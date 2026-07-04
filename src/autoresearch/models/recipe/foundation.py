"""Foundation tabular models as recipe estimators (opt-in, optional extra).

These are transformer foundation models (TabPFN today; TabFM later) that predict
by in-context learning rather than gradient training. They live behind:

* an **optional dependency** — the estimator only registers if its package is
  importable (``pip install -e .[foundation]``), so a machine without the extra
  never sees it; and
* a **per-run opt-in** — the recipe package registers these only when foundation
  models are enabled for the run (``bootstrap-track --enable-foundation-models``
  or ``AUTORESEARCH_FOUNDATION_MODELS=1``). See ``recipe.enable_foundation_models``.

Design notes specific to this codebase:

* **No sample weights.** TabPFN's sklearn interface has no ``sample_weight``. The
  framework fits every stage on the *rate* with exposure weights, so exposure is
  reintroduced here by **exposure-weighted subsampling** of the training context
  (rows drawn with probability ∝ exposure). The framework's downstream
  calibration (Σactual/Σpred) then fixes the aggregate level, exactly as for the
  gradient estimators.
* **Context is subsampled.** In-context learning holds the whole training set in
  memory for a forward pass; the 500k-row search split would not fit the compute
  budget on the target Macs, so the context is capped at ``max_context_rows``.
* **Scoring is batched** to bound peak memory on the full 500k-row score frame.
* **Objective is fixed.** TabPFN has a single regression head; we advertise only
  ``squared_error`` (honest) rather than pretending to optimise Tweedie/Poisson.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import numpy as np

from autoresearch.models.recipe.registry import (
    EstimatorSpec,
    FitContext,
    RecipeError,
    register_estimator,
)


# Two backends (see docs/internal/foundation_models_integration_plan.md):
#   "local" — the `tabpfn` package on this machine's CPU/GPU. On Apple Silicon the
#             MPS path is broken/slow, so local is only practical for small data.
#   "api"   — the `tabpfn_client` package offloading to Prior Labs' GPU. This is
#             the practical backend for the full-scale problem on a Mac.
# `AUTORESEARCH_TABPFN_BACKEND` sets the default; a recipe `params.backend` wins.
_DEFAULT_BACKEND = "local"

# Local default context tuned for the M2 Pro 32GB; the M3 Air should override
# lower. The API runs on a GPU, so it affords a larger context cheaply (credits
# are dominated by the per-query-row cost, ~2.8 cr/row, not the context).
_DEFAULT_MAX_CONTEXT_ROWS = 40_000
_DEFAULT_API_MAX_CONTEXT_ROWS = 20_000
_DEFAULT_PREDICT_BATCH = 50_000
# The API caps test rows per request; batch conservatively under it.
_DEFAULT_API_PREDICT_BATCH = 30_000


def _resolve_backend(params: dict[str, Any]) -> str:
    backend = params.pop("backend", None) or os.environ.get(
        "AUTORESEARCH_TABPFN_BACKEND", _DEFAULT_BACKEND
    )
    backend = str(backend).lower()
    if backend not in {"local", "api"}:
        raise RecipeError(
            f"Unknown tabpfn backend {backend!r}; use 'local' or 'api'."
        )
    return backend


# ── shared helpers (reused by any foundation estimator) ──────────────────────

def _densify(X: Any) -> Any:
    """Foundation models need dense float arrays; the one-hot encoder is sparse."""
    from scipy import sparse

    return X.toarray() if sparse.issparse(X) else X


def _take(matrix: Any, idx: np.ndarray) -> Any:
    import pandas as pd

    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[idx]
    return matrix[idx]


def _select_device() -> str:
    """Pick the TabPFN compute device. ``AUTORESEARCH_TABPFN_DEVICE`` overrides.

    Prefer CUDA when present. **Apple-Silicon MPS is deliberately NOT auto-selected:**
    benchmarking (2026-07-03, M3 Air) found TabPFN's Metal backend both crashes at
    larger contexts (MPSNDArray assertion) and is pathologically slow even when it
    works (~85s per 1000 rows vs ~10-40s on CPU). CPU is the stable default on
    Apple Silicon; set ``AUTORESEARCH_TABPFN_DEVICE=mps`` to force MPS anyway.
    """
    override = os.environ.get("AUTORESEARCH_TABPFN_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def subsample_context(
    X: Any,
    y: np.ndarray,
    w: np.ndarray,
    max_rows: int,
    strategy: str,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    """Cap the in-context training set to ``max_rows`` rows.

    ``strategy="exposure"`` draws rows without replacement with probability ∝ the
    exposure weight, so high-exposure policies are represented in proportion to
    their weight (a stand-in for the sample weighting TabPFN cannot take).
    ``strategy="uniform"`` draws uniformly. Below the cap it is a no-op. Returns
    ``(X_sub, y_sub, w_sub, n_context)`` and is deterministic given ``seed``.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)
    if max_rows <= 0:
        raise RecipeError("max_context_rows must be a positive integer")
    if n <= max_rows:
        return X, y, w, n

    rng = np.random.default_rng(seed)
    if strategy == "exposure":
        p = np.clip(w, 0.0, None)
        total = float(p.sum())
        p = (p / total) if total > 0 else None
    elif strategy == "uniform":
        p = None
    else:
        raise RecipeError(
            f"Unknown subsample_strategy {strategy!r}; use 'exposure' or 'uniform'."
        )
    idx = rng.choice(n, size=max_rows, replace=False, p=p)
    idx.sort()  # keep original row order for reproducibility / locality
    return _take(X, idx), y[idx], w[idx], max_rows


class _BatchedRegressor:
    """Wraps a fitted foundation regressor so ``predict`` runs in row-batches
    (bounding peak memory on the full score frame) and returns non-negative
    rates (the estimator contract requires ≥0; TabPFN's regression mean can dip
    below zero)."""

    def __init__(self, model: Any, batch_size: int = _DEFAULT_PREDICT_BATCH) -> None:
        self._model = model
        self._batch_size = int(batch_size)

    def predict(self, X: Any) -> np.ndarray:
        X = _densify(X)
        n = X.shape[0]
        out: list[np.ndarray] = []
        for start in range(0, n, self._batch_size):
            chunk = X[start : start + self._batch_size]
            out.append(np.asarray(self._model.predict(chunk), dtype=float))
        pred = np.concatenate(out) if out else np.zeros(0, dtype=float)
        return np.clip(pred, 0.0, None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


# ── TabPFN ───────────────────────────────────────────────────────────────────

def _authenticate_api() -> None:
    """Authenticate the tabpfn_client session non-interactively from TABPFN_TOKEN.

    The client caches after the first login, but research cycles run headless, so
    we set the token explicitly. Fail loudly with instructions rather than let the
    client try to open a browser it cannot.
    """
    import tabpfn_client

    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        raise RecipeError(
            "tabpfn backend='api' needs TABPFN_TOKEN (Prior Labs API key). "
            "Get it from https://ux.priorlabs.ai/account, accept the licence, then "
            "export TABPFN_TOKEN=<key>. See scripts/setup_foundation_models.py."
        )
    tabpfn_client.set_access_token(token)


# Thinking-mode params (API only) — extra fit-time compute for higher precision.
# thinking_effort ∈ {"medium","high"}; thinking_timeout_s is the fit budget in
# seconds (client-capped at 2400); thinking_metric is what it optimises toward
# (for regression prefer "spearmanr", the rank metric closest to exposure-weighted
# Gini — "mse"/"rmse"/"mae"/"r2"/"smape" are also valid). thinking_mode=True alone
# means effort="medium". Draws from a SEPARATE Prior Labs daily budget and is much
# slower, so use a config with `[compute] enforce = false` (see frugal_thinking.toml).
_THINKING_PARAMS = ("thinking_mode", "thinking_effort", "thinking_timeout_s", "thinking_metric")


def _pop_thinking(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "thinking_mode" in params:
        out["thinking_mode"] = bool(params.pop("thinking_mode"))
    if "thinking_effort" in params:
        out["thinking_effort"] = str(params.pop("thinking_effort"))
    if "thinking_timeout_s" in params:
        out["thinking_timeout_s"] = float(params.pop("thinking_timeout_s"))
    if "thinking_metric" in params:
        out["thinking_metric"] = str(params.pop("thinking_metric"))
    return out


def _make_regressor(backend: str, params: dict[str, Any], seed: int, device: str | None):
    """Construct a local or API TabPFN regressor from the curated params."""
    if backend == "api":
        _authenticate_api()
        from tabpfn_client import TabPFNRegressor

        # The managed API runs on a GPU and takes no device/pretraining-limit
        # kwargs; keep the passthrough minimal.
        reg_kwargs: dict[str, Any] = {}
        if "n_estimators" in params:
            reg_kwargs["n_estimators"] = int(params.pop("n_estimators"))
        reg_kwargs.update(_pop_thinking(params))  # thinking mode (client-validated)
        return TabPFNRegressor(**reg_kwargs)

    if any(k in params for k in _THINKING_PARAMS):
        raise RecipeError(
            "thinking mode is an API feature; it requires backend='api' "
            f"(got backend={backend!r}). Remove {sorted(k for k in _THINKING_PARAMS if k in params)} "
            "or switch the recipe to the api backend."
        )

    # macOS OpenMP guard (local backend only): the local `tabpfn` package imports
    # torch, which registers its own OpenMP/libomp runtime. On macOS, loading
    # torch's runtime *before* LightGBM's makes a later LightGBM fit segfault
    # (e.g. the framework's interpretation surrogate). Importing LightGBM first
    # makes it register its runtime first, which coexists cleanly. The API backend
    # never imports torch, so it does not need this. Harmless if lightgbm is
    # already imported.
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        pass

    from tabpfn import TabPFNRegressor

    reg_kwargs = {
        "device": device or _select_device(),
        "ignore_pretraining_limits": True,
        "random_state": seed,
    }
    if "n_estimators" in params:
        reg_kwargs["n_estimators"] = int(params.pop("n_estimators"))
    reg_kwargs.update(params)  # any remaining curated passthrough
    return TabPFNRegressor(**reg_kwargs)


def _fit_tabpfn(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    params = dict(ctx.params or {})
    backend = _resolve_backend(params)
    is_api = backend == "api"
    default_ctx = _DEFAULT_API_MAX_CONTEXT_ROWS if is_api else _DEFAULT_MAX_CONTEXT_ROWS
    default_batch = _DEFAULT_API_PREDICT_BATCH if is_api else _DEFAULT_PREDICT_BATCH

    max_context = int(params.pop("max_context_rows", default_ctx))
    strategy = str(params.pop("subsample_strategy", "exposure"))
    seed = int(params.pop("random_state", 42))
    device = params.pop("device", None)  # local-only; ignored by the API
    batch_size = int(params.pop("predict_batch_size", default_batch))

    # Capture thinking config for the notes before _make_regressor pops it.
    thinking = {k: params[k] for k in _THINKING_PARAMS if k in params}

    X = _densify(ctx.X_train)
    X_ctx, y_ctx, _w_ctx, n_context = subsample_context(
        X, ctx.y_train, ctx.w_train, max_context, strategy, seed
    )

    model = _make_regressor(backend, params, seed, device)
    model.fit(X_ctx, np.asarray(y_ctx, dtype=float))

    notes = {
        "estimator": "tabpfn",
        "objective": ctx.objective,
        "backend": backend,
        "device": ("api" if is_api else (device or _select_device())),
        "context_rows": int(n_context),
        "max_context_rows": max_context,
        "subsample_strategy": strategy,
        "subsampled": bool(n_context < len(ctx.y_train)),
        "thinking": thinking or None,
    }
    return _BatchedRegressor(model, batch_size=batch_size), notes


_TABPFN_SPEC = EstimatorSpec(
    name="tabpfn",
    # TabPFN has a single regression head; advertise the honest objective only.
    objectives=frozenset({"squared_error"}),
    encodings=frozenset({"ordinal", "one_hot"}),
    default_encoding="ordinal",
    fit=_fit_tabpfn,
    supports_early_stopping=False,
    native_categorical=False,
    description=(
        "TabPFN-3 foundation model (in-context learning, no gradient training). "
        "Training context is subsampled to max_context_rows (exposure-weighted by "
        "default) and scoring is batched; exposure enters via the subsample, not a "
        "sample weight, with framework calibration fixing the level. backend='api' "
        "offloads to Prior Labs' GPU (needs TABPFN_TOKEN) — the practical choice on "
        "Apple Silicon; backend='local' runs on this machine (CPU is slow at scale). "
        "Requires the [foundation] extra and a per-run opt-in."
    ),
    allowed_params=frozenset({
        "backend", "max_context_rows", "subsample_strategy", "random_state",
        "n_estimators", "device", "predict_batch_size",
        # Thinking mode (api backend only; see _pop_thinking).
        "thinking_mode", "thinking_effort", "thinking_timeout_s", "thinking_metric",
    }),
)


# ── registration ─────────────────────────────────────────────────────────────

def _importable(name: str) -> bool:
    if name in sys.modules:  # already imported (incl. test stubs)
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def tabpfn_available() -> bool:
    """True if either backend is usable: the local ``tabpfn`` package or the
    ``tabpfn_client`` API package."""
    return _importable("tabpfn") or _importable("tabpfn_client")


def register_foundation_estimators() -> list[str]:
    """Register every foundation estimator whose package is installed.

    Idempotent — re-registering the same name overwrites harmlessly. Returns the
    names that were registered so callers can report what became available.
    """
    registered: list[str] = []
    if tabpfn_available():
        register_estimator(_TABPFN_SPEC)
        registered.append("tabpfn")
    return registered
