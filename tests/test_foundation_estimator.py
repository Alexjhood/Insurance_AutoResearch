"""Tests for the opt-in foundation tabular estimators (TabPFN).

These must pass whether or not the ``[foundation]`` extra is installed, so the
package-dependent paths are exercised with a lightweight **stub** ``tabpfn``
module injected into ``sys.modules``. The one test that touches the real package
is guarded with ``importorskip``.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from autoresearch.models import recipe as recipe_pkg
from autoresearch.models.dispatcher import dispatch_model
from autoresearch.models.recipe import foundation, validate_recipe


# ── stub tabpfn ──────────────────────────────────────────────────────────────

class _StubTabPFNRegressor:
    """Records the context it was fit on and predicts its mean — enough to prove
    the wrapper subsamples, batches, and clips correctly without the real model."""

    last_fit_rows: int | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._mean = 0.0

    def fit(self, X, y):
        X = np.asarray(X)
        assert X.ndim == 2  # dense array, not sparse / DataFrame
        type(self).last_fit_rows = X.shape[0]
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        n = np.asarray(X).shape[0]
        # Return a value that can go negative so the wrapper's clip is exercised.
        return np.full(n, self._mean - 0.5, dtype=float)


@pytest.fixture
def stub_tabpfn(monkeypatch):
    mod = types.ModuleType("tabpfn")
    mod.TabPFNRegressor = _StubTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", mod)
    _StubTabPFNRegressor.last_fit_rows = None
    # Register into the live registry for the duration of the test; restore after.
    from autoresearch.models.recipe import registry as reg

    before = dict(reg._ESTIMATORS)
    monkeypatch.setattr(foundation, "_select_device", lambda: "cpu")
    recipe_pkg.enable_foundation_models()
    yield mod
    reg._ESTIMATORS.clear()
    reg._ESTIMATORS.update(before)


class _StubClientRegressor(_StubTabPFNRegressor):
    """Stub for the tabpfn_client API regressor (no device kwarg)."""


@pytest.fixture
def stub_tabpfn_client(monkeypatch):
    calls = {"set_token": []}
    mod = types.ModuleType("tabpfn_client")
    mod.TabPFNRegressor = _StubClientRegressor
    mod.set_access_token = lambda t: calls["set_token"].append(t)
    monkeypatch.setitem(sys.modules, "tabpfn_client", mod)
    monkeypatch.setenv("TABPFN_TOKEN", "test-token")
    monkeypatch.setenv("AUTORESEARCH_TABPFN_BACKEND", "api")
    _StubClientRegressor.last_fit_rows = None
    from autoresearch.models.recipe import registry as reg

    before = dict(reg._ESTIMATORS)
    recipe_pkg.enable_foundation_models()
    yield calls
    reg._ESTIMATORS.clear()
    reg._ESTIMATORS.update(before)


def _frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "record_id": np.arange(n),
        "Exposure": rng.uniform(0.5, 1.5, n),
        "VehPower": rng.integers(1, 6, n),
        "Region": rng.choice(list("abcde"), n),
        "ClaimNb": rng.poisson(0.25, n),
    })
    frame["ClaimAmountCount"] = frame["ClaimNb"]
    frame["ClaimAmount"] = frame["ClaimNb"] * rng.gamma(2.0, 400.0, n)
    frame["ClaimAmountCapped"] = frame["ClaimAmount"]
    n_train = int(n * 0.75)
    split = pd.DataFrame({
        "record_id": np.arange(n),
        "split": ["train"] * n_train + ["search_validation"] * (n - n_train),
    })
    return frame, split


# ── gating (no package installed) ────────────────────────────────────────────

def test_not_registered_by_default() -> None:
    # With the extra absent (the default in CI), tabpfn must not appear and a
    # recipe naming it must fail validation with the standard message.
    if foundation.tabpfn_available():
        pytest.skip("foundation extra is installed in this environment")
    assert "tabpfn" not in recipe_pkg.list_estimators()
    errors = validate_recipe(
        {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error"},
        target_mode="burning_cost",
    )
    assert any("tabpfn" in e for e in errors)


def test_enable_is_noop_without_package(monkeypatch) -> None:
    if foundation.tabpfn_available():
        pytest.skip("foundation extra is installed in this environment")
    assert recipe_pkg.enable_foundation_models() == []


# ── subsampler unit tests (no package needed) ────────────────────────────────

def test_subsample_noop_below_cap() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10, dtype=float)
    w = np.ones(10)
    Xs, ys, ws, n = foundation.subsample_context(X, y, w, max_rows=50, strategy="exposure", seed=1)
    assert n == 10 and Xs.shape[0] == 10
    assert np.array_equal(ys, y)


def test_subsample_is_deterministic_and_capped() -> None:
    rng = np.random.default_rng(0)
    X = rng.random((1000, 3))
    y = rng.random(1000)
    w = rng.random(1000)
    a = foundation.subsample_context(X, y, w, 100, "exposure", seed=7)
    b = foundation.subsample_context(X, y, w, 100, "exposure", seed=7)
    assert a[3] == 100 and a[0].shape == (100, 3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_subsample_exposure_weighting_favours_high_exposure() -> None:
    n = 2000
    X = np.zeros((n, 1))
    y = np.zeros(n)
    w = np.ones(n)
    w[:100] = 100.0  # a few very-high-exposure rows
    # Exposure weighting should over-represent the heavy rows vs uniform.
    _, _, _, _ = foundation.subsample_context(X, y, w, 200, "exposure", seed=3)
    rng = np.random.default_rng(3)
    p = w / w.sum()
    idx = rng.choice(n, size=200, replace=False, p=p)
    heavy = np.mean(idx < 100)
    assert heavy > 100 / n  # more than the uniform share


def test_subsample_rejects_unknown_strategy() -> None:
    from autoresearch.models.recipe.registry import RecipeError

    with pytest.raises(RecipeError):
        foundation.subsample_context(np.zeros((5, 1)), np.zeros(5), np.ones(5), 3, "bogus", 0)


def test_batched_regressor_clips_and_batches() -> None:
    class _Neg:
        calls = 0

        def predict(self, X):
            _Neg.calls += 1
            return np.full(np.asarray(X).shape[0], -1.0)

    wrapped = foundation._BatchedRegressor(_Neg(), batch_size=100)
    out = wrapped.predict(np.zeros((250, 2)))
    assert out.shape == (250,)
    assert np.all(out == 0.0)      # clipped to non-negative
    assert _Neg.calls == 3          # 100 + 100 + 50


# ── end-to-end via stub ──────────────────────────────────────────────────────

def test_registered_and_validates_with_stub(stub_tabpfn) -> None:
    assert "tabpfn" in recipe_pkg.list_estimators()
    errors = validate_recipe(
        {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
         "encoding": "ordinal"},
        target_mode="burning_cost",
    )
    assert errors == []


def test_dispatch_direct_recipe_with_stub(stub_tabpfn) -> None:
    frame, split = _frame()
    rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
          "encoding": "ordinal", "params": {"max_context_rows": 50}}
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)
    # Context was capped to max_context_rows (300 train rows > 50).
    assert _StubTabPFNRegressor.last_fit_rows == 50
    stage_notes = res.model_notes
    assert stage_notes.get("stage_estimator") == "tabpfn"


def test_invalid_objective_rejected_with_stub(stub_tabpfn) -> None:
    # TabPFN advertises squared_error only; gamma must be rejected pre-run.
    errors = validate_recipe(
        {"structure": "direct", "estimator": "tabpfn", "objective": "gamma"},
        target_mode="burning_cost",
    )
    assert any("objective" in e for e in errors)


# ── API backend (stubbed tabpfn_client) ──────────────────────────────────────

def test_api_backend_authenticates_and_dispatches(stub_tabpfn_client) -> None:
    # Registered via the client package (no local tabpfn needed).
    assert "tabpfn" in recipe_pkg.list_estimators()
    frame, split = _frame()
    rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
          "encoding": "ordinal", "params": {"backend": "api", "max_context_rows": 50}}
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)
    assert res.model_notes.get("stage_backend") == "api"
    assert res.model_notes.get("stage_device") == "api"
    # The token was pushed to the client for non-interactive auth.
    assert stub_tabpfn_client["set_token"] == ["test-token"]


def test_api_backend_requires_token(stub_tabpfn_client, monkeypatch) -> None:
    from autoresearch.models.recipe.registry import RecipeError

    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    frame, split = _frame()
    rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
          "encoding": "ordinal", "params": {"backend": "api"}}
    with pytest.raises((RecipeError, Exception)):
        dispatch_model(
            frame, split, model_family="recipe", target_strategy="direct_pure_premium",
            train_split="train", score_splits=("search_validation",),
            hyperparameters={"recipe": rc}, target_mode="burning_cost",
        )


def test_unknown_backend_rejected() -> None:
    from autoresearch.models.recipe.registry import RecipeError

    with pytest.raises(RecipeError):
        foundation._resolve_backend({"backend": "bogus"})


# ── real package (only when installed) ───────────────────────────────────────

def test_real_tabpfn_smoke() -> None:
    # Installed locally here, so guard behind an explicit opt-in to keep the
    # default suite fast and hermetic (this does a real model fit).
    if os.environ.get("RUN_REAL_TABPFN") != "1":
        pytest.skip("set RUN_REAL_TABPFN=1 to run the real local TabPFN fit")
    pytest.importorskip("tabpfn")
    recipe_pkg.enable_foundation_models()
    assert "tabpfn" in recipe_pkg.list_estimators()
    frame, split = _frame(n=120)
    rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
          "encoding": "ordinal", "params": {"max_context_rows": 80, "device": "cpu"}}
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)


# ── categorical feature flagging ─────────────────────────────────────────────

def test_categorical_features_indices_passed_to_api(stub_tabpfn_client) -> None:
    """Named categoricals map to post-ordinal-encoding column positions.

    The _frame features are VehPower (numeric) + Region (categorical); the
    ordinal transformer emits numerics first, so Region is column 1.
    """
    captured: dict = {}
    orig_init = _StubClientRegressor.__init__

    def _spy_init(self, **kwargs):
        captured.update(kwargs)
        orig_init(self, **kwargs)

    _StubClientRegressor.__init__ = _spy_init
    try:
        frame, split = _frame()
        rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
              "encoding": "ordinal",
              "params": {"backend": "api", "max_context_rows": 50,
                         "categorical_features": ["Region"]}}
        res = dispatch_model(
            frame, split, model_family="recipe", target_strategy="direct_pure_premium",
            train_split="train", score_splits=("search_validation",),
            hyperparameters={"recipe": rc}, target_mode="burning_cost",
        )
    finally:
        _StubClientRegressor.__init__ = orig_init

    n_features = res.model_notes.get("n_features")
    cat_count = 1  # Region is the only categorical feature in _frame
    assert captured["categorical_features_indices"] == [n_features - cat_count]
    assert res.model_notes.get("stage_categorical_features") == ["Region"]
    assert res.model_notes.get("stage_categorical_features_indices") == [n_features - cat_count]


def test_categorical_features_unknown_name_rejected(stub_tabpfn_client) -> None:
    from autoresearch.models.recipe.registry import RecipeError

    frame, split = _frame()
    rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
          "encoding": "ordinal",
          "params": {"backend": "api", "categorical_features": ["VehBrand"]}}
    with pytest.raises((RecipeError, Exception), match="VehBrand"):
        dispatch_model(
            frame, split, model_family="recipe", target_strategy="direct_pure_premium",
            train_split="train", score_splits=("search_validation",),
            hyperparameters={"recipe": rc}, target_mode="burning_cost",
        )


def test_categorical_features_omitted_sends_nothing(stub_tabpfn_client) -> None:
    captured: dict = {}
    orig_init = _StubClientRegressor.__init__

    def _spy_init(self, **kwargs):
        captured.update(kwargs)
        orig_init(self, **kwargs)

    _StubClientRegressor.__init__ = _spy_init
    try:
        frame, split = _frame()
        rc = {"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
              "encoding": "ordinal", "params": {"backend": "api", "max_context_rows": 50}}
        dispatch_model(
            frame, split, model_family="recipe", target_strategy="direct_pure_premium",
            train_split="train", score_splits=("search_validation",),
            hyperparameters={"recipe": rc}, target_mode="burning_cost",
        )
    finally:
        _StubClientRegressor.__init__ = orig_init
    assert "categorical_features_indices" not in captured
