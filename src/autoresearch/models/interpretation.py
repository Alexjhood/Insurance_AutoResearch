"""Automatic model interpretation computed by the framework after every experiment.

``compute_automatic_interpretation`` is called by ``experiment_runner`` immediately
after ``dispatch_model`` returns and saves the result as ``interpretation.json``.
No cooperation from model scripts is required.

    from autoresearch.models.interpretation import compute_automatic_interpretation
    interp = compute_automatic_interpretation(eval_df, feature_cols, interpret_fn=fn)

What is always computed (predictions + feature values, no model object needed):

* **One-way A/E analysis** — actual rate, predicted rate, A/E ratio and exposure
  by factor band for every feature.  The standard actuarial exhibit.
* **Spearman rank importance** — |ρ| between each feature and predicted rate.
* **Surrogate SHAP** — a fast GBM surrogate is fitted to the model's own
  ``predicted_rate`` outputs using feature values, then TreeSHAP is run on it.
  Because the surrogate is trained on the original model's predictions, it
  faithfully approximates the original prediction surface without needing the
  model object.

Additionally, when the model script returns an ``interpret_fn`` callable as the
third element of ``fit_predict()``:

* **True PDPs** — marginalised partial dependence via interpret_fn.
* **Permutation importance** — MAE change when each feature is shuffled.

``interpret_fn`` contract: ``(df: pd.DataFrame) -> np.ndarray`` accepting a
DataFrame that contains (at minimum) the model's feature columns and returning
predicted *rates* (not totals) as a 1-D array.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_automatic_interpretation(
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    interpret_fn: Callable | None = None,
    train_df: pd.DataFrame | None = None,
    target_mode: str = "burning_cost",
    n_bins: int = 10,
    n_pdp_sample: int = 3_000,
    n_shap_explain: int = 300,
    max_shap_features: int = 20,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute all interpretation exhibits automatically from predictions + joined features.

    Always computed (requires only predictions + feature values, no model object):

    * **One-way A/E analysis** per feature — actual rate, predicted rate, A/E ratio
      and exposure by factor band.  This is the standard actuarial exhibit.
    * **Spearman rank importance** — |ρ| between each feature and predicted rate.
    * **Surrogate SHAP** — a fast GBM surrogate is fitted to ``predicted_rate``
      using feature values, then TreeSHAP is computed on the surrogate.  The
      surrogate faithfully approximates the original model's prediction surface
      without needing the model object.

    Additionally, when ``interpret_fn`` is supplied:

    * **True PDPs** (marginalised partial dependence via interpret_fn).
    * **Permutation importance** (model-output change when each feature is shuffled).

    Returns a dict with keys ``feature_importance``, ``pdp_data``, and
    (when SHAP is available) ``shap_summary``.  Each ``pdp_data`` entry carries a
    ``source`` field of ``"one_way"`` or ``"pdp_marginal"``.
    """

    from autoresearch.targets import target_spec, normalise_target_mode  # noqa: PLC0415

    spec = target_spec(normalise_target_mode(target_mode))

    avail_cols = [c for c in feature_cols if c in eval_df.columns]
    if not avail_cols:
        return {}

    result: dict[str, Any] = {}

    # ── 1. One-way A/E analysis ───────────────────────────────────────────────
    pdp_data: list[dict[str, Any]] = []
    for col in avail_cols:
        entry = _one_way_analysis(eval_df, col, spec, n_bins)
        if entry is not None:
            pdp_data.append(entry)

    # ── 2. Spearman rank importance ───────────────────────────────────────────
    fi_spearman = _spearman_importance(eval_df, avail_cols, spec.rate_predicted_column)

    # ── 3. Surrogate GBM — feature importance + SHAP ─────────────────────────
    fi_surrogate, shap_summary = _surrogate_interpretation(
        eval_df, avail_cols, spec.rate_predicted_column,
        n_shap_explain=n_shap_explain,
        max_shap_features=max_shap_features,
        random_state=random_state,
    )

    # ── 4. interpret_fn path — true PDPs + permutation importance ────────────
    fi_permutation: list[dict[str, Any]] | None = None
    if interpret_fn is not None:
        ref_df = train_df if train_df is not None else eval_df
        pdp_true = _compute_pdp_marginal(
            interpret_fn, ref_df, avail_cols,
            n_bins=n_bins, n_sample=n_pdp_sample, random_state=random_state,
        )
        if pdp_true:
            pdp_data = pdp_true  # marginalised PDPs replace one-way when available

        fi_permutation = _permutation_importance_from_fn(
            interpret_fn, eval_df, avail_cols, spec.rate_predicted_column,
            n_repeats=2, random_state=random_state,
        )

    # ── 5. Assemble — pick best available importance signal ───────────────────
    fi = fi_permutation or fi_surrogate or fi_spearman

    if fi:
        result["feature_importance"] = fi
    if pdp_data:
        result["pdp_data"] = pdp_data
    if shap_summary:
        result["shap_summary"] = shap_summary

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _one_way_analysis(
    eval_df: pd.DataFrame,
    col: str,
    spec: Any,
    n_bins: int,
) -> dict[str, Any] | None:
    """Compute actual vs predicted rates by factor bands for a single feature."""
    needed = {col, "exposure", spec.rate_actual_column, spec.rate_predicted_column}
    if not needed.issubset(eval_df.columns):
        return None

    df = eval_df[[col, "exposure", spec.rate_actual_column, spec.rate_predicted_column]].copy()
    df = df.dropna(subset=[col, spec.rate_predicted_column])
    if len(df) < 10:
        return None

    exp = df["exposure"].clip(lower=1e-12).values
    actual_tot = df[spec.rate_actual_column].fillna(0).values * exp
    pred_tot = df[spec.rate_predicted_column].values * exp
    col_vals = df[col].values

    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    bands: list[dict[str, Any]] = []

    if is_numeric:
        sort_idx = np.argsort(col_vals)
        col_s = col_vals[sort_idx]
        exp_s = exp[sort_idx]
        act_s = actual_tot[sort_idx]
        prd_s = pred_tot[sort_idx]
        n = len(col_s)
        for b in range(n_bins):
            lo, hi = b * n // n_bins, (b + 1) * n // n_bins
            if lo >= hi:
                continue
            exp_sum = float(exp_s[lo:hi].sum())
            if exp_sum < 1e-12:
                continue
            act_rate = float(act_s[lo:hi].sum()) / exp_sum
            pred_rate = float(prd_s[lo:hi].sum()) / exp_sum
            bands.append({
                "x": round(float(np.median(col_s[lo:hi])), 4),
                "y_pred": round(pred_rate, 6),
                "y_actual": round(act_rate, 6),
                "ae_ratio": round(act_rate / max(pred_rate, 1e-9), 4),
                "exposure": round(exp_sum, 2),
            })
    else:
        df_cp = df.copy()
        df_cp["_act"] = actual_tot
        df_cp["_prd"] = pred_tot
        for val, grp in df_cp.groupby(col, sort=True):
            exp_sum = float(grp["exposure"].clip(lower=1e-12).sum())
            if exp_sum < 1e-12:
                continue
            act_rate = float(grp["_act"].sum()) / exp_sum
            pred_rate = float(grp["_prd"].sum()) / exp_sum
            bands.append({
                "x": str(val),
                "y_pred": round(pred_rate, 6),
                "y_actual": round(act_rate, 6),
                "ae_ratio": round(act_rate / max(pred_rate, 1e-9), 4),
                "exposure": round(exp_sum, 2),
            })

    if len(bands) < 2:
        return None

    return {
        "feature": col,
        "x": [b["x"] for b in bands],
        "y_pred": [b["y_pred"] for b in bands],
        "y_actual": [b["y_actual"] for b in bands],
        "ae_ratio": [b["ae_ratio"] for b in bands],
        "exposure": [b["exposure"] for b in bands],
        "x_is_numeric": is_numeric,
        "source": "one_way",
    }


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    x, y = x.astype(float), y.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    x, y = x[mask], y[mask]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 1e-12 else 0.0


def _spearman_importance(
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    pred_rate_col: str,
) -> list[dict[str, Any]]:
    """Return |Spearman ρ| between each feature and predicted rate, sorted desc."""
    if pred_rate_col not in eval_df.columns:
        return []

    pred_vals = eval_df[pred_rate_col].values
    results: list[dict[str, Any]] = []

    for col in feature_cols:
        if col not in eval_df.columns:
            continue
        col_series = eval_df[col]
        if not pd.api.types.is_numeric_dtype(col_series):
            col_vals = col_series.astype("category").cat.codes.values.astype(float)
        else:
            col_vals = col_series.values.astype(float)

        rho = _spearman_rho(col_vals, pred_vals)
        if not np.isfinite(rho):
            continue
        results.append({
            "feature": col,
            "importance": round(abs(rho), 8),
            "importance_type": "spearman_rho",
        })

    results.sort(key=lambda d: d["importance"], reverse=True)
    return results


def _encode_features(eval_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Label-encode categorical columns; return a numeric-only DataFrame."""
    X = eval_df[feature_cols].copy()
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("category").cat.codes.astype(float)
    return X


def _fit_surrogate(
    X_enc: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[Any, str]:
    """Fit the fastest available tree surrogate. Returns (model, importance_type)."""
    try:
        import lightgbm as lgb  # noqa: PLC0415
        m = lgb.LGBMRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            verbose=-1, random_state=random_state,
        )
        m.fit(X_enc, y)
        return m, "surrogate_gain"
    except Exception:
        pass
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415
        m = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state)
        m.fit(X_enc, y)
        return m, "surrogate_split"
    except Exception:
        pass
    return None, ""


def _surrogate_interpretation(
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    pred_rate_col: str,
    *,
    n_shap_explain: int = 300,
    max_shap_features: int = 20,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Fit a surrogate GBM to predicted rates; return (feature_importance, shap_summary).

    Both outputs are ``None`` when no suitable surrogate library is available.
    The surrogate faithfully approximates the original model's prediction surface
    because it is trained directly on the model's own outputs.
    """
    if pred_rate_col not in eval_df.columns or not feature_cols:
        return None, None

    y = eval_df[pred_rate_col].fillna(0.0).values
    X_enc = _encode_features(eval_df, feature_cols)

    surrogate, imp_type = _fit_surrogate(X_enc, y, random_state=random_state)
    if surrogate is None:
        return None, None

    # Feature importance from the surrogate
    fi: list[dict[str, Any]] | None = None
    if hasattr(surrogate, "feature_importances_"):
        imps = np.asarray(surrogate.feature_importances_, dtype=float)
        imp_max = imps.max()
        if imp_max > 0:
            imps = imps / imp_max
        fi = [
            {"feature": col, "importance": round(float(imp), 8), "importance_type": imp_type}
            for col, imp in zip(feature_cols, imps)
        ]
        fi.sort(key=lambda d: d["importance"], reverse=True)

    # SHAP via TreeExplainer on the surrogate
    shap_summary: dict[str, Any] | None = None
    try:
        import shap  # noqa: PLC0415
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_enc), min(len(X_enc), n_shap_explain), replace=False)
        X_explain = X_enc.iloc[idx].reset_index(drop=True)
        explainer = shap.TreeExplainer(surrogate, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(X_explain)
        shap_values = sv[0] if isinstance(sv, list) else sv
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            mean_abs = np.abs(shap_values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1][:max_shap_features]
            shap_summary = {
                "features": [feature_cols[i] for i in order],
                "mean_abs_shap": [round(float(mean_abs[i]), 8) for i in order],
                "source": "surrogate",
            }
    except Exception:
        pass

    return fi, shap_summary


def _compute_pdp_marginal(
    interpret_fn: Callable,
    ref_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_bins: int = 10,
    n_sample: int = 3_000,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Compute marginalised PDPs using interpret_fn (returns rates, not totals)."""
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(ref_df), min(len(ref_df), n_sample), replace=False)
    sample = ref_df.iloc[idx].copy().reset_index(drop=True)

    exp_vals = (
        sample["exposure"].clip(lower=1e-12).values
        if "exposure" in sample.columns
        else np.ones(len(sample))
    )
    total_exp = float(exp_vals.sum())

    results: list[dict[str, Any]] = []
    for col in feature_cols:
        if col not in ref_df.columns:
            continue
        col_data = ref_df[col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        if is_numeric:
            try:
                _, bin_edges = np.histogram(col_data.values, bins=n_bins)
                grid = [float(0.5 * (bin_edges[i] + bin_edges[i + 1])) for i in range(len(bin_edges) - 1)]
            except Exception:
                continue
        else:
            grid = col_data.value_counts().nlargest(n_bins).index.tolist()

        y_pred_list: list[float] = []
        x_list: list = []
        exp_list: list[float] = []

        for val in grid:
            df_mod = sample.copy()
            df_mod[col] = val
            try:
                rates = np.clip(np.asarray(interpret_fn(df_mod), dtype=float), 0.0, None)
                y_pdp = float(np.sum(rates * exp_vals) / total_exp) if total_exp > 0 else 0.0
                y_pred_list.append(round(y_pdp, 6))
                exp_list.append(round(total_exp, 2))
                x_list.append(float(val) if is_numeric else str(val))
            except Exception:
                continue

        if len(y_pred_list) < 2:
            continue

        results.append({
            "feature": col,
            "x": [round(v, 4) if is_numeric else v for v in x_list],
            "y_pred": y_pred_list,
            "exposure": exp_list,
            "x_is_numeric": is_numeric,
            "source": "pdp_marginal",
        })

    return results


def _permutation_importance_from_fn(
    interpret_fn: Callable,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    pred_rate_col: str,
    *,
    n_repeats: int = 2,
    random_state: int = 42,
) -> list[dict[str, Any]] | None:
    """Permutation importance: measure change in model output when each feature is shuffled."""
    if pred_rate_col not in eval_df.columns:
        return None

    feat_cols_avail = [c for c in feature_cols if c in eval_df.columns]
    if not feat_cols_avail:
        return None

    rng = np.random.default_rng(random_state)
    feat_df = eval_df[feat_cols_avail].copy()

    try:
        baseline = np.clip(np.asarray(interpret_fn(feat_df), dtype=float), 0.0, None)
    except Exception:
        return None

    results: list[dict[str, Any]] = []
    for col in feat_cols_avail:
        deltas: list[float] = []
        for _ in range(n_repeats):
            df_perm = feat_df.copy()
            df_perm[col] = rng.permutation(df_perm[col].values)
            try:
                perm_preds = np.clip(np.asarray(interpret_fn(df_perm), dtype=float), 0.0, None)
                deltas.append(float(np.mean(np.abs(perm_preds - baseline))))
            except Exception:
                pass
        if deltas:
            results.append({
                "feature": col,
                "importance": round(float(np.mean(deltas)), 8),
                "importance_type": "permutation",
            })

    if not results:
        return None
    results.sort(key=lambda d: d["importance"], reverse=True)
    return results
