"""Actuarially-relevant metric panel for burning-cost model evaluation.

Primary metric: Tweedie deviance (power configurable, default 1.5).
Panel also includes: calibration ratio, Gini, double-lift slope, MAE, RMSE.
Per-row pure-premium RMSE is computed but NOT used for ranking.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def full_metric_panel(
    actual_cost: pd.Series,
    predicted_cost: pd.Series,
    exposure: pd.Series,
    *,
    tweedie_power: float = 1.5,
) -> dict[str, Any]:
    """Compute the full actuarial metric panel from claim-cost predictions."""

    actual = actual_cost.astype(float).to_numpy()
    predicted = np.clip(predicted_cost.astype(float).to_numpy(), 1e-9, None)
    exp = np.clip(exposure.astype(float).to_numpy(), 1e-12, None)

    actual_pp = actual / exp
    predicted_pp = predicted / exp

    # Primary: Tweedie deviance on pure premium (exposure-weighted)
    tweedie_dev = _tweedie_deviance(actual_pp, predicted_pp, exp, tweedie_power)
    poisson_dev = _tweedie_deviance(actual_pp, predicted_pp, exp, 1.0)

    # Calibration: predicted/actual ratio (want ≈ 1.0)
    total_actual = float(actual.sum())
    total_predicted = float(predicted.sum())
    pred_to_actual = total_predicted / total_actual if total_actual > 0 else float("nan")

    # Weighted MAE and RMSE on claim cost (not pure premium — more stable)
    cost_error = predicted - actual
    weighted_mae = float(np.average(np.abs(cost_error), weights=exp))
    weighted_rmse = float(np.sqrt(np.average(cost_error**2, weights=exp)))

    # Gini coefficient (discrimination / lift)
    gini = _gini_weighted(actual, predicted, exp)

    # Double-lift slope (regression of actual_pp / pred_pp by decile)
    double_lift_slope = _double_lift_slope(actual_pp, predicted_pp, exp)

    # Legacy RMSE on pure premium (retained for backward compat; do NOT rank on this)
    pp_error = predicted_pp - actual_pp
    rmse_pure_premium = float(np.sqrt(np.mean(pp_error**2)))
    mae_pure_premium = float(np.mean(np.abs(pp_error)))

    return {
        "tweedie_deviance_p15": tweedie_dev,
        "poisson_deviance": poisson_dev,
        "predicted_to_actual_ratio": pred_to_actual,
        "total_actual_claim_cost": total_actual,
        "total_predicted_claim_cost": total_predicted,
        "exposure_sum": float(exp.sum()),
        "weighted_mae_claim_cost": weighted_mae,
        "weighted_rmse_claim_cost": weighted_rmse,
        "gini_weighted": gini,
        "double_lift_slope": double_lift_slope,
        "mean_actual_pure_premium": float(np.average(actual_pp, weights=exp)),
        "mean_predicted_pure_premium": float(np.average(predicted_pp, weights=exp)),
        # Legacy — kept for diagnostics, not primary ranking
        "rmse_pure_premium": rmse_pure_premium,
        "mae_pure_premium": mae_pure_premium,
    }


def regression_metrics(
    actual_claim: pd.Series,
    predicted_claim: pd.Series,
    exposure: pd.Series,
    *,
    tweedie_power: float = 1.5,
) -> dict[str, float]:
    """Backward-compatible wrapper returning a subset of the full panel."""

    panel = full_metric_panel(actual_claim, predicted_claim, exposure, tweedie_power=tweedie_power)
    return {k: v for k, v in panel.items() if isinstance(v, float)}


def evaluate_predictions(
    predictions: pd.DataFrame,
    eval_splits: tuple[str, ...],
    *,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
) -> dict[str, Any]:
    """Evaluate split-level and aggregate scores, excluding milestone holdout."""

    if (predictions["split"] == "milestone_holdout").any():
        raise ValueError("Ordinary evaluation cannot include milestone_holdout rows")

    split_metrics: list[dict[str, Any]] = []
    for split, split_frame in predictions.groupby("split", sort=True):
        panel = full_metric_panel(
            split_frame["actual_claim_cost"],
            split_frame["predicted_claim_cost"],
            split_frame["exposure"],
            tweedie_power=tweedie_power,
        )
        panel["split"] = split
        panel["row_count"] = int(len(split_frame))
        split_metrics.append(panel)

    eval_values = [item[primary_metric] for item in split_metrics if item["split"] in set(eval_splits)]
    if not eval_values:
        raise ValueError(f"No configured evaluation splits found: {eval_splits}")

    return {
        "primary_metric": primary_metric,
        "lower_is_better": True,
        "tweedie_power": tweedie_power,
        "ordinary_eval_splits": list(eval_splits),
        "split_metrics": split_metrics,
        "aggregate": {
            "mean_score": float(np.mean(eval_values)),
            "std_score": float(np.std(eval_values, ddof=0)),
            "split_count": len(eval_values),
        },
    }


# ── private helpers ───────────────────────────────────────────────────────────

def _tweedie_deviance(actual_pp: np.ndarray, predicted_pp: np.ndarray, weights: np.ndarray, power: float) -> float:
    """Exposure-weighted Tweedie deviance on pure premium."""

    try:
        return float(mean_tweedie_deviance(actual_pp, predicted_pp, sample_weight=weights, power=power))
    except Exception:
        return float("nan")


def _gini_weighted(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    """Exposure-weighted Gini coefficient measuring discrimination."""

    if len(actual) < 2:
        return float("nan")
    order = np.argsort(predicted)
    w = weights[order]
    y = actual[order]
    cum_w = np.cumsum(w) / w.sum()
    cum_y = np.cumsum(y) / y.sum() if y.sum() > 0 else cum_w
    # Area under Lorenz curve via trapezoidal rule
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    lorenz_area = float(_trapz(cum_y, cum_w))
    return float(2 * lorenz_area - 1)


def _double_lift_slope(actual_pp: np.ndarray, predicted_pp: np.ndarray, weights: np.ndarray) -> float:
    """Slope of actual/predicted ratio by predicted decile (want ≈ 1.0)."""

    if len(actual_pp) < 20:
        return float("nan")
    n_bins = 10
    order = np.argsort(predicted_pp)
    bin_size = len(order) // n_bins
    if bin_size == 0:
        return float("nan")

    x_vals, y_vals = [], []
    for i in range(n_bins):
        idx = order[i * bin_size: (i + 1) * bin_size]
        w = weights[idx]
        if w.sum() == 0:
            continue
        x_vals.append(float(np.average(predicted_pp[idx], weights=w)))
        y_vals.append(float(np.average(actual_pp[idx], weights=w)))

    if len(x_vals) < 2:
        return float("nan")
    x = np.array(x_vals)
    y = np.array(y_vals)
    # OLS slope of actual on predicted
    cov = np.cov(x, y)
    if cov[0, 0] == 0:
        return float("nan")
    return float(cov[0, 1] / cov[0, 0])
