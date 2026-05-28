"""Actuarially-relevant metric panel for target-model evaluation.

Primary metric is configurable. The default gate uses weighted Gini, where
higher is better. Panel also includes calibration ratio, Tweedie deviance,
double-lift slope, MAE, and RMSE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_tweedie_deviance

from autoresearch.targets import BURNING_COST, normalise_target_mode, target_spec


HIGHER_IS_BETTER_METRICS = {"gini_weighted"}


def lower_is_better(metric_name: str) -> bool:
    """Return whether lower values are better for a metric."""

    return metric_name not in HIGHER_IS_BETTER_METRICS


def full_metric_panel(
    actual_target: pd.Series,
    predicted_target: pd.Series,
    exposure: pd.Series,
    *,
    tweedie_power: float = 1.5,
    target_mode: str = BURNING_COST,
) -> dict[str, Any]:
    """Compute the full actuarial metric panel for the active target."""

    spec = target_spec(target_mode)
    actual = actual_target.astype(float).to_numpy()
    predicted = np.clip(predicted_target.astype(float).to_numpy(), 1e-9, None)
    exp = np.clip(exposure.astype(float).to_numpy(), 1e-12, None)

    actual_rate = actual / exp
    predicted_rate = predicted / exp

    # Deviance on target rate (pure premium for cost, claim frequency for count)
    tweedie_dev = _tweedie_deviance(actual_rate, predicted_rate, exp, tweedie_power)
    poisson_dev = _tweedie_deviance(actual_rate, predicted_rate, exp, 1.0)

    # Calibration: predicted/actual ratio (want ≈ 1.0)
    total_actual = float(actual.sum())
    total_predicted = float(predicted.sum())
    pred_to_actual = total_predicted / total_actual if total_actual > 0 else float("nan")

    # Weighted MAE and RMSE on the target total, not the per-exposure rate
    target_error = predicted - actual
    weighted_mae = float(np.average(np.abs(target_error), weights=exp))
    weighted_rmse = float(np.sqrt(np.average(target_error**2, weights=exp)))

    # Gini coefficient (discrimination / lift)
    gini = _gini_weighted(actual, predicted, exp)

    # Double-lift slope (regression of actual rate on predicted rate by decile)
    double_lift_slope = _double_lift_slope(actual_rate, predicted_rate, exp)

    rate_error = predicted_rate - actual_rate
    rmse_rate = float(np.sqrt(np.mean(rate_error**2)))
    mae_rate = float(np.mean(np.abs(rate_error)))

    panel = {
        "target_mode": spec.mode,
        "tweedie_deviance_p15": tweedie_dev,
        "poisson_deviance": poisson_dev,
        "predicted_to_actual_ratio": pred_to_actual,
        "total_actual_target": total_actual,
        "total_predicted_target": total_predicted,
        "exposure_sum": float(exp.sum()),
        "weighted_mae_target": weighted_mae,
        "weighted_rmse_target": weighted_rmse,
        "gini_weighted": gini,
        "double_lift_slope": double_lift_slope,
        "mean_actual_rate": float(np.average(actual_rate, weights=exp)),
        "mean_predicted_rate": float(np.average(predicted_rate, weights=exp)),
        "rmse_rate": rmse_rate,
        "mae_rate": mae_rate,
    }
    panel[spec.total_actual_key] = total_actual
    panel[spec.total_predicted_key] = total_predicted
    panel[spec.mae_key] = weighted_mae
    panel[spec.rmse_key] = weighted_rmse
    panel[spec.mean_actual_rate_key] = panel["mean_actual_rate"]
    panel[spec.mean_predicted_rate_key] = panel["mean_predicted_rate"]

    if spec.mode == BURNING_COST:
        # Legacy names retained for historical artifacts and dashboards.
        panel["rmse_pure_premium"] = rmse_rate
        panel["mae_pure_premium"] = mae_rate
    else:
        panel["rmse_frequency"] = rmse_rate
        panel["mae_frequency"] = mae_rate
    return panel


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
    target_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate split-level and aggregate scores, excluding milestone holdout."""

    if (predictions["split"] == "milestone_holdout").any():
        raise ValueError("Ordinary evaluation cannot include milestone_holdout rows")
    target_mode = infer_target_mode(predictions, target_mode)
    spec = target_spec(target_mode)

    split_metrics: list[dict[str, Any]] = []
    for split, split_frame in predictions.groupby("split", sort=True):
        actual_col, predicted_col = prediction_target_columns(split_frame, spec.mode)
        panel = full_metric_panel(
            split_frame[actual_col],
            split_frame[predicted_col],
            split_frame["exposure"],
            tweedie_power=tweedie_power,
            target_mode=target_mode,
        )
        panel["split"] = split
        panel["row_count"] = int(len(split_frame))
        split_metrics.append(panel)

    eval_values = [item[primary_metric] for item in split_metrics if item["split"] in set(eval_splits)]
    if not eval_values:
        raise ValueError(f"No configured evaluation splits found: {eval_splits}")

    return {
        "primary_metric": primary_metric,
        "lower_is_better": lower_is_better(primary_metric),
        "target_mode": target_mode,
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
    """Exposure-weighted Tweedie deviance on the target rate."""

    try:
        return float(mean_tweedie_deviance(actual_pp, predicted_pp, sample_weight=weights, power=power))
    except Exception:
        return float("nan")


def _gini_weighted(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    """Exposure-weighted Gini coefficient measuring discrimination.

    Sorts by predicted target rate (predicted / exposure), not by predicted
    target total. Sorting by totals would conflate the exposure-size effect
    with model discrimination.
    """

    if len(actual) < 2:
        return float("nan")
    safe_w = np.clip(weights, 1e-12, None)
    predicted_pp = predicted / safe_w
    order = np.argsort(predicted_pp)
    w = weights[order]
    y = actual[order]
    cum_w = np.cumsum(w) / w.sum()
    cum_y = np.cumsum(y) / y.sum() if y.sum() > 0 else cum_w
    # Prepend origin so the trapezoidal rule covers the full [0,1]×[0,1] square
    cum_w = np.concatenate([[0.0], cum_w])
    cum_y = np.concatenate([[0.0], cum_y])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    lorenz_area = float(_trapz(cum_y, cum_w))
    return float(1 - 2 * lorenz_area)


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


def infer_target_mode(predictions: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        return normalise_target_mode(requested)
    if "target_mode" in predictions.columns:
        modes = {str(value) for value in predictions["target_mode"].dropna().unique()}
        if len(modes) == 1:
            return normalise_target_mode(next(iter(modes)))
    return BURNING_COST


def prediction_target_columns(frame: pd.DataFrame, target_mode: str | None = None) -> tuple[str, str]:
    spec = target_spec(target_mode or infer_target_mode(frame))
    if "actual_target" in frame.columns and "predicted_target" in frame.columns:
        return "actual_target", "predicted_target"
    if spec.actual_alias in frame.columns and spec.predicted_alias in frame.columns:
        return spec.actual_alias, spec.predicted_alias
    raise ValueError(
        f"Predictions are missing target columns for {spec.mode}: "
        f"expected actual_target/predicted_target or {spec.actual_alias}/{spec.predicted_alias}"
    )
