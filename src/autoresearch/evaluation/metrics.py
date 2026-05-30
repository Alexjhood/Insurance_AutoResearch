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


HIGHER_IS_BETTER_METRICS = {
    "gini_weighted",
    "rank_gini_weighted",
    "spearman_rho",
    "kendall_tau",
    "decile_lift_monotonicity",
}

# Asymmetric Pricing Loss penalty weights (fixed product constants).
# Under-pricing a policy costs ~4× a missed quote: a policy written at a loss
# triggers claims that exceed premium, whereas a miss only loses the margin.
TAU_UNDER = 4.0
TAU_OVER = 1.0


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

    # Rank-Gini (Somers' D) — bounded-influence gate metric.
    # Identical to gini_weighted but cumulates rank of actual loss rather than
    # loss amounts, so one 100k claim has O(1/n) influence instead of unbounded.
    rank_gini = _rank_gini_weighted(actual, predicted, exp)

    # Rank correlations — model-agnostic, assume nothing about distribution.
    spearman = _spearman_rho(actual_rate, predicted_rate)
    kendall = _kendall_tau(actual_rate, predicted_rate)

    # Decile lift monotonicity — Spearman of decile-mean-actual vs decile order.
    decile_mono = _decile_lift_monotonicity(actual_rate, predicted_rate, exp)

    # Asymmetric Pricing Loss — penalises under-pricing 4× over-pricing.
    apl_metrics = _asym_pricing_loss(actual_rate, predicted_rate, exp)

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
        "rank_gini_weighted": rank_gini,
        "spearman_rho": spearman,
        "kendall_tau": kendall,
        "decile_lift_monotonicity": decile_mono,
        "asym_pricing_loss": apl_metrics["asym_pricing_loss"],
        "apl_under_cost": apl_metrics["apl_under_cost"],
        "apl_over_cost": apl_metrics["apl_over_cost"],
        "apl_under_over_ratio": apl_metrics["apl_under_over_ratio"],
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


def _asym_pricing_loss(
    actual_rate: np.ndarray,
    predicted_rate: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Exposure-weighted Asymmetric Pricing Loss (lower is better).

    Penalises under-pricing TAU_UNDER× and over-pricing TAU_OVER×.
    Equivalent to a pinball loss at τ = TAU_UNDER / (TAU_UNDER + TAU_OVER) = 0.8,
    rewarding prudent upward loading.

    u > 0 means we wrote at a loss (actual rate > predicted rate = under-priced).
    """

    safe_w = np.clip(weights, 1e-12, None)
    w_sum = float(safe_w.sum())
    if w_sum < 1e-12:
        return {
            "asym_pricing_loss": float("nan"),
            "apl_under_cost": float("nan"),
            "apl_over_cost": float("nan"),
            "apl_under_over_ratio": float("nan"),
        }

    u = actual_rate - predicted_rate
    under = np.maximum(u, 0.0)
    over = np.maximum(-u, 0.0)

    apl_under = float(np.sum(safe_w * under) / w_sum)
    apl_over = float(np.sum(safe_w * over) / w_sum)
    apl = TAU_UNDER * apl_under + TAU_OVER * apl_over
    ratio = apl_under / max(apl_over, 1e-12)

    return {
        "asym_pricing_loss": float(apl),
        "apl_under_cost": apl_under,
        "apl_over_cost": apl_over,
        "apl_under_over_ratio": float(ratio),
    }


def _rank_gini_weighted(actual: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    """Rank-Gini (Somers' D) — bounded-influence discrimination metric.

    Identical structure to _gini_weighted but the Lorenz y-axis cumulates
    exposure-weighted *ranks* of actual loss rather than loss amounts.
    A single extreme claim contributes O(1/n) to this statistic instead of
    O(claim_size / total_claims), making it robust to right-tail volatility.

    Interpretation: the fraction of concordant pairs minus discordant pairs
    (probability that a randomly chosen higher-predicted policy has higher
    actual loss than a randomly chosen lower-predicted policy).
    Range [-1, 1]; higher is better; 0 = random ordering.
    """

    if len(actual) < 2:
        return float("nan")
    safe_w = np.clip(weights, 1e-12, None)
    predicted_pp = predicted / safe_w
    actual_rate = actual / safe_w

    # Sort by predicted rate ascending (same orientation as gini_weighted)
    order = np.argsort(predicted_pp)
    w = safe_w[order]
    actual_rate_ordered = actual_rate[order]

    # Replace loss amounts with exposure-weighted ranks of actual loss: each
    # policy's rank value is the midpoint of its exposure band in actual-rate
    # order, so one extreme claim contributes O(1/n) rather than
    # O(claim / total).  Vectorised equivalent of a cumulative-midpoint loop.
    rank_order = np.argsort(actual_rate_ordered, kind="stable")
    w_in_rank_order = w[rank_order]
    midpoints = np.cumsum(w_in_rank_order) - 0.5 * w_in_rank_order
    rank_values = np.empty(len(w))
    rank_values[rank_order] = midpoints

    cum_w = np.cumsum(w) / w.sum()
    total_rank = rank_values.sum()
    if total_rank < 1e-12:
        return float("nan")
    cum_rank = np.cumsum(rank_values) / total_rank

    cum_w = np.concatenate([[0.0], cum_w])
    cum_rank = np.concatenate([[0.0], cum_rank])
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    lorenz_area = float(_trapz(cum_rank, cum_w))
    return float(1.0 - 2.0 * lorenz_area)


def _spearman_rho(actual_rate: np.ndarray, predicted_rate: np.ndarray) -> float:
    """Spearman rank correlation of predicted rate vs actual loss rate.

    Pure rank correlation — makes no distributional assumptions and is
    invariant to monotone transforms of either series.
    Range [-1, 1]; higher is better; 1 = perfect rank agreement.
    """

    n = len(actual_rate)
    if n < 4:
        return float("nan")
    # A constant series has no rank ordering — correlation is undefined.
    # Guard explicitly to avoid scipy's ConstantInputWarning on flat predictions
    # (e.g. the global-mean baseline predicts an identical rate for every policy).
    if np.ptp(predicted_rate) == 0 or np.ptp(actual_rate) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        result = spearmanr(predicted_rate, actual_rate)
        return float(result.statistic if hasattr(result, "statistic") else result.correlation)
    except Exception:
        return float("nan")


def _kendall_tau(actual_rate: np.ndarray, predicted_rate: np.ndarray) -> float:
    """Kendall's τ-b between predicted rate and actual loss rate.

    Counts concordant minus discordant pairs, normalised by the geometric
    mean of pairs that are not tied on each variable separately.
    Range [-1, 1]; higher is better.

    For large datasets (n > 5000) a 5k stratified subsample is used to keep
    runtime practical; the estimator is unbiased for the population τ.
    """

    n = len(actual_rate)
    if n < 4:
        return float("nan")
    try:
        from scipy.stats import kendalltau
        if n > 5000:
            # Subsample deterministically for speed (O(n²) naive implementation in scipy)
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n, size=5000, replace=False)
            x, y = predicted_rate[idx], actual_rate[idx]
        else:
            x, y = predicted_rate, actual_rate
        result = kendalltau(x, y)
        return float(result.statistic if hasattr(result, "statistic") else result.correlation)
    except Exception:
        return float("nan")


def _decile_lift_monotonicity(
    actual_rate: np.ndarray,
    predicted_rate: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Spearman correlation of per-decile mean actual rate vs decile order.

    Bins policies into 10 equal-exposure deciles ordered by predicted rate.
    Computes mean actual rate per decile, then Spearman of those 10 means
    vs decile rank (1–10).  A perfectly calibrated monotone model → 1.0.
    Aggregation within deciles averages out within-bucket claim noise,
    making this metric more stable than policy-level rank statistics.
    """

    n = len(actual_rate)
    if n < 20:
        return float("nan")
    safe_w = np.clip(weights, 1e-12, None)
    order = np.argsort(predicted_rate)
    w_s = safe_w[order]
    a_s = actual_rate[order]

    n_bins = 10
    cum_w = np.cumsum(w_s)
    total_w = cum_w[-1]
    if total_w < 1e-12:
        return float("nan")

    # Equal-exposure bin edges
    edges = [total_w * i / n_bins for i in range(1, n_bins)]
    bin_ids = np.searchsorted(cum_w, edges, side="left")
    bin_ids = np.clip(bin_ids, 0, n - 1)

    boundaries = [0] + list(bin_ids) + [n]
    decile_means: list[float] = []
    for i in range(n_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        if lo >= hi:
            continue
        w_bin = w_s[lo:hi]
        a_bin = a_s[lo:hi]
        w_sum = w_bin.sum()
        if w_sum < 1e-12:
            continue
        decile_means.append(float(np.average(a_bin, weights=w_bin)))

    if len(decile_means) < 4:
        return float("nan")
    ranks = np.arange(1, len(decile_means) + 1, dtype=float)
    means = np.array(decile_means, dtype=float)
    return _spearman_rho(means, ranks)


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
