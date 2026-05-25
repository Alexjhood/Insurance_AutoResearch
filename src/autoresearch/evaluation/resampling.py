"""Repeated resampling, k-fold CV, paired comparison, and promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.evaluation.metrics import full_metric_panel, lower_is_better, regression_metrics


@dataclass(frozen=True)
class PromotionRules:
    """Configurable thresholds for champion/challenger promotion."""

    minimum_mean_lift: float          # absolute lift (legacy, kept for compat)
    min_relative_lift: float          # lift as fraction of champion score (primary)
    min_absolute_lift: float          # hard floor in metric units
    minimum_win_rate: float
    bootstrap_lower_bound: float
    bootstrap_lower_bound_relative: float
    confidence_level: float
    max_predicted_to_actual_drift: float  # reject if calibration worsens > this
    require_diagnostics: bool
    bonferroni_lookback: int          # number of prior comparisons to adjust for


def repeated_scores(
    predictions: pd.DataFrame,
    *,
    eval_split: str,
    n_resamples: int,
    seed: int,
    resample_fraction: float = 1.0,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
) -> pd.DataFrame:
    """Score one experiment across deterministic search-time resamples."""

    evaluation_frame = _ordinary_eval_frame(predictions, eval_split)
    sample_size = _sample_size(len(evaluation_frame), resample_fraction)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for resample_id in range(n_resamples):
        positions = rng.integers(0, len(evaluation_frame), size=sample_size)
        sample = evaluation_frame.iloc[positions]
        metrics = full_metric_panel(
            sample["actual_claim_cost"], sample["predicted_claim_cost"], sample["exposure"],
            tweedie_power=tweedie_power,
        )
        rows.append({
            "resample_id": resample_id,
            "split": eval_split,
            "sample_size": sample_size,
            "score": metrics[primary_metric],
            **{k: v for k, v in metrics.items() if isinstance(v, float)},
        })
    return pd.DataFrame(rows)


def paired_comparison(
    champion_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    *,
    champion_id: str,
    challenger_id: str,
    eval_split: str,
    n_resamples: int,
    seed: int,
    resample_fraction: float = 1.0,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare two experiments on identical search-time resamples (paired)."""

    champion = _ordinary_eval_frame(champion_predictions, eval_split)
    challenger = _ordinary_eval_frame(challenger_predictions, eval_split)
    paired = champion[["record_id", "actual_claim_cost", "predicted_claim_cost", "exposure"]].merge(
        challenger[["record_id", "predicted_claim_cost"]],
        on="record_id",
        suffixes=("_champion", "_challenger"),
        how="inner",
    )
    if paired.empty:
        raise ValueError("Champion and challenger predictions have no overlapping evaluation rows")

    sample_size = _sample_size(len(paired), resample_fraction)
    metric_lower_is_better = lower_is_better(primary_metric)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for resample_id in range(n_resamples):
        positions = rng.integers(0, len(paired), size=sample_size)
        sample = paired.iloc[positions]
        champ_metrics = full_metric_panel(
            sample["actual_claim_cost"], sample["predicted_claim_cost_champion"], sample["exposure"],
            tweedie_power=tweedie_power,
        )
        chal_metrics = full_metric_panel(
            sample["actual_claim_cost"], sample["predicted_claim_cost_challenger"], sample["exposure"],
            tweedie_power=tweedie_power,
        )
        champ_score = champ_metrics[primary_metric]
        chal_score = chal_metrics[primary_metric]
        lift = champ_score - chal_score if metric_lower_is_better else chal_score - champ_score
        rows.append({
            "resample_id": resample_id,
            "split": eval_split,
            "sample_size": sample_size,
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "champion_score": champ_score,
            "challenger_score": chal_score,
            "lift": lift,
            "challenger_won": bool(lift > 0),
        })

    per_resample = pd.DataFrame(rows)
    summary = {
        "champion_id": champion_id,
        "challenger_id": challenger_id,
        "eval_split": eval_split,
        "primary_metric": primary_metric,
        "lower_is_better": metric_lower_is_better,
        "n_resamples": n_resamples,
        "resample_fraction": resample_fraction,
        "seed": seed,
        "mean_lift": float(per_resample["lift"].mean()),
        "median_lift": float(per_resample["lift"].median()),
        "std_lift": float(per_resample["lift"].std(ddof=0)),
        "challenger_win_rate": float(per_resample["challenger_won"].mean()),
        "champion_mean_score": float(per_resample["champion_score"].mean()),
        "challenger_mean_score": float(per_resample["challenger_score"].mean()),
    }
    return per_resample, summary


def bootstrap_lift_summary(
    lifts: pd.Series,
    *,
    iterations: int,
    seed: int,
    confidence_level: float,
    n_comparisons: int = 1,
) -> dict[str, Any]:
    """Bootstrap the mean paired lift, with optional Bonferroni correction."""

    values = lifts.astype(float).to_numpy()
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty lift series")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Bonferroni: widen alpha by number of comparisons
    adjusted_level = 1.0 - (1.0 - confidence_level) / max(1, n_comparisons)
    adjusted_level = min(adjusted_level, 0.9999)

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()

    alpha = 1.0 - adjusted_level
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "bootstrap_iterations": iterations,
        "seed": seed,
        "confidence_level": confidence_level,
        "adjusted_confidence_level": float(adjusted_level),
        "n_comparisons_bonferroni": n_comparisons,
        "mean_lift": float(values.mean()),
        "interval_lower": lower,
        "interval_upper": upper,
        "probability_challenger_outperforms": float((means > 0).mean()),
    }


def cv_repeated_scores(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    model_factory,
    n_folds: int,
    n_repeats: int = 1,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """K-fold CV scores with variance decomposition.

    ``model_factory(train_df, val_df)`` must return ``(predictions_df)`` containing
    actual_claim_cost, predicted_claim_cost, exposure columns.
    """

    data = frame.merge(fold_assignments[["record_id", "fold"]], on="record_id", how="inner")
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    fold_scores: list[float] = []

    for repeat in range(n_repeats):
        fold_fold_scores = []
        for fold in range(1, n_folds + 1):
            val_ids = set(data.loc[data["fold"] == fold, "record_id"].tolist())
            train_data = data[~data["record_id"].isin(val_ids)].copy()
            val_data = data[data["record_id"].isin(val_ids)].copy()
            if n_repeats > 1:
                # Reshuffle training rows within each repeat
                train_data = train_data.sample(frac=1.0, random_state=rng.integers(0, 2**31)).copy()
            preds = model_factory(train_data, val_data)
            metrics = full_metric_panel(
                preds["actual_claim_cost"], preds["predicted_claim_cost"], preds["exposure"],
                tweedie_power=tweedie_power,
            )
            score = metrics[primary_metric]
            fold_fold_scores.append(score)
            rows.append({
                "repeat": repeat,
                "fold": fold,
                "n_val": len(val_data),
                "score": score,
                **{k: v for k, v in metrics.items() if isinstance(v, float)},
            })
        fold_scores.extend(fold_fold_scores)

    cv_frame = pd.DataFrame(rows)

    # Variance decomposition: between-fold vs total
    fold_means = cv_frame.groupby("fold")["score"].mean().to_numpy()
    between_fold_var = float(np.var(fold_means, ddof=0))
    total_var = float(np.var(cv_frame["score"].to_numpy(), ddof=0))
    within_fold_var = max(0.0, total_var - between_fold_var)
    ratio_between = between_fold_var / total_var if total_var > 0 else float("nan")

    summary = {
        "primary_metric": primary_metric,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "mean_score": float(cv_frame["score"].mean()),
        "std_score": float(cv_frame["score"].std(ddof=0)),
        "between_fold_variance": between_fold_var,
        "within_fold_variance": within_fold_var,
        "total_variance": total_var,
        "ratio_between_to_total": ratio_between,
        "warning_between_dominates": ratio_between > 0.5 if not np.isnan(ratio_between) else False,
    }
    return cv_frame, summary


def promotion_decision(
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    rules: PromotionRules,
    *,
    challenger_diagnostics: dict[str, Any] | None = None,
    n_prior_comparisons: int = 0,
) -> dict[str, Any]:
    """Make an explicit promote/inconclusive decision from configured rules."""

    champion_score = comparison_summary.get("champion_mean_score", 0.0) or 1e-9
    champion_scale = abs(champion_score) or 1e-9
    mean_lift = comparison_summary["mean_lift"]
    win_rate = comparison_summary["challenger_win_rate"]
    boot_lower = bootstrap_summary["interval_lower"]
    n_resamples = comparison_summary.get("n_resamples", 30)

    # Relative lift fractions
    relative_lift = mean_lift / champion_scale
    boot_lower_relative = boot_lower / champion_scale

    # Minimum detectable effect estimate (rough 95% one-sided)
    std_lift = comparison_summary.get("std_lift", 0.0)
    mde_relative = (2 * std_lift / max(n_resamples ** 0.5, 1)) / champion_scale

    if relative_lift > (mde_relative or 1.0):
        power_note = "effect_above_mde"
    elif relative_lift > rules.min_relative_lift:
        power_note = "effect_below_mde"
    else:
        power_note = "effect_below_min_relative_lift"

    checks: dict[str, bool] = {
        "mean_lift_positive": mean_lift >= rules.minimum_mean_lift,
        "relative_lift": relative_lift >= rules.min_relative_lift,
        "absolute_lift": mean_lift >= rules.min_absolute_lift,
        "challenger_win_rate": win_rate >= rules.minimum_win_rate,
        "bootstrap_lower_bound": boot_lower >= rules.bootstrap_lower_bound,
        "bootstrap_lower_bound_relative": boot_lower_relative >= rules.bootstrap_lower_bound_relative,
    }

    # Calibration check
    calibration_ok = True
    if rules.max_predicted_to_actual_drift < 1.0 and challenger_diagnostics:
        chal_ratio = _get_pred_to_actual(challenger_diagnostics)
        if chal_ratio is not None:
            drift = abs(1.0 - chal_ratio)
            calibration_ok = drift <= rules.max_predicted_to_actual_drift
    checks["calibration_ok"] = calibration_ok

    # Diagnostics presence check
    if rules.require_diagnostics and challenger_diagnostics:
        checks["diagnostics_present"] = not bool(challenger_diagnostics.get("error"))
    elif rules.require_diagnostics:
        checks["diagnostics_present"] = False

    promoted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]

    if promoted:
        decision = "promote"
        rationale = "Challenger passed all configured promotion thresholds."
    else:
        decision = "inconclusive"
        rationale = "No promotion: failed thresholds " + ", ".join(failed) + "."

    return {
        "decision": decision,
        "promoted": promoted,
        "rationale": rationale,
        "checks": checks,
        "effect_size": {
            "mean_lift": mean_lift,
            "relative_lift": relative_lift,
            "mde_relative": mde_relative,
            "power_note": power_note,
            "n_resamples": n_resamples,
        },
        "thresholds": {
            "minimum_mean_lift": rules.minimum_mean_lift,
            "min_relative_lift": rules.min_relative_lift,
            "min_absolute_lift": rules.min_absolute_lift,
            "minimum_win_rate": rules.minimum_win_rate,
            "bootstrap_lower_bound": rules.bootstrap_lower_bound,
            "bootstrap_lower_bound_relative": rules.bootstrap_lower_bound_relative,
            "confidence_level": rules.confidence_level,
            "max_predicted_to_actual_drift": rules.max_predicted_to_actual_drift,
        },
        "n_prior_comparisons_bonferroni": n_prior_comparisons,
    }


def _get_pred_to_actual(diagnostics: dict[str, Any]) -> float | None:
    decile_table = diagnostics.get("calibration_by_pred_decile", [])
    if not decile_table:
        return None
    total_actual = sum(d.get("actual_pp", 0) * d.get("n", 0) for d in decile_table)
    total_pred = sum(d.get("pred_pp", 0) * d.get("n", 0) for d in decile_table)
    return total_pred / total_actual if total_actual > 0 else None


# ── private helpers ───────────────────────────────────────────────────────────

def _ordinary_eval_frame(predictions: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    if (predictions["split"] == "milestone_holdout").any():
        raise ValueError("Repeated ordinary evaluation cannot include milestone_holdout rows")
    frame = predictions[predictions["split"] == eval_split].copy()
    if frame.empty:
        raise ValueError(f"No rows found for evaluation split {eval_split!r}")
    return frame.reset_index(drop=True)


def _sample_size(row_count: int, resample_fraction: float) -> int:
    if row_count <= 0:
        raise ValueError("Cannot resample an empty frame")
    if resample_fraction <= 0:
        raise ValueError("resample_fraction must be positive")
    return max(1, int(round(row_count * resample_fraction)))
