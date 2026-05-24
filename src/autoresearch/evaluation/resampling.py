"""Repeated resampling, paired comparison, and promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.evaluation.metrics import regression_metrics


@dataclass(frozen=True)
class PromotionRules:
    """Configurable thresholds for champion/challenger promotion."""

    minimum_mean_lift: float
    minimum_win_rate: float
    bootstrap_lower_bound: float
    confidence_level: float


def repeated_scores(
    predictions: pd.DataFrame,
    *,
    eval_split: str,
    n_resamples: int,
    seed: int,
    resample_fraction: float = 1.0,
) -> pd.DataFrame:
    """Score one experiment across deterministic search-time resamples."""

    evaluation_frame = _ordinary_eval_frame(predictions, eval_split)
    sample_size = _sample_size(len(evaluation_frame), resample_fraction)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for resample_id in range(n_resamples):
        positions = rng.integers(0, len(evaluation_frame), size=sample_size)
        sample = evaluation_frame.iloc[positions]
        metrics = regression_metrics(sample["actual_claim_cost"], sample["predicted_claim_cost"], sample["exposure"])
        rows.append(
            {
                "resample_id": resample_id,
                "split": eval_split,
                "sample_size": sample_size,
                "score": metrics["rmse_pure_premium"],
                **metrics,
            }
        )
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare two experiments on identical search-time resamples."""

    champion = _ordinary_eval_frame(champion_predictions, eval_split)
    challenger = _ordinary_eval_frame(challenger_predictions, eval_split)
    paired = champion[
        ["record_id", "actual_claim_cost", "predicted_claim_cost", "exposure"]
    ].merge(
        challenger[["record_id", "predicted_claim_cost"]],
        on="record_id",
        suffixes=("_champion", "_challenger"),
        how="inner",
    )
    if paired.empty:
        raise ValueError("Champion and challenger predictions have no overlapping evaluation rows")

    sample_size = _sample_size(len(paired), resample_fraction)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for resample_id in range(n_resamples):
        positions = rng.integers(0, len(paired), size=sample_size)
        sample = paired.iloc[positions]
        champion_metrics = regression_metrics(
            sample["actual_claim_cost"],
            sample["predicted_claim_cost_champion"],
            sample["exposure"],
        )
        challenger_metrics = regression_metrics(
            sample["actual_claim_cost"],
            sample["predicted_claim_cost_challenger"],
            sample["exposure"],
        )
        champion_score = champion_metrics["rmse_pure_premium"]
        challenger_score = challenger_metrics["rmse_pure_premium"]
        lift = champion_score - challenger_score
        rows.append(
            {
                "resample_id": resample_id,
                "split": eval_split,
                "sample_size": sample_size,
                "champion_id": champion_id,
                "challenger_id": challenger_id,
                "champion_score": champion_score,
                "challenger_score": challenger_score,
                "lift": lift,
                "challenger_won": bool(lift > 0),
            }
        )

    per_resample = pd.DataFrame(rows)
    summary = {
        "champion_id": champion_id,
        "challenger_id": challenger_id,
        "eval_split": eval_split,
        "primary_metric": "rmse_pure_premium",
        "lower_is_better": True,
        "n_resamples": n_resamples,
        "resample_fraction": resample_fraction,
        "seed": seed,
        "mean_lift": float(per_resample["lift"].mean()),
        "median_lift": float(per_resample["lift"].median()),
        "std_lift": float(per_resample["lift"].std(ddof=0)),
        "challenger_win_rate": float(per_resample["challenger_won"].mean()),
    }
    return per_resample, summary


def bootstrap_lift_summary(
    lifts: pd.Series,
    *,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap the mean paired lift."""

    values = lifts.astype(float).to_numpy()
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty lift series")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        means[index] = sample.mean()

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "bootstrap_iterations": iterations,
        "seed": seed,
        "confidence_level": confidence_level,
        "mean_lift": float(values.mean()),
        "interval_lower": lower,
        "interval_upper": upper,
        "probability_challenger_outperforms": float((means > 0).mean()),
    }


def promotion_decision(
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    rules: PromotionRules,
) -> dict[str, Any]:
    """Make an explicit promote/inconclusive decision from configured rules."""

    checks = {
        "mean_lift": comparison_summary["mean_lift"] >= rules.minimum_mean_lift,
        "challenger_win_rate": comparison_summary["challenger_win_rate"] >= rules.minimum_win_rate,
        "bootstrap_lower_bound": bootstrap_summary["interval_lower"] >= rules.bootstrap_lower_bound,
    }
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
        "thresholds": {
            "minimum_mean_lift": rules.minimum_mean_lift,
            "minimum_win_rate": rules.minimum_win_rate,
            "bootstrap_lower_bound": rules.bootstrap_lower_bound,
            "confidence_level": rules.confidence_level,
        },
    }


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
