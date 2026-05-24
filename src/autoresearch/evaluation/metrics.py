"""Evaluation metrics and aggregation for deterministic baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(
    actual_claim: pd.Series,
    predicted_claim: pd.Series,
    exposure: pd.Series,
) -> dict[str, float]:
    """Compute burning-cost metrics from claim-cost predictions."""

    actual = actual_claim.astype(float).to_numpy()
    predicted = np.clip(predicted_claim.astype(float).to_numpy(), 0.0, None)
    exp = exposure.astype(float).to_numpy()
    exp = np.clip(exp, 1e-12, None)

    actual_pp = actual / exp
    predicted_pp = predicted / exp
    error = predicted_pp - actual_pp

    return {
        "mae_pure_premium": float(np.mean(np.abs(error))),
        "rmse_pure_premium": float(np.sqrt(np.mean(error**2))),
        "weighted_mae_claim_cost": float(np.average(np.abs(predicted - actual), weights=exp)),
        "mean_actual_pure_premium": float(np.average(actual_pp, weights=exp)),
        "mean_predicted_pure_premium": float(np.average(predicted_pp, weights=exp)),
        "total_actual_claim_cost": float(actual.sum()),
        "total_predicted_claim_cost": float(predicted.sum()),
        "exposure_sum": float(exp.sum()),
    }


def evaluate_predictions(predictions: pd.DataFrame, eval_splits: tuple[str, ...]) -> dict[str, Any]:
    """Evaluate split-level and aggregate scores without milestone holdout access."""

    if (predictions["split"] == "milestone_holdout").any():
        raise ValueError("Ordinary evaluation cannot include milestone_holdout rows")

    split_metrics: list[dict[str, Any]] = []
    for split, split_frame in predictions.groupby("split", sort=True):
        metrics = regression_metrics(
            split_frame["actual_claim_cost"],
            split_frame["predicted_claim_cost"],
            split_frame["exposure"],
        )
        metrics["split"] = split
        metrics["row_count"] = int(len(split_frame))
        split_metrics.append(metrics)

    eval_values = [
        item["rmse_pure_premium"]
        for item in split_metrics
        if item["split"] in set(eval_splits)
    ]
    if not eval_values:
        raise ValueError(f"No configured evaluation splits found: {eval_splits}")

    return {
        "primary_metric": "rmse_pure_premium",
        "lower_is_better": True,
        "ordinary_eval_splits": list(eval_splits),
        "split_metrics": split_metrics,
        "aggregate": {
            "mean_score": float(np.mean(eval_values)),
            "std_score": float(np.std(eval_values, ddof=0)),
            "split_count": len(eval_values),
        },
    }
