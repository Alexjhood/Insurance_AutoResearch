"""Global-mean baseline for the active target mode.

The no-model starting point for every research run: predicted target total is
the exposure-weighted mean target-per-unit-exposure on the training rows,
applied uniformly to every scored row.

This is intentionally the simplest possible "model" — it ignores every
feature and produces a constant target rate.  Every proposed experiment
develops relative to this baseline, so the research loop must demonstrate
real lift over a flat rate before introducing any structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoresearch.targets import BURNING_COST, FREQUENCY, SEVERITY, normalise_target_mode


EXPOSURE = "Exposure"
CLAIM_COST = "ClaimAmountCapped"
CLAIM_COUNT = "ClaimNb"
CLAIM_EVENTS = "ClaimAmountCount"


def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, dict]:
    target_mode = normalise_target_mode(hyperparameters.get("target_mode", BURNING_COST))
    # Weight/offset: exposure for population-wide modes, paid claim-event count
    # for severity (the dispatcher has already restricted train/score).
    weight_column = CLAIM_EVENTS if target_mode == SEVERITY else EXPOSURE
    train_weight = train[weight_column].astype(float)
    total_weight = float(train_weight.sum())
    if total_weight <= 0:
        raise ValueError(
            f"Total training {weight_column} must be positive for the global-mean baseline"
        )
    if target_mode == FREQUENCY:
        train_target = train[CLAIM_COUNT].astype(float)
        target_note = "mean_claim_frequency_per_exposure"
    else:
        # burning_cost and severity both average claim cost; severity divides by
        # claim count (cost per claim) rather than exposure.
        train_target = train[CLAIM_COST].astype(float)
        target_note = (
            "mean_severity_per_claim" if target_mode == SEVERITY
            else "mean_burning_cost_per_exposure"
        )
    total_target = float(train_target.sum())
    mean_target_rate = total_target / total_weight
    predicted = mean_target_rate * score[weight_column].astype(float).to_numpy()
    notes = {
        "model_family": "global_mean",
        "target_mode": target_mode,
        target_note: mean_target_rate,
        "train_total_target": total_target,
        "train_total_weight": total_weight,
        "weight_column": weight_column,
        "train_row_count": int(len(train)),
        "uses_features": False,
        "feature_inclusions": feature_inclusions,
        "feature_exclusions": feature_exclusions,
    }
    if target_mode == BURNING_COST:
        notes["mean_burning_cost_per_exposure"] = mean_target_rate
        notes["train_total_claim_cost"] = total_target
        notes["train_total_exposure"] = total_weight
    elif target_mode == SEVERITY:
        notes["mean_severity_per_claim"] = mean_target_rate
        notes["train_total_claim_cost"] = total_target
        notes["train_total_claim_count"] = total_weight
    else:
        notes["mean_claim_frequency_per_exposure"] = mean_target_rate
        notes["train_total_claim_count"] = total_target
        notes["train_total_exposure"] = total_weight
    return predicted, notes
