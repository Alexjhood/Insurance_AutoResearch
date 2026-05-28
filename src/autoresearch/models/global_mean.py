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

from autoresearch.targets import BURNING_COST, FREQUENCY, normalise_target_mode


EXPOSURE = "exposure_term_a"
CLAIM_COST = "claim_cost_capped_active"
CLAIM_COUNT = "claim_count_signal_q"


def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, dict]:
    target_mode = normalise_target_mode(hyperparameters.get("target_mode", BURNING_COST))
    train_exposure = train[EXPOSURE].astype(float)
    total_exposure = float(train_exposure.sum())
    if total_exposure <= 0:
        raise ValueError("Total training exposure must be positive for the global-mean baseline")
    if target_mode == FREQUENCY:
        train_target = train[CLAIM_COUNT].astype(float)
        total_target = float(train_target.sum())
        mean_target_rate = total_target / total_exposure
        predicted = mean_target_rate * score[EXPOSURE].astype(float).to_numpy()
        target_note = "mean_claim_frequency_per_exposure"
    else:
        train_target = train[CLAIM_COST].astype(float)
        total_target = float(train_target.sum())
        mean_target_rate = total_target / total_exposure
        predicted = mean_target_rate * score[EXPOSURE].astype(float).to_numpy()
        target_note = "mean_burning_cost_per_exposure"
    notes = {
        "model_family": "global_mean",
        "target_mode": target_mode,
        target_note: mean_target_rate,
        "train_total_target": total_target,
        "train_total_exposure": total_exposure,
        "train_row_count": int(len(train)),
        "uses_features": False,
        "feature_inclusions": feature_inclusions,
        "feature_exclusions": feature_exclusions,
    }
    if target_mode == BURNING_COST:
        notes["mean_burning_cost_per_exposure"] = mean_target_rate
        notes["train_total_claim_cost"] = total_target
    else:
        notes["mean_claim_frequency_per_exposure"] = mean_target_rate
        notes["train_total_claim_count"] = total_target
    return predicted, notes
