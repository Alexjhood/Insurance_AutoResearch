"""Global-mean burning-cost baseline.

The no-model starting point for every research run: predicted claim cost is
the exposure-weighted mean cost-per-unit-exposure on the training rows,
applied uniformly to every scored row.

This is intentionally the simplest possible "model" — it ignores every
feature and produces a constant burning cost.  Every proposed experiment
develops relative to this baseline, so the research loop must demonstrate
real lift over a flat rate before introducing any structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EXPOSURE = "exposure_term_a"
CLAIM_COST = "claim_cost_capped_active"


def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, dict]:
    train_exposure = train[EXPOSURE].astype(float)
    train_cost = train[CLAIM_COST].astype(float)
    total_exposure = float(train_exposure.sum())
    total_cost = float(train_cost.sum())
    if total_exposure <= 0:
        raise ValueError("Total training exposure must be positive for the global-mean baseline")
    mean_burning_cost = total_cost / total_exposure
    predicted_cost = mean_burning_cost * score[EXPOSURE].astype(float).to_numpy()
    notes = {
        "model_family": "global_mean",
        "mean_burning_cost_per_exposure": mean_burning_cost,
        "train_total_claim_cost": total_cost,
        "train_total_exposure": total_exposure,
        "train_row_count": int(len(train)),
        "uses_features": False,
        "feature_inclusions": feature_inclusions,
        "feature_exclusions": feature_exclusions,
    }
    return predicted_cost, notes
