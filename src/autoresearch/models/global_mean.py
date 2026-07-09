"""Global-mean baseline for the active target mode.

The no-model starting point for every research run: the predicted target total is
the weight-weighted mean target-per-unit-weight on the training rows, applied
uniformly to every scored row.

This is intentionally the simplest possible "model" — it ignores every feature
and produces a constant target rate. Every proposed experiment develops relative
to this baseline, so the research loop must demonstrate real lift over a flat
rate before introducing any structure.

Fully dataset-generic: the source/weight columns come from the active
:class:`~autoresearch.targets.TargetSpec` (French Exposure/ClaimAmountCapped,
AllState unit weight / Claim_Amount, Porto unit weight / target, …).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autoresearch.targets import BURNING_COST, normalise_target_mode, target_spec


def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, dict]:
    target_mode = normalise_target_mode(hyperparameters.get("target_mode", BURNING_COST))
    spec = target_spec(target_mode)
    weight_column = spec.weight_column
    source_column = spec.source_column

    train_weight = train[weight_column].astype(float)
    total_weight = float(train_weight.sum())
    if total_weight <= 0:
        raise ValueError(
            f"Total training {weight_column} must be positive for the global-mean baseline"
        )
    train_target = train[source_column].astype(float)
    total_target = float(train_target.sum())
    mean_target_rate = total_target / total_weight
    predicted = mean_target_rate * score[weight_column].astype(float).to_numpy()

    rate_key = f"mean_{spec.rate_slug}" if spec.rate_slug else "mean_target_rate"
    notes = {
        "model_family": "global_mean",
        "target_mode": target_mode,
        rate_key: mean_target_rate,
        "mean_target_rate": mean_target_rate,
        "train_total_target": total_target,
        "train_total_weight": total_weight,
        "weight_column": weight_column,
        "source_column": source_column,
        "train_row_count": int(len(train)),
        "uses_features": False,
        "feature_inclusions": feature_inclusions,
        "feature_exclusions": feature_exclusions,
    }
    return predicted, notes
