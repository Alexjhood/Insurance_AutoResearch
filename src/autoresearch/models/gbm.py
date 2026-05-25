"""Gradient-boosting Tweedie model for insurance burning-cost modelling.

Uses sklearn's HistGradientBoostingRegressor with Poisson loss (a Tweedie
special case with power=1), which handles missing values natively and works
well on claim data with many zeros.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


RECORD_ID = "record_id"
EXPOSURE = "exposure_term_a"
CLAIM_COUNT = "claim_count_signal_q"
CLAIM_EVENTS = "claim_event_count_l"
CLAIM_COST = "claim_cost_capped_active"
RAW_CLAIM_COST = "claim_cost_observed_k"

_NON_FEATURE = {RECORD_ID, CLAIM_COUNT, CLAIM_EVENTS, CLAIM_COST, RAW_CLAIM_COST}


def run_tweedie_gbm(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    max_iter: int = 500,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 200,
    l2_regularization: float = 0.0,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Tweedie GBM (Poisson loss) and predict claim cost."""

    features = _feature_columns(train, feature_inclusions, feature_exclusions)
    train_x, score_x = _encode(train[features], score[features])

    target = train[CLAIM_COST].astype(float) / train[EXPOSURE].astype(float).clip(lower=1e-9)
    weights = train[EXPOSURE].astype(float).to_numpy()

    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_state,
    )
    model.fit(train_x, target.to_numpy(), sample_weight=weights)
    pred_pp = np.clip(model.predict(score_x), 0.0, None)
    return pred_pp * score[EXPOSURE].astype(float).to_numpy(), {
        "model_family": "tweedie_gbm",
        "target_strategy": "direct_pure_premium",
        "max_iter": max_iter,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "feature_columns": features,
    }


def _feature_columns(
    frame: pd.DataFrame,
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
) -> list[str]:
    base = [c for c in frame.columns if c not in _NON_FEATURE]
    if feature_inclusions:
        unknown = sorted(set(feature_inclusions).difference(base))
        if unknown:
            raise ValueError(f"Unknown included features: {unknown}")
        base = [c for c in base if c in set(feature_inclusions)]
    if feature_exclusions:
        unknown = sorted(set(feature_exclusions).difference(base))
        if unknown:
            raise ValueError(f"Unknown excluded features: {unknown}")
        base = [c for c in base if c not in set(feature_exclusions)]
    if not base:
        raise ValueError("At least one feature column is required")
    return base


def _encode(train_features: pd.DataFrame, score_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Encode categoricals as ordinals for HistGradientBoosting."""

    numeric = [c for c in train_features.columns if pd.api.types.is_numeric_dtype(train_features[c])]
    categorical = [c for c in train_features.columns if c not in numeric]

    if not categorical:
        return train_features.to_numpy(dtype=float), score_features.to_numpy(dtype=float)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cat = enc.fit_transform(train_features[categorical])
    score_cat = enc.transform(score_features[categorical])

    train_num = train_features[numeric].to_numpy(dtype=float) if numeric else np.empty((len(train_features), 0))
    score_num = score_features[numeric].to_numpy(dtype=float) if numeric else np.empty((len(score_features), 0))

    train_x = np.hstack([train_num, train_cat]) if numeric else train_cat
    score_x = np.hstack([score_num, score_cat]) if numeric else score_cat
    return train_x, score_x
