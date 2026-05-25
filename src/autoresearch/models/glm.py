"""Properly-specified GLM baselines for insurance burning-cost modelling.

All GLMs predict in original (claim-cost) space using exposure as sample
weight; no log/exp retransformation hacks are applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Column name constants (shared with baselines.py)
RECORD_ID = "record_id"
EXPOSURE = "exposure_term_a"
CLAIM_COUNT = "claim_count_signal_q"
CLAIM_EVENTS = "claim_event_count_l"
CLAIM_COST = "claim_cost_capped_active"
RAW_CLAIM_COST = "claim_cost_observed_k"
SPLIT = "split"

_NON_FEATURE = {RECORD_ID, CLAIM_COUNT, CLAIM_EVENTS, CLAIM_COST, RAW_CLAIM_COST, SPLIT}


def run_tweedie_glm(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    alpha: float = 1.0,
    power: float = 1.5,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a Tweedie GLM and predict claim cost in original space."""

    features = _feature_columns(train, feature_inclusions, feature_exclusions)
    model = _glm_pipeline(train[features], TweedieRegressor(power=power, alpha=alpha, link="log", max_iter=500))
    y = train[CLAIM_COST].astype(float) / train[EXPOSURE].astype(float).clip(lower=1e-9)
    model.fit(train[features], y, glm__sample_weight=train[EXPOSURE].astype(float))
    pred_pp = np.clip(model.predict(score[features]), 0.0, None)
    return pred_pp * score[EXPOSURE].astype(float).to_numpy(), {
        "model_family": "tweedie_glm",
        "target_strategy": "direct_pure_premium",
        "alpha": alpha,
        "power": power,
        "feature_columns": features,
    }


def run_frequency_severity_glm(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    freq_alpha: float = 1.0,
    sev_alpha: float = 1.0,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit Poisson (frequency) × Gamma (severity) GLM and predict claim cost."""

    features = _feature_columns(train, feature_inclusions, feature_exclusions)

    # Frequency: Poisson on claim rate
    freq_model = _glm_pipeline(train[features], PoissonRegressor(alpha=freq_alpha, max_iter=500))
    freq_target = train[CLAIM_COUNT].astype(float) / train[EXPOSURE].astype(float).clip(lower=1e-9)
    freq_model.fit(train[features], freq_target, glm__sample_weight=train[EXPOSURE].astype(float))
    pred_freq = np.clip(freq_model.predict(score[features]), 0.0, None)

    # Severity: Gamma on cost-per-claim for policies with claims
    claim_rows = train[(train[CLAIM_COUNT] > 0) & (train[CLAIM_COST] > 0)].copy()
    if claim_rows.empty:
        pred_sev = np.full(len(score), float(train[CLAIM_COST].mean()))
    else:
        sev_model = _glm_pipeline(claim_rows[features], GammaRegressor(alpha=sev_alpha, max_iter=500))
        sev_target = (
            claim_rows[CLAIM_COST].astype(float) / claim_rows[CLAIM_COUNT].astype(float).clip(lower=1e-9)
        )
        sev_model.fit(claim_rows[features], sev_target, glm__sample_weight=claim_rows[CLAIM_COUNT].astype(float))
        pred_sev = np.clip(sev_model.predict(score[features]), 0.0, None)

    predicted = pred_freq * pred_sev * score[EXPOSURE].astype(float).to_numpy()
    return predicted, {
        "model_family": "frequency_severity_glm",
        "target_strategy": "frequency_severity",
        "freq_alpha": freq_alpha,
        "sev_alpha": sev_alpha,
        "feature_columns": features,
    }


def _glm_pipeline(features: pd.DataFrame, glm) -> Pipeline:
    numeric = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("glm", glm),
    ])


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
