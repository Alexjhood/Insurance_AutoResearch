"""Simple deterministic baseline modelling paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RECORD_ID = "record_id"
EXPOSURE = "exposure_term_a"
CLAIM_COUNT = "claim_count_signal_q"
CLAIM_EVENTS = "claim_event_count_l"
CLAIM_COST = "claim_cost_capped_active"
RAW_CLAIM_COST = "claim_cost_observed_k"
SPLIT = "split"


@dataclass(frozen=True)
class BaselineResult:
    """Predictions and model notes from a fitted deterministic baseline."""

    predictions: pd.DataFrame
    model_notes: dict[str, object]


def run_baseline_model(
    frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    target_strategy: str,
    train_split: str,
    score_splits: tuple[str, ...],
    alpha: float = 1.0,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
) -> BaselineResult:
    """Fit one supported baseline and return ordinary-search predictions."""

    data = frame.merge(split_frame[["record_id", "split"]], on="record_id", how="inner")
    if (data["split"] == "milestone_holdout").any() and "milestone_holdout" in score_splits:
        raise ValueError("Milestone holdout cannot be requested for ordinary scoring")

    train = data[data["split"] == train_split].copy()
    score = data[data["split"].isin((train_split, *score_splits))].copy()
    if train.empty:
        raise ValueError(f"Training split {train_split!r} is empty")
    if score.empty:
        raise ValueError("No rows available for scoring")

    if target_strategy == "direct_pure_premium":
        predicted_claim = _fit_direct(train, score, alpha, feature_inclusions, feature_exclusions)
    elif target_strategy == "frequency_severity":
        predicted_claim = _fit_frequency_severity(train, score, alpha, feature_inclusions, feature_exclusions)
    else:
        raise ValueError(f"Unsupported target strategy: {target_strategy}")

    predictions = pd.DataFrame(
        {
            "record_id": score[RECORD_ID].to_numpy(),
            "split": score["split"].to_numpy(),
            "exposure": score[EXPOSURE].astype(float).to_numpy(),
            "actual_claim_cost": score[CLAIM_COST].astype(float).to_numpy(),
            "actual_claim_cost_uncapped": score[RAW_CLAIM_COST].astype(float).to_numpy(),
            "predicted_claim_cost": np.clip(predicted_claim, 0.0, None),
        }
    )
    predictions["actual_pure_premium"] = predictions["actual_claim_cost"] / predictions["exposure"].clip(lower=1e-12)
    predictions["predicted_pure_premium"] = predictions["predicted_claim_cost"] / predictions["exposure"].clip(lower=1e-12)

    return BaselineResult(
        predictions=predictions,
        model_notes={
            "target_strategy": target_strategy,
            "model_family": "regularized_linear",
            "alpha": alpha,
            "feature_columns": _feature_columns(frame, feature_inclusions, feature_exclusions),
            "feature_inclusions": feature_inclusions,
            "feature_exclusions": feature_exclusions,
            "train_split": train_split,
            "score_splits": list(score_splits),
        },
    )


def _fit_direct(
    train: pd.DataFrame,
    score: pd.DataFrame,
    alpha: float,
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
) -> np.ndarray:
    features = _feature_columns(train, feature_inclusions, feature_exclusions)
    model = _model_pipeline(train[features], alpha)
    y = np.log1p(train[CLAIM_COST].astype(float) / train[EXPOSURE].astype(float).clip(lower=1e-12))
    model.fit(train[features], y, ridge__sample_weight=train[EXPOSURE].astype(float))
    predicted_pp = np.expm1(model.predict(score[features]))
    return predicted_pp * score[EXPOSURE].astype(float).to_numpy()


def _fit_frequency_severity(
    train: pd.DataFrame,
    score: pd.DataFrame,
    alpha: float,
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
) -> np.ndarray:
    features = _feature_columns(train, feature_inclusions, feature_exclusions)
    freq_model = _model_pipeline(train[features], alpha)
    freq_target = np.log1p(train[CLAIM_COUNT].astype(float) / train[EXPOSURE].astype(float).clip(lower=1e-12))
    freq_model.fit(train[features], freq_target, ridge__sample_weight=train[EXPOSURE].astype(float))

    claim_rows = train[train[CLAIM_COUNT] > 0].copy()
    if claim_rows.empty:
        severity_mean = 0.0
        predicted_severity = np.zeros(len(score))
    else:
        sev_model = _model_pipeline(claim_rows[features], alpha)
        severity_target = np.log1p(
            claim_rows[CLAIM_COST].astype(float) / claim_rows[CLAIM_COUNT].astype(float).clip(lower=1e-12)
        )
        sev_model.fit(claim_rows[features], severity_target, ridge__sample_weight=claim_rows[CLAIM_COUNT].astype(float))
        predicted_severity = np.expm1(sev_model.predict(score[features]))
        severity_mean = float(np.average(np.expm1(severity_target), weights=claim_rows[CLAIM_COUNT].astype(float)))

    predicted_freq = np.expm1(freq_model.predict(score[features]))
    predicted_freq = np.clip(predicted_freq, 0.0, None)
    predicted_severity = np.clip(predicted_severity, 0.0, None)
    if not np.isfinite(predicted_severity).all():
        predicted_severity = np.nan_to_num(predicted_severity, nan=severity_mean, posinf=severity_mean)
    return predicted_freq * predicted_severity * score[EXPOSURE].astype(float).to_numpy()


def _model_pipeline(features: pd.DataFrame, alpha: float) -> Pipeline:
    numeric = [column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])]
    categorical = [column for column in features.columns if column not in numeric]
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                            numeric,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                                ]
                            ),
                            categorical,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("ridge", Ridge(alpha=alpha, random_state=0)),
        ]
    )


def _feature_columns(
    frame: pd.DataFrame,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
) -> list[str]:
    excluded = {RECORD_ID, CLAIM_COUNT, CLAIM_EVENTS, CLAIM_COST, RAW_CLAIM_COST, SPLIT}
    base = [column for column in frame.columns if column not in excluded]
    if feature_inclusions:
        unknown = sorted(set(feature_inclusions).difference(base))
        if unknown:
            raise ValueError(f"Unknown included feature columns: {unknown}")
        base = [column for column in base if column in set(feature_inclusions)]
    if feature_exclusions:
        unknown = sorted(set(feature_exclusions).difference(base))
        if unknown:
            raise ValueError(f"Unknown excluded feature columns: {unknown}")
        base = [column for column in base if column not in set(feature_exclusions)]
    if not base:
        raise ValueError("At least one feature column is required")
    return base
