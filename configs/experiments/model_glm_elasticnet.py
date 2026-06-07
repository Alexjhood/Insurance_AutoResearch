from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoresearch.models.calibration import apply_training_calibration

EXPOSURE = "exposure_term_a"
CLAIM_COST = "claim_cost_capped_active"
CLAIM_COUNT = "claim_count_signal_q"
CLAIM_EVENTS = "claim_event_count_l"
RECORD_ID = "record_id"

LEAKAGE_COLUMNS = {
    RECORD_ID,
    "split",
    CLAIM_COST,
    CLAIM_COUNT,
    CLAIM_EVENTS,
    "claim_cost_observed_k",
}


def _selected_features(
    frame: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
) -> list[str]:
    exclusions = set(feature_exclusions or [])
    exclusions |= LEAKAGE_COLUMNS

    if feature_inclusions:
        missing = [name for name in feature_inclusions if name not in frame.columns]
        if missing:
            raise ValueError(f"Requested feature_inclusions are missing from the frame: {missing}")
        features = [name for name in feature_inclusions if name not in exclusions]
    else:
        features = [
            column
            for column in frame.columns
            if column not in exclusions and not column.startswith("actual_") and not column.startswith("predicted_")
        ]

    features = [column for column in features if column in frame.columns and column not in exclusions]
    if not features:
        raise ValueError("No usable predictors remain after applying feature selections and exclusions")
    return features


def _numeric_and_categorical(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def _build_pipeline(
    frame: pd.DataFrame,
    features: list[str],
    *,
    estimator_family: str,
    hyperparameters: dict[str, object],
) -> tuple[Pipeline, list[str], list[str]]:
    numeric, categorical = _numeric_and_categorical(frame, features)

    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical,
            )
        )

    if not transformers:
        raise ValueError("No numeric or categorical transformers could be constructed")

    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

    if estimator_family == "tweedie_glm":
        model = TweedieRegressor(
            power=float(hyperparameters.get("tweedie_power", 1.5)),
            alpha=float(hyperparameters.get("alpha", 1.0)),
            link="log",
            max_iter=int(hyperparameters.get("max_iter", 5000)),
        )
    elif estimator_family == "elasticnet":
        model = ElasticNet(
            alpha=float(hyperparameters.get("alpha", 0.1)),
            l1_ratio=float(hyperparameters.get("l1_ratio", 0.5)),
            fit_intercept=True,
            max_iter=int(hyperparameters.get("max_iter", 10000)),
            tol=float(hyperparameters.get("tol", 1e-4)),
            random_state=int(hyperparameters.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unsupported estimator_family: {estimator_family!r}")

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("model", model),
    ])
    return pipeline, numeric, categorical


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters):
    estimator_family = str(hyperparameters.get("estimator_family", "tweedie_glm"))

    features = _selected_features(
        train,
        feature_inclusions=feature_inclusions,
        feature_exclusions=feature_exclusions,
    )
    pipeline, numeric, categorical = _build_pipeline(
        train,
        features,
        estimator_family=estimator_family,
        hyperparameters=hyperparameters,
    )

    exp_tr = train[EXPOSURE].astype(float).to_numpy()
    exp_sc = score[EXPOSURE].astype(float).to_numpy()
    y_rate = train[CLAIM_COST].astype(float).to_numpy() / np.clip(exp_tr, 1e-9, None)

    x_train = train[features]
    x_score = score[features]
    pipeline.fit(x_train, y_rate, model__sample_weight=exp_tr)

    rate_tr = np.clip(pipeline.predict(x_train), 0.0, None)
    rate_sc = np.clip(pipeline.predict(x_score), 0.0, None)
    pred_tr = rate_tr * exp_tr
    pred_sc = rate_sc * exp_sc

    pred_sc, calib = apply_training_calibration(pred_sc, pred_tr, train[CLAIM_COST].to_numpy())
    notes = {
        "model_family": f"scripted_{estimator_family}",
        "estimator_family": estimator_family,
        "alpha": float(hyperparameters.get("alpha", 1.0 if estimator_family == "tweedie_glm" else 0.1)),
        "l1_ratio": float(hyperparameters.get("l1_ratio", 0.5)) if estimator_family == "elasticnet" else None,
        "tweedie_power": float(hyperparameters.get("tweedie_power", 1.5)) if estimator_family == "tweedie_glm" else None,
        "selected_feature_count": len(features),
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "feature_exclusions": list(feature_exclusions or []),
        "feature_builder_module": hyperparameters.get("feature_builder_module"),
        "calib_factor": round(float(calib), 4),
    }
    return pred_sc, notes
