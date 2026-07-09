"""Recipe interpreter — turns a declarative recipe into fitted predictions.

Exposes the standard model hook so it plugs straight into the dispatcher's open
registry:

    fit_predict(train, score, *, feature_inclusions=None,
                feature_exclusions=None, **hyperparameters)
        -> (Prediction, notes, interpret_fn)

The recipe is read from ``hyperparameters['recipe']`` and the active target mode
from ``hyperparameters['target_mode']``. Stages are fit on the target *rate*; the
framework (#7) converts rate → total via exposure and calibrates, so the
interpreter never touches exposure conversion or calibration itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoresearch.models.prediction import Prediction, validate_objective_labels
from autoresearch.models.recipe.registry import (
    FitContext,
    RecipeError,
    get_encoding,
    get_estimator,
)
from autoresearch.models.recipe.schema import recipe_summary, validate_recipe
from autoresearch.targets import BURNING_COST, FREQUENCY, SEVERITY, normalise_target_mode


EXPOSURE = "Exposure"
CLAIM_COST = "ClaimAmountCapped"
CLAIM_COUNT = "ClaimNb"
CLAIM_EVENTS = "ClaimAmountCount"
RAW_CLAIM_COST = "ClaimAmount"
RECORD_ID = "record_id"

_LEAKAGE = frozenset({
    RECORD_ID, EXPOSURE, CLAIM_COST, CLAIM_COUNT, CLAIM_EVENTS, RAW_CLAIM_COST,
    "split", "target_mode",
})


def _select_features(
    frame: pd.DataFrame,
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
    id_columns: list[str] | None = None,
) -> list[str]:
    exclusions = set(feature_exclusions or []) | set(id_columns or []) | _LEAKAGE
    if feature_inclusions:
        missing = [c for c in feature_inclusions if c not in frame.columns]
        if missing:
            raise RecipeError(f"feature_inclusions missing from data: {missing}")
        features = [c for c in feature_inclusions if c not in exclusions]
    else:
        features = [
            c for c in frame.columns
            if c not in exclusions
            and not c.startswith("actual_")
            and not c.startswith("predicted_")
        ]
    if not features:
        raise RecipeError("No usable predictors remain after feature selection")
    return features


def _split_types(frame: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in features:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _as_categorical(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in categorical:
        out[col] = out[col].astype("category")
    return out


def _take(matrix: Any, idx: np.ndarray) -> Any:
    if isinstance(matrix, pd.DataFrame):
        return matrix.iloc[idx]
    return matrix[idx]


def _run_stage(
    stage: dict[str, Any],
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    features: list[str],
    y_rate: np.ndarray,
    weight: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = get_estimator(stage["estimator"])
    objective = stage["objective"]
    encoding = stage.get("encoding") or spec.default_encoding
    enc_spec = get_encoding(encoding)
    params = dict(stage.get("params") or {})
    early_stopping = stage.get("early_stopping")
    if early_stopping and not spec.supports_early_stopping:
        early_stopping = None

    validate_objective_labels(objective, y_rate, context=stage["estimator"])

    numeric, categorical = _split_types(train_df, features)
    encoded_column_indices = None
    if enc_spec.native:
        X_train_full = _as_categorical(train_df[features], categorical)
        X_score = _as_categorical(score_df[features], categorical)
        cat_features = categorical or None
    else:
        transformer = enc_spec.builder(numeric, categorical)
        X_train_full = transformer.fit_transform(train_df[features])
        X_score = transformer.transform(score_df[features])
        cat_features = None
        if encoding == "ordinal":
            # The ordinal ColumnTransformer emits the numeric block first, then
            # the categorical block, one column per feature in this order.
            encoded_column_indices = {
                name: i for i, name in enumerate([*numeric, *categorical])
            }

    y_full = np.asarray(y_rate, dtype=float)
    w_full = np.asarray(weight, dtype=float)

    X_tr, y_tr, w_tr = X_train_full, y_full, w_full
    X_val = y_val = w_val = None
    # Only carve an external validation holdout for estimators that consume an
    # eval_set; self-validating estimators (e.g. hist_gbm) make their own split,
    # so an external one would needlessly shrink their training data.
    needs_external_val = (
        early_stopping and spec.supports_early_stopping and not spec.self_validates_early_stopping
    )
    if needs_external_val and len(y_full) >= 50:
        from sklearn.model_selection import train_test_split

        tr_idx, val_idx = train_test_split(
            np.arange(len(y_full)), test_size=0.2, random_state=seed
        )
        X_tr, X_val = _take(X_train_full, tr_idx), _take(X_train_full, val_idx)
        y_tr, y_val = y_full[tr_idx], y_full[val_idx]
        w_tr, w_val = w_full[tr_idx], w_full[val_idx]

    ctx = FitContext(
        objective=objective, params=params,
        X_train=X_tr, y_train=y_tr, w_train=w_tr,
        X_val=X_val, y_val=y_val, w_val=w_val,
        categorical_features=cat_features, early_stopping=early_stopping,
        encoded_column_indices=encoded_column_indices,
    )
    predictor, notes = spec.fit(ctx)
    pred = np.clip(np.asarray(predictor.predict(X_score), dtype=float), 0.0, None)
    notes["encoding"] = encoding
    return pred, notes


def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    **hyperparameters: Any,
) -> tuple[Prediction, dict[str, Any], None]:
    recipe = hyperparameters.get("recipe")
    if not isinstance(recipe, dict):
        raise RecipeError("recipe model family requires a 'recipe' object in the model config")
    target_mode = normalise_target_mode(hyperparameters.get("target_mode", BURNING_COST))

    errors = validate_recipe(recipe, target_mode=target_mode)
    if errors:
        raise RecipeError("Invalid recipe:\n- " + "\n- ".join(errors))

    features = _select_features(
        train,
        feature_inclusions,
        feature_exclusions,
        hyperparameters.get("_id_columns"),
    )
    structure = recipe.get("structure", "direct")
    train_exposure = train[EXPOSURE].astype(float).to_numpy()
    score_exposure = score[EXPOSURE].astype(float).to_numpy()

    notes: dict[str, Any] = {
        "model_family": "recipe",
        "recipe": recipe,
        "recipe_summary": recipe_summary(recipe),
        "structure": structure,
        "n_features": len(features),
        "target_mode": target_mode,
    }

    if structure == "frequency_severity":
        pred_rate, stage_notes = _fit_frequency_severity(recipe, train, score, features)
        notes["stages"] = stage_notes
    elif target_mode == SEVERITY:
        # Direct severity: cost per claim on claim rows (the dispatcher has
        # already filtered to ClaimNb > 0), claim-count-weighted. The framework
        # multiplies the returned rate by claim count to recover the cost total.
        pred_rate, stage_notes = _fit_direct_severity(recipe, train, score, features)
        notes.update({f"stage_{k}": v for k, v in stage_notes.items()})
    else:
        if target_mode == FREQUENCY:
            y_rate = train[CLAIM_COUNT].astype(float).to_numpy() / np.clip(train_exposure, 1e-12, None)
        else:
            y_rate = train[CLAIM_COST].astype(float).to_numpy() / np.clip(train_exposure, 1e-12, None)
        pred_rate, stage_notes = _run_stage(
            recipe, train, score, features, y_rate, train_exposure
        )
        notes.update({f"stage_{k}": v for k, v in stage_notes.items()})

    prediction = Prediction(values=pred_rate, unit="rate", calibrate=True)
    return prediction, notes, None


def _fit_direct_severity(
    recipe: dict[str, Any],
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-stage severity: cost per paid claim event, event-count-weighted.

    ``train`` is already restricted to positive paid-claim rows by the
    dispatcher. Mirrors the severity stage of the freq×sev decomposition but
    returns the cost-per-claim rate directly (no frequency multiplier).
    """

    count = train[CLAIM_EVENTS].astype(float).to_numpy()
    cost = train[CLAIM_COST].astype(float).to_numpy()
    per_claim = cost / np.clip(count, 1e-12, None)
    if recipe.get("objective") == "gamma":
        positive = per_claim > 0
        if not positive.any():
            raise RecipeError("Severity gamma model has no strictly positive cost rows")
        train = train[positive]
        per_claim = per_claim[positive]
        count = count[positive]
    return _run_stage(recipe, train, score, features, per_claim, count)


def _fit_frequency_severity(
    recipe: dict[str, Any],
    train: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    stages = recipe["stages"]
    train_exposure = train[EXPOSURE].astype(float).to_numpy()
    train_count = train[CLAIM_COUNT].astype(float).to_numpy()

    # Frequency stage: claims per exposure, exposure-weighted.
    freq_rate_train = train_count / np.clip(train_exposure, 1e-12, None)
    pred_freq, freq_notes = _run_stage(
        stages["frequency"], train, score, features, freq_rate_train, train_exposure
    )

    # Severity stage: cost per paid claim event on positive paid-claim rows.
    sev_stage = stages["severity"]
    train_event_count = train[CLAIM_EVENTS].astype(float).to_numpy()
    claim_mask = train_event_count > 0
    sev_train = train[claim_mask].copy()
    if sev_train.empty:
        raise RecipeError("Severity stage has no training rows with ClaimAmountCount > 0")
    sev_cost = sev_train[CLAIM_COST].astype(float).to_numpy()
    sev_count = sev_train[CLAIM_EVENTS].astype(float).to_numpy()
    sev_per_claim = sev_cost / np.clip(sev_count, 1e-12, None)
    if sev_stage.get("objective") == "gamma":
        positive = sev_per_claim > 0
        if not positive.any():
            raise RecipeError("Severity gamma stage has no strictly positive cost rows")
        sev_train = sev_train[positive]
        sev_per_claim = sev_per_claim[positive]
        sev_count = sev_count[positive]
    pred_sev, sev_notes = _run_stage(
        sev_stage, sev_train, score, features, sev_per_claim, sev_count
    )

    pred_pp_rate = pred_freq * pred_sev
    return pred_pp_rate, {"frequency": freq_notes, "severity": sev_notes}
