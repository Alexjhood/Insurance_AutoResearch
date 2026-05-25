"""Model dispatcher: routes model_family/target_strategy to the right implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# Column name constants (keep consistent across all model modules)
RECORD_ID = "record_id"
EXPOSURE = "exposure_term_a"
CLAIM_COUNT = "claim_count_signal_q"
CLAIM_EVENTS = "claim_event_count_l"
CLAIM_COST = "claim_cost_capped_active"
RAW_CLAIM_COST = "claim_cost_observed_k"


@dataclass(frozen=True)
class ModelResult:
    """Predictions and notes from any model family."""

    predictions: pd.DataFrame
    model_notes: dict[str, Any]


def dispatch_model(
    frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    *,
    model_family: str,
    target_strategy: str,
    train_split: str,
    score_splits: tuple[str, ...],
    hyperparameters: dict[str, Any] | None = None,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
) -> ModelResult:
    """Fit the requested model family and return scored predictions.

    Works identically for the legacy Ridge, the built-in GLM/GBM families,
    and any new family registered as ``autoresearch.models.<model_family>``.

    Feature engineering: if ``hyperparameters`` contains a
    ``feature_builder_module`` key, the named module's ``build_features``
    function is applied to the full data frame before splitting into
    train / score partitions.
    """

    hp = dict(hyperparameters or {})

    # Apply optional feature builder before splitting
    feature_builder_module = hp.pop("feature_builder_module", None)
    if feature_builder_module:
        import importlib
        builder = importlib.import_module(feature_builder_module)
        frame = builder.build_features(frame)

    data = frame.merge(split_frame[["record_id", "split"]], on="record_id", how="inner")
    if len(data) < len(frame):
        raise ValueError(f"{len(frame) - len(data)} rows dropped during split merge — split pack may be stale")
    if (data["split"] == "milestone_holdout").any() and "milestone_holdout" in score_splits:
        raise ValueError("Milestone holdout cannot be requested for ordinary scoring")

    train = data[data["split"] == train_split].copy()
    score = data[data["split"].isin((train_split, *score_splits))].copy()
    if train.empty:
        raise ValueError(f"Training split {train_split!r} is empty")
    if score.empty:
        raise ValueError("No rows available for scoring")

    predicted_cost, notes = _call_model(
        model_family, target_strategy, train, score, hp, feature_inclusions, feature_exclusions
    )

    predictions = pd.DataFrame({
        RECORD_ID: score[RECORD_ID].to_numpy(),
        "split": score["split"].to_numpy(),
        "exposure": score[EXPOSURE].astype(float).to_numpy(),
        "actual_claim_cost": score[CLAIM_COST].astype(float).to_numpy(),
        "actual_claim_cost_uncapped": score[RAW_CLAIM_COST].astype(float).to_numpy(),
        "predicted_claim_cost": np.clip(predicted_cost, 0.0, None),
    })
    exp = predictions["exposure"].clip(lower=1e-12)
    predictions["actual_pure_premium"] = predictions["actual_claim_cost"] / exp
    predictions["predicted_pure_premium"] = predictions["predicted_claim_cost"] / exp
    return ModelResult(predictions=predictions, model_notes=notes)


def _call_model(
    model_family: str,
    target_strategy: str,
    train: pd.DataFrame,
    score: pd.DataFrame,
    hp: dict[str, Any],
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch to the correct model implementation, returning (predicted_cost, notes)."""

    if model_family == "tweedie_glm":
        from autoresearch.models.glm import run_tweedie_glm
        return run_tweedie_glm(
            train, score,
            alpha=float(hp.get("alpha", 1.0)),
            power=float(hp.get("power", 1.5)),
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
        )

    if model_family == "frequency_severity_glm":
        from autoresearch.models.glm import run_frequency_severity_glm
        return run_frequency_severity_glm(
            train, score,
            freq_alpha=float(hp.get("freq_alpha", 1.0)),
            sev_alpha=float(hp.get("sev_alpha", 1.0)),
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
        )

    if model_family == "tweedie_gbm":
        from autoresearch.models.gbm import run_tweedie_gbm
        return run_tweedie_gbm(
            train, score,
            max_iter=int(hp.get("max_iter", 500)),
            max_depth=int(hp.get("max_depth", 5)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            min_samples_leaf=int(hp.get("min_samples_leaf", 200)),
            l2_regularization=float(hp.get("l2_regularization", 0.0)),
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
        )

    if model_family == "regularized_linear":
        from autoresearch.models.baselines import _fit_direct, _fit_frequency_severity, _feature_columns
        features = _feature_columns(train, feature_inclusions, feature_exclusions)
        if target_strategy == "direct_pure_premium":
            predicted = _fit_direct(train, score, float(hp.get("alpha", 1.0)), feature_inclusions, feature_exclusions)
        elif target_strategy == "frequency_severity":
            predicted = _fit_frequency_severity(train, score, float(hp.get("alpha", 1.0)), feature_inclusions, feature_exclusions)
        else:
            raise ValueError(f"Unsupported target strategy for regularized_linear: {target_strategy!r}")
        notes = {
            "model_family": "regularized_linear",
            "target_strategy": target_strategy,
            "alpha": float(hp.get("alpha", 1.0)),
            "feature_columns": features,
        }
        return predicted, notes

    # Open registry: try to import autoresearch.models.<model_family>
    # The module must expose fit_predict(train, score, *, feature_inclusions,
    # feature_exclusions, **hyperparameters) -> (np.ndarray, dict).
    import importlib
    try:
        mod = importlib.import_module(f"autoresearch.models.{model_family}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown model_family: {model_family!r}. "
            f"Either use a built-in family or create src/autoresearch/models/{model_family}.py "
            "exposing fit_predict(train, score, *, feature_inclusions, feature_exclusions, **hp)."
        )
    if not hasattr(mod, "fit_predict"):
        raise ValueError(
            f"Module autoresearch.models.{model_family} must expose a fit_predict() function."
        )
    return mod.fit_predict(
        train, score,
        feature_inclusions=feature_inclusions,
        feature_exclusions=feature_exclusions,
        **hp,
    )
