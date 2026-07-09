"""Model dispatcher: routes model_family/target_strategy to the right implementation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.models.prediction import Prediction, finalize_prediction
from autoresearch.models import columns as _columns
from autoresearch.models.columns import RECORD_ID
from autoresearch.targets import (
    BURNING_COST,
    normalise_target_mode,
    target_spec,
)


# French-default column constants, re-exported for agent scripts. ``columns.bind``
# reassigns these on config load to the active dataset's names.
EXPOSURE = _columns.EXPOSURE
CLAIM_COUNT = _columns.CLAIM_COUNT
CLAIM_EVENTS = _columns.CLAIM_EVENTS
CLAIM_COST = _columns.CLAIM_COST
RAW_CLAIM_COST = _columns.RAW_CLAIM_COST


def _filter_to_population(frame: pd.DataFrame, spec: Any, *, context: str) -> pd.DataFrame:
    """Restrict a frame to the target's evaluation population.

    A mode with a ``population_column`` (e.g. severity) is modelled and assessed
    on rows where that column is positive; every other mode uses the full
    population. Filtering here — the single dispatch chokepoint shared by the
    search runner, the CV refit factory, and the holdout eval — means every
    surface selects the same rows automatically.
    """

    column = spec.population_column
    if column is None:
        return frame
    filtered = frame[frame[column].astype(float) > 0]
    if filtered.empty:
        raise ValueError(
            f"{context}: no rows with {column} > 0 for target_mode {spec.mode!r}"
        )
    return filtered.copy()


def _add_target_rate_columns(
    predictions: pd.DataFrame, spec: Any, actual_target: np.ndarray, clipped_target: np.ndarray
) -> None:
    """Populate spec-driven alias and per-weight rate columns on a prediction frame.

    Column names come purely from the active :class:`TargetSpec`; the frame's
    ``exposure`` column already holds the active metric weight, so each rate is
    ``total / weight``. Applied to French modes this reproduces the historical
    ``predicted_claim_cost`` / ``actual_pure_premium`` / … columns exactly.
    """

    exp = np.clip(predictions["exposure"].to_numpy(dtype=float), 1e-12, None)
    predictions[spec.predicted_alias] = clipped_target
    predictions[spec.actual_alias] = actual_target
    predictions[spec.rate_actual_column] = actual_target / exp
    predictions[spec.rate_predicted_column] = clipped_target / exp


def _french_companion_columns(predictions: pd.DataFrame, score: pd.DataFrame, spec: Any) -> None:
    """Reconstruct the historical French cross-mode prediction-parquet schema.

    Only fires when the French cost + count source columns are present in the
    score frame (a no-op for datasets without them). Preserves every column the
    pre-multi-dataset dispatcher emitted so old dashboards/reports/tests stay
    valid: both claim-cost and claim-count entity columns (the inactive one
    NaN-filled) and, for population-wide modes, both pure-premium and frequency
    rate pairs.
    """

    cost_col, count_col = _columns.CLAIM_COST, _columns.CLAIM_COUNT
    if cost_col not in score.columns or count_col not in score.columns:
        return

    exp = np.clip(predictions["exposure"].to_numpy(dtype=float), 1e-12, None)

    def _set(name: str, values: np.ndarray) -> None:
        if name not in predictions.columns:
            predictions[name] = values

    nan = np.full(len(predictions), np.nan)
    _set("actual_claim_cost", score[cost_col].astype(float).to_numpy())
    _set("actual_claim_count", score[count_col].astype(float).to_numpy())
    if _columns.RAW_CLAIM_COST in score.columns:
        _set("actual_claim_cost_uncapped", score[_columns.RAW_CLAIM_COST].astype(float).to_numpy())
    if _columns.CLAIM_EVENTS in score.columns:
        _set("actual_claim_event_count", score[_columns.CLAIM_EVENTS].astype(float).to_numpy())
    # Inactive entity's predicted total is NaN (the active one is already set).
    _set("predicted_claim_cost", nan)
    _set("predicted_claim_count", nan)
    if spec.population_column is None:  # population-wide modes carry both rate pairs
        _set("actual_pure_premium", predictions["actual_claim_cost"].to_numpy() / exp)
        _set("predicted_pure_premium", predictions["predicted_claim_cost"].to_numpy() / exp)
        _set("actual_frequency", predictions["actual_claim_count"].to_numpy() / exp)
        _set("predicted_frequency", predictions["predicted_claim_count"].to_numpy() / exp)


@dataclass(frozen=True)
class ModelResult:
    """Predictions and notes from any model family."""

    predictions: pd.DataFrame
    model_notes: dict[str, Any]
    interpret_fn: Any | None = None  # optional callable(feature_df) -> predicted_rates


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
    model_script_path: Path | None = None,
    allow_holdout_split: bool = False,
    target_mode: str = BURNING_COST,
) -> ModelResult:
    """Fit the requested model family and return scored predictions.

    Works identically for the legacy Ridge, the built-in GLM/GBM families,
    and any new family registered as ``autoresearch.models.<model_family>``.

    Feature engineering: if ``hyperparameters`` contains a
    ``feature_builder_module`` key, the named module's ``build_features``
    function is applied to the full data frame before splitting into
    train / score partitions.
    """

    target_mode = normalise_target_mode(target_mode)
    spec = target_spec(target_mode)
    hp = dict(hyperparameters or {})
    hp.setdefault("target_mode", target_mode)

    # Apply optional feature builder before splitting
    feature_builder_module = hp.pop("feature_builder_module", None)
    if feature_builder_module:
        import importlib
        builder = importlib.import_module(feature_builder_module)
        frame = builder.build_features(frame)

    hp.setdefault("_id_columns", _matching_id_columns(frame))
    data = frame.merge(split_frame[["record_id", "split"]], on="record_id", how="inner")
    if len(data) < len(frame):
        raise ValueError(f"{len(frame) - len(data)} rows dropped during split merge — split pack may be stale")
    if not allow_holdout_split and (data["split"] == "milestone_holdout").any() and "milestone_holdout" in score_splits:
        raise ValueError("Milestone holdout cannot be requested for ordinary scoring")

    train = data[data["split"] == train_split].copy()
    score = data[data["split"].isin((train_split, *score_splits))].copy()
    if train.empty:
        raise ValueError(f"Training split {train_split!r} is empty")
    if score.empty:
        raise ValueError("No rows available for scoring")
    # Severity trains and scores on claim rows only; a no-op for other modes.
    train = _filter_to_population(train, spec, context=f"train split {train_split!r}")
    score = _filter_to_population(score, spec, context="score frame")

    predicted_target, notes, interpret_fn = _call_model(
        model_family,
        target_strategy,
        train,
        score,
        hp,
        feature_inclusions,
        feature_exclusions,
        model_script_path,
    )
    clipped_target = _finalize_predicted(
        predicted_target, score, notes, target_mode=target_mode, train_split_label=train_split
    )

    actual_target = score[spec.source_column].astype(float).to_numpy()
    # Metric weight: the target's weight column (exposure, claim count, or the
    # synthesised unit weight).
    exposure = score[spec.weight_column].astype(float).to_numpy()
    predictions = pd.DataFrame({
        RECORD_ID: score[RECORD_ID].to_numpy(),
        "split": score["split"].to_numpy(),
        "target_mode": target_mode,
        "exposure": exposure,
        "actual_target": actual_target,
        "predicted_target": clipped_target,
    })
    _add_target_rate_columns(predictions, spec, actual_target, clipped_target)
    _french_companion_columns(predictions, score, spec)
    return ModelResult(predictions=predictions, model_notes=notes, interpret_fn=interpret_fn)


def dispatch_model_on_explicit_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    model_family: str,
    target_strategy: str,
    hyperparameters: dict[str, Any] | None = None,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    model_script_path: Path | None = None,
    target_mode: str = BURNING_COST,
) -> ModelResult:
    """Fit and score using explicit train/val DataFrames instead of split names.

    Intended for k-fold CV comparisons where fold boundaries are managed by the
    caller rather than by split-pack labels.  Semantics are identical to
    ``dispatch_model`` except the caller supplies ``train_df`` and ``val_df``
    directly; a synthetic 'split' column is added before dispatch.

    The returned predictions contain only val rows (split == 'val').
    """

    target_mode = normalise_target_mode(target_mode)
    spec = target_spec(target_mode)
    hp = dict(hyperparameters or {})
    hp.setdefault("target_mode", target_mode)

    # Apply optional feature builder to both frames before model dispatch
    feature_builder_module = hp.pop("feature_builder_module", None)
    if feature_builder_module:
        import importlib
        builder = importlib.import_module(feature_builder_module)
        train_df = builder.build_features(train_df)
        val_df = builder.build_features(val_df)

    hp.setdefault("_id_columns", _matching_id_columns(train_df))

    # Severity trains and scores on claim rows only; a no-op for other modes.
    train_df = _filter_to_population(train_df, spec, context="cv train frame")
    val_df = _filter_to_population(val_df, spec, context="cv val frame")

    # Tag with synthetic split labels; score frame includes both so fit_predict
    # receives the same (train, score) structure as the standard runner.
    _train = train_df.copy()
    _train["split"] = "_cv_train"
    _val = val_df.copy()
    _val["split"] = "_cv_val"
    score = pd.concat([_train, _val], ignore_index=True)

    predicted_target, notes, interpret_fn = _call_model(
        model_family,
        target_strategy,
        _train,
        score,
        hp,
        feature_inclusions,
        feature_exclusions,
        model_script_path,
    )
    finalised = _finalize_predicted(
        predicted_target, score, notes, target_mode=target_mode, train_split_label="_cv_train"
    )

    # Filter to val rows only for evaluation
    val_mask = score["split"].to_numpy() == "_cv_val"
    score_val = score[val_mask].copy()

    actual_target = score_val[spec.source_column].astype(float).to_numpy()
    clipped_target = finalised[val_mask]
    # Metric weight: the target's weight column.
    exposure = score_val[spec.weight_column].astype(float).to_numpy()

    predictions = pd.DataFrame({
        RECORD_ID: score_val[RECORD_ID].to_numpy(),
        "split": "val",
        "target_mode": target_mode,
        "exposure": exposure,
        "actual_target": actual_target,
        "predicted_target": clipped_target,
    })
    _add_target_rate_columns(predictions, spec, actual_target, clipped_target)
    _french_companion_columns(predictions, score_val, spec)
    return ModelResult(predictions=predictions, model_notes=notes, interpret_fn=interpret_fn)


def _matching_id_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns that duplicate the framework record identifier."""

    if RECORD_ID not in frame.columns:
        return [RECORD_ID]
    record_ids = frame[RECORD_ID]
    matches = [RECORD_ID]
    for column in frame.columns:
        if column == RECORD_ID:
            continue
        series = frame[column]
        if series.is_unique and series.equals(record_ids):
            matches.append(column)
    return matches


def _sanitize_script_frames(
    train: pd.DataFrame, score: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return script-safe copies of the train/score frames.

    The score frame loses every target-bearing column (the active dataset's leak
    set); both frames lose identifier columns that duplicate ``record_id`` (a
    column-sweeping script would otherwise pick up the source id, or worse, the
    realised outcome of the rows it is scored on). The train frame keeps the
    training targets but loses the raw uncapped source, which is never a legal
    training input. The leak set is dataset-driven via ``columns.bind``.
    """

    id_dupes = [c for c in _matching_id_columns(train) if c != RECORD_ID]
    train_drop = [c for c in (*_columns.raw_train_strip_columns(), *id_dupes) if c in train.columns]
    score_drop = [c for c in (*_columns.leak_columns(), *id_dupes) if c in score.columns]
    return train.drop(columns=train_drop), score.drop(columns=score_drop)


def _finalize_predicted(
    predicted: Any,
    score: pd.DataFrame,
    notes: dict[str, Any],
    *,
    target_mode: str,
    train_split_label: str,
) -> np.ndarray:
    """Resolve a model return value to validated, calibrated target totals.

    A bare ``np.ndarray`` is treated as already-final totals (legacy contract:
    the model owns its own exposure conversion and calibration). A
    :class:`Prediction` is finalised by the framework (#7): rate→total via
    exposure, aggregate training calibration, and prediction validation.
    """

    spec = target_spec(target_mode)
    if isinstance(predicted, Prediction):
        totals, fin_notes = finalize_prediction(
            predicted,
            score,
            # For severity the "rate" is cost-per-claim, so rate→total multiplies
            # by claim count, not exposure.
            exposure_column=spec.weight_column,
            source_column=spec.source_column,
            train_split_label=train_split_label,
        )
        notes.update(fin_notes)
        return totals

    arr = np.asarray(predicted, dtype=float)
    if len(arr) != len(score):
        raise ValueError(
            f"Model returned {len(arr)} predictions for {len(score)} scored rows"
        )
    return np.clip(arr, 0.0, None)


def _call_model(
    model_family: str,
    target_strategy: str,
    train: pd.DataFrame,
    score: pd.DataFrame,
    hp: dict[str, Any],
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
    model_script_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Dispatch to the correct model implementation.

    Returns ``(predicted, notes, interpret_fn)`` where ``interpret_fn`` is an
    optional callable ``(feature_df: pd.DataFrame) -> np.ndarray`` that returns
    predicted *rates* (not totals) for arbitrary feature inputs.  Models that
    do not return a third value get ``interpret_fn=None``.
    """

    # A recipe takes precedence over a script path: representations are meant to
    # be mutually exclusive (enforced at proposal validation), and routing the
    # recipe first prevents a stray script_path from silently shadowing it.
    if model_family == "recipe" or (model_family != "global_mean" and "recipe" in hp):
        if model_script_path is not None:
            raise ValueError(
                "A model cannot specify both a recipe and a script_path; choose one."
            )
        from autoresearch.models import recipe as _recipe
        return _normalise_fit_predict_result(
            _recipe.fit_predict(
                train, score,
                feature_inclusions=feature_inclusions,
                feature_exclusions=feature_exclusions,
                **hp,
            )
        )

    if model_script_path is not None:
        train, score = _sanitize_script_frames(train, score)
        return _call_script_model(
            model_script_path,
            train,
            score,
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
            hyperparameters=hp,
        )

    if model_family == "global_mean":
        from autoresearch.models.global_mean import fit_predict
        return _normalise_fit_predict_result(
            fit_predict(train, score, feature_inclusions=feature_inclusions, feature_exclusions=feature_exclusions, **hp)
        )

    # Open registry: try to import autoresearch.models.<model_family>
    # The module must expose fit_predict(train, score, *, feature_inclusions,
    # feature_exclusions, **hyperparameters) -> (np.ndarray, dict[, callable]).
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
    return _normalise_fit_predict_result(
        mod.fit_predict(
            train, score,
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
            **hp,
        )
    )


def _normalise_fit_predict_result(result: Any) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Normalise a fit_predict return value to always have 3 elements."""
    if isinstance(result, tuple):
        if len(result) == 3:
            predicted, notes, interpret_fn = result
        elif len(result) == 2:
            predicted, notes = result
            interpret_fn = None
        else:
            raise ValueError(
                f"fit_predict must return 2 or 3 values (predicted, notes[, interpret_fn]), got {len(result)}"
            )
    else:
        raise ValueError("fit_predict must return a tuple")
    if notes is None:
        notes = {}
    if not isinstance(notes, dict):
        raise ValueError("fit_predict() notes must be a dict")
    return predicted, dict(notes), interpret_fn


def _call_script_model(
    path: Path,
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None,
    feature_exclusions: list[str] | None,
    hyperparameters: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], Any]:
    """Load a run-local modelling script and execute its fit_predict hook."""

    if not path.exists():
        raise ValueError(f"Model script does not exist: {path}")
    module_name = f"_autoresearch_experiment_model_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load model script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "fit_predict"):
        raise ValueError(f"Model script must expose fit_predict(): {path}")
    predicted, notes, interpret_fn = _normalise_fit_predict_result(
        module.fit_predict(
            train,
            score,
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
            **hyperparameters,
        )
    )
    notes.setdefault("model_script_path", str(path))
    notes.setdefault("uses_run_local_script", True)
    return predicted, notes, interpret_fn
