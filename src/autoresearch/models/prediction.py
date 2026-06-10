"""Framework-owned prediction units, exposure conversion, and calibration (#7).

The most safety-critical bookkeeping in every experiment is the same:

* convert a predicted *rate* (per unit exposure) into a *target total*,
* calibrate the aggregate level to the training data,
* validate that predictions are the right length, finite, and non-negative.

Historically every run-local model script re-implemented this by hand, which was
both repeated paid output and the source of recurring bugs (exposure-ranking
artefacts, sign errors, length mismatches). This module centralises it.

A model — whether a declarative recipe or an escape-hatch Python script — may
return a :class:`Prediction` instead of a raw ``np.ndarray``. When it does, the
dispatcher finalises it through :func:`finalize_prediction`:

* ``unit="rate"``         → multiplied by exposure to obtain target totals;
* ``unit="target_total"`` → used directly as totals.

The framework then applies a single-scalar aggregate calibration computed on the
training rows and runs the prediction validators. Returning a bare ``np.ndarray``
preserves the legacy contract (the script owns totals and calibration itself), so
this change is fully backward compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.models.calibration import apply_training_calibration


RATE = "rate"
TARGET_TOTAL = "target_total"
VALID_UNITS = frozenset({RATE, TARGET_TOTAL})


class PredictionUnitError(ValueError):
    """Raised when a model returns predictions in an inconsistent or invalid unit."""


@dataclass(frozen=True)
class Prediction:
    """A structured model prediction tagged with its unit.

    Parameters
    ----------
    values:
        Predicted quantities aligned 1:1 with the scored rows.
    unit:
        Either ``"rate"`` (per unit exposure) or ``"target_total"``.
    calibrate:
        When ``True`` (default) the framework applies aggregate training
        calibration. Set ``False`` for a model that is calibration-free by
        construction (e.g. a flat baseline) or that has already calibrated.
    notes:
        Optional extra notes merged into the experiment's ``model_notes``.
    """

    values: Any
    unit: str = RATE
    calibrate: bool = True
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit not in VALID_UNITS:
            raise PredictionUnitError(
                f"Prediction.unit must be one of {sorted(VALID_UNITS)}, got {self.unit!r}"
            )


def validate_objective_labels(objective: str, labels: np.ndarray, *, context: str = "") -> None:
    """Validate that training labels are compatible with the chosen objective.

    These are stable, domain-level constraints that should never reach the LLM as
    a repair request:

    * ``gamma``   — labels must be strictly positive.
    * ``poisson`` — labels must be non-negative.
    * ``tweedie`` — labels must be non-negative.
    """

    labels = np.asarray(labels, dtype=float)
    where = f" ({context})" if context else ""
    if labels.size == 0:
        raise PredictionUnitError(f"Training labels are empty{where}")
    if not np.all(np.isfinite(labels)):
        raise PredictionUnitError(f"Training labels contain non-finite values{where}")
    obj = (objective or "").strip().lower()
    if obj == "gamma" and not np.all(labels > 0):
        raise PredictionUnitError(
            f"gamma objective requires strictly positive labels{where}; "
            "split frequency × severity or restrict to claim rows"
        )
    if obj in {"poisson", "tweedie"} and np.any(labels < 0):
        raise PredictionUnitError(f"{obj} objective requires non-negative labels{where}")


def validate_predictions(values: np.ndarray, *, n_expected: int, context: str = "") -> np.ndarray:
    """Validate predicted totals: length, finiteness, and sign."""

    arr = np.asarray(values, dtype=float)
    where = f" ({context})" if context else ""
    if arr.ndim != 1:
        raise PredictionUnitError(f"Predictions must be 1-D{where}, got shape {arr.shape}")
    if len(arr) != n_expected:
        raise PredictionUnitError(
            f"Model returned {len(arr)} predictions for {n_expected} scored rows{where}"
        )
    if not np.all(np.isfinite(arr)):
        raise PredictionUnitError(f"Predictions contain non-finite values{where}")
    if np.any(arr < -1e-9):
        raise PredictionUnitError(f"Predictions contain negative values{where}")
    return np.clip(arr, 0.0, None)


def finalize_prediction(
    prediction: Prediction,
    score: pd.DataFrame,
    *,
    exposure_column: str,
    source_column: str,
    train_split_label: str,
    split_column: str = "split",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a :class:`Prediction` to calibrated target totals.

    Returns ``(calibrated_totals, notes)``. ``notes`` records the calibration
    factor and the native (pre-calibration) prediction-to-actual ratio so the
    original model bias remains observable.
    """

    exposure = score[exposure_column].astype(float).to_numpy()
    raw_values = np.asarray(prediction.values, dtype=float)

    if prediction.unit == RATE:
        totals = raw_values * exposure
    else:  # TARGET_TOTAL
        totals = raw_values

    totals = validate_predictions(totals, n_expected=len(score), context=f"unit={prediction.unit}")

    notes: dict[str, Any] = dict(prediction.notes)
    notes.setdefault("prediction_unit", prediction.unit)

    if prediction.calibrate:
        split_values = score[split_column].to_numpy()
        train_mask = split_values == train_split_label
        if train_mask.any():
            actual_train = score.loc[train_mask, source_column].astype(float).to_numpy()
            pred_train = totals[train_mask]
            totals, calib_factor = apply_training_calibration(totals, pred_train, actual_train)
            notes["calib_factor"] = round(float(calib_factor), 6)
            notes["native_pred_to_actual_ratio"] = round(1.0 / calib_factor, 6) if calib_factor else None
        else:
            notes["calib_factor"] = 1.0
            notes["calibration_skipped"] = "no training rows in score frame"
    else:
        notes["calib_factor"] = 1.0

    return totals, notes
