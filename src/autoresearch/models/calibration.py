"""
Standard calibration utility for run-local model scripts.

Apply at the end of every fit_predict function before returning predictions.
The correction is a single aggregate scalar (one degree of freedom) so it
carries no leakage risk, is rank-preserving, and is a no-op when the model
is already perfectly calibrated.

Usage::

    from autoresearch.models.calibration import apply_training_calibration

    pred_train_cost = np.maximum(model.predict(X_train), 0.0) * train_exp
    pred_score_cost = np.maximum(model.predict(X_score), 0.0) * score_exp
    pred_score_cost, calib_factor = apply_training_calibration(
        pred_score_cost, pred_train_cost, train[CLAIM_COST].values
    )
    notes["native_pred_to_actual_ratio"] = round(1.0 / calib_factor, 4)
    notes["calib_factor"] = round(float(calib_factor), 4)
"""
from __future__ import annotations

import numpy as np


def apply_training_calibration(
    pred_score: np.ndarray,
    pred_train_cost: np.ndarray,
    actual_train_cost: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Apply a global scalar calibration to score predictions.

    Computes ``calib_factor = sum(actual_train_cost) / sum(pred_train_cost)``
    and returns ``(pred_score * calib_factor, calib_factor)``.

    Parameters
    ----------
    pred_score:
        Predicted claim *costs* for the score / eval set, shape ``(n_score,)``.
    pred_train_cost:
        Predicted claim *costs* for the training set, shape ``(n_train,)``.
        Must be on the same scale as ``actual_train_cost`` (i.e. cost, not rate).
    actual_train_cost:
        Actual claim costs for the training set, shape ``(n_train,)``.

    Returns
    -------
    calibrated_pred_score : np.ndarray
        Score predictions rescaled so the aggregate level matches training.
    calib_factor : float
        The scalar applied.  ``1.0`` means no correction was needed.
        Always record ``1 / calib_factor`` as ``native_pred_to_actual_ratio``
        in notes so the original model bias remains observable.
    """
    pred_score = np.asarray(pred_score, dtype=float)
    pred_train_cost = np.asarray(pred_train_cost, dtype=float)
    actual_train_cost = np.asarray(actual_train_cost, dtype=float)

    total_actual = float(actual_train_cost.sum())
    total_pred = float(pred_train_cost.sum())
    calib_factor = total_actual / max(total_pred, 1e-9)
    return pred_score * calib_factor, calib_factor
