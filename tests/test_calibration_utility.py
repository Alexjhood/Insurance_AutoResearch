"""Tests for autoresearch.models.calibration — the universal scalar calibration helper."""

import numpy as np
import pytest

from autoresearch.models.calibration import apply_training_calibration


def test_perfect_calibration_returns_factor_one():
    pred_train = np.array([100.0, 200.0, 300.0])
    actual_train = np.array([100.0, 200.0, 300.0])
    pred_score = np.array([150.0, 250.0])
    calibrated, factor = apply_training_calibration(pred_score, pred_train, actual_train)
    assert abs(factor - 1.0) < 1e-9
    np.testing.assert_allclose(calibrated, pred_score)


def test_over_prediction_corrects_downward():
    # model predicts 1000 total, actual is 770 → factor ≈ 0.77
    pred_train = np.full(100, 10.0)   # sum = 1000
    actual_train = np.full(100, 7.7)  # sum = 770
    pred_score = np.array([10.0, 20.0])
    calibrated, factor = apply_training_calibration(pred_score, pred_train, actual_train)
    assert abs(factor - 0.77) < 1e-9
    np.testing.assert_allclose(calibrated, pred_score * 0.77)


def test_under_prediction_corrects_upward():
    pred_train = np.full(100, 5.0)    # sum = 500
    actual_train = np.full(100, 10.0) # sum = 1000
    pred_score = np.array([5.0, 10.0])
    calibrated, factor = apply_training_calibration(pred_score, pred_train, actual_train)
    assert abs(factor - 2.0) < 1e-9
    np.testing.assert_allclose(calibrated, pred_score * 2.0)


def test_rank_preserved():
    """Multiplying by a scalar must not change the rank ordering."""
    rng = np.random.default_rng(42)
    pred_train = rng.exponential(100, 500)
    actual_train = pred_train * 0.77
    pred_score = rng.exponential(100, 200)
    calibrated, _ = apply_training_calibration(pred_score, pred_train, actual_train)
    assert np.all(np.argsort(pred_score) == np.argsort(calibrated))


def test_guard_against_zero_pred():
    """Zero total predicted should not raise; factor is clamped via 1e-9 denominator."""
    pred_train = np.zeros(10)
    actual_train = np.ones(10) * 100.0
    pred_score = np.array([1.0, 2.0])
    calibrated, factor = apply_training_calibration(pred_score, pred_train, actual_train)
    assert np.isfinite(factor)
    assert np.all(np.isfinite(calibrated))


def test_accepts_pandas_series():
    import pandas as pd
    pred_train = pd.Series([10.0, 20.0, 30.0])
    actual_train = pd.Series([12.0, 24.0, 36.0])  # factor = 1.2
    pred_score = np.array([5.0, 10.0])
    calibrated, factor = apply_training_calibration(pred_score, pred_train, actual_train)
    assert abs(factor - 1.2) < 1e-9
    np.testing.assert_allclose(calibrated, pred_score * 1.2)
