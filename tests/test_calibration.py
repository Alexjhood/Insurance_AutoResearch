"""Tests for calibration diagnostics and promotion gating."""

import numpy as np
import pandas as pd
import pytest

from autoresearch.evaluation.diagnostics import compute_diagnostics, _calibration_pass
from autoresearch.evaluation.resampling import PromotionRules, promotion_decision, bootstrap_lift_summary


def _make_preds(actual, predicted, split="search_validation"):
    n = len(actual)
    return pd.DataFrame({
        "record_id": range(n),
        "split": [split] * n,
        "actual_claim_cost": actual,
        "predicted_claim_cost": predicted,
        "exposure": [1.0] * n,
    })


def _rules(**kwargs) -> PromotionRules:
    defaults = dict(
        minimum_mean_lift=0.0, min_relative_lift=0.0, min_absolute_lift=0.0,
        minimum_win_rate=0.0, bootstrap_lower_bound=0.0, bootstrap_lower_bound_relative=0.0,
        confidence_level=0.90, max_predicted_to_actual_drift=0.05,
        require_diagnostics=True, bonferroni_lookback=1,
    )
    defaults.update(kwargs)
    return PromotionRules(**defaults)


def test_diagnostics_returns_decile_table() -> None:
    rng = np.random.default_rng(1)
    actual = rng.exponential(100, 100)
    predicted = actual * (1 + rng.normal(0, 0.1, 100))
    preds = _make_preds(actual, predicted)
    result = compute_diagnostics(preds, "search_validation")
    assert "calibration_by_pred_decile" in result
    assert len(result["calibration_by_pred_decile"]) > 0


def test_calibration_pass_false_when_ratio_extreme() -> None:
    table = [
        {"decile": 1, "n": 10, "actual_pp": 100.0, "pred_pp": 1.0, "ratio": 100.0},  # very bad
        {"decile": 2, "n": 10, "actual_pp": 100.0, "pred_pp": 100.0, "ratio": 1.0},
    ]
    assert not _calibration_pass(table)


def test_calibration_pass_true_when_ratios_reasonable() -> None:
    table = [
        {"decile": i, "n": 10, "actual_pp": 100.0, "pred_pp": 100.0 * (1 + 0.05 * i), "ratio": 1.0 / (1 + 0.05 * i)}
        for i in range(10)
    ]
    # All ratios ~0.5–1.0, should pass
    assert _calibration_pass(table, lo=0.3, hi=3.0)


def test_promotion_fails_when_calibration_worsens() -> None:
    """A challenger with good lift but bad calibration should not promote."""
    # Challenger diagnostics showing extreme calibration drift
    bad_diag = {
        "calibration_by_pred_decile": [
            {"decile": i, "actual_pp": 100.0, "pred_pp": 0.5, "n": 10, "ratio": 200.0}
            for i in range(10)
        ]
    }
    comp_summary = {
        "mean_lift": 10.0, "challenger_win_rate": 0.9,
        "champion_mean_score": 100.0, "std_lift": 1.0, "n_resamples": 30,
    }
    boot = bootstrap_lift_summary(pd.Series([10.0] * 30), iterations=100, seed=1, confidence_level=0.9)
    decision = promotion_decision(comp_summary, boot, _rules(max_predicted_to_actual_drift=0.05), challenger_diagnostics=bad_diag)
    assert decision["decision"] == "inconclusive"
    assert not decision["checks"]["calibration_ok"]


def test_promotion_passes_when_calibration_is_fine() -> None:
    good_diag = {
        "calibration_by_pred_decile": [
            {"decile": i, "actual_pp": 100.0, "pred_pp": 102.0, "n": 10, "ratio": 0.98}
            for i in range(10)
        ]
    }
    comp_summary = {
        "mean_lift": 10.0, "challenger_win_rate": 0.9,
        "champion_mean_score": 100.0, "std_lift": 1.0, "n_resamples": 30,
    }
    boot = bootstrap_lift_summary(pd.Series([10.0] * 30), iterations=100, seed=1, confidence_level=0.9)
    decision = promotion_decision(comp_summary, boot, _rules(max_predicted_to_actual_drift=0.5), challenger_diagnostics=good_diag)
    assert decision["checks"]["calibration_ok"]
