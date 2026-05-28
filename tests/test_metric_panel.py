"""Tests for metric panel statistical properties."""

import numpy as np
import pandas as pd
import pytest

from autoresearch.evaluation.metrics import full_metric_panel, evaluate_predictions


def _panel(actual, predicted, exposure=None):
    n = len(actual)
    exp = exposure or [1.0] * n
    return full_metric_panel(
        pd.Series(actual, dtype=float),
        pd.Series(predicted, dtype=float),
        pd.Series(exp, dtype=float),
    )


def test_perfect_predictions_have_low_deviance() -> None:
    actual = [100.0, 200.0, 50.0, 300.0]
    panel = _panel(actual, actual)
    assert panel["tweedie_deviance_p15"] < 1e-6
    assert abs(panel["predicted_to_actual_ratio"] - 1.0) < 1e-9


def test_inflated_predictions_worsen_deviance_and_ratio() -> None:
    actual = [100.0, 200.0, 50.0, 300.0]
    panel_good = _panel(actual, actual)
    panel_bad = _panel(actual, [v * 1000 for v in actual])
    assert panel_bad["tweedie_deviance_p15"] > panel_good["tweedie_deviance_p15"]
    assert abs(panel_bad["predicted_to_actual_ratio"] - 1000.0) < 1.0


def test_inflating_predictions_does_not_change_gini() -> None:
    """Gini measures rank discrimination, not level calibration."""
    actual = [10.0, 50.0, 200.0, 1000.0]
    base_pred = [5.0, 40.0, 150.0, 900.0]
    panel_base = _panel(actual, base_pred)
    panel_scaled = _panel(actual, [v * 100 for v in base_pred])
    assert panel_base["gini_weighted"] > 0
    assert abs(panel_scaled["gini_weighted"] - panel_base["gini_weighted"]) < 0.01


def test_zero_predictions_make_ratio_near_zero() -> None:
    actual = [100.0, 200.0, 300.0]
    panel = _panel(actual, [1e-6, 1e-6, 1e-6])
    assert panel["predicted_to_actual_ratio"] < 0.01


def test_evaluate_predictions_uses_primary_metric() -> None:
    preds = pd.DataFrame({
        "split": ["search_validation"] * 10,
        "actual_claim_cost": np.random.default_rng(1).exponential(100, 10),
        "predicted_claim_cost": np.random.default_rng(2).exponential(100, 10),
        "exposure": np.ones(10),
    })
    result = evaluate_predictions(preds, ("search_validation",), primary_metric="tweedie_deviance_p15")
    assert result["primary_metric"] == "tweedie_deviance_p15"
    # mean_score should match what the split returned
    sv_metrics = [m for m in result["split_metrics"] if m["split"] == "search_validation"][0]
    assert abs(result["aggregate"]["mean_score"] - sv_metrics["tweedie_deviance_p15"]) < 1e-9


def test_evaluate_predictions_marks_gini_higher_is_better() -> None:
    preds = pd.DataFrame({
        "split": ["search_validation"] * 10,
        "actual_claim_cost": np.linspace(10, 100, 10),
        "predicted_claim_cost": np.linspace(10, 100, 10),
        "exposure": np.ones(10),
    })
    result = evaluate_predictions(preds, ("search_validation",), primary_metric="gini_weighted")
    assert result["primary_metric"] == "gini_weighted"
    assert result["lower_is_better"] is False


def test_evaluate_predictions_supports_frequency_target_mode() -> None:
    preds = pd.DataFrame({
        "split": ["search_validation"] * 10,
        "target_mode": ["frequency"] * 10,
        "actual_claim_count": [0, 1, 0, 2, 0, 1, 0, 0, 1, 0],
        "predicted_claim_count": [0.1, 0.8, 0.2, 1.6, 0.1, 0.9, 0.2, 0.1, 0.7, 0.1],
        "exposure": np.ones(10),
    })
    result = evaluate_predictions(
        preds,
        ("search_validation",),
        primary_metric="poisson_deviance",
        target_mode="frequency",
    )
    sv_metrics = [m for m in result["split_metrics"] if m["split"] == "search_validation"][0]
    assert result["target_mode"] == "frequency"
    assert sv_metrics["total_actual_claim_count"] == 5.0
    assert "mean_predicted_frequency" in sv_metrics
    assert abs(result["aggregate"]["mean_score"] - sv_metrics["poisson_deviance"]) < 1e-9


def test_evaluate_predictions_blocks_milestone_holdout() -> None:
    preds = pd.DataFrame({
        "split": ["search_validation", "milestone_holdout"],
        "actual_claim_cost": [100.0, 200.0],
        "predicted_claim_cost": [100.0, 200.0],
        "exposure": [1.0, 1.0],
    })
    with pytest.raises(ValueError, match="milestone_holdout"):
        evaluate_predictions(preds, ("search_validation",))
