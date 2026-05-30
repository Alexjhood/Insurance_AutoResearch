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


def test_rank_gini_positive_for_correlated_predictions() -> None:
    """Positively correlated predictions give positive rank_gini."""
    actual = [10.0, 20.0, 30.0, 40.0, 50.0]
    predicted = [1.0, 2.0, 3.0, 4.0, 5.0]
    panel = _panel(actual, predicted)
    # rank_gini should be positive (correctly ordered discrimination)
    assert panel["rank_gini_weighted"] > 0.0
    # and agrees in sign with the amount-weighted Gini
    assert panel["gini_weighted"] > 0.0
    assert (panel["rank_gini_weighted"] > 0) == (panel["gini_weighted"] > 0)


def test_rank_gini_negative_for_reversed_predictions() -> None:
    """Anti-correlated predictions give negative rank_gini."""
    actual = [10.0, 20.0, 30.0, 40.0, 50.0]
    predicted = [5.0, 4.0, 3.0, 2.0, 1.0]
    panel = _panel(actual, predicted)
    assert panel["rank_gini_weighted"] < 0.0


def test_rank_gini_bounded_influence_vs_amount_gini() -> None:
    """Inflating one large claim changes gini_weighted dramatically but
    rank_gini_weighted only slightly (O(1/n) influence per observation)."""
    n = 50
    actual_base = list(range(1, n + 1))
    predicted = list(range(1, n + 1))

    panel_base = _panel(actual_base, predicted)
    # Replace the largest actual claim with an extreme value
    actual_extreme = actual_base[:]
    actual_extreme[-1] = 1_000_000.0
    panel_extreme = _panel(actual_extreme, predicted)

    gini_delta = abs(panel_extreme["gini_weighted"] - panel_base["gini_weighted"])
    rank_gini_delta = abs(panel_extreme["rank_gini_weighted"] - panel_base["rank_gini_weighted"])

    # The amount-weighted Gini should change substantially; rank-Gini only slightly
    assert gini_delta > 0.05, f"Expected gini_weighted to shift materially, got delta={gini_delta}"
    assert rank_gini_delta < gini_delta, (
        f"rank_gini_weighted (delta={rank_gini_delta}) should be more stable "
        f"than gini_weighted (delta={gini_delta}) under an extreme single claim"
    )


def test_rank_gini_invariant_to_monotone_scale() -> None:
    """Rank-Gini is invariant to monotone rescaling of predictions (same ordering)."""
    actual = [10.0, 50.0, 200.0, 1000.0]
    predicted = [5.0, 40.0, 150.0, 900.0]
    panel_base = _panel(actual, predicted)
    panel_scaled = _panel(actual, [v * 100 for v in predicted])
    assert abs(panel_scaled["rank_gini_weighted"] - panel_base["rank_gini_weighted"]) < 1e-9


def test_spearman_rho_positive_when_correlated() -> None:
    actual = [float(i) for i in range(20)]
    predicted = [float(i) + 0.1 for i in range(20)]
    panel = _panel(actual, predicted)
    assert panel["spearman_rho"] > 0.95


def test_kendall_tau_positive_when_correlated() -> None:
    actual = [float(i) for i in range(20)]
    predicted = [float(i) + 0.1 for i in range(20)]
    panel = _panel(actual, predicted)
    assert panel["kendall_tau"] > 0.85


def test_decile_lift_monotonicity_high_for_perfect_model() -> None:
    n = 100
    actual = list(range(n))
    predicted = list(range(n))
    panel = _panel(actual, predicted)
    assert panel["decile_lift_monotonicity"] > 0.8


def test_decile_lift_monotonicity_nan_for_small_input() -> None:
    panel = _panel([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    assert np.isnan(panel["decile_lift_monotonicity"])


def test_evaluate_predictions_blocks_milestone_holdout() -> None:
    preds = pd.DataFrame({
        "split": ["search_validation", "milestone_holdout"],
        "actual_claim_cost": [100.0, 200.0],
        "predicted_claim_cost": [100.0, 200.0],
        "exposure": [1.0, 1.0],
    })
    with pytest.raises(ValueError, match="milestone_holdout"):
        evaluate_predictions(preds, ("search_validation",))


# ── Asymmetric Pricing Loss ───────────────────────────────────────────────────

def test_apl_under_pricing_penalised_4x_over() -> None:
    """Under-pricing by 1 unit costs 4× as much as over-pricing by 1 unit."""
    # actual_rate = 10, predicted_rate = 9 → under by 1 unit (exposure 1)
    panel_under = _panel([10.0], [9.0])
    # actual_rate = 10, predicted_rate = 11 → over by 1 unit (exposure 1)
    panel_over = _panel([10.0], [11.0])
    apl_under = panel_under["asym_pricing_loss"]
    apl_over = panel_over["asym_pricing_loss"]
    assert abs(apl_under / apl_over - 4.0) < 1e-6, f"Expected 4:1 ratio, got {apl_under / apl_over}"


def test_apl_perfect_prediction_is_zero() -> None:
    """Perfect predictions incur zero APL."""
    panel = _panel([5.0, 10.0, 20.0], [5.0, 10.0, 20.0])
    assert panel["asym_pricing_loss"] == 0.0
    assert panel["apl_under_cost"] == 0.0
    assert panel["apl_over_cost"] == 0.0


def test_apl_components_sum_correctly() -> None:
    """TAU_UNDER * under_cost + TAU_OVER * over_cost == asym_pricing_loss."""
    from autoresearch.evaluation.metrics import TAU_UNDER, TAU_OVER
    panel = _panel([5.0, 15.0, 10.0], [8.0, 12.0, 10.0])
    expected = TAU_UNDER * panel["apl_under_cost"] + TAU_OVER * panel["apl_over_cost"]
    assert abs(panel["asym_pricing_loss"] - expected) < 1e-9


def test_apl_lower_is_better() -> None:
    """asym_pricing_loss is NOT in HIGHER_IS_BETTER_METRICS."""
    from autoresearch.evaluation.metrics import HIGHER_IS_BETTER_METRICS, lower_is_better
    assert "asym_pricing_loss" not in HIGHER_IS_BETTER_METRICS
    assert lower_is_better("asym_pricing_loss") is True


def test_apl_in_full_panel() -> None:
    """APL sub-metrics are present in the full panel."""
    panel = _panel([10.0, 20.0, 30.0], [9.0, 22.0, 30.0])
    assert "asym_pricing_loss" in panel
    assert "apl_under_cost" in panel
    assert "apl_over_cost" in panel
    assert "apl_under_over_ratio" in panel
    # ratio: under/over
    if panel["apl_over_cost"] > 0:
        expected_ratio = panel["apl_under_cost"] / panel["apl_over_cost"]
        assert abs(panel["apl_under_over_ratio"] - expected_ratio) < 1e-9
