import pandas as pd

from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    paired_comparison,
    promotion_decision,
    repeated_scores,
)


def _predictions(predicted: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": [1, 2, 3, 4],
        "split": ["search_validation"] * 4,
        "actual_claim_cost": [10.0, 20.0, 30.0, 40.0],
        "predicted_claim_cost": predicted,
        "exposure": [1.0, 1.0, 1.0, 1.0],
    })


def _default_rules(**overrides) -> PromotionRules:
    defaults = dict(
        minimum_mean_lift=0.0,
        min_relative_lift=0.0,
        min_absolute_lift=0.0,
        minimum_win_rate=0.55,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.5,
        require_diagnostics=False,
        bonferroni_lookback=10,
    )
    defaults.update(overrides)
    return PromotionRules(**defaults)


def test_repeated_scores_are_reproducible() -> None:
    predictions = _predictions([10.0, 18.0, 35.0, 39.0])

    first = repeated_scores(predictions, eval_split="search_validation", n_resamples=5, seed=123)
    second = repeated_scores(predictions, eval_split="search_validation", n_resamples=5, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_paired_comparison_positive_lift_when_challenger_is_better() -> None:
    # Champion predicts zero (terrible); challenger predicts actuals (perfect)
    champion = _predictions([0.01, 0.01, 0.01, 0.01])
    challenger = _predictions([10.0, 20.0, 30.0, 40.0])

    per_resample, summary = paired_comparison(
        champion,
        challenger,
        champion_id="champ",
        challenger_id="challenger",
        eval_split="search_validation",
        n_resamples=10,
        seed=7,
    )

    assert (per_resample["lift"] > 0).all()
    assert summary["mean_lift"] > 0
    assert summary["challenger_win_rate"] == 1.0


def test_paired_comparison_positive_lift_for_higher_is_better_gini() -> None:
    champion = _predictions([40.0, 30.0, 20.0, 10.0])
    challenger = _predictions([10.0, 20.0, 30.0, 40.0])

    per_resample, summary = paired_comparison(
        champion,
        challenger,
        champion_id="champ",
        challenger_id="challenger",
        eval_split="search_validation",
        n_resamples=10,
        seed=7,
        primary_metric="gini_weighted",
    )

    assert summary["lower_is_better"] is False
    assert (per_resample["lift"] > 0).all()
    assert summary["mean_lift"] > 0


def test_bootstrap_and_promotion_decision() -> None:
    bootstrap = bootstrap_lift_summary(pd.Series([1.0, 2.0, 3.0]), iterations=100, seed=1, confidence_level=0.9)
    decision = promotion_decision(
        {"mean_lift": 2.0, "challenger_win_rate": 1.0, "champion_mean_score": 10.0,
         "std_lift": 0.5, "n_resamples": 30},
        bootstrap,
        _default_rules(),
    )

    assert bootstrap["probability_challenger_outperforms"] == 1.0
    assert decision["decision"] == "promote"


def test_promotion_requires_relative_lift() -> None:
    """Very small absolute lift that passes old 0.0 floor should fail relative threshold."""
    bootstrap = bootstrap_lift_summary(
        pd.Series([1e-7] * 30), iterations=100, seed=1, confidence_level=0.9
    )
    decision = promotion_decision(
        {"mean_lift": 1e-7, "challenger_win_rate": 1.0, "champion_mean_score": 1000.0,
         "std_lift": 1e-8, "n_resamples": 30},
        bootstrap,
        _default_rules(min_relative_lift=0.005),
    )
    assert decision["decision"] == "inconclusive"
    assert "relative_lift" in decision["checks"]
    assert not decision["checks"]["relative_lift"]
