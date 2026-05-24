import pandas as pd

from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    paired_comparison,
    promotion_decision,
    repeated_scores,
)


def _predictions(predicted: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "split": ["search_validation"] * 4,
            "actual_claim_cost": [10.0, 20.0, 30.0, 40.0],
            "predicted_claim_cost": predicted,
            "exposure": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_repeated_scores_are_reproducible() -> None:
    predictions = _predictions([10.0, 18.0, 35.0, 39.0])

    first = repeated_scores(predictions, eval_split="search_validation", n_resamples=5, seed=123)
    second = repeated_scores(predictions, eval_split="search_validation", n_resamples=5, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_paired_comparison_positive_lift_when_challenger_is_better() -> None:
    champion = _predictions([0.0, 0.0, 0.0, 0.0])
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


def test_bootstrap_and_promotion_decision() -> None:
    bootstrap = bootstrap_lift_summary(pd.Series([1.0, 2.0, 3.0]), iterations=100, seed=1, confidence_level=0.9)
    decision = promotion_decision(
        {"mean_lift": 2.0, "challenger_win_rate": 1.0},
        bootstrap,
        PromotionRules(
            minimum_mean_lift=0.0,
            minimum_win_rate=0.55,
            bootstrap_lower_bound=0.0,
            confidence_level=0.9,
        ),
    )

    assert bootstrap["probability_challenger_outperforms"] == 1.0
    assert decision["decision"] == "promote"
