import pandas as pd

from autoresearch.evaluation.metrics import evaluate_predictions


def test_evaluate_predictions_rejects_milestone_holdout() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["milestone_holdout"],
            "actual_claim_cost": [10.0],
            "predicted_claim_cost": [10.0],
            "exposure": [1.0],
        }
    )

    try:
        evaluate_predictions(predictions, ("search_validation",))
    except ValueError as exc:
        assert "milestone_holdout" in str(exc)
    else:
        raise AssertionError("Expected milestone holdout rejection")
