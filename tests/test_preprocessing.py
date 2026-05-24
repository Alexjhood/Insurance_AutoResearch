import pandas as pd

from autoresearch.data.preprocessing import apply_claim_capping


def test_apply_claim_capping_preserves_uncapped_and_reports_diagnostics() -> None:
    frame = pd.DataFrame({"claim_cost_observed_k": [0.0, 50.0, 150.0]})

    capped, diagnostics = apply_claim_capping(frame, "claim_cost_observed_k", threshold=100.0)

    assert capped["claim_cost_observed_k"].tolist() == [0.0, 50.0, 150.0]
    assert capped["claim_cost_capped_active"].tolist() == [0.0, 50.0, 100.0]
    assert diagnostics["capped_row_count"] == 1
    assert diagnostics["total_claim_cost_reduction"] == 50.0
