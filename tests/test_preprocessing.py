import pandas as pd

from autoresearch.data.preprocessing import apply_claim_capping


def test_apply_claim_capping_preserves_uncapped_and_reports_diagnostics() -> None:
    frame = pd.DataFrame({"ClaimAmount": [0.0, 50.0, 150.0]})

    capped, diagnostics = apply_claim_capping(frame, "ClaimAmount", threshold=100.0)

    assert capped["ClaimAmount"].tolist() == [0.0, 50.0, 150.0]
    assert capped["ClaimAmountCapped"].tolist() == [0.0, 50.0, 100.0]
    assert diagnostics["capped_row_count"] == 1
    assert diagnostics["total_claim_cost_reduction"] == 50.0
