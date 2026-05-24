import pandas as pd

from autoresearch.data.anonymise import anonymise_columns


def test_anonymise_columns_creates_private_mapping_and_public_schema() -> None:
    frame = pd.DataFrame(
        {
            "IDpol": [10, 20],
            "ClaimNb": [0, 1],
            "Exposure": [1.0, 0.25],
            "Region": ["A", "B"],
        }
    )

    result = anonymise_columns(frame)

    assert list(result.frame.columns) == [
        "record_id",
        "claim_count_signal_q",
        "exposure_term_a",
        "region_cluster_j",
    ]
    assert result.private_mapping["columns"][0]["original_name"] == "IDpol"
    assert result.private_mapping["columns"][1]["anonymised_name"] == "claim_count_signal_q"
    assert result.agent_schema["columns"][1]["role"] == "target_or_outcome"
    assert "original_name" not in result.agent_schema["columns"][0]
