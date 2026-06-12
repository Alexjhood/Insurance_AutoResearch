import pandas as pd

from autoresearch.data.schema import build_dataset_schema


def test_dataset_schema_retains_source_names_and_assigns_roles() -> None:
    frame = pd.DataFrame(
        {
            "IDpol": [10, 20],
            "record_id": [10, 20],
            "ClaimNb": [0, 1],
            "Exposure": [1.0, 0.25],
            "Region": ["A", "B"],
        }
    )

    schema = build_dataset_schema(
        frame,
        id_column="IDpol",
        exposure_column="Exposure",
        target_columns={"ClaimNb"},
    )

    assert [column["name"] for column in schema["columns"]] == list(frame.columns)
    assert schema["columns"][0]["role"] == "record_id"
    assert schema["columns"][1]["role"] == "record_id"
    assert schema["columns"][2]["role"] == "target_or_outcome"
    assert schema["columns"][3]["role"] == "exposure_offset"
