"""Dataset schema metadata built from source column names."""

from __future__ import annotations

from typing import Any

import pandas as pd


def infer_role(
    column: str,
    series: pd.Series,
    *,
    id_columns: set[str],
    exposure_column: str,
    target_columns: set[str],
) -> str:
    """Infer the modelling role of a source or framework column."""

    if column in id_columns:
        return "record_id"
    if column in target_columns:
        return "target_or_outcome"
    if column == exposure_column:
        return "exposure_offset"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric_feature"
    return "categorical_feature"


def build_dataset_schema(
    frame: pd.DataFrame,
    *,
    id_column: str,
    exposure_column: str,
    target_columns: set[str],
) -> dict[str, Any]:
    """Describe the prepared dataset without renaming its source columns."""

    id_columns = {id_column, "record_id"}
    columns: list[dict[str, Any]] = []
    for name, series in frame.items():
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "role": infer_role(
                    name,
                    series,
                    id_columns=id_columns,
                    exposure_column=exposure_column,
                    target_columns=target_columns,
                ),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
            }
        )

    return {
        "schema_version": 2,
        "row_count": int(len(frame)),
        "columns": columns,
    }
