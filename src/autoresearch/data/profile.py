"""Dataset profile generation for review and dashboard display."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_profile(frame: pd.DataFrame, source_paths: dict[str, str | None]) -> dict[str, Any]:
    """Build a compact JSON-serialisable profile for an anonymised dataset."""

    columns: list[dict[str, Any]] = []
    for name, series in frame.items():
        entry: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            entry["summary"] = {
                key: float(value)
                for key, value in desc.items()
                if key in {"mean", "std", "min", "25%", "50%", "75%", "max"}
            }
        else:
            top_values = series.value_counts(dropna=True).head(10)
            entry["top_values"] = {str(key): int(value) for key, value in top_values.items()}
        columns.append(entry)

    return {
        "profile_version": 1,
        "row_count": int(len(frame)),
        "column_count": int(frame.shape[1]),
        "source_paths": source_paths,
        "columns": columns,
    }
