"""Column anonymisation for agent-facing artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


SEMANTIC_NAME_MAP = {
    "IDpol": "record_id",
    "ClaimNb": "claim_count_signal_q",
    "Exposure": "exposure_term_a",
    "VehPower": "vehicle_power_band_b",
    "VehAge": "vehicle_age_band_c",
    "DrivAge": "driver_age_band_d",
    "BonusMalus": "risk_score_index_e",
    "VehBrand": "vehicle_make_group_f",
    "VehGas": "vehicle_energy_type_g",
    "Area": "territory_band_h",
    "Density": "density_index_i",
    "Region": "region_cluster_j",
    "ClaimAmount": "claim_cost_observed_k",
    "ClaimAmountCount": "claim_event_count_l",
}


@dataclass(frozen=True)
class AnonymisedDataset:
    """Anonymised frame and metadata for private and agent-facing use."""

    frame: pd.DataFrame
    private_mapping: dict[str, Any]
    agent_schema: dict[str, Any]


def infer_role(column: str, series: pd.Series, id_column: str) -> str:
    """Infer a simple semantic role without exposing original names to agents."""

    lower = column.lower()
    if column == id_column:
        return "record_id"
    if lower in {"claimnb", "claimamount", "claimamountcount"}:
        return "target_or_outcome"
    if lower == "exposure":
        return "exposure_offset"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric_feature"
    return "categorical_feature"


def anonymise_columns(frame: pd.DataFrame, id_column: str = "IDpol") -> AnonymisedDataset:
    """Rename columns to lightly obfuscated semantic names and emit metadata."""

    rename: dict[str, str] = {}
    private_columns: list[dict[str, Any]] = []
    public_columns: list[dict[str, Any]] = []

    for index, column in enumerate(frame.columns, start=1):
        anon = SEMANTIC_NAME_MAP.get(column)
        if anon is None:
            role_prefix = "numeric_field" if pd.api.types.is_numeric_dtype(frame[column]) else "categorical_field"
            anon = f"{role_prefix}_{index:03d}"
        role = infer_role(column, frame[column], id_column)
        dtype = str(frame[column].dtype)
        rename[column] = anon
        private_columns.append(
            {
                "original_name": column,
                "anonymised_name": anon,
                "dtype": dtype,
                "role": role,
            }
        )
        public_columns.append(
            {
                "name": anon,
                "dtype": dtype,
                "role": role,
                "missing_count": int(frame[column].isna().sum()),
                "unique_count": int(frame[column].nunique(dropna=True)),
            }
        )

    anonymised = frame.rename(columns=rename)
    private_mapping = {
        "mapping_version": 1,
        "id_column": id_column,
        "columns": private_columns,
    }
    agent_schema = {
        "schema_version": 1,
        "row_count": int(len(frame)),
        "columns": public_columns,
        "notes": (
            "Lightly obfuscated semantic field names are used for agent-facing artifacts. "
            "Private source-column mapping is stored separately and should not be shown to agents."
        ),
    }
    return AnonymisedDataset(anonymised, private_mapping, agent_schema)
