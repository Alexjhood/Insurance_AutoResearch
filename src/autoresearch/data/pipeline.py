"""Deterministic, dataset-driven data preparation pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.holdout_vault import write_vault
from autoresearch.data.preprocessing import apply_claim_capping
from autoresearch.data.profile import build_profile
from autoresearch.data.schema import build_dataset_schema
from autoresearch.data.sources import load_raw
from autoresearch.data.splits import generate_fold_assignments, generate_split_pack
from autoresearch.datasets import DatasetSpec
from autoresearch.targets import UNIT_WEIGHT_COLUMN
from autoresearch.utils.io import write_json


def _target_role_columns(spec: DatasetSpec) -> set[str]:
    """Every target/outcome-bearing column (non-predictive, leak-sensitive)."""

    cols: set[str] = {str(t["source_column"]) for t in spec.target_configs}
    if spec.cap is not None:
        cols.add(spec.cap.column)
        cols.add(spec.cap.output_column)
    if spec.count_column:
        cols.add(spec.count_column)
    if spec.event_count_column:
        cols.add(spec.event_count_column)
    for derived in spec.derived_columns:
        cols.add(derived.name)
    return cols


def _stable_hash01(value: object, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|sample|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _apply_sample(frame: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministically subsample whole units, stratified on the target band."""

    sample = spec.sample
    assert sample is not None
    unit_col = sample.unit_column
    # Per-unit aggregate target for stratification, plus row count for sizing.
    per_unit = frame.groupby(unit_col, sort=True)
    unit_ids = pd.Index(list(per_unit.groups))
    if sample.stratify_column and sample.stratify_column in frame.columns:
        unit_target = per_unit[sample.stratify_column].sum()
        band = (unit_target.astype(float) > 0).map({True: "pos", False: "zero"})
    else:
        band = pd.Series("all", index=unit_ids)

    unit_size = per_unit.size()
    # Deterministic per-unit key; select whole units until the row budget is hit,
    # proportionally within each stratum band so the target tail is preserved.
    order = pd.DataFrame({"unit": unit_ids})
    order["band"] = band.reindex(unit_ids).to_numpy()
    order["size"] = unit_size.reindex(unit_ids).to_numpy()
    order["key"] = order["unit"].map(lambda v: _stable_hash01(v, sample.seed))

    total_rows = int(order["size"].sum())
    target_rows = min(sample.rows, total_rows)
    frac = target_rows / total_rows if total_rows else 1.0

    selected_units: list[Any] = []
    for _, grp in order.groupby("band", sort=True):
        grp = grp.sort_values(["key", "unit"])
        budget = int(round(frac * grp["size"].sum()))
        taken = 0
        for _, row in grp.iterrows():
            if taken >= budget:
                break
            selected_units.append(row["unit"])
            taken += int(row["size"])
    selected = set(selected_units)
    sampled = frame[frame[unit_col].isin(selected)].copy()

    manifest = {
        "seed": sample.seed,
        "unit_column": unit_col,
        "stratify_column": sample.stratify_column,
        "requested_rows": sample.rows,
        "source_rows": total_rows,
        "sampled_rows": int(len(sampled)),
        "source_units": int(len(unit_ids)),
        "sampled_units": int(len(selected)),
        "target_band_composition": {
            str(k): int(v) for k, v in band.reindex(list(selected)).value_counts().items()
        },
    }
    return sampled, manifest


def prepare_data(config: ProjectConfig, *, nrows: int | None = None) -> dict[str, Path]:
    """Run deterministic ingestion, sampling, profiling, capping, and splitting."""

    spec = config.dataset
    ensure_project_dirs(config)
    raw = load_raw(spec, nrows=nrows)
    prepared = raw.frame.copy()
    provenance = dict(raw.provenance)

    # Deterministic ingestion-time subsample (whole units), if configured.
    sample_manifest_path: Path | None = None
    if spec.sample is not None:
        prepared, sample_manifest = _apply_sample(prepared, spec)
        sample_manifest_path = config.metadata_dir / "sample_manifest.json"
        write_json(sample_manifest_path, sample_manifest)
        provenance["sample_manifest"] = sample_manifest

    prepared = prepared.reset_index(drop=True)

    # record_id: framework row identifier (mirrors the dataset id column).
    if "record_id" in prepared.columns and config.id_column != "record_id":
        if not prepared["record_id"].equals(prepared[config.id_column]):
            raise ValueError(
                "Source data contains a record_id column that conflicts with "
                f"the configured id column {config.id_column!r}"
            )
    else:
        prepared["record_id"] = prepared[config.id_column]

    # Synthesise a unit weight ≡ 1.0 when the dataset has no weight column.
    weight_column = spec.weight_column or UNIT_WEIGHT_COLUMN
    if spec.weight_column is None:
        prepared[UNIT_WEIGHT_COLUMN] = 1.0

    processed_path = config.processed_dir / f"{config.agent_dataset_name}.parquet"
    dataset_schema_path = config.metadata_dir / "dataset_schema.json"
    profile_path = config.metadata_dir / "dataset_profile.json"
    capping_diagnostics_path = config.metadata_dir / "capping_diagnostics.json"
    split_path = config.splits_dir / "split_pack.csv"
    split_manifest_path = config.splits_dir / "split_pack_manifest.json"
    folds_path = config.splits_dir / "split_pack_folds.parquet"

    # Capping diagnostics only. The capped column is materialised at model time
    # by the runner/comparison/milestone (historical French behaviour), so the
    # processed parquet and schema stay identical to the pre-multi-dataset ones.
    if spec.cap is not None:
        _, capping_diagnostics = apply_claim_capping(
            prepared,
            claim_column=spec.cap.column,
            threshold=spec.cap.threshold,
            enabled=config.claim_capping_enabled,
            output_column=spec.cap.output_column,
        )
    else:
        capping_diagnostics = {"claim_capping_enabled": False, "reason": "dataset declares no cap"}
    write_json(capping_diagnostics_path, capping_diagnostics)

    prepared.to_parquet(processed_path, index=False)

    dataset_schema = build_dataset_schema(
        prepared,
        id_column=config.id_column,
        exposure_column=weight_column,
        target_columns=_target_role_columns(spec),
    )
    write_json(dataset_schema_path, dataset_schema)

    profile = build_profile(prepared, provenance)
    write_json(profile_path, profile)

    split_frame = _load_or_create_split_pack(
        prepared,
        config=config,
        spec=spec,
        split_path=split_path,
        manifest_path=split_manifest_path,
    )

    # Architectural holdout separation: search vs holdout partitions.
    search_ids = set(split_frame.loc[split_frame["split"] != "milestone_holdout", "record_id"].tolist())
    holdout_ids = set(split_frame.loc[split_frame["split"] == "milestone_holdout", "record_id"].tolist())
    search_frame = prepared[prepared["record_id"].isin(search_ids)].copy()
    holdout_frame = prepared[prepared["record_id"].isin(holdout_ids)].copy()
    vault_paths = write_vault(search_frame, holdout_frame, config.processed_dir, config.holdout_vault_dir)

    # K-fold assignments on the search partition (group-aware when configured).
    if not folds_path.exists():
        fold_frame = generate_fold_assignments(
            search_frame,
            id_column="record_id",
            n_folds=config.cv_folds,
            seed=config.random_seed,
            unit_column=_grouping_unit_column(spec),
            stratify_target=spec.stratify_target,
            stratify_weight=spec.stratify_weight,
            stratify_bands=spec.stratify_bands,
        )
        fold_frame.to_parquet(folds_path, index=False)

    outputs = {
        "processed_dataset": processed_path,
        "search_dataset": vault_paths["search"],
        "holdout_dataset": vault_paths["holdout"],
        "dataset_schema": dataset_schema_path,
        "profile": profile_path,
        "capping_diagnostics": capping_diagnostics_path,
        "split_pack": split_path,
        "split_manifest": split_manifest_path,
        "fold_assignments": folds_path,
    }
    if sample_manifest_path is not None:
        outputs["sample_manifest"] = sample_manifest_path
    return outputs


def _grouping_unit_column(spec: DatasetSpec) -> str | None:
    """The split/fold grouping column, or None when splitting per-record."""

    unit = spec.split_unit_column
    return unit if unit and unit != "record_id" else None


def _load_or_create_split_pack(
    frame: pd.DataFrame,
    *,
    config: ProjectConfig,
    spec: DatasetSpec,
    split_path: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """Reuse the fixed split pack when present, otherwise create it once."""

    if split_path.exists() != manifest_path.exists():
        raise ValueError(
            "Split pack is incomplete: split_pack.csv and split_pack_manifest.json "
            "must either both exist or both be absent"
        )
    if split_path.exists():
        split_frame = pd.read_csv(split_path)
        expected_ids = set(frame[config.id_column].tolist())
        actual_ids = set(split_frame["record_id"].tolist())
        if expected_ids != actual_ids:
            raise ValueError(
                "Existing split pack does not match the loaded source data; "
                "refusing to replace the fixed split"
            )
        return split_frame

    # The grouping column must live on the frame keyed by record_id.
    unit_column = _grouping_unit_column(spec)
    split_frame, manifest = generate_split_pack(
        frame,
        id_column="record_id",
        ratios=config.split_ratios,
        seed=config.random_seed,
        unit_column=unit_column,
        stratify_target=spec.stratify_target,
        stratify_weight=spec.stratify_weight,
        stratify_bands=spec.stratify_bands,
    )
    split_frame.to_csv(split_path, index=False)
    write_json(manifest_path, manifest)
    return split_frame
