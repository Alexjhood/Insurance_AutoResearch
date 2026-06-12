"""Deterministic data preparation pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.holdout_vault import write_vault
from autoresearch.data.loader import load_fremtpl2
from autoresearch.data.preprocessing import DEFAULT_CAPPED_COLUMN, apply_claim_capping
from autoresearch.data.profile import build_profile
from autoresearch.data.schema import build_dataset_schema
from autoresearch.data.splits import generate_split_pack, generate_fold_assignments
from autoresearch.utils.io import write_json


def prepare_data(config: ProjectConfig) -> dict[str, Path]:
    """Run deterministic ingestion, profiling, and splitting."""

    ensure_project_dirs(config)
    raw = load_fremtpl2(config.raw_data_dir, id_column=config.id_column)
    prepared = raw.frame.copy()
    if "record_id" in prepared.columns and config.id_column != "record_id":
        if not prepared["record_id"].equals(prepared[config.id_column]):
            raise ValueError(
                "Source data contains a record_id column that conflicts with "
                f"the configured id column {config.id_column!r}"
            )
    else:
        prepared["record_id"] = prepared[config.id_column]

    # Legacy full-dataset path kept for dashboard / other consumers.
    processed_path = config.processed_dir / f"{config.agent_dataset_name}.parquet"
    dataset_schema_path = config.metadata_dir / "dataset_schema.json"
    profile_path = config.metadata_dir / "dataset_profile.json"
    capping_diagnostics_path = config.metadata_dir / "capping_diagnostics.json"
    split_path = config.splits_dir / "split_pack.csv"
    split_manifest_path = config.splits_dir / "split_pack_manifest.json"
    folds_path = config.splits_dir / "split_pack_folds.parquet"

    prepared.to_parquet(processed_path, index=False)
    dataset_schema = build_dataset_schema(
        prepared,
        id_column=config.id_column,
        exposure_column="Exposure",
        target_columns={
            "ClaimNb",
            "ClaimAmount",
            "ClaimAmountCount",
            DEFAULT_CAPPED_COLUMN,
        },
    )
    write_json(dataset_schema_path, dataset_schema)
    _, capping_diagnostics = apply_claim_capping(
        prepared,
        claim_column="ClaimAmount",
        threshold=config.claim_cap_threshold,
        enabled=config.claim_capping_enabled,
    )
    write_json(capping_diagnostics_path, capping_diagnostics)
    profile = build_profile(
        prepared,
        {
            "frequency_path": str(raw.frequency_path),
            "severity_path": str(raw.severity_path) if raw.severity_path else None,
        },
    )
    write_json(profile_path, profile)

    split_frame = _load_or_create_split_pack(
        raw.frame,
        config=config,
        split_path=split_path,
        manifest_path=split_manifest_path,
    )

    # Architectural holdout separation (§6): write search and holdout partitions.
    search_ids = set(split_frame.loc[split_frame["split"] != "milestone_holdout", "record_id"].tolist())
    holdout_ids = set(split_frame.loc[split_frame["split"] == "milestone_holdout", "record_id"].tolist())
    search_frame = prepared[prepared["record_id"].isin(search_ids)].copy()
    holdout_frame = prepared[prepared["record_id"].isin(holdout_ids)].copy()
    vault_paths = write_vault(search_frame, holdout_frame, config.processed_dir, config.holdout_vault_dir)

    # K-fold assignments on the search partition.
    if not folds_path.exists():
        fold_frame = generate_fold_assignments(
            search_frame,
            id_column="record_id",
            n_folds=config.cv_folds,
            seed=config.random_seed,
        )
        fold_frame.to_parquet(folds_path, index=False)

    return {
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


def _load_or_create_split_pack(
    frame: pd.DataFrame,
    *,
    config: ProjectConfig,
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

    split_frame, manifest = generate_split_pack(
        frame,
        id_column=config.id_column,
        ratios=config.split_ratios,
        seed=config.random_seed,
    )
    split_frame.to_csv(split_path, index=False)
    write_json(manifest_path, manifest)
    return split_frame
