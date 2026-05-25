"""Deterministic data preparation pipeline."""

from __future__ import annotations

from pathlib import Path

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.anonymise import anonymise_columns
from autoresearch.data.holdout_vault import write_vault
from autoresearch.data.loader import load_fremtpl2
from autoresearch.data.preprocessing import apply_claim_capping
from autoresearch.data.profile import build_profile
from autoresearch.data.splits import generate_split_pack, generate_fold_assignments
from autoresearch.utils.io import write_json


def prepare_data(config: ProjectConfig) -> dict[str, Path]:
    """Run deterministic ingestion, anonymisation, profiling, and splitting."""

    ensure_project_dirs(config)
    raw = load_fremtpl2(config.raw_data_dir, id_column=config.id_column)
    anonymised = anonymise_columns(raw.frame, id_column=config.id_column)

    # Legacy full-dataset path kept for dashboard / other consumers.
    processed_path = config.processed_dir / f"{config.agent_dataset_name}.parquet"
    private_mapping_path = config.metadata_dir / "private_column_mapping.json"
    agent_schema_path = config.metadata_dir / "agent_schema.json"
    profile_path = config.metadata_dir / "dataset_profile.json"
    capping_diagnostics_path = config.metadata_dir / "capping_diagnostics.json"
    split_path = config.splits_dir / "split_pack.csv"
    split_manifest_path = config.splits_dir / "split_pack_manifest.json"
    folds_path = config.splits_dir / "split_pack_folds.parquet"

    anonymised.frame.to_parquet(processed_path, index=False)
    write_json(private_mapping_path, anonymised.private_mapping)
    write_json(agent_schema_path, anonymised.agent_schema)
    _, capping_diagnostics = apply_claim_capping(
        anonymised.frame,
        claim_column="claim_cost_observed_k",
        threshold=config.claim_cap_threshold,
        enabled=config.claim_capping_enabled,
    )
    write_json(capping_diagnostics_path, capping_diagnostics)
    profile = build_profile(
        anonymised.frame,
        {
            "frequency_path": str(raw.frequency_path),
            "severity_path": str(raw.severity_path) if raw.severity_path else None,
        },
    )
    write_json(profile_path, profile)

    split_frame, manifest = generate_split_pack(
        raw.frame,
        id_column=config.id_column,
        ratios=config.split_ratios,
        seed=config.random_seed,
    )
    split_frame.to_csv(split_path, index=False)
    write_json(split_manifest_path, manifest)

    # Architectural holdout separation (§6): write search and holdout partitions.
    search_ids = set(split_frame.loc[split_frame["split"] != "milestone_holdout", "record_id"].tolist())
    holdout_ids = set(split_frame.loc[split_frame["split"] == "milestone_holdout", "record_id"].tolist())
    search_frame = anonymised.frame[anonymised.frame["record_id"].isin(search_ids)].copy()
    holdout_frame = anonymised.frame[anonymised.frame["record_id"].isin(holdout_ids)].copy()
    vault_paths = write_vault(search_frame, holdout_frame, config.processed_dir, config.holdout_vault_dir)

    # K-fold assignments on the search partition.
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
        "private_mapping": private_mapping_path,
        "agent_schema": agent_schema_path,
        "profile": profile_path,
        "capping_diagnostics": capping_diagnostics_path,
        "split_pack": split_path,
        "split_manifest": split_manifest_path,
        "fold_assignments": folds_path,
    }
