"""Phase 1 deterministic data preparation pipeline."""

from __future__ import annotations

from pathlib import Path

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.anonymise import anonymise_columns
from autoresearch.data.loader import load_fremtpl2
from autoresearch.data.preprocessing import apply_claim_capping
from autoresearch.data.profile import build_profile
from autoresearch.data.splits import generate_split_pack
from autoresearch.utils.io import write_json


def prepare_data(config: ProjectConfig) -> dict[str, Path]:
    """Run deterministic ingestion, anonymisation, profiling, and splitting."""

    ensure_project_dirs(config)
    raw = load_fremtpl2(config.raw_data_dir, id_column=config.id_column)
    anonymised = anonymise_columns(raw.frame, id_column=config.id_column)

    processed_path = config.processed_dir / f"{config.agent_dataset_name}.parquet"
    private_mapping_path = config.metadata_dir / "private_column_mapping.json"
    agent_schema_path = config.metadata_dir / "agent_schema.json"
    profile_path = config.metadata_dir / "dataset_profile.json"
    capping_diagnostics_path = config.metadata_dir / "capping_diagnostics.json"
    split_path = config.splits_dir / "split_pack.csv"
    split_manifest_path = config.splits_dir / "split_pack_manifest.json"

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

    return {
        "processed_dataset": processed_path,
        "private_mapping": private_mapping_path,
        "agent_schema": agent_schema_path,
        "profile": profile_path,
        "capping_diagnostics": capping_diagnostics_path,
        "split_pack": split_path,
        "split_manifest": split_manifest_path,
    }
