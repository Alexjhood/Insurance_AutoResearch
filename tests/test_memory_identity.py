"""Tests for model identity capture in bootstrap and run manifests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.bootstrap import bootstrap_track
from autoresearch.config import ProjectConfig, ensure_project_dirs
from tests.test_runner import _make_config as _base_config
from tests.test_handoff import _record_direct


def _config_with_identity(tmp_path: Path, **identity_kwargs) -> ProjectConfig:
    config = _base_config(tmp_path)
    # track_base_dir must be set for ensure_project_dirs to write the manifest
    track_base = tmp_path / "tracks" / "test"
    config = replace(
        config,
        track_base_dir=track_base,
        artifacts_dir=track_base / "runs" / config.run_id,
        **identity_kwargs,
    )
    return config


def _write_prepared_data_markers(config: ProjectConfig) -> None:
    for path in (
        config.processed_dir / f"{config.agent_dataset_name}.parquet",
        config.processed_dir / "agent_dataset_search.parquet",
        config.holdout_vault_dir / "agent_dataset_holdout.parquet",
        config.metadata_dir / "agent_schema.json",
        config.metadata_dir / "dataset_profile.json",
        config.metadata_dir / "capping_diagnostics.json",
        config.splits_dir / "split_pack.csv",
        config.splits_dir / "split_pack_manifest.json",
        config.splits_dir / "split_pack_folds.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# bootstrap_track refuses without identity
# ---------------------------------------------------------------------------


def test_bootstrap_track_refuses_without_identity(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    _write_prepared_data_markers(config)
    _record_direct(config)
    with pytest.raises(ValueError, match="model identity"):
        bootstrap_track(config)


def test_bootstrap_track_refuses_with_provider_only(tmp_path: Path) -> None:
    config = _config_with_identity(tmp_path, model_provider="anthropic")
    _write_prepared_data_markers(config)
    _record_direct(config)
    with pytest.raises(ValueError, match="model identity"):
        bootstrap_track(config)


def test_bootstrap_track_refuses_with_name_only(tmp_path: Path) -> None:
    config = _config_with_identity(tmp_path, model_name="claude-sonnet-4-6")
    _write_prepared_data_markers(config)
    _record_direct(config)
    with pytest.raises(ValueError, match="model identity"):
        bootstrap_track(config)


# ---------------------------------------------------------------------------
# Manifest writes model_identity when identity is present
# ---------------------------------------------------------------------------


def test_ensure_project_dirs_writes_identity_into_new_manifest(tmp_path: Path) -> None:
    config = _config_with_identity(
        tmp_path,
        model_provider="Anthropic",
        model_name="Claude-Sonnet-4-6",
        model_version="20251101",
        model_harness="claude-code",
    )
    ensure_project_dirs(config)
    manifest_path = config.artifacts_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "model_identity" in manifest
    identity = manifest["model_identity"]
    assert identity["provider"] == "anthropic"  # lowercased
    assert identity["name"] == "claude-sonnet-4-6"  # lowercased
    assert identity["version"] == "20251101"
    assert identity["harness"] == "claude-code"


def test_ensure_project_dirs_patches_existing_manifest_without_identity(tmp_path: Path) -> None:
    config = _config_with_identity(
        tmp_path,
        model_provider="openai",
        model_name="gpt-4o",
    )
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.track_base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.artifacts_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"track_id": "test", "run_id": "r1"}), encoding="utf-8"
    )
    ensure_project_dirs(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_identity"]["provider"] == "openai"
    assert manifest["model_identity"]["name"] == "gpt-4o"


def test_ensure_project_dirs_does_not_overwrite_existing_identity(tmp_path: Path) -> None:
    config = _config_with_identity(
        tmp_path,
        model_provider="openai",
        model_name="gpt-4o",
    )
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.track_base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.artifacts_dir / "run_manifest.json"
    existing_identity = {"provider": "anthropic", "name": "claude-sonnet-4-6", "version": "", "harness": ""}
    manifest_path.write_text(
        json.dumps({"track_id": "test", "model_identity": existing_identity}),
        encoding="utf-8",
    )
    ensure_project_dirs(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Existing identity must be preserved
    assert manifest["model_identity"]["provider"] == "anthropic"


def test_ensure_project_dirs_no_identity_without_config(tmp_path: Path) -> None:
    config = _base_config(tmp_path)  # no identity fields
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    ensure_project_dirs(config)
    manifest_path = config.artifacts_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "model_identity" not in manifest


# ---------------------------------------------------------------------------
# model_id slug computation
# ---------------------------------------------------------------------------


def test_model_id_slug_format(tmp_path: Path) -> None:
    """model_id in the aggregator must be provider/name (lowercased)."""
    from autoresearch.memory.harvester import harvest_run
    from autoresearch.memory.store import init_memory_store
    import sqlite3

    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"

    import sqlite3 as _sq
    registry.parent.mkdir(parents=True, exist_ok=True)
    with _sq.connect(registry) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY, created_at TEXT, status TEXT,
                model_family TEXT, target_strategy TEXT, target_mode TEXT,
                metrics_path TEXT, fit_wall_seconds REAL,
                compute_budget_seconds REAL, timed_out INTEGER
            );
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY, created_at TEXT, champion_id TEXT,
                challenger_id TEXT, paired_summary TEXT, promotion_decision TEXT,
                decision TEXT, guardrail_status TEXT
            );
            CREATE TABLE IF NOT EXISTS champion_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
                previous_champion_id TEXT, new_champion_id TEXT,
                branch_id TEXT, action TEXT, reason TEXT
            );
            CREATE TABLE IF NOT EXISTS auto_sessions (session_id TEXT PRIMARY KEY, created_at TEXT);
            """
        )

    identity = {"provider": "Anthropic", "name": "Claude-Opus-4-8", "version": "20251101"}
    harvest_run(memory, registry, identity, track_id="t", run_id="r")

    with sqlite3.connect(memory) as con:
        row = con.execute("SELECT model_id, provider, name FROM models").fetchone()
    assert row[0] == "anthropic/claude-opus-4-8"
    assert row[1] == "anthropic"
    assert row[2] == "claude-opus-4-8"
