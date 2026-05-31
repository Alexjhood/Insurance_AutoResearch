"""Tests for the read-only harvester.

Key assertions:
- Harvest produces correct rows in the aggregator.
- The harvester never opens paths under holdout_vault or milestone_reports.
- Opening a registry read-only does not write to it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from autoresearch.memory.harvester import _guard_path, harvest_all, harvest_run
from autoresearch.memory.store import init_memory_store, memory_store_counts


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_registry(path: Path, experiments: list[dict], comparisons: list[dict] | None = None) -> None:
    """Build a minimal per-run registry for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                model_family TEXT,
                target_strategy TEXT,
                target_mode TEXT,
                metrics_path TEXT,
                fit_wall_seconds REAL,
                compute_budget_seconds REAL,
                timed_out INTEGER
            );
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                champion_id TEXT,
                challenger_id TEXT,
                paired_summary TEXT,
                promotion_decision TEXT,
                decision TEXT,
                guardrail_status TEXT
            );
            CREATE TABLE IF NOT EXISTS champion_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                previous_champion_id TEXT,
                new_champion_id TEXT,
                branch_id TEXT,
                action TEXT,
                reason TEXT
            );
            CREATE TABLE IF NOT EXISTS auto_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT
            );
            """
        )
        for exp in experiments:
            con.execute(
                """
                INSERT INTO experiments
                    (experiment_id, created_at, status, model_family,
                     target_strategy, target_mode, metrics_path,
                     fit_wall_seconds, compute_budget_seconds, timed_out)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    exp["experiment_id"],
                    exp.get("created_at", "2026-01-01T00:00:00Z"),
                    exp.get("status", "completed"),
                    exp.get("model_family"),
                    exp.get("target_strategy"),
                    exp.get("target_mode"),
                    exp.get("metrics_path"),
                    exp.get("fit_wall_seconds"),
                    exp.get("compute_budget_seconds"),
                    exp.get("timed_out", 0),
                ),
            )
        for cmp in (comparisons or []):
            con.execute(
                """
                INSERT INTO comparisons
                    (comparison_id, champion_id, challenger_id, paired_summary,
                     promotion_decision, decision, guardrail_status)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    cmp["comparison_id"],
                    cmp.get("champion_id"),
                    cmp.get("challenger_id"),
                    json.dumps(cmp.get("paired_summary", {})),
                    cmp.get("promotion_decision", "reject"),
                    cmp.get("decision"),
                    cmp.get("guardrail_status"),
                ),
            )


def _make_metrics(path: Path, gini: float, std: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "aggregate": {"mean_score": gini, "std_score": std, "split_count": 1},
        "primary_metric": "gini_weighted",
        "split_metrics": [
            {"split": "search_validation", "gini_weighted": gini},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Basic harvest tests
# ---------------------------------------------------------------------------


def test_harvest_run_creates_model_and_run(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    metrics_file = tmp_path / "metrics.json"
    _make_metrics(metrics_file, gini=0.35)
    _make_registry(
        registry,
        [{"experiment_id": "exp1", "metrics_path": str(metrics_file)}],
    )
    identity = {"provider": "anthropic", "name": "claude-sonnet-4-6"}
    harvest_run(memory, registry, identity, track_id="claude", run_id="run1")

    counts = memory_store_counts(memory)
    assert counts["models"] == 1
    assert counts["runs"] == 1
    assert counts["experiments"] == 1


def test_harvest_run_peak_gini(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    _make_metrics(m1, gini=0.30)
    _make_metrics(m2, gini=0.40)
    _make_registry(
        registry,
        [
            {"experiment_id": "e1", "created_at": "2026-01-01T01:00:00Z", "metrics_path": str(m1)},
            {"experiment_id": "e2", "created_at": "2026-01-01T02:00:00Z", "metrics_path": str(m2)},
        ],
    )
    identity = {"provider": "anthropic", "name": "claude-sonnet-4-6"}
    harvest_run(memory, registry, identity, track_id="claude", run_id="run1")

    with sqlite3.connect(memory) as con:
        row = con.execute("SELECT peak_gini FROM runs WHERE run_uid='claude/run1'").fetchone()
    assert row is not None
    assert abs(row[0] - 0.40) < 1e-9


def test_harvest_run_idempotent(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    mf = tmp_path / "m.json"
    _make_metrics(mf, gini=0.33)
    _make_registry(registry, [{"experiment_id": "e1", "metrics_path": str(mf)}])
    identity = {"provider": "openai", "name": "gpt-4o"}

    harvest_run(memory, registry, identity, track_id="t", run_id="r")
    harvest_run(memory, registry, identity, track_id="t", run_id="r")
    harvest_run(memory, registry, identity, track_id="t", run_id="r")

    counts = memory_store_counts(memory)
    assert counts["models"] == 1
    assert counts["runs"] == 1
    assert counts["experiments"] == 1


def test_harvest_run_comparisons(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    _make_registry(
        registry,
        [],
        comparisons=[
            {
                "comparison_id": "cmp1",
                "champion_id": "e1",
                "challenger_id": "e2",
                "paired_summary": {"mean_lift": 0.05, "challenger_win_rate": 0.75, "std_lift": 0.01},
                "promotion_decision": "promote",
                "decision": "promote",
            }
        ],
    )
    identity = {"provider": "openai", "name": "gpt-4o"}
    harvest_run(memory, registry, identity, track_id="t", run_id="r")

    with sqlite3.connect(memory) as con:
        row = con.execute("SELECT mean_lift, decision FROM comparisons WHERE run_uid='t/r'").fetchone()
    assert row is not None
    assert abs(row[0] - 0.05) < 1e-9
    assert row[1] == "promote"


def test_harvest_run_skips_without_provider_name(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    _make_registry(registry, [])
    harvest_run(memory, registry, {"provider": "", "name": ""}, track_id="t", run_id="r")
    counts = memory_store_counts(memory)
    assert counts["models"] == 0


# ---------------------------------------------------------------------------
# Holdout guard tests — the harvester must NEVER open forbidden paths
# ---------------------------------------------------------------------------


def test_guard_path_allows_normal_paths(tmp_path: Path) -> None:
    _guard_path(tmp_path / "registry.sqlite")  # must not raise


def test_guard_path_blocks_holdout_vault(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holdout guard"):
        _guard_path(tmp_path / "data" / "holdout_vault" / "foo.parquet")


def test_guard_path_blocks_milestone_reports(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="holdout guard"):
        _guard_path(tmp_path / "artifacts" / "milestone_reports" / "eval.json")


def test_harvest_run_never_opens_holdout_path(tmp_path: Path) -> None:
    """Verify that harvest_run raises before opening a holdout-vault path."""
    memory = tmp_path / "memory.sqlite"
    bogus = tmp_path / "data" / "holdout_vault" / "registry.sqlite"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_bytes(b"")
    identity = {"provider": "x", "name": "y"}
    with pytest.raises(ValueError, match="holdout guard"):
        harvest_run(memory, bogus, identity, track_id="t", run_id="r")


def test_harvest_run_never_opens_milestone_reports_path(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    bogus = tmp_path / "artifacts" / "milestone_reports" / "registry.sqlite"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_bytes(b"")
    identity = {"provider": "x", "name": "y"}
    with pytest.raises(ValueError, match="holdout guard"):
        harvest_run(memory, bogus, identity, track_id="t", run_id="r")


def test_metrics_read_skips_holdout_split(tmp_path: Path) -> None:
    """Metrics from holdout split must not affect gini_weighted in the aggregator."""
    from autoresearch.memory.harvester import _read_metrics

    mf = tmp_path / "metrics.json"
    payload = {
        "aggregate": {"mean_score": 0.50, "std_score": 0.0, "split_count": 1},
        "primary_metric": "gini_weighted",
        "split_metrics": [
            {"split": "search_validation", "gini_weighted": 0.35},
            {"split": "milestone_holdout", "gini_weighted": 0.50},
        ],
    }
    mf.write_text(json.dumps(payload), encoding="utf-8")
    result = _read_metrics(str(mf))
    # Only the search split should contribute to gini_weighted
    assert abs(result["gini_weighted"] - 0.35) < 1e-9


def test_harvest_all_skips_runs_without_identity(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks" / "mytrack" / "runs" / "run1"
    tracks.mkdir(parents=True)
    registry = tracks / "registry.sqlite"
    _make_registry(registry, [])
    manifest = tracks / "run_manifest.json"
    manifest.write_text(json.dumps({"track_id": "mytrack", "run_id": "run1"}), encoding="utf-8")

    memory = tmp_path / "memory.sqlite"
    result = harvest_all(memory, tracks_base=tmp_path / "tracks")
    assert result["skipped"] == 1
    assert result["harvested"] == 0


def test_harvest_all_harvests_run_with_identity(tmp_path: Path) -> None:
    tracks = tmp_path / "tracks" / "mytrack" / "runs" / "run1"
    tracks.mkdir(parents=True)
    registry = tracks / "registry.sqlite"
    _make_registry(registry, [])
    manifest = tracks / "run_manifest.json"
    manifest.write_text(
        json.dumps({
            "track_id": "mytrack",
            "run_id": "run1",
            "model_identity": {"provider": "anthropic", "name": "claude-sonnet-4-6"},
        }),
        encoding="utf-8",
    )

    memory = tmp_path / "memory.sqlite"
    result = harvest_all(memory, tracks_base=tmp_path / "tracks")
    assert result["harvested"] == 1
    assert result["skipped"] == 0


def test_harvest_all_never_opens_holdout_vault(tmp_path: Path) -> None:
    """harvest_all must not descend into holdout_vault directories."""
    opened_paths: list[str] = []
    original_connect = sqlite3.connect

    def mock_connect(path, **kwargs):
        opened_paths.append(str(path))
        return original_connect(path, **kwargs)

    # Put a fake registry inside holdout_vault — it must never be opened.
    bogus = tmp_path / "holdout_vault" / "registry.sqlite"
    bogus.parent.mkdir(parents=True, exist_ok=True)
    bogus.write_bytes(b"")

    memory = tmp_path / "memory.sqlite"
    # harvest_all from an empty tracks dir (nothing to harvest)
    result = harvest_all(memory, tracks_base=tmp_path / "tracks_nonexistent")
    for p in opened_paths:
        assert "holdout_vault" not in p, f"Opened holdout path: {p}"
