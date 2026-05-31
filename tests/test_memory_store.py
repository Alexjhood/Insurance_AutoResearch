"""Tests for the aggregator store schema and upsert idempotency."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from autoresearch.memory.store import (
    assert_no_holdout_columns,
    init_memory_store,
    memory_store_counts,
)


def test_init_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    assert db.exists()
    with sqlite3.connect(db) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert {"models", "runs", "experiments", "comparisons", "insights"} <= tables


def test_init_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    init_memory_store(db)  # second call must not raise


def test_memory_store_counts_empty(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    counts = memory_store_counts(db)
    assert counts == {
        "models": 0,
        "runs": 0,
        "experiments": 0,
        "comparisons": 0,
        "insights": 0,
    }


def test_memory_store_counts_nonexistent(tmp_path: Path) -> None:
    db = tmp_path / "nonexistent.sqlite"
    counts = memory_store_counts(db)
    assert all(v == 0 for v in counts.values())


def test_assert_no_holdout_columns_passes(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    # Must not raise
    assert_no_holdout_columns(db)


def test_assert_no_holdout_columns_fails_when_added(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE experiments ADD COLUMN holdout_gini REAL")
    with pytest.raises(AssertionError, match="holdout"):
        assert_no_holdout_columns(db)


def test_upsert_models_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    with sqlite3.connect(db) as con:
        for _ in range(3):
            con.execute(
                """
                INSERT INTO models (model_id, provider, name, first_seen, last_seen)
                VALUES ('a/b', 'a', 'b', '2026-01-01', '2026-01-02')
                ON CONFLICT(model_id) DO UPDATE SET last_seen=excluded.last_seen
                """
            )
    counts = memory_store_counts(db)
    assert counts["models"] == 1


def test_upsert_runs_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO models (model_id, provider, name) VALUES ('p/m', 'p', 'm')"
        )
        for _ in range(2):
            con.execute(
                """
                INSERT INTO runs (run_uid, model_id, track_id, run_id, n_experiments, peak_gini)
                VALUES ('t/r', 'p/m', 't', 'r', 5, 0.35)
                ON CONFLICT(run_uid) DO UPDATE SET
                    n_experiments=excluded.n_experiments,
                    peak_gini=excluded.peak_gini
                """
            )
    counts = memory_store_counts(db)
    assert counts["runs"] == 1


def test_schema_has_no_holdout_columns_by_default(tmp_path: Path) -> None:
    """Core invariant: the aggregator schema must never include holdout columns."""
    db = tmp_path / "memory.sqlite"
    init_memory_store(db)
    with sqlite3.connect(db) as con:
        for table in ("models", "runs", "experiments", "comparisons", "insights"):
            cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
            for col in cols:
                assert "holdout" not in col.lower(), f"Holdout column {col!r} found in table {table}"
                assert "milestone" not in col.lower() or col == "model_id", (
                    f"Milestone column {col!r} found in table {table}"
                )
