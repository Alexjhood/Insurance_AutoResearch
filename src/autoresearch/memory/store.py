"""Aggregator store — memory.sqlite schema and initialisation."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    first_seen TEXT,
    last_seen TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_uid TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    track_id TEXT,
    run_id TEXT,
    version TEXT,
    harness TEXT,
    started_at TEXT,
    last_harvested_at TEXT,
    n_experiments INTEGER,
    n_promotions INTEGER,
    peak_gini REAL,
    final_champion_id TEXT,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_uid TEXT PRIMARY KEY,
    run_uid TEXT NOT NULL,
    experiment_id TEXT,
    cycle_index INTEGER,
    model_family TEXT,
    target_strategy TEXT,
    target_mode TEXT,
    features_json TEXT,
    hyperparameters_json TEXT,
    mean_score REAL,
    std_score REAL,
    gini_weighted REAL,
    fit_wall_seconds REAL,
    compute_budget_seconds REAL,
    timed_out INTEGER,
    status TEXT,
    FOREIGN KEY (run_uid) REFERENCES runs(run_uid)
);

CREATE TABLE IF NOT EXISTS comparisons (
    comparison_uid TEXT PRIMARY KEY,
    run_uid TEXT NOT NULL,
    champion_id TEXT,
    challenger_id TEXT,
    mean_lift REAL,
    challenger_win_rate REAL,
    std_lift REAL,
    decision TEXT,
    guardrail_status TEXT,
    created_at TEXT,
    FOREIGN KEY (run_uid) REFERENCES runs(run_uid)
);

CREATE TABLE IF NOT EXISTS insights (
    insight_id TEXT PRIMARY KEY,
    run_uid TEXT NOT NULL,
    model_id TEXT NOT NULL,
    created_at TEXT,
    claim TEXT NOT NULL,
    scope TEXT NOT NULL,
    confidence REAL,
    evidence_json TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    verification_note TEXT,
    supersedes TEXT,
    contradicts TEXT,
    FOREIGN KEY (run_uid) REFERENCES runs(run_uid)
);
"""

_HOLDOUT_COLUMN_NAMES = {
    "holdout_gini",
    "holdout_score",
    "milestone_gini",
    "milestone_score",
    "holdout_mean_score",
}


def init_memory_store(path: Path) -> Path:
    """Create the aggregator database and tables. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
    return path


def memory_store_counts(path: Path) -> dict[str, int]:
    """Return row counts for each table in the aggregator."""
    if not path.exists():
        return {"models": 0, "runs": 0, "experiments": 0, "comparisons": 0, "insights": 0}
    with sqlite3.connect(path) as con:
        return {
            "models": con.execute("SELECT COUNT(*) FROM models").fetchone()[0],
            "runs": con.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "experiments": con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0],
            "comparisons": con.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0],
            "insights": con.execute("SELECT COUNT(*) FROM insights").fetchone()[0],
        }


def assert_no_holdout_columns(path: Path) -> None:
    """Raise AssertionError if any holdout-derived column name exists in the schema."""
    with sqlite3.connect(path) as con:
        for table in ("models", "runs", "experiments", "comparisons", "insights"):
            try:
                cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
            except Exception:
                continue
            leaks = cols & _HOLDOUT_COLUMN_NAMES
            if leaks:
                raise AssertionError(f"Holdout-derived columns found in {table}: {leaks}")
