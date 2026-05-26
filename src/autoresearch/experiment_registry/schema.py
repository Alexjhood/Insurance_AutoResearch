"""Registry schema and initialisation."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    experiment_name TEXT,
    status TEXT NOT NULL,
    parent_experiment_id TEXT,
    model_family TEXT,
    target_strategy TEXT,
    preprocessing_summary TEXT,
    claim_cap_threshold REAL,
    config_snapshot_path TEXT,
    code_version TEXT,
    rationale_path TEXT,
    metrics_path TEXT,
    uncertainty_summary_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS comparisons (
    comparison_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    champion_id TEXT NOT NULL,
    challenger_id TEXT NOT NULL,
    paired_summary TEXT NOT NULL,
    bootstrap_summary TEXT NOT NULL,
    promotion_decision TEXT NOT NULL,
    promotion_rationale TEXT NOT NULL,
    comparison_summary_path TEXT,
    paired_scores_path TEXT,
    bootstrap_summary_path TEXT,
    promotion_decision_path TEXT,
    report_path TEXT
);

CREATE TABLE IF NOT EXISTS champion_state (
    state_id TEXT PRIMARY KEY,
    champion_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reason TEXT NOT NULL,
    comparison_id TEXT,
    proposal_id TEXT
);

CREATE TABLE IF NOT EXISTS champion_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    previous_champion_id TEXT,
    new_champion_id TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reason TEXT NOT NULL,
    comparison_id TEXT,
    proposal_id TEXT
);

CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    parent_branch_id TEXT,
    root_experiment_id TEXT,
    current_experiment_id TEXT,
    status TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL,
    parent_experiment_id TEXT,
    parent_branch_id TEXT,
    branch_id TEXT,
    experiment_id TEXT,
    comparison_id TEXT,
    experiment_name TEXT,
    rationale TEXT,
    change_summary TEXT,
    expected_benefit TEXT,
    key_risk TEXT,
    config_json TEXT,
    validation_errors_json TEXT,
    llm_provider TEXT,
    llm_model TEXT,
    prompt_path TEXT,
    response_path TEXT,
    proposal_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS auto_sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state TEXT NOT NULL,
    current_cycle INTEGER NOT NULL DEFAULT 0,
    max_cycles INTEGER,
    stop_requested INTEGER NOT NULL DEFAULT 0,
    state_path TEXT,
    summary_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS session_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    state TEXT,
    proposal_id TEXT,
    experiment_id TEXT,
    comparison_id TEXT,
    message TEXT,
    details_json TEXT,
    FOREIGN KEY (session_id) REFERENCES auto_sessions(session_id)
);
"""


def init_registry(path: Path) -> Path:
    """Create the registry database, required tables, and integrity manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        con.executescript(SCHEMA)
        _migrate_experiments(con)

    from autoresearch.config import PROJECT_ROOT
    from autoresearch.utils.integrity import write_integrity_manifest
    write_integrity_manifest(PROJECT_ROOT, path.parent)

    return path


def _migrate_experiments(con: sqlite3.Connection) -> None:
    """Add Phase 2 columns to registries created by earlier phases."""

    existing = {row[1] for row in con.execute("PRAGMA table_info(experiments)").fetchall()}
    columns = {
        "updated_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "experiment_name": "TEXT",
        "model_family": "TEXT",
        "target_strategy": "TEXT",
        "preprocessing_summary": "TEXT",
        "claim_cap_threshold": "REAL",
    }
    for column, definition in columns.items():
        if column not in existing:
            con.execute(f"ALTER TABLE experiments ADD COLUMN {column} {definition}")


def registry_counts(path: Path) -> dict[str, int]:
    """Return table counts for dashboard status checks."""

    if not path.exists():
        return {"experiments": 0, "artifacts": 0, "comparisons": 0, "proposals": 0, "branches": 0, "sessions": 0}
    init_registry(path)
    with sqlite3.connect(path) as con:
        exp_count = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        art_count = con.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        cmp_count = con.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
        proposal_count = con.execute("SELECT COUNT(*) FROM proposals").fetchone()[0]
        branch_count = con.execute("SELECT COUNT(*) FROM branches").fetchone()[0]
        session_count = con.execute("SELECT COUNT(*) FROM auto_sessions").fetchone()[0]
    return {
        "experiments": int(exp_count),
        "artifacts": int(art_count),
        "comparisons": int(cmp_count),
        "proposals": int(proposal_count),
        "branches": int(branch_count),
        "sessions": int(session_count),
    }
