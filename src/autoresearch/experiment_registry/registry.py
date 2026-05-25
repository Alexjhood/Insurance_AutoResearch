"""Minimal SQLite registry for reproducible experiment bookkeeping."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

from autoresearch.utils.io import read_json


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

    # Write integrity manifest the first time (or refresh on explicit call)
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


def record_experiment(
    path: Path,
    *,
    experiment_id: str,
    experiment_name: str,
    model_family: str,
    target_strategy: str,
    preprocessing_summary: dict[str, Any],
    claim_cap_threshold: float | None,
    status: str,
    parent_experiment_id: str | None,
    config_snapshot_path: Path,
    metrics_path: Path,
    artifacts: dict[str, Path],
    code_version: str | None = None,
    notes: str | None = None,
) -> None:
    """Insert or replace a completed experiment and its artifact records."""

    import json

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id,
                updated_at,
                experiment_name,
                status,
                parent_experiment_id,
                model_family,
                target_strategy,
                preprocessing_summary,
                claim_cap_threshold,
                config_snapshot_path,
                code_version,
                metrics_path,
                notes
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                experiment_name,
                status,
                parent_experiment_id,
                model_family,
                target_strategy,
                json.dumps(preprocessing_summary, sort_keys=True),
                claim_cap_threshold,
                str(config_snapshot_path),
                code_version,
                str(metrics_path),
                notes,
            ),
        )
        con.execute("DELETE FROM artifacts WHERE experiment_id = ?", (experiment_id,))
        con.executemany(
            """
            INSERT INTO artifacts (experiment_id, artifact_type, path)
            VALUES (?, ?, ?)
            """,
            [(experiment_id, artifact_type, str(artifact_path)) for artifact_type, artifact_path in artifacts.items()],
        )


def record_experiment_artifacts(path: Path, experiment_id: str, artifacts: dict[str, Path]) -> None:
    """Add or replace selected artifact records for an existing experiment."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        for artifact_type, artifact_path in artifacts.items():
            con.execute(
                "DELETE FROM artifacts WHERE experiment_id = ? AND artifact_type = ?",
                (experiment_id, artifact_type),
            )
            con.execute(
                """
                INSERT INTO artifacts (experiment_id, artifact_type, path)
                VALUES (?, ?, ?)
                """,
                (experiment_id, artifact_type, str(artifact_path)),
            )


def get_experiment(path: Path, experiment_id: str) -> dict[str, Any]:
    """Fetch one experiment row by id."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT *
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Unknown experiment id: {experiment_id}")
    return dict(row)


def list_experiments(path: Path) -> list[dict[str, Any]]:
    """Return registry rows with selected metrics hydrated from disk."""

    if not path.exists():
        return []
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                experiment_id,
                created_at,
                updated_at,
                experiment_name,
                status,
                parent_experiment_id,
                model_family,
                target_strategy,
                preprocessing_summary,
                claim_cap_threshold,
                config_snapshot_path,
                metrics_path,
                notes
            FROM experiments
            ORDER BY created_at DESC, experiment_id DESC
            """
        ).fetchall()

    experiments: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metrics_path = Path(item["metrics_path"]) if item.get("metrics_path") else None
        if metrics_path and metrics_path.exists():
            metrics = read_json(metrics_path)
            item["mean_score"] = metrics.get("aggregate", {}).get("mean_score")
            item["std_score"] = metrics.get("aggregate", {}).get("std_score")
            item["primary_metric"] = metrics.get("primary_metric")
            item["ordinary_eval_splits"] = metrics.get("ordinary_eval_splits")
        experiments.append(item)
    return experiments


def list_artifacts(path: Path, experiment_id: str) -> list[dict[str, Any]]:
    """Return artifacts for one experiment."""

    if not path.exists():
        return []
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT artifact_type, path, created_at
            FROM artifacts
            WHERE experiment_id = ?
            ORDER BY artifact_type
            """,
            (experiment_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def record_comparison(
    path: Path,
    *,
    comparison_id: str,
    champion_id: str,
    challenger_id: str,
    paired_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    promotion_decision: str,
    promotion_rationale: str,
    artifacts: dict[str, Path],
) -> None:
    """Insert or replace a volatility-aware comparison record."""

    import json

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO comparisons (
                comparison_id,
                champion_id,
                challenger_id,
                paired_summary,
                bootstrap_summary,
                promotion_decision,
                promotion_rationale,
                comparison_summary_path,
                paired_scores_path,
                bootstrap_summary_path,
                promotion_decision_path,
                report_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_id,
                champion_id,
                challenger_id,
                json.dumps(paired_summary, sort_keys=True),
                json.dumps(bootstrap_summary, sort_keys=True),
                promotion_decision,
                promotion_rationale,
                str(artifacts.get("comparison_summary", "")),
                str(artifacts.get("paired_resample_scores", "")),
                str(artifacts.get("bootstrap_summary", "")),
                str(artifacts.get("promotion_decision", "")),
                str(artifacts.get("promotion_report", "")),
            ),
        )


def list_comparisons(path: Path) -> list[dict[str, Any]]:
    """Return comparison records with JSON summaries decoded."""

    import json

    if not path.exists():
        return []
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM comparisons
            ORDER BY created_at DESC, comparison_id DESC
            """
        ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["paired_summary"] = json.loads(item["paired_summary"])
        item["bootstrap_summary"] = json.loads(item["bootstrap_summary"])
        item["mean_lift"] = item["paired_summary"].get("mean_lift")
        item["challenger_win_rate"] = item["paired_summary"].get("challenger_win_rate")
        item["bootstrap_interval_lower"] = item["bootstrap_summary"].get("interval_lower")
        item["bootstrap_interval_upper"] = item["bootstrap_summary"].get("interval_upper")
        item["probability_challenger_outperforms"] = item["bootstrap_summary"].get(
            "probability_challenger_outperforms"
        )
        results.append(item)
    return results


def set_official_champion(
    path: Path,
    *,
    champion_id: str,
    branch_id: str,
    reason: str,
    action: str,
    comparison_id: str | None = None,
    proposal_id: str | None = None,
) -> None:
    """Set official champion and append a history record."""

    init_registry(path)
    previous = get_official_champion(path)
    previous_id = previous["champion_id"] if previous else None
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO champion_state (
                state_id, champion_id, branch_id, updated_at, reason, comparison_id, proposal_id
            )
            VALUES ('official', ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            """,
            (champion_id, branch_id, reason, comparison_id, proposal_id),
        )
        con.execute(
            """
            INSERT INTO champion_history (
                previous_champion_id,
                new_champion_id,
                branch_id,
                action,
                reason,
                comparison_id,
                proposal_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (previous_id, champion_id, branch_id, action, reason, comparison_id, proposal_id),
        )


def get_official_champion(path: Path) -> dict[str, Any] | None:
    """Return the explicit official champion state."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT *
            FROM champion_state
            WHERE state_id = 'official'
            """
        ).fetchone()
    return dict(row) if row else None


def list_champion_history(path: Path) -> list[dict[str, Any]]:
    """Return official champion history."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM champion_history
            ORDER BY history_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_branch(
    path: Path,
    *,
    branch_id: str,
    parent_branch_id: str | None,
    root_experiment_id: str | None,
    current_experiment_id: str | None,
    status: str,
    description: str | None = None,
) -> None:
    """Create or update a branch lineage record."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO branches (
                branch_id, parent_branch_id, root_experiment_id, current_experiment_id, status, description
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(branch_id) DO UPDATE SET
                parent_branch_id = excluded.parent_branch_id,
                root_experiment_id = COALESCE(branches.root_experiment_id, excluded.root_experiment_id),
                current_experiment_id = excluded.current_experiment_id,
                status = excluded.status,
                description = COALESCE(excluded.description, branches.description)
            """,
            (branch_id, parent_branch_id, root_experiment_id, current_experiment_id, status, description),
        )


def list_branches(path: Path) -> list[dict[str, Any]]:
    """Return branch lineage records."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM branches
            ORDER BY created_at DESC, branch_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def record_proposal(
    path: Path,
    *,
    proposal_id: str,
    status: str,
    parent_experiment_id: str | None,
    parent_branch_id: str | None,
    branch_id: str | None,
    experiment_name: str | None,
    rationale: str | None,
    change_summary: str | None,
    expected_benefit: str | None,
    key_risk: str | None,
    config: dict[str, Any] | None,
    validation_errors: list[str] | None,
    llm_provider: str | None,
    llm_model: str | None,
    prompt_path: Path | None,
    response_path: Path | None,
    proposal_path: Path | None,
    notes: str | None = None,
) -> None:
    """Insert or replace a proposal queue record."""

    import json

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO proposals (
                proposal_id,
                updated_at,
                status,
                parent_experiment_id,
                parent_branch_id,
                branch_id,
                experiment_name,
                rationale,
                change_summary,
                expected_benefit,
                key_risk,
                config_json,
                validation_errors_json,
                llm_provider,
                llm_model,
                prompt_path,
                response_path,
                proposal_path,
                notes
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal_id,
                status,
                parent_experiment_id,
                parent_branch_id,
                branch_id,
                experiment_name,
                rationale,
                change_summary,
                expected_benefit,
                key_risk,
                json.dumps(config or {}, sort_keys=True),
                json.dumps(validation_errors or [], sort_keys=True),
                llm_provider,
                llm_model,
                str(prompt_path) if prompt_path else None,
                str(response_path) if response_path else None,
                str(proposal_path) if proposal_path else None,
                notes,
            ),
        )


def update_proposal_status(
    path: Path,
    proposal_id: str,
    status: str,
    *,
    experiment_id: str | None = None,
    comparison_id: str | None = None,
    notes: str | None = None,
) -> None:
    """Update proposal lifecycle state and optional execution links."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            UPDATE proposals
            SET
                status = ?,
                updated_at = CURRENT_TIMESTAMP,
                experiment_id = COALESCE(?, experiment_id),
                comparison_id = COALESCE(?, comparison_id),
                notes = COALESCE(?, notes)
            WHERE proposal_id = ?
            """,
            (status, experiment_id, comparison_id, notes, proposal_id),
        )


def get_proposal(path: Path, proposal_id: str) -> dict[str, Any]:
    """Fetch one proposal with JSON fields decoded."""

    rows = [row for row in list_proposals(path) if row["proposal_id"] == proposal_id]
    if not rows:
        raise ValueError(f"Unknown proposal id: {proposal_id}")
    return rows[0]


def next_queued_proposal(path: Path) -> dict[str, Any] | None:
    """Return the oldest proposal ready to run."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT *
            FROM proposals
            WHERE status IN ('validated', 'proposed')
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
    return _decode_proposal(dict(row)) if row else None


def list_proposals(path: Path) -> list[dict[str, Any]]:
    """Return proposal queue records with JSON fields decoded."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM proposals
            ORDER BY created_at DESC, proposal_id DESC
            """
        ).fetchall()
    return [_decode_proposal(dict(row)) for row in rows]


def _decode_proposal(row: dict[str, Any]) -> dict[str, Any]:
    import json

    row["config"] = json.loads(row.pop("config_json") or "{}")
    row["validation_errors"] = json.loads(row.pop("validation_errors_json") or "[]")
    return row


def upsert_session(
    path: Path,
    *,
    session_id: str,
    name: str,
    state: str,
    current_cycle: int,
    max_cycles: int | None,
    stop_requested: bool,
    state_path: Path,
    summary_path: Path,
    notes: str | None = None,
) -> None:
    """Create or update an autonomous research session row."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO auto_sessions (
                session_id, name, updated_at, state, current_cycle, max_cycles,
                stop_requested, state_path, summary_path, notes
            )
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                state = excluded.state,
                current_cycle = excluded.current_cycle,
                max_cycles = excluded.max_cycles,
                stop_requested = excluded.stop_requested,
                state_path = excluded.state_path,
                summary_path = excluded.summary_path,
                notes = COALESCE(excluded.notes, auto_sessions.notes)
            """,
            (
                session_id,
                name,
                state,
                current_cycle,
                max_cycles,
                1 if stop_requested else 0,
                str(state_path),
                str(summary_path),
                notes,
            ),
        )


def record_session_event(
    path: Path,
    *,
    session_id: str,
    event_type: str,
    state: str | None = None,
    proposal_id: str | None = None,
    experiment_id: str | None = None,
    comparison_id: str | None = None,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one session event."""

    import json

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO session_events (
                session_id, event_type, state, proposal_id, experiment_id,
                comparison_id, message, details_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event_type,
                state,
                proposal_id,
                experiment_id,
                comparison_id,
                message,
                json.dumps(details or {}, sort_keys=True),
            ),
        )


def list_sessions(path: Path) -> list[dict[str, Any]]:
    """Return autonomous research sessions."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM auto_sessions
            ORDER BY updated_at DESC, session_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_session(path: Path, session_id: str) -> dict[str, Any] | None:
    """Return one autonomous research session row."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM auto_sessions WHERE session_id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_session_events(path: Path, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent session events."""

    import json

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT *
            FROM session_events
            WHERE session_id = ?
            ORDER BY event_id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    events = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json") or "{}")
        events.append(item)
    return events
