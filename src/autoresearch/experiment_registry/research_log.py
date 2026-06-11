"""Structured per-cycle research-log registry operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry


def upsert_research_log_entry(
    path: Path,
    *,
    session_id: str,
    cycle: int,
    proposal_id: str | None,
    experiment_id: str | None,
    comparison_id: str | None,
    hypothesis: str,
    changes: str,
    outcome: str,
    metrics: dict[str, Any],
    interpretation: str | None = None,
    next_step: str | None = None,
    completed_at: str | None = None,
) -> None:
    """Create or update one cycle entry without duplicating it."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO research_log_entries (
                session_id, cycle, updated_at, proposal_id, experiment_id,
                comparison_id, hypothesis, changes, outcome, metrics_json,
                interpretation, next_step, completed_at
            )
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, cycle) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                proposal_id = COALESCE(excluded.proposal_id, proposal_id),
                experiment_id = COALESCE(excluded.experiment_id, experiment_id),
                comparison_id = COALESCE(excluded.comparison_id, comparison_id),
                hypothesis = excluded.hypothesis,
                changes = excluded.changes,
                outcome = excluded.outcome,
                metrics_json = excluded.metrics_json,
                interpretation = COALESCE(excluded.interpretation, interpretation),
                next_step = COALESCE(excluded.next_step, next_step),
                completed_at = COALESCE(excluded.completed_at, completed_at)
            """,
            (
                session_id,
                cycle,
                proposal_id,
                experiment_id,
                comparison_id,
                hypothesis,
                changes,
                outcome,
                dumps(metrics),
                interpretation,
                next_step,
                completed_at,
            ),
        )


def complete_research_log_entry(
    path: Path,
    *,
    session_id: str,
    cycle: int,
    interpretation: str,
    next_step: str,
    completed_at: str,
    outcome: str | None = None,
) -> None:
    """Attach the agent-authored reflection and mark one entry complete."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            UPDATE research_log_entries
            SET interpretation = ?,
                next_step = ?,
                outcome = COALESCE(?, outcome),
                completed_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ? AND cycle = ?
            """,
            (interpretation, next_step, outcome, completed_at, session_id, cycle),
        )
        if con.execute("SELECT changes()").fetchone()[0] == 0:
            raise ValueError(f"Research-log entry not found for session {session_id!r}, cycle {cycle}.")


def list_research_log_entries(path: Path, session_id: str | None = None) -> list[dict[str, Any]]:
    """Return cycle entries in chronological order."""

    if not path.exists():
        return []
    init_registry(path)
    query = "SELECT * FROM research_log_entries"
    params: tuple[Any, ...] = ()
    if session_id is not None:
        query += " WHERE session_id = ?"
        params = (session_id,)
    query += " ORDER BY created_at ASC, session_id ASC, cycle ASC"
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(query, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
        result.append(item)
    return result


def get_research_log_entry(
    path: Path,
    *,
    session_id: str,
    cycle: int,
) -> dict[str, Any] | None:
    """Return one cycle entry."""

    entries = list_research_log_entries(path, session_id)
    return next((entry for entry in entries if int(entry["cycle"]) == int(cycle)), None)


def find_research_log_entry_by_comparison(
    path: Path,
    comparison_id: str,
) -> dict[str, Any] | None:
    """Return the cycle entry linked to a comparison."""

    if not path.exists():
        return None
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM research_log_entries WHERE comparison_id = ?",
            (comparison_id,),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
    return item


def latest_incomplete_research_log_entry(
    path: Path,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the newest entry still missing interpretation or next direction."""

    if not path.exists():
        return None
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        where_session = "AND session_id = ?" if session_id is not None else ""
        params: tuple[Any, ...] = (session_id,) if session_id is not None else ()
        row = con.execute(
            f"""
            SELECT *
            FROM research_log_entries
            WHERE (
                interpretation IS NULL OR TRIM(interpretation) = ''
                OR next_step IS NULL OR TRIM(next_step) = ''
            )
            {where_session}
            ORDER BY created_at DESC, cycle DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
    return item
