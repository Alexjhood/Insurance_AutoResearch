"""Autonomous research session registry operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry


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
                dumps(details or {}),
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
