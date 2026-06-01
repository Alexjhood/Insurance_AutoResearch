"""Per-run research-line operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry


def upsert_research_line(
    path: Path,
    *,
    line_id: str,
    label: str,
    status: str = "active",
    root_node_id: str | None = None,
    parent_line_id: str | None = None,
    hypothesis: str | None = None,
    current_node_id: str | None = None,
    current_experiment_id: str | None = None,
    best_node_id: str | None = None,
    best_experiment_id: str | None = None,
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Insert or update a named local research line for the active run."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO research_lines (
                line_id, label, status, root_node_id, parent_line_id, hypothesis,
                current_node_id, current_experiment_id, best_node_id, best_experiment_id,
                notes, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(line_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                label = COALESCE(excluded.label, label),
                status = excluded.status,
                root_node_id = COALESCE(excluded.root_node_id, root_node_id),
                parent_line_id = COALESCE(excluded.parent_line_id, parent_line_id),
                hypothesis = COALESCE(excluded.hypothesis, hypothesis),
                current_node_id = COALESCE(excluded.current_node_id, current_node_id),
                current_experiment_id = COALESCE(excluded.current_experiment_id, current_experiment_id),
                best_node_id = COALESCE(excluded.best_node_id, best_node_id),
                best_experiment_id = COALESCE(excluded.best_experiment_id, best_experiment_id),
                notes = COALESCE(excluded.notes, notes),
                metadata_json = COALESCE(excluded.metadata_json, metadata_json)
            """,
            (
                line_id,
                label,
                status,
                root_node_id,
                parent_line_id,
                hypothesis,
                current_node_id,
                current_experiment_id,
                best_node_id,
                best_experiment_id,
                notes,
                dumps(metadata) if metadata else None,
            ),
        )


def set_research_line_champion(
    path: Path,
    *,
    line_id: str,
    experiment_id: str,
    node_id: str | None,
    action: str,
    reason: str,
    comparison_id: str | None = None,
    proposal_id: str | None = None,
) -> None:
    """Promote an experiment as the local incumbent for one research line."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT * FROM research_lines WHERE line_id = ?",
            (line_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Research line {line_id!r} not found")
        previous = row["current_experiment_id"] or row["best_experiment_id"]
        con.execute(
            """
            UPDATE research_lines
            SET updated_at = CURRENT_TIMESTAMP,
                status = 'active',
                current_node_id = COALESCE(?, current_node_id),
                current_experiment_id = ?,
                best_node_id = COALESCE(?, best_node_id),
                best_experiment_id = ?,
                notes = ?
            WHERE line_id = ?
            """,
            (node_id, experiment_id, node_id, experiment_id, reason, line_id),
        )
        con.execute(
            """
            INSERT INTO research_line_history (
                line_id, previous_experiment_id, new_experiment_id, node_id,
                action, reason, comparison_id, proposal_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (line_id, previous, experiment_id, node_id, action, reason, comparison_id, proposal_id),
        )


def record_research_line_history(
    path: Path,
    *,
    line_id: str,
    action: str,
    new_experiment_id: str,
    previous_experiment_id: str | None = None,
    node_id: str | None = None,
    reason: str | None = None,
    comparison_id: str | None = None,
    proposal_id: str | None = None,
) -> None:
    """Append an audit event for a local research line."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO research_line_history (
                line_id, previous_experiment_id, new_experiment_id, node_id,
                action, reason, comparison_id, proposal_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (line_id, previous_experiment_id, new_experiment_id, node_id, action, reason, comparison_id, proposal_id),
        )


def get_research_line(path: Path, line_id: str) -> dict[str, Any] | None:
    """Return one research line, if present."""

    if not path.exists():
        return None
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM research_lines WHERE line_id = ?", (line_id,)).fetchone()
    return _decode_line(dict(row)) if row else None


def list_research_lines(path: Path, *, status: str | None = None) -> list[dict[str, Any]]:
    """Return research lines newest first."""

    if not path.exists():
        return []
    init_registry(path)
    params: tuple[Any, ...] = ()
    sql = "SELECT * FROM research_lines"
    if status is not None:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY created_at DESC, line_id DESC"
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    return [_decode_line(dict(row)) for row in rows]


def list_research_line_history(path: Path, *, line_id: str | None = None) -> list[dict[str, Any]]:
    """Return research-line local promotion history newest first."""

    if not path.exists():
        return []
    init_registry(path)
    params: tuple[Any, ...] = ()
    sql = "SELECT * FROM research_line_history"
    if line_id is not None:
        sql += " WHERE line_id = ?"
        params = (line_id,)
    sql += " ORDER BY created_at DESC, history_id DESC"
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def _decode_line(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.pop("metadata_json", None)
    row["metadata"] = json.loads(raw) if raw else {}
    return row
