"""Champion state registry operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry.schema import init_registry


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
