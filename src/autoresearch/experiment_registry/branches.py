"""Branch lineage registry operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry.schema import init_registry


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
