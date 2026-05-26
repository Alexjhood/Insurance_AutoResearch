"""Proposal queue registry operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry


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
                dumps(config or {}),
                dumps(validation_errors or []),
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
            WHERE status IN ('validated', 'proposed', 'needs_repair')
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
    row["config"] = json.loads(row.pop("config_json") or "{}")
    row["validation_errors"] = json.loads(row.pop("validation_errors_json") or "[]")
    return row
