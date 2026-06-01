"""Per-run research idea tree operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry


def upsert_research_node(
    path: Path,
    *,
    node_id: str,
    line_id: str | None = None,
    proposal_id: str | None = None,
    parent_node_id: str | None = None,
    parent_experiment_id: str | None = None,
    experiment_id: str | None = None,
    comparison_id: str | None = None,
    branch_id: str | None = None,
    status: str = "proposed",
    outcome_type: str | None = None,
    hypothesis: str | None = None,
    change_summary: str | None = None,
    expected_benefit: str | None = None,
    key_risk: str | None = None,
    tags: list[str] | None = None,
    tree_metadata: dict[str, Any] | None = None,
    screening: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    guidance: str | None = None,
) -> None:
    """Insert or update one node in the active run's research tree."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT INTO research_nodes (
                node_id, line_id, proposal_id, parent_node_id, parent_experiment_id,
                experiment_id, comparison_id, branch_id, status, outcome_type,
                hypothesis, change_summary, expected_benefit, key_risk,
                tags_json, tree_json, screening_json, metrics_json, guidance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                line_id = COALESCE(excluded.line_id, line_id),
                proposal_id = COALESCE(excluded.proposal_id, proposal_id),
                parent_node_id = COALESCE(excluded.parent_node_id, parent_node_id),
                parent_experiment_id = COALESCE(excluded.parent_experiment_id, parent_experiment_id),
                experiment_id = COALESCE(excluded.experiment_id, experiment_id),
                comparison_id = COALESCE(excluded.comparison_id, comparison_id),
                branch_id = COALESCE(excluded.branch_id, branch_id),
                status = excluded.status,
                outcome_type = COALESCE(excluded.outcome_type, outcome_type),
                hypothesis = COALESCE(excluded.hypothesis, hypothesis),
                change_summary = COALESCE(excluded.change_summary, change_summary),
                expected_benefit = COALESCE(excluded.expected_benefit, expected_benefit),
                key_risk = COALESCE(excluded.key_risk, key_risk),
                tags_json = COALESCE(excluded.tags_json, tags_json),
                tree_json = COALESCE(excluded.tree_json, tree_json),
                screening_json = COALESCE(excluded.screening_json, screening_json),
                metrics_json = COALESCE(excluded.metrics_json, metrics_json),
                guidance = COALESCE(excluded.guidance, guidance)
            """,
            (
                node_id,
                line_id,
                proposal_id,
                parent_node_id,
                parent_experiment_id,
                experiment_id,
                comparison_id,
                branch_id,
                status,
                outcome_type,
                hypothesis,
                change_summary,
                expected_benefit,
                key_risk,
                dumps(tags) if tags is not None else None,
                dumps(tree_metadata) if tree_metadata else None,
                dumps(screening) if screening is not None else None,
                dumps(metrics) if metrics is not None else None,
                guidance,
            ),
        )


def list_research_nodes(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Return research-tree nodes for the active run, newest first."""

    if not path.exists():
        return []
    init_registry(path)
    sql = """
        SELECT *
        FROM research_nodes
        ORDER BY created_at DESC, node_id DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params: tuple[Any, ...] = (int(limit),)
    else:
        params = ()
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    return [_decode_node(dict(row)) for row in rows]


def find_research_node_by_experiment(path: Path, experiment_id: str) -> dict[str, Any] | None:
    """Return the node that produced an experiment, if known."""

    if not path.exists():
        return None
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT *
            FROM research_nodes
            WHERE experiment_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (experiment_id,),
        ).fetchone()
    return _decode_node(dict(row)) if row else None


def _decode_node(row: dict[str, Any]) -> dict[str, Any]:
    for raw_key, out_key, default in (
        ("tags_json", "tags", []),
        ("tree_json", "tree_metadata", {}),
        ("screening_json", "screening", None),
        ("metrics_json", "metrics", None),
    ):
        raw = row.pop(raw_key, None)
        if raw is None:
            row[out_key] = default
        else:
            row[out_key] = json.loads(raw)
    return row
