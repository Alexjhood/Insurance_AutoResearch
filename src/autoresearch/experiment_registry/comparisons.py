"""Comparison registry operations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry


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
    guardrail_status: dict[str, Any] | None = None,
) -> None:
    """Insert or replace a comparison record (initial state: pending_llm)."""

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
                report_path,
                guardrail_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_id,
                champion_id,
                challenger_id,
                dumps(paired_summary),
                dumps(bootstrap_summary),
                promotion_decision,
                promotion_rationale,
                str(artifacts.get("comparison_summary", "")),
                str(artifacts.get("paired_resample_scores", "")),
                str(artifacts.get("bootstrap_summary", "")),
                str(artifacts.get("promotion_decision", "")),
                str(artifacts.get("html_report") or artifacts.get("promotion_report", "")),
                dumps(guardrail_status) if guardrail_status is not None else None,
            ),
        )


def update_comparison_decision(
    path: Path,
    comparison_id: str,
    *,
    decision: str,
    rationale: str,
    decided_by: str = "llm",
    decided_at: str,
    guardrail_status: dict[str, Any] | None = None,
) -> None:
    """Record the LLM's final promote/reject verdict on a pending comparison."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            UPDATE comparisons
            SET decision = ?,
                decision_rationale = ?,
                decided_by = ?,
                decided_at = ?,
                guardrail_status = COALESCE(?, guardrail_status)
            WHERE comparison_id = ?
            """,
            (
                decision,
                rationale,
                decided_by,
                decided_at,
                dumps(guardrail_status) if guardrail_status is not None else None,
                comparison_id,
            ),
        )
        if con.execute("SELECT changes()").fetchone()[0] == 0:
            raise ValueError(f"Comparison {comparison_id!r} not found in registry")


def list_comparisons(path: Path) -> list[dict[str, Any]]:
    """Return comparison records with JSON summaries decoded."""

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
