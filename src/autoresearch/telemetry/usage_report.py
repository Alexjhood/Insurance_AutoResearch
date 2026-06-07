"""Compact incremental and cumulative LLM usage by experiment."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.telemetry.store import connect, telemetry_path


REPORT_NAME = "LLM_USAGE.md"


def write_usage_report(run_dir: Path) -> Path | None:
    run_dir = Path(run_dir)
    if not telemetry_path(run_dir).exists():
        return None
    con = connect(run_dir)
    try:
        _backfill_checkpoints(con, run_dir / "registry.sqlite")
        checkpoints = [
            dict(row)
            for row in con.execute(
                """
                SELECT experiment_id, experiment_name, status, completed_at
                FROM experiment_usage_checkpoints
                ORDER BY completed_at, experiment_id
                """
            ).fetchall()
        ]
        rows = [_usage_at(con, checkpoint) for checkpoint in checkpoints]
    finally:
        con.close()
    if not rows:
        return None

    lines = [
        "# LLM Usage by Experiment",
        "",
        "Automatically refreshed from desktop telemetry. Incremental values are the "
        "change since the previous experiment checkpoint; cumulative values include "
        "records observed at or before the checkpoint time.",
        "",
        "| Experiment | Status | Completed | Incremental tokens | Cumulative tokens | "
        "Cache hit | Model calls (inc/cum) | Tool calls (inc/cum) | Tool failures (inc/cum) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    previous = _zero_usage()
    for row in rows:
        incremental = {
            key: int(row[key]) - int(previous[key])
            for key in ("total_tokens", "model_calls", "tool_calls", "tool_failures")
        }
        cache_ratio = (
            float(row["cached_input_tokens"]) / float(row["input_tokens"])
            if row["input_tokens"]
            else None
        )
        lines.append(
            "| {name} | {status} | {completed} | {delta_tokens} | {total_tokens} | "
            "{cache} | {delta_models}/{models} | {delta_tools}/{tools} | "
            "{delta_failures}/{failures} |".format(
                name=_cell(row["experiment_name"]),
                status=_cell(row["status"]),
                completed=_cell(row["completed_at"]),
                delta_tokens=_count(incremental["total_tokens"]),
                total_tokens=_count(row["total_tokens"]),
                cache=f"{cache_ratio:.1%}" if cache_ratio is not None else "-",
                delta_models=_count(incremental["model_calls"]),
                models=_count(row["model_calls"]),
                delta_tools=_count(incremental["tool_calls"]),
                tools=_count(row["tool_calls"]),
                delta_failures=_count(incremental["tool_failures"]),
                failures=_count(row["tool_failures"]),
            )
        )
        previous = row

    latest = rows[-1]
    lines.extend(
        [
            "",
            "Latest cumulative detail: "
            f"{_count(latest['input_tokens'])} input "
            f"({_count(latest['cached_input_tokens'])} cached), "
            f"{_count(latest['output_tokens'])} output, "
            f"{_count(latest['reasoning_tokens'])} reasoning tokens.",
            "",
        ]
    )
    path = run_dir / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _backfill_checkpoints(con, registry_path: Path) -> None:
    if not registry_path.exists():
        return
    existing = {
        str(row["experiment_id"])
        for row in con.execute(
            "SELECT experiment_id FROM experiment_usage_checkpoints"
        ).fetchall()
    }
    try:
        registry = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
        registry.row_factory = sqlite3.Row
        rows = registry.execute(
            """
            SELECT experiment_id, experiment_name, status, updated_at
            FROM experiments ORDER BY updated_at, experiment_id
            """
        ).fetchall()
        registry.close()
    except sqlite3.Error:
        return
    additions = [
        (
            str(row["experiment_id"]),
            str(row["experiment_name"]),
            str(row["status"]),
            str(row["updated_at"]).replace(" ", "T") + "+00:00",
        )
        for row in rows
        if str(row["experiment_id"]) not in existing and row["updated_at"]
    ]
    if additions:
        con.executemany(
            """
            INSERT OR IGNORE INTO experiment_usage_checkpoints (
                experiment_id, experiment_name, status, completed_at
            ) VALUES (?,?,?,?)
            """,
            additions,
        )
        con.commit()


def _usage_at(con, checkpoint: dict[str, Any]) -> dict[str, Any]:
    at = checkpoint["completed_at"]
    tokens = con.execute(
        """
        SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens,
               COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens,
               COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
               COALESCE(SUM(total_tokens), 0) AS total_tokens,
               COUNT(*) AS model_calls
        FROM llm_model_calls
        WHERE julianday(occurred_at) <= julianday(?)
        """,
        (at,),
    ).fetchone()
    tools = con.execute(
        """
        SELECT COUNT(*) AS tool_calls,
               COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS tool_failures
        FROM llm_tool_calls
        WHERE julianday(started_at) <= julianday(?)
        """,
        (at,),
    ).fetchone()
    return {**checkpoint, **dict(tokens), **dict(tools)}


def _zero_usage() -> dict[str, int]:
    return {
        "total_tokens": 0,
        "model_calls": 0,
        "tool_calls": 0,
        "tool_failures": 0,
    }


def _count(value: Any) -> str:
    return f"{int(value or 0):,}"


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|")
