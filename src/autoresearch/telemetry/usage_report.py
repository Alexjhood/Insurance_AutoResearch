"""Incremental and cumulative LLM usage at experiment and user breakpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.telemetry.store import connect, telemetry_path


REPORT_NAME = "LLM_USAGE.md"
_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "model_calls",
    "tool_calls",
    "tool_failures",
)


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
                SELECT checkpoint_key, checkpoint_type, label, status, occurred_at
                FROM llm_usage_checkpoints
                ORDER BY julianday(occurred_at), checkpoint_key
                """
            ).fetchall()
        ]
        rows: list[dict[str, Any]] = []
        previous_at: str | None = None
        for checkpoint in checkpoints:
            row = _usage_at(con, checkpoint)
            row.update(_model_context(con, after=previous_at, at=checkpoint["occurred_at"]))
            rows.append(row)
            previous_at = str(checkpoint["occurred_at"])

        ledger = _ledger_usage(con)
        latest = rows[-1] if rows else _zero_usage()
        if _has_unassigned_usage(ledger, latest):
            observed_at = _latest_observed_at(con)
            live = {
                "checkpoint_key": "live:unassigned",
                "checkpoint_type": "overhead",
                "label": "Between-cycle / wrap-up overhead",
                "status": "current import",
                "occurred_at": observed_at or "unknown",
                **ledger,
            }
            live.update(_model_context(con, after=previous_at, at=observed_at))
            rows.append(live)
    finally:
        con.close()
    if not rows:
        return None

    lines = [
        "# LLM Usage by Step",
        "",
        "Automatically refreshed from desktop telemetry. Experiment rows stop at the "
        "experiment checkpoint; user-breakpoint rows capture work through each settled "
        "agent turn. Any remaining imported usage is shown explicitly as overhead.",
        "",
        "| Step | Type | Status | Completed | Model | Effort | Input | Cached | Uncached | "
        "Output | Reasoning | Total (inc/cum) | Cache hit | Model calls (inc/cum) | "
        "Tool calls (inc/cum) | Tool failures (inc/cum) |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    previous = _zero_usage()
    for row in rows:
        incremental = {
            key: int(row[key]) - int(previous[key])
            for key in _USAGE_FIELDS
        }
        cache_ratio = (
            float(incremental["cached_input_tokens"]) / float(incremental["input_tokens"])
            if incremental["input_tokens"]
            else None
        )
        lines.append(
            "| {label} | {kind} | {status} | {completed} | {models} | {efforts} | "
            "{input_tokens} | {cached_tokens} | {uncached_tokens} | {output_tokens} | "
            "{reasoning_tokens} | {delta_total}/{total_tokens} | {cache} | "
            "{delta_models}/{models_count} | {delta_tools}/{tools} | "
            "{delta_failures}/{failures} |".format(
                label=_cell(row["label"]),
                kind=_cell(row["checkpoint_type"]),
                status=_cell(row["status"]),
                completed=_cell(row["occurred_at"]),
                models=_cell(row.get("models_str") or "—"),
                efforts=_cell(row.get("efforts_str") or "—"),
                input_tokens=_count(incremental["input_tokens"]),
                cached_tokens=_count(incremental["cached_input_tokens"]),
                uncached_tokens=_count(incremental["uncached_input_tokens"]),
                output_tokens=_count(incremental["output_tokens"]),
                reasoning_tokens=_count(incremental["reasoning_tokens"]),
                delta_total=_count(incremental["total_tokens"]),
                total_tokens=_count(row["total_tokens"]),
                cache=f"{cache_ratio:.1%}" if cache_ratio is not None else "-",
                delta_models=_count(incremental["model_calls"]),
                models_count=_count(row["model_calls"]),
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
            "Current imported ledger: "
            f"{_count(latest['input_tokens'])} input "
            f"({_count(latest['cached_input_tokens'])} cached, "
            f"{_count(latest['uncached_input_tokens'])} uncached), "
            f"{_count(latest['output_tokens'])} output, "
            f"{_count(latest['reasoning_tokens'])} reasoning, "
            f"{_count(latest['total_tokens'])} total tokens.",
            "",
        ]
    )
    path = run_dir / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _backfill_checkpoints(con, registry_path: Path) -> None:
    con.execute(
        """
        INSERT OR IGNORE INTO llm_usage_checkpoints (
            checkpoint_key, checkpoint_type, label, status, occurred_at, experiment_id
        )
        SELECT 'experiment:' || experiment_id, 'experiment', experiment_name,
               status, completed_at, experiment_id
        FROM experiment_usage_checkpoints
        """
    )
    if not registry_path.exists():
        con.commit()
        return
    try:
        registry = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
        registry.row_factory = sqlite3.Row
        registry_rows = registry.execute(
            """
            SELECT experiment_id, experiment_name, status, updated_at
            FROM experiments ORDER BY updated_at, experiment_id
            """
        ).fetchall()
        registry.close()
    except sqlite3.Error:
        con.commit()
        return
    additions = [
        (
            f"experiment:{row['experiment_id']}",
            "experiment",
            str(row["experiment_name"]),
            str(row["status"]),
            str(row["updated_at"]).replace(" ", "T") + "+00:00",
            str(row["experiment_id"]),
        )
        for row in registry_rows
        if row["updated_at"]
    ]
    if additions:
        con.executemany(
            """
            INSERT OR IGNORE INTO llm_usage_checkpoints (
                checkpoint_key, checkpoint_type, label, status, occurred_at,
                experiment_id
            ) VALUES (?,?,?,?,?,?)
            """,
            additions,
        )
    con.commit()


def _usage_at(con, checkpoint: dict[str, Any]) -> dict[str, Any]:
    at = checkpoint["occurred_at"]
    return {
        **checkpoint,
        **_token_usage(con, at=at),
        **_tool_usage(con, at=at),
    }


def _ledger_usage(con) -> dict[str, int]:
    return {
        **_token_usage(con, at=None),
        **_tool_usage(con, at=None),
    }


def _token_usage(con, *, at: str | None) -> dict[str, int]:
    where = "WHERE julianday(occurred_at) <= julianday(?)" if at else ""
    params = (at,) if at else ()
    row = con.execute(
        f"""
        SELECT COALESCE(SUM(input_tokens), 0) AS input_tokens,
               COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
               COALESCE(SUM(uncached_input_tokens), 0) AS uncached_input_tokens,
               COALESCE(SUM(output_tokens), 0) AS output_tokens,
               COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
               COALESCE(SUM(total_tokens), 0) AS total_tokens,
               COUNT(*) AS model_calls
        FROM llm_model_calls
        {where}
        """,
        params,
    ).fetchone()
    return {key: int(row[key] or 0) for key in (
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "model_calls",
    )}


def _tool_usage(con, *, at: str | None) -> dict[str, int]:
    where = "WHERE julianday(started_at) <= julianday(?)" if at else ""
    params = (at,) if at else ()
    row = con.execute(
        f"""
        SELECT COUNT(*) AS tool_calls,
               COALESCE(SUM(CASE WHEN success=0 THEN 1 ELSE 0 END), 0) AS tool_failures
        FROM llm_tool_calls
        {where}
        """,
        params,
    ).fetchone()
    return {
        "tool_calls": int(row["tool_calls"] or 0),
        "tool_failures": int(row["tool_failures"] or 0),
    }


def _model_context(
    con,
    *,
    after: str | None,
    at: str | None,
) -> dict[str, str | None]:
    conditions = []
    params: list[str] = []
    if after:
        conditions.append("julianday(c.occurred_at) > julianday(?)")
        params.append(after)
    if at and at != "unknown":
        conditions.append("julianday(c.occurred_at) <= julianday(?)")
        params.append(at)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = con.execute(
        f"""
        SELECT DISTINCT
               COALESCE(c.model, t.model, s.model) AS model,
               COALESCE(t.effort, s.effort) AS effort
        FROM llm_model_calls c
        LEFT JOIN llm_turns t ON t.turn_key=c.turn_key
        LEFT JOIN llm_sessions s ON s.session_key=c.session_key
        {where}
        """,
        params,
    ).fetchall()
    models = sorted({str(row["model"]) for row in rows if row["model"]})
    efforts = sorted({str(row["effort"]) for row in rows if row["effort"]})
    return {
        "models_str": ", ".join(models) or None,
        "efforts_str": ", ".join(efforts) or None,
    }


def _latest_observed_at(con) -> str | None:
    row = con.execute(
        """
        SELECT observed_at
        FROM (
            SELECT occurred_at AS observed_at FROM llm_model_calls
            UNION ALL
            SELECT COALESCE(completed_at, started_at) FROM llm_tool_calls
            UNION ALL
            SELECT COALESCE(completed_at, started_at) FROM llm_turns
        )
        WHERE observed_at IS NOT NULL
        ORDER BY julianday(observed_at) DESC
        LIMIT 1
        """
    ).fetchone()
    return str(row["observed_at"]) if row and row["observed_at"] else None


def _has_unassigned_usage(current: dict[str, int], latest: dict[str, Any]) -> bool:
    return any(int(current[key]) != int(latest.get(key) or 0) for key in _USAGE_FIELDS)


def _zero_usage() -> dict[str, int]:
    return {key: 0 for key in _USAGE_FIELDS}


def _count(value: Any) -> str:
    return f"{int(value or 0):,}"


def _cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|")
