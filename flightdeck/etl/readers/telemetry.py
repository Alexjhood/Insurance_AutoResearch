"""Read ``telemetry.sqlite`` (+ LLM_USAGE.md fallback) per DATA.md §2.6/§3.

Provides three things:
- :func:`read_delegation_telemetry` — full event lists for ``telemetry_<dNN>.json``.
- :func:`read_telemetry_cost` — per-delegation aggregates (counts, tool-mix, and
  a token fallback for crashed delegations missing ``tool_usage``, quirk 7).
- :func:`read_experiment_usage` — per-experiment usage checkpoints, discovering
  columns with PRAGMA and degrading to the LLM_USAGE.md table when the token
  columns are absent (as in the reference fixture).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from ..schema import (
    DelegationTelemetry,
    ExperimentUsage,
    ModelCallEvent,
    TelemetryCheckpoint,
    TokenTotals,
    ToolCallEvent,
    ToolMixEntry,
    WorkflowEvent,
)
from ..util import (
    Warnings,
    cache_hit_rate,
    has_table,
    ro_connect,
    safe_query,
    table_columns,
    to_float,
    to_int,
)


def _g(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _is_failure(row: Any) -> bool:
    if _g(row, "success") == 0:
        return True
    if _g(row, "error_type"):
        return True
    status = (_g(row, "status") or "").lower()
    return status in ("failed", "error", "errored", "timeout")


def read_delegation_telemetry(
    telemetry_path: Path, delegation_id: str, warnings: Warnings
) -> Optional[DelegationTelemetry]:
    with ro_connect(telemetry_path, warnings) as conn:
        if conn is None:
            return None
        model_calls: list[ModelCallEvent] = []
        if has_table(conn, "llm_model_calls"):
            for r in safe_query(
                conn,
                "SELECT * FROM llm_model_calls ORDER BY occurred_at",
                warnings=warnings, label="llm_model_calls",
            ):
                model_calls.append(
                    ModelCallEvent(
                        at=_g(r, "occurred_at") or "",
                        model=_g(r, "model") or "",
                        input=to_int(_g(r, "input_tokens")) or 0,
                        cached_input=to_int(_g(r, "cached_input_tokens")) or 0,
                        output=to_int(_g(r, "output_tokens")) or 0,
                        reasoning=to_int(_g(r, "reasoning_tokens")) or 0,
                        duration_hint_ms=None,
                        workflow_event_id=to_int(_g(r, "workflow_event_id")),
                    )
                )
        tool_calls: list[ToolCallEvent] = []
        if has_table(conn, "llm_tool_calls"):
            for r in safe_query(
                conn,
                "SELECT * FROM llm_tool_calls ORDER BY started_at",
                warnings=warnings, label="llm_tool_calls",
            ):
                tool_calls.append(
                    ToolCallEvent(
                        name=_g(r, "name") or "",
                        detail=_g(r, "detail"),
                        started_at=_g(r, "started_at"),
                        completed_at=_g(r, "completed_at"),
                        duration_ms=to_float(_g(r, "duration_ms")),
                        status=_g(r, "status"),
                        success=(None if _g(r, "success") is None else bool(_g(r, "success"))),
                        input_bytes=to_int(_g(r, "input_bytes")),
                        output_bytes=to_int(_g(r, "output_bytes")),
                        error_type=_g(r, "error_type"),
                        workflow_event_id=to_int(_g(r, "workflow_event_id")),
                    )
                )
        workflow_events: list[WorkflowEvent] = []
        if has_table(conn, "workflow_events"):
            for r in safe_query(
                conn,
                "SELECT * FROM workflow_events ORDER BY started_at",
                warnings=warnings, label="workflow_events",
            ):
                workflow_events.append(
                    WorkflowEvent(
                        id=to_int(_g(r, "id")) or 0,
                        command=_g(r, "command") or "",
                        started_at=_g(r, "started_at"),
                        completed_at=_g(r, "completed_at"),
                        duration_ms=to_float(_g(r, "duration_ms")),
                        status=_g(r, "status"),
                        error_type=_g(r, "error_type"),
                    )
                )
        checkpoints: list[TelemetryCheckpoint] = []
        if has_table(conn, "experiment_usage_checkpoints"):
            cols = table_columns(conn, "experiment_usage_checkpoints")
            cum_col = "total_cumulative" if "total_cumulative" in cols else None
            for r in safe_query(
                conn,
                "SELECT * FROM experiment_usage_checkpoints ORDER BY completed_at",
                warnings=warnings, label="experiment_usage_checkpoints",
            ):
                checkpoints.append(
                    TelemetryCheckpoint(
                        experiment_name=_g(r, "experiment_name") or "",
                        completed_at=_g(r, "completed_at"),
                        total_cumulative=(to_int(_g(r, cum_col)) or 0) if cum_col else 0,
                    )
                )
        return DelegationTelemetry(
            delegation_id=delegation_id,
            model_calls=model_calls,
            tool_calls=tool_calls,
            workflow_events=workflow_events,
            checkpoints=checkpoints,
        )


def read_telemetry_cost(telemetry_path: Path, warnings: Warnings) -> dict:
    """Aggregate counts, tool-mix and a token fallback from telemetry.sqlite."""
    result = {
        "model_calls": 0,
        "tool_calls": 0,
        "tool_failures": 0,
        "tokens": TokenTotals(),
        "tool_mix": [],  # list of (tool, calls, failures, duration_ms)
        "has_db": False,
    }
    with ro_connect(telemetry_path, warnings) as conn:
        if conn is None:
            return result
        result["has_db"] = True
        if has_table(conn, "llm_model_calls"):
            rows = safe_query(conn, "SELECT * FROM llm_model_calls", warnings=warnings)
            result["model_calls"] = len(rows)
            tok = TokenTotals()
            for r in rows:
                tok.input += to_int(_g(r, "input_tokens")) or 0
                tok.cached_input += to_int(_g(r, "cached_input_tokens")) or 0
                tok.output += to_int(_g(r, "output_tokens")) or 0
                tok.reasoning += to_int(_g(r, "reasoning_tokens")) or 0
            result["tokens"] = tok
        if has_table(conn, "llm_tool_calls"):
            rows = safe_query(conn, "SELECT * FROM llm_tool_calls", warnings=warnings)
            result["tool_calls"] = len(rows)
            failures = 0
            mix: dict[str, list[float]] = {}
            for r in rows:
                fail = _is_failure(r)
                failures += 1 if fail else 0
                name = _g(r, "name") or ""
                entry = mix.setdefault(name, [0, 0, 0.0])
                entry[0] += 1
                entry[1] += 1 if fail else 0
                entry[2] += to_float(_g(r, "duration_ms")) or 0.0
            result["tool_failures"] = failures
            result["tool_mix"] = [
                (name, int(v[0]), int(v[1]), float(v[2])) for name, v in mix.items()
            ]
    return result


# --------------------------------------------------------------------------- #
# Per-experiment usage
# --------------------------------------------------------------------------- #
_TOKEN_COL_MAP = {
    "input": ("input_tokens", "input"),
    "cached_input": ("cached_input_tokens", "cached_input", "cached"),
    "output": ("output_tokens", "output"),
    "reasoning": ("reasoning_tokens", "reasoning_output_tokens", "reasoning"),
}


def _first_present(cols: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def read_experiment_usage(
    telemetry_path: Path, llm_usage_md: Optional[Path], warnings: Warnings
) -> dict[str, ExperimentUsage]:
    """Map experiment_name → per-step usage.

    Primary: ``experiment_usage_checkpoints`` token columns (discovered with
    PRAGMA). Fallback: parse the LLM_USAGE.md table (stable documented format).
    """
    by_name: dict[str, ExperimentUsage] = {}
    with ro_connect(telemetry_path, warnings) as conn:
        if conn is not None and has_table(conn, "experiment_usage_checkpoints"):
            cols = table_columns(conn, "experiment_usage_checkpoints")
            token_cols = {
                key: _first_present(cols, cands) for key, cands in _TOKEN_COL_MAP.items()
            }
            if token_cols["input"] is not None:  # token data present in table
                for r in safe_query(
                    conn, "SELECT * FROM experiment_usage_checkpoints", warnings=warnings
                ):
                    name = _g(r, "experiment_name") or ""
                    tok = TokenTotals(
                        input=to_int(_g(r, token_cols["input"])) or 0,
                        cached_input=to_int(_g(r, token_cols["cached_input"])) or 0
                        if token_cols["cached_input"] else 0,
                        output=to_int(_g(r, token_cols["output"])) or 0
                        if token_cols["output"] else 0,
                        reasoning=to_int(_g(r, token_cols["reasoning"])) or 0
                        if token_cols["reasoning"] else 0,
                    )
                    cum_col = _first_present(cols, ("total_cumulative", "cumulative_total"))
                    by_name[name] = ExperimentUsage(
                        tokens=tok,
                        total_cumulative=(to_int(_g(r, cum_col)) or 0) if cum_col else 0,
                        model_calls=to_int(_g(r, _first_present(cols, ("model_calls",)))) or 0
                        if _first_present(cols, ("model_calls",)) else 0,
                        tool_calls=to_int(_g(r, _first_present(cols, ("tool_calls",)))) or 0
                        if _first_present(cols, ("tool_calls",)) else 0,
                        tool_failures=to_int(_g(r, _first_present(cols, ("tool_failures",)))) or 0
                        if _first_present(cols, ("tool_failures",)) else 0,
                        cache_hit_rate=cache_hit_rate(tok.cached_input, tok.input),
                    )
                if by_name:
                    return by_name
    # Fallback: parse LLM_USAGE.md.
    if llm_usage_md is not None and llm_usage_md.exists():
        parsed = _parse_llm_usage_md(llm_usage_md, warnings)
        by_name.update(parsed)
    return by_name


def _num(cell: str) -> Optional[int]:
    cell = cell.strip().replace(",", "")
    if cell in ("", "-", "—"):
        return None
    try:
        return int(float(cell))
    except ValueError:
        return None


def _inc(cell: str) -> Optional[int]:
    """Parse an 'inc/cum' cell → the incremental value."""
    cell = cell.strip()
    if "/" in cell:
        cell = cell.split("/", 1)[0]
    return _num(cell)


def _cum(cell: str) -> Optional[int]:
    cell = cell.strip()
    if "/" in cell:
        cell = cell.split("/", 1)[1]
    return _num(cell)


def _parse_llm_usage_md(path: Path, warnings: Warnings) -> dict[str, ExperimentUsage]:
    out: dict[str, ExperimentUsage] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        warnings.add(f"cannot read {path}: {exc}")
        return out
    header_cols: list[str] = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not cells:
            continue
        low = [c.lower() for c in cells]
        if "step" in low and "type" in low:
            header_cols = low
            continue
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        if not header_cols:
            continue
        row = dict(zip(header_cols, cells))
        if row.get("type") != "experiment":
            continue
        name = row.get("step", "")
        tok = TokenTotals(
            input=_num(row.get("input", "")) or 0,
            cached_input=_num(row.get("cached", "")) or 0,
            output=_num(row.get("output", "")) or 0,
            reasoning=_num(row.get("reasoning", "")) or 0,
        )
        total_cell = row.get("total (inc/cum)", "")
        out[name] = ExperimentUsage(
            tokens=tok,
            total_cumulative=_cum(total_cell) or 0,
            model_calls=_inc(row.get("model calls (inc/cum)", "")) or 0,
            tool_calls=_inc(row.get("tool calls (inc/cum)", "")) or 0,
            tool_failures=_inc(row.get("tool failures (inc/cum)", "")) or 0,
            cache_hit_rate=cache_hit_rate(tok.cached_input, tok.input),
        )
    return out
