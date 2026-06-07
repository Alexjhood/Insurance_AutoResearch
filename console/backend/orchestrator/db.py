"""orchestrator.db — job and event persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from console.backend.config import ORCHESTRATOR_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    track       TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    surface     TEXT NOT NULL,          -- claude | codex | opencode
    model_provider TEXT,
    model_name  TEXT,
    cycles      INTEGER DEFAULT 3,
    memory_access TEXT DEFAULT 'none',  -- none | own | all
    scope       TEXT DEFAULT 'research',
    guidance    TEXT,
    seed_prompt TEXT,
    env_json    TEXT,
    worktree_path TEXT,
    pid         INTEGER,
    agent_session_id TEXT,              -- surface-native session ID for resume
    agent_model TEXT,                   -- agent model driving the run
    agent_effort TEXT,                  -- thinking/reasoning effort
    status      TEXT DEFAULT 'pending', -- pending | running | paused | stopped | failed | done | interrupted
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Marker table for migration bookkeeping
CREATE TABLE IF NOT EXISTS _schema_migrations (key TEXT PRIMARY KEY);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    ts          TEXT NOT NULL,
    event_type  TEXT NOT NULL,          -- token | tool_use | tool_result | turn_end | agent_exit | steer_sent | steer_queued | system
    payload_json TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS steer_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    message     TEXT NOT NULL,
    interrupt   INTEGER DEFAULT 1,
    queued_at   TEXT NOT NULL,
    sent_at     TEXT,
    status      TEXT DEFAULT 'pending', -- pending | sent | skipped
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS telemetry_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    surface     TEXT NOT NULL,
    session_id  TEXT,
    model       TEXT,
    effort      TEXT,
    started_at  TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL,
    is_error    INTEGER,
    input_tokens INTEGER,
    cached_input_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    uncached_input_tokens INTEGER,
    output_tokens INTEGER,
    reasoning_tokens INTEGER,
    total_tokens INTEGER,
    provider_cost_usd REAL,
    output_chars INTEGER DEFAULT 0,
    raw_usage_json TEXT,
    raw_result_json TEXT,
    UNIQUE(job_id, turn_index),
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS telemetry_tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    turn_id     INTEGER,
    provider_call_id TEXT,
    name        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL,
    status      TEXT,
    success     INTEGER,
    input_bytes INTEGER DEFAULT 0,
    output_bytes INTEGER DEFAULT 0,
    error       TEXT,
    raw_json    TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id),
    FOREIGN KEY (turn_id) REFERENCES telemetry_turns(id)
);

CREATE TABLE IF NOT EXISTS telemetry_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    turn_id     INTEGER,
    signal_type TEXT NOT NULL,
    cause       TEXT,
    attempt     INTEGER,
    raw_excerpt TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(id),
    FOREIGN KEY (turn_id) REFERENCES telemetry_turns(id)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_turns_job
    ON telemetry_turns(job_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_telemetry_tools_job
    ON telemetry_tool_calls(job_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_telemetry_signals_job
    ON telemetry_signals(job_id, turn_id);
"""


_INIT_LOCK = threading.Lock()
_initialized = False


def _ensure_initialized() -> None:
    """Create schema + run migrations exactly once per process."""
    global _initialized
    if _initialized:
        return
    with _INIT_LOCK:
        if _initialized:
            return
        ORCHESTRATOR_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(ORCHESTRATOR_DB))
        try:
            conn.executescript(SCHEMA)
            # WAL mode → concurrent readers (websocket polls) don't block writers
            conn.execute("PRAGMA journal_mode=WAL")
            # Live migrations: add columns to pre-existing DBs
            cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
            for col in ("agent_session_id", "agent_model", "agent_effort"):
                if col not in cols:
                    conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} TEXT")
            conn.commit()
        finally:
            conn.close()
        _initialized = True


def _connect() -> sqlite3.Connection:
    _ensure_initialized()
    conn = sqlite3.connect(str(ORCHESTRATOR_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def create_job(
    track: str,
    run_id: str,
    surface: str,
    model_provider: str | None,
    model_name: str | None,
    cycles: int,
    memory_access: str,
    scope: str,
    guidance: str | None,
    seed_prompt: str,
    env: dict,
    agent_model: str | None = None,
    agent_effort: str | None = None,
) -> str:
    job_id = str(uuid.uuid4())[:8]
    now = _now()
    conn = _connect()
    with conn:
        # Explicit column names — robust against schema additions.
        conn.execute(
            """
            INSERT INTO jobs (
                id, track, run_id, surface, model_provider, model_name,
                cycles, memory_access, scope, guidance, seed_prompt, env_json,
                agent_model, agent_effort, status, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)
            """,
            (job_id, track, run_id, surface, model_provider, model_name,
             cycles, memory_access, scope, guidance, seed_prompt,
             json.dumps(env), agent_model, agent_effort, now, now),
        )
    conn.close()
    return job_id


def update_job(job_id: str, **fields: Any) -> None:
    fields["updated_at"] = _now()
    set_clause = ", ".join(f"{k}=?" for k in fields)
    conn = _connect()
    with conn:
        conn.execute(
            f"UPDATE jobs SET {set_clause} WHERE id=?",
            list(fields.values()) + [job_id],
        )
    conn.close()


def get_job(job_id: str) -> dict | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_jobs() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_event(job_id: str, event_type: str, payload: Any = None) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            "INSERT INTO events (job_id, ts, event_type, payload_json) VALUES (?,?,?,?)",
            (job_id, _now(), event_type, json.dumps(payload) if payload is not None else None),
        )
    conn.close()


def get_events(job_id: str, after_id: int = 0) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id ASC",
        (job_id, after_id),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def start_telemetry_turn(
    job_id: str,
    *,
    surface: str,
    session_id: str | None,
    model: str | None,
    effort: str | None,
) -> int:
    conn = _connect()
    with conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM telemetry_turns WHERE job_id=?",
            (job_id,),
        ).fetchone()
        turn_index = int(row[0])
        cur = conn.execute(
            """
            INSERT INTO telemetry_turns (
                job_id, turn_index, surface, session_id, model, effort, started_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (job_id, turn_index, surface, session_id, model, effort, _now()),
        )
        turn_id = int(cur.lastrowid)
    conn.close()
    return turn_id


def finish_telemetry_turn(
    turn_id: int,
    *,
    session_id: str | None,
    duration_ms: float | None,
    is_error: bool | None,
    output_chars: int,
    usage: dict[str, Any],
    raw_usage: Any,
    raw_result: Any,
) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            UPDATE telemetry_turns
            SET session_id=COALESCE(?, session_id),
                completed_at=?,
                duration_ms=?,
                is_error=?,
                input_tokens=?,
                cached_input_tokens=?,
                cache_creation_input_tokens=?,
                uncached_input_tokens=?,
                output_tokens=?,
                reasoning_tokens=?,
                total_tokens=?,
                provider_cost_usd=?,
                output_chars=?,
                raw_usage_json=?,
                raw_result_json=?
            WHERE id=?
            """,
            (
                session_id,
                _now(),
                duration_ms,
                _bool_int(is_error),
                usage.get("input_tokens"),
                usage.get("cached_input_tokens"),
                usage.get("cache_creation_input_tokens"),
                usage.get("uncached_input_tokens"),
                usage.get("output_tokens"),
                usage.get("reasoning_tokens"),
                usage.get("total_tokens"),
                usage.get("provider_cost_usd"),
                output_chars,
                json.dumps(raw_usage) if raw_usage is not None else None,
                json.dumps(raw_result) if raw_result is not None else None,
                turn_id,
            ),
        )
    conn.close()


def record_telemetry_tool_call(
    job_id: str,
    turn_id: int | None,
    *,
    normalized: dict[str, Any],
    raw_payload: dict[str, Any],
) -> int:
    now = _now()
    completed = _tool_is_completed(normalized)
    conn = _connect()
    with conn:
        cur = conn.execute(
            """
            INSERT INTO telemetry_tool_calls (
                job_id, turn_id, provider_call_id, name, started_at, completed_at,
                duration_ms, status, success, input_bytes, output_bytes, error, raw_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                turn_id,
                normalized.get("provider_call_id"),
                normalized.get("name") or "tool",
                now,
                now if completed else None,
                normalized.get("duration_ms"),
                normalized.get("status"),
                _bool_int(normalized.get("success")),
                normalized.get("input_bytes") or 0,
                normalized.get("output_bytes") or 0,
                normalized.get("error"),
                json.dumps(raw_payload),
            ),
        )
        tool_id = int(cur.lastrowid)
    conn.close()
    return tool_id


def finish_telemetry_tool_call(
    job_id: str,
    turn_id: int | None,
    *,
    normalized: dict[str, Any],
    raw_payload: dict[str, Any],
) -> int:
    provider_call_id = normalized.get("provider_call_id")
    conn = _connect()
    row = None
    if provider_call_id:
        row = conn.execute(
            """
            SELECT id FROM telemetry_tool_calls
            WHERE job_id=? AND provider_call_id=? AND completed_at IS NULL
            ORDER BY id DESC LIMIT 1
            """,
            (job_id, provider_call_id),
        ).fetchone()
    if row is None:
        conn.close()
        return record_telemetry_tool_call(
            job_id,
            turn_id,
            normalized=normalized,
            raw_payload=raw_payload,
        )
    tool_id = int(row["id"])
    completed_at = _now()
    with conn:
        conn.execute(
            """
            UPDATE telemetry_tool_calls
            SET completed_at=?,
                duration_ms=COALESCE(
                    ?,
                    (julianday(?) - julianday(started_at)) * 86400000.0,
                    duration_ms
                ),
                status=COALESCE(?, status),
                success=COALESCE(?, success),
                output_bytes=MAX(output_bytes, ?),
                error=COALESCE(?, error),
                raw_json=?
            WHERE id=?
            """,
            (
                completed_at,
                normalized.get("duration_ms"),
                completed_at,
                normalized.get("status"),
                _bool_int(normalized.get("success")),
                normalized.get("output_bytes") or 0,
                normalized.get("error"),
                json.dumps(raw_payload),
                tool_id,
            ),
        )
    conn.close()
    return tool_id


def record_telemetry_signal(
    job_id: str,
    turn_id: int | None,
    *,
    signal_type: str,
    cause: str | None,
    attempt: int | None,
    raw_excerpt: str | None,
) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """
            INSERT INTO telemetry_signals (
                job_id, turn_id, signal_type, cause, attempt, raw_excerpt, created_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (job_id, turn_id, signal_type, cause, attempt, raw_excerpt, _now()),
        )
    conn.close()


def get_job_telemetry(job_id: str) -> dict[str, Any]:
    conn = _connect()
    turn_rows = conn.execute(
        "SELECT * FROM telemetry_turns WHERE job_id=? ORDER BY turn_index ASC",
        (job_id,),
    ).fetchall()
    tool_rows = conn.execute(
        "SELECT * FROM telemetry_tool_calls WHERE job_id=? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    signal_rows = conn.execute(
        "SELECT * FROM telemetry_signals WHERE job_id=? ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    conn.close()

    turns = [_decode_json_columns(dict(row), "raw_usage_json", "raw_result_json") for row in turn_rows]
    tools = [_decode_json_columns(dict(row), "raw_json") for row in tool_rows]
    signals = [dict(row) for row in signal_rows]

    sums = {
        key: sum(int(turn.get(key) or 0) for turn in turns)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "uncached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
            "output_chars",
        )
    }
    reported_costs = [
        float(turn["provider_cost_usd"])
        for turn in turns
        if turn.get("provider_cost_usd") is not None
    ]
    completed_tools = [tool for tool in tools if tool.get("completed_at")]
    failed_tools = [tool for tool in completed_tools if tool.get("success") == 0]
    tool_duration_values = [
        float(tool["duration_ms"])
        for tool in completed_tools
        if tool.get("duration_ms") is not None
    ]
    cached = sums["cached_input_tokens"]
    total_input = sums["input_tokens"]
    cache_hit_ratio = cached / total_input if total_input > 0 else None

    return {
        "job_id": job_id,
        "summary": {
            "turn_count": len(turns),
            **sums,
            "cache_hit_ratio": cache_hit_ratio,
            "provider_reported_cost_usd": sum(reported_costs) if reported_costs else None,
            "cost_coverage_turns": len(reported_costs),
            "tool_call_count": len(tools),
            "completed_tool_call_count": len(completed_tools),
            "tool_failure_count": len(failed_tools),
            "tool_duration_ms": sum(tool_duration_values) if tool_duration_values else None,
            "signal_count": len(signals),
        },
        "turns": turns,
        "tool_calls": tools,
        "signals": signals,
    }


def enqueue_steer(job_id: str, message: str, interrupt: bool) -> int:
    conn = _connect()
    with conn:
        cur = conn.execute(
            "INSERT INTO steer_queue (job_id, message, interrupt, queued_at) VALUES (?,?,?,?)",
            (job_id, message, int(interrupt), _now()),
        )
        row_id = cur.lastrowid
    conn.close()
    return row_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_int(value: bool | None) -> int | None:
    return int(value) if isinstance(value, bool) else None


def _tool_is_completed(normalized: dict[str, Any]) -> bool:
    return (
        normalized.get("success") is not None
        or normalized.get("duration_ms") is not None
        or normalized.get("output_bytes", 0) > 0
        or str(normalized.get("status") or "").lower()
        in {"completed", "success", "succeeded", "done", "failed", "error"}
    )


def _decode_json_columns(row: dict[str, Any], *columns: str) -> dict[str, Any]:
    for column in columns:
        raw = row.get(column)
        if raw is None:
            continue
        try:
            row[column.removesuffix("_json")] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            row[column.removesuffix("_json")] = raw
    return row
