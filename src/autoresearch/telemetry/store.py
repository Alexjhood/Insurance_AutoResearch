"""Provider-neutral, run-scoped telemetry persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_sessions (
    session_key TEXT PRIMARY KEY,
    surface TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    transcript_path TEXT,
    cwd TEXT,
    model TEXT,
    effort TEXT,
    source_version TEXT,
    started_at TEXT,
    completed_at TEXT,
    last_event_at TEXT,
    last_import_at TEXT NOT NULL,
    import_status TEXT NOT NULL DEFAULT 'active',
    UNIQUE(surface, native_session_id)
);

CREATE TABLE IF NOT EXISTS llm_turns (
    turn_key TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    native_turn_id TEXT,
    turn_index INTEGER NOT NULL,
    model TEXT,
    effort TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms REAL,
    time_to_first_token_ms REAL,
    status TEXT NOT NULL DEFAULT 'active',
    output_chars INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    provider_cost_usd REAL,
    model_call_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(session_key, turn_index),
    FOREIGN KEY (session_key) REFERENCES llm_sessions(session_key)
);

CREATE TABLE IF NOT EXISTS llm_model_calls (
    call_key TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    turn_key TEXT,
    provider_request_id TEXT,
    occurred_at TEXT,
    model TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    uncached_input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    provider_cost_usd REAL,
    output_chars INTEGER NOT NULL DEFAULT 0,
    workflow_event_id INTEGER,
    FOREIGN KEY (session_key) REFERENCES llm_sessions(session_key),
    FOREIGN KEY (turn_key) REFERENCES llm_turns(turn_key)
);

CREATE TABLE IF NOT EXISTS llm_tool_calls (
    call_key TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    turn_key TEXT,
    provider_call_id TEXT,
    name TEXT NOT NULL,
    detail TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms REAL,
    status TEXT,
    success INTEGER,
    input_bytes INTEGER NOT NULL DEFAULT 0,
    output_bytes INTEGER NOT NULL DEFAULT 0,
    stdout_bytes INTEGER NOT NULL DEFAULT 0,
    stderr_bytes INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,
    workflow_event_id INTEGER,
    FOREIGN KEY (session_key) REFERENCES llm_sessions(session_key),
    FOREIGN KEY (turn_key) REFERENCES llm_turns(turn_key)
);

CREATE TABLE IF NOT EXISTS llm_signals (
    signal_key TEXT PRIMARY KEY,
    session_key TEXT NOT NULL,
    turn_key TEXT,
    tool_call_key TEXT,
    signal_type TEXT NOT NULL,
    created_at TEXT,
    FOREIGN KEY (session_key) REFERENCES llm_sessions(session_key)
);

CREATE TABLE IF NOT EXISTS llm_import_cursors (
    source_path TEXT PRIMARY KEY,
    surface TEXT NOT NULL,
    native_session_id TEXT NOT NULL,
    byte_offset INTEGER NOT NULL DEFAULT 0,
    source_size INTEGER NOT NULL DEFAULT 0,
    source_mtime_ns INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS workflow_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL UNIQUE,
    command TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL,
    status TEXT NOT NULL,
    error_type TEXT,
    native_session_id TEXT
);

CREATE TABLE IF NOT EXISTS workflow_entities (
    event_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'created',
    PRIMARY KEY (event_id, entity_type, entity_id, relation),
    FOREIGN KEY (event_id) REFERENCES workflow_events(id)
);

CREATE TABLE IF NOT EXISTS experiment_usage_checkpoints (
    experiment_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    status TEXT NOT NULL,
    completed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_usage_checkpoints (
    checkpoint_key TEXT PRIMARY KEY,
    checkpoint_type TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    experiment_id TEXT,
    session_key TEXT,
    turn_key TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_turns_session ON llm_turns(session_key, turn_index);
CREATE INDEX IF NOT EXISTS idx_llm_model_calls_turn ON llm_model_calls(turn_key);
CREATE INDEX IF NOT EXISTS idx_llm_tools_turn ON llm_tool_calls(turn_key);
CREATE INDEX IF NOT EXISTS idx_workflow_events_time ON workflow_events(started_at, completed_at);
CREATE INDEX IF NOT EXISTS idx_experiment_usage_time
    ON experiment_usage_checkpoints(completed_at, experiment_id);
CREATE INDEX IF NOT EXISTS idx_llm_usage_checkpoint_time
    ON llm_usage_checkpoints(occurred_at, checkpoint_key);
"""


def telemetry_path(run_dir: Path) -> Path:
    return Path(run_dir) / "telemetry.sqlite"


def connect(run_dir: Path) -> sqlite3.Connection:
    path = telemetry_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _ensure_column(con, "llm_turns", "model", "TEXT")
    _ensure_column(con, "llm_turns", "effort", "TEXT")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_key(surface: str, native_session_id: str) -> str:
    return f"{surface}:{native_session_id}"


def upsert_session(
    con: sqlite3.Connection,
    *,
    surface: str,
    native_session_id: str,
    transcript_path: Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    metadata = metadata or {}
    key = session_key(surface, native_session_id)
    con.execute(
        """
        INSERT INTO llm_sessions (
            session_key, surface, native_session_id, transcript_path, cwd, model,
            effort, source_version, started_at, last_event_at, last_import_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(session_key) DO UPDATE SET
            transcript_path=excluded.transcript_path,
            cwd=COALESCE(excluded.cwd, llm_sessions.cwd),
            model=COALESCE(excluded.model, llm_sessions.model),
            effort=COALESCE(excluded.effort, llm_sessions.effort),
            source_version=COALESCE(excluded.source_version, llm_sessions.source_version),
            started_at=COALESCE(llm_sessions.started_at, excluded.started_at),
            last_event_at=COALESCE(excluded.last_event_at, llm_sessions.last_event_at),
            last_import_at=excluded.last_import_at,
            import_status='active'
        """,
        (
            key,
            surface,
            native_session_id,
            str(transcript_path),
            metadata.get("cwd"),
            metadata.get("model"),
            metadata.get("effort"),
            metadata.get("source_version"),
            metadata.get("started_at"),
            metadata.get("last_event_at"),
            now_iso(),
        ),
    )
    return key


def ensure_turn(
    con: sqlite3.Connection,
    *,
    session_key_value: str,
    native_turn_id: str | None,
    started_at: str | None,
    turn_key_value: str | None = None,
) -> str:
    if turn_key_value:
        existing = con.execute(
            "SELECT turn_key FROM llm_turns WHERE turn_key=?", (turn_key_value,)
        ).fetchone()
        if existing:
            return str(existing["turn_key"])
    if native_turn_id:
        existing = con.execute(
            "SELECT turn_key FROM llm_turns WHERE session_key=? AND native_turn_id=?",
            (session_key_value, native_turn_id),
        ).fetchone()
        if existing:
            return str(existing["turn_key"])
    row = con.execute(
        "SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_index FROM llm_turns WHERE session_key=?",
        (session_key_value,),
    ).fetchone()
    index = int(row["next_index"])
    key = turn_key_value or f"{session_key_value}:turn:{native_turn_id or index}"
    con.execute(
        """
        INSERT OR IGNORE INTO llm_turns (
            turn_key, session_key, native_turn_id, turn_index, started_at
        ) VALUES (?,?,?,?,?)
        """,
        (key, session_key_value, native_turn_id, index, started_at),
    )
    return key


def latest_turn(con: sqlite3.Connection, session_key_value: str) -> str | None:
    row = con.execute(
        "SELECT turn_key FROM llm_turns WHERE session_key=? ORDER BY turn_index DESC LIMIT 1",
        (session_key_value,),
    ).fetchone()
    return str(row["turn_key"]) if row else None


def complete_turn(
    con: sqlite3.Connection,
    turn_key_value: str,
    *,
    completed_at: str | None,
    duration_ms: float | None = None,
    time_to_first_token_ms: float | None = None,
    status: str = "completed",
) -> None:
    con.execute(
        """
        UPDATE llm_turns SET
            completed_at=COALESCE(?, completed_at),
            duration_ms=COALESCE(?, duration_ms),
            time_to_first_token_ms=COALESCE(?, time_to_first_token_ms),
            status=?
        WHERE turn_key=?
        """,
        (completed_at, duration_ms, time_to_first_token_ms, status, turn_key_value),
    )


def refresh_session_status(
    con: sqlite3.Connection,
    session_key_value: str,
    *,
    finalize: bool,
) -> None:
    row = con.execute(
        """
        SELECT COUNT(*) AS turn_count,
               SUM(CASE WHEN status != 'completed' THEN 1 ELSE 0 END) AS active_turns,
               MAX(completed_at) AS completed_at
        FROM llm_turns WHERE session_key=?
        """,
        (session_key_value,),
    ).fetchone()
    complete = bool(
        finalize
        and row
        and int(row["turn_count"] or 0) > 0
        and int(row["active_turns"] or 0) == 0
    )
    con.execute(
        """
        UPDATE llm_sessions SET
            import_status=?,
            completed_at=CASE WHEN ? THEN COALESCE(?, completed_at) ELSE NULL END
        WHERE session_key=?
        """,
        (
            "completed" if complete else "active",
            int(complete),
            row["completed_at"] if row else None,
            session_key_value,
        ),
    )


def update_turn_context(
    con: sqlite3.Connection,
    turn_key_value: str,
    *,
    model: str | None,
    effort: str | None,
) -> None:
    con.execute(
        """
        UPDATE llm_turns SET
            model=COALESCE(?, model),
            effort=COALESCE(?, effort)
        WHERE turn_key=?
        """,
        (model, effort, turn_key_value),
    )


def upsert_model_call(con: sqlite3.Connection, values: dict[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO llm_model_calls (
            call_key, session_key, turn_key, provider_request_id, occurred_at, model,
            input_tokens, cached_input_tokens, cache_creation_input_tokens,
            uncached_input_tokens, output_tokens, reasoning_tokens, total_tokens,
            provider_cost_usd, output_chars
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(call_key) DO UPDATE SET
            turn_key=COALESCE(llm_model_calls.turn_key, excluded.turn_key),
            model=COALESCE(excluded.model, llm_model_calls.model),
            output_chars=MAX(llm_model_calls.output_chars, excluded.output_chars)
        """,
        (
            values["call_key"],
            values["session_key"],
            values.get("turn_key"),
            values.get("provider_request_id"),
            values.get("occurred_at"),
            values.get("model"),
            values.get("input_tokens", 0),
            values.get("cached_input_tokens", 0),
            values.get("cache_creation_input_tokens", 0),
            values.get("uncached_input_tokens", 0),
            values.get("output_tokens", 0),
            values.get("reasoning_tokens", 0),
            values.get("total_tokens", 0),
            values.get("provider_cost_usd"),
            values.get("output_chars", 0),
        ),
    )


def upsert_tool_call(con: sqlite3.Connection, values: dict[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO llm_tool_calls (
            call_key, session_key, turn_key, provider_call_id, name, detail,
            started_at, completed_at, duration_ms, status, success, input_bytes,
            output_bytes, stdout_bytes, stderr_bytes, error_type
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(call_key) DO UPDATE SET
            turn_key=COALESCE(llm_tool_calls.turn_key, excluded.turn_key),
            detail=COALESCE(excluded.detail, llm_tool_calls.detail),
            completed_at=COALESCE(excluded.completed_at, llm_tool_calls.completed_at),
            duration_ms=COALESCE(excluded.duration_ms, llm_tool_calls.duration_ms),
            status=COALESCE(excluded.status, llm_tool_calls.status),
            success=COALESCE(excluded.success, llm_tool_calls.success),
            output_bytes=MAX(llm_tool_calls.output_bytes, excluded.output_bytes),
            stdout_bytes=MAX(llm_tool_calls.stdout_bytes, excluded.stdout_bytes),
            stderr_bytes=MAX(llm_tool_calls.stderr_bytes, excluded.stderr_bytes),
            error_type=COALESCE(excluded.error_type, llm_tool_calls.error_type)
        """,
        (
            values["call_key"],
            values["session_key"],
            values.get("turn_key"),
            values.get("provider_call_id"),
            values.get("name") or "tool",
            values.get("detail"),
            values.get("started_at"),
            values.get("completed_at"),
            values.get("duration_ms"),
            values.get("status"),
            _bool_int(values.get("success")),
            values.get("input_bytes", 0),
            values.get("output_bytes", 0),
            values.get("stdout_bytes", 0),
            values.get("stderr_bytes", 0),
            values.get("error_type"),
        ),
    )


def record_signals(
    con: sqlite3.Connection,
    *,
    session_key_value: str,
    turn_key_value: str | None,
    tool_call_key: str,
    signal_types: Iterable[str],
    created_at: str | None,
) -> None:
    con.execute("DELETE FROM llm_signals WHERE tool_call_key=?", (tool_call_key,))
    for signal_type in signal_types:
        con.execute(
            """
            INSERT OR IGNORE INTO llm_signals (
                signal_key, session_key, turn_key, tool_call_key, signal_type, created_at
            ) VALUES (?,?,?,?,?,?)
            """,
            (
                f"{tool_call_key}:{signal_type}",
                session_key_value,
                turn_key_value,
                tool_call_key,
                signal_type,
                created_at,
            ),
        )


def refresh_turn_aggregates(con: sqlite3.Connection, session_key_value: str) -> None:
    turns = con.execute(
        "SELECT turn_key FROM llm_turns WHERE session_key=?", (session_key_value,)
    ).fetchall()
    for row in turns:
        key = row["turn_key"]
        sums = con.execute(
            """
            SELECT
                COUNT(*) AS model_call_count,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens,
                COALESCE(SUM(cache_creation_input_tokens), 0) AS cache_creation_input_tokens,
                COALESCE(SUM(uncached_input_tokens), 0) AS uncached_input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(reasoning_tokens), 0) AS reasoning_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(output_chars), 0) AS output_chars,
                SUM(provider_cost_usd) AS provider_cost_usd
            FROM llm_model_calls WHERE turn_key=?
            """,
            (key,),
        ).fetchone()
        con.execute(
            """
            UPDATE llm_turns SET
                model_call_count=?, input_tokens=?, cached_input_tokens=?,
                cache_creation_input_tokens=?, uncached_input_tokens=?,
                output_tokens=?, reasoning_tokens=?, total_tokens=?, output_chars=?,
                provider_cost_usd=?
            WHERE turn_key=?
            """,
            (
                sums["model_call_count"],
                sums["input_tokens"],
                sums["cached_input_tokens"],
                sums["cache_creation_input_tokens"],
                sums["uncached_input_tokens"],
                sums["output_tokens"],
                sums["reasoning_tokens"],
                sums["total_tokens"],
                sums["output_chars"],
                sums["provider_cost_usd"],
                key,
            ),
        )


def cursor_for(con: sqlite3.Connection, source_path: Path) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT * FROM llm_import_cursors WHERE source_path=?", (str(source_path),)
    ).fetchone()
    return dict(row) if row else None


def update_cursor(
    con: sqlite3.Connection,
    *,
    source_path: Path,
    surface: str,
    native_session_id: str,
    byte_offset: int,
    source_size: int,
    source_mtime_ns: int,
    last_error: str | None = None,
) -> None:
    con.execute(
        """
        INSERT INTO llm_import_cursors (
            source_path, surface, native_session_id, byte_offset, source_size,
            source_mtime_ns, updated_at, last_error
        ) VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(source_path) DO UPDATE SET
            byte_offset=excluded.byte_offset,
            source_size=excluded.source_size,
            source_mtime_ns=excluded.source_mtime_ns,
            updated_at=excluded.updated_at,
            last_error=excluded.last_error
        """,
        (
            str(source_path),
            surface,
            native_session_id,
            byte_offset,
            source_size,
            source_mtime_ns,
            now_iso(),
            last_error,
        ),
    )


def record_workflow_event(
    run_dir: Path,
    *,
    event_key: str,
    command: str,
    started_at: str,
    completed_at: str | None,
    duration_ms: float | None,
    status: str,
    error_type: str | None,
    entities: Iterable[tuple[str, str, str]] = (),
) -> int:
    con = connect(run_dir)
    try:
        with con:
            con.execute(
                """
                INSERT INTO workflow_events (
                    event_key, command, started_at, completed_at, duration_ms,
                    status, error_type
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(event_key) DO UPDATE SET
                    completed_at=excluded.completed_at,
                    duration_ms=excluded.duration_ms,
                    status=excluded.status,
                    error_type=excluded.error_type
                """,
                (
                    event_key,
                    command,
                    started_at,
                    completed_at,
                    duration_ms,
                    status,
                    error_type,
                ),
            )
            event_id = int(
                con.execute(
                    "SELECT id FROM workflow_events WHERE event_key=?", (event_key,)
                ).fetchone()["id"]
            )
            con.executemany(
                """
                INSERT OR IGNORE INTO workflow_entities (
                    event_id, entity_type, entity_id, relation
                ) VALUES (?,?,?,?)
                """,
                ((event_id, kind, entity_id, relation) for kind, entity_id, relation in entities),
            )
        return event_id
    finally:
        con.close()


def link_workflow_events(con: sqlite3.Connection) -> None:
    """Deterministically rebuild CLI workflow attribution."""

    con.execute("UPDATE llm_tool_calls SET workflow_event_id=NULL")
    con.execute("UPDATE llm_model_calls SET workflow_event_id=NULL")
    events = con.execute(
        "SELECT * FROM workflow_events WHERE completed_at IS NOT NULL ORDER BY started_at"
    ).fetchall()
    for event in events:
        tool = con.execute(
            """
            SELECT call_key, turn_key FROM llm_tool_calls
            WHERE workflow_event_id IS NULL
              AND julianday(started_at) <= julianday(?) + (5.0 / 86400.0)
              AND julianday(COALESCE(completed_at, started_at))
                    >= julianday(?) - (5.0 / 86400.0)
              AND (detail=? OR detail LIKE ?)
            ORDER BY ABS(julianday(started_at) - julianday(?)) ASC
            LIMIT 1
            """,
            (
                event["completed_at"],
                event["started_at"],
                f"autoresearch {event['command']}",
                f"%{event['command']}%",
                event["started_at"],
            ),
        ).fetchone()
        if not tool:
            continue
        con.execute(
            "UPDATE llm_tool_calls SET workflow_event_id=? WHERE call_key=?",
            (event["id"], tool["call_key"]),
        )
        model_call = con.execute(
            """
            SELECT call_key FROM llm_model_calls
            WHERE turn_key=? AND workflow_event_id IS NULL
            ORDER BY ABS(julianday(occurred_at) - julianday(?)) ASC
            LIMIT 1
            """,
            (tool["turn_key"], event["started_at"]),
        ).fetchone()
        if model_call:
            con.execute(
                "UPDATE llm_model_calls SET workflow_event_id=? WHERE call_key=?",
                (event["id"], model_call["call_key"]),
            )


def record_experiment_checkpoint(
    run_dir: Path,
    *,
    experiment_id: str,
    experiment_name: str,
    status: str,
    completed_at: str | None = None,
) -> None:
    con = connect(run_dir)
    try:
        checkpoint_at = completed_at or now_iso()
        with con:
            con.execute(
                """
                INSERT INTO experiment_usage_checkpoints (
                    experiment_id, experiment_name, status, completed_at
                ) VALUES (?,?,?,?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    experiment_name=excluded.experiment_name,
                    status=excluded.status
                """,
                (
                    experiment_id,
                    experiment_name,
                    status,
                    checkpoint_at,
                ),
            )
            con.execute(
                """
                INSERT INTO llm_usage_checkpoints (
                    checkpoint_key, checkpoint_type, label, status, occurred_at,
                    experiment_id
                ) VALUES (?,?,?,?,?,?)
                ON CONFLICT(checkpoint_key) DO UPDATE SET
                    label=excluded.label,
                    status=excluded.status,
                    occurred_at=excluded.occurred_at
                """,
                (
                    f"experiment:{experiment_id}",
                    "experiment",
                    experiment_name,
                    status,
                    checkpoint_at,
                    experiment_id,
                ),
            )
    finally:
        con.close()


def record_user_breakpoint(
    run_dir: Path,
    *,
    session_key_value: str,
) -> str | None:
    """Record one idempotent checkpoint for the latest completed native turn."""

    con = connect(run_dir)
    try:
        turn = con.execute(
            """
            SELECT turn_key, turn_index, completed_at
            FROM llm_turns
            WHERE session_key=? AND status='completed' AND completed_at IS NOT NULL
            ORDER BY turn_index DESC
            LIMIT 1
            """,
            (session_key_value,),
        ).fetchone()
        if turn is None:
            return None
        occurred_at = str(turn["completed_at"])
        checkpoint_key = f"user_breakpoint:{turn['turn_key']}"
        session = con.execute(
            "SELECT surface FROM llm_sessions WHERE session_key=?",
            (session_key_value,),
        ).fetchone()
        surface = str(session["surface"] if session else "agent")
        with con:
            con.execute(
                """
                INSERT INTO llm_usage_checkpoints (
                    checkpoint_key, checkpoint_type, label, status, occurred_at,
                    session_key, turn_key
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(checkpoint_key) DO UPDATE SET
                    label=excluded.label,
                    status=excluded.status
                """,
                (
                    checkpoint_key,
                    "user_breakpoint",
                    f"User breakpoint ({surface} turn {int(turn['turn_index'])})",
                    "settled",
                    occurred_at,
                    session_key_value,
                    str(turn["turn_key"]),
                ),
            )
        return checkpoint_key
    finally:
        con.close()


def get_run_telemetry(run_dir: Path) -> dict[str, Any]:
    path = telemetry_path(run_dir)
    if not path.exists():
        return empty_report()
    con = connect(run_dir)
    try:
        sessions = [_public_session(dict(row)) for row in con.execute(
            "SELECT * FROM llm_sessions ORDER BY started_at"
        ).fetchall()]
        turns = [dict(row) for row in con.execute(
            "SELECT * FROM llm_turns ORDER BY started_at, turn_index"
        ).fetchall()]
        tools = [dict(row) for row in con.execute(
            """
            SELECT call_key, session_key, turn_key, name, detail, started_at,
                   completed_at, duration_ms, status, success, input_bytes,
                   output_bytes, stdout_bytes, stderr_bytes, error_type,
                   workflow_event_id
            FROM llm_tool_calls ORDER BY started_at
            """
        ).fetchall()]
        signals = [dict(row) for row in con.execute(
            "SELECT * FROM llm_signals ORDER BY created_at"
        ).fetchall()]
        workflows = _workflow_rows(con)
        cursor_rows = con.execute(
            """
            SELECT surface, native_session_id, byte_offset, source_size,
                   updated_at, last_error FROM llm_import_cursors ORDER BY updated_at
            """
        ).fetchall()
    finally:
        con.close()

    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "output_chars",
        "model_call_count",
    )
    sums = {field: sum(int(turn.get(field) or 0) for turn in turns) for field in token_fields}
    completed_tools = [tool for tool in tools if tool.get("completed_at")]
    failed_tools = [tool for tool in completed_tools if tool.get("success") == 0]
    duration_values = [
        float(tool["duration_ms"]) for tool in completed_tools if tool.get("duration_ms") is not None
    ]
    costs = [
        float(turn["provider_cost_usd"])
        for turn in turns
        if turn.get("provider_cost_usd") is not None
    ]
    total_input = sums["input_tokens"]
    cached = sums["cached_input_tokens"]
    imported = sum(int(row["byte_offset"]) for row in cursor_rows)
    available = sum(int(row["source_size"]) for row in cursor_rows)
    all_turns_complete = bool(turns) and all(turn.get("status") == "completed" for turn in turns)
    summary = {
        "session_count": len(sessions),
        "turn_count": len(turns),
        **sums,
        "cache_hit_ratio": cached / total_input if total_input else None,
        "provider_reported_cost_usd": sum(costs) if costs else None,
        "cost_coverage_turns": len(costs),
        "tool_call_count": len(tools),
        "completed_tool_call_count": len(completed_tools),
        "tool_failure_count": len(failed_tools),
        "tool_duration_ms": sum(duration_values) if duration_values else None,
        "signal_count": len(signals),
        "workflow_event_count": len(workflows),
        "imported_bytes": imported,
        "source_bytes": available,
        "import_complete": bool(cursor_rows) and imported >= available and all_turns_complete,
    }
    return {
        "available": True,
        "summary": summary,
        "sessions": sessions,
        "turns": turns,
        "tool_calls": tools,
        "signals": signals,
        "workflow_events": workflows,
        "coverage": [dict(row) for row in cursor_rows],
    }


def empty_report() -> dict[str, Any]:
    return {
        "available": False,
        "summary": {
            "session_count": 0,
            "turn_count": 0,
            "model_call_count": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "output_chars": 0,
            "cache_hit_ratio": None,
            "provider_reported_cost_usd": None,
            "cost_coverage_turns": 0,
            "tool_call_count": 0,
            "completed_tool_call_count": 0,
            "tool_failure_count": 0,
            "tool_duration_ms": None,
            "signal_count": 0,
            "workflow_event_count": 0,
            "imported_bytes": 0,
            "source_bytes": 0,
            "import_complete": False,
        },
        "sessions": [],
        "turns": [],
        "tool_calls": [],
        "signals": [],
        "workflow_events": [],
        "coverage": [],
    }


def _workflow_rows(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = [dict(row) for row in con.execute(
        "SELECT * FROM workflow_events ORDER BY started_at"
    ).fetchall()]
    for row in rows:
        entities = con.execute(
            """
            SELECT entity_type, entity_id, relation FROM workflow_entities
            WHERE event_id=? ORDER BY entity_type, entity_id
            """,
            (row["id"],),
        ).fetchall()
        row["entities"] = [dict(entity) for entity in entities]
        usage = con.execute(
            """
            SELECT COALESCE(SUM(total_tokens), 0) AS direct_tokens,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens
            FROM llm_model_calls WHERE workflow_event_id=?
            """,
            (row["id"],),
        ).fetchone()
        row.update(dict(usage))
    return rows


def _public_session(row: dict[str, Any]) -> dict[str, Any]:
    raw_path = row.pop("transcript_path", None)
    row["transcript_file"] = Path(raw_path).name if raw_path else None
    return row


def _bool_int(value: bool | None) -> int | None:
    return int(value) if isinstance(value, bool) else None


def _ensure_column(
    con: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
