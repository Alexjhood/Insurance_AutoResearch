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
    status      TEXT DEFAULT 'pending', -- pending | running | paused | stopped | failed | done | interrupted
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Migrate existing DBs that lack agent_session_id column
CREATE TABLE IF NOT EXISTS _schema_migrations (key TEXT PRIMARY KEY);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    ts          TEXT NOT NULL,
    event_type  TEXT NOT NULL,          -- token | tool_use | turn_end | agent_exit | steer_sent | steer_queued | system
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
            # Live migration: add agent_session_id column to pre-existing DBs
            cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
            if "agent_session_id" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN agent_session_id TEXT")
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
                status, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)
            """,
            (job_id, track, run_id, surface, model_provider, model_name,
             cycles, memory_access, scope, guidance, seed_prompt,
             json.dumps(env), now, now),
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
