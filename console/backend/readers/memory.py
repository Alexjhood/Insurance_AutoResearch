"""Read-only access to the cross-run memory aggregator."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from console.backend.config import MEMORY_DB, PLAYBOOK_PATH


def _connect() -> sqlite3.Connection | None:
    if not MEMORY_DB.exists():
        return None
    conn = sqlite3.connect(f"file:{MEMORY_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def leaderboard() -> dict:
    conn = _connect()
    if not conn:
        return {"available": False, "models": [], "runs": [], "top_experiments": []}

    with conn:
        models = [dict(r) for r in conn.execute("SELECT * FROM models ORDER BY last_seen DESC")]
        runs = [dict(r) for r in conn.execute(
            "SELECT r.*, m.provider, m.name AS model_name "
            "FROM runs r JOIN models m USING(model_id) "
            "ORDER BY r.peak_gini DESC NULLS LAST"
        )]
        top_experiments = [dict(r) for r in conn.execute(
            "SELECT e.*, r.track_id, r.run_id, m.provider, m.name AS model_name "
            "FROM experiments e "
            "JOIN runs r USING(run_uid) "
            "JOIN models m USING(model_id) "
            "WHERE e.status IN ('ok', 'completed') "
            "ORDER BY e.gini_weighted DESC NULLS LAST "
            "LIMIT 50"
        )]
        insights = [dict(r) for r in conn.execute(
            "SELECT * FROM insights ORDER BY created_at DESC LIMIT 20"
        )] if _table_exists(conn, "insights") else []

    conn.close()
    return {
        "available": True,
        "models": models,
        "runs": runs,
        "top_experiments": top_experiments,
        "insights": insights,
    }


def playbook_text() -> str | None:
    if PLAYBOOK_PATH.exists():
        return PLAYBOOK_PATH.read_text(encoding="utf-8")
    return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None
