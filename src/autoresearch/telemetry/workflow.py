"""Framework-owned workflow events for experiment attribution."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from autoresearch.telemetry.store import record_workflow_event


@contextmanager
def track_command(config, command: str) -> Iterator[dict[str, Any]]:
    """Record command timing and newly-created registry entities."""

    if not getattr(config, "track_base_dir", None):
        yield {"return_code": 0}
        return
    started = datetime.now(timezone.utc)
    before = _registry_snapshot(config.registry_path)
    event_key = str(uuid.uuid4())
    status = "completed"
    error_type = None
    state: dict[str, Any] = {"return_code": 0}
    started_clock = time.monotonic()
    try:
        yield state
    except BaseException as exc:
        status = "failed"
        error_type = type(exc).__name__
        raise
    finally:
        if int(state.get("return_code") or 0) != 0:
            status = "failed"
            error_type = error_type or f"exit_{state['return_code']}"
        completed = datetime.now(timezone.utc)
        after = _registry_snapshot(config.registry_path)
        entities = []
        for kind in ("proposal", "experiment", "comparison", "promotion"):
            for entity_id in sorted(after[kind] - before[kind]):
                entities.append((kind, entity_id, "created"))
        try:
            record_workflow_event(
                config.artifacts_dir,
                event_key=event_key,
                command=command,
                started_at=started.isoformat(),
                completed_at=completed.isoformat(),
                duration_ms=(time.monotonic() - started_clock) * 1000,
                status=status,
                error_type=error_type,
                entities=entities,
            )
        except Exception:
            pass


def _registry_snapshot(path: Path) -> dict[str, set[str]]:
    empty = {kind: set() for kind in ("proposal", "experiment", "comparison", "promotion")}
    if not Path(path).exists():
        return empty
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        values = {
            "proposal": _ids(con, "SELECT proposal_id FROM proposals"),
            "experiment": _ids(con, "SELECT experiment_id FROM experiments"),
            "comparison": _ids(con, "SELECT comparison_id FROM comparisons"),
            "promotion": _ids(
                con,
                "SELECT CAST(history_id AS TEXT) FROM champion_history WHERE action='promote'",
            ),
        }
        con.close()
        return values
    except sqlite3.Error:
        return empty


def _ids(con: sqlite3.Connection, query: str) -> set[str]:
    try:
        return {str(row[0]) for row in con.execute(query).fetchall()}
    except sqlite3.Error:
        return set()
