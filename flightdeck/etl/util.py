"""Small stdlib-only helpers shared across readers (SPEC §8: stdlib + sqlite3)."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


class Warnings:
    """Accumulator threaded through the build for defensive degradation."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def add(self, message: str) -> None:
        self._items.append(message)

    def extend(self, messages: list[str]) -> None:
        self._items.extend(messages)

    @property
    def items(self) -> list[str]:
        return list(self._items)


def load_json(path: Path, warnings: Warnings, *, label: str | None = None) -> Optional[Any]:
    """Load JSON, returning ``None`` and warning on any error (never raises)."""
    label = label or str(path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        warnings.add(f"missing file: {label}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        warnings.add(f"unreadable json {label}: {exc}")
        return None


@contextmanager
def ro_connect(path: Path, warnings: Warnings) -> Iterator[Optional[sqlite3.Connection]]:
    """Open a sqlite DB read-only via URI mode (DATA.md §3, quirk 5).

    Yields ``None`` (plus a warning) if the file is missing or unopenable, so
    callers can degrade to nulls on integrity-locked/partial DBs (quirk 6).
    """
    conn: Optional[sqlite3.Connection] = None
    if not path.exists():
        warnings.add(f"missing sqlite: {path}")
        yield None
        return
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        yield conn
    except sqlite3.Error as exc:
        warnings.add(f"cannot open sqlite {path}: {exc}")
        yield None
    finally:
        if conn is not None:
            conn.close()


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names for ``table`` via PRAGMA table_info; [] if it doesn't exist."""
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    except sqlite3.Error:
        return []


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def safe_query(
    conn: sqlite3.Connection, sql: str, params: tuple = (), *,
    warnings: Warnings | None = None, label: str = "",
) -> list[sqlite3.Row]:
    """Run a query, returning [] (and optionally warning) on any sqlite error."""
    try:
        return list(conn.execute(sql, params))
    except sqlite3.Error as exc:
        if warnings is not None:
            warnings.add(f"query failed {label or sql[:40]}: {exc}")
        return []


def loads_or_none(raw: Any) -> Optional[Any]:
    """json.loads a possibly-null / already-parsed value; None on failure."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_TS_CLEAN = re.compile(r"[Zz]$")


def parse_ts(value: Any) -> Optional[datetime]:
    """Best-effort parse of the timestamp shapes seen across sources.

    Handles ``2026-07-11 16:52:17`` (sqlite), ``2026-07-11T17:08:19Z`` and
    fractional ISO forms. Used *only* for ordering — the ETL never reformats
    the stored strings it emits (DATA.md general rules).
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    candidate = _TS_CLEAN.sub("+00:00", text)
    for parser in (datetime.fromisoformat,):
        try:
            dt = parser(candidate)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    # Last resort: space-separated without tz.
    try:
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def ts_key(value: Any) -> tuple[int, float]:
    """Sort key that pushes unparseable/None timestamps to the end, stably."""
    dt = parse_ts(value)
    if dt is None:
        return (1, 0.0)
    return (0, dt.timestamp())


def cache_hit_rate(cached_input: int, total_input: int) -> Optional[float]:
    if not total_input:
        return None
    return cached_input / total_input
