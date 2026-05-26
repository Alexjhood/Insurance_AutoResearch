"""Shared helpers for the experiment registry modules."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


def loads(s: str | None, default: Any = None) -> Any:
    if s is None:
        return default
    return json.loads(s)
