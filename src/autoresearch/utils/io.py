"""Small IO helpers used across deterministic setup steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    """Write stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))
