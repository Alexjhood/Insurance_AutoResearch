#!/usr/bin/env python3
"""Codex Stop hook enforcing per-cycle research-log entries for bound runs.

This is intentionally Codex-specific wiring around the shared run-scope state:
the session must already be bound by ``run_scope_guard.py`` before this hook
does anything. Analyst and unbound build sessions remain unrestricted.

Fail-open policy mirrors the run-scope guard. Unexpected errors should never
trap a non-research thread, but a proven missing log entry exits 2 so Codex
continues and asks the agent to update the run's own ``RESEARCH_LOG.md``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _find_root(start: Path) -> Path:
    p = start
    for _ in range(8):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start


ROOT = _find_root(Path(__file__).resolve().parent)
TRACKS_DIR = ROOT / "artifacts" / "tracks"
SCOPE_DIR = TRACKS_DIR / ".scope"
LOG_PATH = SCOPE_DIR / "research-log-guard.log"

REQUIRED_FIELDS = (
    "Hypothesis",
    "Changes",
    "Outcome",
    "Metrics",
    "Interpretation",
    "Next",
)


def _log(message: str) -> None:
    try:
        SCOPE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {message}\n")
    except Exception:
        pass


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120] or "session"


def _scope_path(session_id: str) -> Path:
    return SCOPE_DIR / f"{_safe(session_id)}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_scope(session_id: str) -> dict[str, Any] | None:
    payload = _read_json(_scope_path(session_id))
    return payload if isinstance(payload, dict) else None


def _latest_session_state(run_dir: Path) -> dict[str, Any] | None:
    sessions_dir = run_dir / "sessions"
    latest_path = sessions_dir / "latest_session_id.txt"
    if latest_path.exists():
        session_id = latest_path.read_text(encoding="utf-8").strip()
        payload = _read_json(sessions_dir / session_id / "state.json")
        if isinstance(payload, dict):
            return payload

    states = sorted(sessions_dir.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for state_path in states:
        payload = _read_json(state_path)
        if isinstance(payload, dict):
            return payload
    return None


def _cycle_section(text: str, cycle: int) -> str | None:
    pattern = re.compile(
        rf"^#+\s*Cycle\s+{cycle}\b(?P<body>.*?)(?=^#+\s*Cycle\s+\d+\b|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(0)


def missing_log_requirements(log_text: str, current_cycle: int) -> list[str]:
    missing: list[str] = []
    for cycle in range(1, current_cycle + 1):
        section = _cycle_section(log_text, cycle)
        if section is None:
            missing.append(f"Cycle {cycle} section")
            continue

        absent = []
        for field in REQUIRED_FIELDS:
            field_pattern = re.compile(rf"\*\*{field.rstrip('s')}s?\*\*\s*:", re.IGNORECASE)
            if not field_pattern.search(section):
                absent.append(field)
        if absent:
            missing.append(f"Cycle {cycle} fields: {', '.join(absent)}")
    return missing


def evaluate(payload: dict[str, Any]) -> tuple[int, str]:
    if payload.get("hook_event_name") != "Stop":
        return 0, ""
    if os.environ.get("AUTORESEARCH_SCOPE", "").strip().lower() == "analyst":
        return 0, ""

    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        return 0, ""

    scope = _load_scope(session_id)
    if not scope or scope.get("mode") != "research":
        return 0, ""

    track = str(scope.get("track") or "").strip()
    run_id = str(scope.get("run_id") or "").strip()
    run_dir = Path(str(scope.get("run_dir") or "")) if scope.get("run_dir") else TRACKS_DIR / track / "runs" / run_id
    if not track or not run_id:
        return 0, ""

    state = _latest_session_state(run_dir)
    try:
        current_cycle = int((state or {}).get("current_cycle") or 0)
    except (TypeError, ValueError):
        return 0, ""
    if current_cycle <= 0:
        return 0, ""

    log_path = run_dir / "RESEARCH_LOG.md"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    missing = missing_log_requirements(log_text, current_cycle)
    if not missing:
        _log(f"session {session_id}: research log ok for {track}/{run_id} cycles=1..{current_cycle}")
        return 0, ""

    message = (
        f"Update {log_path} before finishing. Missing required research-log entries: "
        f"{'; '.join(missing)}. Each cycle needs Hypothesis, Changes, Outcome, "
        "Metrics, Interpretation, and Next, using only this run's artifacts."
    )
    _log(f"session {session_id}: missing research log for {track}/{run_id}: {'; '.join(missing)}")
    return 2, message


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        code, message = evaluate(payload)
    except Exception as exc:
        _log(f"ERROR: {exc!r}")
        return 0
    if code == 2 and message:
        sys.stderr.write(f"[research-log-guard] {message}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
