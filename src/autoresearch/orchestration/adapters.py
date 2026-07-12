"""Tool-specific interpretation of headless orchestration backend output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExitObservation:
    """What the backend output says about one completed child process."""

    clean_exit: bool
    usage: dict[str, Any]
    terminal_event: str | None = None
    malformed_lines: int = 0


def inspect_backend_exit(
    tool: str, log_path: Path, *, exit_code: int
) -> ExitObservation:
    """Interpret a backend log after process exit.

    Process status remains authoritative for every backend. Codex adds one
    stronger condition: a zero return code is clean only when its JSONL stream
    reaches ``turn.completed``. This catches truncated streams and launcher
    failures that otherwise look successful to a detached monitor.
    """

    if tool == "codex":
        return _inspect_codex_jsonl(log_path, exit_code=exit_code)
    if tool == "claude":
        return _inspect_claude_jsonl(log_path, exit_code=exit_code)
    return ExitObservation(clean_exit=exit_code == 0, usage={})


def _inspect_claude_jsonl(log_path: Path, *, exit_code: int) -> ExitObservation:
    """Aggregate Claude stream-json message usage into canonical counters."""

    usage: dict[str, Any] = {}
    malformed_lines = 0
    terminal_event: str | None = None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or "")
        if event_type == "result":
            terminal_event = event_type
        if event_type != "assistant":
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        raw = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        uncached = int(raw.get("input_tokens") or 0) + int(
            raw.get("cache_creation_input_tokens") or 0
        )
        cached = int(raw.get("cache_read_input_tokens") or 0)
        usage["input_tokens"] = int(usage.get("input_tokens") or 0) + uncached + cached
        usage["cached_input_tokens"] = int(usage.get("cached_input_tokens") or 0) + cached
        usage["output_tokens"] = int(usage.get("output_tokens") or 0) + int(
            raw.get("output_tokens") or 0
        )

    return ExitObservation(
        clean_exit=exit_code == 0,
        usage=usage,
        terminal_event=terminal_event,
        malformed_lines=malformed_lines,
    )


def _inspect_codex_jsonl(log_path: Path, *, exit_code: int) -> ExitObservation:
    usage: dict[str, Any] = {}
    completed_turns = 0
    terminal_event: str | None = None
    malformed_lines = 0

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (TypeError, json.JSONDecodeError):
            malformed_lines += 1
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        if event_type == "turn.completed":
            terminal_event = event_type
            completed_turns += 1
            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                _merge_numeric_usage(usage, raw_usage)
        elif event_type in {"turn.failed", "error"}:
            terminal_event = str(event_type)

    if completed_turns:
        usage["completed_turns"] = completed_turns
    return ExitObservation(
        clean_exit=exit_code == 0 and terminal_event == "turn.completed",
        usage=usage,
        terminal_event=terminal_event,
        malformed_lines=malformed_lines,
    )


def _merge_numeric_usage(total: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Sum numeric usage leaves while preserving their provider nesting."""

    for key, value in incoming.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict):
            nested = total.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_numeric_usage(nested, value)
