"""Incremental importers for Claude Code and Codex Desktop transcripts."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from autoresearch.telemetry import store
from autoresearch.telemetry.usage_report import write_usage_report


_SIGNALS = {
    "repair_requested": re.compile(
        r"waiting_for_repair|ExperimentNeedsRepair|repair_request(?:_\d+)?\.json",
        re.IGNORECASE,
    ),
    "preflight_failed": re.compile(
        r"PreflightFailed|preflight (?:smoke[- ]?test )?failed", re.IGNORECASE
    ),
    "compute_budget_exceeded": re.compile(
        r"ComputeBudgetExceeded|compute budget exceeded", re.IGNORECASE
    ),
}
_CODEX_EXIT = re.compile(r"Process exited with code\s+(-?\d+)")


def sync_opencode_session(
    *,
    run_dir: Path,
    native_session_id: str,
) -> dict[str, Any]:
    """Import an OpenCode session from the desktop app's SQLite database."""
    db_path = _find_opencode_db()
    if db_path is None:
        raise FileNotFoundError(
            "OpenCode database not found at ~/.local/share/opencode/opencode.db"
        )

    oc_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    oc_con.row_factory = sqlite3.Row
    try:
        session_row = oc_con.execute(
            "SELECT * FROM session WHERE id=?", (native_session_id,)
        ).fetchone()
        if not session_row:
            raise ValueError(f"OpenCode session {native_session_id!r} not found in database")
        message_rows = oc_con.execute(
            "SELECT * FROM message WHERE session_id=? ORDER BY time_created",
            (native_session_id,),
        ).fetchall()
    finally:
        oc_con.close()

    model_json: dict[str, Any] = {}
    try:
        model_json = json.loads(session_row["model"] or "{}") or {}
    except (json.JSONDecodeError, TypeError):
        pass
    model_id = str(model_json.get("id") or "")
    provider_id = str(model_json.get("providerID") or "opencode")
    variant = str(model_json.get("variant") or "")
    model_str = f"{provider_id}/{model_id}" if provider_id and model_id else model_id
    effort = variant if variant and variant.lower() != "default" else None

    run_dir = Path(run_dir)
    con = store.connect(run_dir)
    try:
        metadata: dict[str, Any] = {
            "cwd": session_row["directory"] if "directory" in session_row.keys() else None,
            "model": model_str,
            "effort": effort,
            "started_at": _ms_to_iso(session_row["time_created"]),
            "last_event_at": _ms_to_iso(session_row["time_updated"]),
        }
        sess_key = store.upsert_session(
            con,
            surface="opencode",
            native_session_id=native_session_id,
            transcript_path=db_path,
            metadata=metadata,
        )
        imported = _import_opencode(con, sess_key, model_str, effort, message_rows)
        store.refresh_turn_aggregates(con, sess_key)
        store.refresh_session_status(con, sess_key, finalize=True)
        store.update_cursor(
            con,
            source_path=db_path,
            surface="opencode",
            native_session_id=native_session_id,
            byte_offset=imported,
            source_size=db_path.stat().st_size,
            source_mtime_ns=db_path.stat().st_mtime_ns,
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    write_usage_report(run_dir)
    return {
        "status": "ok",
        "surface": "opencode",
        "native_session_id": native_session_id,
        "records_imported": imported,
        "run_dir": str(run_dir),
    }


def sync_session(
    *,
    run_dir: Path,
    surface: str,
    native_session_id: str,
    transcript_path: Path,
    finalize_turn: bool = False,
    rebuild: bool = False,
    _include_subagents: bool = True,
) -> dict[str, Any]:
    """Import new complete JSONL records for one native desktop session."""

    surface = surface.lower().strip()
    if surface not in {"claude", "codex"}:
        raise ValueError(f"Unsupported telemetry surface: {surface}")
    transcript_path = Path(transcript_path).expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(transcript_path)

    con = store.connect(run_dir)
    result: dict[str, Any]
    try:
        cursor = store.cursor_for(con, transcript_path)
        stat = transcript_path.stat()
        offset = 0 if rebuild else (int(cursor["byte_offset"]) if cursor else 0)
        if offset > stat.st_size or (
            cursor and int(cursor["source_mtime_ns"]) > stat.st_mtime_ns
        ):
            offset = 0

        records, new_offset = _read_complete_jsonl(transcript_path, offset)
        metadata = _session_metadata(surface, records)
        session_key = store.upsert_session(
            con,
            surface=surface,
            native_session_id=native_session_id,
            transcript_path=transcript_path,
            metadata=metadata,
        )
        if surface == "codex":
            imported = _import_codex(con, session_key, native_session_id, records)
        else:
            imported = _import_claude(con, session_key, native_session_id, records)

        if finalize_turn and surface != "codex":
            latest = store.latest_turn(con, session_key)
            if latest:
                completed_at = metadata.get("last_event_at") or _last_timestamp(records)
                store.complete_turn(con, latest, completed_at=completed_at)

        store.refresh_turn_aggregates(con, session_key)
        store.refresh_session_status(
            con,
            session_key,
            finalize=finalize_turn or surface == "codex",
        )
        store.link_workflow_events(con)
        store.update_cursor(
            con,
            source_path=transcript_path,
            surface=surface,
            native_session_id=native_session_id,
            byte_offset=new_offset,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
        )
        con.commit()
        result = {
            "status": "ok",
            "surface": surface,
            "native_session_id": native_session_id,
            "transcript": str(transcript_path),
            "records_read": len(records),
            "records_imported": imported,
            "byte_offset": new_offset,
            "source_size": stat.st_size,
            "run_dir": str(run_dir),
        }
    except Exception as exc:
        con.rollback()
        try:
            stat = transcript_path.stat()
            store.update_cursor(
                con,
                source_path=transcript_path,
                surface=surface,
                native_session_id=native_session_id,
                byte_offset=0,
                source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                last_error=type(exc).__name__,
            )
            con.commit()
        except Exception:
            pass
        raise
    finally:
        con.close()

    write_usage_report(run_dir)
    if surface == "claude" and _include_subagents:
        imported_children = 0
        child_errors = 0
        for child_path in _claude_subagent_transcripts(transcript_path):
            agent_id = child_path.stem.removeprefix("agent-")
            try:
                sync_session(
                    run_dir=run_dir,
                    surface="claude",
                    native_session_id=f"{native_session_id}:subagent:{agent_id}",
                    transcript_path=child_path,
                    finalize_turn=True,
                    rebuild=rebuild,
                    _include_subagents=False,
                )
                imported_children += 1
            except Exception:
                child_errors += 1
        result["subagent_transcripts"] = imported_children
        result["subagent_errors"] = child_errors
    return result


def find_transcript(
    *,
    surface: str,
    native_session_id: str,
    cwd: Path | None = None,
    explicit_path: str | Path | None = None,
) -> Path | None:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path.resolve()
    if surface == "codex":
        return _find_codex_transcript(native_session_id)
    if surface == "claude":
        return _find_claude_transcript(native_session_id, cwd)
    return None


def _find_codex_transcript(native_session_id: str) -> Path | None:
    state_path = Path.home() / ".codex" / "state_5.sqlite"
    if state_path.exists():
        try:
            con = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
            row = con.execute(
                "SELECT rollout_path FROM threads WHERE id=?", (native_session_id,)
            ).fetchone()
            con.close()
            if row and row[0] and Path(row[0]).exists():
                return Path(row[0]).resolve()
        except sqlite3.Error:
            pass
    for base in (Path.home() / ".codex" / "sessions", Path.home() / ".codex" / "archived_sessions"):
        if not base.exists():
            continue
        matches = list(base.rglob(f"*{native_session_id}*.jsonl"))
        if matches:
            return matches[-1].resolve()
    return None


def _find_claude_transcript(native_session_id: str, cwd: Path | None) -> Path | None:
    projects = Path.home() / ".claude" / "projects"
    if cwd:
        encoded = str(cwd.resolve()).replace("/", "-")
        candidate = projects / encoded / f"{native_session_id}.jsonl"
        if candidate.exists():
            return candidate.resolve()
    matches = list(projects.glob(f"*/{native_session_id}.jsonl")) if projects.exists() else []
    return matches[-1].resolve() if matches else None


def _claude_subagent_transcripts(parent_transcript: Path) -> list[Path]:
    base = parent_transcript.parent / parent_transcript.stem / "subagents"
    return sorted(base.glob("agent-*.jsonl")) if base.exists() else []


def _read_complete_jsonl(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(offset)
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                return records, handle.tell()
            if not line.endswith(b"\n"):
                return records, line_start
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                records.append(value)


def _import_codex(
    con: sqlite3.Connection,
    session_key: str,
    native_session_id: str,
    records: list[dict[str, Any]],
) -> int:
    current_turn = store.latest_turn(con, session_key)
    imported = 0
    pending_output_chars = 0
    for record in records:
        timestamp = _str(record.get("timestamp"))
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        record_type = record.get("type")
        payload_type = payload.get("type")

        if record_type == "event_msg" and payload_type == "task_started":
            native_turn_id = _str(payload.get("turn_id"))
            current_turn = store.ensure_turn(
                con,
                session_key_value=session_key,
                native_turn_id=native_turn_id,
                started_at=timestamp,
                turn_key_value=f"{session_key}:turn:{native_turn_id}",
            )
            imported += 1
            continue

        if record_type == "turn_context":
            if current_turn:
                store.update_turn_context(
                    con,
                    current_turn,
                    model=_str(payload.get("model")),
                    effort=_str(payload.get("effort") or payload.get("reasoning_effort")),
                )
            imported += 1
            continue

        if record_type == "event_msg" and payload_type == "task_complete":
            native_turn_id = _str(payload.get("turn_id"))
            current_turn = store.ensure_turn(
                con,
                session_key_value=session_key,
                native_turn_id=native_turn_id,
                started_at=None,
                turn_key_value=f"{session_key}:turn:{native_turn_id}",
            )
            store.complete_turn(
                con,
                current_turn,
                completed_at=timestamp,
                duration_ms=_float(payload.get("duration_ms")),
                time_to_first_token_ms=_float(payload.get("time_to_first_token_ms")),
            )
            imported += 1
            continue

        if record_type == "event_msg" and payload_type == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            usage = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
            total_usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
            if not usage:
                continue
            if current_turn is None:
                current_turn = store.ensure_turn(
                    con,
                    session_key_value=session_key,
                    native_turn_id=None,
                    started_at=timestamp,
                )
            fingerprint = json.dumps(total_usage or usage, sort_keys=True, separators=(",", ":"))
            store.upsert_model_call(
                con,
                {
                    "call_key": f"{session_key}:usage:{_hash(fingerprint)}",
                    "session_key": session_key,
                    "turn_key": current_turn,
                    "occurred_at": timestamp,
                    "input_tokens": _int(usage.get("input_tokens")),
                    "cached_input_tokens": _int(usage.get("cached_input_tokens")),
                    "uncached_input_tokens": max(
                        _int(usage.get("input_tokens")) - _int(usage.get("cached_input_tokens")), 0
                    ),
                    "output_tokens": _int(usage.get("output_tokens")),
                    "reasoning_tokens": _int(usage.get("reasoning_output_tokens")),
                    "total_tokens": _int(usage.get("total_tokens")),
                    "output_chars": pending_output_chars,
                },
            )
            pending_output_chars = 0
            imported += 1
            continue

        if record_type != "response_item":
            continue
        if payload_type == "message" and payload.get("role") == "assistant":
            pending_output_chars += _codex_assistant_text_chars(payload.get("content"))
            imported += 1
            continue
        call_id = _str(payload.get("call_id"))
        if not call_id:
            continue
        call_key = f"{session_key}:tool:{call_id}"
        if payload_type in {"function_call", "custom_tool_call"}:
            arguments = payload.get("arguments")
            if arguments is None:
                arguments = payload.get("input")
            name = _str(payload.get("name")) or "tool"
            store.upsert_tool_call(
                con,
                {
                    "call_key": call_key,
                    "session_key": session_key,
                    "turn_key": current_turn,
                    "provider_call_id": call_id,
                    "name": name,
                    "detail": _tool_detail(name, arguments),
                    "started_at": timestamp,
                    "status": _str(payload.get("status")) or "started",
                    "input_bytes": _json_bytes(arguments),
                },
            )
            imported += 1
        elif payload_type in {"function_call_output", "custom_tool_call_output"}:
            output = payload.get("output")
            output_text = _text(output)
            exit_match = _CODEX_EXIT.search(output_text)
            success = int(exit_match.group(1)) == 0 if exit_match else None
            started = con.execute(
                "SELECT started_at, name, detail FROM llm_tool_calls WHERE call_key=?",
                (call_key,),
            ).fetchone()
            store.upsert_tool_call(
                con,
                {
                    "call_key": call_key,
                    "session_key": session_key,
                    "turn_key": current_turn,
                    "provider_call_id": call_id,
                    "name": "tool",
                    "completed_at": timestamp,
                    "duration_ms": _duration_ms(started["started_at"], timestamp) if started else None,
                    "status": "completed" if success is not False else "failed",
                    "success": success,
                    "output_bytes": len(output_text.encode("utf-8")),
                    "error_type": "nonzero_exit" if success is False else None,
                },
            )
            store.record_signals(
                con,
                session_key_value=session_key,
                turn_key_value=current_turn,
                tool_call_key=call_key,
                signal_types=_signal_types(output_text)
                if _should_extract_signals(started, success)
                else [],
                created_at=timestamp,
            )
            imported += 1
    return imported


def _import_claude(
    con: sqlite3.Connection,
    session_key: str,
    native_session_id: str,
    records: list[dict[str, Any]],
) -> int:
    current_turn = store.latest_turn(con, session_key)
    imported = 0
    for record in records:
        timestamp = _str(record.get("timestamp"))
        record_type = record.get("type")
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        content = message.get("content")

        if record_type == "user" and _is_external_prompt(record):
            native_turn_id = _str(record.get("uuid")) or _hash(f"{timestamp}:{len(records)}")
            current_turn = store.ensure_turn(
                con,
                session_key_value=session_key,
                native_turn_id=native_turn_id,
                started_at=timestamp,
                turn_key_value=f"{session_key}:turn:{native_turn_id}",
            )
            imported += 1
            continue

        if record_type == "assistant":
            request_id = _str(record.get("requestId"))
            usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
            if current_turn is None:
                current_turn = store.ensure_turn(
                    con,
                    session_key_value=session_key,
                    native_turn_id=None,
                    started_at=timestamp,
                )
            model_name = _str(message.get("model"))
            if model_name or any(
                isinstance(b, dict) and b.get("type") == "thinking"
                for b in (content if isinstance(content, list) else [])
            ):
                has_thinking = any(
                    isinstance(b, dict) and b.get("type") == "thinking"
                    for b in (content if isinstance(content, list) else [])
                )
                store.update_turn_context(
                    con,
                    current_turn,
                    model=model_name,
                    effort="thinking" if has_thinking else None,
                )
            if request_id and usage:
                uncached = _int(usage.get("input_tokens"))
                cached = _int(usage.get("cache_read_input_tokens"))
                created = _int(usage.get("cache_creation_input_tokens"))
                total_input = uncached + cached + created
                output_tokens = _int(usage.get("output_tokens"))
                store.upsert_model_call(
                    con,
                    {
                        "call_key": f"{session_key}:request:{request_id}",
                        "session_key": session_key,
                        "turn_key": current_turn,
                        "provider_request_id": request_id,
                        "occurred_at": timestamp,
                        "model": _str(message.get("model")),
                        "input_tokens": total_input,
                        "cached_input_tokens": cached,
                        "cache_creation_input_tokens": created,
                        "uncached_input_tokens": uncached,
                        "output_tokens": output_tokens,
                        "total_tokens": total_input + output_tokens,
                        "output_chars": _assistant_text_chars(content),
                    },
                )
                imported += 1
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = _str(block.get("id"))
                if not call_id:
                    continue
                name = _str(block.get("name")) or "tool"
                arguments = block.get("input")
                store.upsert_tool_call(
                    con,
                    {
                        "call_key": f"{session_key}:tool:{call_id}",
                        "session_key": session_key,
                        "turn_key": current_turn,
                        "provider_call_id": call_id,
                        "name": name,
                        "detail": _tool_detail(name, arguments),
                        "started_at": timestamp,
                        "status": "started",
                        "input_bytes": _json_bytes(arguments),
                    },
                )
                imported += 1
            continue

        if record_type == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = _str(block.get("tool_use_id"))
                if not call_id:
                    continue
                call_key = f"{session_key}:tool:{call_id}"
                output_text = _text(block.get("content"))
                structured = record.get("toolUseResult")
                stdout = structured.get("stdout") if isinstance(structured, dict) else None
                stderr = structured.get("stderr") if isinstance(structured, dict) else None
                is_error = block.get("is_error")
                if not isinstance(is_error, bool) and isinstance(structured, dict):
                    is_error = bool(structured.get("interrupted"))
                started = con.execute(
                    "SELECT started_at, name, detail FROM llm_tool_calls WHERE call_key=?",
                    (call_key,),
                ).fetchone()
                store.upsert_tool_call(
                    con,
                    {
                        "call_key": call_key,
                        "session_key": session_key,
                        "turn_key": current_turn,
                        "provider_call_id": call_id,
                        "name": "tool",
                        "completed_at": timestamp,
                        "duration_ms": _duration_ms(started["started_at"], timestamp) if started else None,
                        "status": "failed" if is_error else "completed",
                        "success": not is_error,
                        "output_bytes": _json_bytes(block.get("content")),
                        "stdout_bytes": len(_text(stdout).encode("utf-8")),
                        "stderr_bytes": len(_text(stderr).encode("utf-8")),
                        "error_type": "tool_error" if is_error else None,
                    },
                )
                store.record_signals(
                    con,
                    session_key_value=session_key,
                    turn_key_value=current_turn,
                    tool_call_key=call_key,
                    signal_types=_signal_types(output_text + _text(stderr))
                    if _should_extract_signals(started, not is_error)
                    else [],
                    created_at=timestamp,
                )
                imported += 1
    return imported


def _session_metadata(surface: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"last_event_at": _last_timestamp(records)}
    if surface == "codex":
        for record in records:
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            if record.get("type") == "session_meta":
                metadata.update(
                    {
                        "cwd": payload.get("cwd"),
                        "source_version": payload.get("cli_version"),
                        "started_at": payload.get("timestamp") or record.get("timestamp"),
                    }
                )
            elif record.get("type") == "turn_context":
                metadata["model"] = payload.get("model") or metadata.get("model")
                metadata["effort"] = (
                    payload.get("effort")
                    or payload.get("reasoning_effort")
                    or metadata.get("effort")
                )
    else:
        for record in records:
            metadata["cwd"] = metadata.get("cwd") or record.get("cwd")
            metadata["source_version"] = metadata.get("source_version") or record.get("version")
            metadata["started_at"] = metadata.get("started_at") or record.get("timestamp")
            message = record.get("message") if isinstance(record.get("message"), dict) else {}
            metadata["model"] = metadata.get("model") or message.get("model")
    return metadata


def _is_external_prompt(record: dict[str, Any]) -> bool:
    message = record.get("message") if isinstance(record.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return any(not isinstance(item, dict) or item.get("type") != "tool_result" for item in content)


def _assistant_text_chars(content: Any) -> int:
    if not isinstance(content, list):
        return len(content) if isinstance(content, str) else 0
    return sum(
        len(str(block.get("text") or ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _codex_assistant_text_chars(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        len(str(block.get("text") or ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "output_text"
    )


def _tool_detail(name: str, arguments: Any) -> str:
    if name in {"apply_patch", "multi_tool_use.parallel"}:
        return name
    data = arguments
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments)
        except json.JSONDecodeError:
            data = {"command": arguments}
    data = data if isinstance(data, dict) else {}
    command = data.get("cmd") or data.get("command")
    if isinstance(command, str):
        subcommand = _autoresearch_subcommand(command)
        if subcommand:
            return f"autoresearch {subcommand}"
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if tokens:
            executable = Path(tokens[0]).name
            if executable in {"python", "python3"} and len(tokens) > 1:
                return f"{executable} {Path(tokens[1]).name}"
            return executable[:120]
    path = data.get("file_path") or data.get("filePath") or data.get("path")
    if isinstance(path, str):
        return f"{name} {Path(path).name}"[:160]
    return name[:160]


def _autoresearch_subcommand(command: str) -> str | None:
    for segment in re.split(r"(?:&&|\|\||[;|])", command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        index = 0
        while index < len(tokens) and "=" in tokens[index] and not tokens[index].startswith("-"):
            index += 1
        if index >= len(tokens):
            continue
        executable = Path(tokens[index]).name
        if executable == "autoresearch":
            args = tokens[index + 1 :]
        elif (
            executable.startswith("python")
            and index + 2 < len(tokens)
            and tokens[index + 1] == "-m"
            and tokens[index + 2] in {"autoresearch.cli", "src.autoresearch.cli"}
        ):
            args = tokens[index + 3 :]
        else:
            continue
        skip_next = False
        for token in args:
            if skip_next:
                skip_next = False
                continue
            if token in {"--config", "--track", "--run-id", "--target-mode"}:
                skip_next = True
                continue
            if token == "--new-run" or token.startswith("--"):
                continue
            return token
    return None


def _signal_types(text: str) -> list[str]:
    return [name for name, pattern in _SIGNALS.items() if pattern.search(text)]


def _should_extract_signals(tool_row: Any, success: bool | None) -> bool:
    if success is False:
        return True
    if tool_row is None:
        return False
    return str(tool_row["detail"] or "").startswith("autoresearch ")


def _last_timestamp(records: list[dict[str, Any]]) -> str | None:
    for record in reversed(records):
        timestamp = record.get("timestamp")
        if isinstance(timestamp, str):
            return timestamp
    return None


def _duration_ms(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return max((datetime.fromisoformat(end.replace("Z", "+00:00")) -
                    datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds() * 1000, 0)
    except ValueError:
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _json_bytes(value: Any) -> int:
    return len(_text(value).encode("utf-8"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _import_opencode(
    con: sqlite3.Connection,
    session_key: str,
    model_str: str,
    effort: str | None,
    message_rows: list,
) -> int:
    """Process OpenCode message rows into telemetry tables."""
    current_turn: str | None = store.latest_turn(con, session_key)
    imported = 0

    for msg_row in message_rows:
        try:
            data: dict[str, Any] = json.loads(msg_row["data"] or "{}") or {}
        except (json.JSONDecodeError, TypeError):
            continue

        role = data.get("role")
        time_ms = (data.get("time") or {}).get("created") or msg_row["time_created"]
        timestamp = _ms_to_iso(time_ms)
        msg_id = str(msg_row["id"])

        if role == "user":
            current_turn = store.ensure_turn(
                con,
                session_key_value=session_key,
                native_turn_id=msg_id,
                started_at=timestamp,
                turn_key_value=f"{session_key}:turn:{msg_id}",
            )
            store.update_turn_context(con, current_turn, model=model_str, effort=effort)
            imported += 1
            continue

        if role != "assistant":
            continue

        if current_turn is None:
            current_turn = store.ensure_turn(
                con,
                session_key_value=session_key,
                native_turn_id=None,
                started_at=timestamp,
            )
            store.update_turn_context(con, current_turn, model=model_str, effort=effort)

        tokens: dict[str, Any] = data.get("tokens") or {}
        cache: dict[str, Any] = tokens.get("cache") or {}
        msg_provider = _str(data.get("providerID")) or ""
        msg_model_id = _str(data.get("modelID")) or ""
        full_model = (
            f"{msg_provider}/{msg_model_id}"
            if msg_provider and msg_model_id
            else msg_model_id or model_str
        )
        input_tok = _int(tokens.get("input"))
        output_tok = _int(tokens.get("output"))
        reasoning_tok = _int(tokens.get("reasoning"))
        cache_read = _int(cache.get("read"))
        cache_write = _int(cache.get("write"))
        total_tok = _int(tokens.get("total")) or (input_tok + output_tok)
        cost = data.get("cost")

        store.upsert_model_call(
            con,
            {
                "call_key": f"{session_key}:msg:{msg_id}",
                "session_key": session_key,
                "turn_key": current_turn,
                "occurred_at": timestamp,
                "model": full_model,
                "input_tokens": input_tok,
                "cached_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
                "uncached_input_tokens": max(input_tok - cache_read, 0),
                "output_tokens": output_tok,
                "reasoning_tokens": reasoning_tok,
                "total_tokens": total_tok,
                "provider_cost_usd": float(cost) if cost is not None else None,
            },
        )
        store.update_turn_context(con, current_turn, model=full_model, effort=effort)
        imported += 1

    return imported


def _find_opencode_db() -> Path | None:
    path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    return path if path.exists() else None


def _ms_to_iso(ms: Any) -> str | None:
    from datetime import datetime, timezone
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
