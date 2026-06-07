#!/usr/bin/env python3
"""Fail-open hook and manual entrypoint for desktop agent telemetry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch.telemetry.importer import find_transcript, sync_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True, choices=("claude", "codex"))
    parser.add_argument("--session-id")
    parser.add_argument("--transcript-path")
    parser.add_argument("--track")
    parser.add_argument("--run-id")
    parser.add_argument("--finalize-turn", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--deferred", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--settle-timeout", type=float, default=8.0, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        hook_payload = _stdin_payload()
        session_id = args.session_id or hook_payload.get("session_id") or hook_payload.get("sessionId")
        if not session_id:
            return _finish(args, {"status": "skipped", "reason": "missing_session_id"})

        scope = _scope_for(str(session_id))
        track = args.track or scope.get("track")
        run_id = args.run_id or scope.get("run_id")
        if scope.get("mode") != "research" or not track or not run_id:
            return _finish(args, {"status": "skipped", "reason": "session_not_bound"})

        run_dir = ROOT / "artifacts" / "tracks" / str(track) / "runs" / str(run_id)
        transcript = find_transcript(
            surface=args.surface,
            native_session_id=str(session_id),
            cwd=Path(hook_payload.get("cwd") or ROOT),
            explicit_path=args.transcript_path
            or hook_payload.get("transcript_path")
            or hook_payload.get("transcriptPath"),
        )
        if not transcript:
            return _finish(args, {"status": "skipped", "reason": "transcript_not_found"})

        is_stop_hook = hook_payload.get("hook_event_name") == "Stop"
        if is_stop_hook and not args.deferred:
            _spawn_deferred_import(
                surface=args.surface,
                session_id=str(session_id),
                track=str(track),
                run_id=str(run_id),
                transcript=transcript,
                settle_timeout=args.settle_timeout,
            )
            return _finish(args, {"status": "deferred", "reason": "waiting_for_final_record"})

        if args.deferred:
            _wait_for_transcript_settle(
                transcript,
                surface=args.surface,
                timeout=args.settle_timeout,
            )

        result = sync_session(
            run_dir=run_dir,
            surface=args.surface,
            native_session_id=str(session_id),
            transcript_path=transcript,
            finalize_turn=args.finalize_turn or is_stop_hook,
            rebuild=args.rebuild,
        )
        return _finish(args, result)
    except Exception as exc:
        return _finish(
            args,
            {"status": "error", "error_type": type(exc).__name__, "detail": str(exc)[:300]},
            failed=True,
        )


def _stdin_payload() -> dict:
    if sys.stdin.isatty():
        return {}
    try:
        value = json.loads(sys.stdin.read() or "{}")
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _scope_for(session_id: str) -> dict:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)[:120]
    path = ROOT / "artifacts" / "tracks" / ".scope" / f"{safe}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _finish(args: argparse.Namespace, result: dict, *, failed: bool = False) -> int:
    if sys.stdout.isatty() or args.strict:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failed and args.strict else 0


def _spawn_deferred_import(
    *,
    surface: str,
    session_id: str,
    track: str,
    run_id: str,
    transcript: Path,
    settle_timeout: float,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--surface",
        surface,
        "--session-id",
        session_id,
        "--track",
        track,
        "--run-id",
        run_id,
        "--transcript-path",
        str(transcript),
        "--finalize-turn",
        "--deferred",
        "--settle-timeout",
        str(settle_timeout),
    ]
    subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def _wait_for_transcript_settle(
    path: Path,
    *,
    surface: str,
    timeout: float,
    poll_interval: float = 0.2,
) -> None:
    deadline = time.monotonic() + max(timeout, poll_interval)
    last_size = -1
    stable_checks = 0
    while time.monotonic() < deadline:
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        stable_checks = stable_checks + 1 if size == last_size else 0
        last_size = size
        complete = surface != "codex" or _codex_latest_turn_complete(path)
        if complete and stable_checks >= 2:
            return
        time.sleep(poll_interval)


def _codex_latest_turn_complete(path: Path) -> bool:
    latest_started: str | None = None
    latest_completed: str | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict) or record.get("type") != "event_msg":
                    continue
                if payload.get("type") == "task_started":
                    latest_started = str(payload.get("turn_id") or "")
                elif payload.get("type") == "task_complete":
                    latest_completed = str(payload.get("turn_id") or "")
    except OSError:
        return False
    return bool(latest_started and latest_started == latest_completed)


if __name__ == "__main__":
    raise SystemExit(main())
