#!/usr/bin/env python3
"""Run one orchestration backend and persist its exit status for later monitors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(
    path: Path, *, exit_code: int, clean_exit: bool, usage: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "exit_code": exit_code,
                "clean_exit": clean_exit,
                "usage": usage,
                "ended_at": _stamp(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--prompt-via", choices=("stdin", "argv"), required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("backend command is required after --")

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_path.read_bytes()
    try:
        with args.log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if args.prompt_via == "stdin" else subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            if args.prompt_via == "stdin" and process.stdin is not None:
                process.stdin.write(prompt)
                process.stdin.close()
            exit_code = process.wait()
    except OSError as exc:
        with args.log_path.open("ab") as log_handle:
            log_handle.write(f"orchestration child launch failed: {exc}\n".encode("utf-8"))
        exit_code = 127

    from autoresearch.orchestration.adapters import inspect_backend_exit

    observation = inspect_backend_exit(args.tool, args.log_path, exit_code=exit_code)
    _write_status(
        args.status_path,
        exit_code=exit_code,
        clean_exit=observation.clean_exit,
        usage=observation.usage,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
