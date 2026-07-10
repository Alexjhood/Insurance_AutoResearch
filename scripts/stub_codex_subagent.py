#!/usr/bin/env python3
"""Zero-cost Codex-shaped backend that runs the scripted research stub."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> int:
    prompt = sys.stdin.read()
    stub = Path(__file__).with_name("stub_subagent.py")

    _emit({"type": "thread.started", "thread_id": "stub-codex-thread"})
    _emit({"type": "turn.started"})
    result = subprocess.run(
        [sys.executable, str(stub)],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.stdout:
        _emit(
            {
                "type": "item.completed",
                "item": {
                    "id": "stub-workflow",
                    "type": "agent_message",
                    "text": result.stdout,
                },
            }
        )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    if result.returncode == 0:
        _emit(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                },
            }
        )
    else:
        _emit({"type": "turn.failed", "error": {"message": "scripted stub failed"}})
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
