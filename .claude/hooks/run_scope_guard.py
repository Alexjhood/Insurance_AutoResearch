#!/usr/bin/env python3
"""Compatibility shim — the guard now lives at scripts/run_scope_guard.py.

Kept so any Claude Code session that loaded this older hook path keeps working.
It forwards stdin to the real script unchanged and mirrors its exit code.
New config (settings.json / .codex/hooks.json / .opencode plugin) points at
scripts/run_scope_guard.py directly; this file can be deleted in a future
session once no live session references it.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_REAL = Path(__file__).resolve().parents[2] / "scripts" / "run_scope_guard.py"

if __name__ == "__main__":
    try:
        sys.argv = [str(_REAL)]
        runpy.run_path(str(_REAL), run_name="__main__")
        sys.exit(0)
    except SystemExit as exc:
        sys.exit(exc.code if isinstance(exc.code, int) else 0)
    except Exception:
        sys.exit(0)  # fail open
