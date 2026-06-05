"""Orchestrator configuration — reads from env with sensible defaults."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


# Repo root (two levels up from this file: console/backend/ → console/ → repo)
REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "tracks"

# Where orchestrator.db lives
ORCHESTRATOR_DB = Path(os.environ.get("AUTORESEARCH_CONSOLE_DB", REPO_ROOT / "console" / "orchestrator.db"))

# ── Agent binary resolution ──────────────────────────────────────────────────

def _claude_app_bundles() -> list[str]:
    """Glob all installed desktop-app claude versions, newest first.

    Avoids hardcoding a version like 2.1.161 that breaks on update.
    """
    base = Path.home() / "Library/Application Support/Claude/claude-code"
    if not base.is_dir():
        return []
    bundles = sorted(
        base.glob("*/claude.app/Contents/MacOS/claude"),
        key=lambda p: p.parent.parent.parent.parent.name,
        reverse=True,
    )
    return [str(p) for p in bundles]


_CLAUDE_CANDIDATES = [
    os.environ.get("CLAUDE_BIN", ""),
    shutil.which("claude") or "",            # npm global (preferred — has CLI auth)
    *_claude_app_bundles(),                  # desktop app bundles, newest first
    str(Path.home() / ".npm-global/bin/claude"),
    "/usr/local/bin/claude",
    "/opt/homebrew/bin/claude",
]


def _find_binary(candidates: list[str]) -> str | None:
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        # Resolve symlinks — catches dangling symlinks like the codex Cask install
        try:
            resolved = p.resolve(strict=True)
            if resolved.is_file():
                return str(p)
        except (OSError, RuntimeError):
            continue
    return None


CLAUDE_BIN: str | None = _find_binary(_CLAUDE_CANDIDATES)

_CODEX_CANDIDATES = [
    os.environ.get("CODEX_BIN", ""),
    shutil.which("codex") or "",
    "/opt/homebrew/bin/codex",
    str(Path.home() / ".npm-global/bin/codex"),
]
CODEX_BIN: str | None = _find_binary(_CODEX_CANDIDATES)

_OPENCODE_CANDIDATES = [
    os.environ.get("OPENCODE_BIN", ""),
    shutil.which("opencode") or "",
    "/opt/homebrew/bin/opencode",
]
OPENCODE_BIN: str | None = _find_binary(_OPENCODE_CANDIDATES)

# The autoresearch CLI — prefer the project venv, fall back to PATH.
_AUTORESEARCH_CANDIDATES = [
    os.environ.get("AUTORESEARCH_BIN", ""),
    str(REPO_ROOT / ".venv" / "bin" / "autoresearch"),
    shutil.which("autoresearch") or "",
]
AUTORESEARCH_BIN: str = _find_binary(_AUTORESEARCH_CANDIDATES) or "autoresearch"

# Cross-run memory aggregator (matches autoresearch.memory.store logic)
def memory_dir() -> Path:
    env = os.environ.get("AUTORESEARCH_MEMORY_DIR", "").strip()
    if env:
        return Path(env)
    project_name = REPO_ROOT.name
    return Path.home() / ".autoresearch" / project_name / "memory"


MEMORY_DB = memory_dir() / "memory.sqlite"
PLAYBOOK_PATH = memory_dir() / "playbook" / "latest.md"

# Worktrees live beside the repo
WORKTREES_DIR = REPO_ROOT.parent / ".autoresearch_worktrees"

HOST = os.environ.get("AUTORESEARCH_CONSOLE_HOST", "127.0.0.1")
PORT = int(os.environ.get("AUTORESEARCH_CONSOLE_PORT", "8765"))

CORS_ORIGINS = os.environ.get(
    "AUTORESEARCH_CONSOLE_CORS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")
