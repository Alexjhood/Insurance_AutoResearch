"""
Runner abstraction — local subprocess now, container/remote later.

Swap out the runner by setting AUTORESEARCH_RUNNER env var:
  local      (default) — subprocess in a git worktree
  docker     — docker run with repo mounted
  remote     — POST to a remote job API (cloud)

Current implementation: local only.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol


class Runner(Protocol):
    def create_worktree(self, track: str, run_id: str) -> Path: ...
    def remove_worktree(self, path: Path) -> None: ...


class LocalRunner:
    def __init__(self, repo_root: Path, worktrees_dir: Path) -> None:
        self._repo = repo_root
        self._wt_root = worktrees_dir

    def create_worktree(self, track: str, run_id: str) -> Path:
        self._wt_root.mkdir(parents=True, exist_ok=True)
        wt = self._wt_root / f"{track}-{run_id}"
        subprocess.run(
            ["git", "worktree", "add", str(wt), "HEAD"],
            cwd=str(self._repo),
            check=True,
            capture_output=True,
        )
        return wt

    def remove_worktree(self, path: Path) -> None:
        try:
            status = subprocess.run(
                ["git", "status", "--short"], cwd=str(path),
                capture_output=True, text=True,
            )
            if not status.stdout.strip():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(path)],
                    cwd=str(self._repo), capture_output=True,
                )
        except Exception:
            pass


def get_runner(repo_root: Path, worktrees_dir: Path) -> Runner:
    """Return the configured runner. Extend here for docker/remote."""
    mode = os.environ.get("AUTORESEARCH_RUNNER", "local")
    if mode == "local":
        return LocalRunner(repo_root, worktrees_dir)
    # Placeholder — additional runners registered here when built
    raise ValueError(f"Unknown AUTORESEARCH_RUNNER={mode!r}. Currently only 'local' is supported.")
