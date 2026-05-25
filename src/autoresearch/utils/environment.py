"""Reproducibility manifest capture: git state, dep versions, file hashes."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def capture_environment(root: Path, data_files: dict[str, Path] | None = None) -> dict[str, Any]:
    """Capture runtime environment for reproducibility."""

    manifest: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_sha": _git_sha(root),
        "git_dirty": _git_dirty(root),
        "pip_freeze": _pip_freeze(),
        "key_versions": _key_versions(),
        "file_hashes": {},
    }
    if data_files:
        manifest["file_hashes"] = {
            label: _sha256(path) for label, path in data_files.items() if path.exists()
        }
    return manifest


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(result.strip())
    except Exception:
        return None


def _pip_freeze() -> list[str]:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip().splitlines()
    except Exception:
        return []


def _key_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "sklearn", "pyarrow"):
        try:
            if pkg == "sklearn":
                import sklearn
                versions["scikit-learn"] = sklearn.__version__
            else:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
