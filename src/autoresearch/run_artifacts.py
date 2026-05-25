"""Run-scoped artifact path helpers."""

from __future__ import annotations

from pathlib import Path
import re

from autoresearch.config import ProjectConfig


def next_iteration_dir(config: ProjectConfig, label: str) -> Path:
    """Allocate the next chronological iteration directory for this run."""

    base = config.artifacts_dir / "iterations"
    base.mkdir(parents=True, exist_ok=True)
    index = 1
    for path in base.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^(\d{3})", path.name)
        if match:
            index = max(index, int(match.group(1)) + 1)
    path = base / f"{index:03d}_{_safe_label(label)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def bootstrap_iteration_dir(config: ProjectConfig) -> Path:
    """Return the fixed bootstrap iteration directory."""

    path = config.artifacts_dir / "iterations" / "000_bootstrap"
    path.mkdir(parents=True, exist_ok=True)
    return path


def proposal_iteration_dir(config: ProjectConfig, proposal: dict) -> Path:
    """Find or allocate the iteration directory associated with a proposal."""

    proposal_path = proposal.get("proposal_path")
    if proposal_path:
        path = Path(proposal_path).parent.parent
        path.mkdir(parents=True, exist_ok=True)
        return path
    return next_iteration_dir(config, str(proposal.get("proposal_id") or "proposal"))


def _safe_label(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return (safe or "iteration")[:80]
