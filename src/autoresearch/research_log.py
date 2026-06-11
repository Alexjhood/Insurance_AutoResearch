"""Deterministic rendering and completion checks for the run research log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig
from autoresearch.experiment_registry.registry import (
    complete_research_log_entry,
    latest_incomplete_research_log_entry,
    list_research_log_entries,
)


def render_research_log(config: ProjectConfig) -> Path:
    """Regenerate ``RESEARCH_LOG.md`` atomically from structured entries."""

    entries = list_research_log_entries(config.registry_path)
    lines = [
        "# Research Log",
        "",
        "<!-- Generated from registry.sqlite. Do not edit by hand. -->",
    ]
    for entry in entries:
        lines.extend(_render_entry(entry))
    text = "\n".join(lines).rstrip() + "\n"
    path = config.research_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def complete_pending_reflection_from_proposal(
    config: ProjectConfig,
    proposal: dict[str, Any],
) -> bool:
    """Complete the previous auto-reject entry from the next proposal."""

    entry = latest_incomplete_research_log_entry(config.registry_path)
    if entry is None:
        return True
    if entry.get("comparison_id"):
        return False
    reflection = proposal.get("previous_cycle_reflection")
    if not isinstance(reflection, dict):
        return False
    if int(reflection.get("cycle") or -1) != int(entry["cycle"]):
        return False
    interpretation = _clean(reflection.get("interpretation"))
    next_step = _clean(reflection.get("next"))
    if not interpretation or not next_step:
        return False
    complete_research_log_entry(
        config.registry_path,
        session_id=entry["session_id"],
        cycle=int(entry["cycle"]),
        interpretation=interpretation,
        next_step=next_step,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    render_research_log(config)
    return True


def _render_entry(entry: dict[str, Any]) -> list[str]:
    interpretation = _clean(entry.get("interpretation")) or "_Pending agent reflection._"
    next_step = _clean(entry.get("next_step")) or "_Pending agent direction._"
    return [
        "",
        f"## Cycle {entry['cycle']}",
        "",
        f"**Hypothesis**: {_clean(entry.get('hypothesis'))}",
        "",
        f"**Changes**: {_clean(entry.get('changes'))}",
        "",
        f"**Outcome**: {_clean(entry.get('outcome'))}",
        "",
        f"**Metrics**: {_format_metrics(entry.get('metrics') or {})}",
        "",
        f"**Interpretation**: {interpretation}",
        "",
        f"**Next**: {next_step}",
    ]


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "No quantitative metrics were produced."
    parts = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, float):
            rendered = f"{value:.6g}"
        elif isinstance(value, (dict, list)):
            rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            rendered = str(value)
        parts.append(f"`{key}={rendered}`")
    return ", ".join(parts)


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())
