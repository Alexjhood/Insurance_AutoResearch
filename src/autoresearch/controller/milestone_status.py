"""Operator-facing milestone status without exposing protected holdout metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.utils.io import read_json


def milestone_status(config, champion: dict[str, Any] | None) -> dict[str, Any]:
    """Summarise whether the search champion has completed trusted evaluation."""

    champion = champion or {}
    champion_id = champion.get("champion_id")
    comparison_id = champion.get("comparison_id")
    if not champion_id or not comparison_id:
        return {
            "status": "not_applicable",
            "champion_id": champion_id,
            "operator_action": None,
        }

    report_dir = config.artifacts_dir / "milestone_reports"
    reports = _matching_reports(report_dir, str(champion_id))
    if any(report.get("status") == "completed" for report in reports):
        return {
            "status": "completed",
            "champion_id": champion_id,
            "operator_action": None,
        }

    return {
        "status": "pending_operator",
        "champion_id": champion_id,
        "operator_action": (
            f"autoresearch --track {config.track_id} --run-id {config.run_id} "
            f"evaluate-milestone {champion_id}"
        ),
    }


def _matching_reports(report_dir: Path, champion_id: str) -> list[dict[str, Any]]:
    if not report_dir.exists():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*.json")):
        try:
            report = read_json(path)
        except (OSError, ValueError, TypeError):
            continue
        if report.get("champion_id") == champion_id:
            reports.append(report)
    return reports
