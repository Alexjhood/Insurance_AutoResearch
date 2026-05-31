"""Dynamic playbook generator for the cross-run memory aggregator.

Compiles verified insights + leaderboard-derived facts into (under the out-of-tree
memory dir; override with AUTORESEARCH_MEMORY_DIR):
  <memory_dir>/playbook/latest.md
  <memory_dir>/playbook/<timestamp>.md (timestamped copy)

Only verified=1 insights are included. Every bullet cites evidence IDs and
the source model_id (full attribution, per the locked decision).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _load_verified_insights(memory_path: Path) -> list[dict[str, Any]]:
    if not memory_path.exists():
        return []
    with sqlite3.connect(memory_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(
            "SELECT * FROM insights WHERE verified = 1 ORDER BY created_at DESC"
        )
        return [dict(row) for row in cur.fetchall()]


def _load_leaderboard_facts(memory_path: Path, threshold: float = 0.37) -> dict[str, Any]:
    """Return compact leaderboard facts for the playbook header."""
    if not memory_path.exists():
        return {}
    with sqlite3.connect(memory_path) as con:
        # Peak quality per model
        cur = con.execute(
            """
            SELECT r.model_id, MAX(r.peak_gini) AS best_gini, COUNT(*) AS n_runs,
                   SUM(r.n_experiments) AS n_experiments
            FROM runs r
            GROUP BY r.model_id
            ORDER BY best_gini DESC
            """
        )
        rows = cur.fetchall()
        if not rows:
            return {}
        top_model = rows[0][0]
        top_gini = rows[0][1]

        # Models that never crossed the threshold
        plateau_models = [
            r[0] for r in rows if r[1] is not None and r[1] < threshold
        ]

    return {
        "top_model": top_model,
        "top_gini": top_gini,
        "plateau_models": plateau_models,
        "threshold": threshold,
    }


def _categorise_insights(
    insights: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split insights into works / plateaus / leverage buckets by claim keywords."""
    works: list[dict[str, Any]] = []
    plateaus: list[dict[str, Any]] = []
    leverage: list[dict[str, Any]] = []

    for ins in insights:
        claim = (ins.get("claim") or "").lower()
        if any(kw in claim for kw in ("plateau", "ceiling", "cap", "stuck", "max", "limit")):
            plateaus.append(ins)
        elif any(kw in claim for kw in ("high", "leverage", "highest", "best move", "break")):
            leverage.append(ins)
        else:
            works.append(ins)

    # Any insight not already in a bucket goes into 'works'
    covered = set(ins["insight_id"] for ins in plateaus + leverage)
    works = [ins for ins in works if ins["insight_id"] not in covered] + [
        ins for ins in insights if ins["insight_id"] not in covered and ins not in works
    ]
    return works, plateaus, leverage


def _render_insight_bullet(ins: dict[str, Any]) -> str:
    evidence = {}
    try:
        evidence = json.loads(ins.get("evidence_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        pass
    exp_ids = evidence.get("experiment_ids") or []
    cmp_ids = evidence.get("comparison_ids") or []
    model_id = ins.get("model_id", "unknown")
    confidence = ins.get("confidence")
    conf_str = f" (confidence {confidence:.0%})" if confidence is not None else ""

    evidence_parts = []
    if exp_ids:
        evidence_parts.append("experiments: " + ", ".join(f"`{e}`" for e in exp_ids[:3]))
    if cmp_ids:
        evidence_parts.append("comparisons: " + ", ".join(f"`{c}`" for c in cmp_ids[:3]))
    evidence_str = "; ".join(evidence_parts) if evidence_parts else "no cited evidence"

    return (
        f"- {ins['claim']}{conf_str}  \n"
        f"  Source: `{model_id}` | Evidence: {evidence_str}"
    )


def build_playbook(
    memory_path: Path,
    *,
    structural_gini_threshold: float = 0.37,
    model_id_filter: str | None = None,
) -> Path | None:
    """Compile verified insights + facts into a playbook markdown file.

    Returns the path to the playbook `latest.md`, or None if
    there are no verified insights.

    Parameters
    ----------
    memory_path:
        Path to the memory.sqlite aggregator.
    structural_gini_threshold:
        Threshold used for labelling plateau models.
    model_id_filter:
        When set, produce a filtered own-model variant that includes only
        insights attributed to this model_id.
    """
    insights = _load_verified_insights(memory_path)
    if model_id_filter:
        insights = [ins for ins in insights if ins.get("model_id") == model_id_filter]

    if not insights:
        logger.info("build_playbook: no verified insights — skipping")
        return None

    facts = _load_leaderboard_facts(memory_path, threshold=structural_gini_threshold)
    works, plateaus, leverage = _categorise_insights(insights)

    lines: list[str] = [
        "# Research Playbook",
        "",
        f"_Generated {_now_human()} from verified insights in the cross-run memory aggregator._",
        "_Only insights with verified=1 are included. All bullets cite evidence and source model._",
    ]

    if model_id_filter:
        lines += ["", f"_Scope: own-model insights for `{model_id_filter}` only._"]

    if facts and facts.get("top_gini") is not None:
        gini_str = f"{facts['top_gini']:.4f}"
        lines += [
            "",
            "## Leaderboard summary",
            "",
            f"- Best observed gini: **{gini_str}** (`{facts['top_model']}`)",
            f"- Structural threshold (rate-Tweedie escape band): **{facts['threshold']:.3f}**",
        ]
        if facts["plateau_models"]:
            plist = ", ".join(f"`{m}`" for m in facts["plateau_models"][:5])
            lines.append(
                f"- Models that never crossed the threshold: {plist}"
            )

    lines += ["", "## What works", ""]
    if works:
        for ins in works:
            lines.append(_render_insight_bullet(ins))
    else:
        lines.append("_No verified insights in this category yet._")

    lines += ["", "## What plateaus / known ceilings", ""]
    if plateaus:
        for ins in plateaus:
            lines.append(_render_insight_bullet(ins))
    else:
        lines.append("_No verified plateau insights yet._")

    lines += ["", "## Highest-leverage moves", ""]
    if leverage:
        for ins in leverage:
            lines.append(_render_insight_bullet(ins))
    else:
        lines.append("_No verified high-leverage insights yet._")

    content = "\n".join(lines) + "\n"

    # Write latest.md and timestamped copy
    playbook_dir = memory_path.parent / "playbook"
    playbook_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{model_id_filter.replace('/', '_')}" if model_id_filter else ""
    latest_path = playbook_dir / f"latest{suffix}.md"
    timestamp_path = playbook_dir / f"{_now_stamp()}{suffix}.md"

    latest_path.write_text(content, encoding="utf-8")
    timestamp_path.write_text(content, encoding="utf-8")

    logger.info("build_playbook: wrote %s", latest_path)
    return latest_path


def playbook_needs_rebuild(memory_path: Path, playbook_path: Path | None) -> bool:
    """Return True if any new verified insight appeared after the last playbook build."""
    if playbook_path is None or not playbook_path.exists():
        return True
    if not memory_path.exists():
        return False
    last_built = playbook_path.stat().st_mtime
    with sqlite3.connect(memory_path) as con:
        row = con.execute(
            "SELECT MAX(created_at) FROM insights WHERE verified = 1"
        ).fetchone()
        if not row or not row[0]:
            return False
    # Compare latest insight timestamp to file mtime
    # Rough check: if the playbook file is older than 1 second before the
    # latest insight, rebuild. This is conservative but safe.
    try:
        latest_insight_ts = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        ).timestamp()
        return latest_insight_ts > last_built
    except (ValueError, TypeError):
        return True
