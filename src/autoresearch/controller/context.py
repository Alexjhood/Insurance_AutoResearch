"""Build concise LLM proposal context."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from autoresearch.config import ProjectConfig
from autoresearch.controller.proposal_schema import allowed_search_space
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_research_nodes,
    list_research_lines,
    list_research_line_history,
    list_comparisons,
    list_experiments,
    list_proposals,
)
from autoresearch.utils.io import read_json


def build_llm_context(config: ProjectConfig) -> dict[str, Any]:
    """Build the bounded context that is safe to provide to the proposer."""
    from autoresearch.memory import resolve_memory_access

    schema_path = config.metadata_dir / "agent_schema.json"
    latest_nonpromotion_path = config.handoff_results_dir / "latest_nonpromotion_summary.json"
    latest_cycle_path = config.handoff_results_dir / "latest_cycle_result.json"
    raw_schema = read_json(schema_path) if schema_path.exists() else None
    champion = get_official_champion(config.registry_path)
    experiments = list_experiments(config.registry_path)[:10]
    comparisons = list_comparisons(config.registry_path)[:10]
    all_proposals = list_proposals(config.registry_path)
    proposals = all_proposals[:10]
    research_nodes = list_research_nodes(config.registry_path, limit=25)
    research_lines = list_research_lines(config.registry_path)
    research_line_history = list_research_line_history(config.registry_path)[:20]
    compact_nodes = _compact_research_nodes(research_nodes)

    raw_cycle = read_json(latest_cycle_path) if latest_cycle_path.exists() else None
    latest_cycle_result = _flatten_cycle_result(raw_cycle) if raw_cycle else None

    ctx: dict[str, Any] = {
        "project_goal": (
            f"Improve {config.target_mode} prediction while protecting reproducibility and holdout integrity. "
            "Every run starts from the global-mean no-model baseline; progress through many small, "
            "well-motivated steps with a broad search before committing to any single direction. "
            "Claim cap is fixed at 100,000. exposure_term_a is an exposure offset for weights, "
            "response denominators, and converting predicted rates to target totals; it must not be "
            "used as a predictive feature because it is unavailable at quote time."
        ),
        "official_champion": champion,
        "recent_experiments": _compact_experiments(experiments),
        "recent_comparisons": _compact_comparisons(comparisons),
        "recent_proposals": _compact_proposals(proposals),
        "research_tree": {
            "scope": "active run only",
            "principles": [
                "Choose an idea parent from this run's own tree: champion, near-miss, informative failure, or unexplored node.",
                "Prefer genuine exploration over small retunes when the tree is narrow or repetitive.",
                "Do not repeat failed configurations; use failures as evidence for materially different child ideas.",
                "The official champion remains the evaluation parent unless the proposal schema says otherwise.",
            ],
            "tree_policy": _build_tree_policy(champion, compact_nodes),
            "recent_nodes": compact_nodes,
        },
        "research_lines": {
            "scope": "active run only",
            "max_active_lines": 5,
            "principles": [
                "Self-sort proposals into a small number of named research lines.",
                "Use local promotion to advance a promising line without replacing the global champion.",
                "Use global promotion only when the experiment should become the whole run's official champion.",
                "The single-split hurdle compares against the local line incumbent when one exists.",
            ],
            "active_lines": _compact_research_lines([line for line in research_lines if line.get("status") == "active"]),
            "recent_line_history": _compact_research_line_history(research_line_history),
        },
        "proposal_count": len(all_proposals),
        "active_queue": _active_queue_summary(all_proposals, config.running_stale_minutes),
        "latest_cycle_result": latest_cycle_result,
        "latest_nonpromotion_summary": read_json(latest_nonpromotion_path) if latest_nonpromotion_path.exists() else None,
        "agent_schema": _compact_agent_schema(raw_schema),
        "allowed_search_space": allowed_search_space(config, raw_schema),
        "evaluation_rules": {
            "ordinary_train_split": config.ordinary_train_split,
            "ordinary_eval_splits": list(config.ordinary_eval_splits),
            "milestone_holdout_access": "forbidden during ordinary search",
            "target_mode": config.target_mode,
            "primary_metric": config.primary_metric,
            "lower_is_better": config.primary_metric != "gini_weighted",
            "promotion_gate": {
                "minimum_mean_lift": config.minimum_mean_lift,
                "minimum_win_rate": config.minimum_win_rate,
                "bootstrap_lower_bound": config.bootstrap_lower_bound,
                "confidence_level": config.confidence_level,
            },
        },
    }

    # Add memory_access block only when access is granted.
    # With 'none' (default) the context dict is unchanged — run isolation is sacrosanct.
    access = resolve_memory_access(config)
    if access in ("own", "all"):
        ctx["memory_access"] = _build_memory_access_block(access)

    return ctx


def _build_memory_access_block(access: str) -> dict[str, Any]:
    """Return a compact memory_access context block for the LLM proposer."""
    scope_desc = (
        "your own model's history across all its runs"
        if access == "own"
        else "all models' history, fully attributed"
    )
    return {
        "scope": access,
        "description": (
            f"Cross-run memory is available ({scope_desc}). "
            "Use the query tool to retrieve relevant insights or analyse experiment history. "
            "Do NOT assume facts not returned by the tool."
        ),
        "how_to_query": {
            "retrieval": "autoresearch memory query --insights [--verified-only] [--family X] [--model Z]",
            "analytical": (
                "autoresearch memory query "
                "--analysis [peak-gini-by-framing | plateau-families | biggest-single-jumps | efficiency-by-model]"
            ),
        },
        "playbook_hint": (
            "If a dynamic playbook has been generated, the handoff links it directly; "
            "otherwise it can be produced with `autoresearch memory build-playbook`."
        ),
    }


def _compact_agent_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if not schema:
        return None
    return {
        "row_count": schema.get("row_count"),
        "columns": [
            {"name": c["name"], "role": c["role"]}
            for c in schema.get("columns", [])
        ],
    }


def _flatten_cycle_result(raw: dict[str, Any]) -> dict[str, Any]:
    inner = raw.get("cycle_result") or {}
    return {
        "completed_at": raw.get("completed_at"),
        "proposal_id": inner.get("proposal_id"),
        "experiment_id": inner.get("experiment_id"),
        "comparison_id": inner.get("comparison_id"),
        "decision": inner.get("decision"),
        "metrics_summary": inner.get("metrics_summary"),
        "comparison_report": inner.get("comparison_report"),
    }


def _compact_experiments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["experiment_id", "experiment_name", "target_strategy", "model_family", "mean_score", "std_score"]
    return [{key: row.get(key) for key in keys} for row in rows]


def _compact_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["comparison_id", "champion_id", "challenger_id", "mean_lift", "challenger_win_rate", "promotion_decision"]
    return [{key: row.get(key) for key in keys} for row in rows]


def _compact_proposals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["proposal_id", "status", "experiment_name", "change_summary", "experiment_id", "comparison_id"]
    result = []
    for row in rows:
        item = {key: row.get(key) for key in keys}
        cs = item.get("change_summary") or ""
        if len(cs) > 200:
            item["change_summary"] = cs[:200] + "…"
        result.append(item)
    return result


def _active_queue_summary(rows: list[dict[str, Any]], stale_minutes: int) -> dict[str, Any]:
    active_statuses = {"validated", "proposed", "needs_repair", "running", "awaiting_decision"}
    active = []
    stale_after = int(stale_minutes) * 60
    now = datetime.now(timezone.utc)
    for row in rows:
        status = row.get("status")
        if status not in active_statuses:
            continue
        item = {
            "proposal_id": row.get("proposal_id"),
            "status": status,
            "experiment_name": row.get("experiment_name"),
            "parent_experiment_id": row.get("parent_experiment_id"),
            "updated_at": row.get("updated_at"),
        }
        if status == "running":
            updated_at = _parse_registry_time(row.get("updated_at"))
            running_for = int((now - updated_at).total_seconds()) if updated_at else None
            item.update({
                "running_for_seconds": running_for,
                "stale_after_seconds": stale_after,
                "is_stale": bool(running_for is not None and running_for > stale_after),
            })
        active.append(item)
    return {
        "active_count": len(active),
        "has_running": any(item["status"] == "running" for item in active),
        "has_awaiting_decision": any(item["status"] == "awaiting_decision" for item in active),
        "items": active[:5],
    }


def _parse_registry_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _compact_research_nodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "node_id",
        "line_id",
        "parent_node_id",
        "proposal_id",
        "status",
        "outcome_type",
        "experiment_id",
        "comparison_id",
        "change_summary",
        "expected_benefit",
        "key_risk",
        "metrics",
        "guidance",
        "tree_metadata",
    ]
    result = []
    for row in rows:
        item = {key: row.get(key) for key in keys}
        for text_key in ("change_summary", "expected_benefit", "key_risk", "guidance"):
            value = item.get(text_key) or ""
            if len(value) > 220:
                item[text_key] = value[:220] + "…"
        result.append(item)
    return result


def _compact_research_lines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "line_id",
        "label",
        "status",
        "root_node_id",
        "hypothesis",
        "current_node_id",
        "current_experiment_id",
        "best_node_id",
        "best_experiment_id",
        "notes",
    ]
    result = []
    for row in rows[:8]:
        item = {key: row.get(key) for key in keys}
        for text_key in ("hypothesis", "notes"):
            value = item.get(text_key) or ""
            if len(value) > 220:
                item[text_key] = value[:220] + "…"
        result.append(item)
    return result


def _compact_research_line_history(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "created_at",
        "line_id",
        "previous_experiment_id",
        "new_experiment_id",
        "node_id",
        "action",
        "reason",
        "comparison_id",
        "proposal_id",
    ]
    result = []
    for row in rows[:12]:
        item = {key: row.get(key) for key in keys}
        reason = item.get("reason") or ""
        if len(reason) > 180:
            item["reason"] = reason[:180] + "…"
        result.append(item)
    return result


def _build_tree_policy(champion: dict[str, Any] | None, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact active-run tree-walk guidance without prescribing model types."""

    chronological = list(reversed(nodes))
    axis_counts: dict[str, int] = {}
    for node in chronological:
        axis = _node_axis(node)
        if axis:
            axis_counts[axis] = axis_counts.get(axis, 0) + 1

    recent_axis = None
    streak = 0
    for node in reversed(chronological):
        axis = _node_axis(node)
        if not axis:
            break
        if recent_axis is None:
            recent_axis = axis
            streak = 1
            continue
        if axis != recent_axis:
            break
        streak += 1

    actions: list[dict[str, Any]] = []
    active_nodes = [
        node for node in nodes
        if node.get("status") in {"awaiting_decision", "promoted", "completed", "screened"}
        or node.get("experiment_id")
    ]
    failed_nodes = [
        node for node in nodes
        if node.get("outcome_type") in {"clear_loser", "failed_run", "needs_repair", "system_error"}
        or node.get("status") in {"rejected", "failed", "needs_repair"}
    ]
    champion_node = _find_champion_node(champion, nodes)

    if not nodes:
        actions.append({
            "action_id": "start_first_root",
            "tree_action": "new_root",
            "requires_parent_node_id": False,
            "parent_node_id": None,
            "reason": "No active-run research nodes exist yet; start a first explicit hypothesis.",
        })
    elif streak >= 2:
        actions.append({
            "action_id": f"rotate_after_{recent_axis}_streak",
            "tree_action": "rotate_axis",
            "requires_parent_node_id": True,
            "parent_node_id": (active_nodes[0]["node_id"] if active_nodes else nodes[0]["node_id"]),
            "avoid_axis": recent_axis,
            "reason": "Recent nodes repeat the same exploration axis; choose a different axis with a clear learning goal.",
        })

    if champion_node:
        actions.append({
            "action_id": "extend_current_champion",
            "tree_action": "exploit_champion",
            "requires_parent_node_id": True,
            "parent_node_id": champion_node["node_id"],
            "reason": "Build from the node that produced the current champion if there is direct diagnostic evidence.",
        })
    elif active_nodes:
        actions.append({
            "action_id": "extend_recent_successful_node",
            "tree_action": "extend_node",
            "requires_parent_node_id": True,
            "parent_node_id": active_nodes[0]["node_id"],
            "reason": "Extend a completed or screened node only if the child changes the hypothesis, not just a small dial.",
        })

    if failed_nodes:
        actions.append({
            "action_id": "learn_from_recent_failure",
            "tree_action": "revisit_failure",
            "requires_parent_node_id": True,
            "parent_node_id": failed_nodes[0]["node_id"],
            "reason": "Use a recent failure as evidence for a materially different child idea.",
        })

    if len(axis_counts) < 3 and nodes:
        actions.append({
            "action_id": "open_underexplored_axis",
            "tree_action": "new_root",
            "requires_parent_node_id": False,
            "parent_node_id": None,
            "reason": "The active-run tree is still narrow; a genuinely different axis can add useful information.",
        })

    return {
        "scope": "active run only",
        "one_proposal_per_context_refresh": True,
        "axis_counts": axis_counts,
        "recent_axis_streak": {"axis": recent_axis, "count": streak},
        "recommended_actions": actions[:4],
        "override_policy": (
            "If no recommended action fits, set selected_tree_action_id to the closest action "
            "and provide tree_policy_override_rationale."
        ),
    }


def _node_axis(node: dict[str, Any]) -> str | None:
    metadata = node.get("tree_metadata") or {}
    axis = metadata.get("exploration_axis")
    return str(axis) if axis else None


def _find_champion_node(champion: dict[str, Any] | None, nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not champion:
        return None
    champion_id = champion.get("champion_id")
    if not champion_id:
        return None
    for node in nodes:
        if node.get("experiment_id") == champion_id:
            return node
    return None
