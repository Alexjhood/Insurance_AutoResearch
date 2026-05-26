"""Build concise LLM proposal context."""

from __future__ import annotations

from typing import Any

from autoresearch.config import ProjectConfig
from autoresearch.controller.proposal_schema import allowed_search_space
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_comparisons,
    list_experiments,
    list_proposals,
)
from autoresearch.utils.io import read_json


def build_llm_context(config: ProjectConfig) -> dict[str, Any]:
    """Build the bounded context that is safe to provide to the proposer."""

    schema_path = config.metadata_dir / "agent_schema.json"
    latest_nonpromotion_path = config.handoff_results_dir / "latest_nonpromotion_summary.json"
    latest_cycle_path = config.handoff_results_dir / "latest_cycle_result.json"
    raw_schema = read_json(schema_path) if schema_path.exists() else None
    champion = get_official_champion(config.registry_path)
    experiments = list_experiments(config.registry_path)[:10]
    comparisons = list_comparisons(config.registry_path)[:10]
    all_proposals = list_proposals(config.registry_path)
    proposals = all_proposals[:10]

    raw_cycle = read_json(latest_cycle_path) if latest_cycle_path.exists() else None
    latest_cycle_result = _flatten_cycle_result(raw_cycle) if raw_cycle else None

    return {
        "project_goal": (
            "Improve burning-cost prediction while protecting reproducibility and holdout integrity. "
            "Every run starts from the global-mean no-model baseline; progress through many small, "
            "well-motivated steps with a broad search before committing to any single direction. "
            "Claim cap is fixed at 100,000."
        ),
        "official_champion": champion,
        "recent_experiments": _compact_experiments(experiments),
        "recent_comparisons": _compact_comparisons(comparisons),
        "recent_proposals": _compact_proposals(proposals),
        "proposal_count": len(all_proposals),
        "latest_cycle_result": latest_cycle_result,
        "latest_nonpromotion_summary": read_json(latest_nonpromotion_path) if latest_nonpromotion_path.exists() else None,
        "agent_schema": _compact_agent_schema(raw_schema),
        "allowed_search_space": allowed_search_space(config, raw_schema),
        "evaluation_rules": {
            "ordinary_train_split": config.ordinary_train_split,
            "ordinary_eval_splits": list(config.ordinary_eval_splits),
            "milestone_holdout_access": "forbidden during ordinary search",
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
