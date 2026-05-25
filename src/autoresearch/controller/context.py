"""Build concise LLM proposal context."""

from __future__ import annotations

from typing import Any

from autoresearch.config import ProjectConfig
from autoresearch.controller.proposal_schema import allowed_search_space
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_champion_history,
    list_comparisons,
    list_experiments,
    list_proposals,
    list_sessions,
)
from autoresearch.utils.io import read_json


def _read_research_log_tail(config: ProjectConfig, n_lines: int = 60) -> str | None:
    """Return the last n_lines of RESEARCH_LOG.md, or None if absent."""
    log_path = config.root / "docs" / "RESEARCH_LOG.md"
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-n_lines:])


def build_llm_context(config: ProjectConfig) -> dict[str, Any]:
    """Build the bounded context that is safe to provide to the proposer."""

    schema_path = config.metadata_dir / "agent_schema.json"
    capping_path = config.metadata_dir / "capping_diagnostics.json"
    latest_nonpromotion_path = config.handoff_results_dir / "latest_nonpromotion_summary.json"
    latest_session_path = config.handoff_results_dir / "latest_session_summary.json"
    latest_cycle_path = config.handoff_results_dir / "latest_cycle_result.json"
    agent_schema = read_json(schema_path) if schema_path.exists() else None
    champion = get_official_champion(config.registry_path)
    experiments = list_experiments(config.registry_path)[:10]
    comparisons = list_comparisons(config.registry_path)[:10]
    proposals = list_proposals(config.registry_path)[:10]
    history = list_champion_history(config.registry_path)[:10]

    return {
        "project_goal": "Improve burning-cost prediction while protecting reproducibility and holdout integrity.",
        "official_champion": champion,
        "recent_experiments": _compact_experiments(experiments),
        "recent_comparisons": _compact_comparisons(comparisons),
        "recent_proposals": _compact_proposals(proposals),
        "champion_history": history,
        "latest_session_summary": read_json(latest_session_path) if latest_session_path.exists() else None,
        "recent_sessions": list_sessions(config.registry_path)[:5],
        "latest_cycle_result": read_json(latest_cycle_path) if latest_cycle_path.exists() else None,
        "latest_nonpromotion_summary": read_json(latest_nonpromotion_path) if latest_nonpromotion_path.exists() else None,
        "agent_schema": agent_schema,
        "default_capping_diagnostics": read_json(capping_path) if capping_path.exists() else None,
        "allowed_search_space": allowed_search_space(config, agent_schema),
        "research_log_tail": _read_research_log_tail(config),
        "evaluation_rules": {
            "ordinary_train_split": config.ordinary_train_split,
            "ordinary_eval_splits": list(config.ordinary_eval_splits),
            "milestone_holdout_access": "forbidden during ordinary search",
            "primary_metric": "rmse_pure_premium",
            "lower_is_better": True,
            "promotion_gate": {
                "minimum_mean_lift": config.minimum_mean_lift,
                "minimum_win_rate": config.minimum_win_rate,
                "bootstrap_lower_bound": config.bootstrap_lower_bound,
                "confidence_level": config.confidence_level,
            },
        },
    }


def _compact_experiments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["experiment_id", "experiment_name", "target_strategy", "model_family", "mean_score", "std_score"]
    return [{key: row.get(key) for key in keys} for row in rows]


def _compact_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["comparison_id", "champion_id", "challenger_id", "mean_lift", "challenger_win_rate", "promotion_decision"]
    return [{key: row.get(key) for key in keys} for row in rows]


def _compact_proposals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ["proposal_id", "status", "experiment_name", "change_summary", "experiment_id", "comparison_id"]
    return [{key: row.get(key) for key in keys} for row in rows]
