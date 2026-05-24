"""Machine-validated proposal schema and allowed search space."""

from __future__ import annotations

import re
from typing import Any


VALID_STATUSES = {
    "proposed",
    "validated",
    "running",
    "completed",
    "failed",
    "compared",
    "promoted",
    "rejected",
    "inconclusive",
}

TARGET_COLUMNS = {
    "record_id",
    "claim_count_signal_q",
    "claim_event_count_l",
    "claim_cost_observed_k",
    "claim_cost_capped_active",
}


def allowed_search_space(config, agent_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the explicit search space exposed to proposal generators."""

    feature_columns = []
    if agent_schema:
        feature_columns = [
            item["name"]
            for item in agent_schema.get("columns", [])
            if item["name"] not in TARGET_COLUMNS and item.get("role") != "target_or_outcome"
        ]
    return {
        "model_families": list(config.search_space["model_families"]),
        "target_strategies": list(config.search_space["target_strategies"]),
        "alpha_range": [float(config.search_space["min_alpha"]), float(config.search_space["max_alpha"])],
        "claim_cap_thresholds": list(config.search_space["claim_cap_thresholds"]),
        "allow_disable_claim_capping": bool(config.search_space["allow_disable_claim_capping"]),
        "feature_columns": feature_columns,
        "branch_actions": ["extend_current", "new_branch"],
    }


def validate_proposal(proposal: dict[str, Any], search_space: dict[str, Any]) -> list[str]:
    """Return validation errors for a structured experiment proposal."""

    errors: list[str] = []
    required_text = [
        "proposal_id",
        "parent_experiment_id",
        "experiment_name",
        "rationale",
        "change_summary",
        "expected_benefit",
        "key_risk",
    ]
    for field in required_text:
        if not isinstance(proposal.get(field), str) or not proposal[field].strip():
            errors.append(f"{field} is required")

    proposal_id = proposal.get("proposal_id", "")
    if proposal_id and not re.fullmatch(r"[A-Za-z0-9_\-]{3,80}", proposal_id):
        errors.append("proposal_id must be 3-80 chars using letters, numbers, hyphen, or underscore")

    branch_action = proposal.get("branch_action", "extend_current")
    if branch_action not in search_space["branch_actions"]:
        errors.append(f"branch_action must be one of {search_space['branch_actions']}")

    exp_config = proposal.get("experiment_config")
    if not isinstance(exp_config, dict):
        errors.append("experiment_config must be an object")
        return errors

    if exp_config.get("model_family") not in search_space["model_families"]:
        errors.append("model_family is not allowed")
    if exp_config.get("target_strategy") not in search_space["target_strategies"]:
        errors.append("target_strategy is not allowed")
    if exp_config.get("experiment_name") != proposal.get("experiment_name"):
        errors.append("experiment_config.experiment_name must match proposal experiment_name")
    if exp_config.get("parent_experiment_id") != proposal.get("parent_experiment_id"):
        errors.append("experiment_config.parent_experiment_id must match proposal parent_experiment_id")

    preprocessing = exp_config.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        errors.append("preprocessing must be an object")
    else:
        enabled = preprocessing.get("claim_capping_enabled")
        if not isinstance(enabled, bool):
            errors.append("preprocessing.claim_capping_enabled must be boolean")
        if enabled is False and not search_space["allow_disable_claim_capping"]:
            errors.append("claim capping cannot be disabled in this search space")
        threshold = preprocessing.get("claim_cap_threshold")
        if threshold not in search_space["claim_cap_thresholds"]:
            errors.append("claim_cap_threshold is outside the allowed set")

    model = exp_config.get("model", {})
    if not isinstance(model, dict):
        errors.append("model must be an object")
    else:
        alpha = model.get("alpha")
        low, high = search_space["alpha_range"]
        if not isinstance(alpha, (int, float)) or not low <= float(alpha) <= high:
            errors.append(f"model.alpha must be in [{low}, {high}]")
        allowed_features = set(search_space["feature_columns"])
        for key in ("feature_inclusions", "feature_exclusions"):
            value = model.get(key)
            if value is None:
                continue
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"model.{key} must be a list of strings")
                continue
            unknown = sorted(set(value).difference(allowed_features))
            if unknown:
                errors.append(f"model.{key} contains unknown features: {unknown}")

    serialised = str(proposal).lower()
    if "milestone_holdout" in serialised:
        errors.append("proposal must not reference milestone_holdout")
    return errors


def normalise_proposal(proposal: dict[str, Any], *, branch_id: str, parent_branch_id: str | None) -> dict[str, Any]:
    """Fill derived branch fields after validation."""

    result = dict(proposal)
    result["branch_id"] = branch_id
    result["parent_branch_id"] = parent_branch_id
    result["experiment_config"] = dict(result["experiment_config"])
    result["experiment_config"]["parent_experiment_id"] = result["parent_experiment_id"]
    return result
