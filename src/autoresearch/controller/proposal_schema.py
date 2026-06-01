"""Machine-validated proposal schema and allowed search space."""

from __future__ import annotations

import json
import re
from typing import Any

from autoresearch.feature_policy import NON_PREDICTIVE_COLUMNS, predictive_columns


VALID_STATUSES = {
    "proposed", "validated", "running", "completed", "failed",
    "compared", "promoted", "local_promoted", "rejected", "inconclusive", "duplicate", "needs_repair",
    "screened", "awaiting_decision", "stale_parent",
}

TREE_ACTIONS = {
    "new_root",
    "extend_node",
    "revisit_failure",
    "exploit_champion",
    "rotate_axis",
}

EXPLORATION_AXES = {
    "model_family",
    "target_framing",
    "feature_representation",
    "calibration",
    "hyperparameter",
    "diagnostic_probe",
    "data_slice",
    "ensemble",
    "other",
}

RESEARCH_LINE_ACTIONS = {
    "create_line",
    "extend_line",
    "revisit_line",
    "close_line",
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

    ss = config.search_space
    feature_columns = []
    if agent_schema:
        feature_columns = [
            name
            for name in predictive_columns(agent_schema.get("columns", []))
            if name not in TARGET_COLUMNS
        ]

    families = list(ss.get("model_families", ["global_mean"]))

    space: dict[str, Any] = {
        "model_families": families,
        "target_strategies": list(ss.get("target_strategies", ["direct_pure_premium", "frequency_severity"])),
        "target_modes": ["burning_cost", "frequency"],
        "active_target_mode": config.target_mode,
        "feature_columns": feature_columns,
        "non_predictive_columns": sorted(NON_PREDICTIVE_COLUMNS),
        "feature_policy": {
            "exposure_term_a": (
                "Use only for exposure weights, frequency/severity denominators, "
                "and converting predicted rates to target totals. "
                "Do not use as a predictive model feature because it is unavailable at quote time."
            )
        },
        "branch_actions": ["extend_current", "new_branch"],
        "research_line_actions": sorted(RESEARCH_LINE_ACTIONS),
        "allow_legacy_baselines": bool(ss.get("allow_legacy_baselines", False)),
        "allow_open_model_families": bool(ss.get("allow_open_model_families", False)),
        "requires_model_script": bool(ss.get("requires_model_script", False)),
    }

    # Preprocessing
    prep = ss.get("preprocessing", {})
    space["claim_cap_thresholds"] = list(prep.get("claim_cap_thresholds", [100000]))
    space["allow_disable_claim_capping"] = bool(prep.get("allow_disable_claim_capping", False))
    space["allow_log1p_features"] = list(prep.get("allow_log1p_features", []))

    return space


def validate_proposal(proposal: dict[str, Any], search_space: dict[str, Any]) -> list[str]:
    """Return validation errors for a structured experiment proposal."""

    errors: list[str] = []
    required_text = [
        "proposal_id", "parent_experiment_id", "experiment_name",
        "rationale", "change_summary", "expected_benefit", "key_risk",
        "tree_action", "parent_rationale", "exploration_axis",
        "approach_family", "target_framing", "feature_representation",
        "expected_learning", "selected_tree_action_id",
        "research_line_action", "research_line_id", "research_line_label",
        "research_line_hypothesis", "line_membership_rationale",
    ]
    for field in required_text:
        if not isinstance(proposal.get(field), str) or not proposal[field].strip():
            errors.append(f"{field} is required")

    proposal_id = proposal.get("proposal_id", "")
    if proposal_id and not re.fullmatch(r"[A-Za-z0-9_\-]{3,80}", proposal_id):
        errors.append("proposal_id must be 3-80 chars using letters, numbers, hyphen, or underscore")

    line_id = proposal.get("research_line_id", "")
    if line_id and not re.fullmatch(r"[A-Za-z0-9_\-]{3,80}", line_id):
        errors.append("research_line_id must be 3-80 chars using letters, numbers, hyphen, or underscore")

    line_action = proposal.get("research_line_action")
    if line_action not in RESEARCH_LINE_ACTIONS:
        errors.append(f"research_line_action must be one of {sorted(RESEARCH_LINE_ACTIONS)}")

    tree_action = proposal.get("tree_action")
    if tree_action not in TREE_ACTIONS:
        errors.append(f"tree_action must be one of {sorted(TREE_ACTIONS)}")

    exploration_axis = proposal.get("exploration_axis")
    if exploration_axis not in EXPLORATION_AXES:
        errors.append(f"exploration_axis must be one of {sorted(EXPLORATION_AXES)}")

    if "research_parent_node_id" in proposal:
        research_parent = proposal.get("research_parent_node_id")
        if research_parent is not None and (
            not isinstance(research_parent, str) or not research_parent.strip()
        ):
            errors.append("research_parent_node_id must be a non-empty string or null")

    branch_action = proposal.get("branch_action", "extend_current")
    if branch_action not in search_space["branch_actions"]:
        errors.append(f"branch_action must be one of {search_space['branch_actions']}")

    exp_config = proposal.get("experiment_config")
    if not isinstance(exp_config, dict):
        errors.append("experiment_config must be an object")
        return errors

    family = exp_config.get("model_family")
    if family not in search_space["model_families"] and not search_space.get("allow_open_model_families", False):
        errors.append(f"model_family {family!r} is not in allowed families: {search_space['model_families']}")
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
            errors.append(f"claim_cap_threshold {threshold} is outside the allowed set: {search_space['claim_cap_thresholds']}")

    model = exp_config.get("model", {})
    if not isinstance(model, dict):
        errors.append("model must be an object")
    else:
        script_path = model.get("script_path") or model.get("model_script_path")
        if search_space.get("requires_model_script", False) and family != "global_mean":
            if not isinstance(script_path, str) or not script_path.strip():
                errors.append("model.script_path is required for non-global_mean autonomous experiments")
        allowed_features = set(search_space.get("feature_columns", []))
        non_predictive = set(search_space.get("non_predictive_columns", []))
        for key in ("feature_inclusions", "feature_exclusions"):
            value = model.get(key)
            if value is None:
                continue
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"model.{key} must be a list of strings")
                continue
            forbidden = sorted(set(value).intersection(non_predictive))
            if forbidden:
                errors.append(
                    f"model.{key} must not contain non-predictive columns reserved for weighting/response: {forbidden}"
                )
            if allowed_features:
                unknown = sorted(set(value).difference(allowed_features).difference(non_predictive))
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


def proposal_fingerprint(proposal: dict[str, Any]) -> str:
    """Return a stable fingerprint for duplicate detection (float-tolerant)."""

    cfg = dict(proposal.get("config") or {})
    cfg.pop("experiment_name", None)
    cfg.pop("parent_experiment_id", None)
    model = dict(cfg.get("model") or {})
    for key in ("feature_inclusions", "feature_exclusions"):
        if isinstance(model.get(key), list):
            model[key] = sorted(model[key])
    # Round floats to 6 sig figs to avoid numeric-jitter duplicates
    model = {k: _round_if_float(v) for k, v in model.items()}
    cfg["model"] = model
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _round_if_float(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 6)
    return v
