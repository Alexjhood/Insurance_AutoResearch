"""Machine-validated proposal schema and allowed search space."""

from __future__ import annotations

import json
import re
from typing import Any


VALID_STATUSES = {
    "proposed", "validated", "running", "completed", "failed",
    "compared", "promoted", "rejected", "inconclusive", "duplicate",
}

TARGET_COLUMNS = {
    "record_id",
    "claim_count_signal_q",
    "claim_event_count_l",
    "claim_cost_observed_k",
    "claim_cost_capped_active",
}

_SUPPORTED_FAMILIES = {"tweedie_glm", "frequency_severity_glm", "tweedie_gbm", "regularized_linear"}


def allowed_search_space(config, agent_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the explicit search space exposed to proposal generators."""

    ss = config.search_space
    feature_columns = []
    if agent_schema:
        feature_columns = [
            item["name"]
            for item in agent_schema.get("columns", [])
            if item["name"] not in TARGET_COLUMNS and item.get("role") != "target_or_outcome"
        ]

    families = list(ss.get("model_families", ["tweedie_glm", "frequency_severity_glm"]))

    space: dict[str, Any] = {
        "model_families": families,
        "target_strategies": list(ss.get("target_strategies", ["direct_pure_premium", "frequency_severity"])),
        "feature_columns": feature_columns,
        "branch_actions": ["extend_current", "new_branch"],
        "allow_legacy_baselines": bool(ss.get("allow_legacy_baselines", False)),
    }

    # Per-family hyperparameter ranges
    for family in families:
        family_cfg = ss.get(family, {})
        if isinstance(family_cfg, dict) and family_cfg:
            space[f"{family}_params"] = family_cfg

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

    family = exp_config.get("model_family")
    if family not in search_space["model_families"]:
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
        errors.extend(_validate_model_hyperparameters(family, model, search_space))
        allowed_features = set(search_space.get("feature_columns", []))
        for key in ("feature_inclusions", "feature_exclusions"):
            value = model.get(key)
            if value is None:
                continue
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"model.{key} must be a list of strings")
                continue
            if allowed_features:
                unknown = sorted(set(value).difference(allowed_features))
                if unknown:
                    errors.append(f"model.{key} contains unknown features: {unknown}")

    serialised = str(proposal).lower()
    if "milestone_holdout" in serialised:
        errors.append("proposal must not reference milestone_holdout")
    return errors


def _validate_model_hyperparameters(family: str, model: dict[str, Any], space: dict[str, Any]) -> list[str]:
    errors = []
    family_params = space.get(f"{family}_params", {})

    if family == "regularized_linear":
        alpha = model.get("alpha")
        low = float(family_params.get("min_alpha", 0.001))
        high = float(family_params.get("max_alpha", 100.0))
        if not isinstance(alpha, (int, float)) or not low <= float(alpha) <= high:
            errors.append(f"model.alpha must be in [{low}, {high}]")

    elif family == "tweedie_glm":
        alpha = model.get("alpha")
        low = float(family_params.get("min_alpha", 0.001))
        high = float(family_params.get("max_alpha", 10.0))
        if alpha is not None and (not isinstance(alpha, (int, float)) or not low <= float(alpha) <= high):
            errors.append(f"model.alpha must be in [{low}, {high}] for tweedie_glm")
        power = model.get("power")
        allowed_powers = [float(p) for p in family_params.get("power_choices", [1.1, 1.3, 1.5, 1.7, 1.9])]
        if power is not None and float(power) not in allowed_powers:
            errors.append(f"model.power must be one of {allowed_powers}")

    elif family == "frequency_severity_glm":
        for param in ("freq_alpha", "sev_alpha"):
            val = model.get(param)
            low = float(family_params.get(f"min_{param}", 0.001))
            high = float(family_params.get(f"max_{param}", 10.0))
            if val is not None and (not isinstance(val, (int, float)) or not low <= float(val) <= high):
                errors.append(f"model.{param} must be in [{low}, {high}]")

    elif family == "tweedie_gbm":
        if "power" in model:
            errors.append("tweedie_gbm proposals must not set 'power' (it is fixed by the GBM loss)")
        max_iter = model.get("max_iter")
        low_iter = int(family_params.get("min_max_iter", 100))
        high_iter = int(family_params.get("max_max_iter", 2000))
        if max_iter is not None and (not isinstance(max_iter, int) or not low_iter <= max_iter <= high_iter):
            errors.append(f"model.max_iter must be an integer in [{low_iter}, {high_iter}]")
        max_depth = model.get("max_depth")
        allowed_depths = [int(d) for d in family_params.get("max_depth_choices", [3, 5, 7, 9])]
        if max_depth is not None and int(max_depth) not in allowed_depths:
            errors.append(f"model.max_depth must be one of {allowed_depths}")

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
