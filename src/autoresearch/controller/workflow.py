"""Controlled propose -> execute -> compare -> promote workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import json
import re
import sqlite3
from pathlib import Path
import shutil
import hashlib
import traceback as _tb_module
from typing import Any

from autoresearch.comparison_runner import compare_experiments, screen_challenger_single_split
from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.milestone import evaluate_on_holdout
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.proposal_schema import (
    TREE_ACTIONS,
    allowed_search_space,
    normalise_proposal,
    validate_proposal,
)

from autoresearch.experiment_registry.registry import (
    get_research_line,
    get_official_champion,
    latest_incomplete_research_log_entry,
    list_artifacts,
    list_proposals,
    list_research_lines,
    list_research_nodes,
    next_queued_proposal,
    record_proposal,
    record_experiment_artifacts,
    set_official_champion,
    park_research_line,
    upsert_research_line,
    upsert_research_node,
    update_proposal_status,
    upsert_branch,
)
from autoresearch.experiment_runner import ComputeBudgetExceeded, PreflightFailed, run_experiment
from autoresearch.evaluation.validation import ValidationRules, validate_experiment_outputs
from autoresearch.run_artifacts import next_iteration_dir, proposal_iteration_dir
from autoresearch.research_log import complete_pending_reflection_from_proposal
from autoresearch.utils.io import read_json, write_json


class ExperimentNeedsRepair(ValueError):
    """Raised when a script attempt failed validation and another attempt is needed."""


MAX_ACTIVE_RESEARCH_LINES = 5


def enqueue_proposal_from_file(config: ProjectConfig, proposal_path: Path) -> dict[str, Any]:
    """Validate and enqueue a manually supplied proposal JSON file."""

    champion = _require_champion(config)
    parsed = read_json(proposal_path)
    parsed.setdefault("proposal_id", _proposal_id(parsed))

    # ── Dedup: skip proposals whose proposal_id already exists in the registry ─
    from autoresearch.experiment_registry.proposals import list_proposals as _list_proposals
    _existing = {p["proposal_id"] for p in _list_proposals(config.registry_path)
                 if p.get("status") not in {"failed", "duplicate"}}
    if parsed["proposal_id"] in _existing:
        record_proposal(
            config.registry_path,
            proposal_id=parsed["proposal_id"],
            status="duplicate",
            parent_experiment_id=parsed.get("parent_experiment_id"),
            parent_branch_id=parsed.get("parent_branch_id"),
            branch_id=parsed.get("branch_id"),
            experiment_name=parsed.get("experiment_name"),
            rationale=parsed.get("rationale"),
            change_summary=parsed.get("change_summary"),
            expected_benefit=parsed.get("expected_benefit"),
            key_risk=parsed.get("key_risk"),
            config=parsed.get("experiment_config"),
            validation_errors=[],
            llm_provider="manual_file",
            llm_model=None,
            prompt_path=None,
            response_path=None,
            proposal_path=proposal_path,
            notes=f"Duplicate: proposal_id {parsed['proposal_id']!r} already exists in registry.",
        )
        _upsert_proposal_node(config, parsed, status="duplicate", outcome_type="duplicate")
        return {"proposal_id": parsed["proposal_id"], "status": "duplicate", "validation_errors": []}
    iteration_dir = next_iteration_dir(config, parsed["proposal_id"])
    out_dir = iteration_dir / "proposal"
    out_dir.mkdir(parents=True, exist_ok=True)
    _materialise_referenced_model_script(parsed, proposal_path, out_dir)
    proposal, errors = _validate_and_normalise(config, parsed, champion)
    stored_path = out_dir / "proposal.json"
    errors_path = out_dir / "validation_errors.json"
    write_json(stored_path, proposal)
    write_json(errors_path, errors)
    status = "validated" if not errors else "failed"
    record_proposal(
        config.registry_path,
        proposal_id=proposal["proposal_id"],
        status=status,
        parent_experiment_id=proposal.get("parent_experiment_id"),
        parent_branch_id=proposal.get("parent_branch_id"),
        branch_id=proposal.get("branch_id"),
        experiment_name=proposal.get("experiment_name"),
        rationale=proposal.get("rationale"),
        change_summary=proposal.get("change_summary"),
        expected_benefit=proposal.get("expected_benefit"),
        key_risk=proposal.get("key_risk"),
        config=proposal.get("experiment_config"),
        validation_errors=errors,
        llm_provider="manual_file",
        llm_model=None,
        prompt_path=None,
        response_path=None,
        proposal_path=stored_path,
        notes="Manual proposal enqueued." if not errors else "Manual proposal failed validation.",
    )
    _upsert_proposal_node(
        config,
        proposal,
        status=status,
        outcome_type=None if not errors else "invalid",
    )
    return {"proposal_id": proposal["proposal_id"], "status": status, "validation_errors": errors}


def _reconcile_stale_running_proposals(config: ProjectConfig) -> None:
    """Flip any proposals stuck in 'running' past the stale threshold to 'failed'."""
    stale_minutes = getattr(config, "running_stale_minutes", 30)
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(config.registry_path) as con:
        con.execute(
            """
            UPDATE proposals
            SET status = 'failed',
                updated_at = CURRENT_TIMESTAMP,
                notes = 'Reconciled: stale running proposal (process likely died).'
            WHERE status = 'running'
            AND updated_at < ?
            """,
            (cutoff,),
        )


def _compute_experiment_budget(config: ProjectConfig) -> float | None:
    """Compute the per-experiment wall-clock budget in seconds for the current run."""
    if not getattr(config, "compute_enforce", True):
        return None
    base_min = getattr(config, "base_budget_minutes", 10)
    inc_min = getattr(config, "budget_increment_minutes", 5)
    per_inc = getattr(config, "experiments_per_increment", 5)
    # Count prior experiments in this run's registry
    n_prior = 0
    if config.registry_path.exists():
        try:
            with sqlite3.connect(config.registry_path) as con:
                row = con.execute("SELECT COUNT(*) FROM experiments").fetchone()
                n_prior = int(row[0]) if row else 0
        except Exception:
            pass
    budget_minutes = base_min + inc_min * (n_prior // per_inc)
    return float(budget_minutes * 60)


def _awaiting_decision_proposals(config: ProjectConfig) -> list[dict[str, Any]]:
    """Return proposals whose comparison still needs an explicit decision."""

    return [
        proposal
        for proposal in list_proposals(config.registry_path)
        if proposal.get("status") == "awaiting_decision"
    ]


def run_next_queued_proposal(config: ProjectConfig) -> dict[str, Any]:
    """Run the next validated proposal and gate it against the official champion."""

    _reconcile_stale_running_proposals(config)
    unresolved = _awaiting_decision_proposals(config)
    if unresolved:
        first = unresolved[0]
        raise ValueError(
            "Cannot run another proposal while a comparison is awaiting decision: "
            f"proposal {first.get('proposal_id')} / comparison {first.get('comparison_id')}. "
            "Review the comparison and call `record-decision` before advancing the run."
        )
    champion = _require_champion(config)
    proposal = next_queued_proposal(config.registry_path)
    if proposal is None:
        raise ValueError("No validated proposals are queued")
    proposal = _hydrate_proposal_from_path(proposal)
    if not complete_pending_reflection_from_proposal(config, proposal):
        raise ValueError(
            "The previous cycle still needs reflection. Include "
            "`previous_cycle_reflection` in this proposal or run "
            "`record-cycle-reflection` before executing another experiment."
        )

    proposal_id = proposal["proposal_id"]
    if proposal.get("parent_experiment_id") != champion["champion_id"]:
        reason = (
            f"Proposal parent {proposal.get('parent_experiment_id')!r} is stale; "
            f"current champion is {champion['champion_id']!r}."
        )
        update_proposal_status(config.registry_path, proposal_id, "stale_parent", notes=reason)
        _upsert_proposal_node(
            config,
            proposal,
            status="stale_parent",
            outcome_type="stale_parent",
            guidance="Refresh context and redesign the proposal from the current champion before running it.",
        )
        _write_nonpromotion_summary(
            config,
            proposal_id=proposal_id,
            outcome_type="stale_parent",
            reason=reason,
            quantitative_signal={"current_champion_id": champion["champion_id"]},
        )
        return {
            "proposal_id": proposal_id,
            "experiment_id": None,
            "comparison_id": None,
            "decision": "auto_reject",
            "auto_rejected": True,
            "auto_reject_reason": reason,
            "metrics_summary": {},
        }

    update_proposal_status(config.registry_path, proposal_id, "running", notes="Deterministic execution started.")
    _upsert_proposal_node(config, proposal, status="running")
    iteration_dir = proposal_iteration_dir(config, proposal)
    proposal_dir = iteration_dir / "proposal"
    proposal_dir.mkdir(parents=True, exist_ok=True)

    compute_budget_sec = _compute_experiment_budget(config)
    try:
        outputs = _run_validated_experiment_attempts(
            config,
            proposal,
            champion["champion_id"],
            proposal_dir,
            iteration_dir,
            compute_budget_sec=compute_budget_sec,
        )
        experiment_id = read_json(outputs["config_snapshot"])["experiment_id"]
        update_proposal_status(config.registry_path, proposal_id, "completed", experiment_id=experiment_id)
        _upsert_proposal_node(config, proposal, status="completed", experiment_id=experiment_id)
        upsert_branch(
            config.registry_path,
            branch_id=proposal["branch_id"],
            parent_branch_id=proposal["parent_branch_id"],
            root_experiment_id=proposal["parent_experiment_id"],
            current_experiment_id=experiment_id,
            status="active",
            description=proposal.get("change_summary"),
        )

        screening: dict[str, Any] | None = None
        local_champion_id = _local_research_line_champion(config, proposal, champion["champion_id"])
        if getattr(config, "screening_enabled", True):
            screening = screen_challenger_single_split(config, local_champion_id, experiment_id)
            screening_path = iteration_dir / "comparison" / "single_split_screening.json"
            screening_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(screening_path, screening)
            record_experiment_artifacts(
                config.registry_path,
                experiment_id,
                {"single_split_screening": screening_path},
            )
            _upsert_proposal_node(
                config,
                proposal,
                status="screened",
                experiment_id=experiment_id,
                screening=screening,
                metrics=_screening_metrics_summary(screening),
            )
            if not screening.get("passed", False):
                reason = screening.get("reason", "Failed single-split screen.")
                diagnostic_comparison = _write_screening_failure_comparison_report(
                    config,
                    champion_id=local_champion_id,
                    challenger_id=experiment_id,
                    iteration_dir=iteration_dir,
                )
                update_proposal_status(
                    config.registry_path,
                    proposal_id,
                    "rejected",
                    experiment_id=experiment_id,
                    notes=f"Auto-rejected by single-split screen: {reason}",
                )
                _upsert_proposal_node(
                    config,
                    proposal,
                    status="rejected",
                    outcome_type="clear_loser",
                    experiment_id=experiment_id,
                    screening=screening,
                    metrics=_screening_metrics_summary(screening),
                    guidance=(
                        "Treat this as evidence about the hypothesis, not just a failed run. "
                        "Use the screening metrics and diagnostics to decide whether a materially "
                        "different child idea is warranted."
                    ),
                )
                _write_nonpromotion_summary(
                    config,
                    proposal_id=proposal_id,
                    outcome_type="clear_loser",
                    reason=f"Auto-rejected by single-split screen: {reason}",
                    quantitative_signal=screening,
                )
                _record_recipe_outcome(config, proposal, experiment_id, "auto_reject", screening)
                return {
                    "proposal_id": proposal_id,
                    "experiment_id": experiment_id,
                    "comparison_id": None,
                    "decision": "auto_reject",
                    "auto_rejected": True,
                    "auto_reject_reason": reason,
                    "screening_report": str(screening_path),
                    "comparison_report": str(diagnostic_comparison.get("html_report", "")),
                    "diagnostic_comparison_report": str(diagnostic_comparison.get("promotion_report", "")),
                    "diagnostic_gate_mode": "single_partition",
                    "local_research_line_champion_id": local_champion_id,
                    "metrics_summary": _screening_metrics_summary(screening),
                }

        comparison_outputs = compare_experiments(
            config,
            champion["champion_id"],
            experiment_id,
            output_dir=iteration_dir / "comparison",
        )
        report = read_json(comparison_outputs["promotion_report"])
        comparison_id = report["comparison_id"]
        # The comparison is written as decision=pending_llm. The framework does NOT
        # decide here — the agent reviews the metrics and calls `record-decision`,
        # which performs the champion/proposal/holdout bookkeeping and re-renders
        # the report. We only link the proposal to the comparison and pause.
        update_proposal_status(
            config.registry_path,
            proposal_id,
            "awaiting_decision",
            experiment_id=experiment_id,
            comparison_id=comparison_id,
            notes="Awaiting LLM decision (call record-decision).",
        )
        # Include key metrics in the return so the agent can assess without
        # additional file reads of the comparison report.
        comp_summary = report.get("comparison_summary") or {}
        guardrail = report.get("guardrail_result") or {}
        metrics_summary = {
            "target_mode": config.target_mode,
            "primary_metric": config.primary_metric,
            "comparison_gate_mode": comp_summary.get("gate_mode"),
            "gate_primary_metric": comp_summary.get("gate_primary_metric"),
            "cv_challenger_score": round(float(comp_summary.get("challenger_mean_score") or 0), 6),
            "cv_champion_score": round(float(comp_summary.get("champion_mean_score") or 0), 6),
            "cv_mean_lift": round(float(comp_summary.get("mean_lift") or 0), 6),
            "cv_win_rate": round(float(comp_summary.get("challenger_win_rate") or 0), 4),
        }
        _upsert_proposal_node(
            config,
            proposal,
            status="awaiting_decision",
            experiment_id=experiment_id,
            comparison_id=comparison_id,
            screening=screening,
            metrics={**_screening_metrics_summary(screening or {}), **metrics_summary},
        )
        return {
            "proposal_id": proposal_id,
            "experiment_id": experiment_id,
            "comparison_id": comparison_id,
            "decision": "pending_llm",
            "advisory_decision": (report.get("advisory_promotion_decision") or {}).get("decision"),
            "guardrail_passed": guardrail.get("passed"),
            "guardrail_failures": guardrail.get("failures", []),
            "escalated": report.get("escalated", False),
            "comparison_report": str(comparison_outputs.get("html_report", "")),
            "screening": screening,
            "local_research_line_champion_id": local_champion_id,
            "metrics_summary": metrics_summary,
        }
    except ExperimentNeedsRepair as exc:
        update_proposal_status(config.registry_path, proposal_id, "needs_repair", notes=str(exc))
        _upsert_proposal_node(config, proposal, status="needs_repair", outcome_type="needs_repair", guidance=str(exc))
        raise
    except Exception as exc:
        reason = str(exc)
        if "Experiment output validation failed" not in reason:
            update_proposal_status(config.registry_path, proposal_id, "failed", notes=reason)
            _upsert_proposal_node(config, proposal, status="failed", outcome_type="system_error", guidance=reason)
            raise
        update_proposal_status(config.registry_path, proposal_id, "rejected", notes=f"Auto-rejected failed run: {reason}")
        _upsert_proposal_node(
            config,
            proposal,
            status="rejected",
            outcome_type="failed_run",
            guidance=(
                "Reflect on the failure mode before proposing a related child idea. "
                "Avoid repeating the same execution, schema, or modelling failure."
            ),
        )
        _write_nonpromotion_summary(
            config,
            proposal_id=proposal_id,
            outcome_type="failed",
            reason=f"Auto-rejected failed run: {reason}",
            quantitative_signal=None,
        )
        return {
            "proposal_id": proposal_id,
            "experiment_id": None,
            "comparison_id": None,
            "decision": "auto_reject",
            "auto_rejected": True,
            "auto_reject_reason": reason,
            "metrics_summary": {},
        }



def _is_blank(value: Any) -> bool:
    """True when a proposal text field is missing or empty."""

    return not isinstance(value, str) or not value.strip()


def _fixed_claim_cap(config: ProjectConfig) -> int:
    """Return the run's fixed claim-cap threshold from the search space."""

    prep = (getattr(config, "search_space", {}) or {}).get("preprocessing", {})
    thresholds = prep.get("claim_cap_thresholds") or [100000]
    return thresholds[0]


def _hydrate_research_line(config: ProjectConfig, parsed: dict[str, Any]) -> None:
    """Fill research-line fields the agent omitted.

    An explicit ``research_line_id`` is honoured: an existing line inherits its
    stored label/hypothesis (so the agent need not restate them), and an unknown
    id is treated as a new line. With no id, the proposal extends the most recent
    active line, or opens a first line when none exist.
    """

    fallback_hypothesis = (
        parsed.get("rationale")
        or parsed.get("expected_learning")
        or "Exploration line for this run."
    )

    def _open_new_line(new_id: str) -> None:
        parsed["research_line_id"] = new_id
        parsed["research_line_action"] = "create_line"
        if _is_blank(parsed.get("research_line_label")):
            parsed["research_line_label"] = parsed.get("experiment_name") or new_id
        if _is_blank(parsed.get("research_line_hypothesis")):
            parsed["research_line_hypothesis"] = fallback_hypothesis

    def _bind_existing_line(line: dict[str, Any]) -> None:
        parsed["research_line_id"] = line["line_id"]
        parsed.setdefault("research_line_action", "extend_line")
        if _is_blank(parsed.get("research_line_label")):
            parsed["research_line_label"] = line.get("label") or line["line_id"]
        if _is_blank(parsed.get("research_line_hypothesis")):
            parsed["research_line_hypothesis"] = line.get("hypothesis") or fallback_hypothesis

    line_id = parsed.get("research_line_id")
    if isinstance(line_id, str) and line_id.strip():
        existing = get_research_line(config.registry_path, line_id)
        if existing is not None:
            _bind_existing_line(existing)
        else:
            # An explicit but unknown id is a new line.
            parsed.setdefault("research_line_action", "create_line")
            if _is_blank(parsed.get("research_line_label")):
                parsed["research_line_label"] = parsed.get("experiment_name") or line_id
            if _is_blank(parsed.get("research_line_hypothesis")):
                parsed["research_line_hypothesis"] = fallback_hypothesis
        return

    # No id supplied. Mint a new id when there is nothing to extend or when the
    # agent explicitly asked to create a line; otherwise extend the most recent
    # active line. This keeps research_line_action and research_line_id
    # consistent — an agent that says `create_line` without an id never gets
    # bound to a pre-existing line.
    active = list_research_lines(config.registry_path, status="active")
    if not active or parsed.get("research_line_action") == "create_line":
        _open_new_line(f"line_{parsed['proposal_id']}"[:80])
    else:
        _bind_existing_line(active[0])


def _hydrate_derived_fields(
    config: ProjectConfig,
    parsed: dict[str, Any],
    champion: dict[str, Any],
    context: dict[str, Any],
) -> None:
    """Fill controller-derivable proposal fields the agent may omit.

    Only absent/blank fields are filled, so a fully specified (legacy) proposal
    is left untouched and any value the agent does supply always wins. Validation
    runs afterwards as a safety net (see ``REQUIRED_PROPOSAL_TEXT_FIELDS``).
    """

    # Identity + parentage: the official champion is the only valid parent.
    parsed.setdefault("proposal_id", _proposal_id(parsed))
    if _is_blank(parsed.get("parent_experiment_id")):
        parsed["parent_experiment_id"] = champion["champion_id"]
    if _is_blank(parsed.get("parent_branch_id")):
        parsed["parent_branch_id"] = champion["branch_id"]
    parsed.setdefault("branch_action", "new_branch")

    # Tree-walk defaults from the top recommended action; an agent that wants a
    # different action supplies it (with tree_policy_override_rationale when it
    # diverges from the recommendation).
    policy = (context.get("research_tree") or {}).get("tree_policy") or {}
    recommended = policy.get("recommended_actions") or []
    rec = recommended[0] if recommended else {}
    if _is_blank(parsed.get("tree_action")):
        parsed["tree_action"] = rec.get("tree_action") or "new_root"
    if _is_blank(parsed.get("selected_tree_action_id")):
        parsed["selected_tree_action_id"] = rec.get("action_id") or "start_first_root"
    if "research_parent_node_id" not in parsed:
        parsed["research_parent_node_id"] = rec.get("parent_node_id")
    if _is_blank(parsed.get("parent_rationale")):
        parsed["parent_rationale"] = (
            f"Follows recommended tree action {parsed['selected_tree_action_id']}."
        )

    _hydrate_research_line(config, parsed)
    if _is_blank(parsed.get("line_membership_rationale")):
        parsed["line_membership_rationale"] = (
            parsed.get("rationale") or "Belongs to this research line."
        )

    # experiment_config: mirror the single experiment_name / parent id the agent
    # gave and apply the run's fixed preprocessing so the agent need not restate
    # any of it.
    exp_config = parsed.get("experiment_config")
    if isinstance(exp_config, dict):
        if _is_blank(exp_config.get("experiment_name")) and not _is_blank(
            parsed.get("experiment_name")
        ):
            exp_config["experiment_name"] = parsed["experiment_name"]
        if _is_blank(exp_config.get("parent_experiment_id")):
            exp_config["parent_experiment_id"] = parsed["parent_experiment_id"]
        prep = exp_config.get("preprocessing")
        if not isinstance(prep, dict) or not prep:
            exp_config["preprocessing"] = {
                "claim_capping_enabled": True,
                "claim_cap_threshold": _fixed_claim_cap(config),
            }


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge JSON-object overrides without mutating either input."""

    result = json.loads(json.dumps(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value))
    return result


def _hydrate_recipe_reference(
    config: ProjectConfig,
    parsed: dict[str, Any],
    champion: dict[str, Any],
    errors: list[str],
) -> None:
    """Expand ``model.recipe_ref=champion`` into a validated full recipe."""

    exp_config = parsed.get("experiment_config")
    model = exp_config.get("model") if isinstance(exp_config, dict) else None
    if not isinstance(model, dict) or "recipe_ref" not in model:
        return

    recipe_ref = model.get("recipe_ref")
    if recipe_ref != "champion":
        errors.append("model.recipe_ref currently supports only 'champion'")
        return
    if "recipe" in model or "script_path" in model or "model_script_path" in model:
        errors.append("model.recipe_ref cannot be combined with model.recipe or model.script_path")
        return

    overrides = model.get("recipe_overrides", {})
    if not isinstance(overrides, dict):
        errors.append("model.recipe_overrides must be an object")
        return

    artifact_path = config.handoff_proposal_inbox_dir / "champion_recipe.json"
    try:
        artifact = read_json(artifact_path)
    except Exception as exc:
        errors.append(f"model.recipe_ref could not load champion_recipe.json: {exc}")
        return
    if artifact.get("experiment_id") != champion.get("champion_id"):
        errors.append("model.recipe_ref points to a stale champion_recipe.json")
        return

    artifact_model = artifact.get("model")
    if not isinstance(artifact_model, dict):
        recipe = artifact.get("recipe")
        artifact_model = {"recipe": recipe} if isinstance(recipe, dict) else {}
    base_recipe = artifact_model.get("recipe")
    if not isinstance(base_recipe, dict) or not base_recipe:
        errors.append("model.recipe_ref champion is not recipe-based")
        return

    resolved_model = {
        key: json.loads(json.dumps(value))
        for key, value in artifact_model.items()
        if key not in {"recipe_ref", "recipe_overrides"}
    }
    resolved_model["recipe"] = _deep_merge(base_recipe, overrides)
    for key in ("feature_inclusions", "feature_exclusions"):
        if key in model:
            value = model[key]
            if value is None:
                resolved_model.pop(key, None)
            else:
                resolved_model[key] = value

    exp_config["model"] = resolved_model
    exp_config["model_family"] = "recipe"
    if _is_blank(exp_config.get("target_strategy")):
        exp_config["target_strategy"] = artifact.get("target_strategy")


def _validate_and_normalise(
    config: ProjectConfig,
    parsed: dict[str, Any],
    champion: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    parsed = dict(parsed)
    parsed.setdefault("proposal_id", _proposal_id(parsed))
    if not isinstance(parsed.get("experiment_config"), dict):
        parsed["experiment_config"] = {}
    context = build_llm_context(config)
    _hydrate_derived_fields(config, parsed, champion, context)
    reference_errors: list[str] = []
    _hydrate_recipe_reference(config, parsed, champion, reference_errors)
    space = allowed_search_space(config, context.get("dataset_schema"))
    errors = reference_errors + validate_proposal(parsed, space)
    _validate_previous_cycle_reflection(config, parsed, errors)
    if parsed.get("parent_experiment_id") != champion["champion_id"]:
        errors.append("parent_experiment_id must match the current official champion")
    parent_branch = parsed.get("parent_branch_id") or champion["branch_id"]
    if parent_branch != champion["branch_id"]:
        errors.append("parent_branch_id must match the current official champion branch")
    _validate_tree_navigation(config, parsed, context, errors)
    _validate_research_line_navigation(config, parsed, errors)
    branch_action = parsed.get("branch_action", "extend_current")
    branch_id = parsed.get("branch_id") or (parsed.get("proposal_id") if branch_action == "new_branch" else parent_branch)
    proposal = normalise_proposal(parsed, branch_id=branch_id, parent_branch_id=parent_branch)
    _ensure_research_line(config, proposal, errors)
    return proposal, errors


def _validate_previous_cycle_reflection(
    config: ProjectConfig,
    parsed: dict[str, Any],
    errors: list[str],
) -> None:
    """Require an auto-reject reflection on the next proposal."""

    pending = latest_incomplete_research_log_entry(config.registry_path)
    if pending is None or pending.get("comparison_id"):
        return
    reflection = parsed.get("previous_cycle_reflection")
    if not isinstance(reflection, dict):
        errors.append(
            "previous_cycle_reflection is required because the prior auto-rejected "
            f"cycle {pending['cycle']} still needs interpretation and next direction"
        )
        return
    if int(reflection.get("cycle") or -1) != int(pending["cycle"]):
        errors.append(f"previous_cycle_reflection.cycle must be {pending['cycle']}")
    if _is_blank(reflection.get("interpretation")):
        errors.append("previous_cycle_reflection.interpretation is required")
    if _is_blank(reflection.get("next")):
        errors.append("previous_cycle_reflection.next is required")


def _validate_research_line_navigation(
    config: ProjectConfig,
    parsed: dict[str, Any],
    errors: list[str],
) -> None:
    line_action = parsed.get("research_line_action")
    line_id = parsed.get("research_line_id")
    if not isinstance(line_id, str) or not line_id.strip():
        return

    existing = get_research_line(config.registry_path, line_id)
    active_lines = list_research_lines(config.registry_path, status="active")
    if line_action == "create_line":
        if existing is not None:
            errors.append("research_line_action=create_line requires a new research_line_id")
        park_line_id = parsed.get("park_research_line_id")
        if len(active_lines) >= MAX_ACTIVE_RESEARCH_LINES:
            if not isinstance(park_line_id, str) or not park_line_id.strip():
                errors.append(
                    f"At most {MAX_ACTIVE_RESEARCH_LINES} active research lines are allowed; "
                    "set park_research_line_id to park one existing active line before creating another."
                )
            elif park_line_id == line_id:
                errors.append("park_research_line_id cannot equal the new research_line_id")
            elif not any(line.get("line_id") == park_line_id for line in active_lines):
                errors.append("park_research_line_id must refer to an existing active research line")
        elif park_line_id:
            if not isinstance(park_line_id, str) or not park_line_id.strip():
                errors.append("park_research_line_id must be a non-empty string or null")
            elif not any(line.get("line_id") == park_line_id for line in active_lines):
                errors.append("park_research_line_id must refer to an existing active research line")
    elif line_action in {"extend_line", "revisit_line", "close_line"}:
        if existing is None:
            errors.append(f"research_line_action={line_action} requires an existing research_line_id")
        elif existing.get("status") == "parked" and not parsed.get("tree_policy_override_rationale"):
            errors.append(
                f"research_line_id {line_id!r} is parked; include tree_policy_override_rationale to revive it."
            )

    research_parent = parsed.get("research_parent_node_id") or parsed.get("parent_node_id")
    if research_parent and existing is not None:
        parent_node = next(
            (node for node in list_research_nodes(config.registry_path) if node.get("node_id") == research_parent),
            None,
        )
        parent_line = parent_node.get("line_id") if parent_node else None
        if parent_line and parent_line != line_id and not parsed.get("tree_policy_override_rationale"):
            errors.append(
                "research_parent_node_id belongs to a different research_line_id; "
                "include tree_policy_override_rationale to cross lines."
            )


def _ensure_research_line(config: ProjectConfig, proposal: dict[str, Any], errors: list[str]) -> None:
    if errors:
        return
    line_id = proposal.get("research_line_id")
    if not line_id:
        return
    action = proposal.get("research_line_action")
    root_node_id = proposal["proposal_id"] if action == "create_line" else None
    if action == "create_line" and proposal.get("park_research_line_id"):
        park_research_line(
            config.registry_path,
            line_id=str(proposal["park_research_line_id"]),
            reason=(
                proposal.get("park_research_line_rationale")
                or f"Parked to open new research line {line_id}."
            ),
            proposal_id=proposal.get("proposal_id"),
        )
    status = "parked" if action == "close_line" else "active"
    upsert_research_line(
        config.registry_path,
        line_id=line_id,
        label=proposal.get("research_line_label") or line_id,
        status=status,
        root_node_id=root_node_id,
        hypothesis=proposal.get("research_line_hypothesis"),
        current_node_id=proposal["proposal_id"] if action == "create_line" else None,
        notes=proposal.get("line_membership_rationale"),
        metadata={
            "created_or_updated_by_proposal": proposal.get("proposal_id"),
            "research_line_action": action,
        },
    )


def _validate_tree_navigation(
    config: ProjectConfig,
    parsed: dict[str, Any],
    context: dict[str, Any],
    errors: list[str],
) -> None:
    """Validate explicit active-run tree-walk fields."""

    tree_action = parsed.get("tree_action")
    research_parent = parsed.get("research_parent_node_id") or parsed.get("parent_node_id")
    nodes = list_research_nodes(config.registry_path)
    node_ids = {node["node_id"] for node in nodes}

    if research_parent is not None:
        if not isinstance(research_parent, str) or not research_parent.strip():
            errors.append("research_parent_node_id must be a non-empty string or null")
        elif research_parent not in node_ids:
            errors.append("research_parent_node_id must refer to a node in this active run's research_tree")

    if tree_action in TREE_ACTIONS and tree_action != "new_root" and not research_parent:
        errors.append(f"tree_action={tree_action} requires research_parent_node_id")

    policy = (context.get("research_tree") or {}).get("tree_policy") or {}
    recommended = policy.get("recommended_actions") or []
    recommended_ids = {str(item.get("action_id")) for item in recommended if item.get("action_id")}
    selected = parsed.get("selected_tree_action_id")
    selected_recommendation = next((item for item in recommended if item.get("action_id") == selected), None)
    if (
        selected_recommendation
        and selected_recommendation.get("tree_action") != tree_action
        and not parsed.get("tree_policy_override_rationale")
    ):
        errors.append("tree_action must match selected_tree_action_id or include tree_policy_override_rationale")
    if (
        tree_action == "new_root"
        and len(nodes) >= 3
        and not parsed.get("tree_policy_override_rationale")
        and not (selected_recommendation and selected_recommendation.get("tree_action") == "new_root")
    ):
        errors.append(
            "tree_action=new_root after the early tree requires tree_policy_override_rationale "
            "explaining the materially new axis"
        )
    if recommended_ids and selected not in recommended_ids and not parsed.get("tree_policy_override_rationale"):
        errors.append("selected_tree_action_id must match research_tree.tree_policy or include tree_policy_override_rationale")


def _run_validated_experiment_attempts(
    config: ProjectConfig,
    proposal: dict[str, Any],
    champion_id: str,
    proposal_dir: Path,
    iteration_dir: Path,
    *,
    compute_budget_sec: float | None = None,
) -> dict[str, Path]:
    """Run a proposal script, validating outputs before comparison.

    When validation fails, the framework writes a repair request and looks for
    the next numbered attempt script. File-handoff agents can respond by
    creating ``model_attempt_2.py`` or ``model_attempt_3.py`` and rerunning the
    cycle. API-driven agents can also provide those attempts up front.
    """

    last_report: dict[str, Any] | None = None
    noise_eps = getattr(config, "repair_noise_floor_eps", 0.002)
    auto_abandon = getattr(config, "repair_auto_abandon_enabled", True)
    # A recipe-origin experiment repairs as a recipe (the framework owns units &
    # calibration); only an explicit escape-hatch script forces the script path.
    recipe_origin = _proposal_recipe(proposal) is not None

    # On a repaired rerun, resume at the prepared attempt instead of re-fitting
    # the known-failing earlier attempts from scratch. Replay their persisted
    # lifts so the auto-abandon history is preserved.
    start_attempt = _resume_attempt_index(proposal_dir)
    attempt_lifts: list[float | None] = _reconstruct_attempt_lifts(proposal_dir, start_attempt)

    for attempt in range(start_attempt, 4):
        attempt_script = proposal_dir / f"model_attempt_{attempt}.py"
        attempt_recipe = proposal_dir / f"recipe_attempt_{attempt}.json"
        cfg = dict(proposal["config"])
        model_cfg = dict(cfg.get("model") or {})
        if attempt_script.exists():
            # An explicit script attempt (the escape hatch) always wins.
            model_cfg.pop("recipe", None)
            model_cfg["script_path"] = attempt_script.name
            if recipe_origin:
                cfg["model_family"] = "scripted_challenger"
        elif attempt_recipe.exists():
            # A corrected recipe supplied by the agent for this attempt.
            model_cfg.pop("script_path", None)
            model_cfg.pop("model_script_path", None)
            model_cfg["recipe"] = read_json(attempt_recipe)
            cfg["model_family"] = "recipe"
        elif attempt == 1:
            raw_script = model_cfg.get("script_path") or model_cfg.get("model_script_path")
            if raw_script:
                raw_path = proposal_dir / str(raw_script)
                if raw_path.exists():
                    model_cfg["script_path"] = raw_path.name
            # else: a recipe-origin proposal already carries model.recipe — run as-is.
        elif last_report is not None:
            _write_repair_request(proposal_dir, attempt, last_report, recipe_origin=recipe_origin)
            break
        cfg["model"] = model_cfg
        experiment_config_path = proposal_dir / f"experiment_config_attempt_{attempt}.toml"
        experiment_config_path.write_text(_to_toml(cfg), encoding="utf-8")

        # ── Run experiment, catching compute/preflight failures ────────────────
        try:
            outputs = run_experiment(
                config,
                experiment_config_path,
                output_dir=iteration_dir / "experiment" / f"attempt_{attempt}",
                compute_budget_sec=compute_budget_sec,
            )
        except (ComputeBudgetExceeded, PreflightFailed) as exc:
            error_type = "compute_budget_exceeded" if isinstance(exc, ComputeBudgetExceeded) else "runtime_exception"
            tb_str = getattr(exc, "traceback_str", _tb_module.format_exc())
            exc_report: dict[str, Any] = {
                "attempt": attempt,
                "valid": False,
                "reason": str(exc)[:400],
                "error_type": error_type,
                "exception_class": type(exc).__name__,
                "traceback": tb_str[-4000:],
                "checks": [],
            }
            attempt_lifts.append(None)
            last_report = exc_report
            if attempt < 3:
                _write_repair_request(proposal_dir, attempt + 1, exc_report, recipe_origin=recipe_origin)
                if not _next_attempt_available(proposal_dir, attempt + 1):
                    raise ExperimentNeedsRepair(
                        f"Attempt {attempt} failed with {type(exc).__name__}: {str(exc)[:200]}. "
                        f"{_repair_handoff_message(proposal_dir, attempt + 1, recipe_origin)}"
                    )
            continue

        report = _validate_attempt_outputs(config, champion_id, outputs, attempt)
        validation_path = Path(outputs["config_snapshot"]).parent / "validation_report.json"
        write_json(validation_path, report)
        outputs["validation_report"] = validation_path
        experiment_id = read_json(outputs["config_snapshot"])["experiment_id"]
        record_experiment_artifacts(
            config.registry_path,
            experiment_id,
            {"validation_report": validation_path},
        )
        if report["valid"]:
            return outputs
        # Produce a diagnostic comparison report even for failed attempts so the
        # agent can see the actual metrics (Gini, lift curve, A/E) before deciding
        # how to repair.  record=False keeps this out of the official comparison
        # registry and avoids inflating the Bonferroni count.
        report = _attach_failed_attempt_comparison(
            config, champion_id, experiment_id, report,
            comparison_dir=iteration_dir / f"failed_attempt_{attempt}_comparison",
        )
        # Record the numeric lift for auto-abandon check.
        raw_lift = report.get("lift_summary", {}) or {}
        attempt_lifts.append(raw_lift.get("lift") if raw_lift else None)
        last_report = report

        # ── Auto-abandon: two consecutive attempts at/below noise floor ────────
        if auto_abandon and len(attempt_lifts) >= 2:
            last_two = attempt_lifts[-2:]
            if all(v is not None and abs(v) < noise_eps for v in last_two):
                abandon_reason = (
                    "Auto-abandoned: two consecutive attempts at/below noise floor "
                    f"(|lift| < {noise_eps}); structural change required."
                )
                last_report = dict(last_report)
                last_report["reason"] = abandon_reason
                raise ValueError(
                    f"Experiment output validation failed after repair attempts: {abandon_reason}"
                )

        if attempt < 3:
            _write_repair_request(proposal_dir, attempt + 1, report, recipe_origin=recipe_origin)
            if not _next_attempt_available(proposal_dir, attempt + 1):
                raise ExperimentNeedsRepair(
                    f"Experiment output validation failed: {report['reason']}. "
                    f"{_repair_handoff_message(proposal_dir, attempt + 1, recipe_origin)}"
                )
    reason = last_report["reason"] if last_report else "Experiment validation failed"
    raise ValueError(f"Experiment output validation failed after repair attempts: {reason}")


def _validate_attempt_outputs(
    config: ProjectConfig,
    champion_id: str,
    outputs: dict[str, Path],
    attempt: int,
) -> dict[str, Any]:
    import pandas as pd

    challenger_predictions = pd.read_parquet(outputs["predictions"])
    champion_predictions = pd.read_parquet(_artifact_path(config, champion_id, "predictions"))
    config_snapshot = read_json(outputs["config_snapshot"])
    model_family = config_snapshot.get("experiment", {}).get("model_family")
    return {
        "attempt": attempt,
        "experiment_id": config_snapshot.get("experiment_id"),
        **validate_experiment_outputs(
            challenger_predictions,
            eval_split=config.ordinary_eval_splits[0],
            primary_metric=config.primary_metric,
            tweedie_power=config.tweedie_power,
            champion_predictions=champion_predictions,
            rules=ValidationRules(require_positive_lift=False),
            allow_constant_predictions=model_family == "global_mean",
            target_mode=config.target_mode,
        ),
    }


def _attach_failed_attempt_comparison(
    config: ProjectConfig,
    champion_id: str,
    experiment_id: str,
    report: dict[str, Any],
    comparison_dir: Path,
) -> dict[str, Any]:
    """Generate a diagnostic comparison report for a failed validation attempt.

    Produces the full HTML comparison report without recording the comparison in
    the registry (record=False), so it does not inflate Bonferroni counts.
    Attaches the report path and a compact metrics summary to the report dict so
    they appear in the repair_request JSON.

    Returns an updated copy of ``report`` with ``comparison_report`` and
    ``metrics_summary`` fields added.
    """
    report = dict(report)
    try:
        artifacts = compare_experiments(
            config,
            champion_id,
            experiment_id,
            output_dir=comparison_dir,
            record=False,
        )
        report["comparison_report"] = str(artifacts.get("html_report", ""))
        # Pull the key numbers directly from the validation's already-computed
        # lift_summary and metric_panel so no extra work is done.
        lift = report.get("lift_summary") or {}
        panel = report.get("metric_panel") or {}
        report["metrics_summary"] = {
            "target_mode": config.target_mode,
            "primary_metric": config.primary_metric,
            "challenger_score": round(float(panel.get(config.primary_metric) or 0), 6),
            "champion_score": round(float(lift.get("champion_score") or 0), 6),
            "raw_lift": round(float(lift.get("lift") or 0), 6),
            "predicted_to_actual_ratio": round(float(panel.get("predicted_to_actual_ratio") or 0), 4),
        }
    except Exception:
        pass  # best-effort; never block the repair flow
    return report


def _write_screening_failure_comparison_report(
    config: ProjectConfig,
    *,
    champion_id: str,
    challenger_id: str,
    iteration_dir: Path,
) -> dict[str, Path]:
    """Write a cheap diagnostic report for challengers that fail screening.

    The low-hurdle screen has already established that the challenger is a
    clear loser, so this report is for diagnosis only. It uses one paired
    sample on the ordinary eval split and is not recorded as an official
    pending comparison.
    """

    out_dir = iteration_dir / "comparison"
    diagnostic_config = replace(
        config,
        gate_mode="single_partition",
        repeated_resamples=1,
        bootstrap_iterations=1,
        resample_fraction=1.0,
        escalation_partitions=0,
    )
    try:
        return compare_experiments(
            diagnostic_config,
            champion_id,
            challenger_id,
            output_dir=out_dir,
            record=False,
        )
    except Exception as exc:
        error_path = out_dir / "diagnostic_comparison_error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            error_path,
            {
                "champion_id": champion_id,
                "challenger_id": challenger_id,
                "reason": str(exc),
                "gate_mode": "single_partition",
                "repeated_resamples": 1,
            },
        )
        return {"diagnostic_comparison_error": error_path}


def _artifact_path(config: ProjectConfig, experiment_id: str, artifact_type: str) -> Path:
    for artifact in list_artifacts(config.registry_path, experiment_id):
        if artifact["artifact_type"] == artifact_type:
            return Path(artifact["path"])
    raise ValueError(f"Experiment {experiment_id} has no {artifact_type!r} artifact")


def _next_attempt_available(proposal_dir: Path, next_attempt: int) -> bool:
    """A repaired attempt is ready if the agent supplied either a corrected recipe
    or an escape-hatch script for it."""
    return (
        (proposal_dir / f"recipe_attempt_{next_attempt}.json").exists()
        or (proposal_dir / f"model_attempt_{next_attempt}.py").exists()
    )


def _resume_attempt_index(proposal_dir: Path) -> int:
    """Where the attempt loop should start when a repaired cycle is rerun.

    A previous process already ran (and recorded) attempts 1..N-1 before stopping
    with ``repair_request_<N>.json``. Restarting at attempt 1 would re-fit the
    known-failing attempt from scratch on every rerun (and again per repair).
    Resume at the highest attempt that has both a pending repair request *and* the
    corrected input the agent was asked to supply; otherwise start fresh at 1.
    """
    for candidate in (3, 2):
        if (proposal_dir / f"repair_request_{candidate}.json").exists() and _next_attempt_available(
            proposal_dir, candidate
        ):
            return candidate
    return 1


def _reconstruct_attempt_lifts(proposal_dir: Path, start_attempt: int) -> list[float | None]:
    """Rebuild the per-attempt lift history for attempts skipped on resume.

    Attempt ``k``'s lift was persisted into ``repair_request_<k+1>.json`` when it
    failed. Replaying it keeps the two-consecutive-attempts auto-abandon check
    correct across a resumed run instead of silently resetting it.
    """
    lifts: list[float | None] = []
    for k in range(1, start_attempt):
        request_path = proposal_dir / f"repair_request_{k + 1}.json"
        lift: float | None = None
        if request_path.exists():
            try:
                lift = read_json(request_path).get("failed_attempt_lift")
            except Exception:
                lift = None
        lifts.append(lift)
    return lifts


def _repair_handoff_message(proposal_dir: Path, next_attempt: int, recipe_origin: bool) -> str:
    if recipe_origin:
        return (
            f"Fix the recipe and write {proposal_dir / f'recipe_attempt_{next_attempt}.json'} "
            "(a JSON file containing just the corrected model.recipe object), then rerun the "
            "proposal. Only write a model_attempt_*.py script if the recipe vocabulary genuinely "
            "cannot express the fix."
        )
    return (
        f"Write {proposal_dir / f'model_attempt_{next_attempt}.py'} and rerun the proposal."
    )


def _write_repair_request(
    proposal_dir: Path,
    next_attempt: int,
    report: dict[str, Any],
    *,
    recipe_origin: bool = False,
) -> Path:
    error_type = report.get("error_type", "output_validation_failed")
    lift_summary = report.get("lift_summary")
    failed_attempt_lift = (
        lift_summary.get("lift") if isinstance(lift_summary, dict) else None
    )
    payload: dict[str, Any] = {
        "next_attempt": next_attempt,
        "error_type": error_type,
        "reason": report.get("reason"),
        "failed_checks": [check for check in report.get("checks", []) if not check.get("passed")],
        "metrics_summary": report.get("metrics_summary"),
        "comparison_report": report.get("comparison_report"),
        # Lift of the attempt that just failed (= attempt next_attempt - 1). Lets a
        # resumed run reconstruct the auto-abandon history without re-fitting it.
        "failed_attempt_lift": failed_attempt_lift,
    }
    if recipe_origin:
        payload["repair_kind"] = "recipe"
        payload["write_recipe"] = f"recipe_attempt_{next_attempt}.json"
        payload["escape_hatch_script"] = f"model_attempt_{next_attempt}.py"
        payload["instruction"] = (
            "This experiment is a declarative recipe. Fix the RECIPE and write the corrected "
            f"recipe object to recipe_attempt_{next_attempt}.json (just the model.recipe object). "
            "The framework owns feature handling, units, and calibration — do NOT add exposure "
            "conversion or calibration. Only fall back to writing "
            f"model_attempt_{next_attempt}.py (a fit_predict script) if the recipe vocabulary "
            "genuinely cannot express the fix. Read comparison_report (HTML) for Gini/lift detail first."
        )
    else:
        payload["repair_kind"] = "script"
        payload["write_script"] = f"model_attempt_{next_attempt}.py"
        payload["instruction"] = (
            "Revise the model script to fix the failed checks. Keep the same fit_predict "
            "interface and do not access holdout data. The next run will use this script. "
            "Read comparison_report (HTML) for full Gini curves and lift charts before deciding "
            "on the fix strategy."
        )
    if error_type in ("runtime_exception", "compute_budget_exceeded"):
        payload["exception_class"] = report.get("exception_class", "")
        payload["traceback"] = report.get("traceback", "")
    path = proposal_dir / f"repair_request_{next_attempt}.json"
    write_json(path, payload)
    return path


def _materialise_referenced_model_script(parsed: dict[str, Any], source_proposal_path: Path, proposal_dir: Path) -> None:
    model = parsed.setdefault("experiment_config", {}).setdefault("model", {})
    raw = model.get("script_path") or model.get("model_script_path")
    if not raw:
        return
    source = Path(str(raw))
    if not source.is_absolute():
        source = source_proposal_path.parent / source
    if not source.exists():
        raise FileNotFoundError(f"Referenced model script does not exist: {source}")
    destination = proposal_dir / "model_attempt_1.py"
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    model["script_path"] = destination.name
    model["script_sha256"] = _sha256(destination)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _slugify_id(text: str) -> str:
    """Reduce free text to the proposal-id charset ([A-Za-z0-9_-])."""

    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return slug[:60]


def _proposal_id(parsed: dict[str, Any] | None) -> str:
    """Return the proposal id, deriving a meaningful one when none was supplied.

    Under the minimal proposal contract the agent legitimately omits
    ``proposal_id`` (the controller derives it), so the fallback is built from the
    agent's ``experiment_name`` plus a timestamp uniquifier rather than being
    labelled ``invalid_*``. Only a proposal with neither id nor usable name — i.e.
    a genuinely malformed one — falls back to a neutral stamped id.
    """

    if parsed and isinstance(parsed.get("proposal_id"), str) and parsed["proposal_id"].strip():
        return parsed["proposal_id"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = parsed.get("experiment_name") if parsed else None
    slug = _slugify_id(name) if isinstance(name, str) else ""
    if slug:
        return f"{slug}_{stamp}"
    return f"proposal_{stamp}"


def _hydrate_proposal_from_path(proposal: dict[str, Any]) -> dict[str, Any]:
    """Merge queue registry fields with the original proposal JSON metadata."""

    result = dict(proposal)
    proposal_path = result.get("proposal_path")
    if proposal_path:
        try:
            stored = read_json(Path(proposal_path))
            if isinstance(stored, dict):
                merged = dict(stored)
                merged.update({key: value for key, value in result.items() if value is not None})
                if "config" not in merged and "experiment_config" in merged:
                    merged["config"] = merged["experiment_config"]
                if "experiment_config" not in merged and "config" in merged:
                    merged["experiment_config"] = merged["config"]
                return merged
        except Exception:
            pass
    if "experiment_config" not in result and "config" in result:
        result["experiment_config"] = result["config"]
    return result


def _local_research_line_champion(
    config: ProjectConfig,
    proposal: dict[str, Any],
    fallback_champion_id: str,
) -> str:
    """Return the local incumbent used for cheap screening."""

    line_id = proposal.get("research_line_id")
    if not line_id:
        return fallback_champion_id
    line = get_research_line(config.registry_path, line_id)
    if not line:
        return fallback_champion_id
    if line.get("status") != "active":
        return fallback_champion_id
    local_id = line.get("current_experiment_id") or line.get("best_experiment_id")
    return str(local_id) if local_id else fallback_champion_id


def _upsert_proposal_node(
    config: ProjectConfig,
    proposal: dict[str, Any],
    *,
    status: str,
    outcome_type: str | None = None,
    experiment_id: str | None = None,
    comparison_id: str | None = None,
    screening: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    guidance: str | None = None,
) -> None:
    """Mirror proposal lifecycle into the active run's research tree."""

    proposal_id = proposal.get("proposal_id") or _proposal_id(proposal)
    cfg = proposal.get("config") or proposal.get("experiment_config") or {}
    tags = proposal.get("exploration_tags")
    if tags is not None and not isinstance(tags, list):
        tags = [str(tags)]
    tree_metadata = {
        "tree_action": proposal.get("tree_action"),
        "selected_tree_action_id": proposal.get("selected_tree_action_id"),
        "parent_rationale": proposal.get("parent_rationale"),
        "exploration_axis": proposal.get("exploration_axis"),
        "approach_family": proposal.get("approach_family"),
        "target_framing": proposal.get("target_framing"),
        "feature_representation": proposal.get("feature_representation"),
        "expected_learning": proposal.get("expected_learning"),
        "tree_policy_override_rationale": proposal.get("tree_policy_override_rationale"),
        "research_line_action": proposal.get("research_line_action"),
        "research_line_label": proposal.get("research_line_label"),
        "research_line_hypothesis": proposal.get("research_line_hypothesis"),
        "line_membership_rationale": proposal.get("line_membership_rationale"),
    }
    tree_metadata = {key: value for key, value in tree_metadata.items() if value is not None}
    if not tree_metadata:
        tree_metadata = None
    upsert_research_node(
        config.registry_path,
        node_id=proposal_id,
        line_id=proposal.get("research_line_id"),
        proposal_id=proposal_id,
        parent_node_id=proposal.get("research_parent_node_id") or proposal.get("parent_node_id"),
        parent_experiment_id=proposal.get("parent_experiment_id") or cfg.get("parent_experiment_id"),
        experiment_id=experiment_id,
        comparison_id=comparison_id,
        branch_id=proposal.get("branch_id"),
        status=status,
        outcome_type=outcome_type,
        hypothesis=proposal.get("rationale"),
        change_summary=proposal.get("change_summary"),
        expected_benefit=proposal.get("expected_benefit"),
        key_risk=proposal.get("key_risk"),
        tags=tags,
        tree_metadata=tree_metadata,
        screening=screening,
        metrics=metrics,
        guidance=guidance,
    )


def _proposal_recipe(proposal: dict[str, Any]) -> dict[str, Any] | None:
    """Return the recipe object from a proposal's experiment_config, if any."""
    cfg = proposal.get("experiment_config") or proposal.get("config") or {}
    model = cfg.get("model") if isinstance(cfg, dict) else None
    recipe = model.get("recipe") if isinstance(model, dict) else None
    return recipe if isinstance(recipe, dict) and recipe else None


def _record_recipe_outcome(
    config: ProjectConfig,
    proposal: dict[str, Any],
    experiment_id: str,
    outcome: str,
    screening: dict[str, Any] | None = None,
) -> None:
    """Record a recipe's terminal outcome to the reuse library. Best-effort.

    Covers terminal paths that bypass ``record_decision`` (notably single-split
    auto-rejection), so a losing recipe is not left mis-recorded as ``completed``.
    """
    recipe = _proposal_recipe(proposal)
    if recipe is None:
        return
    try:
        from autoresearch.models.recipe_library import record_recipe

        score = None
        if screening:
            score = screening.get("challenger_score")
        cfg = proposal.get("experiment_config") or proposal.get("config") or {}
        model = cfg.get("model") if isinstance(cfg, dict) else {}
        record_recipe(
            config,
            recipe,
            experiment_id=experiment_id,
            outcome=outcome,
            score=score,
            target_strategy=cfg.get("target_strategy"),
            feature_inclusions=model.get("feature_inclusions"),
            feature_exclusions=model.get("feature_exclusions"),
            model_spec={"target_strategy": cfg.get("target_strategy"), **model},
        )
    except Exception:
        pass


def _screening_metrics_summary(screening: dict[str, Any]) -> dict[str, Any]:
    if not screening:
        return {}
    keys = ["gate_mode", "gate_metric", "target_mode", "passed", "overlap_rows"]
    renamed = {
        "champion_score": "split_champion_score",
        "challenger_score": "split_challenger_score",
        "lift": "split_lift",
        "relative_lift": "split_relative_lift",
    }
    result: dict[str, Any] = {}
    for key in keys:
        if key in screening:
            value = screening[key]
            result[key] = round(float(value), 6) if isinstance(value, float) else value
    for source, target in renamed.items():
        if source in screening:
            value = screening[source]
            result[target] = round(float(value), 6) if isinstance(value, float) else value
    return result


def _write_nonpromotion_summary(
    config: ProjectConfig,
    *,
    proposal_id: str,
    outcome_type: str,
    reason: str,
    quantitative_signal: dict[str, Any] | None,
) -> None:
    # Lazy import avoids a module import cycle: handoff imports this workflow.
    from autoresearch.controller.handoff import write_nonpromotion_summary

    write_nonpromotion_summary(
        config,
        proposal_id=proposal_id,
        outcome_type=outcome_type,
        reason=reason,
        quantitative_signal=quantitative_signal,
    )


def _require_champion(config: ProjectConfig) -> dict[str, Any]:
    champion = get_official_champion(config.registry_path)
    if champion is None:
        raise ValueError("Official champion is not initialised. Run init-official-champion first.")
    return champion


def _to_toml(data: dict[str, Any]) -> str:
    import tomli_w

    def _sanitise(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [_sanitise(i) for i in obj if i is not None]
        return obj

    return tomli_w.dumps(_sanitise(data))
