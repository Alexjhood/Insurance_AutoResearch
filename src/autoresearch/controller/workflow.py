"""Controlled propose -> execute -> compare -> promote workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from autoresearch.comparison_runner import compare_experiments
from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.proposal_schema import allowed_search_space, normalise_proposal, validate_proposal
from autoresearch.controller.proposer import build_prompt, proposer_from_config
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    next_queued_proposal,
    record_proposal,
    set_official_champion,
    update_proposal_status,
    upsert_branch,
)
from autoresearch.experiment_runner import run_experiment
from autoresearch.utils.io import read_json, write_json


def generate_and_enqueue_proposal(config: ProjectConfig) -> dict[str, Any]:
    """Generate one structured proposal, validate it, and enqueue it if valid."""

    ensure_project_dirs(config)
    champion = _require_champion(config)
    context = build_llm_context(config)
    proposer = proposer_from_config(config)
    raw_text: str
    parsed: dict[str, Any] | None
    prompt = build_prompt(context)

    try:
        raw_text, parsed = proposer.propose(context)
    except Exception as exc:
        parsed = None
        raw_text = str(exc)

    proposal_id = _proposal_id(parsed)
    out_dir = config.artifacts_dir / "proposals" / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    context_path = out_dir / "context.json"
    prompt_path = out_dir / "prompt.txt"
    response_path = out_dir / "response.txt"
    proposal_path = out_dir / "proposal.json"
    errors_path = out_dir / "validation_errors.json"

    write_json(context_path, context)
    prompt_path.write_text(prompt, encoding="utf-8")
    response_path.write_text(raw_text, encoding="utf-8")

    if parsed is None:
        errors = ["LLM response was not valid JSON or provider failed"]
        write_json(errors_path, errors)
        record_proposal(
            config.registry_path,
            proposal_id=proposal_id,
            status="failed",
            parent_experiment_id=champion["champion_id"],
            parent_branch_id=champion["branch_id"],
            branch_id=None,
            experiment_name=None,
            rationale=None,
            change_summary=None,
            expected_benefit=None,
            key_risk=None,
            config=None,
            validation_errors=errors,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            prompt_path=prompt_path,
            response_path=response_path,
            proposal_path=None,
            notes="Invalid proposer output.",
        )
        return {"proposal_id": proposal_id, "status": "failed", "validation_errors": errors}

    proposal, errors = _validate_and_normalise(config, parsed, champion)
    write_json(proposal_path, proposal)
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
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        prompt_path=prompt_path,
        response_path=response_path,
        proposal_path=proposal_path,
        notes="Proposal validated and queued." if not errors else "Proposal failed validation.",
    )
    return {"proposal_id": proposal["proposal_id"], "status": status, "validation_errors": errors}


def enqueue_proposal_from_file(config: ProjectConfig, proposal_path: Path) -> dict[str, Any]:
    """Validate and enqueue a manually supplied proposal JSON file."""

    champion = _require_champion(config)
    parsed = read_json(proposal_path)
    proposal, errors = _validate_and_normalise(config, parsed, champion)
    out_dir = config.artifacts_dir / "proposals" / proposal["proposal_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
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
    return {"proposal_id": proposal["proposal_id"], "status": status, "validation_errors": errors}


def run_next_queued_proposal(config: ProjectConfig) -> dict[str, Any]:
    """Run the next validated proposal and gate it against the official champion."""

    champion = _require_champion(config)
    proposal = next_queued_proposal(config.registry_path)
    if proposal is None:
        raise ValueError("No validated proposals are queued")

    proposal_id = proposal["proposal_id"]
    update_proposal_status(config.registry_path, proposal_id, "running", notes="Deterministic execution started.")
    out_dir = config.artifacts_dir / "proposals" / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_config_path = out_dir / "experiment_config.toml"
    experiment_config_path.write_text(_to_toml(proposal["config"]), encoding="utf-8")

    try:
        outputs = run_experiment(config, experiment_config_path)
        experiment_id = outputs["metrics"].parent.name
        update_proposal_status(config.registry_path, proposal_id, "completed", experiment_id=experiment_id)
        upsert_branch(
            config.registry_path,
            branch_id=proposal["branch_id"],
            parent_branch_id=proposal["parent_branch_id"],
            root_experiment_id=proposal["parent_experiment_id"],
            current_experiment_id=experiment_id,
            status="active",
            description=proposal.get("change_summary"),
        )

        comparison_outputs = compare_experiments(config, champion["champion_id"], experiment_id)
        report = read_json(comparison_outputs["promotion_report"])
        comparison_id = report["comparison_id"]
        decision = report["promotion_decision"]
        if decision["decision"] == "promote":
            set_official_champion(
                config.registry_path,
                champion_id=experiment_id,
                branch_id=proposal["branch_id"],
                reason=decision["rationale"],
                action="promoted",
                comparison_id=comparison_id,
                proposal_id=proposal_id,
            )
            update_proposal_status(
                config.registry_path,
                proposal_id,
                "promoted",
                comparison_id=comparison_id,
                notes=decision["rationale"],
            )
        else:
            set_official_champion(
                config.registry_path,
                champion_id=champion["champion_id"],
                branch_id=champion["branch_id"],
                reason=decision["rationale"],
                action="retained",
                comparison_id=comparison_id,
                proposal_id=proposal_id,
            )
            update_proposal_status(
                config.registry_path,
                proposal_id,
                "inconclusive",
                comparison_id=comparison_id,
                notes=decision["rationale"],
            )
        return {
            "proposal_id": proposal_id,
            "experiment_id": experiment_id,
            "comparison_id": comparison_id,
            "decision": decision["decision"],
        }
    except Exception as exc:
        update_proposal_status(config.registry_path, proposal_id, "failed", notes=str(exc))
        raise


def run_one_cycle(config: ProjectConfig) -> dict[str, Any]:
    """Generate, enqueue, execute, compare, and gate one challenger."""

    generated = generate_and_enqueue_proposal(config)
    if generated["status"] != "validated":
        return generated
    executed = run_next_queued_proposal(config)
    return {**generated, **executed}


def run_n_cycles(config: ProjectConfig, count: int) -> list[dict[str, Any]]:
    """Run a bounded number of controlled cycles."""

    if count <= 0:
        raise ValueError("count must be positive")
    results = []
    for _ in range(count):
        results.append(run_one_cycle(config))
    return results


def _validate_and_normalise(
    config: ProjectConfig,
    parsed: dict[str, Any],
    champion: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    parsed = dict(parsed)
    parsed.setdefault("proposal_id", _proposal_id(parsed))
    if not isinstance(parsed.get("experiment_config"), dict):
        parsed["experiment_config"] = {}
    space = allowed_search_space(config, build_llm_context(config).get("agent_schema"))
    errors = validate_proposal(parsed, space)
    if parsed.get("parent_experiment_id") != champion["champion_id"]:
        errors.append("parent_experiment_id must match the current official champion")
    parent_branch = parsed.get("parent_branch_id") or champion["branch_id"]
    if parent_branch != champion["branch_id"]:
        errors.append("parent_branch_id must match the current official champion branch")
    branch_action = parsed.get("branch_action", "extend_current")
    branch_id = parsed.get("branch_id") or (parsed.get("proposal_id") if branch_action == "new_branch" else parent_branch)
    proposal = normalise_proposal(parsed, branch_id=branch_id, parent_branch_id=parent_branch)
    return proposal, errors


def _proposal_id(parsed: dict[str, Any] | None) -> str:
    if parsed and isinstance(parsed.get("proposal_id"), str):
        return parsed["proposal_id"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"invalid_proposal_{stamp}"


def _require_champion(config: ProjectConfig) -> dict[str, Any]:
    champion = get_official_champion(config.registry_path)
    if champion is None:
        raise ValueError("Official champion is not initialised. Run init-official-champion first.")
    return champion


def _to_toml(data: dict[str, Any]) -> str:
    import tomli_w

    # tomli_w requires values to be TOML-compatible; replace None with empty string
    def _sanitise(obj: Any) -> Any:
        if obj is None:
            return ""
        if isinstance(obj, dict):
            return {k: _sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitise(i) for i in obj]
        return obj

    return tomli_w.dumps(_sanitise(data))
