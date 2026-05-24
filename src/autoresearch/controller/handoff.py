"""File-based handoff workflow for external coding agents."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.proposal_schema import allowed_search_space
from autoresearch.controller.workflow import enqueue_proposal_from_file, run_next_queued_proposal
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_branches,
    list_comparisons,
    list_proposals,
    record_proposal,
    update_proposal_status,
)
from autoresearch.utils.io import read_json, write_json


def export_context_bundle(config: ProjectConfig) -> dict[str, Path]:
    """Export JSON and Markdown context for Codex or Claude Code."""

    ensure_project_dirs(config)
    context = build_llm_context(config)
    stamp = _stamp()
    context_path = config.handoff_context_dir / f"context_{stamp}.json"
    latest_context = config.handoff_context_dir / "latest_context.json"
    handoff_path = config.handoff_handoffs_dir / f"handoff_{stamp}.md"
    latest_handoff = config.handoff_handoffs_dir / "latest_handoff.md"
    champion_summary = config.handoff_context_dir / "current_champion_summary.json"
    comparisons_summary = config.handoff_context_dir / "recent_comparisons_summary.json"
    branch_summary = config.handoff_context_dir / "recent_branch_summary.json"

    write_json(context_path, context)
    write_json(latest_context, context)
    handoff_text = render_handoff_markdown(config, context)
    handoff_path.write_text(handoff_text, encoding="utf-8")
    latest_handoff.write_text(handoff_text, encoding="utf-8")
    write_json(champion_summary, context.get("official_champion") or {})
    write_json(comparisons_summary, context.get("recent_comparisons") or [])
    write_json(branch_summary, list_branches(config.registry_path))
    write_proposal_template(config)
    return {
        "context_json": context_path,
        "latest_context_json": latest_context,
        "handoff_markdown": handoff_path,
        "latest_handoff_markdown": latest_handoff,
        "champion_summary": champion_summary,
        "comparisons_summary": comparisons_summary,
        "branch_summary": branch_summary,
    }


def write_proposal_template(config: ProjectConfig) -> dict[str, Path]:
    """Write proposal template, schema description, and instructions."""

    ensure_project_dirs(config)
    context = build_llm_context(config)
    champion = context.get("official_champion") or {}
    template = {
        "proposal_id": "short_unique_id",
        "parent_experiment_id": champion.get("champion_id", "OFFICIAL_CHAMPION_ID"),
        "parent_branch_id": champion.get("branch_id", "main"),
        "branch_action": "new_branch",
        "experiment_name": "concise_experiment_name",
        "rationale": "Why this change is worth trying.",
        "change_summary": "Exact modelling/preprocessing change from parent.",
        "expected_benefit": "Expected improvement mechanism.",
        "key_risk": "Most likely failure mode.",
        "experiment_config": {
            "experiment_name": "concise_experiment_name",
            "model_family": "regularized_linear",
            "target_strategy": "direct_pure_premium",
            "parent_experiment_id": champion.get("champion_id", "OFFICIAL_CHAMPION_ID"),
            "preprocessing": {
                "claim_capping_enabled": True,
                "claim_cap_threshold": 100000,
            },
            "model": {
                "alpha": 1.0,
                "feature_exclusions": [],
            },
        },
    }
    schema = proposal_schema_document(config, context)
    template_path = config.handoff_handoffs_dir / "proposal_template.json"
    schema_path = config.handoff_handoffs_dir / "proposal_schema.json"
    instructions_path = config.handoff_handoffs_dir / "proposal_instructions.md"
    inbox_template_path = config.handoff_proposal_inbox_dir / "proposal_template.json"
    write_json(template_path, template)
    write_json(schema_path, schema)
    write_json(inbox_template_path, template)
    instructions_path.write_text(render_proposal_instructions(config, context), encoding="utf-8")
    return {
        "proposal_template": template_path,
        "proposal_schema": schema_path,
        "proposal_instructions": instructions_path,
        "inbox_template": inbox_template_path,
    }


def ingest_proposals(config: ProjectConfig) -> dict[str, Any]:
    """Validate proposal files from inbox, enqueue valid ones, and move all files."""

    ensure_project_dirs(config)
    valid_dir = config.handoff_proposal_processed_dir / "valid"
    invalid_dir = config.handoff_proposal_processed_dir / "invalid"
    duplicate_dir = config.handoff_proposal_processed_dir / "duplicate"
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for proposal_file in sorted(config.handoff_proposal_inbox_dir.glob("*.json")):
        if proposal_file.name == "proposal_template.json":
            continue
        try:
            result = enqueue_proposal_from_file(config, proposal_file)
            status = result["status"]
            if status == "validated":
                duplicate = detect_duplicate_proposal(config, result["proposal_id"])
                if duplicate and config.deduplication_policy == "reject":
                    reason = f"Duplicate proposal: {duplicate['reason']}"
                    update_proposal_status(config.registry_path, result["proposal_id"], "duplicate", notes=reason)
                    result["status"] = "duplicate"
                    result["duplicate_reason"] = reason
                    write_nonpromotion_summary(
                        config,
                        proposal_id=result["proposal_id"],
                        outcome_type="duplicate",
                        reason=reason,
                        quantitative_signal=None,
                    )
                    destination_dir = duplicate_dir
                else:
                    destination_dir = valid_dir
            else:
                write_nonpromotion_summary(
                    config,
                    proposal_id=result["proposal_id"],
                    outcome_type="invalid",
                    reason="; ".join(result.get("validation_errors", [])) or "Proposal failed validation.",
                    quantitative_signal=None,
                )
                destination_dir = invalid_dir
            result["validation_errors"] = result.get("validation_errors", [])
        except Exception as exc:
            status = "failed"
            destination_dir = invalid_dir
            result = {
                "proposal_id": proposal_file.stem,
                "status": status,
                "validation_errors": [str(exc)],
            }
            record_proposal(
                config.registry_path,
                proposal_id=result["proposal_id"],
                status="failed",
                parent_experiment_id=None,
                parent_branch_id=None,
                branch_id=None,
                experiment_name=None,
                rationale=None,
                change_summary=None,
                expected_benefit=None,
                key_risk=None,
                config=None,
                validation_errors=result["validation_errors"],
                llm_provider="file_handoff",
                llm_model=config.llm_model,
                prompt_path=None,
                response_path=None,
                proposal_path=proposal_file,
                notes="Malformed inbox proposal.",
            )
            write_nonpromotion_summary(
                config,
                proposal_id=result["proposal_id"],
                outcome_type="invalid",
                reason=str(exc),
                quantitative_signal=None,
            )
        destination = destination_dir / f"{_stamp()}_{proposal_file.name}"
        shutil.move(str(proposal_file), destination)
        result["source_path"] = str(proposal_file)
        result["processed_path"] = str(destination)
        results.append(result)
    summary = {
        "ingested_at": _now(),
        "inbox": str(config.handoff_proposal_inbox_dir),
        "processed_dir": str(config.handoff_proposal_processed_dir),
        "results": results,
        "valid_count": sum(1 for item in results if item["status"] == "validated"),
        "duplicate_count": sum(1 for item in results if item["status"] == "duplicate"),
        "invalid_count": sum(1 for item in results if item["status"] not in {"validated", "duplicate"}),
    }
    write_json(config.handoff_results_dir / "latest_ingest_summary.json", summary)
    export_context_bundle(config)
    return summary


def run_latest_proposal_cycle(config: ProjectConfig) -> dict[str, Any]:
    """Ingest inbox proposals, run the next queued proposal, and write cycle summary."""

    ingest_summary = ingest_proposals(config)
    result = run_next_queued_proposal(config)
    summary = {
        "completed_at": _now(),
        "ingest_summary": ingest_summary,
        "cycle_result": result,
        "official_champion": get_official_champion(config.registry_path),
    }
    write_json(config.handoff_results_dir / "latest_cycle_result.json", summary)
    (config.handoff_results_dir / "latest_cycle_result.md").write_text(render_cycle_summary(summary), encoding="utf-8")
    if result.get("decision") != "promote":
        write_nonpromotion_summary(
            config,
            proposal_id=result.get("proposal_id", "unknown"),
            outcome_type=result.get("decision", "inconclusive"),
            reason="Proposal did not pass the promotion gate.",
            quantitative_signal={"comparison_id": result.get("comparison_id")},
        )
    export_context_bundle(config)
    return summary


def inbox_status(config: ProjectConfig) -> dict[str, Any]:
    """Return handoff inbox and processed-folder status."""

    ensure_project_dirs(config)
    valid_dir = config.handoff_proposal_processed_dir / "valid"
    invalid_dir = config.handoff_proposal_processed_dir / "invalid"
    return {
        "provider": config.llm_provider,
        "mode": "file-based handoff" if config.llm_provider == "file_handoff" else config.llm_provider,
        "inbox_dir": str(config.handoff_proposal_inbox_dir),
        "inbox_json_count": len([p for p in config.handoff_proposal_inbox_dir.glob("*.json") if p.name != "proposal_template.json"]),
        "inbox_files": [str(p) for p in sorted(config.handoff_proposal_inbox_dir.glob("*.json"))],
        "processed_valid_count": len(list(valid_dir.glob("*.json"))) if valid_dir.exists() else 0,
        "processed_invalid_count": len(list(invalid_dir.glob("*.json"))) if invalid_dir.exists() else 0,
        "processed_duplicate_count": len(list((config.handoff_proposal_processed_dir / "duplicate").glob("*.json")))
        if (config.handoff_proposal_processed_dir / "duplicate").exists()
        else 0,
        "latest_context": str(config.handoff_context_dir / "latest_context.json"),
        "latest_handoff": str(config.handoff_handoffs_dir / "latest_handoff.md"),
        "latest_cycle_result": str(config.handoff_results_dir / "latest_cycle_result.md"),
    }


def render_handoff_markdown(config: ProjectConfig, context: dict[str, Any]) -> str:
    """Render human-readable handoff instructions for external agents."""

    champion = context.get("official_champion") or {}
    return "\n".join(
        [
            "# Auto-Research Handoff",
            "",
            "You are acting as the external experiment-design agent for this local lab.",
            "Write exactly one valid proposal JSON file into the proposal inbox.",
            "",
            "## Project Goal",
            context["project_goal"],
            "",
            "## Current Official Champion",
            f"- champion_id: `{champion.get('champion_id')}`",
            f"- branch_id: `{champion.get('branch_id')}`",
            f"- reason: {champion.get('reason')}",
            "",
            "## Important Rules",
            "- Do not request or reference `milestone_holdout`.",
            "- Do not redefine evaluation metrics.",
            "- Stay within the allowed search space in `latest_context.json`.",
            "- The Python framework will validate, run, compare, and decide promotion.",
            "- Avoid near-duplicate proposals; inspect recent proposals and non-promotion summaries first.",
            "",
            "## Where To Read",
            f"- Context JSON: `{config.handoff_context_dir / 'latest_context.json'}`",
            f"- Proposal schema: `{config.handoff_handoffs_dir / 'proposal_schema.json'}`",
            f"- Proposal template: `{config.handoff_handoffs_dir / 'proposal_template.json'}`",
            f"- Proposal instructions: `{config.handoff_handoffs_dir / 'proposal_instructions.md'}`",
            "",
            "## Where To Write",
            f"- Write one proposal JSON file to: `{config.handoff_proposal_inbox_dir}`",
            "- Use a unique filename such as `proposal_alpha_10.json`.",
            "",
            "## How To Continue The Session",
            "- After writing a proposal file, run `autoresearch run-session-cycle`.",
            "- If the session reports `waiting_for_proposal`, write the next proposal and run it again.",
            "- Use `autoresearch session-status` to inspect state.",
            "- Use `autoresearch pause-session` or `autoresearch stop-session` when needed.",
            "",
            "## Recent Comparisons",
            json.dumps(context.get("recent_comparisons", []), indent=2, sort_keys=True),
        ]
    ) + "\n"


def render_proposal_instructions(config: ProjectConfig, context: dict[str, Any]) -> str:
    """Instructions focused on producing a proposal file."""

    return "\n".join(
        [
            "# Proposal Instructions",
            "",
            "Create one JSON file matching `proposal_template.json`.",
            "The proposal should be a modest, interpretable challenger to the official champion.",
            "",
            "Do not use Markdown fences in the proposal file. The file content must be JSON only.",
            "Do not repeat a recent executable configuration or identical change summary.",
            "",
            "Allowed search space is embedded below:",
            "",
            "```json",
            json.dumps(context["allowed_search_space"], indent=2, sort_keys=True),
            "```",
            "",
            f"Write proposal files to `{config.handoff_proposal_inbox_dir}`.",
            "Then run `autoresearch run-session-cycle` to let the lab ingest, execute, compare, and refresh handoff files.",
        ]
    ) + "\n"


def proposal_schema_document(config: ProjectConfig, context: dict[str, Any]) -> dict[str, Any]:
    """Export an inspectable proposal schema description."""

    return {
        "type": "object",
        "required": [
            "proposal_id",
            "parent_experiment_id",
            "parent_branch_id",
            "branch_action",
            "experiment_name",
            "rationale",
            "change_summary",
            "expected_benefit",
            "key_risk",
            "experiment_config",
        ],
        "allowed_search_space": context["allowed_search_space"],
        "notes": [
            "proposal_id must use letters, numbers, hyphen, or underscore.",
            "experiment_config.experiment_name must equal experiment_name.",
            "experiment_config.parent_experiment_id must equal parent_experiment_id.",
            "Do not reference milestone_holdout.",
        ],
    }


def render_cycle_summary(summary: dict[str, Any]) -> str:
    result = summary["cycle_result"]
    champion = summary.get("official_champion") or {}
    return "\n".join(
        [
            "# Latest Cycle Result",
            "",
            f"- completed_at: {summary['completed_at']}",
            f"- proposal_id: `{result.get('proposal_id')}`",
            f"- experiment_id: `{result.get('experiment_id')}`",
            f"- comparison_id: `{result.get('comparison_id')}`",
            f"- decision: `{result.get('decision')}`",
            f"- official_champion: `{champion.get('champion_id')}`",
        ]
    ) + "\n"


def detect_duplicate_proposal(config: ProjectConfig, proposal_id: str) -> dict[str, str] | None:
    """Detect obvious duplicate proposals using recent normalised configs."""

    proposals = list_proposals(config.registry_path)
    current = next((item for item in proposals if item["proposal_id"] == proposal_id), None)
    if current is None:
        return None
    current_key = proposal_fingerprint(current)
    current_summary = _normalise_text(current.get("change_summary"))
    checked = 0
    for item in proposals:
        if item["proposal_id"] == proposal_id:
            continue
        if item.get("status") in {"failed", "duplicate"}:
            continue
        checked += 1
        if proposal_fingerprint(item) == current_key:
            return {"matched_proposal_id": item["proposal_id"], "reason": "same executable experiment configuration"}
        if current_summary and current_summary == _normalise_text(item.get("change_summary")):
            return {"matched_proposal_id": item["proposal_id"], "reason": "same change summary"}
        if checked >= config.deduplication_lookback:
            break
    return None


def proposal_fingerprint(proposal: dict[str, Any]) -> str:
    """Return a stable fingerprint for obvious duplicate detection."""

    cfg = dict(proposal.get("config") or {})
    cfg.pop("experiment_name", None)
    cfg.pop("parent_experiment_id", None)
    model = dict(cfg.get("model") or {})
    for key in ("feature_inclusions", "feature_exclusions"):
        if isinstance(model.get(key), list):
            model[key] = sorted(model[key])
    cfg["model"] = model
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def write_nonpromotion_summary(
    config: ProjectConfig,
    *,
    proposal_id: str,
    outcome_type: str,
    reason: str,
    quantitative_signal: dict[str, Any] | None,
) -> dict[str, Path]:
    """Write compact non-promotion summaries for future handoff context."""

    out_dir = config.handoff_results_dir / "non_promoted"
    out_dir.mkdir(parents=True, exist_ok=True)
    proposals = list_proposals(config.registry_path)
    proposal = next((item for item in proposals if item["proposal_id"] == proposal_id), {})
    payload = {
        "proposal_id": proposal_id,
        "outcome_type": outcome_type,
        "reason": reason,
        "experiment_name": proposal.get("experiment_name"),
        "change_summary": proposal.get("change_summary"),
        "expected_benefit": proposal.get("expected_benefit"),
        "key_risk": proposal.get("key_risk"),
        "quantitative_signal": quantitative_signal,
        "agent_guidance": _guidance_for_outcome(outcome_type),
        "created_at": _now(),
    }
    json_path = out_dir / f"{proposal_id}.json"
    md_path = out_dir / f"{proposal_id}.md"
    write_json(json_path, payload)
    md_path.write_text(render_nonpromotion_markdown(payload), encoding="utf-8")
    write_json(config.handoff_results_dir / "latest_nonpromotion_summary.json", payload)
    (config.handoff_results_dir / "latest_nonpromotion_summary.md").write_text(
        render_nonpromotion_markdown(payload),
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": md_path}


def render_nonpromotion_markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Non-Promotion Summary",
            "",
            f"- proposal_id: `{payload['proposal_id']}`",
            f"- outcome_type: `{payload['outcome_type']}`",
            f"- experiment_name: {payload.get('experiment_name')}",
            f"- change_summary: {payload.get('change_summary')}",
            f"- reason: {payload['reason']}",
            f"- quantitative_signal: `{payload.get('quantitative_signal')}`",
            "",
            "## Guidance For Next Proposal",
            payload["agent_guidance"],
        ]
    ) + "\n"


def _guidance_for_outcome(outcome_type: str) -> str:
    if outcome_type == "duplicate":
        return "Avoid repeating the same executable config or change summary; try a materially different hypothesis."
    if outcome_type in {"invalid", "failed"}:
        return "Fix schema, allowed-search-space, or execution issues before proposing similar changes."
    return "Treat this as weak evidence; propose a clearer change with a plausible variance or bias reduction mechanism."


def _normalise_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
