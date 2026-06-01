"""File-based handoff workflow for external coding agents."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.workflow import enqueue_proposal_from_file, run_next_queued_proposal
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_proposals,
    record_proposal,
    update_proposal_status,
)
from autoresearch.utils.io import read_json, write_json


def export_context_bundle(config: ProjectConfig) -> dict[str, Path]:
    """Export JSON and Markdown context for Codex or Claude Code."""

    ensure_project_dirs(config)
    context = build_llm_context(config)
    latest_context = config.handoff_context_dir / "latest_context.json"
    latest_handoff = config.handoff_handoffs_dir / "latest_handoff.md"

    write_json(latest_context, context)
    latest_handoff.write_text(render_handoff_markdown(config, context), encoding="utf-8")
    template_outputs = _write_proposal_template_files(config, context)
    return {
        "latest_context_json": latest_context,
        "latest_handoff_markdown": latest_handoff,
        **template_outputs,
    }


def write_proposal_template(config: ProjectConfig) -> dict[str, Path]:
    """Write proposal template and schema description."""

    ensure_project_dirs(config)
    context = build_llm_context(config)
    return _write_proposal_template_files(config, context)


def _write_proposal_template_files(config: ProjectConfig, context: dict[str, Any]) -> dict[str, Path]:
    """Write proposal template/schema files from an already-built context."""

    template = _proposal_template(config, context)
    schema = proposal_schema_document(config, context)
    template_path = config.handoff_handoffs_dir / "proposal_template.json"
    schema_path = config.handoff_handoffs_dir / "proposal_schema.json"
    inbox_template_path = config.handoff_proposal_inbox_dir / "proposal_template.json"
    write_json(template_path, template)
    write_json(schema_path, schema)
    write_json(inbox_template_path, template)
    return {
        "proposal_template": template_path,
        "proposal_schema": schema_path,
        "inbox_template": inbox_template_path,
    }


def _proposal_template(config: ProjectConfig, context: dict[str, Any]) -> dict[str, Any]:
    """Return a proposal template aligned to the current champion and tree policy."""

    champion = context.get("official_champion") or {}
    recommended_actions = ((context.get("research_tree") or {}).get("tree_policy") or {}).get("recommended_actions") or []
    recommended_action = recommended_actions[0] if recommended_actions else {}
    research_lines = (context.get("research_lines") or {}).get("active_lines") or []
    selected_line = research_lines[0] if research_lines else {}
    return {
        "proposal_id": "short_unique_id",
        "parent_experiment_id": champion.get("champion_id", "OFFICIAL_CHAMPION_ID"),
        "parent_branch_id": champion.get("branch_id", "main"),
        "research_line_action": "extend_line" if selected_line else "create_line",
        "research_line_id": selected_line.get("line_id", "line_short_name"),
        "research_line_label": selected_line.get("label", "Short research line label"),
        "research_line_hypothesis": selected_line.get("hypothesis", "What this line is trying to learn."),
        "line_membership_rationale": "Why this proposal belongs in this research line.",
        "tree_action": recommended_action.get("tree_action", "new_root"),
        "research_parent_node_id": recommended_action.get("parent_node_id"),
        "selected_tree_action_id": recommended_action.get("action_id", "start_first_root"),
        "parent_rationale": "Why this parent or new root is the right next tree step.",
        "exploration_axis": "model_family",
        "approach_family": "Broad approach family, not a prescribed implementation.",
        "target_framing": "direct_pure_premium",
        "feature_representation": "raw",
        "expected_learning": "What this experiment should teach even if it fails.",
        "branch_action": "new_branch",
        "experiment_name": "concise_experiment_name",
        "rationale": "Why this change is worth trying.",
        "change_summary": "Exact modelling/preprocessing change from parent.",
        "expected_benefit": "Expected improvement mechanism.",
        "key_risk": "Most likely failure mode.",
        "experiment_config": {
            "experiment_name": "concise_experiment_name",
            "model_family": "scripted_challenger",
            "target_strategy": "direct_pure_premium",
            "parent_experiment_id": champion.get("champion_id", "OFFICIAL_CHAMPION_ID"),
            "preprocessing": {
                "claim_capping_enabled": True,
                "claim_cap_threshold": 100000,
            },
            "model": {
                "script_path": "model.py",
                "feature_exclusions": [],
            },
        },
    }


def ingest_proposals(config: ProjectConfig) -> dict[str, Any]:
    """Validate inbox proposals while keeping at most one active queued proposal."""

    ensure_project_dirs(config)
    valid_dir = config.handoff_proposal_processed_dir / "valid"
    invalid_dir = config.handoff_proposal_processed_dir / "invalid"
    duplicate_dir = config.handoff_proposal_processed_dir / "duplicate"
    valid_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    accepted_valid = _active_queued_count(config) > 0
    for proposal_file in sorted(config.handoff_proposal_inbox_dir.glob("*.json")):
        if proposal_file.name == "proposal_template.json":
            continue
        if accepted_valid:
            results.append({
                "proposal_id": proposal_file.stem,
                "status": "deferred_pending_context_refresh",
                "validation_errors": [],
                "source_path": str(proposal_file),
                "processed_path": None,
                "reason": "A validated proposal is already queued; refresh context before ingesting another.",
            })
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
                    accepted_valid = True
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
                llm_model=None,
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
        "deferred_count": sum(1 for item in results if item["status"] == "deferred_pending_context_refresh"),
        "invalid_count": sum(
            1
            for item in results
            if item["status"] not in {"validated", "duplicate", "deferred_pending_context_refresh"}
        ),
    }
    if summary["deferred_count"]:
        summary["agent_warning"] = (
            "Additional proposal JSON files were left in the inbox because one proposal is already active. "
            "Refresh context before deciding whether to keep or rewrite them."
        )
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
    if result.get("decision") not in {"promote", "auto_reject"}:
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
    """Render a self-contained handoff file for external agents.

    Embeds the proposal template, key constraints, and champion metrics inline
    so the agent can start writing a proposal immediately without reading
    additional files (proposal_schema.json, proposal_template.json, context.json).
    """
    champion = context.get("official_champion") or {}
    champion_id = champion.get("champion_id", "FILL_IN_CHAMPION_ID")
    branch_id = champion.get("branch_id", "main")

    # Find champion Gini from recent_experiments list
    gini_str = ""
    for exp in context.get("recent_experiments") or []:
        if exp.get("experiment_id") == champion_id:
            score = exp.get("mean_score")
            if score is not None:
                gini_str = f", Gini {float(score):.4f}"
            break

    search_space = context.get("allowed_search_space") or {}
    features = search_space.get("feature_columns") or []
    target_strategies = search_space.get("target_strategies") or ["direct_pure_premium"]
    target_mode = search_space.get("active_target_mode") or config.target_mode
    return_instruction = (
        "Return expected claim counts (not rates): if predicting annual claim frequency, multiply by "
        "`score['exposure_term_a']`"
        if target_mode == "frequency"
        else "Return claim costs (not rates): if predicting pure premium, multiply by `score['exposure_term_a']`"
    )
    feature_list = ", ".join(f"`{f}`" for f in features)
    tree = context.get("research_tree") or {}
    research_lines = context.get("research_lines") or {}
    node_lines = _render_tree_node_lines(tree.get("recent_nodes") or [])
    research_line_lines = _render_research_line_lines(research_lines.get("active_lines") or [])
    recommended_actions = (tree.get("tree_policy") or {}).get("recommended_actions") or []
    recommended_action = recommended_actions[0] if recommended_actions else {}
    action_lines = _render_tree_policy_lines(recommended_actions)
    deferred_lines = _render_deferred_proposal_warning(config)

    template_json = json.dumps({
        "proposal_id": "<short_unique_id>",
        "parent_experiment_id": champion_id,
        "parent_branch_id": branch_id,
        "research_line_action": "<create_line|extend_line|revisit_line|close_line>",
        "research_line_id": "<line_short_name>",
        "research_line_label": "<human readable line label>",
        "research_line_hypothesis": "<what this local line is exploring>",
        "line_membership_rationale": "<why this proposal belongs in this line>",
        "tree_action": recommended_action.get("tree_action", "<tree_action>"),
        "research_parent_node_id": recommended_action.get("parent_node_id"),
        "selected_tree_action_id": recommended_action.get("action_id", "<recommended action_id>"),
        "parent_rationale": "<why this tree parent or new root is appropriate>",
        "exploration_axis": "<model_family|target_framing|feature_representation|calibration|hyperparameter|diagnostic_probe|data_slice|ensemble|other>",
        "approach_family": "<broad approach family, without relying on another run's details>",
        "target_framing": "<target framing used by the proposal>",
        "feature_representation": "<feature representation used by the proposal>",
        "expected_learning": "<what this experiment should teach even if it fails>",
        "branch_action": "new_branch",
        "experiment_name": "<concise_name>",
        "rationale": "<why this change is worth trying>",
        "change_summary": "<exact modelling/preprocessing change from parent>",
        "expected_benefit": "<expected improvement mechanism>",
        "key_risk": "<most likely failure mode>",
        "experiment_config": {
            "experiment_name": "<concise_name>",
            "model_family": "scripted_challenger",
            "target_strategy": "direct_pure_premium",
            "parent_experiment_id": champion_id,
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": {"script_path": "model_<name>.py"},
        },
    }, indent=2)

    from autoresearch.memory import resolve_memory_access
    from autoresearch.memory.store import default_playbook_dir

    _memory_access = resolve_memory_access(config)
    _playbook_link_lines: list[str] = []
    if _memory_access in ("own", "all"):
        _playbook_base = default_playbook_dir()
        _suffix = ""
        if _memory_access == "own":
            manifest_path = config.artifacts_dir / "run_manifest.json"
            try:
                import json as _json
                _manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
                _ident = _manifest.get("model_identity") or {}
                _provider = (_ident.get("provider") or "").lower().strip()
                _name = (_ident.get("name") or "").lower().strip()
                if _provider and _name:
                    _own_id = f"{_provider}/{_name}"
                    _own_suffix = f"_{_own_id.replace('/', '_')}"
                    _own_path = _playbook_base / f"latest{_own_suffix}.md"
                    if _own_path.exists():
                        _suffix = _own_suffix
            except (OSError, Exception):
                pass
        _playbook_path = _playbook_base / f"latest{_suffix}.md"
        if _playbook_path.exists():
            _playbook_link_lines = [
                "",
                "## Research playbook",
                "",
                f"Cross-run memory access is enabled (scope: `{_memory_access}`). "
                "A compiled playbook of verified insights is available:",
                f"`{_playbook_path}`",
                "",
                "Query the memory store for more detail:",
                "```bash",
                "autoresearch memory query --insights",
                "autoresearch memory query --analysis peak-gini-by-framing",
                "```",
            ]

    lines = [
        "# Auto-Research Handoff",
        "",
        "Read `AGENT.md` for the full operating manual.",
        "",
        "## Current state",
        "",
        f"- **Champion**: `{champion_id}` (branch `{branch_id}`{gini_str})",
        f"- **Inbox**: `{config.handoff_proposal_inbox_dir}`  ← write proposal JSON + model script here",
        f"- **Next command**: `autoresearch --track {config.track_id} --run-id {config.run_id} run-latest-proposal-cycle`",
        "",
        "## Proposal quick-start",
        "",
        f"Copy this to `{config.handoff_proposal_inbox_dir}/proposal_<name>.json` and fill in the `<...>` fields.",
        "Also write `model_<name>.py` (same directory) with a `fit_predict(train, score, ...)` function.",
        "Write exactly one proposal for this context refresh.",
        *deferred_lines,
        "",
        "```json",
        template_json,
        "```",
        "",
        "## Key constraints",
        "",
        f"- **Target mode**: `{target_mode}`",
        f"- **Features available**: {feature_list}",
        "- **Exposure policy**: `exposure_term_a` is not a predictive feature. Use it only for sample weights, response denominators, and multiplying predicted rates back to target totals.",
        f"- **Target strategies**: {', '.join(f'`{s}`' for s in target_strategies)}",
        "- **Claim cap**: `100000` (fixed — never change `claim_cap_threshold`)",
        "- **`model.py` interface**: must expose `fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters) -> tuple[np.ndarray, dict]`",
        f"- **{return_instruction}**",
        "- **Always apply** `apply_training_calibration` from `autoresearch.models.calibration` before returning",
        "- **Never reference** `milestone_holdout`, `holdout_vault`, or `AUTORESEARCH_MILESTONE_TOKEN`",
        "",
        "## Exploration tree",
        "",
        "- Scope: active run only. Do not use results from other runs as proposal evidence.",
        "- Choose `research_parent_node_id` from this tree when the next idea builds on a prior hypothesis; use `null` only for a genuinely new line of attack.",
        "- Cross-run memory, when enabled, may inform broad strategy but must not supply tree parent IDs or evidence for this run.",
        "- Choose `research_line_action` and `research_line_id` so the run maintains a small number of coherent local research lines.",
        "- A future `record-decision` can be `promote` for the whole run, `local_promote` for this line only, or `reject`.",
        "- The single-split hurdle uses this line's local incumbent where available; full comparison still reports against the official champion.",
        "- Set `tree_action`, `selected_tree_action_id`, and `parent_rationale`; if you ignore the recommended action, include `tree_policy_override_rationale`.",
        "- Prefer genuine exploration over repetitive small retunes. A useful child idea should change the hypothesis, representation, target framing, or error mode it addresses.",
        "- Clear failures and auto-rejections are evidence. Reflect on them, then branch only when the child idea is materially different.",
        "",
        *research_line_lines,
        "",
        *action_lines,
        "",
        *node_lines,
        *_playbook_link_lines,
        "",
        "## Context JSON (full detail)",
        "",
        f"`{config.handoff_context_dir / 'latest_context.json'}`",
    ]
    return "\n".join(lines) + "\n"


def proposal_schema_document(config: ProjectConfig, context: dict[str, Any]) -> dict[str, Any]:
    """Export an inspectable proposal schema description."""

    return {
        "type": "object",
        "required": [
            "proposal_id",
            "parent_experiment_id",
            "parent_branch_id",
            "branch_action",
            "research_line_action",
            "research_line_id",
            "research_line_label",
            "research_line_hypothesis",
            "line_membership_rationale",
            "tree_action",
            "selected_tree_action_id",
            "parent_rationale",
            "exploration_axis",
            "approach_family",
            "target_framing",
            "feature_representation",
            "expected_learning",
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
            "experiment_config.model.script_path is required for non-global_mean autonomous experiments.",
            "Do not use exposure_term_a as a predictive feature; it is reserved for weights and response calculations.",
            "research_parent_node_id is optional and may only point to a node from this active run's research_tree.",
            "tree_action=new_root may use research_parent_node_id=null; all other tree actions must point to a valid active-run node.",
            "selected_tree_action_id should match a recommended action from research_tree.tree_policy, unless tree_policy_override_rationale explains the deviation.",
            "research_line_action=create_line must use a new research_line_id; extend_line/revisit_line/close_line must use an existing active-run line.",
            "Keep the active run to a small number of coherent research lines; use local promotion for progress inside a line without replacing the global champion.",
            "Only one validated proposal is ingested per context refresh while a proposal is queued or awaiting decision; additional proposal JSON files remain deferred in the inbox.",
            f"Active target_mode is {config.target_mode}; use frequency only when the run was explicitly configured for it.",
            "Do not reference milestone_holdout.",
        ],
    }


def _render_tree_node_lines(nodes: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return ["No research-tree nodes yet. Start with a small, well-motivated first hypothesis."]
    lines = ["Recent active-run nodes:"]
    for node in nodes[:8]:
        metrics = node.get("metrics") or {}
        lift = metrics.get("lift") if "lift" in metrics else metrics.get("mean_lift")
        lift_text = f", lift={lift}" if lift is not None else ""
        summary = node.get("change_summary") or node.get("expected_benefit") or ""
        if len(summary) > 120:
            summary = summary[:120] + "..."
        lines.append(
            f"- `{node.get('node_id')}` status={node.get('status')}"
            f", outcome={node.get('outcome_type')}{lift_text}: {summary}"
        )
    return lines


def _render_research_line_lines(lines_in: list[dict[str, Any]]) -> list[str]:
    if not lines_in:
        return ["Research lines: none yet. Use `research_line_action=create_line` for the first coherent line of attack."]
    lines = ["Active research lines:"]
    for item in lines_in[:5]:
        label = item.get("label") or item.get("line_id")
        incumbent = item.get("current_experiment_id") or item.get("best_experiment_id") or "none"
        hypothesis = item.get("hypothesis") or ""
        if len(hypothesis) > 120:
            hypothesis = hypothesis[:120] + "..."
        lines.append(
            f"- `{item.get('line_id')}` ({label}); local_incumbent=`{incumbent}`; hypothesis: {hypothesis}"
        )
    return lines


def _render_tree_policy_lines(actions: list[dict[str, Any]]) -> list[str]:
    if not actions:
        return ["Recommended tree actions: none yet."]
    lines = ["Recommended tree actions:"]
    for action in actions:
        parent = action.get("parent_node_id")
        parent_text = f", parent=`{parent}`" if parent else ""
        lines.append(
            f"- `{action.get('action_id')}`: {action.get('tree_action')}{parent_text} — {action.get('reason')}"
        )
    return lines


def _render_deferred_proposal_warning(config: ProjectConfig) -> list[str]:
    summary_path = config.handoff_results_dir / "latest_ingest_summary.json"
    if not summary_path.exists():
        return []
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return []
    deferred = [
        item
        for item in summary.get("results", [])
        if item.get("status") == "deferred_pending_context_refresh"
    ]
    if not deferred:
        return []
    names = ", ".join(f"`{Path(item.get('source_path') or item.get('proposal_id') or '').name}`" for item in deferred)
    return [
        "",
        f"Deferred proposal warning: {len(deferred)} proposal JSON file(s) remain in the inbox ({names}).",
        "Refresh context and rewrite or remove them before running another cycle.",
    ]


def render_cycle_summary(summary: dict[str, Any]) -> str:
    result = summary["cycle_result"]
    champion = summary.get("official_champion") or {}
    metrics = result.get("metrics_summary") or {}
    lines = [
        "# Latest Cycle Result",
        "",
        f"- completed_at: {summary['completed_at']}",
        f"- proposal_id: `{result.get('proposal_id')}`",
        f"- experiment_id: `{result.get('experiment_id')}`",
        f"- comparison_id: `{result.get('comparison_id')}`",
        f"- **decision**: `{result.get('decision')}`",
        f"- official_champion: `{champion.get('champion_id')}`",
    ]
    if metrics:
        lines += [
            "",
            "## Key metrics",
            f"- target_mode:     {metrics.get('target_mode', 'n/a')}",
            f"- primary_metric:  {metrics.get('primary_metric', 'n/a')}",
            f"- challenger score: {metrics.get('challenger_score', 'n/a')}",
            f"- champion score:   {metrics.get('champion_score', 'n/a')}",
            f"- mean lift:       {metrics.get('mean_lift', 'n/a'):+.6f}" if isinstance(metrics.get('mean_lift'), float) else f"- mean lift:       {metrics.get('mean_lift', 'n/a')}",
            f"- win rate:        {metrics.get('win_rate', 'n/a')}",
        ]
    if result.get("comparison_report"):
        lines += ["", f"- comparison report: `{result['comparison_report']}`"]
    return "\n".join(lines) + "\n"


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
    """Delegate to schema module for consistent float-tolerant fingerprinting."""

    from autoresearch.controller.proposal_schema import proposal_fingerprint as _fp
    return _fp(proposal)


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
    if outcome_type == "stale_parent":
        return "Refresh context and redesign from the current champion before running this idea."
    if outcome_type == "clear_loser":
        return "Treat the single-split loss as evidence; only branch from it with a materially different hypothesis."
    return "Treat this as weak evidence; propose a clearer change with a plausible variance or bias reduction mechanism."


def _active_queued_count(config: ProjectConfig) -> int:
    return sum(
        1
        for item in list_proposals(config.registry_path)
        if item["status"] in {"validated", "proposed", "needs_repair", "running", "awaiting_decision"}
    )


def _normalise_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
