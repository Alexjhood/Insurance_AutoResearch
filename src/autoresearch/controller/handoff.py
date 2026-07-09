"""File-based handoff workflow for external coding agents."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.controller.champion_template import RESERVED_INBOX_FILENAMES
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.workflow import enqueue_proposal_from_file, run_next_queued_proposal
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    latest_incomplete_research_log_entry,
    list_proposals,
    list_sessions,
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
    """Return the minimal proposal template the agent fills in.

    Only the scientific fields plus the ``experiment_config`` machine block are
    required; the controller hydrates ids, parentage, tree-walk, research-line
    label/hypothesis, and the fixed preprocessing at ingestion. See
    :func:`autoresearch.controller.workflow._hydrate_derived_fields` and the
    optional-override example in :func:`build_handoff`.
    """

    search_space = context.get("allowed_search_space") or {}
    strategies = search_space.get("target_strategies") or ["direct_pure_premium"]
    default_strategy = strategies[0]
    champion_recipe_path = config.handoff_proposal_inbox_dir / "champion_recipe.json"
    if champion_recipe_path.exists():
        try:
            champion_artifact = read_json(champion_recipe_path)
            default_strategy = champion_artifact.get("target_strategy") or default_strategy
        except Exception:
            pass
        model = {
            "recipe_ref": "champion",
            "recipe_overrides": {},
            "feature_exclusions": [],
        }
        model_family = "recipe"
    else:
        model = {
            "recipe": {
                "structure": "direct",
                "estimator": "lightgbm",
                "objective": "tweedie",
                "encoding": "native_categorical",
                "early_stopping": 50,
                "params": {"num_leaves": 63, "learning_rate": 0.05},
            },
            "feature_exclusions": [],
        }
        model_family = "recipe"
    template = {
        "experiment_name": "concise_experiment_name",
        "rationale": "Why this change is worth trying.",
        "change_summary": "Exact modelling/preprocessing change from parent.",
        "expected_benefit": "Expected improvement mechanism.",
        "key_risk": "Most likely failure mode.",
        "exploration_axis": "model_family",
        "approach_family": "Broad approach family, not a prescribed implementation.",
        "target_framing": "direct_pure_premium",
        "feature_representation": "raw",
        "expected_learning": "What this experiment should teach even if it fails.",
        "experiment_config": {
            "model_family": model_family,
            "target_strategy": default_strategy,
            "model": model,
        },
    }
    pending = _pending_auto_reject_reflection(config)
    if pending is not None:
        template["previous_cycle_reflection"] = {
            "cycle": int(pending["cycle"]),
            "interpretation": "What the previous result taught.",
            "next": "How that learning shaped this proposal.",
        }
    return template


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
        if proposal_file.name == "proposal_template.json" or proposal_file.name in RESERVED_INBOX_FILENAMES:
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
        "inbox_json_count": len([
            p for p in config.handoff_proposal_inbox_dir.glob("*.json")
            if p.name != "proposal_template.json" and p.name not in RESERVED_INBOX_FILENAMES
        ]),
        "inbox_files": [
            str(p) for p in sorted(config.handoff_proposal_inbox_dir.glob("*.json"))
            if p.name not in RESERVED_INBOX_FILENAMES
        ],
        "processed_valid_count": len(list(valid_dir.glob("*.json"))) if valid_dir.exists() else 0,
        "processed_invalid_count": len(list(invalid_dir.glob("*.json"))) if invalid_dir.exists() else 0,
        "processed_duplicate_count": len(list((config.handoff_proposal_processed_dir / "duplicate").glob("*.json")))
        if (config.handoff_proposal_processed_dir / "duplicate").exists()
        else 0,
        "latest_context": str(config.handoff_context_dir / "latest_context.json"),
        "latest_handoff": str(config.handoff_handoffs_dir / "latest_handoff.md"),
        "latest_cycle_result": str(config.handoff_results_dir / "latest_cycle_result.md"),
    }


def _weight_policy_line(context: dict[str, Any]) -> str:
    active = context.get("active_dataset") or {}
    return active.get("weight_policy") or (
        "the weight column is for weights/response denominators/rate→total conversion, never a feature."
    )


def _capping_constraint_line(context: dict[str, Any]) -> str:
    active = context.get("active_dataset") or {}
    cap = active.get("capping")
    if cap and cap != "no capping.":
        return f"{cap} Never change `claim_cap_threshold`."
    return "no capping for this dataset (`claim_capping_enabled=false`)."


def _render_active_dataset(active: dict[str, Any] | None) -> list[str]:
    """Render the binding "Active dataset" block from the context."""

    if not active:
        return []
    lines = [
        "## Active dataset",
        "",
        f"- **Dataset**: `{active['name']}` — {active['display_name']}",
        f"- **Target mode**: `{active['target_mode']}` → source column `{active['target_source_column']}` "
        f"(rate: {active['rate_label']}); available modes: {', '.join(f'`{m}`' for m in active['available_target_modes'])}",
        f"- **Weight**: {active['weight_policy']}",
        f"- **Population**: {active['population']}",
        f"- **Capping**: {active['capping']}",
        f"- **Frequency×severity recipes**: {'available' if active['frequency_severity_available'] else 'unavailable (no claim-count column)'}",
    ]
    for caution in active.get("cautions") or []:
        lines.append(f"- **Caution**: {caution}")
    lines.append("")
    lines.append("These facts are binding: build features only from the named feature list; never use the weight/id/target columns as predictors.")
    lines.append("")
    return lines


def render_handoff_markdown(
    config: ProjectConfig, context: dict[str, Any], *, delta: bool = False
) -> str:
    """Render a self-contained handoff file for external agents.

    Embeds the proposal template, key constraints, and champion metrics inline
    so the agent can start writing a proposal immediately without reading
    additional files (proposal_schema.json, proposal_template.json, context.json).

    When ``delta`` is true, the static blocks (proposal template, escape hatch,
    override block, key constraints, exploration-tree guidance) are omitted and
    only the dynamic run state (champion, next command, active lines, recommended
    actions, recent nodes, learnings, recipe ledger) is rendered. The static
    content is duplicated from the AGENT.md/CLAUDE.md contract and does not change
    cycle-to-cycle, so a mid-run refresh need not re-pay for it.
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

    # Recipe vocabulary (generated from the live registry, so it never drifts).
    from autoresearch.models.recipe import menu as _recipe_menu

    _menu = _recipe_menu()
    estimator_menu = "; ".join(
        f"`{name}` (obj {sorted(info['objectives'])})"
        for name, info in _menu["estimators"].items()
    )
    _default_objective = {
        "frequency": "poisson",
        "severity": "gamma",
    }.get(target_mode, "tweedie")
    feature_list = ", ".join(f"`{f}`" for f in features)
    tree = context.get("research_tree") or {}
    research_lines = context.get("research_lines") or {}
    node_lines = _render_tree_node_lines(tree.get("recent_nodes") or [])
    research_line_lines = _render_research_line_lines(research_lines.get("active_lines") or [])
    parked_line_lines = _render_parked_line_lines(research_lines.get("parked_lines") or [])
    recommended_actions = (tree.get("tree_policy") or {}).get("recommended_actions") or []
    action_lines = _render_tree_policy_lines(recommended_actions)
    deferred_lines = _render_deferred_proposal_warning(config)
    learning_lines = _render_recent_learnings(config, context)

    # Default target strategy for the active mode (severity is claim-rows-only).
    _default_strategy = "direct_severity" if target_mode == "severity" else "direct_pure_premium"
    champion_recipe_path = config.handoff_proposal_inbox_dir / "champion_recipe.json"
    if champion_recipe_path.exists():
        try:
            champion_artifact = read_json(champion_recipe_path)
            template_target_strategy = (
                champion_artifact.get("target_strategy") or _default_strategy
            )
        except Exception:
            template_target_strategy = _default_strategy
        template_model = {
            "recipe_ref": "champion",
            "recipe_overrides": {
                "params": {"<param>": "<value>"}
            },
        }
    else:
        template_target_strategy = _default_strategy
        template_model = {
            "recipe": {
                "structure": "direct",
                "estimator": "lightgbm",
                "objective": _default_objective,
                "encoding": "native_categorical",
                "early_stopping": 50,
                "params": {"num_leaves": 63, "learning_rate": 0.05},
            }
        }

    template_payload = {
        "experiment_name": "<concise_name>",
        "rationale": "<why this change is worth trying>",
        "change_summary": "<exact modelling/preprocessing change from parent>",
        "expected_benefit": "<expected improvement mechanism>",
        "key_risk": "<most likely failure mode>",
        "exploration_axis": "<model_family|target_framing|feature_representation|calibration|hyperparameter|diagnostic_probe|data_slice|ensemble|other>",
        "approach_family": "<broad approach family, without relying on another run's details>",
        "target_framing": "<target framing used by the proposal>",
        "feature_representation": "<feature representation used by the proposal>",
        "expected_learning": "<what this experiment should teach even if it fails>",
        "experiment_config": {
            "model_family": "recipe",
            "target_strategy": template_target_strategy,
            "model": template_model,
        },
    }
    pending_reflection = _pending_auto_reject_reflection(config)
    if pending_reflection is not None:
        template_payload["previous_cycle_reflection"] = {
            "cycle": int(pending_reflection["cycle"]),
            "interpretation": "<what the previous auto-rejection taught>",
            "next": "<how that learning shaped this proposal>",
        }
    template_json = json.dumps(template_payload, indent=2)

    # Escape-hatch shape, shown only as the alternative for novel models.
    script_config_json = json.dumps({
        "model_family": "scripted_challenger",
        "target_strategy": _default_strategy,
        "model": {"script_path": "model_<name>.py"},
    }, indent=2)

    # Optional block — only include the keys you want to override. The controller
    # otherwise derives ids/parentage, follows the top recommended tree action,
    # and extends the most recent active research line.
    # Placeholders only: concrete values here get copy-pasted into proposals
    # wholesale (observed in run 20260612T105643Z), turning derivable defaults
    # into validation failures.
    override_json = json.dumps({
        "research_line_id": "<existing line_id to extend, or a short new id when creating>",
        "research_line_action": "<create_line|extend_line|revisit_line|close_line>",
        "research_line_label": "<label, only required when creating a new line>",
        "research_line_hypothesis": "<hypothesis, only required when creating a new line>",
        "tree_action": "<only when diverging from the recommended action>",
        "selected_tree_action_id": "<an action_id from the recommended tree actions below>",
        "research_parent_node_id": "<a node_id from this run's exploration tree, or null>",
        "tree_policy_override_rationale": "<required only when diverging from the recommended action>",
    }, indent=2)

    from autoresearch.memory import resolve_memory_access
    from autoresearch.memory.store import default_playbook_dir

    _memory_access = resolve_memory_access(config)
    _playbook_link_lines: list[str] = []
    if _memory_access in ("own", "all"):
        _playbook_base = default_playbook_dir(config.dataset_name)
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

    # Dynamic run-state intro that the agent needs every cycle: a pending
    # auto-reject reflection requirement and any deferred-proposal warning.
    pending_reflection_lines = (
        [
            "",
            f"Cycle {pending_reflection['cycle']} was auto-rejected and still needs reflection. "
            "The `previous_cycle_reflection` block in the proposal is required; the framework will "
            "use it to complete that cycle's log before running this proposal.",
        ]
        if pending_reflection is not None
        else []
    )

    # Static blocks (proposal template, escape hatch, override block, key
    # constraints, exploration-tree guidance). These are duplicated from the
    # AGENT.md/CLAUDE.md contract and do not change cycle-to-cycle, so the
    # `--delta` handoff omits them.
    static_lines = [
        "## Proposal quick-start",
        "",
        f"Copy this to `{config.handoff_proposal_inbox_dir}/proposal_<name>.json` and fill in the `<...>` fields.",
        "Write exactly one proposal for this context refresh.",
        "",
        "**Prefer a declarative `model.recipe`** (below): trusted framework code builds the model "
        "and owns exposure→total conversion and calibration — no Python file, no hand-calibration. "
        f"Estimators: {estimator_menu}. Structures: `direct`, `frequency_severity` (burning-cost only; "
        "in `severity` target mode use `direct` — the population is already claim rows only). "
        "Encoding defaults per estimator; `early_stopping` uses a framework train-internal split.",
        "When a recipe champion exists, use `model.recipe_ref: \"champion\"` plus only the nested "
        "`recipe_overrides` you are changing. The controller resolves and validates the full recipe "
        "before execution. Put `feature_inclusions`/`feature_exclusions` beside the reference, not inside it.",
        "",
        "Supply only the scientific fields below. The controller derives the rest "
        "(`proposal_id`, parentage, `branch_action`, the tree-walk fields, the "
        "research-line `label`/`hypothesis`, and the fixed `preprocessing`) — you "
        "do not repeat them.",
        "",
        "```json",
        template_json,
        "```",
        "",
        "**Escape hatch (novel models only):** if the recipe vocabulary cannot express your idea, "
        "write `model_<name>.py` (same directory) exposing "
        "`fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters)` "
        "and set `experiment_config.model.script_path` instead of `recipe`. Return a "
        "`Prediction(values=rates, unit=\"rate\")` from `autoresearch.models.prediction` and the framework "
        "converts to totals and calibrates for you (a recipe and a script are mutually exclusive):",
        "",
        "```json",
        script_config_json,
        "```",
        "",
        "By default the controller follows the top recommended tree action and "
        "extends the most recent active research line — **the usual move is to omit "
        "this block entirely**. Do not copy it verbatim: add only the keys you want "
        "to change (every `<...>` is a placeholder, not a value). To "
        "open a *new* line set `research_line_action` to `create_line` with a short "
        "new `research_line_id` plus a `research_line_label`/`research_line_hypothesis`; "
        "to extend an existing line use its id and you can omit the label/hypothesis:",
        "",
        "```json",
        override_json,
        "```",
        "",
        "## Key constraints",
        "",
        f"- **Target mode**: `{target_mode}` (see the Active dataset block above for source/weight/cap)",
        f"- **Features available**: {feature_list}",
        f"- **Weight policy**: {_weight_policy_line(context)}",
        f"- **Target strategies**: {', '.join(f'`{s}`' for s in target_strategies)} (must agree with the recipe `structure`: `direct_pure_premium`/`frequency`/`direct_severity`→`direct`, `frequency_severity`→`frequency_severity`)",
        f"- **Capping**: {_capping_constraint_line(context)}",
        "- **Units & calibration are framework-owned**: a recipe (or a script returning `Prediction`) needs no exposure conversion or `apply_training_calibration` call. Only a script returning a raw `np.ndarray` must return totals and calibrate itself.",
        "- **Never reference** `milestone_holdout`, `holdout_vault`, or `AUTORESEARCH_MILESTONE_TOKEN`",
        "",
        "## Exploration tree",
        "",
        "- Scope: active run only. Do not use results from other runs as proposal evidence.",
        "- Choose `research_parent_node_id` from this tree when the next idea builds on a prior hypothesis; use `null` only for a genuinely new line of attack.",
        "- Cross-run memory, when enabled, may inform broad strategy but must not supply tree parent IDs or evidence for this run.",
        "- By default a proposal extends the most recent active research line. To organise differently, set `research_line_id` (and `research_line_action`) via the optional override block so the run keeps a small number of coherent lines.",
        "- Keep at most 5 research lines active. When creating a new line at the cap, set `park_research_line_id` and explain why it should be parked.",
        "- A future `record-decision` can be `promote` for the whole run, `local_promote` for this line only, or `reject`.",
        "- The single-split hurdle uses this line's local incumbent where available; full comparison still reports against the official champion.",
        "- By default the controller takes the top recommended tree action. To choose a different one, set `tree_action`/`selected_tree_action_id` in the override block; if you ignore the recommended action, include `tree_policy_override_rationale`.",
        "- Prefer genuine exploration over repetitive small retunes. A useful child idea should change the hypothesis, representation, target framing, or error mode it addresses.",
        "- Clear failures and auto-rejections are evidence. Reflect on them, then branch only when the child idea is materially different.",
    ]

    # Dynamic run-state blocks: active/parked research lines, recommended
    # actions, recent nodes, learnings, recipe ledger, playbook. Always shown.
    run_state_lines = [
        *research_line_lines,
        *parked_line_lines,
        "",
        *action_lines,
        "",
        *node_lines,
        *learning_lines,
        *_render_recipe_reuse(config),
        *_playbook_link_lines,
    ]

    current_state_lines = [
        "## Current state",
        "",
        f"- **Champion**: `{champion_id}` (branch `{branch_id}`{gini_str})",
        f"- **Inbox**: `{config.handoff_proposal_inbox_dir}`  ← write the proposal JSON here",
        f"- **Next command**: `{_next_supervised_command(config, context)}`",
        *_render_champion_followup(config),
        *pending_reflection_lines,
        *deferred_lines,
        "",
        *_render_active_dataset(context.get("active_dataset")),
    ]

    drilldown_lines = [
        "## Optional drill-down",
        "",
        "The handoff above is authoritative for proposing. Read these only when you "
        "need detail it does not already carry:",
        f"- Full context JSON: `{config.handoff_context_dir / 'latest_context.json'}`",
        f"- Narrative research log: `{config.research_log_path}`",
    ]

    if delta:
        lines = [
            "# Auto-Research Handoff (delta)",
            "",
            "Dynamic run state only. The proposal template, key constraints, and "
            "exploration-tree guidance are unchanged from the full handoff (and the "
            "`AGENT.md`/`CLAUDE.md` contract already in context) — run "
            "`show-latest-handoff` without `--delta` for the full version.",
            "",
            *current_state_lines,
            "## Run state",
            "",
            *run_state_lines,
            "",
            *drilldown_lines,
        ]
        return "\n".join(lines) + "\n"

    lines = [
        "# Auto-Research Handoff",
        "",
        "This handoff is the authoritative context for proposing the next experiment — "
        "champion, recent results and learnings, recommended tree actions, key constraints, "
        "and the full proposal template are all inline below. Read `AGENT.md` for the runtime "
        "contract; the files under \"Optional drill-down\" only when you need more detail.",
        "",
        *current_state_lines,
        *static_lines,
        "",
        *run_state_lines,
        "",
        *drilldown_lines,
    ]
    return "\n".join(lines) + "\n"


def proposal_schema_document(config: ProjectConfig, context: dict[str, Any]) -> dict[str, Any]:
    """Export an inspectable proposal schema description."""

    from autoresearch.controller.proposal_schema import (
        DERIVED_PROPOSAL_FIELDS,
        SCIENTIFIC_PROPOSAL_FIELDS,
    )

    return {
        "type": "object",
        "required": [*SCIENTIFIC_PROPOSAL_FIELDS, "experiment_config"],
        "controller_derived": [
            *DERIVED_PROPOSAL_FIELDS,
            "parent_branch_id",
            "branch_action",
            "research_parent_node_id",
            "experiment_config.experiment_name",
            "experiment_config.parent_experiment_id",
            "experiment_config.preprocessing",
        ],
        "experiment_config_required": [
            "model_family",
            "target_strategy",
            "model.recipe OR model.recipe_ref OR model.script_path",
        ],
        "allowed_search_space": context["allowed_search_space"],
        "notes": [
            "Supply only the required scientific fields plus experiment_config "
            "(model_family, target_strategy, and a model implementation); the controller "
            "hydrates every controller_derived field at ingestion.",
            "Any controller_derived field may still be supplied to override its default.",
            "proposal_id, when supplied, must use letters, numbers, hyphen, or underscore.",
            "experiment_config.experiment_name (derived) mirrors experiment_name.",
            "experiment_config.parent_experiment_id (derived) mirrors parent_experiment_id.",
            "Provide exactly one of model.recipe, model.recipe_ref='champion', or model.script_path. "
            "recipe_ref accepts an optional nested recipe_overrides object and is resolved before validation.",
            "A recipe's structure must agree with target_strategy (direct↔direct_pure_premium/frequency/direct_severity; "
            "frequency_severity↔frequency_severity).",
            "Do not use the weight/offset column as a predictive feature; it is reserved for weights and response calculations (see the Active dataset block).",
            "research_parent_node_id is optional and may only point to a node from this active run's research_tree.",
            "tree_action=new_root may use research_parent_node_id=null; all other tree actions must point to a valid active-run node.",
            "selected_tree_action_id should match a recommended action from research_tree.tree_policy, unless tree_policy_override_rationale explains the deviation.",
            "research_line_action=create_line must use a new research_line_id; extend_line/revisit_line/close_line must use an existing active-run line.",
            "If 5 research lines are already active, research_line_action=create_line must include park_research_line_id for an existing active line.",
            "Parked lines remain in history and reports, but new proposals should not extend them unless tree_policy_override_rationale explains why.",
            "Keep the active run to a small number of coherent research lines; use local promotion for progress inside a line without replacing the global champion.",
            "Only one validated proposal is ingested per context refresh while a proposal is queued or awaiting decision; additional proposal JSON files remain deferred in the inbox.",
            "When the previous cycle was auto-rejected, the next proposal must include "
            "previous_cycle_reflection with cycle, interpretation, and next. The framework "
            "uses it to complete the prior cycle before running the new proposal.",
            f"Active target_mode is {config.target_mode}; use a non-default mode only when the run was explicitly configured for it. "
            "In severity mode the population is claim rows only (ClaimNb > 0); ClaimNb is the metric weight and rate→total offset, never a feature; "
            "the response is cost per claim (strictly positive) so objectives are gamma or squared_error.",
            "Do not reference milestone_holdout.",
        ],
    }


def _next_supervised_command(config: ProjectConfig, context: dict[str, Any]) -> str:
    active_queue = context.get("active_queue") or {}
    for item in active_queue.get("items") or []:
        if item.get("status") == "awaiting_decision":
            comparison_id = item.get("comparison_id") or "<comparison_id>"
            return (
                f"autoresearch --track {config.track_id} --run-id {config.run_id} "
                f"record-decision {comparison_id} --decision <promote|local_promote|reject> "
                '--rationale "<why>" --interpretation "<what this taught>" '
                '--next "<next direction>"'
            )

    sessions = list_sessions(config.registry_path)
    if not sessions:
        return f"autoresearch --track {config.track_id} --run-id {config.run_id} start-session main"

    pending = _pending_auto_reject_reflection(config)
    latest = sessions[0]
    if (
        pending is not None
        and latest.get("max_cycles") is not None
        and int(latest.get("current_cycle") or 0) >= int(latest["max_cycles"])
    ):
        return (
            f"autoresearch --track {config.track_id} --run-id {config.run_id} "
            'record-cycle-reflection --interpretation "<what this taught>" '
            '--next "<next direction or stop reason>"'
        )

    return f"autoresearch --track {config.track_id} --run-id {config.run_id} run-session-cycles 1"


def _pending_auto_reject_reflection(config: ProjectConfig) -> dict[str, Any] | None:
    pending = latest_incomplete_research_log_entry(config.registry_path)
    if pending is None or pending.get("comparison_id"):
        return None
    return pending


def _render_champion_followup(config: ProjectConfig) -> list[str]:
    """Point the agent at the generated champion follow-up artifacts when present."""
    inbox = config.handoff_proposal_inbox_dir
    recipe_path = inbox / "champion_recipe.json"
    template_path = inbox / "champion_template.py"
    if recipe_path.exists():
        return [
            f"- **Champion follow-up**: set `model.recipe_ref` to `\"champion\"` and provide only "
            "`model.recipe_overrides` for changed recipe fields. The controller resolves "
            f"`{recipe_path.name}` and preserves provenance — do not rewrite the recipe."
        ]
    if template_path.exists():
        return [
            f"- **Champion follow-up**: `{template_path.name}` (script) reproduces the champion; "
            "edit `PARAM_OVERRIDES` for a small follow-up."
        ]
    return []


def _render_recipe_reuse(config: ProjectConfig, *, limit: int = 8) -> list[str]:
    """Inline a compact ranked summary of recipes tried so far (the reuse library).

    Surfacing this in the authoritative handoff is what makes the recipe library
    actually influence the next proposal — both winners to build on and dead-ends
    to avoid repeating. Scope (run-local vs cross-run) follows ``[recipes]``.
    """
    try:
        from autoresearch.models.recipe_library import effective_reuse_scope, list_recipes

        rows = list_recipes(config, limit=limit)
    except Exception:
        return []
    if not rows:
        return []
    scope = effective_reuse_scope(config)
    lines = [
        "## Recipe reuse",
        "",
        f"Recipes tried so far (scope: `{scope}`), current champion first. Build on what worked; "
        "do not re-run a recipe that already lost or failed:",
        "",
    ]
    for r in rows:
        champion = " **CURRENT CHAMPION**" if r.get("is_current_champion") else ""
        screen_score = r.get("screen_score", r.get("score"))
        comparison_score = r.get("comparison_score")
        comparison_lift = r.get("comparison_lift")
        metrics = []
        if isinstance(comparison_score, (int, float)):
            metrics.append(f"CV score {comparison_score:.4f}")
        if isinstance(comparison_lift, (int, float)):
            metrics.append(f"CV lift {comparison_lift:+.4f}")
        if isinstance(screen_score, (int, float)):
            metrics.append(f"split score {screen_score:.4f}")
        metric_text = ", ".join(metrics) if metrics else "no score recorded"
        lines.append(
            f"- `{r.get('summary')}`{champion} — {r.get('outcome')} ({metric_text})"
        )
    lines.append("")
    return lines


def _render_recent_learnings(config: ProjectConfig, context: dict[str, Any]) -> list[str]:
    """Inline the most recent non-promotions + last research-log entry.

    Surfacing these in the handoff removes the agent's need to read the full
    ``RESEARCH_LOG.md`` or per-proposal non-promotion files on every cycle.
    """
    learning_lines: list[str] = []

    non_promoted_dir = config.handoff_results_dir / "non_promoted"
    summaries: list[dict[str, Any]] = []
    if non_promoted_dir.exists():
        files = sorted(
            non_promoted_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files[:3]:
            try:
                summaries.append(read_json(path))
            except (OSError, json.JSONDecodeError):
                continue
    if summaries:
        learning_lines.append("Recent non-promotions (most recent first):")
        for item in summaries:
            name = item.get("experiment_name") or item.get("proposal_id") or "unknown"
            outcome = item.get("outcome_type") or "inconclusive"
            reason = (item.get("reason") or "").strip()
            if len(reason) > 160:
                reason = reason[:160] + "…"
            guidance = (item.get("agent_guidance") or "").strip()
            if len(guidance) > 160:
                guidance = guidance[:160] + "…"
            entry = f"- `{name}` ({outcome}): {reason}"
            if guidance:
                entry += f" — guidance: {guidance}"
            learning_lines.append(entry)

    log_tail = _research_log_tail(config)
    if log_tail:
        if learning_lines:
            learning_lines.append("")
        learning_lines.append("Last research-log entry:")
        learning_lines.extend(f"> {line}" if line else ">" for line in log_tail.splitlines())

    if not learning_lines:
        return []
    return ["", "## Recent learnings", "", *learning_lines]


def _research_log_tail(config: ProjectConfig, max_chars: int = 600) -> str:
    """Return the final ``## ``-delimited entry of this run's research log, bounded."""
    path = config.research_log_path
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not text:
        return ""
    marker = "\n## "
    idx = text.rfind(marker)
    entry = text[idx + 1:] if idx != -1 else text
    entry = entry.strip()
    if len(entry) > max_chars:
        entry = entry[-max_chars:].lstrip()
    return entry


def _render_tree_node_lines(nodes: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return ["No research-tree nodes yet. Start with a small, well-motivated first hypothesis."]
    lines = ["Recent active-run nodes:"]
    for node in nodes[:8]:
        cv_lift = node.get("cv_lift")
        split_lift = node.get("split_lift")
        if cv_lift is not None and split_lift is not None:
            lift_text = f", cv_lift={float(cv_lift):+.6f} (split_lift={float(split_lift):+.6f})"
        elif cv_lift is not None:
            lift_text = f", cv_lift={float(cv_lift):+.6f}"
        elif split_lift is not None:
            lift_text = f", split_lift={float(split_lift):+.6f}"
        else:
            lift_text = ""
        axis = node.get("exploration_axis")
        axis_text = f", axis={axis}" if axis else ""
        summary = node.get("learning") or ""
        if len(summary) > 120:
            summary = summary[:120] + "..."
        lines.append(
            f"- `{node.get('node_id')}` status={node.get('status')}"
            f", outcome={node.get('outcome_type')}{axis_text}{lift_text}: {summary}"
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


def _render_parked_line_lines(lines_in: list[dict[str, Any]]) -> list[str]:
    if not lines_in:
        return []
    lines = ["Parked research lines:"]
    for item in lines_in[:5]:
        label = item.get("label") or item.get("line_id")
        notes = item.get("notes") or ""
        if len(notes) > 100:
            notes = notes[:100] + "..."
        lines.append(f"- `{item.get('line_id')}` ({label}); parked_reason: {notes}")
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
        challenger_score = metrics.get("cv_challenger_score", metrics.get("challenger_score"))
        champion_score = metrics.get("cv_champion_score", metrics.get("champion_score"))
        mean_lift = metrics.get("cv_mean_lift", metrics.get("mean_lift"))
        win_rate = metrics.get("cv_win_rate", metrics.get("win_rate"))
        lines += [
            "",
            "## Key metrics",
            f"- target_mode:     {metrics.get('target_mode', 'n/a')}",
            f"- primary_metric:  {metrics.get('primary_metric', 'n/a')}",
            f"- CV challenger score: {challenger_score if challenger_score is not None else 'n/a'}",
            f"- CV champion score:   {champion_score if champion_score is not None else 'n/a'}",
            f"- CV mean lift:       {mean_lift:+.6f}" if isinstance(mean_lift, float) else f"- CV mean lift:       {mean_lift if mean_lift is not None else 'n/a'}",
            f"- CV win rate:        {win_rate if win_rate is not None else 'n/a'}",
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
