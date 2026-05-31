"""Controlled propose -> execute -> compare -> promote workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
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
from autoresearch.controller.proposal_schema import allowed_search_space, normalise_proposal, validate_proposal

from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_artifacts,
    list_research_nodes,
    next_queued_proposal,
    record_proposal,
    record_experiment_artifacts,
    set_official_champion,
    upsert_research_node,
    update_proposal_status,
    upsert_branch,
)
from autoresearch.experiment_runner import ComputeBudgetExceeded, PreflightFailed, run_experiment
from autoresearch.evaluation.validation import ValidationRules, validate_experiment_outputs
from autoresearch.run_artifacts import next_iteration_dir, proposal_iteration_dir
from autoresearch.utils.io import read_json, write_json


class ExperimentNeedsRepair(ValueError):
    """Raised when a script attempt failed validation and another attempt is needed."""


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


def run_next_queued_proposal(config: ProjectConfig) -> dict[str, Any]:
    """Run the next validated proposal and gate it against the official champion."""

    _reconcile_stale_running_proposals(config)
    champion = _require_champion(config)
    proposal = next_queued_proposal(config.registry_path)
    if proposal is None:
        raise ValueError("No validated proposals are queued")

    proposal_id = proposal["proposal_id"]
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
        if getattr(config, "screening_enabled", True):
            screening = screen_challenger_single_split(config, champion["champion_id"], experiment_id)
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
                return {
                    "proposal_id": proposal_id,
                    "experiment_id": experiment_id,
                    "comparison_id": None,
                    "decision": "auto_reject",
                    "auto_rejected": True,
                    "auto_reject_reason": reason,
                    "screening_report": str(screening_path),
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
            "gate_primary_metric": comp_summary.get("gate_primary_metric"),
            "challenger_score": round(float(comp_summary.get("challenger_mean_score") or 0), 6),
            "champion_score": round(float(comp_summary.get("champion_mean_score") or 0), 6),
            "mean_lift": round(float(comp_summary.get("mean_lift") or 0), 6),
            "win_rate": round(float(comp_summary.get("challenger_win_rate") or 0), 4),
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
    research_parent = parsed.get("research_parent_node_id") or parsed.get("parent_node_id")
    if research_parent is not None:
        if not isinstance(research_parent, str) or not research_parent.strip():
            errors.append("research_parent_node_id must be a non-empty string or null")
        else:
            node_ids = {node["node_id"] for node in list_research_nodes(config.registry_path)}
            if research_parent not in node_ids:
                errors.append("research_parent_node_id must refer to a node in this active run's research_tree")
    branch_action = parsed.get("branch_action", "extend_current")
    branch_id = parsed.get("branch_id") or (parsed.get("proposal_id") if branch_action == "new_branch" else parent_branch)
    proposal = normalise_proposal(parsed, branch_id=branch_id, parent_branch_id=parent_branch)
    return proposal, errors


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
    # Track per-attempt numeric lifts for auto-abandon check.
    attempt_lifts: list[float | None] = []
    noise_eps = getattr(config, "repair_noise_floor_eps", 0.002)
    auto_abandon = getattr(config, "repair_auto_abandon_enabled", True)

    for attempt in range(1, 4):
        attempt_script = proposal_dir / f"model_attempt_{attempt}.py"
        cfg = dict(proposal["config"])
        model_cfg = dict(cfg.get("model") or {})
        if attempt_script.exists():
            model_cfg["script_path"] = attempt_script.name
        elif attempt == 1:
            raw_script = model_cfg.get("script_path") or model_cfg.get("model_script_path")
            if raw_script:
                raw_path = proposal_dir / str(raw_script)
                if raw_path.exists():
                    model_cfg["script_path"] = raw_path.name
        elif last_report is not None:
            _write_repair_request(proposal_dir, attempt, last_report)
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
                _write_repair_request(proposal_dir, attempt + 1, exc_report)
                next_script = proposal_dir / f"model_attempt_{attempt + 1}.py"
                if not next_script.exists():
                    raise ExperimentNeedsRepair(
                        f"Attempt {attempt} failed with {type(exc).__name__}: {str(exc)[:200]}. "
                        f"Write {next_script} and rerun the proposal."
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
            _write_repair_request(proposal_dir, attempt + 1, report)
            next_script = proposal_dir / f"model_attempt_{attempt + 1}.py"
            if not next_script.exists():
                raise ExperimentNeedsRepair(
                    f"Experiment output validation failed: {report['reason']}. "
                    f"Write {next_script} and rerun the proposal."
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


def _artifact_path(config: ProjectConfig, experiment_id: str, artifact_type: str) -> Path:
    for artifact in list_artifacts(config.registry_path, experiment_id):
        if artifact["artifact_type"] == artifact_type:
            return Path(artifact["path"])
    raise ValueError(f"Experiment {experiment_id} has no {artifact_type!r} artifact")


def _write_repair_request(proposal_dir: Path, next_attempt: int, report: dict[str, Any]) -> Path:
    error_type = report.get("error_type", "positive_lift_failed")
    payload: dict[str, Any] = {
        "next_attempt": next_attempt,
        "write_script": f"model_attempt_{next_attempt}.py",
        "error_type": error_type,
        "reason": report.get("reason"),
        "failed_checks": [check for check in report.get("checks", []) if not check.get("passed")],
        "metrics_summary": report.get("metrics_summary"),
        "comparison_report": report.get("comparison_report"),
        "instruction": (
            "Revise the model script to fix the failed checks. Keep the same fit_predict "
            "interface and do not access holdout data. The next run will use this script. "
            "Read comparison_report (HTML) for full Gini curves and lift charts before deciding "
            "on the fix strategy."
        ),
    }
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


def _proposal_id(parsed: dict[str, Any] | None) -> str:
    if parsed and isinstance(parsed.get("proposal_id"), str):
        return parsed["proposal_id"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"invalid_proposal_{stamp}"


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
    upsert_research_node(
        config.registry_path,
        node_id=proposal_id,
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
        screening=screening,
        metrics=metrics,
        guidance=guidance,
    )


def _screening_metrics_summary(screening: dict[str, Any]) -> dict[str, Any]:
    if not screening:
        return {}
    keys = [
        "gate_mode",
        "gate_metric",
        "target_mode",
        "passed",
        "champion_score",
        "challenger_score",
        "lift",
        "relative_lift",
        "overlap_rows",
    ]
    result: dict[str, Any] = {}
    for key in keys:
        if key in screening:
            value = screening[key]
            result[key] = round(float(value), 6) if isinstance(value, float) else value
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
