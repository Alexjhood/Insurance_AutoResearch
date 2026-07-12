"""Read orchestration.json / campaign_report.json / notes.json / reports / briefs.

Builds :class:`Campaign`, :class:`Delegation` and :class:`OperatorNote` objects
(DATA.md §2.1, §2.2, §2.4). The ledger (``orchestration.json .delegations[]``)
is authoritative; report and brief files enrich it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..schema import (
    Brief,
    Campaign,
    Consolidation,
    Delegation,
    DelegationBudget,
    DelegationChampion,
    DelegationCost,
    DelegationFiles,
    Distress,
    ModelIdentity,
    OperatorNote,
    OrchestratorIdentity,
    RepairSummary,
    TokenTotals,
)
from ..util import Warnings, cache_hit_rate, load_json, to_float, to_int


def load_sources(orch_dir: Path, warnings: Warnings) -> tuple[dict, dict, dict]:
    ledger = load_json(orch_dir / "orchestration.json", warnings,
                       label="orchestration.json") or {}
    campaign_report = load_json(orch_dir / "campaign_report.json", warnings,
                                label="campaign_report.json") or {}
    notes = load_json(orch_dir / "notes.json", warnings, label="notes.json") or {}
    if not ledger:
        warnings.add("orchestration.json missing or empty — campaign degraded")
    return ledger, campaign_report, notes


def _orchestrator_model(campaign_report: dict, ledger: dict) -> str:
    orch = campaign_report.get("orchestrator")
    if isinstance(orch, dict):
        provider = orch.get("model_provider") or ""
        name = orch.get("model_name") or ""
        if provider or name:
            return f"{provider}/{name}".strip("/")
    if isinstance(orch, str) and orch:
        return orch
    provider = ledger.get("model_provider") or ""
    name = ledger.get("model_name") or ""
    return f"{provider}/{name}".strip("/")


def _orchestrator_identity(campaign_report: dict, ledger: dict) -> OrchestratorIdentity:
    report = campaign_report.get("orchestrator")
    source = report if isinstance(report, dict) else ledger.get("orchestrator")
    if not isinstance(source, dict):
        source = {}
    return OrchestratorIdentity(
        provider=source.get("provider") or source.get("model_provider") or ledger.get("model_provider") or "",
        model=source.get("model") or source.get("model_name") or ledger.get("model_name") or "",
        effort=source.get("effort") or ledger.get("model_effort"),
    )


def build_campaign(
    orch_id: str, orch_dir: Path, ledger: dict, campaign_report: dict,
    ended_at: Optional[str], files_lookup: dict[str, str], warnings: Warnings,
) -> Campaign:
    cons_doc = ledger.get("consolidation")
    consolidation = None
    if isinstance(cons_doc, dict) and cons_doc.get("run_id"):
        consolidation = Consolidation(
            run_id=cons_doc.get("run_id", ""),
            track=cons_doc.get("track", ""),
        )
    return Campaign(
        orch_id=orch_id,
        dataset=ledger.get("dataset") or campaign_report.get("dataset") or "",
        target_mode=campaign_report.get("target_mode") or ledger.get("target_mode") or "",
        status=ledger.get("status") or campaign_report.get("status") or "",
        created_at=ledger.get("created_at") or "",
        ended_at=ended_at,
        cycles_committed=to_int(ledger.get("cycles_committed")) or 0,
        orchestrator_model=_orchestrator_model(campaign_report, ledger),
        orchestrator=_orchestrator_identity(campaign_report, ledger),
        consolidation=consolidation,
        framework_computed=campaign_report.get("framework_computed"),
        campaign_report_md=files_lookup.get("CAMPAIGN_REPORT.md"),
        orchestration_log_md=files_lookup.get("ORCHESTRATION_LOG.md"),
    )


def build_notes(notes_doc: dict) -> list[OperatorNote]:
    out: list[OperatorNote] = []
    for n in notes_doc.get("notes") or []:
        if not isinstance(n, dict):
            continue
        out.append(
            OperatorNote(
                at=n.get("timestamp") or n.get("at") or "",
                kind=n.get("kind") or "note",
                delegation_id=n.get("delegation_id"),
                text=n.get("text") or n.get("note") or "",
            )
        )
    return out


def _brief(orch_dir: Path, ledger_deleg: dict, warnings: Warnings) -> Brief:
    brief_path = ledger_deleg.get("brief_path")
    doc: dict = {}
    if brief_path:
        # brief_path is repo-relative; resolve against repo root via orch_dir.
        candidate = Path(brief_path)
        if not candidate.is_absolute():
            candidate = _repo_root(orch_dir) / brief_path
        loaded = load_json(candidate, warnings, label=brief_path)
        if isinstance(loaded, dict):
            doc = loaded
    name = None
    src = doc.get("_source_path")
    if src:
        name = Path(src).stem
    return Brief(
        name=name,
        direction=doc.get("direction") or "",
        constraints=list(doc.get("constraints") or []),
        cycle_budget=to_int(doc.get("cycle_budget")) or 0,
        starting_knowledge=list(doc.get("starting_knowledge") or []),
        foundation_models=bool(doc.get("foundation_models")),
        seed_champion=doc.get("seed_champion"),
    )


def _repo_root(orch_dir: Path) -> Path:
    # orch_dir = <repo>/artifacts/orchestrations/<id>
    return orch_dir.parent.parent.parent


def _model_identity(orch_dir: Path, delegation_id: str, warnings: Warnings) -> Optional[ModelIdentity]:
    manifest = load_json(
        orch_dir / "runs" / delegation_id / "run_manifest.json", warnings,
        label=f"run_manifest {delegation_id}",
    )
    if not isinstance(manifest, dict):
        return None
    mid = manifest.get("model_identity")
    if not isinstance(mid, dict):
        return None
    return ModelIdentity(
        provider=mid.get("provider", ""),
        name=mid.get("name", ""),
        harness=mid.get("harness", ""),
    )


def _budget(ledger_deleg: dict, report: dict) -> DelegationBudget:
    cycles = report.get("cycles") or {}
    return DelegationBudget(
        committed=to_int(ledger_deleg.get("cycle_budget")) or 0,
        attempted=to_int(cycles.get("attempted")) or 0,
        completed=to_int(cycles.get("completed")) or 0,
        decided=to_int(cycles.get("decided")) or 0,
        used=to_int(cycles.get("used")) or 0,
        forfeited=to_int(ledger_deleg.get("cycles_forfeited")) or 0,
        refunded=bool(ledger_deleg.get("budget_refunded")),
    )


def _champion(report: dict) -> Optional[DelegationChampion]:
    ch = report.get("champion")
    if not isinstance(ch, dict):
        return None
    return DelegationChampion(
        experiment_id=ch.get("experiment_id", ""),
        gini_weighted=to_float(ch.get("gini_weighted")) or 0.0,
        rank_gini_weighted=to_float(ch.get("rank_gini_weighted")),
        asym_pricing_loss=to_float(ch.get("asym_pricing_loss")),
        calibration_ratio=to_float(ch.get("calibration_ratio")),
        model_family=ch.get("model_family", ""),
        target_strategy=ch.get("target_strategy", ""),
        beat_seed_baseline=ch.get("beat_seed_baseline"),
    )


def _distress(report: dict) -> Distress:
    d = report.get("distress") or {}
    return Distress(
        active=list(d.get("active") or []),
        all_flags=list(d.get("flags") or []),
        detail=d.get("detail"),
    )


def _cost(ledger_deleg: dict, report: dict, telemetry_cost: dict) -> DelegationCost:
    usage = ledger_deleg.get("tool_usage")
    if isinstance(usage, dict) and usage:
        tokens = TokenTotals(
            input=to_int(usage.get("input_tokens")) or 0,
            cached_input=to_int(usage.get("cached_input_tokens")) or 0,
            output=to_int(usage.get("output_tokens")) or 0,
            reasoning=to_int(usage.get("reasoning_output_tokens")) or 0,
        )
    else:
        # quirk 7: crashed delegation missing tool_usage → sum model calls.
        tokens = telemetry_cost.get("tokens") or TokenTotals()
    report_cost = report.get("cost") or {}
    llm_usage = report_cost.get("llm_usage") or {}
    cost_usd = to_float(
        ledger_deleg.get("estimated_cost_usd")
        or ledger_deleg.get("cost_usd")
        or llm_usage.get("cost_usd")
        or report_cost.get("cost_usd")
    )
    return DelegationCost(
        tokens=tokens,
        model_calls=telemetry_cost.get("model_calls", 0),
        tool_calls=telemetry_cost.get("tool_calls", 0),
        tool_failures=telemetry_cost.get("tool_failures", 0),
        cache_hit_rate=cache_hit_rate(tokens.cached_input, tokens.input),
        wall_clock_minutes=to_float(report_cost.get("wall_clock_minutes")),
        cost_usd=cost_usd,
        cost_estimated=bool(ledger_deleg.get("cost_estimated") or llm_usage.get("cost_estimated") or report_cost.get("cost_estimated") or cost_usd is not None),
    )


def _files(delegation_id: str, files_paths: set[str]) -> DelegationFiles:
    def present(rel: str) -> Optional[str]:
        return rel if rel in files_paths else None

    return DelegationFiles(
        prompt=present(f"prompts/{delegation_id}.md"),
        brief=present(f"briefs/{delegation_id}.json"),
        report=present(f"reports/{delegation_id}.json"),
        stdout_log=present(f"logs/{delegation_id}.stdout.log"),
        exit_json=present(f"logs/{delegation_id}.exit.json"),
        research_log=present(f"runs/{delegation_id}/RESEARCH_LOG.md"),
        llm_usage=present(f"runs/{delegation_id}/LLM_USAGE.md"),
    )


def build_delegation(
    orch_dir: Path, ledger_deleg: dict, telemetry_cost: dict,
    files_paths: set[str], warnings: Warnings,
) -> Delegation:
    did = ledger_deleg.get("delegation_id") or ""
    report = load_json(orch_dir / "reports" / f"{did}.json", warnings,
                       label=f"report {did}") or {}
    repairs_doc = report.get("repairs")
    repairs = None
    if isinstance(repairs_doc, dict):
        repairs = RepairSummary(
            max_attempts_in_a_cycle=to_int(repairs_doc.get("max_attempts_in_a_cycle")) or 0,
            total_attempts=to_int(repairs_doc.get("total_attempts")) or 0,
        )
    return Delegation(
        delegation_id=did,
        backend=ledger_deleg.get("backend") or "",
        track=ledger_deleg.get("track") or "",
        run_id=ledger_deleg.get("run_id") or "",
        run_path=ledger_deleg.get("run_path") or "",
        command=list(ledger_deleg.get("command") or []),
        resolved_executable=ledger_deleg.get("resolved_executable"),
        resolved_version=ledger_deleg.get("resolved_version"),
        model_identity=_model_identity(orch_dir, did, warnings),
        status=ledger_deleg.get("status") or "",
        clean_exit=ledger_deleg.get("clean_exit"),
        exit_code=to_int(ledger_deleg.get("exit_code")),
        spawned_at=ledger_deleg.get("spawned_at"),
        ended_at=ledger_deleg.get("ended_at"),
        timeout_minutes=to_float(ledger_deleg.get("timeout_minutes")),
        continue_run=bool(ledger_deleg.get("continue_run")),
        respawn_of=ledger_deleg.get("respawn_of"),
        taken_over=bool(ledger_deleg.get("taken_over")),
        budget=_budget(ledger_deleg, report),
        brief=_brief(orch_dir, ledger_deleg, warnings),
        agent_summary=report.get("agent_summary") or ledger_deleg.get("agent_summary"),
        champion=_champion(report),
        distress=_distress(report),
        repairs=repairs,
        cost=_cost(ledger_deleg, report, telemetry_cost),
        files=_files(did, files_paths),
    )
