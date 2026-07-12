"""Delegation reports — built from the child's registry, never from its claims.

A sub-agent's ``finish-delegation`` summary is *testimony*: it is stored verbatim
as ``agent_summary`` and never used to compute a number. Every metric, decision,
and distress flag in the report is a mechanical read of the child run's registry
and its predictions artifact, so a sub-agent cannot flatter its own report.

Distress flags (design §4.5) are cheap, unambiguous predicates. The orchestrator
contract maps each to a response — refine the brief and respawn, or take over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig, load_config
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    report_path as delegation_report_path,
)
from autoresearch.utils.io import write_json


#: Every flag the report may raise. Kept as a closed set so the orchestrator
#: contract's distress-response table can be exhaustive.
DISTRESS_FLAGS = (
    "repair_exhausted",
    "all_rejected",
    "budget_overrun",
    "crashed",
    "no_finish_delegation",
    "champion_is_baseline",
    "calibration_anomaly",
    "cycles_forfeited",
)
INFORMATIONAL_FLAGS = ("early_stop", "auto_rejected")

#: Model family of the flat-rate experiment every run starts from.
BASELINE_MODEL_FAMILY = "global_mean"

#: |predicted/actual - 1| beyond this is a calibration anomaly worth a look.
CALIBRATION_TOLERANCE = 0.10

#: The framework allows 3 model attempts per cycle; a 3rd repair request means
#: the cycle burned every attempt it had.
MAX_REPAIR_ATTEMPTS = 3

_PROMOTING_DECISIONS = frozenset({"promote", "local_promote"})


@dataclass(frozen=True)
class DistressAssessment:
    """Which distress predicates fired, and why."""

    active: tuple[str, ...]
    detail: str
    informational: tuple[str, ...] = ()
    informational_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": list(DISTRESS_FLAGS),
            "active": list(self.active),
            "detail": self.detail,
            "informational": list(self.informational),
            "informational_detail": self.informational_detail,
        }


# ── distress predicates (pure; unit-tested) ──────────────────────────────────


def assess_distress(
    *,
    status: str,
    exit_code: int | None,
    agent_summary: str | None,
    cycles_used: int,
    cycle_budget: int,
    decisions: list[str],
    champion_model_family: str | None,
    calibration_ratio: float | None,
    max_repair_attempts_seen: int,
    cycles_attempted: int | None = None,
    nonterminal_proposals: int = 0,
) -> DistressAssessment:
    """Evaluate every distress predicate against mechanical child-run state.

    Pure by construction: all inputs are plain values read elsewhere, so the
    policy is testable without a registry, a process, or a filesystem.
    """

    active: list[str] = []
    details: list[str] = []
    informational: list[str] = []
    informational_details: list[str] = []

    if status == "failed" or (exit_code is not None and exit_code != 0):
        active.append("crashed")
        details.append(f"child exited with status={status}, exit_code={exit_code}")

    if status == "timed_out":
        active.append("budget_overrun")
        details.append("wall-clock timeout reached before the child finished")
    elif cycles_used > cycle_budget:
        active.append("budget_overrun")
        details.append(f"{cycles_used} cycles used against a budget of {cycle_budget}")

    if agent_summary is None or not str(agent_summary).strip():
        active.append("no_finish_delegation")
        details.append("child never called `orchestrate finish-delegation`")

    if max_repair_attempts_seen >= MAX_REPAIR_ATTEMPTS:
        active.append("repair_exhausted")
        details.append(
            f"a cycle needed its {MAX_REPAIR_ATTEMPTS}rd and final model attempt"
        )

    auto_rejected = sum(1 for d in decisions if d == "auto_reject")
    if auto_rejected:
        informational.append("auto_rejected")
        informational_details.append(f"{auto_rejected} screening-gate decision(s) recorded by the framework")

    if cycles_used > 0 and decisions and not any(d in _PROMOTING_DECISIONS for d in decisions):
        active.append("all_rejected")
        details.append(f"{len(decisions)}/{cycles_used} decided cycles produced no promotion")

    if champion_model_family == BASELINE_MODEL_FAMILY:
        active.append("champion_is_baseline")
        details.append("the run never beat its flat-rate starting baseline")

    if calibration_ratio is not None and abs(calibration_ratio - 1.0) > CALIBRATION_TOLERANCE:
        active.append("calibration_anomaly")
        details.append(f"champion predicted/actual ratio is {calibration_ratio:.3f}")

    decided = len([d for d in decisions if d])
    forfeit_details: list[str] = []
    if cycles_attempted is not None and cycles_attempted > decided:
        forfeit_details.append(
            f"{cycles_attempted} experiments attempted but only {decided} reached a decision"
        )
    if nonterminal_proposals > 0:
        forfeit_details.append(
            f"{nonterminal_proposals} proposal(s) left nonterminal at exit"
        )
    if forfeit_details:
        active.append("cycles_forfeited")
        details.append("; ".join(forfeit_details))

    if (status == "completed" and cycles_used < cycle_budget and nonterminal_proposals == 0
            and (cycles_attempted is None or cycles_attempted == decided)):
        informational.append("early_stop")
        informational_details.append(f"brief ended cleanly after {cycles_used}/{cycle_budget} budgeted cycles")

    unknown = set(active) - set(DISTRESS_FLAGS)
    if unknown:  # defensive: keeps the closed set honest as flags are added
        raise AssertionError(f"assess_distress produced unknown flags: {sorted(unknown)}")

    return DistressAssessment(
        active=tuple(active),
        detail="; ".join(details) if details else "no distress signals",
        informational=tuple(informational),
        informational_detail="; ".join(informational_details),
    )


# ── registry reads ───────────────────────────────────────────────────────────


def _child_config(delegation: Delegation) -> ProjectConfig:
    return load_config(track_id=delegation.track, run_id=delegation.run_id)


def _cycles_used(registry_path: Path) -> int:
    from autoresearch.experiment_registry.sessions import list_sessions

    sessions = list_sessions(registry_path)
    if not sessions:
        return 0
    return sum(int(s.get("current_cycle") or 0) for s in sessions)


def _cycle_accounting(config: ProjectConfig, delegation: Delegation) -> dict[str, int]:
    """Separate reserved budget from attempted, completed, and decided work."""

    from autoresearch.experiment_registry.registry import list_experiments

    attempted = sum(
        1
        for experiment in list_experiments(config.registry_path)
        if experiment.get("model_family") != BASELINE_MODEL_FAMILY
        and "delegation_seed_" not in str(experiment.get("experiment_name") or "")
    )
    attempted = max(0, attempted - delegation.cycles_at_start)
    completed = max(0, _cycles_used(config.registry_path) - delegation.cycles_at_start)
    rows = _experiment_rows(config)
    if delegation.continue_run:
        rows = rows[delegation.cycles_at_start :]
    decided = sum(1 for row in rows if row.get("decision"))
    return {
        "budget": delegation.cycle_budget,
        "attempted": attempted,
        "completed": completed,
        "decided": decided,
        # Backward-compatible alias. A used cycle reached a framework result;
        # whether the LLM supplied a verdict is reported separately.
        "used": completed,
    }


def _nonterminal_proposal_count(config: ProjectConfig) -> int:
    """Proposals that never reached a terminal outcome in this run."""

    from autoresearch.controller.workflow import INFLIGHT_PROPOSAL_STATUSES
    from autoresearch.experiment_registry.registry import list_proposals

    nonterminal = {"proposed", "validated", "queued", "needs_repair", "awaiting_decision"} | set(
        INFLIGHT_PROPOSAL_STATUSES
    )
    return sum(
        1
        for proposal in list_proposals(config.registry_path)
        if proposal.get("status") in nonterminal
    )


def _max_repair_attempts_seen(run_dir: Path) -> int:
    """Highest ``repair_request_<N>.json`` index written anywhere in this run."""

    highest = 0
    for path in run_dir.rglob("repair_request_*.json"):
        stem = path.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest


def _repair_requests_seen(run_dir: Path) -> int:
    """How many repair requests this run's cycles provoked in total.

    ``_max_repair_attempts_seen`` answers "did any one cycle burn all its
    attempts" (the ``repair_exhausted`` predicate). The scorecard's
    "repair attempts per cycle" wants the total instead, so both are recorded.
    """

    return sum(
        1
        for path in run_dir.rglob("repair_request_*.json")
        if path.stem.rsplit("_", 1)[-1].isdigit()
    )


def _champion_metrics(config: ProjectConfig, experiment_id: str) -> dict[str, float]:
    """Score the champion's predictions on the search-validation split.

    Uses the same privileged read as :mod:`autoresearch.tracks` — the predictions
    artifact plus ``full_metric_panel`` — so the numbers are computed, not copied
    from whatever the sub-agent wrote down.
    """

    import pandas as pd

    from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns
    from autoresearch.experiment_registry.registry import list_artifacts

    predictions_path: Path | None = None
    for artifact in list_artifacts(config.registry_path, experiment_id):
        if artifact["artifact_type"] == "predictions":
            predictions_path = Path(artifact["path"])
            break
    if predictions_path is None or not predictions_path.exists():
        return {}

    frame = pd.read_parquet(predictions_path)
    eval_split = config.ordinary_eval_splits[0]
    frame = frame[frame["split"] == eval_split]
    if frame.empty:
        return {}

    actual_col, predicted_col = prediction_target_columns(frame, config.target_mode)
    panel = full_metric_panel(
        frame[actual_col],
        frame[predicted_col],
        frame["exposure"],
        tweedie_power=config.tweedie_power,
        target_mode=config.target_mode,
    )
    return {
        "gini_weighted": float(panel["gini_weighted"]),
        "rank_gini_weighted": float(panel["rank_gini_weighted"]),
        "asym_pricing_loss": float(panel["asym_pricing_loss"]),
        "calibration_ratio": float(panel["predicted_to_actual_ratio"]),
    }


def _experiment_rows(config: ProjectConfig) -> list[dict[str, Any]]:
    """One row per completed cycle: what was tried, decided, and learned."""

    from autoresearch.experiment_registry.comparisons import list_comparisons
    from autoresearch.experiment_registry.registry import get_experiment
    from autoresearch.experiment_registry.research_log import list_research_log_entries

    comparisons = {c["comparison_id"]: c for c in list_comparisons(config.registry_path)}

    rows: list[dict[str, Any]] = []
    for entry in list_research_log_entries(config.registry_path):
        comparison = comparisons.get(entry.get("comparison_id")) or {}
        outcome = str(entry.get("outcome") or "")
        decision = comparison.get("decision")
        if not decision and outcome.startswith("auto_reject"):
            decision = "auto_reject"
        paired = comparison.get("paired_summary") or {}
        experiment_name = None
        if entry.get("experiment_id"):
            try:
                experiment = get_experiment(config.registry_path, entry["experiment_id"])
                experiment_name = (experiment or {}).get("experiment_name")
            except Exception:
                experiment_name = None
        vs_baseline = None
        if paired.get("champion_id"):
            try:
                incumbent = get_experiment(config.registry_path, paired["champion_id"]) or {}
                vs_baseline = incumbent.get("model_family") == BASELINE_MODEL_FAMILY
            except Exception:
                vs_baseline = None
        rows.append(
            {
                "cycle": int(entry.get("cycle") or 0),
                "experiment_id": entry.get("experiment_id"),
                "name": experiment_name or entry.get("hypothesis") or "",
                "decision": decision,
                "reason_code": comparison.get("decision_reason_code"),
                "lift_vs_champion": (
                    float(paired["mean_lift"]) if paired.get("mean_lift") is not None else None
                ),
                # Baseline-relative lifts (vs the flat global_mean start) are an
                # order of magnitude larger than incremental champion-vs-champion
                # lifts; downstream aggregation must not average the two together.
                "vs_baseline": vs_baseline,
                "interpretation": entry.get("interpretation"),
            }
        )
    return rows


def _champion_facts(config: ProjectConfig) -> dict[str, Any]:
    from autoresearch.experiment_registry.registry import get_experiment, get_official_champion

    champion = get_official_champion(config.registry_path)
    if champion is None:
        return {}
    champion_id = champion["champion_id"]
    experiment = get_experiment(config.registry_path, champion_id) or {}
    facts: dict[str, Any] = {
        "experiment_id": champion_id,
        "model_family": experiment.get("model_family"),
        "target_strategy": experiment.get("target_strategy"),
    }
    facts.update(_champion_metrics(config, champion_id))
    facts["beat_seed_baseline"] = experiment.get("model_family") != BASELINE_MODEL_FAMILY
    return facts


# ── report ───────────────────────────────────────────────────────────────────


def build_report(orch: Orchestration, delegation: Delegation) -> dict[str, Any]:
    """Assemble the delegation report from child-run state (no side effects)."""

    config = _child_config(delegation)
    # ``artifacts_dir`` *is* the run folder for a tracked run; deriving it from the
    # config rather than rebuilding the path keeps this testable against a fixture.
    run_dir = config.artifacts_dir

    cycle_accounting = _cycle_accounting(config, delegation)
    cycles_used = cycle_accounting["completed"]
    experiments = _experiment_rows(config)
    if delegation.continue_run:
        # Positional slice: research-log entries' own cycle numbers restart per
        # session, so the entry count is the only stable offset across sessions.
        experiments = experiments[delegation.cycles_at_start :]
    champion = _champion_facts(config)
    decisions = [row["decision"] for row in experiments if row.get("decision")]

    distress = assess_distress(
        status=delegation.status,
        exit_code=delegation.exit_code,
        agent_summary=delegation.agent_summary,
        cycles_used=cycles_used,
        cycle_budget=delegation.cycle_budget,
        decisions=decisions,
        champion_model_family=champion.get("model_family"),
        calibration_ratio=champion.get("calibration_ratio"),
        max_repair_attempts_seen=_max_repair_attempts_seen(run_dir),
        cycles_attempted=cycle_accounting["attempted"],
        nonterminal_proposals=_nonterminal_proposal_count(config),
    )

    return {
        "delegation_id": delegation.delegation_id,
        "orchestration_id": orch.orchestration_id,
        "run_id": delegation.run_id,
        "track": delegation.track,
        "backend": delegation.backend,
        "respawn_of": delegation.respawn_of,
        "continue_run": delegation.continue_run,
        "taken_over": delegation.taken_over,
        "status": delegation.status,
        "cycles": cycle_accounting,
        "champion": champion,
        "experiments": experiments,
        "repairs": {
            "requests": _repair_requests_seen(run_dir),
            "max_attempts_in_a_cycle": _max_repair_attempts_seen(run_dir),
        },
        "agent_summary": delegation.agent_summary,
        "distress": distress.to_dict(),
        "cost": _cost(delegation, config),
    }


def should_refund_budget(payload: dict[str, Any]) -> bool:
    """A delegation that crashed before doing any work costs the campaign nothing.

    Refund iff the delegation ended in a non-completed terminal state, ran zero
    cycles, and made zero recorded LLM calls — i.e. the failure was environmental
    (auth, sandbox, missing binary), not scientific. A ``completed`` delegation
    that chose to do nothing is not refunded; the distress flags cover that.
    """

    if payload.get("status") not in {"failed", "killed", "timed_out"}:
        return False
    if payload.get("taken_over"):
        return False  # the orchestrator is spending this budget in the run itself
    if int((payload.get("cycles") or {}).get("used") or 0) != 0:
        return False
    llm_usage = (payload.get("cost") or {}).get("llm_usage") or {}
    if int(llm_usage.get("calls") or 0) != 0:
        return False
    backend_usage = llm_usage.get("backend") or {}
    tokens = int(backend_usage.get("input_tokens") or 0) + int(
        backend_usage.get("output_tokens") or 0
    )
    return tokens == 0


def collect_report(orch: Orchestration, delegation: Delegation) -> Path:
    """Build the delegation report and persist it under ``reports/``."""

    payload = build_report(orch, delegation)
    path = delegation_report_path(orch.orchestration_id, delegation.delegation_id)
    write_json(path, payload)
    return path


def _cost(delegation: Delegation, config: ProjectConfig) -> dict[str, Any]:
    wall_clock = _wall_clock_minutes(delegation)
    llm_usage = _llm_usage(config)
    if delegation.tool_usage:
        llm_usage["backend"] = delegation.tool_usage
        if llm_usage.get("cost_usd") is None:
            backend_cost = delegation.tool_usage.get("cost_usd")
            if backend_cost is None:
                backend_cost = delegation.tool_usage.get("provider_cost_usd")
            if backend_cost is not None:
                llm_usage["cost_usd"] = float(backend_cost)
                llm_usage["cost_source"] = "backend_exit"
        if llm_usage.get("cost_usd") is None:
            estimated = _estimated_backend_cost(delegation.backend, delegation.tool_usage)
            if estimated is not None:
                llm_usage["cost_usd"] = estimated
                llm_usage["cost_source"] = "backend_pricing"
                llm_usage["cost_estimated"] = True
    return {
        "wall_clock_minutes": wall_clock,
        "llm_usage": llm_usage,
    }


def _estimated_backend_cost(backend_name: str, usage: dict[str, Any]) -> float | None:
    from autoresearch.orchestration.backends import get_backend

    try:
        backend = get_backend(backend_name)
    except (KeyError, ValueError, FileNotFoundError):
        return None
    rates = (backend.usd_per_mtok_input, backend.usd_per_mtok_cached, backend.usd_per_mtok_output)
    if all(rate is None for rate in rates):
        return None
    details = usage.get("details") if isinstance(usage.get("details"), dict) else {}
    cached = int(usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens")
                 or details.get("cached_input_tokens") or details.get("cache_read_input_tokens") or 0)
    total_input = int(usage.get("input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    cost = (max(0, total_input - cached) * (backend.usd_per_mtok_input or 0.0)
            + cached * (backend.usd_per_mtok_cached or 0.0)
            + output * (backend.usd_per_mtok_output or 0.0)) / 1_000_000
    return round(cost, 6)


def _wall_clock_minutes(delegation: Delegation) -> float | None:
    from datetime import datetime

    if not delegation.spawned_at or not delegation.ended_at:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    try:
        started = datetime.strptime(delegation.spawned_at, fmt)
        ended = datetime.strptime(delegation.ended_at, fmt)
    except ValueError:
        return None
    return round((ended - started).total_seconds() / 60.0, 2)


def _llm_usage(config: ProjectConfig) -> dict[str, Any]:
    """Best-effort usage rollup from the child run's telemetry DB."""

    import sqlite3

    from autoresearch.telemetry.store import telemetry_path

    db = telemetry_path(config.artifacts_dir)
    if not db.exists():
        return {}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT COUNT(*) AS calls,
                   COALESCE(SUM(input_tokens), 0) AS input_tokens,
                   COALESCE(SUM(output_tokens), 0) AS output_tokens,
                   SUM(provider_cost_usd) AS cost_usd
            FROM llm_model_calls
            """
        ).fetchone()
        models = con.execute(
            "SELECT DISTINCT model FROM llm_model_calls WHERE model IS NOT NULL"
        ).fetchall()
        con.close()
    except Exception:
        return {}
    return {
        "calls": int(row["calls"]),
        "input_tokens": int(row["input_tokens"]),
        "output_tokens": int(row["output_tokens"]),
        "cost_usd": (float(row["cost_usd"]) if row["cost_usd"] is not None else None),
        "models": sorted({str(m["model"]) for m in models if m["model"]}),
    }
