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
)

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "flags": list(DISTRESS_FLAGS),
            "active": list(self.active),
            "detail": self.detail,
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
) -> DistressAssessment:
    """Evaluate every distress predicate against mechanical child-run state.

    Pure by construction: all inputs are plain values read elsewhere, so the
    policy is testable without a registry, a process, or a filesystem.
    """

    active: list[str] = []
    details: list[str] = []

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
            f"a cycle consumed all {MAX_REPAIR_ATTEMPTS} model attempts without passing validation"
        )

    if cycles_used > 0 and not any(d in _PROMOTING_DECISIONS for d in decisions):
        active.append("all_rejected")
        rejected = len([d for d in decisions if d])
        details.append(f"{rejected}/{cycles_used} decided cycles produced no promotion")

    if champion_model_family == BASELINE_MODEL_FAMILY:
        active.append("champion_is_baseline")
        details.append("the run never beat its flat-rate starting baseline")

    if calibration_ratio is not None and abs(calibration_ratio - 1.0) > CALIBRATION_TOLERANCE:
        active.append("calibration_anomaly")
        details.append(f"champion predicted/actual ratio is {calibration_ratio:.3f}")

    unknown = set(active) - set(DISTRESS_FLAGS)
    if unknown:  # defensive: keeps the closed set honest as flags are added
        raise AssertionError(f"assess_distress produced unknown flags: {sorted(unknown)}")

    return DistressAssessment(
        active=tuple(active),
        detail="; ".join(details) if details else "no distress signals",
    )


# ── registry reads ───────────────────────────────────────────────────────────


def _child_config(delegation: Delegation) -> ProjectConfig:
    return load_config(track_id=delegation.track, run_id=delegation.run_id)


def _cycles_used(registry_path: Path) -> int:
    from autoresearch.experiment_registry.sessions import list_sessions

    sessions = list_sessions(registry_path)
    if not sessions:
        return 0
    return max(int(s.get("current_cycle") or 0) for s in sessions)


def _max_repair_attempts_seen(run_dir: Path) -> int:
    """Highest ``repair_request_<N>.json`` index written anywhere in this run."""

    highest = 0
    for path in run_dir.rglob("repair_request_*.json"):
        stem = path.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest


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
        paired = comparison.get("paired_summary") or {}
        experiment_name = None
        if entry.get("experiment_id"):
            try:
                experiment = get_experiment(config.registry_path, entry["experiment_id"])
                experiment_name = (experiment or {}).get("experiment_name")
            except Exception:
                experiment_name = None
        rows.append(
            {
                "cycle": int(entry.get("cycle") or 0),
                "experiment_id": entry.get("experiment_id"),
                "name": experiment_name or entry.get("hypothesis") or "",
                "decision": comparison.get("decision"),
                "reason_code": comparison.get("decision_reason_code"),
                "lift_vs_champion": (
                    float(paired["mean_lift"]) if paired.get("mean_lift") is not None else None
                ),
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

    cycles_used = _cycles_used(config.registry_path)
    experiments = _experiment_rows(config)
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
    )

    return {
        "delegation_id": delegation.delegation_id,
        "orchestration_id": orch.orchestration_id,
        "run_id": delegation.run_id,
        "track": delegation.track,
        "backend": delegation.backend,
        "status": delegation.status,
        "cycles": {"budget": delegation.cycle_budget, "used": cycles_used},
        "champion": champion,
        "experiments": experiments,
        "agent_summary": delegation.agent_summary,
        "distress": distress.to_dict(),
        "cost": _cost(delegation, config),
    }


def collect_report(orch: Orchestration, delegation: Delegation) -> Path:
    """Build the delegation report and persist it under ``reports/``."""

    payload = build_report(orch, delegation)
    path = delegation_report_path(orch.orchestration_id, delegation.delegation_id)
    write_json(path, payload)
    return path


def _cost(delegation: Delegation, config: ProjectConfig) -> dict[str, Any]:
    wall_clock = _wall_clock_minutes(delegation)
    return {
        "wall_clock_minutes": wall_clock,
        "llm_usage": _llm_usage(config),
    }


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
