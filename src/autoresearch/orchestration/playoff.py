"""Playoff consolidation and privileged cross-run experiment replay.

The replay path is shared by campaign playoffs and delegation seed champions.
It imports only model/proposal artifacts, runs a fresh experiment in the
destination run, and writes an audit record that preserves the source run,
experiment, and (when applicable) orchestration delegation.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import tomli_w

from autoresearch.config import PROJECT_ROOT, ProjectConfig, load_config
from autoresearch.orchestration.manifest import (
    Consolidation,
    Orchestration,
    load_orchestration,
    manifest_lock,
    playoff_dir,
    report_path as delegation_report_path,
    save_orchestration,
    write_consolidation_backpointer,
)
from autoresearch.utils.io import read_json, write_json


PLAYOFF_JSON = "playoff_report.json"
PLAYOFF_MARKDOWN = "playoff_report.md"


@dataclass(frozen=True)
class ReplaySource:
    """One source experiment eligible for privileged replay."""

    track: str
    run_id: str
    experiment_id: str
    delegation_id: str | None = None
    orchestration_id: str | None = None

    @property
    def run_ref(self) -> str:
        return f"{self.track}/{self.run_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "delegation_id": self.delegation_id,
            "orchestration_id": self.orchestration_id,
        }


@dataclass(frozen=True)
class Finalist:
    """A non-baseline delegation champion ordered by validation Gini."""

    delegation_id: str
    source: ReplaySource
    gini_weighted: float
    model_family: str
    report_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "delegation_id": self.delegation_id,
            "source": self.source.to_dict(),
            "gini_weighted": self.gini_weighted,
            "model_family": self.model_family,
            "report_path": self.report_path,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Finalist":
        source = raw["source"]
        return cls(
            delegation_id=str(raw["delegation_id"]),
            source=ReplaySource(
                track=str(source["track"]),
                run_id=str(source["run_id"]),
                experiment_id=str(source["experiment_id"]),
                delegation_id=source.get("delegation_id"),
                orchestration_id=source.get("orchestration_id"),
            ),
            gini_weighted=float(raw["gini_weighted"]),
            model_family=str(raw["model_family"]),
            report_path=str(raw["report_path"]),
        )


def collect_finalists(
    orch: Orchestration, include: Iterable[str] | None = None
) -> tuple[list[Finalist], list[dict[str, Any]]]:
    """Collect and rank eligible delegation champions from persisted reports."""

    selected = list(include) if include is not None else [d.delegation_id for d in orch.delegations]
    if len(selected) != len(set(selected)):
        raise ValueError("--include contains duplicate delegation ids")
    known = {d.delegation_id: d for d in orch.delegations}
    unknown = sorted(set(selected) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown delegation(s) in --include: {', '.join(unknown)}; "
            f"known: {', '.join(sorted(known)) or '(none)'}"
        )

    finalists: list[Finalist] = []
    exclusions: list[dict[str, Any]] = []
    for delegation_id in selected:
        delegation = known[delegation_id]
        path = _delegation_report_file(orch, delegation)
        if not path.exists():
            exclusions.append(
                {"delegation_id": delegation_id, "reason": "report_missing", "path": str(path)}
            )
            continue
        payload = read_json(path)
        champion = payload.get("champion") or {}
        distress = set((payload.get("distress") or {}).get("active") or ())
        family = str(champion.get("model_family") or "")
        if family == "global_mean" or "champion_is_baseline" in distress or bool(
            champion.get("champion_is_baseline")
        ):
            exclusions.append(
                {"delegation_id": delegation_id, "reason": "champion_is_baseline"}
            )
            continue
        experiment_id = champion.get("experiment_id")
        gini = champion.get("gini_weighted")
        if not experiment_id or gini is None:
            exclusions.append(
                {"delegation_id": delegation_id, "reason": "champion_or_gini_missing"}
            )
            continue
        finalists.append(
            Finalist(
                delegation_id=delegation_id,
                source=ReplaySource(
                    track=delegation.track,
                    run_id=delegation.run_id,
                    experiment_id=str(experiment_id),
                    delegation_id=delegation_id,
                    orchestration_id=orch.orchestration_id,
                ),
                gini_weighted=float(gini),
                model_family=family,
                report_path=str(path),
            )
        )
    finalists.sort(key=lambda item: (item.gini_weighted, item.delegation_id))
    return finalists, exclusions


def replay_source_for_delegation(orch: Orchestration, delegation_id: str) -> ReplaySource:
    """Resolve ``from:dNN`` to that delegation's registry-backed champion."""

    delegation = orch.delegation(delegation_id)
    from autoresearch.experiment_registry.registry import get_official_champion

    config = load_config(track_id=delegation.track, run_id=delegation.run_id)
    champion = get_official_champion(config.registry_path)
    if champion is None:
        raise ValueError(f"Delegation {delegation_id} has no official champion to seed from")
    return ReplaySource(
        track=delegation.track,
        run_id=delegation.run_id,
        experiment_id=str(champion["champion_id"]),
        delegation_id=delegation_id,
        orchestration_id=orch.orchestration_id,
    )


def replay_experiment(
    destination: ProjectConfig,
    source: ReplaySource,
    *,
    label: str,
) -> dict[str, Any]:
    """Refit one source experiment inside *destination* and record its lineage.

    Recipe specifications are portable JSON. Script finalists additionally copy
    their exact run-local script and originating proposal into the destination's
    ``orchestration_replay`` audit folder before execution.
    """

    from autoresearch.experiment_registry.registry import (
        get_experiment,
        get_official_champion,
        list_artifacts,
        list_proposals,
        record_experiment_artifacts,
    )
    from autoresearch.experiment_runner import run_experiment

    source_config = load_config(track_id=source.track, run_id=source.run_id)
    _validate_replay_compatibility(source_config, destination)
    experiment = get_experiment(source_config.registry_path, source.experiment_id)
    snapshot_path = Path(str(experiment.get("config_snapshot_path") or ""))
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Source experiment {source.run_ref}/{source.experiment_id} has no config snapshot"
        )
    snapshot = read_json(snapshot_path)
    snapshot_target_mode = str(snapshot.get("target_mode") or source_config.target_mode)
    if snapshot_target_mode != destination.target_mode:
        raise ValueError(
            f"Cannot replay across target modes: source={snapshot_target_mode}, "
            f"destination={destination.target_mode}"
        )
    source_exp = snapshot.get("experiment")
    if not isinstance(source_exp, dict):
        raise ValueError(f"Invalid source config snapshot at {snapshot_path}: missing experiment")
    source_model = source_exp.get("model")
    if not isinstance(source_model, dict):
        raise ValueError(f"Invalid source config snapshot at {snapshot_path}: missing model")
    _validate_fixed_preprocessing(snapshot, source_exp, destination)

    replay_dir = destination.artifacts_dir / "orchestration_replay" / _safe_label(label)
    existing_audit = replay_dir / "replay_manifest.json"
    if existing_audit.exists():
        existing = read_json(existing_audit)
        destination_snapshot = replay_dir / "experiment" / "config_snapshot.json"
        destination_id = existing.get("destination_experiment_id")
        if not destination_id and destination_snapshot.exists():
            destination_id = read_json(destination_snapshot).get("experiment_id")
            existing["destination_experiment_id"] = destination_id
            write_json(existing_audit, existing)
        if destination_id:
            try:
                get_experiment(destination.registry_path, str(destination_id))
            except ValueError:
                pass
            else:
                recovered_artifacts = {"orchestration_replay_manifest": existing_audit}
                copied_proposal = existing.get("source_proposal")
                if copied_proposal and Path(str(copied_proposal)).exists():
                    recovered_artifacts["orchestration_source_proposal"] = Path(
                        str(copied_proposal)
                    )
                record_experiment_artifacts(
                    destination.registry_path,
                    str(destination_id),
                    recovered_artifacts,
                )
                _append_lineage(destination, existing)
                return existing
        raise RuntimeError(
            f"Incomplete replay audit already exists at {existing_audit}; "
            "inspect the failed import before retrying"
        )
    replay_dir.mkdir(parents=True, exist_ok=False)
    source_snapshot_copy = replay_dir / "source_config_snapshot.json"
    shutil.copy2(snapshot_path, source_snapshot_copy)

    proposals = [
        proposal
        for proposal in list_proposals(source_config.registry_path)
        if proposal.get("experiment_id") == source.experiment_id
    ]
    source_proposal_path: Path | None = None
    copied_proposal: Path | None = None
    if proposals and proposals[0].get("proposal_path"):
        source_proposal_path = Path(str(proposals[0]["proposal_path"]))
        if source_proposal_path.exists():
            copied_proposal = replay_dir / "source_proposal.json"
            shutil.copy2(source_proposal_path, copied_proposal)

    model = json.loads(json.dumps(source_model))
    is_recipe = isinstance(model.get("recipe"), dict)
    _ensure_foundation_support(model, source, destination)
    copied_script: Path | None = None
    if not is_recipe:
        script_source = _source_script_path(
            snapshot,
            list_artifacts(source_config.registry_path, source.experiment_id),
        )
        if script_source is None or not script_source.exists():
            raise FileNotFoundError(
                f"Script finalist {source.run_ref}/{source.experiment_id} "
                "has no readable model script"
            )
        if copied_proposal is None:
            raise FileNotFoundError(
                f"Script finalist {source.run_ref}/{source.experiment_id} "
                "has no readable source proposal"
            )
        copied_script = replay_dir / "model_replay.py"
        shutil.copy2(script_source, copied_script)
        model.pop("model_script_path", None)
        model["script_path"] = copied_script.name
        model["script_sha256"] = _sha256(copied_script)

    champion = get_official_champion(destination.registry_path)
    if champion is None:
        raise ValueError("Destination run has no official champion")
    replay_config = json.loads(json.dumps(source_exp))
    replay_config["experiment_name"] = _replay_experiment_name(label, source_exp)
    replay_config["parent_experiment_id"] = champion["champion_id"]
    replay_config["model"] = model
    config_path = replay_dir / "experiment_config.toml"
    config_path.write_text(_to_toml(replay_config), encoding="utf-8")

    audit_path = replay_dir / "replay_manifest.json"
    audit: dict[str, Any] = {
        "source": source.to_dict(),
        "destination": {"track": destination.track_id, "run_id": destination.run_id},
        "representation": "recipe" if is_recipe else "script",
        "source_config_snapshot": str(source_snapshot_copy),
        "source_config_sha256": _sha256(source_snapshot_copy),
        "source_proposal": str(copied_proposal) if copied_proposal else None,
        "source_proposal_sha256": _sha256(copied_proposal) if copied_proposal else None,
        "copied_script": str(copied_script) if copied_script else None,
        "copied_script_sha256": _sha256(copied_script) if copied_script else None,
        "destination_experiment_id": None,
    }
    write_json(audit_path, audit)
    outputs = run_experiment(destination, config_path, output_dir=replay_dir / "experiment")
    destination_snapshot = read_json(outputs["config_snapshot"])
    destination_experiment_id = str(destination_snapshot["experiment_id"])
    audit["destination_experiment_id"] = destination_experiment_id
    write_json(audit_path, audit)

    replay_artifacts = {"orchestration_replay_manifest": audit_path}
    if copied_proposal is not None:
        replay_artifacts["orchestration_source_proposal"] = copied_proposal
    record_experiment_artifacts(
        destination.registry_path, destination_experiment_id, replay_artifacts
    )
    _append_lineage(destination, audit)
    return audit


def _recipe_foundation_estimators(model: dict[str, Any]) -> set[str]:
    """Foundation estimator names a recipe model references (empty for scripts)."""

    from autoresearch.models.recipe.foundation_specs import FOUNDATION_ESTIMATOR_NAMES

    recipe = model.get("recipe")
    if not isinstance(recipe, dict):
        return set()
    names: set[str] = set()
    estimator = recipe.get("estimator")
    if isinstance(estimator, str):
        names.add(estimator)
    stages = recipe.get("stages")
    if isinstance(stages, dict):
        for stage in stages.values():
            if isinstance(stage, dict) and isinstance(stage.get("estimator"), str):
                names.add(stage["estimator"])
    return names & set(FOUNDATION_ESTIMATOR_NAMES)


def _ensure_foundation_support(
    model: dict[str, Any], source: ReplaySource, destination: ProjectConfig
) -> None:
    """Make a foundation finalist replayable, or fail with a clear message.

    A foundation estimator only appears in the recipe interpreter's registry when
    the package is importable and it has been enabled. Replaying a TabPFN finalist
    into a destination that cannot support it would otherwise fail deep in
    ``run_experiment`` as a misleading "unknown estimator". Instead: if the extra
    is missing, refuse the replay here naming the fix; if it is present, register
    the estimators so the fresh fit can interpret the portable recipe.
    """

    needed = _recipe_foundation_estimators(model)
    if not needed:
        return

    from autoresearch.models.recipe.foundation import tabpfn_available

    if not tabpfn_available():
        raise RuntimeError(
            f"Cannot replay {source.run_ref}/{source.experiment_id} into "
            f"{destination.track_id}/{destination.run_id}: it uses foundation "
            f"estimator(s) {sorted(needed)} but the [foundation] extra is not "
            "importable in this environment. Install it "
            "(`pip install -e '.[foundation]'`) before consolidating a foundation "
            "finalist."
        )
    from autoresearch.models.recipe import enable_foundation_models

    enable_foundation_models()


def seed_champion_from_source(
    destination: ProjectConfig,
    source: ReplaySource,
    *,
    label: str,
) -> dict[str, Any]:
    """Replay *source* and install it as the destination's starting champion."""

    from autoresearch.experiment_registry.registry import (
        get_official_champion,
        set_official_champion,
        upsert_branch,
    )

    replay = replay_experiment(destination, source, label=label)
    experiment_id = str(replay["destination_experiment_id"])
    official = get_official_champion(destination.registry_path) or {}
    branch_id = str(official.get("branch_id") or "main")
    upsert_branch(
        destination.registry_path,
        branch_id=branch_id,
        parent_branch_id=None,
        root_experiment_id=official.get("champion_id"),
        current_experiment_id=experiment_id,
        status="active",
        description=f"Seeded by privileged replay from {source.run_ref}/{source.experiment_id}.",
    )
    set_official_champion(
        destination.registry_path,
        champion_id=experiment_id,
        branch_id=branch_id,
        reason=(
            f"Seeded by orchestration replay from {source.run_ref}/{source.experiment_id}"
            + (f" ({source.delegation_id})" if source.delegation_id else "")
        ),
        action="seeded",
    )
    return replay


def run_playoff(
    orchestration_id: str,
    *,
    include: Iterable[str] | None = None,
    auto_decide: bool = False,
) -> dict[str, Any]:
    """Create or resume a campaign playoff and return its durable report."""

    orch = load_orchestration(orchestration_id)
    json_path = playoff_dir(orchestration_id) / PLAYOFF_JSON
    requested_include = list(include) if include is not None else None

    if orch.status == "completed" and json_path.exists():
        return read_json(json_path)

    if orch.consolidation.run_id is None:
        finalists, exclusions = collect_finalists(orch, requested_include)
        if not finalists:
            raise ValueError("No eligible finalists: every included delegation was excluded")
        with manifest_lock(orchestration_id):
            orch = load_orchestration(orchestration_id)
            if orch.consolidation.run_id is not None:
                raise RuntimeError(
                    "Another playoff initialised this campaign concurrently; rerun the command"
                )
            consolidation = _create_consolidation_run(
                orch,
                finalists[0].source.track,
                len(finalists) - 1,
                enable_foundation_models=_any_finalist_needs_foundation(finalists),
            )
            write_consolidation_backpointer(
                consolidation.artifacts_dir,
                orchestration_id=orchestration_id,
                target_mode=orch.target_mode,
            )
            orch = replace(
                orch,
                status="consolidating",
                consolidation=Consolidation(
                    track=consolidation.track_id,
                    run_id=consolidation.run_id,
                    playoff_report=_stored_path(json_path),
                ),
            )
            save_orchestration(orch)
        state = {
            "orchestration_id": orchestration_id,
            "status": "consolidating",
            "decision_mode": "auto" if auto_decide else "interactive",
            "included": requested_include,
            "exclusions": exclusions,
            "finalists": [item.to_dict() for item in finalists],
            "consolidation": {
                "track": consolidation.track_id,
                "run_id": consolidation.run_id,
            },
            "seed": None,
            "pairings": [],
            "final_champion_lineage": None,
        }
        _write_playoff_reports(orchestration_id, state)
    else:
        if not json_path.exists():
            raise RuntimeError(
                f"Consolidation run {orch.consolidation.track}/{orch.consolidation.run_id} "
                "exists but its playoff report is missing"
            )
        state = read_json(json_path)
        if requested_include is not None and requested_include != state.get("included"):
            raise ValueError("Cannot change --include after a playoff has started")
        state["decision_mode"] = "auto" if auto_decide else "interactive"
        consolidation = _load_consolidation_config(orch)

    finalists = [Finalist.from_dict(item) for item in state["finalists"]]
    if state.get("seed") is None:
        seed = seed_champion_from_source(
            consolidation,
            finalists[0].source,
            label=f"seed_{finalists[0].delegation_id}",
        )
        state["seed"] = {
            "finalist": finalists[0].to_dict(),
            "destination_experiment_id": seed["destination_experiment_id"],
            "replay_manifest": str(
                consolidation.artifacts_dir
                / "orchestration_replay"
                / _safe_label(f"seed_{finalists[0].delegation_id}")
                / "replay_manifest.json"
            ),
        }
        _write_playoff_reports(orchestration_id, state)

    pending = _sync_pairing_decisions(consolidation, state)
    _write_playoff_reports(orchestration_id, state)
    if pending is not None:
        if not auto_decide:
            return state
        _auto_decide_pairing(consolidation, pending)
        _sync_pairing_decisions(consolidation, state)
        _write_playoff_reports(orchestration_id, state)

    while len(state["pairings"]) < len(finalists) - 1:
        finalist = finalists[len(state["pairings"]) + 1]
        replay = replay_experiment(
            consolidation,
            finalist.source,
            label=f"challenger_{len(state['pairings']) + 1:02d}_{finalist.delegation_id}",
        )
        challenger_id = str(replay["destination_experiment_id"])
        comparison_outputs = _compare_replayed_experiment(consolidation, challenger_id)
        comparison_report = read_json(comparison_outputs["promotion_report"])
        gate_evidence = _gate_evidence(comparison_report)
        pairing = {
            "order": len(state["pairings"]) + 1,
            "finalist": finalist.to_dict(),
            "incumbent_experiment_id": comparison_report.get("champion_id")
            or (comparison_report.get("comparison_summary") or {}).get("champion_id"),
            "replayed_experiment_id": challenger_id,
            "comparison_id": comparison_report["comparison_id"],
            "comparison_summary": comparison_report.get("comparison_summary") or {},
            "gate_evidence": gate_evidence,
            "decision": None,
        }
        state["pairings"].append(pairing)
        _write_playoff_reports(orchestration_id, state)
        if not auto_decide:
            return state
        _auto_decide_pairing(consolidation, pairing)
        _sync_pairing_decisions(consolidation, state)
        _write_playoff_reports(orchestration_id, state)

    _complete_playoff(orch, consolidation, state)
    return state


def _any_finalist_needs_foundation(finalists: list[Finalist]) -> bool:
    """True if any finalist's source run opted into foundation estimators.

    Read from each source run's manifest flag, so the consolidation run enables
    foundation models iff it will actually need to replay a foundation finalist.
    """

    from autoresearch.bootstrap import read_run_manifest

    for finalist in finalists:
        source_config = load_config(
            track_id=finalist.source.track, run_id=finalist.source.run_id
        )
        if read_run_manifest(source_config).get("foundation_models"):
            return True
    return False


def _create_consolidation_run(
    orch: Orchestration,
    track: str,
    challenger_count: int,
    *,
    enable_foundation_models: bool = False,
) -> ProjectConfig:
    from autoresearch.bootstrap import bootstrap_track

    config = load_config(track_id=track, new_run=True, dataset=orch.dataset)
    config = replace(
        config,
        model_provider=orch.model_provider or "orchestration",
        model_name=orch.model_name or "playoff",
        model_harness="orchestrate-playoff",
        target_mode=orch.target_mode,
    )
    bootstrap_track(
        config,
        default_max_cycles=max(1, challenger_count),
        enable_foundation_models=enable_foundation_models,
    )
    return config


def _load_consolidation_config(orch: Orchestration) -> ProjectConfig:
    config = load_config(
        track_id=str(orch.consolidation.track), run_id=str(orch.consolidation.run_id)
    )
    return replace(config, target_mode=orch.target_mode)


def _compare_replayed_experiment(
    config: ProjectConfig, challenger_id: str
) -> dict[str, Path]:
    from autoresearch.comparison_runner import compare_experiments
    from autoresearch.experiment_registry.registry import get_official_champion

    champion = get_official_champion(config.registry_path)
    if champion is None:
        raise ValueError("Consolidation run has no official champion")
    return compare_experiments(config, str(champion["champion_id"]), challenger_id)


def _record_playoff_decision(
    config: ProjectConfig,
    comparison_id: str,
    *,
    decision: str,
    rationale: str,
    reason_code: str,
    decided_by: str = "llm",
) -> dict[str, Any]:
    from autoresearch.comparison_runner import record_decision

    return record_decision(
        config,
        comparison_id,
        decision=decision,
        rationale=rationale,
        reason_code=reason_code,
        interpretation=f"Mechanical playoff decision: {rationale}",
        next_step="Continue the ascending finalist gauntlet.",
        decided_by=decided_by,
    )


def _auto_decide_pairing(config: ProjectConfig, pairing: dict[str, Any]) -> None:
    passed = bool((pairing.get("gate_evidence") or {}).get("all_standard_gates_pass"))
    decision = "promote" if passed else "reject"
    rationale = (
        "Auto-promoted because every standard comparison gate and hard guardrail passed."
        if passed
        else (
            "Auto-rejected because one or more standard comparison gates or hard "
            "guardrails failed."
        )
    )
    _record_playoff_decision(
        config,
        str(pairing["comparison_id"]),
        decision=decision,
        rationale=rationale,
        reason_code="clear_win" if passed else "inferior",
        decided_by="auto_playoff",
    )


def _sync_pairing_decisions(
    config: ProjectConfig, state: dict[str, Any]
) -> dict[str, Any] | None:
    from autoresearch.experiment_registry.registry import list_comparisons

    comparisons = {
        row["comparison_id"]: row for row in list_comparisons(config.registry_path)
    }
    pending: dict[str, Any] | None = None
    for pairing in state.get("pairings") or ():
        row = comparisons.get(pairing.get("comparison_id")) or {}
        decision = row.get("decision")
        if decision:
            pairing["decision"] = {
                "decision": decision,
                "rationale": row.get("decision_rationale"),
                "reason_code": row.get("decision_reason_code"),
                "decided_by": row.get("decided_by"),
                "decided_at": row.get("decided_at"),
            }
        elif pending is None:
            pending = pairing
    return pending


def _gate_evidence(report: dict[str, Any]) -> dict[str, Any]:
    advisory = report.get("advisory_promotion_decision") or {}
    checks = advisory.get("checks") or {}
    guardrail = report.get("guardrail_result") or {}
    all_pass = bool(
        checks
        and all(value is True for value in checks.values())
        and advisory.get("decision") == "promote"
        and guardrail.get("passed") is True
    )
    return {
        "standard_checks": checks,
        "advisory_decision": advisory.get("decision"),
        "advisory_rationale": advisory.get("rationale"),
        "guardrail": guardrail,
        "all_standard_gates_pass": all_pass,
    }


def _complete_playoff(
    orch: Orchestration, config: ProjectConfig, state: dict[str, Any]
) -> None:
    from autoresearch.experiment_registry.registry import get_official_champion

    champion = get_official_champion(config.registry_path)
    if champion is None:
        raise ValueError("Consolidation completed without an official champion")
    champion_id = str(champion["champion_id"])
    lineage = read_json(_lineage_path(config))
    source = next(
        (
            item.get("source")
            for item in lineage.get("replays") or ()
            if item.get("destination_experiment_id") == champion_id
        ),
        None,
    )
    state["status"] = "completed"
    state["final_champion_lineage"] = {
        "consolidation_experiment_id": champion_id,
        "source": source,
    }
    _write_playoff_reports(orch.orchestration_id, state)
    with manifest_lock(orch.orchestration_id):
        latest = load_orchestration(orch.orchestration_id)
        latest = replace(latest, status="completed")
        save_orchestration(latest)


def _append_lineage(config: ProjectConfig, replay: dict[str, Any]) -> None:
    path = _lineage_path(config)
    payload = read_json(path) if path.exists() else {"replays": []}
    replays = payload.setdefault("replays", [])
    destination_id = replay.get("destination_experiment_id")
    if not any(item.get("destination_experiment_id") == destination_id for item in replays):
        replays.append(replay)
    write_json(path, payload)


def _lineage_path(config: ProjectConfig) -> Path:
    return config.artifacts_dir / "orchestration_replay" / "lineage.json"


def _validate_replay_compatibility(source: ProjectConfig, destination: ProjectConfig) -> None:
    if source.dataset_name != destination.dataset_name:
        raise ValueError(
            f"Cannot replay across datasets: source={source.dataset_name}, "
            f"destination={destination.dataset_name}"
        )


def _validate_fixed_preprocessing(
    snapshot: dict[str, Any], source_exp: dict[str, Any], destination: ProjectConfig
) -> None:
    """Refuse a replay whose recorded fixed claim-cap policy differs."""

    preprocessing = snapshot.get("effective_preprocessing") or source_exp.get(
        "preprocessing"
    )
    if not isinstance(preprocessing, dict):
        return
    expected_enabled = getattr(destination, "claim_capping_enabled", None)
    recorded_enabled = preprocessing.get("claim_capping_enabled")
    if (
        expected_enabled is not None
        and recorded_enabled is not None
        and bool(recorded_enabled) != bool(expected_enabled)
    ):
        raise ValueError(
            "Cannot replay an experiment with a different fixed claim-capping policy"
        )
    expected_threshold = getattr(destination, "claim_cap_threshold", None)
    recorded_threshold = preprocessing.get("claim_cap_threshold")
    if (
        bool(expected_enabled)
        and expected_threshold is not None
        and recorded_threshold is not None
        and float(recorded_threshold) != float(expected_threshold)
    ):
        raise ValueError(
            "Cannot replay an experiment with a different fixed claim-cap threshold"
        )


def _source_script_path(
    snapshot: dict[str, Any], artifacts: list[dict[str, Any]]
) -> Path | None:
    artifact = next(
        (
            Path(str(item["path"]))
            for item in artifacts
            if item.get("artifact_type") == "model_script"
        ),
        None,
    )
    if artifact is not None and artifact.exists():
        return artifact
    raw = snapshot.get("model_script_path")
    return Path(str(raw)) if raw else None


def _delegation_report_file(orch: Orchestration, delegation: Any) -> Path:
    if delegation.report_path:
        path = Path(str(delegation.report_path))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            return path
    return delegation_report_path(orch.orchestration_id, delegation.delegation_id)


def _write_playoff_reports(orchestration_id: str, state: dict[str, Any]) -> None:
    directory = playoff_dir(orchestration_id)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / PLAYOFF_JSON, state)
    (directory / PLAYOFF_MARKDOWN).write_text(_render_markdown(state), encoding="utf-8")


def _render_markdown(state: dict[str, Any]) -> str:
    lines = [
        "# Orchestration Playoff Report",
        "",
        f"Campaign: `{state['orchestration_id']}`",
        f"Status: **{state['status']}**",
        f"Decision mode: `{state['decision_mode']}`",
        "",
        "## Finalist order (ascending search-validation gini_weighted)",
        "",
        "| Order | Delegation | Gini | Model | Source experiment |",
        "|---:|---|---:|---|---|",
    ]
    for index, finalist in enumerate(state.get("finalists") or (), start=1):
        source = finalist["source"]
        lines.append(
            f"| {index} | `{finalist['delegation_id']}` | {float(finalist['gini_weighted']):.6f} "
            f"| `{finalist['model_family']}` | `{source['track']}/{source['run_id']}/"
            f"{source['experiment_id']}` |"
        )
    lines.extend(["", "## Exclusions", ""])
    exclusions = state.get("exclusions") or ()
    if exclusions:
        lines.extend(
            f"- `{item['delegation_id']}`: {item['reason']}" for item in exclusions
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Pairings", ""])
    pairings = state.get("pairings") or ()
    if not pairings:
        lines.append("No challenger pairings recorded yet.")
    for pairing in pairings:
        evidence = pairing.get("gate_evidence") or {}
        decision = pairing.get("decision") or {}
        lines.extend(
            [
                f"### {pairing['order']}. {pairing['finalist']['delegation_id']}",
                "",
                f"Comparison: `{pairing['comparison_id']}`",
                f"Decision: `{decision.get('decision') or 'pending_llm'}`",
                f"Decision rationale: {decision.get('rationale') or 'Pending orchestrator verdict.'}",
                f"All standard gates pass: `{evidence.get('all_standard_gates_pass', False)}`",
                "",
                "| Gate | Result |",
                "|---|---|",
            ]
        )
        for name, passed in (evidence.get("standard_checks") or {}).items():
            lines.append(f"| `{name}` | {'pass' if passed else 'FAIL'} |")
        guardrail = evidence.get("guardrail") or {}
        lines.append(f"| `hard_guardrails` | {'pass' if guardrail.get('passed') else 'FAIL'} |")
        lines.append("")
    lines.extend(["## Final champion lineage", ""])
    lineage = state.get("final_champion_lineage")
    if lineage:
        source = lineage.get("source") or {}
        lines.append(f"Consolidation experiment: `{lineage['consolidation_experiment_id']}`")
        lines.append(
            f"Origin: delegation `{source.get('delegation_id')}`, "
            f"`{source.get('track')}/{source.get('run_id')}/{source.get('experiment_id')}`"
        )
    else:
        lines.append("Pending.")
    return "\n".join(lines) + "\n"


def _replay_experiment_name(label: str, source_exp: dict[str, Any]) -> str:
    source_name = str(source_exp.get("experiment_name") or "finalist")
    return f"orchestration_{_safe_label(label)}_{_safe_label(source_name)}"[:120]


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:100] or "replay"


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _to_toml(data: dict[str, Any]) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [clean(item) for item in value if item is not None]
        return value

    return tomli_w.dumps(clean(data))


def _stored_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
