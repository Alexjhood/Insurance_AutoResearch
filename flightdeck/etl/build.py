"""Flight Deck ETL — build snapshots from ``artifacts/orchestrations/``.

CLI (SPEC §2): ``python -m flightdeck.etl [--all | --orchestration <id>] [--force]``.

Emits, under ``flightdeck/snapshots/``:
- ``index.json`` — one headline entry per built orchestration (incremental).
- ``<id>/snapshot.json`` — everything the UI needs except lazy telemetry + files.
- ``<id>/telemetry_<dNN>.json`` — per-delegation event lists (Flight Recorder).
- ``<id>/files/**`` — verbatim prompts/briefs/logs/reports.

Idempotent + incremental (skip when sources unchanged unless ``--force``) and
atomic (write to ``<id>.tmp/``, rename). Defensive throughout: missing sources
degrade to nulls + ``snapshot.build.warnings[]`` (DATA.md general rules).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import (
    BuildInfo,
    ChampionEvent,
    Experiment,
    IndexEntry,
    Lift,
    Snapshot,
    SnapshotIndex,
    SubagentSummary,
    TelemetryByDelegation,
    TelemetrySummary,
    TelemetryTotals,
    TokenTotals,
    ToolMixEntry,
    UsageByModel,
    OrchestratorIdentity,
    SNAPSHOT_SCHEMA_VERSION,
    to_jsonable,
)
from .readers import files as files_reader
from .readers import orchestration as orch_reader
from .readers import playoff as playoff_reader
from .readers import registry as registry_reader
from .readers import solo as solo_reader
from .readers import telemetry as telemetry_reader
from .cost import compute_run_cost, load_pricing_index
from .util import Warnings, load_json, parse_ts, ts_key


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def find_repo_root(start: Optional[Path] = None) -> Path:
    start = (start or Path(__file__)).resolve()
    for parent in [start, *start.parents]:
        if (parent / "artifacts" / "orchestrations").is_dir():
            return parent
        if (parent / ".git").exists() and (parent / "flightdeck").is_dir():
            return parent
    # Fall back to the flightdeck package's grandparent (repo root).
    return Path(__file__).resolve().parents[2]


def snapshots_dir(repo_root: Path) -> Path:
    return repo_root / "flightdeck" / "snapshots"


def orchestrations_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "orchestrations"


def tracks_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "tracks"


def backends_config_path(repo_root: Path) -> Path:
    return repo_root / "configs" / "orchestration" / "backends.toml"


# --------------------------------------------------------------------------- #
# Source mtime (for incremental skip)
# --------------------------------------------------------------------------- #
# Volatile sidecars whose mtime changes merely from opening a DB read-only
# (or is OS/noise) — excluded so incremental skip stays stable.
_MTIME_IGNORE_SUFFIXES = ("-wal", "-shm", ".DS_Store")


def _max_source_mtime(orch_dir: Path, consolidation_registry: Optional[Path]) -> float:
    latest = 0.0
    for path in orch_dir.rglob("*"):
        if path.is_file():
            if path.name.endswith(_MTIME_IGNORE_SUFFIXES):
                continue
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
    if consolidation_registry is not None and consolidation_registry.exists():
        try:
            latest = max(latest, consolidation_registry.stat().st_mtime)
        except OSError:
            pass
    return latest


def _iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Experiment post-processing (seq + lift)
# --------------------------------------------------------------------------- #
def _assign_seq(experiments: list[Experiment]) -> None:
    ordered = sorted(experiments, key=lambda e: ts_key(e.created_at))
    for i, exp in enumerate(ordered):
        exp.seq = i


def _fill_lift(experiments: list[Experiment], baseline_gini: Optional[float]) -> None:
    baseline_ids = {e.experiment_id for e in experiments if e.is_baseline}
    for exp in experiments:
        # Seeds/baselines are excluded from lift aggregates (quirk 2).
        if exp.is_seed or exp.is_baseline:
            exp.lift = Lift(None, None, None)
            continue
        gini = exp.metrics.gini_weighted
        vs_baseline = (
            gini - baseline_gini if gini is not None and baseline_gini is not None else None
        )
        vs_then: Optional[float] = None
        kind: Optional[str] = None
        if exp.comparison is not None:
            vs_then = exp.comparison.cv_mean_lift
            kind = "baseline_relative" if exp.comparison.champion_id in baseline_ids else "incremental"
        elif exp.screening is not None:
            vs_then = exp.screening.lift
            kind = "incremental"
        exp.lift = Lift(vs_then_champion=vs_then, vs_baseline=vs_baseline, kind=kind)


# --------------------------------------------------------------------------- #
# Champion timeline
# --------------------------------------------------------------------------- #
def _build_champion_timeline(
    runs: list[tuple[Optional[str], str, Path]],  # (delegation_id, scope, registry_path)
    gini_lookup: dict[str, Optional[float]],
    seed_ids: set[str],
    warnings: Warnings,
) -> list[ChampionEvent]:
    events: list[ChampionEvent] = []
    seen_initialised = False
    for delegation_id, scope, registry_path in runs:
        for row in registry_reader.read_champion_history(registry_path, warnings):
            action = row.get("action") or ""
            if action == "retained":
                continue  # champion unchanged — not a champion *change*
            new_id = row.get("new_champion_id") or ""
            if action == "initialised":
                if seen_initialised:
                    continue
                seen_initialised = True
            events.append(
                ChampionEvent(
                    at=row.get("created_at") or "",
                    delegation_id=delegation_id,
                    action=action,
                    previous_champion_id=row.get("previous_champion_id"),
                    new_champion_id=new_id,
                    new_champion_gini=gini_lookup.get(new_id),
                    comparison_id=row.get("comparison_id"),
                    scope=scope,
                    is_seed_transfer=(action == "seeded" or new_id in seed_ids),
                )
            )
    events.sort(key=lambda e: ts_key(e.at))
    return events


# --------------------------------------------------------------------------- #
# Telemetry summary
# --------------------------------------------------------------------------- #
def _add_tokens(acc: TokenTotals, other: TokenTotals) -> None:
    acc.input += other.input
    acc.cached_input += other.cached_input
    acc.output += other.output
    acc.reasoning += other.reasoning


def _build_telemetry_summary(
    delegations, per_deleg_cost, campaign_report,
    subagents: Optional[SubagentSummary] = None,
) -> TelemetrySummary:
    totals = TelemetryTotals()
    by_delegation: list[TelemetryByDelegation] = []
    tool_mix: list[ToolMixEntry] = []
    for deleg in delegations:
        cost = deleg.cost
        totals.input += cost.tokens.input
        totals.cached_input += cost.tokens.cached_input
        totals.output += cost.tokens.output
        totals.reasoning += cost.tokens.reasoning
        totals.model_calls += cost.model_calls
        totals.tool_calls += cost.tool_calls
        totals.tool_failures += cost.tool_failures
        by_delegation.append(
            TelemetryByDelegation(
                delegation_id=deleg.delegation_id,
                tokens=cost.tokens,
                model_calls=cost.model_calls,
                tool_calls=cost.tool_calls,
                tool_failures=cost.tool_failures,
                cache_hit_rate=cost.cache_hit_rate,
            )
        )
        for tool, calls, failures, duration_ms in per_deleg_cost.get(
            deleg.delegation_id, {}
        ).get("tool_mix", []):
            tool_mix.append(
                ToolMixEntry(
                    delegation_id=deleg.delegation_id,
                    tool=tool,
                    calls=calls,
                    failures=failures,
                    total_duration_ms=duration_ms,
                )
            )
    cache_rate = totals.cached_input / totals.input if totals.input else None
    cost_doc = (campaign_report.get("framework_computed") or {}).get("cost") or {}
    usage_by_model: list[UsageByModel] = []
    for key, row in (cost_doc.get("usage_by_model") or {}).items():
        if not isinstance(row, dict):
            continue
        unmeasured = bool(row.get("unmeasured"))
        tokens = None if unmeasured else TokenTotals(
            input=int(row.get("input_tokens") or 0),
            cached_input=int(row.get("cached_tokens") or 0),
            output=int(row.get("output_tokens") or 0),
            reasoning=int(row.get("reasoning_tokens") or 0),
        )
        usage_by_model.append(UsageByModel(
            key=key,
            provider=row.get("provider") or "",
            model=row.get("model") or key,
            effort=row.get("effort"),
            role=row.get("role") or "delegation",
            tokens=tokens,
            unmeasured=unmeasured,
            cost_usd=row.get("cost_usd"),
            cost_estimated=bool(row.get("cost_estimated")),
        ))
    return TelemetrySummary(
        totals=totals,
        cache_hit_rate=cache_rate,
        by_delegation=by_delegation,
        tool_mix=tool_mix,
        usage_by_model=usage_by_model,
        cost_usd=cost_doc.get("cost_usd"),
        cost_estimated=bool(cost_doc.get("cost_estimated")),
        subagents=subagents or SubagentSummary(),
    )


# --------------------------------------------------------------------------- #
# Index entry
# --------------------------------------------------------------------------- #
def _wall_clock_minutes(created_at: str, ended_at: Optional[str]) -> Optional[float]:
    start = parse_ts(created_at)
    end = parse_ts(ended_at)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 60.0


def _build_index_entry(
    orch_id, campaign, delegations, playoff, champion_timeline, telemetry_summary,
    baseline_gini, alias, notes,
) -> IndexEntry:
    ended_at = campaign.ended_at
    total_tokens = TokenTotals()
    for d in delegations:
        _add_tokens(total_tokens, d.cost.tokens)
    cache_rate = total_tokens.cached_input / total_tokens.input if total_tokens.input else None

    final_gini = None
    if playoff is not None and playoff.final is not None:
        fd = playoff.final.delegation_id
        for fin in playoff.finalists:
            if fin.delegation_id == fd:
                final_gini = fin.gini_weighted
                break
    if final_gini is None:
        champ_ginis = [d.champion.gini_weighted for d in delegations if d.champion is not None]
        final_gini = max(champ_ginis) if champ_ginis else None

    # takeover_count: delegations taken over OR with a takeover note, dedup by delegation.
    takeover_delegations = {d.delegation_id for d in delegations if d.taken_over}
    for n in notes:
        if n.kind == "takeover" and n.delegation_id:
            takeover_delegations.add(n.delegation_id)

    spark = [
        e.new_champion_gini
        for e in champion_timeline
        if e.new_champion_gini is not None
    ]

    return IndexEntry(
        orch_id=orch_id,
        alias=alias,
        kind=campaign.kind,
        dataset=campaign.dataset,
        target_mode=campaign.target_mode,
        status=campaign.status,
        created_at=campaign.created_at,
        ended_at=ended_at,
        orchestrator_model=campaign.orchestrator_model,
        orchestrator=campaign.orchestrator,
        stale=False,
        backends=sorted({d.backend for d in delegations if d.backend}),
        n_delegations=len(delegations),
        cycles_committed=campaign.cycles_committed,
        cycles_used=min(campaign.cycles_committed, sum(d.budget.decided for d in delegations)),
        cycles_attempted=sum(d.budget.attempted for d in delegations),
        seed_evals=sum(1 for e in champion_timeline if e.is_seed_transfer),
        cycles_forfeited=sum(d.budget.forfeited for d in delegations),
        final_gini=final_gini,
        baseline_gini=baseline_gini,
        total_tokens=total_tokens,
        cache_hit_rate=cache_rate,
        wall_clock_minutes=_wall_clock_minutes(campaign.created_at, ended_at),
        distress_count=sum(len(d.distress.active) for d in delegations),
        takeover_count=len(takeover_delegations),
        champion_spark=spark,
        cost_usd=telemetry_summary.cost_usd,
        cost_estimated=telemetry_summary.cost_estimated,
    )


# --------------------------------------------------------------------------- #
# Core build
# --------------------------------------------------------------------------- #
def build_orchestration(
    orch_id: str,
    repo_root: Path,
    *,
    force: bool = False,
    existing_index_entry: Optional[dict] = None,
    aliases: Optional[dict] = None,
    log=print,
) -> tuple[Optional[IndexEntry], bool]:
    """Build one orchestration's snapshot. Returns (index_entry, skipped)."""
    orch_dir = orchestrations_dir(repo_root) / orch_id
    if not orch_dir.is_dir():
        raise FileNotFoundError(f"orchestration not found: {orch_dir}")

    out_dir = snapshots_dir(repo_root) / orch_id
    ledger_probe = load_json(orch_dir / "orchestration.json", Warnings()) or {}
    cons = ledger_probe.get("consolidation") or {}
    consolidation_registry = None
    if cons.get("run_id") and cons.get("track"):
        consolidation_registry = (
            repo_root / "artifacts" / "tracks" / cons["track"] / "runs"
            / cons["run_id"] / "registry.sqlite"
        )

    source_mtime = _max_source_mtime(orch_dir, consolidation_registry)
    snapshot_json = out_dir / "snapshot.json"
    if not force and snapshot_json.exists() and existing_index_entry is not None:
        existing = load_json(snapshot_json, Warnings())
        if isinstance(existing, dict):
            prior = parse_ts(existing.get("build", {}).get("source_mtime"))
            if prior is not None and source_mtime <= prior.timestamp() + 1e-6:
                log(f"  skip {orch_id} (unchanged)")
                return _index_entry_from_dict(existing_index_entry), True

    warnings = Warnings()
    tmp_dir = snapshots_dir(repo_root) / f"{orch_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_entry = _do_build(
            orch_id, orch_dir, tmp_dir, repo_root, consolidation_registry,
            source_mtime, aliases or {}, warnings, log,
        )
        # Atomic swap.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.replace(out_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return index_entry, False


def _do_build(
    orch_id, orch_dir, out_dir, repo_root, consolidation_registry,
    source_mtime, aliases, warnings, log,
) -> IndexEntry:
    ledger, campaign_report, notes_doc = orch_reader.load_sources(orch_dir, warnings)

    # 1. Files (needed for delegation.files + campaign md links).
    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    files_manifest = files_reader.copy_files(orch_dir, files_dir, repo_root, warnings)
    files_paths = {e.path for e in files_manifest}
    files_lookup = {
        "CAMPAIGN_REPORT.md": "CAMPAIGN_REPORT.md" if "CAMPAIGN_REPORT.md" in files_paths else None,
        "ORCHESTRATION_LOG.md": "ORCHESTRATION_LOG.md" if "ORCHESTRATION_LOG.md" in files_paths else None,
    }

    # 2. Delegations (+ telemetry cost, + per-delegation usage + event dumps).
    ledger_delegs = ledger.get("delegations") or []
    per_deleg_cost: dict[str, dict] = {}
    usage_maps: dict[str, dict] = {}
    delegations = []
    for ld in ledger_delegs:
        did = ld.get("delegation_id") or ""
        telemetry_path = orch_dir / "runs" / did / "telemetry.sqlite"
        cost = telemetry_reader.read_telemetry_cost(telemetry_path, warnings)
        per_deleg_cost[did] = cost
        usage_maps[did] = telemetry_reader.read_experiment_usage(
            telemetry_path, orch_dir / "runs" / did / "LLM_USAGE.md", warnings
        )
        delegations.append(
            orch_reader.build_delegation(orch_dir, ld, cost, files_paths, warnings)
        )
        # Emit lazy telemetry_<dNN>.json.
        dt = telemetry_reader.read_delegation_telemetry(telemetry_path, did, warnings)
        if dt is not None:
            (out_dir / f"telemetry_{did}.json").write_text(
                json.dumps(to_jsonable(dt), indent=1), encoding="utf-8"
            )

    ended_at = _max_ended_at(delegations)
    campaign = orch_reader.build_campaign(
        orch_id, orch_dir, ledger, campaign_report, ended_at, files_lookup, warnings
    )

    # 3. Experiments — delegation registries then consolidation, deduped.
    experiments: list[Experiment] = []
    seen_ids: set[str] = set()
    timeline_runs: list[tuple[Optional[str], str, Path]] = []
    for ld in ledger_delegs:
        did = ld.get("delegation_id") or ""
        registry_path = orch_dir / "runs" / did / "registry.sqlite"
        run_dir = orch_dir / "runs" / did
        timeline_runs.append((did, "delegation", registry_path))
        for exp in registry_reader.read_registry_experiments(
            registry_path, run_dir, did, warnings
        ):
            if exp.experiment_id in seen_ids:
                continue
            seen_ids.add(exp.experiment_id)
            exp.usage = usage_maps.get(did, {}).get(exp.name)
            experiments.append(exp)

    consolidation_usage: dict = {}
    if consolidation_registry is not None:
        cons_run_dir = consolidation_registry.parent
        if consolidation_registry.exists():
            consolidation_usage = telemetry_reader.read_experiment_usage(
                cons_run_dir / "telemetry.sqlite", cons_run_dir / "LLM_USAGE.md", warnings
            )
        else:
            warnings.add(f"consolidation registry unreadable: {consolidation_registry}")
        timeline_runs.append((None, "consolidation", consolidation_registry))
        for exp in registry_reader.read_registry_experiments(
            consolidation_registry, cons_run_dir, None, warnings
        ):
            if exp.experiment_id in seen_ids:
                continue
            seen_ids.add(exp.experiment_id)
            exp.usage = consolidation_usage.get(exp.name)
            experiments.append(exp)

    _assign_seq(experiments)
    baseline_gini = _baseline_gini(experiments)
    _fill_lift(experiments, baseline_gini)

    gini_lookup = {e.experiment_id: e.metrics.gini_weighted for e in experiments}
    seed_ids = {e.experiment_id for e in experiments if e.is_seed}
    champion_timeline = _build_champion_timeline(
        timeline_runs, gini_lookup, seed_ids, warnings
    )

    notes = orch_reader.build_notes(notes_doc)

    # 4. Playoff.
    consolidation_exp_ids = [
        e.experiment_id for e in experiments if e.delegation_id is None
    ]
    playoff = playoff_reader.read_playoff(
        orch_dir, consolidation_exp_ids,
        "playoff/playoff_report.md" if "playoff/playoff_report.md" in files_paths else None,
        warnings,
    )

    telemetry_summary = _build_telemetry_summary(delegations, per_deleg_cost, campaign_report)

    snapshot = Snapshot(
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        build=BuildInfo(
            built_at=_now_iso(),
            source_mtime=_iso(source_mtime),
            warnings=warnings.items,
        ),
        campaign=campaign,
        delegations=delegations,
        experiments=experiments,
        champion_timeline=champion_timeline,
        notes=notes,
        playoff=playoff,
        telemetry_summary=telemetry_summary,
        files=files_manifest,
    )
    (out_dir / "snapshot.json").write_text(
        json.dumps(to_jsonable(snapshot), indent=1), encoding="utf-8"
    )

    alias = (aliases or {}).get(orch_id)
    index_entry = _build_index_entry(
        orch_id, campaign, delegations, playoff, champion_timeline,
        telemetry_summary, baseline_gini, alias, notes,
    )
    log(f"  built {orch_id}: {len(delegations)} delegations, "
        f"{len(experiments)} experiments, {len(warnings.items)} warnings")
    return index_entry


# --------------------------------------------------------------------------- #
# Solo-run build (spec §7.1–7.2)
# --------------------------------------------------------------------------- #
_SOLO_COPY_FILES = ("RESEARCH_LOG.md", "LLM_USAGE.md", "run_manifest.json")


def discover_solo_runs(repo_root: Path) -> list[tuple[str, str, Path]]:
    """Return ``(track, run_id, run_dir)`` for every solo track run.

    A solo run is a ``artifacts/tracks/<track>/runs/<id>/`` whose
    ``run_manifest.json`` has no ``orchestration_id`` (orchestration children and
    playoff-consolidation runs carry one and belong to a campaign).
    """
    root = tracks_dir(repo_root)
    if not root.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for track_dir in sorted(root.iterdir()):
        runs = track_dir / "runs"
        if not track_dir.is_dir() or not runs.is_dir():
            continue
        for run_dir in sorted(runs.iterdir()):
            # Skip old→new rename aliases and delegation compatibility links;
            # a symlinked run belongs to whatever it points at, not here.
            if run_dir.is_symlink():
                continue
            manifest_path = run_dir / "run_manifest.json"
            if not run_dir.is_dir() or not manifest_path.exists():
                continue
            manifest = load_json(manifest_path, Warnings()) or {}
            if not isinstance(manifest, dict):
                continue
            # Orchestration children and playoff-consolidation runs belong to a
            # campaign — they carry an ``orchestration_id`` or a ``delegation_id``
            # (a delegation manifest may omit the id but never the delegation),
            # or are flagged ``consolidation``.
            if (
                manifest.get("orchestration_id")
                or manifest.get("delegation_id")
                or manifest.get("consolidation")
            ):
                continue
            out.append((track_dir.name, run_dir.name, run_dir))
    return out


def _copy_solo_files(run_dir: Path, out_files_dir: Path, repo_root: Path,
                     warnings: Warnings) -> list:
    from .schema import FileEntry
    manifest: list = []
    for name in _SOLO_COPY_FILES:
        src = run_dir / name
        if not src.is_file():
            continue
        dst = out_files_dir / name
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            try:
                source_rel = src.relative_to(repo_root).as_posix()
            except ValueError:
                source_rel = str(src)
            kind = "markdown" if name.endswith(".md") else "json"
            manifest.append(FileEntry(path=name, kind=kind, bytes=dst.stat().st_size,
                                      truncated=False, source=source_rel))
        except OSError as exc:
            warnings.add(f"failed to copy {name}: {exc}")
    manifest.sort(key=lambda e: e.path)
    return manifest


def build_solo_run(
    track: str,
    run_id: str,
    repo_root: Path,
    *,
    force: bool = False,
    existing_index_entry: Optional[dict] = None,
    aliases: Optional[dict] = None,
    pricing_index: Optional[dict] = None,
    log=print,
) -> tuple[Optional[IndexEntry], bool]:
    """Build one solo run's snapshot. Returns (index_entry, skipped)."""
    run_dir = tracks_dir(repo_root) / track / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"solo run not found: {run_dir}")

    out_dir = snapshots_dir(repo_root) / run_id
    source_mtime = _max_source_mtime(run_dir, None)
    snapshot_json = out_dir / "snapshot.json"
    if not force and snapshot_json.exists() and existing_index_entry is not None:
        existing = load_json(snapshot_json, Warnings())
        if isinstance(existing, dict):
            prior = parse_ts(existing.get("build", {}).get("source_mtime"))
            if prior is not None and source_mtime <= prior.timestamp() + 1e-6:
                log(f"  skip {run_id} (unchanged, solo)")
                return _index_entry_from_dict(existing_index_entry), True

    warnings = Warnings()
    tmp_dir = snapshots_dir(repo_root) / f"{run_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        index_entry = _do_build_solo(
            track, run_id, run_dir, tmp_dir, repo_root, source_mtime,
            aliases or {}, pricing_index, warnings, log,
        )
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.replace(out_dir)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return index_entry, False


def _do_build_solo(
    track, run_id, run_dir, out_dir, repo_root, source_mtime,
    aliases, pricing_index, warnings, log,
) -> IndexEntry:
    manifest = solo_reader.load_manifest(run_dir, warnings)

    files_dir = out_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    files_manifest = _copy_solo_files(run_dir, files_dir, repo_root, warnings)

    registry_path = run_dir / "registry.sqlite"
    telemetry_path = run_dir / "telemetry.sqlite"

    experiments = registry_reader.read_registry_experiments(
        registry_path, run_dir, None, warnings
    )
    _assign_seq(experiments)
    baseline_gini = _baseline_gini(experiments)
    _fill_lift(experiments, baseline_gini)

    gini_lookup = {e.experiment_id: e.metrics.gini_weighted for e in experiments}
    seed_ids = {e.experiment_id for e in experiments if e.is_seed}
    champion_timeline = _build_champion_timeline(
        [(None, "principal", registry_path)], gini_lookup, seed_ids, warnings
    )
    champion_new_id = champion_timeline[-1].new_champion_id if champion_timeline else None
    champion = solo_reader.solo_champion(experiments, champion_new_id)

    if pricing_index is None:
        pricing_index = load_pricing_index(backends_config_path(repo_root))
    run_cost = compute_run_cost(telemetry_path, pricing_index=pricing_index)
    telemetry_cost = telemetry_reader.read_telemetry_cost(telemetry_path, warnings)
    telemetry_summary = solo_reader.build_solo_telemetry_summary(run_cost, telemetry_cost)

    # Per-experiment usage checkpoints (same source as delegations).
    usage_map = telemetry_reader.read_experiment_usage(
        telemetry_path, run_dir / "LLM_USAGE.md", warnings
    )
    for exp in experiments:
        exp.usage = usage_map.get(exp.name)

    dt = telemetry_reader.read_delegation_telemetry(telemetry_path, "principal", warnings)
    if dt is not None:
        (out_dir / "telemetry_principal.json").write_text(
            json.dumps(to_jsonable(dt), indent=1), encoding="utf-8"
        )

    ended_at = max((e.created_at for e in experiments if e.created_at), default=None,
                   key=lambda a: ts_key(a) if a else "")
    campaign = solo_reader.build_solo_campaign(run_id, manifest, ended_at, warnings)

    try:
        run_path = run_dir.relative_to(repo_root).as_posix()
    except ValueError:
        run_path = str(run_dir)
    principal_run = solo_reader.build_principal_run(
        run_id, track, run_path, manifest, run_cost, champion
    )

    snapshot = Snapshot(
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
        build=BuildInfo(
            built_at=_now_iso(),
            source_mtime=_iso(source_mtime),
            warnings=warnings.items,
        ),
        campaign=campaign,
        delegations=[],
        experiments=experiments,
        champion_timeline=champion_timeline,
        notes=[],
        playoff=None,
        telemetry_summary=telemetry_summary,
        files=files_manifest,
        principal_run=principal_run,
    )
    (out_dir / "snapshot.json").write_text(
        json.dumps(to_jsonable(snapshot), indent=1), encoding="utf-8"
    )

    alias = (aliases or {}).get(run_id)
    index_entry = _build_solo_index_entry(
        run_id, campaign, experiments, champion, champion_timeline,
        telemetry_summary, baseline_gini, alias,
    )
    log(f"  built {run_id} (solo): {len(experiments)} experiments, "
        f"{len(warnings.items)} warnings")
    return index_entry


def _build_solo_index_entry(
    run_id, campaign, experiments, champion, champion_timeline,
    telemetry_summary, baseline_gini, alias,
) -> IndexEntry:
    tt = telemetry_summary.totals
    total_tokens = TokenTotals(
        input=tt.input, cached_input=tt.cached_input,
        output=tt.output, reasoning=tt.reasoning,
    )
    cache_rate = total_tokens.cached_input / total_tokens.input if total_tokens.input else None

    non_baseline = [e for e in experiments if not e.is_baseline and not e.is_seed]
    decided = [e for e in non_baseline
               if e.comparison is not None and e.comparison.decision is not None]
    spark = [e.new_champion_gini for e in champion_timeline
             if e.new_champion_gini is not None]

    return IndexEntry(
        orch_id=run_id,
        alias=alias,
        kind="solo",
        dataset=campaign.dataset,
        target_mode=campaign.target_mode,
        status=campaign.status,
        created_at=campaign.created_at,
        ended_at=campaign.ended_at,
        orchestrator_model=campaign.orchestrator_model,
        orchestrator=campaign.orchestrator,
        stale=False,
        backends=[],
        n_delegations=0,
        cycles_committed=campaign.cycles_committed,
        cycles_used=len(decided),
        cycles_attempted=len(non_baseline),
        seed_evals=sum(1 for e in champion_timeline if e.is_seed_transfer),
        cycles_forfeited=0,
        final_gini=champion.gini_weighted if champion is not None else None,
        baseline_gini=baseline_gini,
        total_tokens=total_tokens,
        cache_hit_rate=cache_rate,
        wall_clock_minutes=_wall_clock_minutes(campaign.created_at, campaign.ended_at),
        distress_count=0,
        takeover_count=0,
        champion_spark=spark,
        cost_usd=telemetry_summary.cost_usd,
        cost_estimated=telemetry_summary.cost_estimated,
    )


def _max_ended_at(delegations) -> Optional[str]:
    candidates = [d.ended_at for d in delegations if d.ended_at]
    if not candidates:
        return None
    return max(candidates, key=ts_key)


def _baseline_gini(experiments: list[Experiment]) -> Optional[float]:
    baselines = [e for e in experiments if e.is_baseline]
    if not baselines:
        return None
    earliest = min(baselines, key=lambda e: ts_key(e.created_at))
    return earliest.metrics.gini_weighted


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Index assembly
# --------------------------------------------------------------------------- #
def _load_aliases(repo_root: Path) -> dict:
    path = snapshots_dir(repo_root) / "aliases.json"
    data = load_json(path, Warnings())
    return data if isinstance(data, dict) else {}


def _load_index(repo_root: Path) -> dict:
    path = snapshots_dir(repo_root) / "index.json"
    data = load_json(path, Warnings())
    return data if isinstance(data, dict) else {}


def _index_entry_from_dict(d: dict) -> IndexEntry:
    tok = d.get("total_tokens") or {}
    orch = d.get("orchestrator") or {}
    legacy_model = d.get("orchestrator_model", "")
    return IndexEntry(
        orch_id=d.get("orch_id", ""),
        alias=d.get("alias"),
        kind=d.get("kind") or "orchestrated",
        dataset=d.get("dataset", ""),
        target_mode=d.get("target_mode", ""),
        status=d.get("status", ""),
        created_at=d.get("created_at", ""),
        ended_at=d.get("ended_at"),
        orchestrator_model=d.get("orchestrator_model", ""),
        orchestrator=OrchestratorIdentity(
            provider=orch.get("provider") or (legacy_model.split("/", 1)[0] if "/" in legacy_model else ""),
            model=orch.get("model") or (legacy_model.split("/", 1)[-1]),
            effort=orch.get("effort"),
            source=orch.get("source"),
            recorded_at=orch.get("recorded_at"),
            revision=orch.get("revision", 0),
        ),
        stale=bool(d.get("stale")),
        backends=list(d.get("backends") or []),
        n_delegations=d.get("n_delegations", 0),
        cycles_committed=d.get("cycles_committed", 0),
        cycles_used=d.get("cycles_used", 0),
        cycles_attempted=d.get("cycles_attempted", d.get("cycles_used", 0)),
        seed_evals=d.get("seed_evals", 0),
        cycles_forfeited=d.get("cycles_forfeited", 0),
        final_gini=d.get("final_gini"),
        baseline_gini=d.get("baseline_gini"),
        total_tokens=TokenTotals(
            input=tok.get("input", 0), cached_input=tok.get("cached_input", 0),
            output=tok.get("output", 0), reasoning=tok.get("reasoning", 0),
        ),
        cache_hit_rate=d.get("cache_hit_rate"),
        wall_clock_minutes=d.get("wall_clock_minutes"),
        distress_count=d.get("distress_count", 0),
        takeover_count=d.get("takeover_count", 0),
        champion_spark=list(d.get("champion_spark") or []),
        cost_usd=d.get("cost_usd"),
        cost_estimated=bool(d.get("cost_estimated")),
    )


def _write_index(repo_root: Path, entries: dict[str, IndexEntry]) -> None:
    ordered = sorted(entries.values(), key=lambda e: e.created_at, reverse=True)
    index = SnapshotIndex(
        orchestrations=ordered,
        built_at=_now_iso(),
        snapshot_schema_version=SNAPSHOT_SCHEMA_VERSION,
    )
    sd = snapshots_dir(repo_root)
    sd.mkdir(parents=True, exist_ok=True)
    tmp = sd / "index.json.tmp"
    tmp.write_text(json.dumps(to_jsonable(index), indent=1), encoding="utf-8")
    tmp.replace(sd / "index.json")


def build(
    repo_root: Path,
    orch_ids: list[str],
    *,
    solo_runs: Optional[list[tuple[str, str]]] = None,
    force: bool = False,
    log=print,
) -> SnapshotIndex:
    aliases = _load_aliases(repo_root)
    existing_index = _load_index(repo_root)
    existing_entries = {
        e.get("orch_id"): e for e in existing_index.get("orchestrations", [])
    }
    entries: dict[str, IndexEntry] = {
        oid: _index_entry_from_dict(e) for oid, e in existing_entries.items() if oid
    }
    for orch_id in orch_ids:
        log(f"building {orch_id} …")
        entry, _skipped = build_orchestration(
            orch_id, repo_root, force=force,
            existing_index_entry=existing_entries.get(orch_id),
            aliases=aliases, log=log,
        )
        if entry is not None:
            entries[orch_id] = entry

    # Solo (non-orchestration) track runs share the unified league (spec §7).
    pricing_index = None
    for track, run_id in solo_runs or []:
        log(f"building {run_id} (solo) …")
        if pricing_index is None:
            pricing_index = load_pricing_index(backends_config_path(repo_root))
        entry, _skipped = build_solo_run(
            track, run_id, repo_root, force=force,
            existing_index_entry=existing_entries.get(run_id),
            aliases=aliases, pricing_index=pricing_index, log=log,
        )
        if entry is not None:
            entries[run_id] = entry

    _write_index(repo_root, entries)
    return SnapshotIndex(orchestrations=list(entries.values()), built_at=_now_iso())


def discover_orchestrations(repo_root: Path) -> list[str]:
    root = orchestrations_dir(repo_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        # Skip old→new rename aliases (symlinks) so a migrated campaign is not
        # ingested twice under both its old and new id.
        if p.is_dir() and not p.is_symlink() and (p / "orchestration.json").exists()
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m flightdeck.etl",
        description="Build Flight Deck snapshots from artifacts/orchestrations/.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="build every orchestration and solo run present")
    group.add_argument("--orchestration", metavar="ID", help="build a single orchestration id")
    group.add_argument("--solo", metavar="TRACK/ID",
                       help="build a single solo run, e.g. codex/20260713T073805Z")
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    parser.add_argument("--repo-root", type=Path, default=None, help="override repo root")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()

    orch_ids: list[str] = []
    solo_runs: list[tuple[str, str]] = []
    if args.all:
        orch_ids = discover_orchestrations(repo_root)
        solo_runs = [(track, run_id) for track, run_id, _ in discover_solo_runs(repo_root)]
        if not orch_ids and not solo_runs:
            print("no orchestrations or solo runs found", file=sys.stderr)
            return 0
    elif args.orchestration:
        orch_ids = [args.orchestration]
    elif args.solo:
        if "/" not in args.solo:
            parser.error("--solo expects TRACK/ID, e.g. codex/20260713T073805Z")
            return 2
        track, run_id = args.solo.split("/", 1)
        solo_runs = [(track, run_id)]
    else:
        parser.error("specify --all, --orchestration <id>, or --solo <track/id>")
        return 2

    build(repo_root, orch_ids, solo_runs=solo_runs, force=args.force)
    print(f"done — snapshots at {snapshots_dir(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
