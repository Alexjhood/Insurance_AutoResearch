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
    TelemetryByDelegation,
    TelemetrySummary,
    TelemetryTotals,
    TokenTotals,
    ToolMixEntry,
    SNAPSHOT_SCHEMA_VERSION,
    to_jsonable,
)
from .readers import files as files_reader
from .readers import orchestration as orch_reader
from .readers import playoff as playoff_reader
from .readers import registry as registry_reader
from .readers import telemetry as telemetry_reader
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


def _build_telemetry_summary(delegations, per_deleg_cost) -> TelemetrySummary:
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
    return TelemetrySummary(
        totals=totals,
        cache_hit_rate=cache_rate,
        by_delegation=by_delegation,
        tool_mix=tool_mix,
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
        dataset=campaign.dataset,
        target_mode=campaign.target_mode,
        status=campaign.status,
        created_at=campaign.created_at,
        ended_at=ended_at,
        orchestrator_model=campaign.orchestrator_model,
        backends=sorted({d.backend for d in delegations if d.backend}),
        n_delegations=len(delegations),
        cycles_committed=campaign.cycles_committed,
        cycles_used=sum(d.budget.used for d in delegations),
        cycles_forfeited=sum(d.budget.forfeited for d in delegations),
        final_gini=final_gini,
        baseline_gini=baseline_gini,
        total_tokens=total_tokens,
        cache_hit_rate=cache_rate,
        wall_clock_minutes=_wall_clock_minutes(campaign.created_at, ended_at),
        distress_count=sum(len(d.distress.active) for d in delegations),
        takeover_count=len(takeover_delegations),
        champion_spark=spark,
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

    telemetry_summary = _build_telemetry_summary(delegations, per_deleg_cost)

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
    return IndexEntry(
        orch_id=d.get("orch_id", ""),
        alias=d.get("alias"),
        dataset=d.get("dataset", ""),
        target_mode=d.get("target_mode", ""),
        status=d.get("status", ""),
        created_at=d.get("created_at", ""),
        ended_at=d.get("ended_at"),
        orchestrator_model=d.get("orchestrator_model", ""),
        backends=list(d.get("backends") or []),
        n_delegations=d.get("n_delegations", 0),
        cycles_committed=d.get("cycles_committed", 0),
        cycles_used=d.get("cycles_used", 0),
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
    _write_index(repo_root, entries)
    return SnapshotIndex(orchestrations=list(entries.values()), built_at=_now_iso())


def discover_orchestrations(repo_root: Path) -> list[str]:
    root = orchestrations_dir(repo_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "orchestration.json").exists()
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
    group.add_argument("--all", action="store_true", help="build every orchestration present")
    group.add_argument("--orchestration", metavar="ID", help="build a single orchestration id")
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    parser.add_argument("--repo-root", type=Path, default=None, help="override repo root")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()

    if args.all:
        orch_ids = discover_orchestrations(repo_root)
        if not orch_ids:
            print("no orchestrations found", file=sys.stderr)
            return 0
    elif args.orchestration:
        orch_ids = [args.orchestration]
    else:
        parser.error("specify --all or --orchestration <id>")
        return 2

    build(repo_root, orch_ids, force=args.force)
    print(f"done — snapshots at {snapshots_dir(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
