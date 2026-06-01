"""Command line entrypoint for Phase 0/1 project operations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from autoresearch.config import ensure_project_dirs, load_config
from autoresearch.bootstrap import bootstrap_track
from autoresearch.comparison_runner import compare_against_current_champion, compare_experiments, record_decision, run_repeated_evaluation
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import (
    export_context_bundle,
    inbox_status,
    ingest_proposals,
    run_latest_proposal_cycle,
    write_proposal_template,
)
from autoresearch.controller.session import (
    create_session,
    pause_session,
    resume_session,
    run_session_cycle,
    run_session_cycles,
    session_status,
    stop_session,
)
from autoresearch.controller.workflow import (
    enqueue_proposal_from_file,
    run_next_queued_proposal,
)
from autoresearch.data.pipeline import prepare_data
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    get_proposal,
    init_registry,
    list_branches,
    list_champion_history,
    list_comparisons,
    list_experiments,
    list_proposals,
)
from autoresearch.experiment_runner import run_all_baselines, run_experiment
from autoresearch.milestone import manual_evaluate_on_holdout
from autoresearch.utils.integrity import write_integrity_manifest


def _cmd_prepare_data(config, args) -> int:
    outputs = prepare_data(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _inject_identity(config, args):
    """Return config with model identity injected from args if provided."""
    provider = getattr(args, "model_provider", None)
    name = getattr(args, "model_name", None)
    version = getattr(args, "model_version", None)
    harness = getattr(args, "harness", None)
    if provider or name or version or harness:
        return replace(
            config,
            model_provider=provider or config.model_provider,
            model_name=name or config.model_name,
            model_version=version or config.model_version,
            model_harness=harness or config.model_harness,
        )
    return config


def _cmd_bootstrap_track(config, args) -> int:
    parser = build_parser()
    config = _inject_identity(config, args)
    try:
        result = bootstrap_track(
            config,
            prepare_shared_data=not args.skip_data,
            force_prepare_data=args.force_data,
            run_baselines=not args.skip_baselines,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"\nReady: read {result['context']} and continue with "
        f"`autoresearch --track {config.track_id} --run-id {config.run_id} run-session-cycles 10`."
    )
    return 0


def _cmd_init_registry(config, args) -> int:
    ensure_project_dirs(config)
    path = init_registry(config.registry_path)
    print(f"registry: {path}")
    return 0


def _cmd_run_baseline(config, args) -> int:
    outputs = run_experiment(config, Path(args.experiment_config))
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _cmd_run_all_baselines(config, args) -> int:
    runs = run_all_baselines(config)
    for index, outputs in enumerate(runs, start=1):
        print(f"run {index}")
        for name, path in outputs.items():
            print(f"{name}: {path}")
    return 0


def _cmd_list_experiments(config, args) -> int:
    rows = list_experiments(config.registry_path)
    if not rows:
        print("No experiments registered.")
        return 0
    for row in rows:
        print(
            "\t".join([
                row["experiment_id"],
                str(row.get("experiment_name")),
                str(row.get("target_strategy")),
                str(row.get("mean_score")),
                str(row.get("claim_cap_threshold")),
                str(row.get("status")),
            ])
        )
    return 0


def _cmd_run_repeated_evaluation(config, args) -> int:
    outputs = run_repeated_evaluation(config, args.experiment_id)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _cmd_compare_experiments(config, args) -> int:
    outputs = compare_experiments(config, args.champion_id, args.challenger_id)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _cmd_compare_to_champion(config, args) -> int:
    outputs = compare_against_current_champion(config, args.challenger_id)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _cmd_record_decision(config, args) -> int:
    result = record_decision(
        config,
        args.comparison_id,
        decision=args.decision,
        rationale=args.rationale,
    )
    print(f"Decision recorded: {result['decision']}")
    print(f"Rationale: {result['rationale']}")
    print(f"Decided at: {result['decided_at']}")
    if not result.get("guardrail_result", {}).get("passed", True):
        print(f"Guardrail failures: {result['guardrail_result']['failures']}")
    return 0


def _cmd_list_promotions(config, args) -> int:
    rows = list_comparisons(config.registry_path)
    if not rows:
        print("No promotion comparisons registered.")
        return 0
    for row in rows:
        print(
            "\t".join([
                row["comparison_id"],
                row["champion_id"],
                row["challenger_id"],
                str(row.get("mean_lift")),
                str(row.get("challenger_win_rate")),
                row["promotion_decision"],
            ])
        )
    return 0


def _cmd_init_official_champion(config, args) -> int:
    state = initialise_official_champion(config, args.experiment_id)
    export_context_bundle(config)
    print(f"official_champion: {state['champion_id']}")
    print(f"branch: {state['branch_id']}")
    return 0


def _cmd_enqueue_proposal(config, args) -> int:
    result = enqueue_proposal_from_file(config, Path(args.proposal_path))
    print(result)
    return 0


def _cmd_run_next_proposal(config, args) -> int:
    result = run_next_queued_proposal(config)
    export_context_bundle(config)
    print(result)
    return 0


def _cmd_list_proposals(config, args) -> int:
    rows = list_proposals(config.registry_path)
    if not rows:
        print("No proposals registered.")
        return 0
    for row in rows:
        print(
            "\t".join([
                row["proposal_id"],
                row["status"],
                str(row.get("experiment_name")),
                str(row.get("experiment_id")),
                str(row.get("comparison_id")),
            ])
        )
    return 0


def _cmd_list_champion_history(config, args) -> int:
    current = get_official_champion(config.registry_path)
    if current:
        print(f"current\t{current['champion_id']}\t{current['branch_id']}\t{current['reason']}")
    rows = list_champion_history(config.registry_path)
    if not rows:
        print("No champion history registered.")
        return 0
    for row in rows:
        print(
            "\t".join([
                str(row["history_id"]),
                row["action"],
                str(row.get("previous_champion_id")),
                row["new_champion_id"],
                row["branch_id"],
                row["reason"],
            ])
        )
    return 0


def _cmd_list_branches(config, args) -> int:
    rows = list_branches(config.registry_path)
    if not rows:
        print("No branches registered.")
        return 0
    for row in rows:
        print(
            "\t".join([
                row["branch_id"],
                str(row.get("parent_branch_id")),
                str(row.get("root_experiment_id")),
                str(row.get("current_experiment_id")),
                row["status"],
            ])
        )
    return 0


def _cmd_inspect_proposal(config, args) -> int:
    print(json.dumps(get_proposal(config.registry_path, args.proposal_id), indent=2, sort_keys=True))
    return 0


def _cmd_evaluate_milestone(config, args) -> int:
    result = manual_evaluate_on_holdout(config, args.experiment_id)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_update_integrity_manifest(config, args) -> int:
    manifest_path = write_integrity_manifest(config.root, config.artifacts_dir)
    print(f"Integrity manifest updated: {manifest_path}")
    return 0


def _cmd_export_context(config, args) -> int:
    outputs = export_context_bundle(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _cmd_write_proposal_template(config, args) -> int:
    outputs = write_proposal_template(config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _cmd_ingest_proposals(config, args) -> int:
    print(json.dumps(ingest_proposals(config), indent=2, sort_keys=True))
    return 0


def _cmd_run_latest_proposal_cycle(config, args) -> int:
    print(json.dumps(run_latest_proposal_cycle(config), indent=2, sort_keys=True))
    return 0


def _cmd_show_latest_handoff(config, args) -> int:
    path = config.handoff_handoffs_dir / "latest_handoff.md"
    if not path.exists():
        outputs = export_context_bundle(config)
        path = outputs["latest_handoff_markdown"]
    print(path.read_text(encoding="utf-8"))
    return 0


def _cmd_show_proposal_inbox_status(config, args) -> int:
    print(json.dumps(inbox_status(config), indent=2, sort_keys=True))
    return 0


def _cmd_start_session(config, args) -> int:
    config = _inject_identity(config, args)
    ensure_project_dirs(config)
    print(json.dumps(create_session(config, args.name, args.max_cycles), indent=2, sort_keys=True))
    return 0


def _cmd_session_status(config, args) -> int:
    print(json.dumps(session_status(config, args.session_id), indent=2, sort_keys=True))
    return 0


def _cmd_pause_session(config, args) -> int:
    print(json.dumps(pause_session(config, args.session_id), indent=2, sort_keys=True))
    return 0


def _cmd_resume_session(config, args) -> int:
    print(json.dumps(resume_session(config, args.session_id), indent=2, sort_keys=True))
    return 0


def _cmd_stop_session(config, args) -> int:
    print(json.dumps(stop_session(config, args.session_id), indent=2, sort_keys=True))
    return 0


def _cmd_run_session_cycle(config, args) -> int:
    print(json.dumps(run_session_cycle(config, args.session_id), indent=2, sort_keys=True))
    return 0


def _cmd_run_session_cycles(config, args) -> int:
    print(json.dumps(run_session_cycles(config, args.count, args.session_id), indent=2, sort_keys=True))
    return 0


def _cmd_compare_tracks(config, args) -> int:
    from autoresearch.tracks import compare_tracks

    config_a = load_config(args.config, track_id=args.track_a)
    config_b = load_config(args.config, track_id=args.track_b)
    result = compare_tracks(config_a, config_b)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if result.get("report_path"):
        print(f"\nFull report: {result['report_path']}")
    return 0 if result.get("status") != "error" else 1


def _cmd_memory(config, args) -> int:
    """Dispatch memory sub-commands."""
    from autoresearch.memory.harvester import harvest_all, harvest_run
    from autoresearch.memory.store import default_memory_store_path, init_memory_store, memory_store_counts

    memory_path = default_memory_store_path()
    sub = getattr(args, "memory_subcommand", None)

    if sub == "harvest":
        if getattr(args, "all", False):
            result = harvest_all(memory_path)
            print(json.dumps(result, indent=2))
        else:
            # Harvest current run
            manifest_path = config.artifacts_dir / "run_manifest.json"
            if not manifest_path.exists():
                print("No run_manifest.json found. Run bootstrap-track first.")
                return 1
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Cannot read manifest: {exc}")
                return 1
            identity = manifest.get("model_identity")
            if not identity:
                print(
                    "run_manifest.json has no model_identity. "
                    "Run `autoresearch memory backfill-identity` first."
                )
                return 1
            harvest_run(
                memory_path,
                config.registry_path,
                identity,
                track_id=config.track_id,
                run_id=config.run_id,
            )
            counts = memory_store_counts(memory_path)
            print(json.dumps({"status": "ok", "memory_path": str(memory_path), "counts": counts}, indent=2))
        return 0

    if sub == "backfill-identity":
        return _memory_backfill_identity(config, args, memory_path)

    if sub == "status":
        if not memory_path.exists():
            print("Memory store does not exist yet. Run `autoresearch memory harvest --all`.")
            return 0
        counts = memory_store_counts(memory_path)
        print(json.dumps({"memory_path": str(memory_path), "counts": counts}, indent=2))
        return 0

    if sub == "record-insight":
        return _memory_record_insight(config, args, memory_path)

    if sub == "list-insights":
        return _memory_list_insights(config, args, memory_path)

    if sub == "query":
        return _memory_query(config, args, memory_path)

    if sub == "build-playbook":
        return _memory_build_playbook(config, args, memory_path)

    print(f"Unknown memory subcommand: {sub}")
    return 2


def _memory_record_insight(config, args, memory_path: "Path") -> int:
    """Record an evidence-bound insight into the aggregator."""
    from autoresearch.memory.insights import record_insight

    file_path = Path(getattr(args, "file", None) or "")
    if not file_path.exists():
        print(f"Insight JSON file not found: {file_path}")
        return 1

    try:
        insight_dict = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Cannot read insight file: {exc}")
        return 1

    # Read model_identity and run_uid from run_manifest.json
    manifest_path = config.artifacts_dir / "run_manifest.json"
    if not manifest_path.exists():
        print("No run_manifest.json found. Run bootstrap-track first.")
        return 1
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Cannot read manifest: {exc}")
        return 1

    model_identity = manifest.get("model_identity")
    if not model_identity:
        print("run_manifest.json has no model_identity. Run `autoresearch memory backfill-identity` first.")
        return 1

    track_id = manifest.get("track_id") or config.track_id
    run_id = manifest.get("run_id") or config.run_id
    run_uid = f"{track_id}/{run_id}" if track_id and run_id else str(config.artifacts_dir)

    result = record_insight(
        memory_path,
        run_uid,
        model_identity,
        insight_dict,
        run_registry_path=config.registry_path,
    )
    print(json.dumps(result, indent=2))
    return 0


def _memory_list_insights(config, args, memory_path: "Path") -> int:
    """List insights from the aggregator."""
    from autoresearch.memory.insights import list_insights

    verified_only = not getattr(args, "include_unverified", False)
    run_filter = getattr(args, "run", None)
    rows = list_insights(memory_path, verified_only=verified_only, run_uid=run_filter)
    print(json.dumps(rows, indent=2))
    return 0


def _memory_build_playbook(config, args, memory_path: "Path") -> int:
    """Build or regenerate the verified-insights playbook."""
    from autoresearch.memory.playbook import build_playbook

    threshold = getattr(config, "structural_gini_threshold", 0.37)
    model_filter = getattr(args, "model_filter", None)
    path = build_playbook(memory_path, structural_gini_threshold=threshold, model_id_filter=model_filter)
    if path is None:
        print("No verified insights found — playbook not generated.")
        return 0
    print(f"Playbook written to: {path}")
    return 0


def _memory_query(config, args, memory_path: "Path") -> int:
    """Dispatch memory query subcommands (respects access gate)."""
    from autoresearch.config import PROJECT_ROOT
    from autoresearch.memory import resolve_memory_access
    from autoresearch.memory.query import AccessDeniedError, query_insights, query_experiments, run_analysis

    access = resolve_memory_access(config)
    if access == "none":
        print(
            "Memory query refused: AUTORESEARCH_MEMORY_ACCESS is not set (default: none). "
            "Set it to 'own' or 'all' to enable memory queries."
        )
        return 1

    # Derive own_model_id from manifest if available
    own_model_id: str | None = None
    manifest_path = config.artifacts_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity = manifest.get("model_identity") or {}
            provider = (identity.get("provider") or "").lower().strip()
            name = (identity.get("name") or "").lower().strip()
            if provider and name:
                own_model_id = f"{provider}/{name}"
        except (OSError, json.JSONDecodeError):
            pass

    try:
        if getattr(args, "insights", False):
            rows = query_insights(
                memory_path,
                access,
                model_id=getattr(args, "model", None),
                verified_only=not getattr(args, "include_unverified", False),
                own_model_id=own_model_id,
            )
            print(json.dumps(rows, indent=2))
            return 0

        if getattr(args, "experiments", False):
            rows = query_experiments(
                memory_path,
                access,
                own_model_id=own_model_id,
                filter_str=getattr(args, "filter", None),
            )
            print(json.dumps(rows, indent=2))
            return 0

        analysis_name = getattr(args, "analysis", None)
        if analysis_name:
            rows = run_analysis(
                memory_path,
                access,
                analysis_name,
                own_model_id=own_model_id,
                threshold=getattr(config, "structural_gini_threshold", 0.37),
            )
            print(json.dumps(rows, indent=2))
            return 0

    except AccessDeniedError as exc:
        print(f"Access denied: {exc}")
        return 1

    print("Specify --insights, --experiments, or --analysis <name>.")
    return 2


def _memory_backfill_identity(config, args, memory_path: "Path") -> int:
    """Write model_identity into run_manifest.json files that lack it."""
    from autoresearch.config import PROJECT_ROOT

    provider = getattr(args, "provider", None)
    name = getattr(args, "name", None)
    version = getattr(args, "version", None) or ""
    harness = getattr(args, "harness", None) or ""
    run_dir_arg = getattr(args, "run_dir", None)
    all_missing = getattr(args, "all_missing", False)

    if not provider or not name:
        print("--provider and --name are required for backfill-identity.")
        return 1

    identity = {
        "provider": provider.lower().strip(),
        "name": name.lower().strip(),
        "version": version,
        "harness": harness,
    }

    if run_dir_arg:
        targets = [Path(run_dir_arg) / "run_manifest.json"]
    elif all_missing:
        tracks_base = PROJECT_ROOT / "artifacts" / "tracks"
        targets = list(tracks_base.rglob("run_manifest.json"))
    else:
        targets = [config.artifacts_dir / "run_manifest.json"]

    patched = 0
    skipped = 0
    for mpath in targets:
        if not mpath.exists():
            print(f"  not found: {mpath}")
            skipped += 1
            continue
        try:
            existing = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  error reading {mpath}: {exc}")
            skipped += 1
            continue
        if "model_identity" in existing and not getattr(args, "force", False):
            print(f"  already has identity: {mpath}")
            skipped += 1
            continue
        existing["model_identity"] = identity
        mpath.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  patched: {mpath}")
        patched += 1

    print(f"backfill-identity: patched={patched} skipped={skipped}")
    return 0


def _cmd_list_tracks(config, args) -> int:
    from autoresearch.config import load_config as _lc
    from autoresearch.experiment_registry.registry import registry_counts

    base_cfg = _lc(args.config)
    tracks_dir = base_cfg.artifacts_dir / "tracks"
    if not tracks_dir.exists():
        print("No tracks found (artifacts/tracks/ does not exist).")
        return 0
    for track_dir in sorted(tracks_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        tc = load_config(args.config, track_id=track_dir.name)
        registry = tc.registry_path
        has_registry = "✓" if registry.exists() else "✗"
        log = tc.research_log_path
        has_log = "✓" if log.exists() else "✗"
        try:
            champ = get_official_champion(tc.registry_path)
            counts = registry_counts(tc.registry_path)
            champ_id = champ["champion_id"] if champ else "none"
            n_exp = counts["experiments"]
        except Exception:
            champ_id = "?"
            n_exp = "?"
        print(
            f"{track_dir.name}\trun={tc.run_id}\tregistry={has_registry}\t"
            f"log={has_log}\tchampion={champ_id}\texperiments={n_exp}"
        )
    return 0


COMMANDS = {
    "prepare-data": _cmd_prepare_data,
    "bootstrap-track": _cmd_bootstrap_track,
    "init-registry": _cmd_init_registry,
    "run-baseline": _cmd_run_baseline,
    "run-all-baselines": _cmd_run_all_baselines,
    "list-experiments": _cmd_list_experiments,
    "run-repeated-evaluation": _cmd_run_repeated_evaluation,
    "compare-experiments": _cmd_compare_experiments,
    "compare-to-champion": _cmd_compare_to_champion,
    "record-decision": _cmd_record_decision,
    "list-promotions": _cmd_list_promotions,
    "init-official-champion": _cmd_init_official_champion,
    "enqueue-proposal": _cmd_enqueue_proposal,
    "run-next-proposal": _cmd_run_next_proposal,
    "list-proposals": _cmd_list_proposals,
    "list-champion-history": _cmd_list_champion_history,
    "list-branches": _cmd_list_branches,
    "inspect-proposal": _cmd_inspect_proposal,
    "evaluate-milestone": _cmd_evaluate_milestone,
    "update-integrity-manifest": _cmd_update_integrity_manifest,
    "export-context": _cmd_export_context,
    "write-proposal-template": _cmd_write_proposal_template,
    "ingest-proposals": _cmd_ingest_proposals,
    "enqueue-ingested-proposals": _cmd_ingest_proposals,
    "run-latest-proposal-cycle": _cmd_run_latest_proposal_cycle,
    "show-latest-handoff": _cmd_show_latest_handoff,
    "show-proposal-inbox-status": _cmd_show_proposal_inbox_status,
    "start-session": _cmd_start_session,
    "session-status": _cmd_session_status,
    "pause-session": _cmd_pause_session,
    "resume-session": _cmd_resume_session,
    "stop-session": _cmd_stop_session,
    "run-session-cycle": _cmd_run_session_cycle,
    "run-session-cycles": _cmd_run_session_cycles,
    "compare-tracks": _cmd_compare_tracks,
    "list-tracks": _cmd_list_tracks,
    "memory": _cmd_memory,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch")
    parser.add_argument("--config", default=None, help="Path to TOML config file.")
    parser.add_argument(
        "--track",
        default=None,
        metavar="NAME",
        help=(
            "Research track name (e.g. 'claude', 'codex'). "
            "All artifact paths and the registry are scoped under "
            "artifacts/tracks/<NAME>/ so multiple agents can run in "
            "complete isolation.  Omit to use the default (untracked) paths."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help=(
            "Run id within the selected track. Tracked artifacts are written to "
            "artifacts/tracks/<track>/runs/<ID>/. Omit to continue the latest run "
            "or combine --track with --new-run to force a fresh timestamped run."
        ),
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Create a fresh timestamped run within the selected track. "
            "Use this for new agent sessions; omit it to continue the latest run."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=("burning_cost", "frequency"),
        default=None,
        help="Override the configured evaluation target. Default is burning_cost unless the config says otherwise.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare-data", help="Build Phase 1 data artifacts.")
    bootstrap = subparsers.add_parser(
        "bootstrap-track",
        help="Idempotently prepare data, registry, baselines, champion, templates, and context for a named track.",
    )
    bootstrap.add_argument("--skip-data", action="store_true", help="Do not run prepare-data even if shared data is missing.")
    bootstrap.add_argument("--force-data", action="store_true", help="Rebuild shared data artifacts before bootstrapping.")
    bootstrap.add_argument("--skip-baselines", action="store_true", help="Do not run baseline experiments if the registry is empty.")
    bootstrap.add_argument("--model-provider", default=None, metavar="PROVIDER", help="LLM provider (e.g. anthropic, openai). Required.")
    bootstrap.add_argument("--model-name", default=None, metavar="NAME", help="LLM model name (e.g. claude-sonnet-4-6). Required.")
    bootstrap.add_argument("--model-version", default=None, metavar="VERSION", help="LLM model version string (optional).")
    bootstrap.add_argument("--harness", default=None, metavar="HARNESS", help="Agent harness name (e.g. claude-code, codex, opencode).")
    subparsers.add_parser("init-registry", help="Create the SQLite experiment registry.")
    run_parser = subparsers.add_parser("run-baseline", help="Run one deterministic baseline experiment.")
    run_parser.add_argument("experiment_config", help="Path to an experiment TOML config.")
    subparsers.add_parser("run-all-baselines", help="Run all baseline configs under configs/experiments.")
    subparsers.add_parser("list-experiments", help="Print registered experiment summaries.")
    repeated_parser = subparsers.add_parser("run-repeated-evaluation", help="Resample one experiment's search-time predictions.")
    repeated_parser.add_argument("experiment_id")
    compare_parser = subparsers.add_parser("compare-experiments", help="Compare champion and challenger experiment ids.")
    compare_parser.add_argument("champion_id")
    compare_parser.add_argument("challenger_id")
    champion_parser = subparsers.add_parser(
        "compare-to-champion",
        help="Compare a challenger against the official champion when initialised.",
    )
    champion_parser.add_argument("challenger_id")
    decision_parser = subparsers.add_parser(
        "record-decision",
        help="Record the LLM's promote/local_promote/reject decision for a pending comparison.",
    )
    decision_parser.add_argument("comparison_id", help="comparison_id from compare-experiments output.")
    decision_parser.add_argument("--decision", required=True, choices=["promote", "local_promote", "reject"],
                                 help="LLM's final verdict.")
    decision_parser.add_argument("--rationale", required=True,
                                 help="Written justification for the decision.")
    subparsers.add_parser("list-promotions", help="Print volatility-aware comparison and promotion decisions.")
    init_champion = subparsers.add_parser("init-official-champion", help="Initialise official champion as the global-mean baseline for the active target.")
    init_champion.add_argument("--experiment-id", default=None)
    enqueue = subparsers.add_parser("enqueue-proposal", help="Validate and enqueue a proposal JSON file.")
    enqueue.add_argument("proposal_path")
    subparsers.add_parser("run-next-proposal", help="Run the next queued proposal through comparison and promotion gate.")
    subparsers.add_parser("list-proposals", help="Print proposal queue status.")
    subparsers.add_parser("list-champion-history", help="Print official champion history.")
    subparsers.add_parser("list-branches", help="Print branch lineage records.")
    inspect = subparsers.add_parser("inspect-proposal", help="Print one proposal record as JSON.")
    inspect.add_argument("proposal_id")
    eval_ms = subparsers.add_parser("evaluate-milestone", help="Run holdout evaluation for an experiment (token required).")
    eval_ms.add_argument("experiment_id")
    subparsers.add_parser("update-integrity-manifest", help="Recompute integrity manifest after accepting protected-file changes.")
    subparsers.add_parser("export-context", help="Export file-based handoff context for Codex/Claude Code.")
    subparsers.add_parser("write-proposal-template", help="Write proposal template and schema.")
    subparsers.add_parser("ingest-proposals", help="Validate proposal files from the handoff inbox and enqueue valid ones.")
    subparsers.add_parser(
        "enqueue-ingested-proposals",
        help="Alias for ingest-proposals; validates inbox files and queues valid proposals.",
    )
    subparsers.add_parser("run-latest-proposal-cycle", help="Ingest newest inbox proposal and run one gated cycle.")
    subparsers.add_parser("show-latest-handoff", help="Print the latest handoff Markdown summary.")
    subparsers.add_parser("show-proposal-inbox-status", help="Print handoff inbox and processed-folder status.")
    start_session = subparsers.add_parser("start-session", help="Create a supervised autonomous research session.")
    start_session.add_argument("name")
    start_session.add_argument("--max-cycles", type=int, default=None)
    start_session.add_argument("--model-provider", default=None, metavar="PROVIDER", help="LLM provider (e.g. anthropic). Written into run_manifest.json if not already present.")
    start_session.add_argument("--model-name", default=None, metavar="NAME", help="LLM model name. Written into run_manifest.json if not already present.")
    start_session.add_argument("--model-version", default=None, metavar="VERSION", help="LLM model version string (optional).")
    start_session.add_argument("--harness", default=None, metavar="HARNESS", help="Agent harness name (optional).")
    session_status_parser = subparsers.add_parser("session-status", help="Inspect latest or specified session status.")
    session_status_parser.add_argument("--session-id", default=None)
    pause_parser = subparsers.add_parser("pause-session", help="Pause latest or specified session.")
    pause_parser.add_argument("--session-id", default=None)
    resume_parser = subparsers.add_parser("resume-session", help="Resume latest or specified session.")
    resume_parser.add_argument("--session-id", default=None)
    stop_parser = subparsers.add_parser("stop-session", help="Stop latest or specified session cleanly.")
    stop_parser.add_argument("--session-id", default=None)
    step_parser = subparsers.add_parser("run-session-cycle", help="Run one local-side cycle for a session.")
    step_parser.add_argument("--session-id", default=None)
    multi_parser = subparsers.add_parser("run-session-cycles", help="Run up to N local-side session cycles.")
    multi_parser.add_argument("count", type=int)
    multi_parser.add_argument("--session-id", default=None)
    compare_tracks_parser = subparsers.add_parser(
        "compare-tracks",
        help="Compare the official champions of two research tracks without promoting either.",
    )
    compare_tracks_parser.add_argument("track_a", help="First track name (e.g. 'claude').")
    compare_tracks_parser.add_argument("track_b", help="Second track name (e.g. 'codex').")
    subparsers.add_parser("list-tracks", help="List all tracks that have a registry under artifacts/tracks/.")

    # Memory subcommand group
    memory_parser = subparsers.add_parser(
        "memory",
        help="Cross-run memory aggregator commands (harvest, backfill-identity, status).",
    )
    memory_subs = memory_parser.add_subparsers(dest="memory_subcommand", required=True)

    mem_harvest = memory_subs.add_parser(
        "harvest",
        help="Harvest one run (or --all) into the aggregator store.",
    )
    mem_harvest.add_argument(
        "--all",
        action="store_true",
        help="Discover and harvest every run under artifacts/tracks/.",
    )

    mem_backfill = memory_subs.add_parser(
        "backfill-identity",
        help="Write model_identity into run_manifest.json files that lack it.",
    )
    mem_backfill.add_argument("--provider", required=True, help="LLM provider (e.g. anthropic).")
    mem_backfill.add_argument("--name", required=True, help="LLM model name (e.g. claude-sonnet-4-6).")
    mem_backfill.add_argument("--version", default="", help="LLM model version (optional).")
    mem_backfill.add_argument("--harness", default="", help="Agent harness name (optional).")
    mem_backfill.add_argument("--run-dir", default=None, metavar="PATH", help="Patch a specific run directory only.")
    mem_backfill.add_argument("--all-missing", action="store_true", help="Patch all run manifests under artifacts/tracks/ that lack model_identity.")
    mem_backfill.add_argument("--force", action="store_true", help="Overwrite existing model_identity entries.")

    memory_subs.add_parser(
        "status",
        help="Print row counts for each table in the aggregator store.",
    )

    mem_record_insight = memory_subs.add_parser(
        "record-insight",
        help="Record an evidence-bound insight from a JSON file into the aggregator.",
    )
    mem_record_insight.add_argument(
        "--file",
        required=True,
        metavar="PATH",
        help="Path to a JSON file containing the insight (see docs/CLI.md for schema).",
    )

    mem_list_insights = memory_subs.add_parser(
        "list-insights",
        help="List insights stored in the aggregator.",
    )
    mem_list_insights.add_argument(
        "--verified-only",
        action="store_true",
        default=True,
        help="Return only verified=1 insights (default).",
    )
    mem_list_insights.add_argument(
        "--include-unverified",
        action="store_true",
        default=False,
        help="Include insights with verified=0.",
    )
    mem_list_insights.add_argument(
        "--run",
        default=None,
        metavar="RUN_UID",
        help="Filter to a specific run_uid.",
    )

    mem_build_playbook = memory_subs.add_parser(
        "build-playbook",
        help="Compile verified insights into the cross-run playbook (under the out-of-tree memory dir; override with AUTORESEARCH_MEMORY_DIR).",
    )
    mem_build_playbook.add_argument(
        "--model-filter",
        default=None,
        metavar="MODEL_ID",
        help="Produce a filtered own-model variant (e.g. 'anthropic/claude-sonnet-4-6').",
    )

    mem_query = memory_subs.add_parser(
        "query",
        help="Query the aggregator (respects AUTORESEARCH_MEMORY_ACCESS gate).",
    )
    query_mode = mem_query.add_mutually_exclusive_group()
    query_mode.add_argument(
        "--insights",
        action="store_true",
        default=False,
        help="Retrieve insights.",
    )
    query_mode.add_argument(
        "--experiments",
        action="store_true",
        default=False,
        help="Retrieve experiments.",
    )
    query_mode.add_argument(
        "--analysis",
        metavar="NAME",
        help="Run a named canned analysis (peak-gini-by-framing, plateau-families, "
             "biggest-single-jumps, efficiency-by-model).",
    )
    mem_query.add_argument("--model", default=None, metavar="MODEL_ID", help="Filter to a model_id.")
    mem_query.add_argument("--filter", default=None, metavar="STR", help="Free-text filter hint for experiments.")
    mem_query.add_argument("--include-unverified", action="store_true", default=False,
                           help="Include unverified insights (only applies to --insights).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "new_run", False) and getattr(args, "run_id", None):
        parser.error("--new-run cannot be used together with --run-id")
    config = load_config(
        args.config,
        track_id=getattr(args, "track", None),
        run_id=getattr(args, "run_id", None),
        new_run=getattr(args, "new_run", False),
    )
    if getattr(args, "target_mode", None):
        config = replace(config, target_mode=args.target_mode)
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2
    return handler(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
