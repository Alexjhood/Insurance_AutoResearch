"""Command line entrypoint for Phase 0/1 project operations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from autoresearch.config import ensure_project_dirs, load_config
from autoresearch.bootstrap import bootstrap_track
from autoresearch.comparison_runner import compare_against_current_champion, compare_experiments, run_repeated_evaluation
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


def _cmd_bootstrap_track(config, args) -> int:
    parser = build_parser()
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
