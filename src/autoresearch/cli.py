"""Command line entrypoint for Phase 0/1 project operations."""

from __future__ import annotations

import argparse
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
    generate_and_enqueue_proposal,
    run_n_cycles,
    run_next_queued_proposal,
    run_one_cycle,
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
            "or create a timestamped run when none exists."
        ),
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
    init_champion = subparsers.add_parser("init-official-champion", help="Initialise official champion as direct pure premium.")
    init_champion.add_argument("--experiment-id", default=None)
    subparsers.add_parser("generate-proposal", help="Generate, validate, and enqueue one LLM-designed proposal.")
    enqueue = subparsers.add_parser("enqueue-proposal", help="Validate and enqueue a proposal JSON file.")
    enqueue.add_argument("proposal_path")
    subparsers.add_parser("run-next-proposal", help="Run the next queued proposal through comparison and promotion gate.")
    subparsers.add_parser("run-cycle", help="Run one generate -> execute -> compare -> gate cycle.")
    cycles = subparsers.add_parser("run-cycles", help="Run N bounded auto-research cycles.")
    cycles.add_argument("count", type=int)
    subparsers.add_parser("list-proposals", help="Print proposal queue status.")
    subparsers.add_parser("list-champion-history", help="Print official champion history.")
    subparsers.add_parser("list-branches", help="Print branch lineage records.")
    inspect = subparsers.add_parser("inspect-proposal", help="Print one proposal record as JSON.")
    inspect.add_argument("proposal_id")
    eval_ms = subparsers.add_parser("evaluate-milestone", help="Run holdout evaluation for an experiment (token required).")
    eval_ms.add_argument("experiment_id")
    subparsers.add_parser("update-integrity-manifest", help="Recompute integrity manifest after accepting protected-file changes.")
    subparsers.add_parser("export-context", help="Export file-based handoff context for Codex/Claude Code.")
    subparsers.add_parser("write-proposal-template", help="Write proposal template, schema, and instructions.")
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
    config = load_config(
        args.config,
        track_id=getattr(args, "track", None),
        run_id=getattr(args, "run_id", None),
    )

    if args.command == "prepare-data":
        outputs = prepare_data(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "bootstrap-track":
        import json

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
            f"`autoresearch --track {config.track_id} --run-id {config.run_id} run-cycles 10`."
        )
        return 0

    if args.command == "init-registry":
        ensure_project_dirs(config)
        path = init_registry(config.registry_path)
        print(f"registry: {path}")
        return 0

    if args.command == "run-baseline":
        outputs = run_experiment(config, Path(args.experiment_config))
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "run-all-baselines":
        runs = run_all_baselines(config)
        for index, outputs in enumerate(runs, start=1):
            print(f"run {index}")
            for name, path in outputs.items():
                print(f"{name}: {path}")
        return 0

    if args.command == "list-experiments":
        rows = list_experiments(config.registry_path)
        if not rows:
            print("No experiments registered.")
            return 0
        for row in rows:
            print(
                "\t".join(
                    [
                        row["experiment_id"],
                        str(row.get("experiment_name")),
                        str(row.get("target_strategy")),
                        str(row.get("mean_score")),
                        str(row.get("claim_cap_threshold")),
                        str(row.get("status")),
                    ]
                )
            )
        return 0

    if args.command == "run-repeated-evaluation":
        outputs = run_repeated_evaluation(config, args.experiment_id)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "compare-experiments":
        outputs = compare_experiments(config, args.champion_id, args.challenger_id)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "compare-to-champion":
        outputs = compare_against_current_champion(config, args.challenger_id)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "list-promotions":
        rows = list_comparisons(config.registry_path)
        if not rows:
            print("No promotion comparisons registered.")
            return 0
        for row in rows:
            print(
                "\t".join(
                    [
                        row["comparison_id"],
                        row["champion_id"],
                        row["challenger_id"],
                        str(row.get("mean_lift")),
                        str(row.get("challenger_win_rate")),
                        row["promotion_decision"],
                    ]
                )
            )
        return 0

    if args.command == "init-official-champion":
        state = initialise_official_champion(config, args.experiment_id)
        export_context_bundle(config)
        print(f"official_champion: {state['champion_id']}")
        print(f"branch: {state['branch_id']}")
        return 0

    if args.command == "generate-proposal":
        result = generate_and_enqueue_proposal(config)
        print(result)
        return 0

    if args.command == "enqueue-proposal":
        result = enqueue_proposal_from_file(config, Path(args.proposal_path))
        print(result)
        return 0

    if args.command == "run-next-proposal":
        result = run_next_queued_proposal(config)
        export_context_bundle(config)
        print(result)
        return 0

    if args.command == "run-cycle":
        result = run_one_cycle(config)
        print(result)
        return 0

    if args.command == "run-cycles":
        results = run_n_cycles(config, args.count)
        for result in results:
            print(result)
        return 0

    if args.command == "list-proposals":
        rows = list_proposals(config.registry_path)
        if not rows:
            print("No proposals registered.")
            return 0
        for row in rows:
            print(
                "\t".join(
                    [
                        row["proposal_id"],
                        row["status"],
                        str(row.get("experiment_name")),
                        str(row.get("experiment_id")),
                        str(row.get("comparison_id")),
                    ]
                )
            )
        return 0

    if args.command == "list-champion-history":
        current = get_official_champion(config.registry_path)
        if current:
            print(f"current\t{current['champion_id']}\t{current['branch_id']}\t{current['reason']}")
        rows = list_champion_history(config.registry_path)
        if not rows:
            print("No champion history registered.")
            return 0
        for row in rows:
            print(
                "\t".join(
                    [
                        str(row["history_id"]),
                        row["action"],
                        str(row.get("previous_champion_id")),
                        row["new_champion_id"],
                        row["branch_id"],
                        row["reason"],
                    ]
                )
            )
        return 0

    if args.command == "list-branches":
        rows = list_branches(config.registry_path)
        if not rows:
            print("No branches registered.")
            return 0
        for row in rows:
            print(
                "\t".join(
                    [
                        row["branch_id"],
                        str(row.get("parent_branch_id")),
                        str(row.get("root_experiment_id")),
                        str(row.get("current_experiment_id")),
                        row["status"],
                    ]
                )
            )
        return 0

    if args.command == "inspect-proposal":
        import json

        print(json.dumps(get_proposal(config.registry_path, args.proposal_id), indent=2, sort_keys=True))
        return 0

    if args.command == "evaluate-milestone":
        import json
        result = manual_evaluate_on_holdout(config, args.experiment_id)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "update-integrity-manifest":
        manifest_path = write_integrity_manifest(config.root, config.artifacts_dir)
        print(f"Integrity manifest updated: {manifest_path}")
        return 0

    if args.command == "export-context":
        outputs = export_context_bundle(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "write-proposal-template":
        outputs = write_proposal_template(config)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command in {"ingest-proposals", "enqueue-ingested-proposals"}:
        import json

        print(json.dumps(ingest_proposals(config), indent=2, sort_keys=True))
        return 0

    if args.command == "run-latest-proposal-cycle":
        import json

        print(json.dumps(run_latest_proposal_cycle(config), indent=2, sort_keys=True))
        return 0

    if args.command == "show-latest-handoff":
        path = config.handoff_handoffs_dir / "latest_handoff.md"
        if not path.exists():
            outputs = export_context_bundle(config)
            path = outputs["latest_handoff_markdown"]
        print(path.read_text(encoding="utf-8"))
        return 0

    if args.command == "show-proposal-inbox-status":
        import json

        print(json.dumps(inbox_status(config), indent=2, sort_keys=True))
        return 0

    if args.command == "start-session":
        import json

        print(json.dumps(create_session(config, args.name, args.max_cycles), indent=2, sort_keys=True))
        return 0

    if args.command == "session-status":
        import json

        print(json.dumps(session_status(config, args.session_id), indent=2, sort_keys=True))
        return 0

    if args.command == "pause-session":
        import json

        print(json.dumps(pause_session(config, args.session_id), indent=2, sort_keys=True))
        return 0

    if args.command == "resume-session":
        import json

        print(json.dumps(resume_session(config, args.session_id), indent=2, sort_keys=True))
        return 0

    if args.command == "stop-session":
        import json

        print(json.dumps(stop_session(config, args.session_id), indent=2, sort_keys=True))
        return 0

    if args.command == "run-session-cycle":
        import json

        print(json.dumps(run_session_cycle(config, args.session_id), indent=2, sort_keys=True))
        return 0

    if args.command == "run-session-cycles":
        import json

        print(json.dumps(run_session_cycles(config, args.count, args.session_id), indent=2, sort_keys=True))
        return 0

    if args.command == "compare-tracks":
        import json
        from autoresearch.tracks import compare_tracks

        config_a = load_config(args.config, track_id=args.track_a)
        config_b = load_config(args.config, track_id=args.track_b)
        result = compare_tracks(config_a, config_b)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if result.get("report_path"):
            print(f"\nFull report: {result['report_path']}")
        return 0 if result.get("status") != "error" else 1

    if args.command == "list-tracks":
        from autoresearch.config import load_config as _lc
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
                from autoresearch.experiment_registry.registry import registry_counts
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
