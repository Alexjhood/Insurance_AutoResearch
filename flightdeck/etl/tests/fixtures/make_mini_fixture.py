"""Build a synthetic mini-orchestration so the ETL tests run anywhere.

Creates a tiny but structurally-complete orchestration tree (JSON artifacts +
minimal sqlite DBs) under an arbitrary ``repo_root``, deliberately exercising
every DATA.md §4 quirk:

1. takeover run with ``attempted > decided`` and an experiment with no decision.
2. seed experiment duplicating a prior champion (``is_seed``).
3. an experiment id present in a delegation registry *and* the consolidation
   registry (dedup + replay link).
4. mixed lift bases: first challenger is baseline-relative, later incremental.
5. ``telemetry.sqlite-wal`` present (read-only URI mode).
6. an unreadable/partial registry (degrade to nulls + warning).
7. a crashed delegation whose ledger has no ``tool_usage`` (token fallback).

Public entry: :func:`make_mini_fixture(repo_root, orch_id=...) -> orch_id`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ORCH_ID = "MINI0101T000000Z"
TRACK = "codex"
CONS_RUN = "MINI0101T010000Z"

# Experiment ids (timestamp-style prefixes to mirror real data).
GM = "T000100Z_global_mean_baseline"
EXP_A = "T000200Z_lgbm_poisson_native_baseline"
EXP_B = "T000300Z_lgbm_tweedie_native_followup"
SEED_D02 = "T000400Z_orchestration_delegation_seed_d02_lgbm_poisson_native_baseline"
EXP_C = "T000500Z_xgb_poisson_ordinal_root"
EXP_UND = "T000600Z_forfeited_undecided"
CONS_GM = "T010100Z_global_mean_baseline"
# Mirrors real playoff naming: the overall winner (d01) is seeded, the other
# finalist (d02) enters as a challenger — so each finalist matches exactly one
# consolidation experiment by its "_dNN_" tag.
CONS_SEED = "T010200Z_orchestration_seed_d01_lgbm_poisson_native_baseline"
CONS_CH = "T010300Z_orchestration_challenger_02_d02_xgb_poisson_ordinal_root"


def _panel(gini: float, rank: float = 0.3, asym: float = 0.45, cal: float = 0.99) -> dict:
    return {
        "gini_weighted": gini,
        "rank_gini_weighted": rank,
        "asym_pricing_loss": asym,
        "predicted_to_actual_ratio": cal,
    }


def _screening(challenger_gini: float, champion_id: str, champion_gini: float,
               lift: float, win_rate: float) -> str:
    return json.dumps({
        "challenger_id": "x",
        "challenger_score": challenger_gini,
        "challenger_metric_panel": _panel(challenger_gini),
        "champion_id": champion_id,
        "champion_score": champion_gini,
        "champion_metric_panel": _panel(champion_gini, rank=0.04, cal=0.987),
        "lift": lift,
        "bootstrap_win_rate": win_rate,
    })


def _recipe(objective: str, encoding: str, estimator: str = "lightgbm") -> dict:
    return {
        "experiment_name": "x",
        "model": {"recipe": {
            "structure": "direct", "estimator": estimator,
            "objective": objective, "encoding": encoding,
            "params": {"num_leaves": 63}, "early_stopping": 50,
        }},
        "model_family": "recipe",
        "target_strategy": "direct_pure_premium",
    }


# --------------------------------------------------------------------------- #
# sqlite helpers
# --------------------------------------------------------------------------- #
def _make_registry(path: Path, experiments, proposals, comparisons,
                   champion_history, nodes, log_entries, session_events=()):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE experiments (experiment_id TEXT, created_at TEXT, "
        "experiment_name TEXT, status TEXT, model_family TEXT, target_strategy TEXT, "
        "parent_experiment_id TEXT, metrics_path TEXT, fit_wall_seconds REAL)"
    )
    cur.executemany("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?)", experiments)
    cur.execute(
        "CREATE TABLE proposals (experiment_id TEXT, config_json TEXT, "
        "change_summary TEXT, expected_benefit TEXT, key_risk TEXT, rationale TEXT)"
    )
    cur.executemany("INSERT INTO proposals VALUES (?,?,?,?,?,?)", proposals)
    cur.execute(
        "CREATE TABLE comparisons (comparison_id TEXT, created_at TEXT, champion_id TEXT, "
        "challenger_id TEXT, bootstrap_summary TEXT, decision TEXT, decided_by TEXT, "
        "decision_reason_code TEXT, decision_rationale TEXT, guardrail_status TEXT)"
    )
    cur.executemany("INSERT INTO comparisons VALUES (?,?,?,?,?,?,?,?,?,?)", comparisons)
    cur.execute(
        "CREATE TABLE champion_history (history_id INTEGER, created_at TEXT, "
        "previous_champion_id TEXT, new_champion_id TEXT, action TEXT, comparison_id TEXT)"
    )
    cur.executemany("INSERT INTO champion_history VALUES (?,?,?,?,?,?)", champion_history)
    cur.execute(
        "CREATE TABLE research_nodes (node_id TEXT, parent_node_id TEXT, experiment_id TEXT, "
        "line_id TEXT, hypothesis TEXT, change_summary TEXT, expected_benefit TEXT, "
        "key_risk TEXT, screening_json TEXT, metrics_json TEXT)"
    )
    cur.executemany("INSERT INTO research_nodes VALUES (?,?,?,?,?,?,?,?,?,?)", nodes)
    cur.execute(
        "CREATE TABLE research_log_entries (cycle INTEGER, experiment_id TEXT, "
        "comparison_id TEXT, interpretation TEXT, next_step TEXT, outcome TEXT)"
    )
    cur.executemany("INSERT INTO research_log_entries VALUES (?,?,?,?,?,?)", log_entries)
    cur.execute(
        "CREATE TABLE session_events (event_id INTEGER, event_type TEXT, "
        "experiment_id TEXT, message TEXT, details_json TEXT)"
    )
    cur.executemany("INSERT INTO session_events VALUES (?,?,?,?,?)", session_events)
    conn.commit()
    conn.close()


def _make_telemetry(path: Path, model_calls, tool_calls, workflow_events,
                    checkpoints, checkpoint_has_tokens=False, wal=False):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE llm_model_calls (occurred_at TEXT, model TEXT, input_tokens INTEGER, "
        "cached_input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, "
        "workflow_event_id INTEGER)"
    )
    cur.executemany("INSERT INTO llm_model_calls VALUES (?,?,?,?,?,?,?)", model_calls)
    cur.execute(
        "CREATE TABLE llm_tool_calls (name TEXT, detail TEXT, started_at TEXT, "
        "completed_at TEXT, duration_ms REAL, status TEXT, success INTEGER, "
        "input_bytes INTEGER, output_bytes INTEGER, error_type TEXT, workflow_event_id INTEGER)"
    )
    cur.executemany(
        "INSERT INTO llm_tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)", tool_calls
    )
    cur.execute(
        "CREATE TABLE workflow_events (id INTEGER, command TEXT, started_at TEXT, "
        "completed_at TEXT, duration_ms REAL, status TEXT, error_type TEXT)"
    )
    cur.executemany("INSERT INTO workflow_events VALUES (?,?,?,?,?,?,?)", workflow_events)
    if checkpoint_has_tokens:
        cur.execute(
            "CREATE TABLE experiment_usage_checkpoints (experiment_id TEXT, "
            "experiment_name TEXT, status TEXT, completed_at TEXT, input_tokens INTEGER, "
            "cached_input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, "
            "total_cumulative INTEGER, model_calls INTEGER, tool_calls INTEGER, "
            "tool_failures INTEGER)"
        )
        cur.executemany(
            "INSERT INTO experiment_usage_checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            checkpoints,
        )
    else:
        # Token-less shape (like the real fixture) → forces LLM_USAGE.md fallback.
        cur.execute(
            "CREATE TABLE experiment_usage_checkpoints (experiment_id TEXT, "
            "experiment_name TEXT, status TEXT, completed_at TEXT)"
        )
        cur.executemany(
            "INSERT INTO experiment_usage_checkpoints VALUES (?,?,?,?)", checkpoints
        )
    conn.commit()
    conn.close()
    if wal:  # quirk 5: presence of a -wal sidecar must not stop read-only reads.
        path.with_name(path.name + "-wal").write_bytes(b"")
        path.with_name(path.name + "-shm").write_bytes(b"")


LLM_USAGE_D01 = """# LLM Usage by Step

| Step | Type | Status | Completed | Model | Effort | Input | Cached | Uncached | Output | Reasoning | Total (inc/cum) | Cache hit | Model calls (inc/cum) | Tool calls (inc/cum) | Tool failures (inc/cum) |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| global_mean_baseline | experiment | completed | 2026-01-01T00:01:00Z | — | — | 0 | 0 | 0 | 0 | 0 | 0/0 | - | 0/0 | 0/0 | 0/0 |
| lgbm_poisson_native_baseline | experiment | completed | 2026-01-01T00:02:00Z | m | medium | 1,000 | 800 | 200 | 50 | 10 | 1,050/1,050 | 80.0% | 3/3 | 4/4 | 0/0 |
| lgbm_tweedie_native_followup | experiment | completed | 2026-01-01T00:03:00Z | m | medium | 2,000 | 1,700 | 300 | 60 | 12 | 2,060/3,110 | 85.0% | 4/7 | 5/9 | 1/1 |
"""


def make_mini_fixture(repo_root: Path, orch_id: str = ORCH_ID) -> str:
    orch = repo_root / "artifacts" / "orchestrations" / orch_id
    (orch / "runs").mkdir(parents=True, exist_ok=True)
    (orch / "reports").mkdir(exist_ok=True)
    (orch / "briefs").mkdir(exist_ok=True)
    (orch / "prompts").mkdir(exist_ok=True)
    (orch / "logs").mkdir(exist_ok=True)
    (orch / "playoff").mkdir(exist_ok=True)

    # ---- top-level JSON artifacts ----
    ledger = {
        "orchestration_id": orch_id,
        "created_at": "2026-01-01T00:00:00Z",
        "dataset": "mini_motor",
        "target_mode": "frequency",
        "status": "completed",
        "model_provider": "openai",
        "model_name": "gpt-mini",
        "cycles_committed": 6,
        "total_cycle_budget": 6,
        "consolidation": {
            "run_id": CONS_RUN, "track": TRACK,
            "playoff_report": f"artifacts/orchestrations/{orch_id}/playoff/playoff_report.json",
        },
        "delegations": [
            {
                "delegation_id": "d01", "backend": "codex-gpt-mini", "track": TRACK,
                "run_id": "R01", "run_path": f"artifacts/orchestrations/{orch_id}/runs/d01",
                "command": ["codex", "exec"], "resolved_executable": "/bin/codex",
                "resolved_version": "0.1.0", "status": "completed", "clean_exit": True,
                "exit_code": 0, "spawned_at": "2026-01-01T00:00:30Z",
                "ended_at": "2026-01-01T00:04:00Z", "timeout_minutes": 30.0,
                "continue_run": False, "respawn_of": None, "taken_over": False,
                "cycle_budget": 3, "cycles_forfeited": 0, "budget_refunded": False,
                "brief_path": f"artifacts/orchestrations/{orch_id}/briefs/d01.json",
                "tool_usage": {"input_tokens": 3000, "cached_input_tokens": 2500,
                               "output_tokens": 110, "reasoning_output_tokens": 22},
            },
            {
                "delegation_id": "d02", "backend": "codex-gpt-mini", "track": TRACK,
                "run_id": "R02", "run_path": f"artifacts/orchestrations/{orch_id}/runs/d02",
                "command": ["codex", "exec"], "resolved_executable": "/bin/codex",
                "resolved_version": "0.1.0", "status": "completed", "clean_exit": True,
                "exit_code": 0, "spawned_at": "2026-01-01T00:04:30Z",
                "ended_at": "2026-01-01T00:07:00Z", "timeout_minutes": 30.0,
                "continue_run": False, "respawn_of": None, "taken_over": True,
                "cycle_budget": 3, "cycles_forfeited": 1, "budget_refunded": False,
                "brief_path": f"artifacts/orchestrations/{orch_id}/briefs/d02.json",
                "tool_usage": {"input_tokens": 5000, "cached_input_tokens": 4500,
                               "output_tokens": 120, "reasoning_output_tokens": 20},
            },
            {
                # quirk 6+7: crashed delegation, no tool_usage, unreadable registry.
                "delegation_id": "d03", "backend": "codex-gpt-mini", "track": TRACK,
                "run_id": "R03", "run_path": f"artifacts/orchestrations/{orch_id}/runs/d03",
                "command": ["codex", "exec"], "resolved_executable": "/bin/codex",
                "resolved_version": "0.1.0", "status": "crashed", "clean_exit": False,
                "exit_code": 1, "spawned_at": "2026-01-01T00:07:30Z",
                "ended_at": "2026-01-01T00:08:00Z", "timeout_minutes": 30.0,
                "continue_run": False, "respawn_of": None, "taken_over": False,
                "cycle_budget": 0, "cycles_forfeited": 0, "budget_refunded": True,
                "brief_path": f"artifacts/orchestrations/{orch_id}/briefs/d03.json",
                # NOTE: no "tool_usage" key → quirk 7 token fallback.
            },
        ],
    }
    _write_json(orch / "orchestration.json", ledger)
    _write_json(orch / "campaign_report.json", {
        "orchestration_id": orch_id, "dataset": "mini_motor", "target_mode": "frequency",
        "status": "completed", "created_at": "2026-01-01T00:00:00Z",
        "orchestrator": {"model_provider": "openai", "model_name": "gpt-mini"},
        "framework_computed": {"final_champion": {"gini_weighted": 0.29}},
    })
    _write_json(orch / "notes.json", {"notes": [
        {"timestamp": "2026-01-01T00:04:10Z", "kind": "reflection",
         "delegation_id": "d01", "text": "d01 built a strong baseline."},
        {"timestamp": "2026-01-01T00:05:00Z", "kind": "takeover",
         "delegation_id": "d02", "text": "Diagnosis-only takeover recovered d02."},
        {"timestamp": "2026-01-01T00:08:30Z", "kind": "decision",
         "delegation_id": None, "text": "Playoff selected the d01 champion."},
    ]})
    (orch / "CAMPAIGN_REPORT.md").write_text("# Mini Campaign\nfinal champion: EXP_A\n")
    (orch / "ORCHESTRATION_LOG.md").write_text("# Log\nspawned d01, d02, d03\n")

    # briefs
    _write_json(orch / "briefs" / "d01.json", {
        "_source_path": f"artifacts/orchestrations/{orch_id}/briefs/frequency_baseline.json",
        "direction": "Build a frequency baseline.",
        "constraints": ["Use recipes only.", "Stop after two rejects."],
        "cycle_budget": 3, "foundation_models": False,
        "starting_knowledge": ["First delegation."],
    })
    _write_json(orch / "briefs" / "d02.json", {
        "_source_path": f"artifacts/orchestrations/{orch_id}/briefs/structural_families.json",
        "direction": "Try a different estimator family.",
        "constraints": ["Beat the seeded champion."], "cycle_budget": 3,
        "foundation_models": False, "starting_knowledge": ["Seeded from d01."],
        "seed_champion": {"experiment_id": EXP_A, "from_run": f"{TRACK}/R01"},
    })
    _write_json(orch / "briefs" / "d03.json", {
        "_source_path": f"artifacts/orchestrations/{orch_id}/briefs/feature_probe.json",
        "direction": "Feature probe.", "constraints": [], "cycle_budget": 3,
        "foundation_models": False, "starting_knowledge": [],
    })

    # reports
    _write_json(orch / "reports" / "d01.json", {
        "delegation_id": "d01", "backend": "codex-gpt-mini",
        "agent_summary": "Established a strong LightGBM Poisson champion.",
        "champion": {"experiment_id": EXP_A, "gini_weighted": 0.30,
                     "rank_gini_weighted": 0.30, "asym_pricing_loss": 0.45,
                     "calibration_ratio": 0.99, "model_family": "recipe",
                     "target_strategy": "direct_pure_premium", "beat_seed_baseline": True},
        "cycles": {"attempted": 3, "budget": 3, "completed": 3, "decided": 2, "used": 3},
        "distress": {"active": [], "flags": ["all_rejected", "cycles_forfeited"],
                     "detail": None},
        "cost": {"llm_usage": {"cost_usd": 0.12}, "wall_clock_minutes": 3.5},
    })
    _write_json(orch / "reports" / "d02.json", {
        "delegation_id": "d02", "backend": "codex-gpt-mini",
        "agent_summary": "XGBoost was inferior to the seeded champion.",
        "champion": {"experiment_id": SEED_D02, "gini_weighted": 0.30,
                     "rank_gini_weighted": 0.30, "asym_pricing_loss": 0.45,
                     "calibration_ratio": 0.99, "model_family": "recipe",
                     "target_strategy": "direct_pure_premium", "beat_seed_baseline": True},
        # quirk 1: attempted > decided.
        "cycles": {"attempted": 2, "budget": 3, "completed": 1, "decided": 1, "used": 1},
        "distress": {"active": ["all_rejected", "cycles_forfeited"],
                     "flags": ["all_rejected", "cycles_forfeited"],
                     "detail": "1 decided cycle produced no promotion; 2 attempted."},
        "repairs": {"max_attempts_in_a_cycle": 2, "total_attempts": 2},
        "cost": {"llm_usage": {"cost_usd": 0.20}, "wall_clock_minutes": 2.5},
    })
    # d03 report intentionally absent (crashed) → degrade.

    for d in ("d01", "d02", "d03"):
        (orch / "prompts" / f"{d}.md").write_text(f"# Prompt {d}\nrun the delegation\n")
        (orch / "logs" / f"{d}.stdout.log").write_text(f"log line for {d}\n" * 5)
        _write_json(orch / "logs" / f"{d}.exit.json", {"exit_code": 0 if d != "d03" else 1})

    # ---- per-run dirs ----
    _build_d01(orch)
    _build_d02(orch)
    _build_d03(orch)
    _build_consolidation(repo_root)

    # ---- playoff ----
    _write_json(orch / "playoff" / "playoff_report.json", {
        "orchestration_id": orch_id, "decision_mode": "auto", "status": "completed",
        "consolidation": {"run_id": CONS_RUN, "track": TRACK},
        "exclusions": [],
        "finalists": [
            {"delegation_id": "d01", "gini_weighted": 0.30, "model_family": "recipe",
             "source": {"delegation_id": "d01", "experiment_id": EXP_A,
                        "run_id": "R01", "track": TRACK}},
            {"delegation_id": "d02", "gini_weighted": 0.30, "model_family": "recipe",
             "source": {"delegation_id": "d02", "experiment_id": SEED_D02,
                        "run_id": "R02", "track": TRACK}},
        ],
        "final_champion_lineage": {
            "consolidation_experiment_id": CONS_SEED,
            "source": {"delegation_id": "d01", "experiment_id": EXP_A,
                       "run_id": "R01", "track": TRACK},
        },
    })
    (orch / "playoff" / "playoff_report.md").write_text("# Playoff\nfinal: d01 EXP_A\n")
    return orch_id


def _build_d01(orch: Path) -> None:
    run = orch / "runs" / "d01"
    run.mkdir(exist_ok=True)
    experiments = [
        (GM, "2026-01-01 00:01:00", "global_mean_baseline", "completed",
         "global_mean", "direct_pure_premium", None, "", 0.1),
        (EXP_A, "2026-01-01 00:02:00", "lgbm_poisson_native_baseline", "completed",
         "recipe", "direct_pure_premium", GM, "", 7.5),
        (EXP_B, "2026-01-01 00:03:00", "lgbm_tweedie_native_followup", "completed",
         "recipe", "direct_pure_premium", EXP_A, "", 2.8),
    ]
    proposals = [
        (EXP_A, json.dumps({**_recipe("poisson", "native_categorical"),
                            "exploration_axis": "objective", "approach_family": "gbm"}),
         "Switch to LightGBM Poisson.", "Strong ranking.", "Overfit.", "Baseline probe."),
        (EXP_B, json.dumps(_recipe("tweedie", "native_categorical")),
         "Try Tweedie objective.", "Maybe better tails.", "Worse gini.", "Objective test."),
    ]
    comparisons = [
        ("CMP1", "2026-01-01 00:02:30", GM, EXP_A,
         json.dumps({"mean_lift": 0.30}), "promote", "llm", "clear_win",
         "Clean win over global mean.", "passed"),
    ]
    champion_history = [
        (1, "2026-01-01 00:01:00", None, GM, "initialised", None),
        (2, "2026-01-01 00:02:30", GM, EXP_A, "promoted", "CMP1"),
        (3, "2026-01-01 00:03:30", EXP_A, EXP_A, "retained", None),
    ]
    nodes = [
        ("n_a", None, EXP_A, "lgbm_frequency", "Establish a Poisson baseline.",
         "Switch to LightGBM Poisson.", "Strong ranking.", "Overfit.",
         _screening(0.30, GM, 0.001, 0.299, 1.0),
         json.dumps({"cv_mean_lift": 0.30, "cv_win_rate": 1.0})),
        ("n_b", "n_a", EXP_B, "lgbm_frequency", "Isolate objective choice.",
         "Try Tweedie.", "Maybe tails.", "Worse gini.",
         _screening(0.28, EXP_A, 0.30, -0.02, 0.1), None),
    ]
    log_entries = [
        (1, EXP_A, "CMP1", "Poisson captures strong signal.",
         "Hold native categoricals.", "promote: clean win over global mean."),
        (2, EXP_B, None, "Tweedie lost to Poisson.", "Keep Poisson.",
         "auto_reject: below low hurdle"),
    ]
    session_events = [
        (1, "repair_requested", EXP_B, "recipe repair",
         json.dumps({"repair_kind": "recipe", "attempt": 1,
                     "failed_checks": ["calibration_sane"], "resolved": True})),
    ]
    _make_registry(run / "registry.sqlite", experiments, proposals, comparisons,
                   champion_history, nodes, log_entries, session_events)
    # token-less checkpoints → LLM_USAGE.md fallback (quirk in DATA.md §2.6 note).
    _make_telemetry(
        run / "telemetry.sqlite",
        model_calls=[
            ("2026-01-01T00:01:30Z", "m", 500, 400, 20, 5, 1),
            ("2026-01-01T00:02:10Z", "m", 800, 700, 30, 8, 1),
        ],
        tool_calls=[
            ("exec", "const", "2026-01-01T00:01:31Z", "2026-01-01T00:01:32Z", 1000.0,
             "completed", None, 100, 200, None, 1),
            ("wait", None, "2026-01-01T00:02:00Z", "2026-01-01T00:02:05Z", 5000.0,
             "completed", None, 0, 0, None, 1),
        ],
        workflow_events=[
            (1, "run-session-cycles 1", "2026-01-01T00:01:30Z", "2026-01-01T00:02:30Z",
             60000.0, "completed", None),
        ],
        checkpoints=[
            (GM, "global_mean_baseline", "completed", "2026-01-01T00:01:00Z"),
            (EXP_A, "lgbm_poisson_native_baseline", "completed", "2026-01-01T00:02:00Z"),
            (EXP_B, "lgbm_tweedie_native_followup", "completed", "2026-01-01T00:03:00Z"),
        ],
        checkpoint_has_tokens=False,
        wal=True,  # quirk 5
    )
    (run / "LLM_USAGE.md").write_text(LLM_USAGE_D01)
    (run / "RESEARCH_LOG.md").write_text("# Research Log d01\ncycle 1: promote\n")
    _write_json(run / "run_manifest.json", {
        "delegation_id": "d01", "run_id": "R01",
        "model_identity": {"provider": "openai", "name": "gpt-mini",
                           "harness": "codex", "version": ""},
    })


def _build_d02(orch: Path) -> None:
    run = orch / "runs" / "d02"
    run.mkdir(exist_ok=True)
    experiments = [
        (GM, "2026-01-01 00:04:40", "global_mean_baseline", "completed",
         "global_mean", "direct_pure_premium", None, "", 0.1),
        (SEED_D02, "2026-01-01 00:05:00",
         "orchestration_delegation_seed_d02_lgbm_poisson_native_baseline", "completed",
         "recipe", "direct_pure_premium", GM, "", 7.5),
        (EXP_C, "2026-01-01 00:05:30", "xgb_poisson_ordinal_root", "completed",
         "recipe", "direct_pure_premium", SEED_D02, "", 5.0),
        # quirk 1: registered but never decided → no comparison, no log entry.
        (EXP_UND, "2026-01-01 00:06:00", "forfeited_undecided", "completed",
         "recipe", "direct_pure_premium", SEED_D02, "", 0.0),
    ]
    proposals = [
        (SEED_D02, json.dumps(_recipe("poisson", "native_categorical")),
         "Seeded champion.", "Carry forward.", "None.", "Seed."),
        (EXP_C, json.dumps(_recipe("poisson", "ordinal", estimator="xgboost")),
         "Try XGBoost.", "Different family.", "May be inferior.", "Family test."),
        (EXP_UND, json.dumps(_recipe("poisson", "one_hot")),
         "Attempted but forfeited.", "n/a", "n/a", "Forfeited."),
    ]
    comparisons = [
        ("CMP2", "2026-01-01 00:05:45", SEED_D02, EXP_C,
         json.dumps({"mean_lift": -0.008}), "reject", "llm", "inferior",
         "XGBoost is reliably inferior.", "passed"),
    ]
    champion_history = [
        (1, "2026-01-01 00:04:40", None, GM, "initialised", None),
        (2, "2026-01-01 00:05:00", GM, SEED_D02, "seeded", None),
        (3, "2026-01-01 00:05:45", SEED_D02, SEED_D02, "retained", "CMP2"),
    ]
    nodes = [
        ("n_c", None, EXP_C, "xgb_frequency", "Probe XGBoost family.",
         "Try XGBoost.", "Different family.", "Inferior.",
         _screening(0.29, SEED_D02, 0.30, -0.01, 0.0),
         json.dumps({"cv_mean_lift": -0.008, "cv_win_rate": 0.0})),
    ]
    log_entries = [
        (1, EXP_C, "CMP2", "XGBoost inferior to seed.", "Return to LightGBM.",
         "reject: reliably inferior"),
    ]
    _make_registry(run / "registry.sqlite", experiments, proposals, comparisons,
                   champion_history, nodes, log_entries)
    _make_telemetry(
        run / "telemetry.sqlite",
        model_calls=[("2026-01-01T00:05:10Z", "m", 1200, 1080, 40, 8, None)],
        tool_calls=[
            ("exec", None, "2026-01-01T00:05:11Z", "2026-01-01T00:05:12Z", 900.0,
             "failed", 0, 10, 0, "nonzero_exit", None),
        ],
        workflow_events=[],
        checkpoints=[
            (SEED_D02, "orchestration_delegation_seed_d02_lgbm_poisson_native_baseline",
             "completed", "2026-01-01T00:05:00Z", 0, 0, 0, 0, 0, 1, 1, 0),
            (EXP_C, "xgb_poisson_ordinal_root", "completed",
             "2026-01-01T00:05:30Z", 1200, 1080, 40, 8, 1328, 1, 1, 1),
        ],
        checkpoint_has_tokens=True,  # exercises the in-DB token path
    )
    (run / "RESEARCH_LOG.md").write_text("# Research Log d02\ncycle 1: reject\n")
    _write_json(run / "run_manifest.json", {
        "delegation_id": "d02", "run_id": "R02",
        "model_identity": {"provider": "openai", "name": "gpt-mini",
                           "harness": "codex", "version": ""},
    })


def _build_d03(orch: Path) -> None:
    # quirk 6: unreadable/partial registry → not a valid sqlite file.
    run = orch / "runs" / "d03"
    run.mkdir(exist_ok=True)
    (run / "registry.sqlite").write_bytes(b"not a database -- partially written\x00")
    # quirk 7: telemetry present with model calls to sum for the token fallback.
    _make_telemetry(
        run / "telemetry.sqlite",
        model_calls=[
            ("2026-01-01T00:07:40Z", "m", 700, 600, 25, 6, None),
            ("2026-01-01T00:07:50Z", "m", 300, 250, 10, 3, None),
        ],
        tool_calls=[],
        workflow_events=[],
        checkpoints=[],
        checkpoint_has_tokens=False,
    )
    (run / "run_manifest.json").write_text("{ this is : not valid json ]")  # degrade


def _build_consolidation(repo_root: Path) -> None:
    cons = repo_root / "artifacts" / "tracks" / TRACK / "runs" / CONS_RUN
    cons.mkdir(parents=True, exist_ok=True)
    experiments = [
        (CONS_GM, "2026-01-01 01:01:00", "global_mean_baseline", "completed",
         "global_mean", "direct_pure_premium", None, "", 0.1),
        (CONS_SEED, "2026-01-01 01:02:00",
         "orchestration_seed_d01_lgbm_poisson_native_baseline", "completed",
         "recipe", "direct_pure_premium", CONS_GM, "", 7.5),
        (CONS_CH, "2026-01-01 01:03:00",
         "orchestration_challenger_02_d02_xgb_poisson_ordinal_root", "completed",
         "recipe", "direct_pure_premium", CONS_SEED, "", 7.5),
        # quirk 3: same id as a delegation experiment (EXP_A) → must dedup.
        (EXP_A, "2026-01-01 01:04:00", "lgbm_poisson_native_baseline", "completed",
         "recipe", "direct_pure_premium", CONS_SEED, "", 7.5),
    ]
    proposals = [
        (CONS_CH, json.dumps(_recipe("poisson", "native_categorical")),
         "Replay d01 champion.", "Confirm.", "None.", "Replay."),
    ]
    comparisons = [
        ("CCMP1", "2026-01-01 01:03:30", CONS_SEED, CONS_CH,
         json.dumps({"mean_lift": 0.0}), "reject", "auto", "noise",
         "No improvement over seed.", "passed"),
    ]
    champion_history = [
        (1, "2026-01-01 01:01:00", None, CONS_GM, "initialised", None),
        (2, "2026-01-01 01:02:00", CONS_GM, CONS_SEED, "seeded", None),
        (3, "2026-01-01 01:03:30", CONS_SEED, CONS_SEED, "retained", "CCMP1"),
    ]
    nodes = [
        ("cn", None, CONS_CH, "consolidation", "Replay d01 champion.",
         "Replay.", "Confirm.", "None.",
         _screening(0.30, CONS_SEED, 0.30, 0.0, 0.5),
         json.dumps({"cv_mean_lift": 0.0, "cv_win_rate": 0.5})),
    ]
    _make_registry(cons / "registry.sqlite", experiments, proposals, comparisons,
                   champion_history, nodes, [])
    _make_telemetry(
        cons / "telemetry.sqlite",
        model_calls=[("2026-01-01T01:02:10Z", "m", 400, 350, 15, 4, None)],
        tool_calls=[], workflow_events=[],
        checkpoints=[(CONS_CH, "orchestration_challenger_02_d02_xgb_poisson_ordinal_root",
                      "completed", "2026-01-01T01:03:00Z")],
        checkpoint_has_tokens=False,
    )
    (cons / "LLM_USAGE.md").write_text(
        "# LLM Usage by Step\n\n"
        "| Step | Type | Status | Completed | Model | Effort | Input | Cached | "
        "Uncached | Output | Reasoning | Total (inc/cum) | Cache hit | "
        "Model calls (inc/cum) | Tool calls (inc/cum) | Tool failures (inc/cum) |\n"
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| orchestration_challenger_02_d02_xgb_poisson_ordinal_root | experiment | "
        "completed | 2026-01-01T01:03:00Z | m | medium | 400 | 350 | 50 | 15 | 4 | "
        "419/419 | 87.5% | 1/1 | 0/0 | 0/0 |\n"
    )


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


if __name__ == "__main__":  # manual smoke build into a scratch dir
    import sys
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fd_mini")
    make_mini_fixture(target)
    print(f"mini fixture at {target}")
