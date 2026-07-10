# Architecture

## Module Map

```
src/autoresearch/
├── config.py                  # ProjectConfig frozen dataclass (all settings)
├── datasets.py                # DatasetSpec registry — configs/datasets/<name>.toml
├── targets.py                 # TargetSpec generation (build_target_spec) + dataset binding
├── experiment_runner.py       # Load data → dispatch model → evaluate → persist
├── comparison_runner.py       # Paired comparison → promotion decision
├── data/
│   ├── pipeline.py            # prepare-data: load, sample, cap, split, fold assignments
│   ├── loaders/single_table.py# generic single-table loader (na handling, derived cols)
│   ├── adapters/french_motor.py # freMTPL2 freq/sev two-file adapter
│   ├── splits.py              # split pack + fold assignments (group-aware, stratified)
│   └── holdout_vault.py       # write_vault / load_search_dataset / load_holdout_dataset
├── models/
│   ├── dispatcher.py          # dispatch_model() → ModelResult
│   └── global_mean.py         # built-in global-mean baseline
├── evaluation/
│   ├── metrics.py             # full_metric_panel(), evaluate_predictions()
│   ├── diagnostics.py         # compute_diagnostics() — calibration, PSI, segments
│   └── resampling.py          # paired_comparison(), promotion_decision(),
│                              #   bootstrap_lift_summary(), cv_repeated_scores()
├── experiment_registry/
│   └── registry.py            # SQLite: record_experiment, set_official_champion, …
├── controller/
│   ├── proposal_schema.py     # allowed_search_space(), validate_proposal()
│   ├── proposer.py            # FileProposer (file-handoff inbox)
│   ├── workflow.py            # enqueue_proposal_from_file(), run_next_queued_proposal()
│   ├── champion.py            # initialise_official_champion()
│   ├── context.py             # build_llm_context() — gated memory_access block
│   ├── session.py             # run_session_cycle() — every-5-cycle checkpoint hook
│   └── handoff.py             # export_context(), write_proposal_template()
├── memory/
│   ├── __init__.py            # maybe_memory_checkpoint(), resolve_memory_access()
│   ├── store.py               # init_memory_store(), assert_no_holdout_columns()
│   ├── harvester.py           # harvest_run(), harvest_all() — read-only, search splits only
│   ├── insights.py            # validate_insight(), record_insight(), list_insights()
│   ├── query.py               # query_insights(), run_analysis() — access-gated
│   └── playbook.py            # build_playbook() — verified insights → latest.md
├── orchestration/
│   ├── manifest.py            # Orchestration/Delegation records + manifest lock
│   ├── backends.py            # Backend registry — configs/orchestration/backends.toml
│   ├── brief.py               # Brief validation + the handoff's Orchestration block
│   ├── spawner.py             # Pre-bootstrap child run, compose prompt, launch process
│   ├── adapters.py            # Per-tool clean-exit detection and usage parsing
│   ├── monitor.py             # Liveness/progress polling, wall-clock timeouts
│   ├── report.py              # Delegation report + distress predicates (from registry)
│   ├── playoff.py             # Consolidation: replay finalists, rank, promote
│   ├── campaign_log.py        # orchestrate note → ORCHESTRATION_LOG.md
│   ├── campaign_report.py     # orchestrate report → CAMPAIGN_REPORT.md
│   └── stats.py               # orchestrate backend-stats — cross-campaign scorecard
├── utils/
│   └── environment.py         # capture_environment() — git SHA, pip freeze, SHA256s
└── dashboard/
    └── app.py                 # Streamlit dashboard (incl. Memory & Leaderboard page)
```

## High-Level Flow

```
prepare-data
  → load configured source data → retain source columns → compute capping diagnostics
  → reuse or create split pack (train/sv/holdout)
  → write agent_dataset_search.parquet (no holdout rows; raw uncapped target)
  → write holdout_vault/agent_dataset_holdout.parquet (token-gated)
  → write split_pack_folds.parquet (5-fold CV assignments)
  (the fixed claim cap is applied uniformly at scoring time, not baked into the
   persisted artifacts, so search/holdout stay a single canonical source)

run-baseline / run-next-proposal
  → load_search_dataset()  ← only train + search_validation rows visible
  → dispatch_model()       ← routes to global_mean built-in or a run-local script
  → evaluate_predictions() ← target-aware metric panel + Gini + double-lift
  → compute_diagnostics()  ← calibration by pred decile, PSI, segment ratios
  → capture_environment()  ← git SHA, pip freeze, file SHA256s
  → record_experiment()    ← SQLite registry

compare-experiments / compare-to-champion
  → paired_comparison()    ← bootstrap lift distribution, Bonferroni-adjusted CI
  → promotion_decision()   ← 8-check gate: relative lift, calibration, win rate, …
  → set_official_champion() if promoted
```

## Cross-Run Memory Aggregator

A separate read-only harvesting layer accumulates search-split experiment results across runs into a cross-run aggregator that lives **outside the repo working tree** (default `~/.autoresearch/<project>/memory/memory.sqlite`, overridable with `AUTORESEARCH_MEMORY_DIR`). Keeping it out of the working tree means per-run agents cannot reach other runs' results through the aggregator. The memory access gate is a separate control that governs the query tool and context/handoff injection. Raw reads of *other runs' in-repo folders* are blocked independently by the run-scope guard (see below). Per-run registries are never written to; the harvester opens them `mode=ro`.

```
artifacts/tracks/<track>/runs/<run-id>/registry.sqlite   (per-run, isolated, in repo)
        | read-only harvest (every 5 cycles + autoresearch memory harvest)
        v
~/.autoresearch/<project>/memory/memory.sqlite    (aggregator, OUTSIDE the repo:
                                   models, runs, experiments, comparisons, insights)
~/.autoresearch/<project>/memory/playbook/latest.md   (compiled verified insights)
```

**Key invariants:**
- Only `ordinary_eval_splits` (search-split) metrics are stored — train and holdout splits are excluded at harvest time.
- `AUTORESEARCH_MEMORY_ACCESS` (env var) controls agent access: `none` (default, context unchanged), `own` (filtered to own model), `all` (all models, attributed). The agent cannot set this itself.
- Model identity (`--model-provider`, `--model-name`) is required at `bootstrap-track` so results are attributed to a model in the aggregator.
- The every-5-cycle checkpoint also writes `pending_reflection.md` to prompt the agent to record evidence-bound insights, and regenerates the playbook when new verified insights land.

## Orchestration

A campaign splits research into taste and mechanics. An interactive
**orchestrator** conversation plans directions and reads results; headless
**sub-agents** spawned by `autoresearch orchestrate spawn` execute experiment
cycles. A sub-agent run *is* a normal tracked run — same contract, same proposal
inbox, same `run-session-cycles`/`record-decision`, same repair flow. Orchestrated
mode adds only three things: the run is pre-bootstrapped by the spawner, its
handoff carries an **Orchestration brief** block (rendered only when the run
manifest has an `orchestration_id`, so single-agent runs are byte-identical to
before), and a report is generated from its registry when it ends. That report is
built from the child's registry and predictions artifact, never from the
sub-agent's claims; the `finish-delegation` summary is stored verbatim as
testimony and never feeds a computed number.

Campaign state lives in `artifacts/orchestrations/<orchestration-id>/`:
`orchestration.json` (the manifest — the single source of truth for which child
runs belong to the campaign, which the run-scope guard reads), `briefs/`,
`prompts/`, `logs/`, `reports/`, `playoff/`, `notes.json`, the generated
`ORCHESTRATION_LOG.md`, and the final `campaign_report.json` +
`CAMPAIGN_REPORT.md`. Mutations happen under a manifest lock so parallel detached
spawns cannot race on bootstrap or on the delegation list.

Parallel delegations are independent tracked runs, so nothing changes about
sequential comparison semantics inside a run. Cross-run consolidation is a
`playoff`: finalists are replayed into a fresh consolidation run in ascending
`gini_weighted` order and re-tested head-to-head under the standard gates, so the
orchestration champion is a fresh statistical result rather than a cross-run
eyeball. Like `tracks.py`, the whole package is a *caller* of public framework
entry points — it never touches the integrity-protected evaluation files, and the
consolidation champion's promotion fires the ordinary protected holdout
evaluation.

Backend choice is registry-driven. `configs/orchestration/backends.toml` is the
human-owned list of spawnable model × effort combinations, carrying selection
metadata (`tier`, `good_for`, `cost_hint`, `status`). `orchestrate backend-stats`
aggregates collected delegation reports across campaigns into an empirical
scorecard — promotion rate, distress rate by flag, repair attempts per cycle,
mean promoted Gini lift, cost per cycle and per promotion — which `list-backends`
appends beneath each entry. The scorecard is evidence only: it never adds or
removes a spawnable backend.

## Run-Scope Guard

Within the repo working tree, runs are physically siblings under `artifacts/tracks/<track>/runs/<run-id>/`. The framework's own commands are already run-scoped, but an agent's *free-form* file access (shell `cat`/`grep`, file reads/edits) could still reach into a sibling run's folder and contaminate an otherwise independent experiment. A harness-level guard closes that channel.

One shared decision script (`scripts/run_scope_guard.py`) is wired into all three agent harnesses as a pre-tool-use hook:

- **Claude Code** — `.claude/settings.json` (stdin payload, exit code `2` denies)
- **Codex** — `.codex/hooks.json` (same stdin/exit-code contract)
- **OpenCode** — `.opencode/plugins/run-scope-guard.js` (a thin adapter that shells out to the same script and `throw`s to deny)

Policy:

- A session is **unbound** until it runs its first valid `autoresearch --track …` command, at which point it is automatically **bound** to exactly that run. Binding is keyed on the harness session id, so parallel runs in separate threads stay independent.
- A **bound research** session is blocked from reading, grepping, or listing any *other* run's folder — in any track, including its own track's siblings — and from enumerating the `runs/` directory. Its own run, `src/`, data, configs, and tests stay fully accessible. The block is enforced by the harness and cannot be overridden by the model.
- An **unbound** session may inspect source, docs, configs, data, and tests, but raw run artifacts under `artifacts/tracks/<track>/runs` are blocked until binding. Analyst mode is requested with `AUTORESEARCH_SCOPE=analyst` in the launch environment and is the supported way to run a deliberate cross-run analysis thread.
- An **orchestrator** session (`AUTORESEARCH_SCOPE=orchestrator`, or auto-bound on a successful `orchestrate new`/`orchestrate spawn`) may read its own campaign folder, every child run listed in that campaign's manifest, and the consolidation run — and may drive those runs by explicit `--run-id` for takeover. The child list is re-read from `orchestration.json` on every hook invocation, so newly spawned children are covered immediately. Other orchestrations' folders, runs outside the manifest, implicit run selection, and `runs/` enumeration are denied. Sub-agents are unaffected: they bind as ordinary `research` sessions from the environment the spawner gives them.
- Because run artifacts are blocked before binding, a research agent must bootstrap before inspecting its own run outputs.

Cross-run knowledge therefore reaches a research agent only through the memory aggregator (when enabled), never through raw reads. Session scope files and the guard log live under `artifacts/tracks/.scope/` (gitignored). The guard **fails open**: any internal error allows the call, so a guard bug can never block legitimate research.

## Holdout Vault

The milestone holdout is architecturally separated from the search partition. `experiment_runner.py` calls `load_search_dataset()` which reads only `agent_dataset_search.parquet` — a file that never contains holdout record IDs. The holdout file lives in a separate per-dataset directory (`data/datasets/<name>/holdout_vault/`) with a `.locked` sentinel. Reading it requires the `AUTORESEARCH_MILESTONE_TOKEN` environment variable. This makes accidental holdout contamination fail loudly rather than silently.

## Model Layer

An experiment supplies its model in one of two ways:

1. **Declarative recipe (preferred).** `experiment_config.model.recipe` is a
   small validated object (estimator, objective, encoding, params, structure)
   interpreted by trusted code in `autoresearch.models.recipe`. A curated,
   decorator-extensible registry (`estimators.py`, `encoders.py`) covers the
   common search space; the validity matrix in `schema.py` rejects illegal
   combinations before anything runs. `model_family = "recipe"`.
2. **Run-local script (escape hatch).** A proposal points
   `experiment_config.model.script_path` at a Python file beside the proposal
   exposing `fit_predict(train, score, *, feature_inclusions=None,
   feature_exclusions=None, **hyperparameters)`. Used for novel models the
   recipe vocabulary cannot express.

A model returns **either** a raw `np.ndarray` of target totals (legacy contract:
it owns its own exposure conversion and calibration) **or** a
`Prediction(values, unit)` (from `autoresearch.models.prediction`). For a
`Prediction`, the framework owns the safety-critical bookkeeping (#7): rate→total
conversion via exposure, single-scalar training calibration, native-bias
recording, and prediction validation. Recipes always return a `Prediction`, so
both paths share one finalisation stage in `dispatch_model()`.

The framework still owns data loading, split application, capping, evaluation,
comparison, and registry writes. The `global_mean` no-model baseline is the
built-in bootstrap starting point.

`dispatch_model()` in `models/dispatcher.py` routes `model_family` (recipe /
global_mean / open registry / script), finalises `Prediction` returns,
constructs the prediction DataFrame, and asserts row counts to catch silent
drops. Promoted recipes are saved to a run-local (or, under memory access,
cross-run) ledger and re-emitted as `champion_template.py` for cheap follow-ups.

The active dataset is selected per run with `--dataset` and pinned in the run
manifest; a `DatasetSpec` (`configs/datasets/<name>.toml`, loaded by
`autoresearch.datasets`) supplies column roles, weight/exposure, capping, split
grouping, and the available target modes, and is bound into the target/column/
feature-policy modules at config load. The active target is the dataset's
default mode or the CLI `--target-mode` override; each mode's `TargetSpec`
(generated by `targets.build_target_spec`) defines its source column, weight,
and population (e.g. French `burning_cost` interprets outputs as claim costs,
`frequency` as expected claim counts, Porto `claim_incidence` as claim
probabilities).

## Output Validation

After each queued proposal run, the controller writes `validation_report.json`
before any promotion comparison. The report checks that predictions are finite,
non-negative, non-empty, have sensible aggregate scale, produce a finite primary
metric, and show positive lift against the current champion. Failed checks write
`repair_request_2.json` or `repair_request_3.json`; the agent can revise the
next script attempt up to three total attempts before the proposal is marked as
failed.

## Evaluation and Promotion

**Primary metric**: configured in `[evaluation]` and currently defaults to
`gini_weighted`. Burning-cost runs also record Tweedie deviance on pure premium;
frequency runs record Poisson deviance on claim frequency.

**Promotion gate** (all 8 checks must pass):
1. Mean lift > 0 (challenger improves the configured primary metric)
2. Relative lift ≥ `min_relative_lift` (default 0.5%) — prevents noise promotions
3. Absolute lift ≥ `min_absolute_lift`
4. Challenger win rate ≥ `minimum_win_rate` across paired resamples
5. Bootstrap 90% CI lower bound ≥ `bootstrap_lower_bound`
6. Bootstrap 90% CI relative lower bound ≥ `bootstrap_lower_bound_relative`
7. Calibration OK (all predicted decile ratios in [0.3, 3.0])
8. Diagnostics present (when `require_diagnostics = true`)

The CI is Bonferroni-adjusted using `bonferroni_lookback` to account for multiple comparisons across the autonomous search history.

## Reproducibility

Every experiment artifacts folder includes `environment_manifest.json` capturing: Python version, platform, git SHA, git dirty flag, pip freeze output, key dependency versions (numpy, pandas, sklearn, pyarrow), and SHA256 hashes of the input data files. This makes any result reproducible given the captured environment — the manifest records the exact resolved versions even though `pyproject.toml` only lower-bounds them (there is no lockfile), so a faithful rerun means recreating the environment the manifest describes.

## K-Fold CV

5-fold fold assignments are generated deterministically from `record_id` via a stable hash (`stable_unit()`). When `use_cv = true`, `cv_repeated_scores()` runs `n_folds` train/val splits and returns a variance decomposition: `between_fold_variance` measures how much the score varies across folds, `within_fold_variance` measures residual noise within each fold, and `total_variance` is their sum. A `warning_between_dominates` flag fires when between-fold variance exceeds 70% of total — this indicates potential leakage, unstable feature engineering, or severe distributional shift across folds.

## Constraints

- Local Python project; no distributed compute
- Streamlit dashboard for interactive inspection
- No holdout access during ordinary search (enforced architecturally)
- Every experiment is fully resumable and reproducible from its `environment_manifest.json`
- Promotion decisions are non-reversible in the registry (append-only history)
