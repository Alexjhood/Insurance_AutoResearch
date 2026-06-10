# Specification — Compute Budgets, Timing Telemetry & Open-Model Robustness

**Audience:** an implementing engineer (Claude Sonnet) working in the
`Insurance_AutoResearch` repo.
**Repo root:** `/Users/alexhood/Documents/Insurance_AutoResearch`
**Status:** ready to implement. Two independent parts; Part 1 and Part 2 can be
landed separately.

---

## Background / context

This is an autonomous insurance burning-cost modelling loop on the French Motor
dataset (~678K policies). An LLM agent proposes model scripts; the framework
fits them, evaluates them with a `cv_bootstrap` gate, and the agent records a
promote/reject decision. Multiple agent "tracks" run the same loop (`claude`,
`codex`, `opencode`, …), each storing runs under
`artifacts/tracks/<track>/runs/<run-id>/`.

Two problem classes motivate this spec, both observed empirically:

1. **Unbounded compute.** There is no timeout anywhere in the package
   (`grep -rE "timeout|TimeoutError|n_iter_no_change|early_stopping" src/autoresearch/`
   returns nothing on the cycle/comparison path). A single proposal with
   `n_estimators=5000, learning_rate=0.003` and no early stopping ran for
   ~30+ minutes. It is made worse because the **challenger is refit ~5×** (one
   experiment fit + 4 cv folds; champion folds are cached, challenger is not),
   so "5000 trees" is really ~25,000 tree builds over 430K rows.

2. **Open models repeatedly trip on the same errors.** Across the saved
   `codex` / `opencode` runs the recurring failures are: applying Tweedie/gamma
   losses to estimators that don't support them
   (`HistGradientBoostingRegressor.__init__() got an unexpected keyword argument
   'tweedie_power'`; `loss='gamma' requires strictly positive y`), unencoded
   categoricals (`could not convert string to float: 'B12'`), a framework
   doubled-path bug (`.../artifacts/tracks/artifacts/tracks/...`), blend
   components missing predictions, repair attempts asymptoting to exactly `0`
   lift, and orphaned `running` proposals after crashes.

---

## Key files (verified locations)

All paths/symbols below are **verified** against the current tree.

| Concern | File / symbol |
|---|---|
| Per-experiment fit entrypoint | `src/autoresearch/experiment_runner.py` → `run_experiment()` (def ~line 37). The actual model fit happens inside `dispatch_model(...)` (~line 167), called from `run_experiment`. The metrics payload (`metrics_payload`) is assembled **inline** ~line 216 and written with `write_json(metrics_path, metrics_payload)` ~line 245. There are **no** `_build_metrics_payload`/`_write_metrics_outputs` helpers — add the `timing` block to the inline `metrics_payload` dict. |
| Experiment-row persistence | `src/autoresearch/experiment_registry/experiments.py` → `record_experiment()` (def ~line 14; keyword-only args). Called from `run_experiment` ~line 263. Add timing kwargs here (`fit_wall_seconds`, `fit_cpu_seconds`, `compute_budget_seconds`, `timed_out`). |
| Run-local model-script path resolution | `experiment_runner.py` → `_resolve_model_script_path(experiment_config_path, model_cfg)` (~line 119). This is where to fix the doubled-path bug (Part 2.5). |
| Cycle orchestration + repair loop | `src/autoresearch/controller/workflow.py` → `run_next_queued_proposal()` (~line 78), `_run_validated_experiment_attempts()` (~line 187, the `for attempt in range(1, 4)` loop), `_validate_attempt_outputs()` (~line 259), `_attach_failed_attempt_comparison()` (~line 286), `_write_repair_request()` (~line 337). `ExperimentNeedsRepair` is raised ~line 251. |
| Cycle CLI / handoff wrapper | `src/autoresearch/controller/handoff.py` → `run_latest_proposal_cycle()` (~line 187, calls `run_next_queued_proposal`). |
| Comparison runner | `src/autoresearch/comparison_runner.py` → calls `write_comparison_html_report(...)` (~line 259) to render `comparison_report.html`. |
| HTML report renderer | `src/autoresearch/reporting/comparison.py` → `write_comparison_html_report()` (def ~line 77; re-exported from `autoresearch.reporting`). This is where the new "Compute" section is added. |
| Registry schema (DDL) | `src/autoresearch/experiment_registry/schema.py` → `SCHEMA` string: `experiments` table DDL starts ~line 10, `comparisons` ~line 39. Add new columns to the `experiments` CREATE **and** add an idempotent `PRAGMA table_info` + `ALTER TABLE` migration at registry init (see 1.2). |
| Central config | `configs/default.toml`. Existing sections: `[paths]`, `[data]`, `[preprocessing]`, `[splits]`, `[evaluation]`, `[resampling]`, `[promotion]`. Add a new `[compute]` and `[repair]` section. |
| Operating manual (agent-facing) | `AGENT.md`. Relevant existing headers: `## Exploration philosophy` (~line 15), `### E. Repair policy — don't asymptote to zero` (~line 214), `### Step 2 — Implement` (~line 249), `### Step 4 — Run the experiment` (~line 352). |

> Line numbers are approximate — locate by symbol name, not by number.

---

# PART 1 — Compute budgets, timing telemetry, early-stopping guidance

## 1.1 Escalating per-experiment wall-clock budget

**Rule:** the budget starts at **10 minutes** and **increases by 5 minutes
every 5 experiments** within a run.

Define "experiment number" `N` as the count of experiments already run in the
current run (i.e. rows in the `experiments` table for this run's registry, or
the iteration index — pick the registry count for robustness). The budget for
the experiment about to run is:

```
budget_minutes = 10 + 5 * (N // 5)
```

So experiments 1–5 → 10 min, 6–10 → 15 min, 11–15 → 20 min, etc.
(`N` is zero-based count of *prior* experiments; the first experiment of a run
has `N=0` → 10 min.)

**Where to enforce:** wrap the `dispatch_model(...)` call inside
`run_experiment()` (`experiment_runner.py` ~line 167) — that is the single
chokepoint every attempt and every cv refit passes through. The budget value
should be computed once per *proposal cycle* in `workflow.py`
(`run_next_queued_proposal` / `_run_validated_experiment_attempts`) and threaded
down into `run_experiment` as a new parameter (e.g.
`compute_budget_sec: float | None = None`). Note `run_experiment` already takes
`config`, `experiment_config_path`, `output_dir` — add the budget as an optional
kwarg so existing callers (`run-baseline`, etc.) keep working with `None`.

Compute `N` (prior experiment count) from the registry — there is no existing
counter, so query the `experiments` table for this run's registry (e.g.
`SELECT COUNT(*) FROM experiments`). Do this in the cycle code, not inside
`run_experiment`.

**Important nuance — the budget covers the *experiment fit*, not the whole cv
comparison.** The challenger is refit ~5×. Decide and document one of:
- (Recommended) Budget applies to the **single experiment fit** in
  `run_experiment`. The cv comparison refits each get the same per-fit budget.
  Simpler and bounds the worst case per fit.
- Alternatively a whole-cycle budget — more complex, not required.

Implement the recommended option.

**Enforcement mechanism:** the model fit is CPU-bound Python calling into
native libs (lightgbm/xgboost/statsmodels), so a pure-Python timer cannot
interrupt it. Use one of:
- **`signal.SIGALRM`** (POSIX; this is a macOS/Linux project) set around the
  `model_module.fit_predict(...)` call. Simple, in-process, but only fires on
  the main thread and only between Python bytecode / library callback points.
  Acceptable for a first cut.
- **A subprocess with `timeout`** (more robust, can hard-kill native code) —
  better but heavier. Optional upgrade.

Start with `SIGALRM`. On expiry raise a `ComputeBudgetExceeded(Exception)` with
a message including the budget and elapsed time.

**On timeout:**
- Mark the experiment/attempt **failed** with a clear, structured reason:
  `"Compute budget exceeded: ran <elapsed>s, budget <budget>s. Reduce
  n_estimators / use early stopping / lower model complexity."`
- This must flow into the **repair request** (see Part 2.3) so the agent learns
  to back off, and into the proposal `notes` in the registry.
- Do **not** crash the whole cycle with an unhandled traceback — convert to the
  normal "needs repair / failed" path used by `_run_validated_experiment_attempts`.

**Config:** add a `[compute]` section to `configs/default.toml`:

```toml
[compute]
base_budget_minutes = 10
budget_increment_minutes = 5
experiments_per_increment = 5
enforce = true   # allow disabling for debugging
```

Read these in the cycle code; do not hard-code the constants.

## 1.2 Track and surface wall-clock + CPU time per experiment

**Measure** around the `fit_predict` call in `run_experiment()`:
- wall-clock: `time.perf_counter()` delta
- CPU time: `time.process_time()` delta (covers this process; note in code that
  child-thread CPU from `n_jobs=-1` native libs may not be fully captured by
  `process_time` — capture `time.thread_time()` too if cheap, but `perf_counter`
  is the authoritative budget metric).

Optionally also record peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
(nice-to-have, not required).

**Persist** these in three places:

1. **`metrics.json`** (add to the inline `metrics_payload` dict in
   `run_experiment`, ~line 216, before `write_json(metrics_path, ...)` at
   ~line 245): add a `timing` block:
   ```json
   "timing": {
     "fit_wall_seconds": 412.3,
     "fit_cpu_seconds": 2890.1,
     "compute_budget_seconds": 600,
     "budget_utilisation": 0.69,
     "timed_out": false
   }
   ```
2. **Registry `experiments` table**
   (`src/autoresearch/experiment_registry/schema.py`): add columns
   `fit_wall_seconds REAL`, `fit_cpu_seconds REAL`, `compute_budget_seconds REAL`,
   `timed_out INTEGER` to the `experiments` DDL. The schema uses
   `CREATE TABLE IF NOT EXISTS`, so for **existing** registries also add a small
   idempotent migration: on registry open (`init_registry` in
   `experiment_registry/schema.py`), `PRAGMA table_info(experiments)`, and
   `ALTER TABLE experiments ADD COLUMN ...` for any missing columns. Then
   populate the values via `record_experiment()` in
   `src/autoresearch/experiment_registry/experiments.py` (add matching
   keyword-only args). **Note:** `record_experiment` builds an explicit
   `INSERT OR REPLACE INTO experiments (...)` column list (~line 38) — you must
   add the new columns to that INSERT statement and its value bindings too, not
   just to the function signature. `run_experiment` passes the values at the
   call site ~line 263.
   (If you prefer zero schema change, the values are already inside the metrics
   payload JSON on disk — but explicit columns make reporting/querying much
   easier, so add them.)
3. **Comparison report** (`src/autoresearch/reporting/comparison.py` →
   `write_comparison_html_report()`, called from `comparison_runner.py` ~line
   259): add a small **"Compute"** row/section showing, for both champion and
   challenger where available: fit wall-clock, fit CPU time, the budget in
   force, % of budget used, and a ⚠️ flag if `timed_out`. The challenger's
   timing comes from its `metrics.json` `timing` block / registry row; the
   champion's from its stored row. You may need to thread the timing values
   into the renderer's call signature from `comparison_runner.py`.

## 1.3 Tell the agent about the budget + bias toward cheap experiments

Edit **`AGENT.md`**. In the "Exploration philosophy" section (and cross-ref from
the cheat sheet added in Part 2.1), add a **"Compute budget"** subsection
stating:

- There is a per-experiment wall-clock budget, currently
  `10 min, +5 min per 5 experiments` (reference the `[compute]` config so it
  stays truthful if changed).
- The challenger is **refit ~5×** per comparison (1 experiment fit + 4 cv
  folds), so effective cost ≈ 5× a single fit. Budget the single fit
  accordingly.
- **Sequencing guidance:** start with cheap, fast models (GLMs, shallow trees,
  small `n_estimators`) to map the signal, then escalate to higher-capacity
  models only once cheap ideas are exhausted. This dovetails with the existing
  "small steps, broad search" philosophy.
- Cost drivers to watch: `n_estimators × (1/learning_rate)`, `num_leaves` /
  `max_depth`, and dataset size. A "5000-tree, lr=0.003, no early stopping"
  model is ~25k tree builds and will likely blow the budget.

## 1.4 Encourage early stopping

In **`AGENT.md`** (cheat sheet + the "Implement" step of the research cycle),
instruct the agent to **use early stopping whenever the estimator supports it**:
- lightgbm/xgboost: hold out a validation slice from `train`, pass
  `early_stopping_rounds` / callbacks, and let the round count be data-driven
  rather than a large fixed `n_estimators`.
- Explain the dual benefit: it stays within the compute budget *and* tends to
  improve calibration/generalisation (over-boosting hurt `double_lift_slope`
  in prior runs).
- Make clear early stopping must use a **train-internal** split — never the
  search-validation or holdout data.

---

# PART 2 — Open-model robustness

## 2.1 Add a maintained "Cheat sheet" section to AGENT.md

Create a clearly delimited top-level section in `AGENT.md` titled
**`## Cheat sheet — gotchas & tips`**, placed prominently (suggest immediately
after the intro, before "Starting point"). Add a visible comment marker so the
user can append to it over time, e.g.:

```markdown
## Cheat sheet — gotchas & tips
<!-- USER-MAINTAINED: add new tips here as they come up. Keep entries short and concrete. -->
```

Seed it with these (all derived from observed failures):

**Library × loss capability matrix (target has exact zeros):**
- The burning-cost target (`claim_cost_capped_active`) **contains exact
  zeros** (most policies have no claim). Losses requiring strictly positive `y`
  (gamma, log) will error or need a frequency/severity split.
- **Tweedie objective is supported by:** `lightgbm` (`objective="tweedie"`,
  `tweedie_variance_power`), `xgboost` (`reg:tweedie`,
  `tweedie_variance_power`), statsmodels `GLM(family=Tweedie)`, and sklearn
  `TweedieRegressor` (GLM, no trees).
- **Tweedie is NOT supported by** sklearn `HistGradientBoostingRegressor`
  (valid losses: `squared_error`, `absolute_error`, `gamma`, `poisson`,
  `quantile` — and `gamma`/`poisson` need non-negative / positive `y`). Do not
  pass `tweedie_power` to it.
- For pure-premium with zeros: prefer Tweedie (lightgbm/xgboost/statsmodels) or
  a Poisson-frequency × severity decomposition.

**Encoding:**
- Categorical features (e.g. values like `'B12'`) must be encoded before
  estimators that need numeric input. lightgbm accepts `category` dtype;
  xgboost/sklearn need explicit ordinal/one-hot encoding.

**Other recurring traps:**
- Always multiply predicted rates by `exposure_term_a` to return totals.
- Always apply `apply_training_calibration` before returning.
- Build feature lists with care (`list + int` concatenation bug seen).

## 2.2 Cheap preflight smoke-test before the full fit

Before the full `fit_predict` in `run_experiment()` (or as a pre-step in
`_run_validated_experiment_attempts`), run the model on a **small sample**
(e.g. `min(5000, len(train))` rows for train and score) to surface API / type /
path / loss-compatibility errors in **seconds** instead of after a long fit.

- If the smoke-test raises, **skip the full fit** and route straight to the
  repair path with the **full traceback** (see 2.3).
- The smoke-test must be cheap: subsample rows, and if the model honours
  `n_estimators`, you may also cap it for the smoke run (optional). Keep it
  simple — the goal is to catch exceptions, not to validate metrics.
- Guard with config `[compute] preflight_enabled = true` and
  `preflight_sample_rows = 5000`.
- Note: a few models legitimately need enough rows (e.g. rare categories). The
  smoke-test only needs to *execute*, not produce good numbers; catch and
  classify exceptions, don't assert on quality.

## 2.3 Put the full traceback into the repair request

Currently `_write_repair_request()` / the repair JSON centre on the lift check.
Extend the repair request (`repair_request_<n>.json`) and the
`ExperimentNeedsRepair` message to include, when the failure was a runtime
exception (from preflight or the full fit) or a compute-timeout:

- `error_type` (e.g. `"runtime_exception"`, `"compute_budget_exceeded"`,
  `"positive_lift_failed"`)
- `exception_class` (e.g. `TypeError`, `ValueError`)
- `traceback` (full Python traceback string, truncated to a sane length, e.g.
  last ~4000 chars)
- the existing `failed_checks` / `reason` fields for the lift case

This lets the agent fix the *actual* error instead of guessing. Make sure
`_validate_attempt_outputs` distinguishes "experiment raised" from "experiment
ran but failed lift", and tags `error_type` accordingly.

## 2.4 Mechanically enforce repair abandonment (don't asymptote to zero)

In `_run_validated_experiment_attempts` (workflow.py), add a guardrail:

- Track the lift of each attempt.
- If two consecutive attempts produce lift that is **≤ 0 or within the
  resampling noise floor** (`abs(lift) < noise_eps`, where `noise_eps` defaults
  to a small config value, e.g. `0.002`, or is read from the latest comparison's
  `std_lift` if available), **auto-abandon** the proposal instead of consuming
  attempt 3.
- Mark it `failed`/`abandoned` with reason
  `"Auto-abandoned: two consecutive attempts at/below noise floor; structural
  change required."`
- This codifies the existing `AGENT.md` §E advice ("Repair policy — don't
  asymptote to zero") into an actual rule.
- Config: `[repair] noise_floor_eps = 0.002`, `auto_abandon_enabled = true`.

## 2.5 Fix the doubled-path bug

Symptom (from `codex` run notes):
`FileNotFoundError: .../artifacts/tracks/artifacts/tracks/codex/runs/...` — a
run-relative path is being joined onto a cwd that is already inside
`artifacts/tracks/...`.

- Audit how script paths and artifact/prediction paths are resolved in
  `experiment_runner._resolve_model_script_path()` (~line 119) and anywhere
  blend components read prior predictions (the `ValueError: Blend component
  m25_glm_score is missing predictions ...` failure lives in the same area).
  Resolve all such paths to **absolute** (anchored at the run directory or repo
  root) at the point of use, rather than relying on the process cwd. The
  doubled-path string `artifacts/tracks/artifacts/tracks/...` is the tell that a
  run-relative path is being joined onto a cwd already inside `artifacts/tracks`.
- Add a regression test that runs a proposal from a cwd inside
  `artifacts/tracks/` and asserts no path doubling.

## 2.6 Reconcile orphaned `running` proposals + dedup on enqueue

**Orphans:** runs contain proposals stuck in `status='running'` after a crash
(e.g. `opencode/runs/20260530T180022Z` had 3). Add:
- A reconciliation step at cycle start (`run_next_queued_proposal` or
  `handoff.run_latest_proposal_cycle`) that flips any `running` proposal whose
  `updated_at` is older than a threshold (e.g. `2 × current_budget`, or a
  config `[handoff] running_stale_minutes = 30`) to `failed` with reason
  `"Reconciled: stale running proposal (process likely died)."`
- Optionally a heartbeat (`updated_at` touch) during long fits.

**Dedup:** a `duplicate` proposal slipped through (`opencode/runs/20260530T165120Z`).
On enqueue/ingest, skip proposals whose `proposal_id` already exists in the
registry (or whose config hash matches an existing one) and mark them
`duplicate` *before* running, not after.

---

## Testing / acceptance criteria

**Part 1**
- A proposal configured to exceed the budget is killed and marked `failed`
  with a `compute_budget_exceeded` reason; the cycle does not crash.
- `metrics.json` contains a populated `timing` block; the registry
  `experiments` row has the new timing columns populated; existing registries
  open without error after migration.
- `comparison_report.html` shows the compute/timing section.
- The escalation formula is unit-tested: N=0→10, N=4→10, N=5→15, N=10→20.
- `AGENT.md` documents the budget, the ~5× refit cost, cheap-first sequencing,
  and early stopping.

**Part 2**
- `AGENT.md` has a clearly marked, user-maintainable `## Cheat sheet` section
  seeded with the library/loss matrix and encoding/zeros notes.
- A model that raises (e.g. passes `tweedie_power` to HistGBM) is caught by the
  preflight in seconds, and the repair request contains `error_type`,
  `exception_class`, and a `traceback`.
- Two consecutive ≤noise-floor attempts auto-abandon without consuming attempt 3.
- Regression test proves the doubled-path bug is fixed.
- Stale `running` proposals are reconciled to `failed` at cycle start;
  duplicate proposals are marked `duplicate` before running.

## Notes for the implementer
- This is a macOS/Linux Python project (Python 3.13). `signal.SIGALRM` is
  available.
- Native libs (`lightgbm`, `xgboost`) use `n_jobs=-1`; warn in `AGENT.md` and
  consider capping threads to avoid CPU oversubscription when cycles overlap,
  but a hard thread cap is out of scope for this spec.
- Keep all new knobs in `configs/default.toml`; do not hard-code constants.
- Schema changes must be backward-compatible (idempotent `ADD COLUMN`
  migration) — existing run registries must keep working.
- Make all new failure paths funnel through the existing
  `needs_repair`/`failed` machinery so the agent-facing contract is unchanged
  except for the richer repair payload.
