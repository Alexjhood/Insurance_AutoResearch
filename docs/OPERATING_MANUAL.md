# Auto-Research Operating Manual (human reference)

> **This is the full human-facing manual.** The agent does not read this file at
> runtime — it reads the compact, generated `AGENT.md` runtime contract. Keep
> this manual for background, rationale, worked examples, and drill-down. When
> the two disagree about a command name, path, schema, budget, or constant, the
> generated `AGENT.md` (built from code/config by
> `scripts/generate_agent_contract.py`) is authoritative.

You are the research agent for an autonomous tabular target-modelling loop on a per-run selected dataset (see the Datasets chapter below; `french_motor` is the default). The active dataset's default target mode applies unless the run selects another with `--target-mode`. Your goal is to progressively improve predictions measured by **weight-weighted Gini** on the search-validation split, ultimately assessed on a protected holdout on every promotion.

Research run ids must be UTC timestamps in `YYYYMMDDTHHMMSSZ` form. Use
`--new-run` to create that id; do not invent descriptive run ids.

**Run intent rule:** A request to "run N experiments", "start an experiment",
"test N ideas", or begin a new investigation means a **fresh run** unless the
user explicitly says "continue", "resume", or supplies a run id. Do not inspect
`latest_run.json` or existing run artifacts to infer intent. For a fresh run,
the first shell command after reading this file must be:

```bash
autoresearch --track <your-agent-name> --new-run bootstrap-track \
  --model-provider <provider> --model-name <model-name>
```

Capture the timestamped `run_id` returned by bootstrap and pass
`--run-id <returned-run-id>` to every subsequent `autoresearch` command. This
prevents another session or stale `latest_run.json` state from redirecting the
work into a pre-existing run.

Read this file at the start of every session. Keep it open as reference. After
reading this file only, your first shell command in any research run should be an
`autoresearch --track ...` bootstrap or `start-session` command so the
run-scope guard can bind the session before you inspect artifacts.

---

## Datasets — the loop runs on any registered tabular dataset

The framework is dataset-agnostic. Each dataset is one config file under
`configs/datasets/<name>.toml` (a `DatasetSpec`), and its data lives under
`data/datasets/<name>/{raw,processed,metadata,splits,holdout_vault}/`. A
`DatasetSpec` declares the loader, id column, weight column (omit → a synthesised
`unit_weight ≡ 1.0`), optional claim count, missing-value handling, the split
unit (for group-aware splitting), any fixed cap, and the target modes with their
`entity_label`/`rate_label` (from which every metric key/alias is generated).

Registered datasets (see `autoresearch list-datasets`):

| Dataset | Rows | Weight | Default target | Notes |
|---|---|---|---|---|
| `french_motor` | 678K | `Exposure` | `burning_cost` | freMTPL2; cap 100,000; freq×sev available |
| `allstate` | 2M (sampled) | unit | `pure_premium` | household-grouped split; `?` missing; no cap |
| `allstate_full` | 13.2M | unit | `pure_premium` | full variant, enlarged compute budget |
| `porto_seguro` | 595K | unit | `claim_incidence` | binary target as a rate; `-1` missing in `*_cat` |

**Selecting a dataset.** Pass `--dataset <name>` to `bootstrap-track` /
`prepare-data` / `start-session`; it defaults to `french_motor`. `bootstrap-track`
pins the dataset into `run_manifest.json`, and every later command in the run
resolves it from there — you never repeat `--dataset`, and passing a
contradicting one is a hard error (same as run-id pinning). Prepare a dataset's
artifacts with `autoresearch prepare-data --dataset <name>` (bootstrap
auto-prepares when missing).

**Adding a dataset.** Drop a `configs/datasets/<name>.toml`; if the raw shape is
not a single labelled table, add a small loader adapter under
`data/adapters/<name>.py` exposing `load(spec) -> RawDataset`. No other framework
code changes.

The handoff's **"Active dataset"** block is authoritative for the run: it prints
the active columns, target/weight policy, population, cap statement, and
dataset-specific cautions. Read it before proposing.

---

## Cheat sheet — gotchas & tips
<!-- USER-MAINTAINED: add new tips here as they come up. Keep entries short and concrete. -->

### Library × loss capability matrix (target has exact zeros)

The burning-cost target (`ClaimAmountCapped`) **contains exact zeros** — most policies have no claim. Losses requiring strictly positive `y` (gamma, log) will error or need a frequency/severity split.

| Estimator | Tweedie | Gamma / Poisson | Notes |
|---|---|---|---|
| `lightgbm` | ✓ (`objective="tweedie"`, `tweedie_variance_power`) | ✓ (gamma needs `y > 0`) | Preferred for pure-premium with zeros |
| `xgboost` | ✓ (`reg:tweedie`, `tweedie_variance_power`) | ✓ | Preferred for pure-premium with zeros |
| `statsmodels GLM` | ✓ (`family=Tweedie`) | ✓ | Good for GLM baselines |
| `sklearn TweedieRegressor` | ✓ (GLM only, no trees) | — | |
| `sklearn HistGradientBoostingRegressor` | **✗ — no `tweedie_power` arg** | gamma/poisson valid but need `y > 0` | Valid losses: `squared_error`, `absolute_error`, `gamma`, `poisson`, `quantile` |

**Do not pass `tweedie_power` to `HistGradientBoostingRegressor`** — it will raise `TypeError` immediately.

For pure-premium with zeros: prefer Tweedie objective (lightgbm/xgboost/statsmodels) or Poisson-frequency × severity decomposition.

### Categorical encoding

Features with string values (e.g. `'B12'`) must be encoded before estimators that need numeric input.
- **lightgbm**: cast to `category` dtype; lightgbm handles it natively.
- **xgboost / sklearn**: use ordinal or one-hot encoding explicitly.

### Other recurring traps

- These traps apply to **hand-written scripts that return a raw array**; a
  recipe or a `Prediction` return has the framework handle both automatically.
- Always multiply predicted rates by `Exposure` to return totals.
- Always apply `apply_training_calibration` before returning.
- Build feature lists with care — `list + int` concatenation raises `TypeError`.
- `blend` components require predictions from prior experiments to exist on disk.

### Compute budget (see also "Exploration philosophy" below)

There is a per-experiment wall-clock budget controlled by `[compute]` in `configs/default.toml`:
- Default: **10 minutes for the first 5 experiments**, +5 minutes every 5 experiments (`10 + 5 × (N // 5)` minutes).
- The challenger is **refit ~5×** per comparison (1 experiment fit + 4 CV folds), so the effective cost is ~5× a single fit. Budget your single fit accordingly.
- Use **early stopping** whenever the estimator supports it (see "Exploration philosophy → Compute budget & early stopping").
- Cost drivers: `n_estimators × (1/learning_rate)`, `num_leaves`/`max_depth`, dataset size. A 5000-tree, `lr=0.003`, no-early-stopping model is ~25k tree builds and will likely exceed the budget.

---

## Starting point — every run begins with no model

Each run is bootstrapped with the **`global_mean` baseline** for the active target: burning-cost mode predicts `(total training claim cost / total training exposure) × exposure`; frequency mode predicts `(total training claim count / total training exposure) × exposure`. It is the flat exposure-weighted rate, the simplest possible "model", and it is the official champion at the start of every run.

Everything you build develops relative to this. The first real model you propose only has to beat a constant rate; you do not need to start with a sophisticated method. Take the smallest interpretable step that could plausibly outperform the global mean and iterate from there.

## Exploration philosophy — small steps, broad search

The research loop rewards **many small, well-motivated improvements** over a few large jumps. When you choose what to try next:

- **Design experiments from results, not in advance.** This is the most important habit. Treat the cycle count as a **budget, not a to-do list**: submit one experiment, read its result, and let that result shape the next hypothesis. Do **not** plan a fixed slate of N experiments at the start of a run and then execute it regardless of what you see. After each cycle, before proposing the next, say what the last result changed about your thinking — a surprising, degenerate, or near-miss result is a signal to run a quick diagnostic on *why* (e.g. a model that scores no better than the flat baseline is almost certainly broken, not merely weak — investigate before moving on), not a cue to advance to a pre-decided next idea. You **may** deliberately commit to a short, explicit sequence when you want comprehensive coverage of a defined set (e.g. a feature-engineering sweep, or a head-to-head of a few model families) — but name the sequence and why, still read results between steps, and abandon it early if a result makes it moot. Let breadth **emerge** from these adaptive choices, not from a coverage checklist written up front.
- **Use the active run's research tree.** The handoff context contains a `research_tree` for this run only. Choose a `research_parent_node_id` when a new idea builds on a prior hypothesis, near-miss, auto-rejection, or informative failure. Use `null` only for a genuinely new line of attack. Do not use other runs as proposal evidence unless the user explicitly asks for cross-run analysis.
- **Follow the tree policy.** `research_tree.tree_policy.recommended_actions` gives a small set of active-run tree-walk options. By default the controller takes the top recommendation (and fills `tree_action`, `selected_tree_action_id`, `research_parent_node_id`, `parent_rationale` for you). To pick a *different* option, set `selected_tree_action_id` (and the matching `tree_action`) via the optional-override block; if you diverge from the recommendation, include `tree_policy_override_rationale`.
- **Submit one idea per context refresh.** The framework ingests at most one valid proposal while any proposal is queued or awaiting decision. Extra JSON files stay in the inbox as `deferred_pending_context_refresh`; refresh context before relying on them.
- **Bias toward breadth over depth.** Try a range of different ideas before doubling down on any one direction. A run that explores many distinct hypotheses in a session is better than one that iterates narrowly.
- **Prioritise variety of approach, not just variety of dial.** Small steps are fine and similar space is fine to revisit — but the *priority* is covering genuinely different modelling paradigms: different model families (linear/GLM, single trees, bagged trees, boosted trees, GAMs, nearest-neighbour, neural), different target framings (direct pure-premium vs frequency–severity vs two-stage, rate-target vs total-target), and different feature representations (raw, binned, interacted, encoded). Re-tuning one estimator's hyperparameters is the *lowest-information* move available — reach for it only when a distinct approach has been ruled out, not as the default next step. Past runs have stalled by submitting ~10 near-identical boosted-tree variants in a row; do not repeat that pattern.
- **Analyse the problem before tuning it.** Before proposing, spend a little effort understanding *why* the champion misses: look at calibration residuals by segment (region, age band, vehicle type, exposure), check where the largest errors concentrate, and check what signal a simple model is and isn't capturing. A targeted diagnostic that tells you *where* the model is wrong is worth more than a blind hyperparameter sweep. This is especially important at a plateau (see routine D) — a plateau is a signal to *investigate*, not to tune harder.
- **One change at a time.** Every experiment should be readable as "X relative to the current champion". If you change multiple things at once, the next cycle has no clean signal to learn from.
- **Prefer lower-cost approaches before higher-cost ones.** Cheap, fast experiments tell you what the data can support before you commit compute to expensive methods.
- **Use the research log.** Log what each step taught you, not just whether it promoted. A non-promotion that taught you something about a segment is valuable.

This applies to every cycle, including the very first proposal of a fresh run.

### Compute budget & early stopping

**Per-experiment wall-clock budget** (configured in `configs/default.toml` `[compute]`):

```
budget_minutes = 10 + 5 × (N // 5)
```

where `N` = number of experiments already run in this run (zero-based). So experiments 1–5 get 10 min, 6–10 get 15 min, etc.

**Sequencing guidance:** start with cheap, fast models (GLMs, shallow trees, small `n_estimators`) to map the available signal, then escalate to higher-capacity models only once cheap ideas are exhausted. This dovetails with the "small steps, broad search" philosophy.

**The challenger is refit ~5×** per comparison (one experiment fit + 4 CV folds; champion folds are cached). Effective cost ≈ 5× a single fit — budget accordingly.

**Cost drivers to watch:** `n_estimators × (1/learning_rate)`, `num_leaves`/`max_depth`, and dataset size. A "5000-tree, lr=0.003, no early stopping" model is ~25k tree builds over 430K rows and will likely blow the budget.

**Use early stopping whenever the estimator supports it.** This saves compute *and* tends to improve calibration:

- **lightgbm / xgboost:** hold out a validation slice from `train` (e.g. 10%), pass `early_stopping_rounds` / callbacks, and let the round count be data-driven rather than a large fixed `n_estimators`.
- Use a **train-internal** split for early stopping — **never** the search-validation or holdout data.
- Record the chosen `n_estimators` in `model_notes` so future runs can use it as a starting point.

If an experiment times out, the framework marks it `failed` with a `compute_budget_exceeded` reason and the repair request will contain the budget and elapsed time. Fix: reduce `n_estimators`, increase `learning_rate`, or use early stopping.

---

## What you are optimising

**Primary KPI & gate metric**: `gini_weighted` — higher is better. The exposure-weighted Lorenz-area Gini is both the headline business KPI and the metric the `cv_bootstrap` gate ranks challengers on (win rate, lift, escalation trigger). `rank_gini_weighted` is still computed alongside it for reference (bounded-influence Somers' D), but it no longer drives the gate. Since 2026-06-12 the Gini is tie-aware (tied predictions carry no ordering signal; a constant model scores exactly 0); scores recorded before that date used input-order tie-breaking and can be inflated by ~0.02 for near-flat models.

**The decision is yours, not a threshold.** The framework computes the full metric panel across all bootstrap×fold samples and a set of *advisory* gates, but it does not auto-promote. You review everything and call `record-decision` (see "You own the decision" below). Hard guardrails can only block clearly-broken promotions — they never promote for you.

**Full metric panel** (all computed on every evaluation, all visible in the multi-metric exhibit):

| Metric | Type | Notes |
|---|---|---|
| `gini_weighted` | Discrimination (KPI + gate) | Business KPI and the cv_bootstrap gate metric |
| `rank_gini_weighted` | Discrimination (reference) | Bounded-influence Somers' D; robust to tail placement |
| `asym_pricing_loss` | Pricing risk (lower=better) | Penalises under-pricing 4× over-pricing (see below) |
| `spearman_rho` | Rank correlation | Model-agnostic; no distributional assumptions |
| `kendall_tau` | Rank correlation | Concordant/discordant pair count |
| `decile_lift_monotonicity` | Monotonicity | Spearman of decile-mean actual vs decile order |
| `tweedie_deviance_p15` | Loss fit (p=1.5) | Primary deviance for burning-cost models |
| `poisson_deviance` | Loss fit (p=1.0) | Frequency model quality |
| `double_lift_slope` | Calibration | Regression slope of actual on predicted by decile (want ≈ 1.0) |
| `predicted_to_actual_ratio` | Calibration | Aggregate level (want ≈ 1.0) |
| MAE/RMSE variants | Error magnitude | Both rate and total; burning cost and frequency |

In `burning_cost` mode, model scripts return predicted claim-cost totals. In
`frequency` mode, model scripts return expected claim-count totals. Scripts
should model rates internally if useful, then multiply by `Exposure`
before returning.

---

## You own the decision

After every comparison, the framework writes `decision = "pending_llm"`. **You must review the metric summary and call `record-decision` before the cycle can advance.** The mechanical advisory gates are informative, not binding.

### What to review before deciding
1. **Full metric table**: check `gini_weighted`, `rank_gini_weighted`, `asym_pricing_loss`, calibration ratio.
2. **Advisory gate panel**: did the challenger pass or fail the configured thresholds?
3. **Guardrail status** (shown in the report banner): any hard-fail blocks promotion regardless of your choice.
4. **Escalation**: if win rate was in the close-call band [0.50, 0.75] (one fold-unit either side of the 0.60 promotion threshold), escalation added extra partitions — the post-escalation win rate is the one to read.
5. **Independent evidence**: confidence intervals and the gate win rate operate on whole `(partition, fold)` units. The effective evidence count is normally 4 folds, or 12 after escalation; the 20 within-fold bootstraps describe row noise rather than 20 independent model fits.
6. **Asymmetric Pricing Loss (APL)**: lower is better. `asym_pricing_loss` penalises under-pricing 4× over-pricing. A challenger with a good Gini but high APL is writing profitable policies in the wrong segments.

### How to record your decision
```bash
autoresearch --track <track> record-decision <comparison_id> --decision promote --reason-code clear_win --rationale "Clear panel improvement." --interpretation "The new model captured stable signal." --next "Build from the promoted model."
autoresearch --track <track> record-decision <comparison_id> --decision local_promote --reason-code line_progress --rationale "Useful line-local progress." --interpretation "This framing helps within the line." --next "Continue the line without replacing the global champion."
autoresearch --track <track> record-decision <comparison_id> --decision reject --reason-code noise --rationale "Insufficient evidence." --interpretation "The apparent lift was not stable." --next "Rotate to a materially different approach."
```

The comparison_id appears in the `compare-experiments` output and in `list-promotions`.

On `promote`: guardrails are re-checked; hard fails block the promotion with an error message. On success, the holdout evaluation fires automatically and the proposal also becomes the local incumbent for its research line.

On `local_promote`: the proposal becomes the local incumbent for its research line, but the official champion and holdout remain unchanged. When it differs from the global champion, the report includes a second cluster-bootstrap comparison against the line incumbent on the same partitions.

On `reject`: the official champion is retained. Treat the result as evidence for future line design.

---

## Research lines

Every proposal belongs to a local research line. By default the controller
extends the most recent active line (or opens a first line when none exist), so
you can omit all of these. Set them via the optional-override block only when you
want to organise the run differently:

- `research_line_action`: `create_line`, `extend_line`, `revisit_line`, or `close_line`. Defaulted to `extend_line` (or `create_line` for a brand-new `research_line_id`).
- `research_line_id`: a short stable identifier for the line. Defaulted to the most recent active line.
- `research_line_label`, `research_line_hypothesis`, `line_membership_rationale`: for an existing line these are inherited from the registry; supply them only when you create a new line.
- `park_research_line_id`: optional; required when creating a new line while 5 lines are already active.

Keep the run organised into at most 5 active lines. A line is a local sequence of related hypotheses, not a prescribed model family. It should describe what the run is trying to learn, while leaving implementation choices open. Park weak or exhausted lines instead of keeping them active indefinitely.

The framework tracks two kinds of promotion:

- **Global promotion** (`promote`): replaces the official champion for the whole run and triggers holdout evaluation.
- **Local promotion** (`local_promote`): advances only the proposal's research line and becomes that line's incumbent for future screening.

If later evidence shows a local incumbent was an artefact, clear it:

```bash
autoresearch --track <track> clear-line-champion <line_id> --reason "Local incumbent appears artefactual; future screening should fall back to the official champion."
```

To park an exhausted line manually:

```bash
autoresearch --track <track> park-research-line <line_id> --reason "No useful near-term follow-up."
```

---

## Asymmetric Pricing Loss (APL)

`asym_pricing_loss` = Σ w·(4·under + 1·over) / Σ w, where under = max(actual_rate − predicted_rate, 0) and over = max(predicted_rate − actual_rate, 0). Lower is better (not in HIGHER_IS_BETTER_METRICS).

The 4:1 ratio reflects the economic reality that a policy written at a loss generates claims that exceed the premium, costing ~4× more than a missed quote (which only loses margin). A model that systematically under-prices high-risk segments will have a high `asym_pricing_loss` even if its Gini looks acceptable.

Diagnostic sub-metrics: `apl_under_cost` (mean exposure-weighted shortfall), `apl_over_cost` (mean excess), `apl_under_over_ratio` (realised under/over balance — close to 4 is expected at optimal pricing).

---

## Gate modes — how comparisons are adjudicated

The comparison gate has three modes, configured via `gate_mode` in `default.toml`.

### `cv_bootstrap` (default)

Generates a deterministic fold partition from the run id, then bootstrap-resamples each fold ×20. Confidence intervals use a hierarchical cluster bootstrap that treats each `(partition, fold)` as one independent unit. Gate metric: `gini_weighted`.

- Base comparison: 1 partition × 4 folds × 20 bootstrap = **80 samples**.
- Close call (win rate in [0.50, 0.75]): escalation adds 2 extra partitions → **240 samples**. The band spans one fold-unit either side of the 0.60 promotion threshold because base-path win rates are quantised to quarters.
- Effective independent units: **4 folds** on the base path and **12 folds** after escalation.
- The base partition rotates after every 5 recorded full comparisons to limit long-run adaptive overfitting.
- Champion fold predictions are **cached** — only the challenger needs to be refit (4 fits vs old 16–32).
- **~8× cheaper** than the old `repeated_cv` default on the common path.

### `repeated_cv` (legacy)

Refits both models on cv_n_repeats × cv_folds stratified partitions. Costs cv_n_repeats × cv_folds refits per comparison per model. Gate metric: `rank_gini_weighted`. Use when you need to compare against pre-cached repeated-CV results.

### `single_partition` (legacy / fast)

Evaluates on the fixed `search_validation` split with 30 bootstrap resamples. CI measures within-split noise only. Use for quick sanity checks or expensive-to-refit models.

### Single-split screening

Before the expensive CV/bootstrap comparison, every valid challenger is screened on the full `search_validation` split. A paired row bootstrap estimates uncertainty from the existing predictions without refitting. The challenger is auto-rejected only when the interval's upper bound is below the configured clear-loser hurdle; uncertain cases proceed to full comparison. Very small diagnostic datasets use an explicit point-estimate fallback. If the line has a local incumbent, screening compares against it; otherwise it falls back to the official champion.

---

## Quick-start — how to interpret short user instructions

Your default track is your current tool name: **`codex`** when running in Codex, **`claude`** when running in Claude Code, and **`opencode`** when running in OpenCode. These are the only valid research-session track folders; `default` and custom track names are reserved for human/admin analyst work. Your default cycle count is **3**.

For a **new run**, always pass `--new-run`; this creates a fresh timestamped
folder such as `artifacts/tracks/codex/runs/20260527T211530Z/`. Capture the
returned run id and pin it with `--run-id` on every later command. For explicit
**continue** instructions without a supplied run id, the first `start-session`
command may omit both `--new-run` and `--run-id` to resolve the track's latest
run; capture the resolved run id from its output and pin all later commands.
Never inspect existing artifacts to decide whether an otherwise new request
should continue an old run.

A cycle count (the **3** default, or an explicit X/Y below) is a **budget of up to that many experiments to spend adaptively**, not a slate to design in advance. Design each experiment after reading the previous one's result (see "Exploration philosophy → Design experiments from results, not in advance"). You may stop early if a line is clearly exhausted, or commit to a short explicit sequence when you deliberately want coverage.

| User says | What to do |
|-----------|-----------|
| "Go!" / "Start" | Bootstrap a new timestamped run in your tool track, read handoff, run up to 3 experiments (adaptively) |
| "Run X experiments" | Bootstrap a new timestamped run in your tool track, read handoff, run up to X experiments (adaptively) |
| "Continue" / "Keep going" | Read handoff, run up to 3 experiments (skip bootstrap) |
| "Continue and run Y" | Read handoff, run up to Y experiments (skip bootstrap) |
| "Bootstrap only" | Bootstrap and read handoff, then stop |

**Bootstrap** — run once at the start of every fresh conversation. Model identity is required for run attribution:
```bash
autoresearch --track <your-agent-name> --new-run bootstrap-track \
  --model-provider <provider> --model-name <model-name>
# e.g. --model-provider anthropic --model-name claude-sonnet-4-6
# e.g. --model-provider openai   --model-name codex-mini-latest
# Capture the run_id printed by bootstrap as <returned-run-id>.
```

**Read handoff** — always do this after bootstrap or at the start of a continuing session:
```bash
# The handoff path is printed by bootstrap — it ends in handoffs/latest_handoff.md
# Read it to understand current champion state before proposing anything.
```

**Run N cycles** — `run-session-cycles` requires an active session. On a fresh run, create one first (idempotent name is fine):
```bash
autoresearch --track <your-agent-name> --run-id <returned-run-id> start-session main
autoresearch --track <your-agent-name> --run-id <returned-run-id> run-session-cycles <N>
```

**Each cycle now pauses for your decision.** `run-session-cycles` runs a proposal through experiment + comparison and then **stops in the `awaiting_decision` state** — it does not auto-promote. You must review the metric summary and call `record-decision` (see "You own the decision") before the next cycle. So to run N experiments you loop: `run-session-cycles 1` → review → `record-decision …` → repeat. A larger N still stops after the first comparison that needs a verdict.

If the user supplies a specific timestamped `--run-id`, pass it to every
command. Otherwise use `--new-run` for fresh starts, capture its returned run
id, and pin that id thereafter. For an explicit continuation without a
supplied id, resolve latest once with `start-session`, then pin the returned id.

---

## Session start — first shell command

> **Bind before you look around.** After reading this file, do not run discovery
> commands such as `ls`, `find`, `rg`, `cat`, `sed`, or `git status` against the
> repository or `artifacts/` until you have run one of the `autoresearch --track
> ...` commands below. That first `autoresearch` command binds your session to a
> run; until it runs, the run-scope guard cannot tell which run is yours. (See
> "Run-scope isolation" near the end of this file.)

For a fresh run:

```bash
autoresearch --track <your-agent-name> --new-run bootstrap-track \
  --model-provider <provider> --model-name <model-name>               # first command; capture returned run_id
autoresearch --track <your-agent-name> --run-id <returned-run-id> start-session main
autoresearch --track <your-agent-name> --run-id <returned-run-id> list-champion-history
autoresearch --track <your-agent-name> --run-id <returned-run-id> list-experiments
```

For an explicit continuing request, skip `bootstrap-track`. If the user did not
provide a run id, resolve latest exactly once and capture the run id printed by
`start-session`:

```bash
autoresearch --track <your-agent-name> start-session main             # first command only; capture resolved run_id
autoresearch --track <your-agent-name> --run-id <resolved-run-id> list-champion-history
autoresearch --track <your-agent-name> --run-id <resolved-run-id> list-experiments
```

Then read the handoff file printed by bootstrap (or the latest handoff for a continuing run) and this run's `RESEARCH_LOG.md` before forming any hypothesis.

**Also check for** `artifacts/tracks/<track>/runs/<run-id>/OPERATING_NOTES.md` — if present, it's a per-run cheatsheet (current champion, registry path, std_lift, proposal schema) curated by prior sessions to skip rediscovery.

---

## Operating routines — lessons from prior runs

These rules are encoded from process audits and prevent recurring failure modes. Follow them by default; deviate only with a clear reason.

### A. Baseline-first rule (prevents target-column / dispatcher bugs)

Before writing **any** new model family or your first proposal in a fresh run, read these three files **once**:

- `src/autoresearch/models/global_mean.py` — confirms the active training-target column name
- `src/autoresearch/models/dispatcher.py` — confirms what's in the `score` DataFrame, exposure handling, and feature constants
- `src/autoresearch/models/calibration.py` — confirms the `apply_training_calibration` signature

The agent schema in `context/latest_context.json` lists *all* historical target columns. Do not pick the training target from it — use whatever `global_mean.py` uses. Past runs have wasted 2 full experiments on this single bug.

### B. Proposal-schema first (prevents inbox ingestion failures)

The inbox JSON has a small, strict schema. Before writing your first proposal in a fresh run, read `proposal_inbox/proposal_template.json` *once*. You only supply the fields that encode your scientific choice: `experiment_name`, `rationale`, `change_summary`, `expected_benefit`, `key_risk`, `exploration_axis`, `approach_family`, `target_framing`, `feature_representation`, `expected_learning`, plus an `experiment_config` carrying `model_family`, `target_strategy`, and a model implementation — either `model.recipe` (preferred; set `model_family = "recipe"`) or `model.script_path` (escape hatch). The controller **derives** the rest at ingestion — `proposal_id`, `parent_experiment_id`/`parent_branch_id`, `branch_action`, the tree-walk fields (`tree_action`, `selected_tree_action_id`, `research_parent_node_id`, `parent_rationale`), the research-line `research_line_action`/`research_line_label`/`research_line_hypothesis`/`line_membership_rationale`, the fixed `preprocessing` block, and the duplicate `experiment_config.experiment_name`/`experiment_config.parent_experiment_id`. You do not need to repeat any of those. To deviate from a default, supply that field explicitly (see the optional-override block in the handoff); when you diverge from the recommended tree action also include `tree_policy_override_rationale`.

Tree fields are behavioural metadata, not prescriptions for implementation. Use them to describe what the experiment is trying to learn and where it sits in this run's tree. `tree_action=new_root` may use `research_parent_node_id=null`; every other `tree_action` must point to a valid node in this run's `research_tree`.

The handoff export refreshes both `handoffs/proposal_template.json` and `proposal_inbox/proposal_template.json`. If a proposal is already queued, running, or awaiting decision, treat any other proposal JSON left in the inbox as stale until the next context refresh says otherwise.

### C. Inbox audit (prevents re-submitting stale stubs)

List **this run's** inbox at session start — `artifacts/tracks/<track>/runs/<run-id>/proposal_inbox/` (the exact path is printed in the handoff as `Inbox: ...`), not the repo-root `proposal_inbox/`, which is the legacy untracked path and stays empty for agent runs. Any pre-existing `model_*.py` may contain bugs from prior sessions (wrong target column, exposure incorrectly used as a predictive feature, outdated calibration). Either:
- pick a fresh filename with a session prefix (e.g. `s3_<name>.py`), or
- `Read` the existing file in full and audit it before reuse.

The framework only auto-ingests `*.json` files from the inbox, so leftover `.py` files are dormant but easy to misuse.

### D. Plateau detection — the std_lift gate

When the champion sits at a metric plateau, marginal tuning is provably below the noise floor of the resampling gate and **cannot promote**.

Before submitting an experiment, check the latest champion's `std_lift` from `paired_summary` in the most recent comparison (`comparisons` table in `registry.sqlite`, or the latest `comparison_report.html`). If the expected effect of your change is < `2 × std_lift`, the bootstrap CI will straddle zero and it cannot pass the gate — **switch to a structural change instead**:

- new model family (XGBoost, CatBoost, neural net)
- new target decomposition (freq/sev vs direct PP vs two-stage)
- new feature (engineered interaction, new transformation)
- new sample subset (high-exposure-only, claimants-only refinement)

Rule of thumb: if you have 2 consecutive same-axis experiments at the plateau, the next experiment must change axis.

### E. Repair policy — don't asymptote to zero

`run-latest-proposal-cycle` allows up to 3 attempts. When attempt 1 fails:

- **Attempt 2 must move in the opposite direction**, not "halfway back" to champion. If 127 leaves was too deep, try fewer leaves or stronger regularization — not 95 leaves. Halfway-back attempts converge to zero lift without ever beating champion.
- **Attempt 3 is for abandonment or a genuinely different angle.** If attempt 2 is still negative, prefer to let it fail rather than burn the slot on a minor variation. A failed experiment is cheaper than a third near-zero attempt.

### F. Avoid re-writing champion scaffolding (eliminates boilerplate)

Do not restate ~100 lines of identical model scaffolding in every proposal. For
any method the recipe registry covers, **use a `model.recipe`** (see Step 2,
Option A) — it is a few lines, validated before running, and the framework owns
exposure conversion and calibration. Reserve hand-written scripts for genuinely
novel models the recipe vocabulary cannot express. Each verbatim re-write of the
champion script costs ~3K output tokens for zero information gain.

> After every promotion the framework now writes
> `proposal_inbox/champion_recipe.json` and `proposal_inbox/champion_template.py`
> into this run's inbox. For a one-parameter follow-up, edit `PARAM_OVERRIDES`
> (or `RECIPE`) in `champion_template.py` rather than rewriting a model. (Cost
> review recommendation #8 — now implemented.)

### G. Axis rotation & approach diversity

Track which **axis** each experiment changes (`model_family`, `target_framing`, `feature_representation`, `calibration`, `hyperparameter`, `diagnostic_probe`, `data_slice`, `ensemble`, or `other`). After 2 same-axis experiments — promoted or not — rotate to a different axis. This prevents the failure mode where 3+ consecutive experiments all tune the same dial. The framework now computes this from proposal metadata and surfaces recommendations in `research_tree.tree_policy`.

**Stronger rule for model family / paradigm (the most common waste):** maintain a short ledger in your research log of which *distinct approaches* you have tried, along these axes:

- **Model family** — linear/GLM · single tree · bagged trees (RF/ExtraTrees) · boosted trees (LightGBM/XGBoost/CatBoost/HistGBR) · GAM · k-NN · neural · ensemble/stack
- **Target framing** — direct pure-premium · frequency–severity · two-stage · rate-target vs total-target
- **Feature representation** — raw · binned · interactions · target/one-hot encoding

Rules:
- **Cap same-family tuning at 3 experiments.** After 3 experiments within one model family (whether you promoted or not), you must try a *different family* before returning to it. Submitting a 4th near-identical variant of the same estimator is not allowed unless you can name a specific, evidence-backed reason from a diagnostic.
- **Prefer the unexplored cell.** When choosing the next experiment, favour a (family × framing) cell you have not yet tried over one you have. Breadth of paradigm beats depth of tuning, especially before a champion is well established.
- **A plateau forces a paradigm change, not a tuning change.** If the champion has held for 2+ cycles, the next experiment must change model family *or* target framing — re-tuning the current family at a plateau is provably below the noise floor (routine D).

Reason: runs have plateaued ~6 Gini points below what the data supports by committing to one paradigm (e.g. rate-target boosted trees) and exhausting the budget on hyperparameter and ensembling variants within it, while a different family on the same data scored materially higher. Diversity of approach is the highest-leverage habit in this loop.

---

---

## The research cycle

> **Which workflow is authoritative.** The supervised loop is the one to use:
> propose into the inbox, then `run-session-cycles 1` → review →
> `record-decision` (see "Quick-start" and "You own the decision"). The
> `run-baseline` / `compare-to-champion` commands shown in Steps 4–5 below are
> the **legacy direct path**: `run-baseline` explicitly bypasses the proposal,
> comparison, and decision workflow, and `compare-to-champion`'s auto-promote
> behaviour is superseded by the supervised gate that always stops at
> `pending_llm` for your `record-decision`. Use the direct path only for manual
> diagnostics, never inside a research run. Run artifacts live under the
> run-scoped `iterations/` layout, not the legacy `artifacts/experiments/` and
> `artifacts/comparisons/` paths some older examples below still show.

### Step 1 — Form a hypothesis
Read the research log and recent experiment metrics. Ask, in roughly this order:
- Is there an obvious feature transformation (log, binning, indicator) that the current champion misses?
- Is there a single interaction (e.g. factor_a × factor_b, factor_c × factor_d) that I have not yet tried?
- Could a simple model with few features clarify which signal the data actually carries?
- Is the current champion's calibration breaking down on a specific segment (by region, age band, vehicle type)?
- Have I exhausted the cheap interpretable ideas before reaching for higher-capacity approaches?

Quick data investigations on the training set can be valuable for forming and sharpening hypotheses before committing to an experiment. When progress stalls, *analysis beats more experiments*: run a targeted diagnostic (segment residuals, error concentration, decile calibration) to locate where the champion is wrong, and let that finding drive the next proposal rather than guessing.

Also ask: **which distinct approaches have I not yet tried?** Check your approach ledger (routine G) — if the last few experiments clustered in one model family or target framing, deliberately pick an unexplored (family × framing) cell next.

Prefer the smallest change that would credibly improve on the current champion. If you have not yet seen what a simple model with a few features does, do that before reaching for something more complex. If you have not yet looked at calibration residuals, do that before adding more capacity.

Write your hypothesis — and why it is the cheapest next step — at the top of your next research log entry before coding.

### Step 2 — Implement

You provide the model **either** as a declarative recipe (preferred for any
method the registry covers) **or** as a run-local Python script (the escape hatch
for novel models). Both flow through the same framework units/calibration stage,
so you never write exposure conversion or calibration yourself.

**Option A (preferred): Declarative model recipe**

Set `model_family = "recipe"` and put a `model.recipe` object in the proposal —
no Python file needed. Trusted framework code (`autoresearch.models.recipe`)
interprets it: feature selection, encoding, estimator construction, train-internal
early stopping, rate→total conversion, and aggregate calibration are all handled
for you. Invalid building-block combinations (e.g. gamma loss on pure premium,
native categoricals on xgboost, Tweedie on `hist_gbm`) are rejected before
anything runs.

```toml
experiment_name = "lgbm_tweedie_v1"
model_family = "recipe"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model.recipe]
structure = "direct"          # or "frequency_severity" (burning-cost mode only)
estimator = "lightgbm"        # lightgbm | xgboost | hist_gbm | tweedie_glm | elasticnet | constant
objective = "tweedie"         # tweedie | poisson | gamma | squared_error
encoding = "native_categorical"   # native_categorical | one_hot | ordinal (defaults per estimator)
early_stopping = 50

[model.recipe.params]
num_leaves = 63
learning_rate = 0.05
```

Frequency × severity nests two stages, each a normal recipe:

```toml
[model.recipe]
structure = "frequency_severity"
[model.recipe.stages.frequency]
estimator = "lightgbm"
objective = "poisson"
[model.recipe.stages.severity]
estimator = "lightgbm"
objective = "gamma"
```

Target → objective rules: **pure premium** (has exact zeros) → `tweedie` /
`squared_error`; **frequency** → `poisson` / `tweedie` / `squared_error`;
**severity** (claim rows, strictly positive) → `gamma` / `squared_error`. Restrict
features with `model.feature_inclusions` / `feature_exclusions`. Recipes used
earlier in the run are recorded with their outcome and surfaced for reuse
(`[recipes] reuse_scope`; cross-run reuse engages only under memory access). To
add a method not in the registry, register it in
`src/autoresearch/models/recipe/estimators.py` or use Option B.

**Option B: Autonomous proposal with a run-local model script (escape hatch)**
Create one proposal JSON and one neighbouring Python script in the proposal
inbox. The JSON must set `experiment_config.model.script_path` to the script
filename. Do not rely on pre-existing model implementations in
`src/autoresearch/models`; write the modelling logic into this run's script.

Proposal config shape:

```toml
experiment_name = "my_descriptive_name"
model_family = "scripted_model"        # descriptive label for this script
target_strategy = "direct_pure_premium"
parent_experiment_id = ""              # fill in after first run

[preprocessing]
# Fixed by product decision — do not change.  100,000 is applied identically to
# training, search-validation, and milestone-holdout rows.
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
script_path = "model_my_descriptive_name.py"
# Any hyperparameters declared here are passed as **hyperparameters to fit_predict
# Optional: feature subset
# feature_inclusions = ["feature_a", "feature_b", "feature_c"]
```

The model script must expose:

```python
def fit_predict(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    **hyperparameters,
) -> tuple[np.ndarray, dict]:
    """Fit on train, return active-target total predictions and notes."""
    ...
```

A script may instead return a `Prediction(values=rates, unit="rate")` (from
`autoresearch.models.prediction`) and let the framework convert rate→total via
exposure and calibrate for you — the same finalisation recipes use. If you return
a raw `np.ndarray` you must return target totals (multiply pure-premium/frequency
rates by `score["Exposure"]`) and calibrate yourself (below).

**Option C: New feature engineering module**
Create `src/autoresearch/features/<name>.py`. Must expose:

```python
def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to the frame. Must not drop existing columns.
    Must not access holdout data. Must be deterministic."""
    frame = frame.copy()
    # e.g. interaction term:
    frame["feature_a_x_feature_b"] = frame["feature_a"] * frame["feature_b"]
    return frame
```

Reference it in the TOML: `feature_builder_module = "autoresearch.features.<name>"`.

Column constants (import from `autoresearch.models.dispatcher`):
- `EXPOSURE = "Exposure"`
- `CLAIM_COST = "ClaimAmountCapped"` (training target)
- `CLAIM_COUNT = "ClaimNb"`
- `CLAIM_EVENTS = "ClaimAmountCount"`
- `RECORD_ID = "record_id"`

**Calibration — always apply (unless you return a `Prediction`)**

Recipes and scripts that return a `Prediction` are calibrated by the framework
automatically — skip this. A script that returns a raw `np.ndarray` must apply a
training-total calibration scalar before returning predictions. This is a single
aggregate correction (one degree of freedom, no leakage risk) that guarantees the
aggregate gate passes and preserves visibility of the model's native bias in the
comparison report.

```python
from autoresearch.models.calibration import apply_training_calibration

# pred_train and pred_score must be target totals (not rates)
target = train[CLAIM_COUNT].values if hyperparameters.get("target_mode") == "frequency" else train[CLAIM_COST].values
pred_score, calib_factor = apply_training_calibration(
    pred_score, pred_train, target
)
notes["native_pred_to_actual_ratio"] = round(1.0 / calib_factor, 4)
notes["calib_factor"] = round(float(calib_factor), 4)
```

**Early stopping — use it whenever the estimator supports it**

For lightgbm/xgboost, hold out a validation slice from `train` internally and pass early-stopping parameters:

```python
# lightgbm example
n_val = max(1000, int(0.1 * len(train)))
val_slice = train.iloc[-n_val:]
train_slice = train.iloc[:-n_val]
callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
model.fit(X_tr, y_tr, sample_weight=w_tr,
          eval_set=[(X_val, y_val)], callbacks=callbacks)
notes["n_estimators_chosen"] = model.best_iteration_
```

Use a **train-internal** split (never search_validation or holdout). This keeps compute within budget *and* avoids over-boosting (which hurt `double_lift_slope` in prior runs).

If you need a new library: add it to `pyproject.toml` and run:
```bash
pip install -e ".[dev]"
```

### Step 3 — Validate your changes

The experiment runner runs pytest automatically inside `run-latest-proposal-cycle` and fails immediately if any test fails — **you do not need to run pytest manually before calling that command**. Only run it yourself when you have added a new module or test file and want to verify it passes before submitting, or when debugging a test failure:

```bash
pytest --tb=short -q   # optional: only when you've added/changed test files
```

New model families or feature builders should come with at least a smoke test in `tests/`.

### Step 4 — Run the experiment

```bash
autoresearch run-baseline configs/experiments/<your_config>.toml
```

This will:
- Scan your model/feature code for holdout references (integrity check)
- Run pytest
- Fit the model and score on search_validation
- Compute the full actuarial metric panel
- For queued autonomous proposals, validate prediction sanity and positive lift
  against the champion before final comparison. If validation fails, inspect
  `repair_request_*.json` and follow its `repair_kind`: for a recipe experiment
  write the corrected recipe to `recipe_attempt_N.json` (recipes stay recipes —
  the framework keeps owning units/calibration); for a script experiment write
  the next `model_attempt_N.py`. Rerun. The framework allows up to three attempts.
- Write artifacts to `artifacts/experiments/<id>/`

Read the results:
```bash
# Get the experiment ID from the output, then:
cat artifacts/experiments/<id>/metrics.json | python3 -m json.tool
cat artifacts/experiments/<id>/diagnostics.json | python3 -m json.tool
```

### Step 5 — Compare to champion

```bash
autoresearch compare-to-champion <experiment_id>
```

This runs 30 paired resamples + 1000 bootstrap iterations + Bonferroni-adjusted 90% CI. If all 8 gate checks pass, the challenger is promoted automatically.

Read the decision:
```bash
cat artifacts/comparisons/<comparison_id>/promotion_report.json | python3 -m json.tool
```

If promoted: a holdout report is auto-written to `artifacts/milestone_reports/<comparison_id>.md`. **Read it** — it tells you the SV→holdout overfitting gap.

### Step 6 — Complete the structured research log

The framework generates
`artifacts/tracks/<track>/runs/<run-id>/RESEARCH_LOG.md`; do not edit it by
hand. Proposal, outcome, and metric fields come from registry state.

For a comparison that needs an LLM verdict, provide the scientific reflection
with the decision:

```bash
autoresearch --track <track> --run-id <run-id> record-decision <comparison-id> \
  --decision promote|local_promote|reject \
  --rationale "<decision justification>" \
  --interpretation "<what this result taught>" \
  --next "<next research direction>"
```

For an auto-rejected cycle, include the `previous_cycle_reflection` block shown
in the next handoff proposal template. If the auto-rejection is the final
cycle, complete it directly:

```bash
autoresearch --track <track> --run-id <run-id> record-cycle-reflection \
  --interpretation "<what this result taught>" \
  --next "<next direction or why the run should stop>"
```

---

## Foundation tabular models (TabPFN) — opt-in

Foundation tabular models predict by **in-context learning** (a single forward
pass over the training rows) instead of gradient training. They are an optional,
per-run capability — off unless the operator turns them on.

**Backend choice (important on Apple Silicon).** TabPFN is GPU-bound. Two backends:
- **`api` (recommended on a Mac)** — offloads inference to Prior Labs' GPU via
  `tabpfn_client`. A full comparison runs in ~1-2 min. Needs `TABPFN_TOKEN` and
  spends metered credits (see below). Set `AUTORESEARCH_TABPFN_BACKEND=api` for
  the run, or `params.backend="api"` per recipe.
- **`local`** — runs in-process. Benchmarked on the M3 Air: MPS crashes / is
  pathologically slow, and CPU predict is ~10-40s per 1000 rows, so a full
  108k-row comparison is ~1.5h — impractical. Local is only sensible for a CUDA
  box or small-data experiments.

**API credit budget.** Credits scale ~linearly with rows scored (~3-4 credits per
scored row, all-in). The framework scores the whole frame (train, for
calibration, **plus** search-validation) each fit, and a comparison refits ~5×
(up to ~13× on escalation). A full-scale comparison is therefore very roughly
**2-10M credits**; the default daily quota is 50M, so budget for **a handful of
full comparisons per day**. Watch usage at
<https://ux.priorlabs.ai/account/usage>. To economise, keep `max_context_rows`
modest (the per-scored-row cost dominates, not the context) and prefer fewer,
decisive experiments.

**Enabling them (operator).**

1. Install the extra. The **default is API-only and torch-free** — a thin HTTP
   client, no local ML stack, so it cannot hit the macOS OpenMP/libomp crash and
   is the recommended install on any Mac:
   ```bash
   pip install -e '.[foundation]'                 # api backend only (no torch)
   python scripts/setup_foundation_models.py      # checks the client + Prior Labs token
   ```
   The only gate for the API is a **Prior Labs API key**: register at
   <https://ux.priorlabs.ai>, accept the licence, copy the key from
   <https://ux.priorlabs.ai/account>, and `export TABPFN_TOKEN=<key>` (put it in
   `~/.zshenv` so non-interactive research shells inherit it).

   **Local backend (optional, GPU boxes only).** `pip install -e
   '.[foundation-local]'` adds `torch` + the local `tabpfn` for in-machine
   inference. On Apple Silicon it is both slow and OpenMP-fragile (torch's libomp
   clashes with LightGBM's — the framework preloads LightGBM to order the imports,
   but the API path avoids the issue entirely by not installing torch). The local
   backend also needs the **HuggingFace** weight gate (accept terms at
   <https://huggingface.co/Prior-Labs/tabpfn_3> + `hf auth login`), and on
   python.org macOS Python an SSL `CERTIFICATE_VERIFY_FAILED` in the licence check
   is fixed once with `/Applications/Python <ver>/Install Certificates.command`.
2. Start the run with the opt-in flag (and, on a Mac, select the API backend):
   ```bash
   export AUTORESEARCH_TABPFN_BACKEND=api   # recommended on Apple Silicon
   autoresearch --track <t> --new-run bootstrap-track \
     --model-provider <p> --model-name <m> --cycles <N> --enable-foundation-models
   ```
   The flag is written into `run_manifest.json`; every later command for the run
   re-reads it. The `tabpfn` estimator then appears in the recipe menu, the
   handoff, and validation. Without the flag (or without the extra installed) it
   is invisible and recipes naming it are rejected. Dev override:
   `AUTORESEARCH_FOUNDATION_MODELS=1`.

**Using it (agent).** It is a normal recipe estimator:
```json
{"structure": "direct", "estimator": "tabpfn", "objective": "squared_error",
 "encoding": "ordinal", "params": {"backend": "api", "max_context_rows": 20000}}
```
Key differences from the gradient estimators:
- **Objective is `squared_error` only** (TabPFN has a single regression head).
- **No early stopping.**
- **Context is subsampled** to `max_context_rows` (default 40000, drawn with
  probability ∝ exposure). Exposure enters through the subsample, not a sample
  weight; framework calibration fixes the aggregate level as usual. Override the
  strategy with `subsample_strategy: "uniform"`.
- **Backend:** `params.backend` (`api`/`local`) or `AUTORESEARCH_TABPFN_BACKEND`.
  On the API a fit+score is ~15-20s; locally on a Mac it is minutes-to-hours.
- Keep `max_context_rows` modest (default 20000 on the API, 40000 local) and
  leave budget headroom — a comparison refits ~5× and up to ~13× on escalation.
- Strong fit for the **severity** stage of a `frequency_severity` recipe: claim
  rows (~25k) sit under the cap and are used unsubsampled.
- Curated params: `backend`, `max_context_rows`, `subsample_strategy`,
  `random_state`, `n_estimators`, `device`, `predict_batch_size`, plus the
  thinking-mode params below. Else → a script.

**Thinking mode (api backend only).** Extra fit-time compute for higher precision
(the "TabPFN-3-Plus" behaviour). Params:
- `thinking_effort`: `"medium"` or `"high"` (setting it enables thinking).
- `thinking_mode`: `true` = effort `"medium"` if no effort given.
- `thinking_timeout_s`: fit budget in seconds, client-capped at 2400 (40 min).
  This is the knob to sweep for "more thinking".
- `thinking_metric`: what it optimises toward. **Use `"spearmanr"`** — a rank
  metric, the closest available to exposure-weighted Gini (the promotion gate).
  `"mse"`/`"rmse"`/`"mae"`/`"r2"`/`"smape"` are also valid but optimise a level
  metric, not ranking, so they may not move Gini.

Thinking draws from a **separate Prior Labs daily budget** and is much slower per
fit. Because a fit can run up to 40 min, run thinking experiments under
`--config configs/frugal_thinking.toml` (single_partition gate + `[compute]
enforce = false`, so the framework's wall-clock alarm does not kill a long fit),
and give the agent harness a generous command timeout. Example recipe:
`{"structure":"direct","estimator":"tabpfn","objective":"squared_error","encoding":"ordinal","params":{"backend":"api","n_estimators":1,"max_context_rows":4000,"thinking_effort":"high","thinking_metric":"spearmanr","thinking_timeout_s":120}}`

**Machine defaults.** The 40000-row default is tuned for the M2 Pro 32GB. On the
M3 Air 16GB, set `max_context_rows` to ~15000 in the recipe. Re-run
`scripts/benchmark_foundation.py` on a machine to confirm the fit+predict fits
~1/5 of the compute budget before relying on a larger context.

**Licence.** TabPFN weights are licensed for research / internal evaluation
(incl. benchmarking on proprietary data) and are distributed via a gated
HuggingFace repo (accept terms + `hf auth login`). Commercial/production use
needs a separate Prior Labs enterprise licence — out of scope for research runs.

---

## Dataset schema (french_motor)

Each dataset's authoritative schema is `data/datasets/<name>/metadata/dataset_schema.json`,
summarised in the handoff's feature list and "Active dataset" block. The table
below documents the default French Motor dataset as a worked example.

| Column | Role | Notes |
|--------|------|-------|
| `record_id` | Framework ID | Stable join key copied from the configured source identifier |
| `IDpol` | Source ID | Source policy identifier; never use as a predictive feature |
| `Exposure` | Offset | Policy duration in years. Use only for exposure weights, response denominators, and multiplying predicted rates back to target totals; do **not** use as a predictive feature because it is unavailable at quote time. |
| `VehPower` | Feature | Vehicle power |
| `VehAge` | Feature | Vehicle age |
| `DrivAge` | Feature | Driver age |
| `BonusMalus` | Feature | Bonus-malus value |
| `VehBrand` | Feature | Vehicle brand |
| `VehGas` | Feature | Fuel type |
| `Area` | Feature | Area |
| `Density` | Feature | Population density |
| `Region` | Feature | Region |
| `ClaimNb` | Target | Count of claims |
| `ClaimAmountCount` | Target | Number of observed claim-amount records |
| `ClaimAmount` | Target | Raw claim cost |
| `ClaimAmountCapped` | Target | Capped claim cost (active training target in burning-cost mode) |

---

## Safety rules — never break these

1. **Never read the holdout vault.** Do not import from `autoresearch.data.holdout_vault` in model or feature files. Do not reference `milestone_holdout`, `holdout_vault`, or `AUTORESEARCH_MILESTONE_TOKEN` in your code. The integrity scanner will catch this and fail the experiment.

2. **Never edit protected files.** These files define the evaluation and promotion logic:
   - `src/autoresearch/evaluation/metrics.py`
   - `src/autoresearch/evaluation/resampling.py`
   - `src/autoresearch/data/holdout_vault.py`
   - `src/autoresearch/experiment_registry/registry.py`
   
   If you edit them, comparisons will be blocked until the user runs `autoresearch update-integrity-manifest`. Only edit them to fix genuine bugs, and document why in your research log.

3. **Always pass pytest.** The experiment runner won't proceed if tests fail. Fix failures before running new experiments. New code should have tests.

4. **Never mutate `split_pack.csv` or `data/datasets/<name>/processed/`.** The split is fixed. Reproducibility depends on it.

5. **Never change the primary metric or promotion gate thresholds** in a proposal or experiment config. These are controlled by `configs/default.toml` and the protected registry.

6. **Never change the claim cap.** It is fixed at 100,000 and applied identically to training, search-validation, and milestone-holdout rows. The search space lists `claim_cap_thresholds = [100000]` and proposals that diverge from this will be rejected. Every model is evaluated against the same capped target so cycles remain comparable.

---

## Useful commands reference

```bash
autoresearch list-experiments              # all registered experiments (current track)
autoresearch list-champion-history         # champion evolution (current track)
autoresearch list-promotions               # all comparison decisions
autoresearch session-status               # current session state
autoresearch export-context               # refresh context bundle
autoresearch evaluate-milestone <id>      # manual holdout eval (needs token)
autoresearch update-integrity-manifest    # accept intentional protected-file changes
pytest --tb=short -q                      # run test suite
```

---

## Research tracks — isolation between agents

Each agent (Claude, Codex, or any future platform) **must** run under its own
named track.  Tracks are fully isolated: separate registry, separate artifacts
directory, separate research log.  An agent in one track cannot see the
experiments, champion history, or metrics of any other track.

### Starting a track session

```bash
# One-command setup for a new isolated run (model identity flags required)
autoresearch --track <your-agent-name> --new-run bootstrap-track \
  --model-provider <provider> --model-name <model-name>

# Replace 'claude' with the agent identifier for your session
autoresearch --track <your-agent-name> init-registry
autoresearch --track <your-agent-name> run-all-baselines
autoresearch --track <your-agent-name> init-official-champion
autoresearch --track <your-agent-name> export-context   # read this at session start
autoresearch --track <your-agent-name> start-session main
autoresearch --track <your-agent-name> run-session-cycles 10
```

`bootstrap-track` is idempotent. It prepares shared data if needed, creates or
migrates the run registry, runs baselines only when the run has no
experiments, initializes the official champion if missing, writes proposal
templates, and exports the latest context bundle. Use it at the start of a new
ClaudeCode/Codex conversation when you want the agent to configure its own run.

All standard commands accept `--track <name> --run-id <id>`. New agent runs
should use `--track <tool-name> --new-run bootstrap-track`, which creates a
timestamped run id. Capture that id and pass it explicitly thereafter. If
`--run-id` and `--new-run` are both omitted, the command continues the track's
latest run; use this implicit resolution only for the first command of an
explicit continuation request. Without `--track`, commands operate on the
legacy default paths (backward-compatible).

Tracked run layout:

```text
artifacts/tracks/<track>/runs/<run-id>/
  registry.sqlite
  RESEARCH_LOG.md
  run_manifest.json
  context/
  handoffs/
  proposal_inbox/
  proposal_processed/
  results/
  iterations/
    000_bootstrap/
    001_<proposal-id>/
      proposal/
      experiment/
      comparison/
```

Anything created for a run should live under that run directory. If a run goes
wrong, deleting `artifacts/tracks/<track>/runs/<run-id>/` clears its registry,
context, proposals, experiments, comparisons, and logs.

### What is isolated per track

| Isolated (per track run) | Shared (all tracks/runs) |
|----------------------|---------------------|
| SQLite registry | Raw & processed data |
| Artifacts (experiments, comparisons) | Fixed split pack |
| Champion state & history | Protected eval code (metrics.py etc.) |
| Research log | Holdout vault |
| Proposal inbox & context bundle | Model/feature source files |

### Comparing tracks (human-only operation)

After each agent has run independently, a human can compare them:

```bash
autoresearch compare-tracks claude codex
# Writes a full report to artifacts/cross_track/<timestamp>/comparison_report.md
# No promotion is performed.

autoresearch list-tracks   # see all tracks and their current champion
```

### Safety rules for tracked sessions

6. **Always pass `--track <your-agent-name>` to every command.** Use `codex`
   for Codex, `claude` for Claude Code, and `opencode` for OpenCode. The
   run-scope guard only binds research sessions to those three folders. Use
   `--new-run` only on the first command of a fresh run, capture the returned
   timestamped run id, and pass `--run-id` on every later command. Resolve the
   latest run implicitly only when the user explicitly requests continuation,
   then pin the resolved id. Running without `--track`, with `--track default`,
   or with a custom track name is reserved for human/admin analyst operations.

7. **Never read another track's context bundle.**  The files under
   `artifacts/tracks/<other-agent>/` are off-limits during your session.

### Run-scope isolation (enforced — not just advice)

To keep every run an *independent* experiment, a bound research session may only
touch **its own** run folder. This is enforced at the harness layer by a shared
guard (`scripts/run_scope_guard.py`) wired into **all three harnesses** —
Claude Code (`.claude/settings.json`), Codex (`.codex/hooks.json`), and OpenCode
(`.opencode/plugins/run-scope-guard.js`) — so the rule holds whichever agent runs
the track, not left to good behaviour:

- **Bootstrap binds you; confinement starts then.** The moment you run your first
  `autoresearch --track <you> ... bootstrap-track` (or `start-session`) command,
  the session is automatically *bound* to exactly that run. **From that point on**,
  any attempt to `Read`/`Grep`/`Glob`/`ls`/`cat` a path under
  `artifacts/tracks/<...>/runs/<other-run>/` — in *any* track, including your own
  — is **blocked** by the harness, and enumerating the `runs/` directory is
  blocked too (you cannot even list sibling runs).
- **Before you bootstrap, run artifacts are blocked.** A session may inspect
  source, docs, configs, and tests before it binds, but raw access to
  `artifacts/tracks/<...>/runs` is denied until the first valid
  `autoresearch --track <you> ... bootstrap-track` (or `start-session`) command
  binds the session. Bootstrap, *then* inspect your own run artifacts.
- **Everything else is unaffected.** Source (`src/`), data, configs, tests, and
  your *own* run folder are fully accessible. The framework's `autoresearch`
  commands are already run-scoped, so they keep working normally.
- **Cross-run knowledge reaches you only through memory.** If you "know" about a
  prior run, it is because the optional memory feature surfaced it — never
  because you read another run's files. Do not try to; once bound, the attempt
  will be denied.
- **Parallel runs are safe.** Scope is keyed to the session, so two runs in two
  threads never see each other.

**Analysis sessions** (a human wanting to compare/inspect many runs) are exempt:
launch the agent with `AUTORESEARCH_SCOPE=analyst` in the environment and the
guard allows access to every run. Use this only for deliberate cross-run
analysis, never for an experimental run.

---

Note: the claim cap is fixed at 100,000; do not propose alternative thresholds or disabling capping.
