# AGENT.md — Auto-Research Operating Manual

You are the research agent for an autonomous insurance target-modelling loop on the French Motor dataset (freMTPL2, ~678K policies). Burning cost is the default target; claim frequency is used only when the user or run configuration explicitly selects `target_mode = "frequency"`. Your goal is to progressively improve predictions measured by **exposure-weighted Gini** on the search-validation split, ultimately assessed on a protected holdout on every promotion.

Read this file at the start of every session. Keep it open as reference.

---

## Cheat sheet — gotchas & tips
<!-- USER-MAINTAINED: add new tips here as they come up. Keep entries short and concrete. -->

### Library × loss capability matrix (target has exact zeros)

The burning-cost target (`claim_cost_capped_active`) **contains exact zeros** — most policies have no claim. Losses requiring strictly positive `y` (gamma, log) will error or need a frequency/severity split.

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

- Always multiply predicted rates by `exposure_term_a` to return totals.
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

- **Use the active run's research tree.** The handoff context contains a `research_tree` for this run only. Choose a `research_parent_node_id` when a new idea builds on a prior hypothesis, near-miss, auto-rejection, or informative failure. Use `null` for a genuinely new line of attack. Do not use other runs as proposal evidence unless the user explicitly asks for cross-run analysis.
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

**Primary KPI & gate metric**: `gini_weighted` — higher is better. The exposure-weighted Lorenz-area Gini is both the headline business KPI and the metric the `cv_bootstrap` gate ranks challengers on (win rate, lift, escalation trigger). `rank_gini_weighted` is still computed alongside it for reference (bounded-influence Somers' D), but it no longer drives the gate.

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
should model rates internally if useful, then multiply by `exposure_term_a`
before returning.

---

## You own the decision

After every comparison, the framework writes `decision = "pending_llm"`. **You must review the metric summary and call `record-decision` before the cycle can advance.** The mechanical advisory gates are informative, not binding.

### What to review before deciding
1. **Full metric table**: check `gini_weighted`, `rank_gini_weighted`, `asym_pricing_loss`, calibration ratio.
2. **Advisory gate panel**: did the challenger pass or fail the configured thresholds?
3. **Guardrail status** (shown in the report banner): any hard-fail blocks promotion regardless of your choice.
4. **Escalation**: if win rate was in the close-call band [0.40, 0.60], escalation added extra partitions — the post-escalation win rate is the one to read.
5. **Asymmetric Pricing Loss (APL)**: lower is better. `asym_pricing_loss` penalises under-pricing 4× over-pricing. A challenger with a good Gini but high APL is writing profitable policies in the wrong segments.

### How to record your decision
```bash
autoresearch --track <track> record-decision <comparison_id> --decision promote --rationale "Challenger improved gini_weighted by X and reduced APL by Y, indicating better rank discrimination and safer pricing."
autoresearch --track <track> record-decision <comparison_id> --decision reject  --rationale "Win rate 0.48 in the close-call band even after escalation; insufficient evidence."
```

The comparison_id appears in the `compare-experiments` output and in `list-promotions`.

On `promote`: guardrails are re-checked; hard fails block the promotion with an error message. On success, the holdout evaluation fires automatically.

---

## Asymmetric Pricing Loss (APL)

`asym_pricing_loss` = Σ w·(4·under + 1·over) / Σ w, where under = max(actual_rate − predicted_rate, 0) and over = max(predicted_rate − actual_rate, 0). Lower is better (not in HIGHER_IS_BETTER_METRICS).

The 4:1 ratio reflects the economic reality that a policy written at a loss generates claims that exceed the premium, costing ~4× more than a missed quote (which only loses margin). A model that systematically under-prices high-risk segments will have a high `asym_pricing_loss` even if its Gini looks acceptable.

Diagnostic sub-metrics: `apl_under_cost` (mean exposure-weighted shortfall), `apl_over_cost` (mean excess), `apl_under_over_ratio` (realised under/over balance — close to 4 is expected at optimal pricing).

---

## Gate modes — how comparisons are adjudicated

The comparison gate has three modes, configured via `gate_mode` in `default.toml`.

### `cv_bootstrap` (default)

Generates a **unique fold partition per run** (seed derived from run_id), then bootstrap-resamples each fold ×20. All (fold × bootstrap) samples contribute to the win rate and CI. Gate metric: `gini_weighted`.

- Base comparison: 1 partition × 4 folds × 20 bootstrap = **80 samples**.
- Close call (win rate in [0.40, 0.60]): escalation adds 2 extra partitions → **240 samples**.
- Champion fold predictions are **cached** — only the challenger needs to be refit (4 fits vs old 16–32).
- **~8× cheaper** than the old `repeated_cv` default on the common path.

### `repeated_cv` (legacy)

Refits both models on cv_n_repeats × cv_folds stratified partitions. Costs cv_n_repeats × cv_folds refits per comparison per model. Gate metric: `rank_gini_weighted`. Use when you need to compare against pre-cached repeated-CV results.

### `single_partition` (legacy / fast)

Evaluates on the fixed `search_validation` split with 30 bootstrap resamples. CI measures within-split noise only. Use for quick sanity checks or expensive-to-refit models.

### Single-split screening

Before the expensive CV/bootstrap comparison, every valid challenger is screened once on the full `search_validation` split. This is a low hurdle, not a promotion gate: clearly worse challengers are auto-rejected, while similar or better challengers proceed to full comparison and LLM decision. Auto-rejections are still written to the research tree and non-promotion summaries; read them as evidence for the next proposal.

---

## Quick-start — how to interpret short user instructions

Your default track is your current tool name: **`codex`** when running in Codex, **`claude`** when running in Claude Code. Your default cycle count is **3**.

For a **new run**, always pass `--new-run`; this creates a fresh timestamped folder such as `artifacts/tracks/codex/runs/20260527T211530Z/`. For **continue** instructions, omit both `--new-run` and `--run-id` so the framework picks up that track's latest run. Only pass `--run-id` when the user explicitly supplies one.

| User says | What to do |
|-----------|-----------|
| "Go!" / "Start" | Bootstrap a new timestamped run in your tool track, read handoff, run 3 cycles |
| "Run X experiments" | Bootstrap a new timestamped run in your tool track, read handoff, run X cycles |
| "Continue" / "Keep going" | Read handoff, run 3 cycles (skip bootstrap) |
| "Continue and run Y" | Read handoff, run Y cycles (skip bootstrap) |
| "Bootstrap only" | Bootstrap and read handoff, then stop |

**Bootstrap** — run once at the start of every fresh conversation. Model identity is required for run attribution:
```bash
autoresearch --track <codex-or-claude> --new-run bootstrap-track \
  --model-provider <provider> --model-name <model-name>
# e.g. --model-provider anthropic --model-name claude-sonnet-4-6
# e.g. --model-provider openai   --model-name codex-mini-latest
```

**Read handoff** — always do this after bootstrap or at the start of a continuing session:
```bash
# The handoff path is printed by bootstrap — it ends in handoffs/latest_handoff.md
# Read it to understand current champion state before proposing anything.
```

**Run N cycles** — `run-session-cycles` requires an active session. On a fresh run, create one first (idempotent name is fine):
```bash
autoresearch --track <codex-or-claude> start-session main         # only needed once per run
autoresearch --track <codex-or-claude> run-session-cycles <N>
```

**Each cycle now pauses for your decision.** `run-session-cycles` runs a proposal through experiment + comparison and then **stops in the `awaiting_decision` state** — it does not auto-promote. You must review the metric summary and call `record-decision` (see "You own the decision") before the next cycle. So to run N experiments you loop: `run-session-cycles 1` → review → `record-decision …` → repeat. A larger N still stops after the first comparison that needs a verdict.

If the user supplies a specific `--run-id` (e.g. `CC20260526_01`), pass it to every command. Otherwise use `--new-run` for fresh starts and omit `--run-id` for continues.

---

## Session start — always do these first

For a fresh run:

```bash
autoresearch --track <codex-or-claude> --new-run bootstrap-track      # fresh run with timestamped folder
autoresearch --track <codex-or-claude> start-session main             # idempotent name; required before run-session-cycles
autoresearch --track <codex-or-claude> list-champion-history
autoresearch --track <codex-or-claude> list-experiments
```

For a continuing run, skip `bootstrap-track` and omit `--new-run`:

```bash
autoresearch --track <codex-or-claude> start-session main
autoresearch --track <codex-or-claude> list-champion-history
autoresearch --track <codex-or-claude> list-experiments
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

The inbox JSON has a strict schema. Before writing your first proposal in a fresh run, read `proposal_inbox/proposal_template.json` *once*. The required top-level keys include `parent_experiment_id`, `parent_branch_id`, `branch_action`, `experiment_config`, `change_summary`, `expected_benefit`, `key_risk`, `rationale`, `proposal_id`, `experiment_name`. Slimmer JSON is rejected at ingestion and counts as a wasted cycle.

### C. Inbox audit (prevents re-submitting stale stubs)

`ls proposal_inbox/` at session start. Any pre-existing `model_*.py` may contain bugs from prior sessions (wrong target column, exposure incorrectly used as a predictive feature, outdated calibration). Either:
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

### F. Use the champion template (eliminates boilerplate)

`proposal_inbox/champion_template.py` is a parameterised version of the current champion. New proposals should import or copy-then-diff from it rather than restating ~100 lines of identical scaffolding. Each verbatim re-write of the champion costs ~3K output tokens for zero information gain.

### G. Axis rotation & approach diversity

Track which **axis** each experiment changes (hyperparameter / preprocessing / target / features / family). After 2 same-axis experiments — promoted or not — rotate to a different axis. This prevents the failure mode where 3+ consecutive experiments all tune the same dial.

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

**Option A: Autonomous proposal with a run-local model script**
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

If the model predicts pure premium or annual claim frequency, multiply by
`score["exposure_term_a"]` before returning.

**Option B: New feature engineering module**
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
- `EXPOSURE = "exposure_term_a"`
- `CLAIM_COST = "claim_cost_capped_active"` (training target)
- `CLAIM_COUNT = "claim_count_signal_q"`
- `CLAIM_EVENTS = "claim_event_count_l"`
- `RECORD_ID = "record_id"`

**Calibration — always apply**

Every model script must apply a training-total calibration scalar before
returning predictions.  This is a single aggregate correction (one degree of
freedom, no leakage risk) that guarantees the aggregate gate passes and
preserves visibility of the model's native bias in the comparison report.

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
  `repair_request_*.json`, write the next `model_attempt_N.py`, and rerun. The
  framework allows up to three attempts.
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

### Step 6 — Update the research log

Append to `artifacts/tracks/<track>/runs/<run-id>/RESEARCH_LOG.md` (this run's log only — do not read or write logs from other runs or prior sessions):

```markdown
## Cycle N — YYYY-MM-DD
**Hypothesis**: ...
**Changes**: ...
**Outcome**: promoted / inconclusive / failed
**Metrics**: SV Gini = X.XXXXX (vs champion Y.YYYYY, Δ = ...)
**Holdout**: (if promoted) Gini = X.XXXXX, SV→holdout gap = ±...
**Interpretation**: ...
**Next**: ...
```

---

## Dataset schema (anonymised)

| Column | Role | Notes |
|--------|------|-------|
| `record_id` | ID | Float; policy identifier |
| `exposure_term_a` | Offset | Policy duration in years. Use only for exposure weights, response denominators, and multiplying predicted rates back to target totals; do **not** use as a predictive feature because it is unavailable at quote time. |
| `vehicle_power_band_b` | Feature | Numeric 1–12 |
| `vehicle_age_band_c` | Feature | Numeric (years) |
| `driver_age_band_d` | Feature | Numeric (years) |
| `risk_score_index_e` | Feature | Numeric risk score |
| `vehicle_make_group_f` | Feature | Categorical (11 levels) |
| `vehicle_energy_type_g` | Feature | Categorical (2 levels: fuel type) |
| `territory_band_h` | Feature | Categorical (6 zones) |
| `density_index_i` | Feature | Numeric (1607 unique; urban density proxy) |
| `region_cluster_j` | Feature | Categorical (21 regions) |
| `claim_count_signal_q` | Target | Count of claims |
| `claim_event_count_l` | Target | Alternative claim count |
| `claim_cost_observed_k` | Target | Raw claim cost (£) |
| `claim_cost_capped_active` | Target | Capped claim cost (active training target in burning-cost mode) |

Raw mapping is private; use anonymised names only in model code.

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

4. **Never mutate `split_pack.csv` or `data/processed/`.** The split is fixed. Reproducibility depends on it.

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
autoresearch --track <codex-or-claude> --new-run bootstrap-track \
  --model-provider <provider> --model-name <model-name>

# Replace 'claude' with the agent identifier for your session
autoresearch --track <codex-or-claude> init-registry
autoresearch --track <codex-or-claude> run-all-baselines
autoresearch --track <codex-or-claude> init-official-champion
autoresearch --track <codex-or-claude> export-context   # read this at session start
autoresearch --track <codex-or-claude> start-session main
autoresearch --track <codex-or-claude> run-session-cycles 10
```

`bootstrap-track` is idempotent. It prepares shared data if needed, creates or
migrates the run registry, runs baselines only when the run has no
experiments, initializes the official champion if missing, writes proposal
templates, and exports the latest context bundle. Use it at the start of a new
ClaudeCode/Codex conversation when you want the agent to configure its own run.

All standard commands accept `--track <name> --run-id <id>`. New agent runs
should use `--track <tool-name> --new-run bootstrap-track`, which creates a
timestamped run id. If `--run-id` and `--new-run` are both omitted, the command
continues the track's latest run. Without `--track`, commands operate on the
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
   for Codex and `claude` for Claude Code. Use `--new-run` only on the first
   command of a fresh run, then omit `--run-id` to continue that track's latest
   timestamped run unless the user explicitly gives a run id. Running without
   `--track` writes to the shared default registry and is reserved for
   human/admin operations.

7. **Never read another track's context bundle.**  The files under
   `artifacts/tracks/<other-agent>/` are off-limits during your session.

---

Note: the claim cap is fixed at 100,000; do not propose alternative thresholds or disabling capping.
