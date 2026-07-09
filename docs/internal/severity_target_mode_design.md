# Design: a `severity` target mode (model & assess on claim rows only)

## Question answered

**Can severity be reached by prompt/LLM changes alone?** No. Today the harness
has exactly two target modes — `burning_cost` and `frequency`
(`src/autoresearch/targets.py`). "Severity" exists only as the *inner stage* of a
`frequency_severity` recipe: it fits cost-per-claim on claim rows
(`interpreter.py:_fit_frequency_severity`), but the experiment is still **scored
on the full population** against burning cost. There is no code path that trains
*and evaluates* on claim rows only, weights the metric panel by claim count, or
gates/holdout-checks a severity model. That requires framework work, some of it
in the protected evaluation files. An LLM prompt cannot create it.

## The core insight that keeps the change small

Every metric in the protected panel already consumes a **generic weight** — the
prediction-frame column literally named `exposure` — and a generic
`actual_target`/`predicted_target` pair. Nothing in `metrics.py`,
`resampling.py`, or `validation.py` knows that `exposure` is *Exposure*; it just
treats it as "the weight, and the denominator that turns a total into a rate."

Severity is the same shape with two substitutions:

| concept            | burning_cost / frequency | **severity**            |
|--------------------|--------------------------|-------------------------|
| population         | all rows                 | rows with `ClaimNb > 0` |
| weight / offset    | `Exposure`               | `ClaimNb` (claim count) |
| target total       | `ClaimAmountCapped`      | `ClaimAmountCapped`     |
| rate = total/weight| pure premium (cost/exp)  | **severity** (cost/claim) |
| calibration        | Σcost / Σpred            | Σcost / Σpred (unchanged) |

So if the **dispatcher** (a) filters to claim rows and (b) writes `ClaimNb` into
the prediction frame's `exposure` (weight) slot for severity mode, then the
entire metric/gate/holdout machinery works **unchanged** — Gini weighted by
claim count, APL on the severity rate, calibration on cost. That confines the
change to a handful of non-protected files and *ideally zero edits to the
protected evaluation modules*.

## What changes

### 1. `targets.py` — add the mode and generalise `TargetSpec` (non-protected)

Add `SEVERITY = "severity"` to `VALID_TARGET_MODES` and a third `TargetSpec`.
Extend `TargetSpec` with two new fields that make the weight/population explicit
instead of assuming Exposure everywhere:

```python
weight_column: str          # "Exposure" for burning_cost/frequency, "ClaimNb" for severity
population: str              # "all" or "claim_rows" (ClaimNb > 0)
```

`SEVERITY` spec: `source_column="ClaimAmountCapped"`, `weight_column="ClaimNb"`,
`population="claim_rows"`, `rate_label="severity"`, rate columns
`actual_severity`/`predicted_severity`, aliases `actual_claim_cost` /
`predicted_claim_cost` (cost totals, same as burning cost),
`default_primary_metric="gini_weighted"`. Burning_cost/frequency get
`weight_column="Exposure"`, `population="all"` (behaviour identical to today).

### 2. `models/dispatcher.py` — filter + reweight at the single chokepoint (non-protected)

Both `dispatch_model` and `dispatch_model_on_explicit_frames` build the
prediction frame here. Add a shared helper that, for `spec.population ==
"claim_rows"`, filters `train` and `score` to `ClaimNb > 0` **before** the model
call, and populates the frame's weight slot from `spec.weight_column`:

- `train = train[train[CLAIM_COUNT] > 0]`, likewise `score`, when severity.
- `exposure = score[spec.weight_column]` (claim count for severity) — this is the
  value written to the frame's `"exposure"` column and used by every downstream
  metric as the weight.
- `_finalize_predicted`: pass `exposure_column=spec.weight_column` (currently
  hardcoded `EXPOSURE` at `dispatcher.py:283`) so rate→total is
  `severity_rate × ClaimNb = cost`, and calibration is Σcost/Σpred_cost.
- Add `predicted_severity`/`actual_severity` derived columns for the panel; keep
  `predicted_claim_cost = clipped_target` (cost total) so existing cost-alias
  consumers keep working.

Because CV refits flow through `dispatch_model_on_explicit_frames` (via
`cv_factory._factory`) and the holdout eval through `dispatch_model`, putting the
filter here means **every surface — search eval, CV gate, holdout — filters
identically and automatically.** The fold splitter in `resampling.py` still
partitions the full frame, but only claim rows survive scoring, so the paired
merge on `record_id` naturally yields claim rows only. No change needed there.

### 3. `targets.py` rate columns → `metrics.py` panel (check, likely non-protected)

`full_metric_panel` already derives `actual_rate = actual/exp`,
`predicted_rate = predicted/exp` generically and writes
`spec.mean_actual_rate_key` etc. The only mode-specific branch is the
legacy `rmse_pure_premium`/`rmse_frequency` naming (`metrics.py:129-135`) — add a
`severity` arm writing `rmse_severity`/`mae_severity`. This is a **1-line-ish
addition in a protected file** (`metrics.py`). If we want strictly zero protected
edits, fall back to the generic `rmse_rate`/`mae_rate` keys the panel already
emits and skip the legacy alias — then `metrics.py` needs no change at all.

**Recommendation:** do the zero-protected-edit path first (generic keys only);
add the legacy alias later via `update-integrity-manifest` if dashboards need it.

### 4. `models/global_mean.py` — severity baseline (non-protected)

The champion of every run starts as `global_mean`. Add a severity arm: mean
severity = Σcost / Σcount on claim training rows, predicted as a flat
severity-rate × `ClaimNb`. Mirrors the existing frequency/burning arms; the
dispatcher has already filtered to claim rows so `train`/`score` here are
claim-only.

### 5. `models/recipe/schema.py` + `interpreter.py` — legal recipes for severity (non-protected)

- `schema.py`: for `target_mode == severity`, only `structure="direct"` is legal
  (reject `frequency_severity` — it's a burning-cost decomposition). Legal
  objectives are the strictly-positive ones (`gamma`, `squared_error`, and
  `tweedie` degenerates to gamma at p→2) since severity `> 0` on claim rows.
  `_stage_target` returns a new `"severity"` target label.
- `interpreter.py`: add the `direct` severity branch to `fit_predict` —
  `y_rate = cost / ClaimNb` on the (already claim-filtered) train frame, weighted
  by `ClaimNb`. This is exactly the existing severity-stage math at
  `interpreter.py:243-245`, lifted to a top-level direct structure.

### 6. `config.py` — accept the mode (non-protected)

`normalise_target_mode` already validates against `VALID_TARGET_MODES`, so
`target_mode: severity` in the evaluation config block works once (1) lands.
Set it in the run's project config to activate a severity run.

### 7. Controller prompt / handoff surface (non-protected)

- `controller/handoff.py`, `proposal_schema.py`, `champion.py`, `context.py`:
  surface `severity` in the target-mode line, the estimator/objective menu
  (positive-target objectives only), and the target-strategy mapping
  (`direct_severity → direct`). Add a target→objective hint: *severity (claim
  rows, cost/claim, > 0) → gamma / squared_error*.
- `CLAIM_COUNT` becomes an **offset/weight, never a feature** in severity mode —
  add it to the interpreter/script leakage set for that mode (it already is in
  `_LEAKAGE`), and note in the prompt that the population is claim rows only.

## What deliberately does **not** change

- **The three protected metric files (`metrics.py`, `resampling.py`,
  `validation.py`) need no logic change** on the recommended zero-alias path,
  because the weight is threaded through the generic `exposure` frame column.
  The `validation.py` `exposure_positive` check passes automatically (`ClaimNb ≥
  1` on claim rows).
- Split pack, claim cap, primary metric, gate thresholds — untouched.
- `comparison_runner.py` / holdout vault — untouched; they read `config.target_mode`
  and the frame's generic columns, both already mode-agnostic.

## Risks / edge cases to handle in implementation

1. **Fold assignments are exposure-stratified over the full population.** Claim
   rows (~a few % of policies) scatter thinly across folds; a fold could get few
   claim rows → noisy per-fold Gini. Mitigation: acceptable at first (the gate is
   CV-bootstrap and reports variance); a later refinement is a severity-specific
   fold assignment stratified on claim rows.
2. **Zero-claim-row folds / empty severity train.** The interpreter already
   raises on empty severity train; ensure the dispatcher's pre-filter produces a
   clear error rather than a downstream crash if a fold has no claim rows.
3. **Gamma needs strictly positive.** `ClaimAmountCapped` can be 0 on a claim row
   (rare, capped/settled-at-zero). Reuse the interpreter's existing
   positive-mask guard (`interpreter.py:246-251`) for the direct severity branch.
4. **Calibration semantics.** Σcost/Σpred_cost on claim rows — sound; it targets
   total claim cost among claimants, which is the right severity aggregate.
5. **Dashboards / reporting** that assume `predicted_pure_premium` exist should
   fall back gracefully; severity emits `predicted_severity` + the cost aliases.

## Rollout order (smallest safe increments)

1. `targets.py` spec + `config.py` (mode exists, validates).
2. `dispatcher.py` filter + weight threading (+ `_finalize_predicted` exposure_column).
3. `global_mean.py` severity baseline.
4. `recipe/schema.py` + `interpreter.py` direct-severity.
5. Prompt/handoff surfacing.
6. Tests: a severity run end-to-end (bootstrap → global_mean champion →
   direct gamma challenger → CV gate → holdout), asserting the panel weights by
   claim count and the population is claim rows only.
7. *(Optional, later, operator-gated)* legacy `rmse_severity` alias in
   `metrics.py` + `update-integrity-manifest`.

After steps 1–5 an LLM run **can** be launched purely by setting
`target_mode = severity` in the project config and pointing it at the updated
prompt — no per-experiment code needed for standard recipes.
