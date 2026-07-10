<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with: python scripts/generate_agent_contract.py
     Source of truth: cli.COMMANDS, proposal_schema, dispatcher, integrity,
     configs/default.toml. Full human manual: docs/OPERATING_MANUAL.md. -->

# AGENT.md — Auto-Research Runtime Contract

You are the research agent for an autonomous tabular target-modelling loop on a
**per-run selected dataset**. The active dataset, target mode, column roles,
weight policy, and any fixed preprocessing (e.g. a claim cap) are printed in the
handoff's **"Active dataset"** block; read it first — those facts are binding.
Maximise **weight-weighted Gini** (`gini_weighted`) on the search-validation
split; every promotion is re-checked on a protected holdout. Each run starts with
the `global_mean` baseline as champion — beat a flat weighted rate first.

Escalation only — most runs never need it: **docs/OPERATING_MANUAL.md** has the full manual
(dataset schema, metric panel, gate modes, research-line mechanics, worked examples).

## Hard safety constraints — never break

1. **Never read the holdout.** Do not import `autoresearch.data.holdout_vault`
   or reference `milestone_holdout` / `AUTORESEARCH_MILESTONE_TOKEN` in model or
   feature code. The integrity scanner fails the experiment.
2. **Never edit protected evaluation files** (comparisons block until an
   operator runs `autoresearch update-integrity-manifest`):
- `src/autoresearch/evaluation/metrics.py`
- `src/autoresearch/evaluation/resampling.py`
- `src/autoresearch/evaluation/diagnostics.py`
- `src/autoresearch/evaluation/validation.py`
- `src/autoresearch/comparison_runner.py`
- `src/autoresearch/data/holdout_vault.py`
- `src/autoresearch/milestone.py`
- `src/autoresearch/experiment_registry/comparisons.py`
- `src/autoresearch/experiment_registry/champions.py`
3. **Never change** the primary metric, gate thresholds, the fixed split
   (`split_pack.csv`, `data/datasets/<name>/`), or any dataset-fixed
   preprocessing (e.g. the claim cap) — the handoff says what is fixed.
4. **Always pass pytest** — the runner refuses to proceed on a failing suite.
5. **Always stay in your own run.** Pass `--track <your-tool-name>`
   (`claude` / `codex` / `opencode`) and `--run-id <id>` to every command. After
   your first `bootstrap-track`/`start-session`, the harness blocks reads of any
   other run's `artifacts/tracks/.../runs/<other>` folder.

## The workflow (this is the only one)

A request to "run N experiments" / "start" / "test N ideas" means a **fresh
run** unless the user says "continue"/"resume" or gives a run id. Do not inspect
existing artifacts to infer intent.

1. **Bootstrap (fresh run only).** First shell command — binds the run scope:
   ```bash
   autoresearch --track <t> --new-run bootstrap-track \
     --model-provider <provider> --model-name <model-name> --cycles <N>
   ```
   `--cycles <N>` pins the requested experiment budget — the framework stops
   the run at N cycles so you never have to count.
   Capture the returned timestamped `run_id`; pass `--run-id <id>` thereafter.
   For an explicit continuation, skip bootstrap and resolve latest once with
   `start-session`, then pin the resolved id.
2. **Read the handoff** (`show-latest-handoff`) before forming any hypothesis.
   It is authoritative: champion, recent results and learnings, recommended tree
   actions, key constraints, and the exact proposal template are all embedded
   inline. Open this run's `RESEARCH_LOG.md` or `latest_context.json` only when
   you need older detail than the handoff already carries.
3. **Propose one idea per context refresh.** Write a proposal JSON + a
   neighbouring model script into **this run's** inbox. The handoff prints its
   exact path (`Inbox: ... ← write proposal JSON + model script here`); for a
   tracked run it is
   `artifacts/tracks/<track>/runs/<run-id>/proposal_inbox/` — **not** the
   repo-root `proposal_inbox/` (that legacy path is used only for untracked
   runs and stays empty here). The handoff prints the exact proposal template
   inline — copy it and fill the `<...>` fields; no separate schema-file read is
   needed. At most one queued proposal is ingested per refresh; extra JSON files
   stay deferred.
4. **Run + decide, looping one cycle at a time:**
   ```bash
   autoresearch --track <t> --run-id <id> run-session-cycles 1   # stops at awaiting_decision
   # review the metric summary, then:
   autoresearch --track <t> --run-id <id> record-decision <comparison_id> \
     --decision promote|local_promote|reject --rationale "..." \
     --reason-code clear_win|line_progress|noise|inferior|artifact_suspected|calibration|other \
     --interpretation "what the result taught" --next "next direction"
   ```
   `run-session-cycles` **never auto-promotes** — it always stops for your
   `record-decision`. Repeat propose -> cycle -> decide. **Design the next
   experiment only after reading the current one's result** — N is a budget of
   cycles to spend adaptively, not a slate to plan in advance (see "Adaptive
   search"). `record-decision` prints the post-decision champion and the next
   command in its own output, so you do **not** need to follow it with
   `show-latest-handoff`, `list-champion-history`, or `session-status` — those
   are for the rare case you need detail the decision output did not carry. On a
   mid-run refresh, prefer `show-latest-handoff --delta` (dynamic state only;
   the template and constraints are already in this contract).
5. **Research logging is framework-owned.** The framework writes hypothesis,
   changes, outcome, and metrics from registry state. Supply interpretation and
   next direction with `record-decision`. After an auto-rejection, put
   `previous_cycle_reflection` in the next proposal; if it was the final cycle,
   run `record-cycle-reflection --interpretation "..." --next "..."`.
   `RESEARCH_LOG.md` is generated from these structured records; do not edit it.

### Workflow commands

All take `--track <your-tool-name> --run-id <id>` (omit `--run-id` only on the
first `bootstrap-track`, which prints the id):

- `bootstrap-track` — Create + prepare a fresh run (first command of a new run).
- `start-session` — Open the supervised session for the run.
- `show-latest-handoff` — Read current champion/state before proposing.
- `list-experiments` — List this run's registered experiments.
- `list-champion-history` — Show how the champion evolved this run.
- `run-session-cycles` — Run N cycles; STOPS at awaiting_decision (no auto-promote).
- `record-decision` — Your verdict: promote | local_promote | reject.
- `record-cycle-reflection` — Complete the final reflection after an auto-rejected cycle.
- `park-research-line` — Park an exhausted research line.
- `clear-line-champion` — Drop a local incumbent that looks artefactual.
- `export-context` — Refresh the handoff/context bundle on demand.

### Orchestrated mode

If you were asked to **orchestrate** — run a campaign, delegate experiments to
sub-agents — you are not a research agent at all: read `ORCHESTRATOR.md` **before
running any command** and never run `bootstrap-track` (it would bind this session
to a research run and lock you out of every `orchestrate` command).

If your launch prompt names a **pre-bootstrapped run**, you are a sub-agent under
an orchestrator. Skip step 1: never run `bootstrap-track`, never start another
run, and pass the given `--track`/`--run-id` to every command. Your handoff
carries an **"Orchestration brief"** block — treat its direction, constraints,
and stop conditions as binding, on par with the Active dataset block, and spend
your cycles adaptively inside it. When the budget is exhausted (or a brief
stop-condition fires), finish with:

```bash
autoresearch --track <t> --run-id <id> orchestrate finish-delegation \
  --summary "<3–6 sentences: what you learned, what you'd try next, anything artifactual>"
```

and then stop. Your report is built from the registry either way — the summary is
your testimony, not your score.

### When a cycle needs repair

A cycle can stop in **`needs_repair`** (instead of `awaiting_decision`) when the
model fails preflight, the compute budget, or output validation. A negative lift
is **not** a repair trigger — such models proceed to screening. The framework
writes `repair_request_<N>.json` into the proposal directory. Recover
without guessing — the file tells you what to do via its `repair_kind`:

1. Read `repair_request_<N>.json`: `repair_kind`, `failed_checks`, `instruction`.
2. `repair_kind == "recipe"` → write the corrected recipe to
   `recipe_attempt_<N>.json` (framework still owns units/calibration; only write
   `model_attempt_<N>.py` if the recipe truly can't express the fix).
   `repair_kind == "script"` → write `model_attempt_<N>.py` (same `fit_predict`).
   Never touch holdout data.
3. Rerun `run-session-cycles 1` — it picks up the new attempt automatically.

You get up to **3 attempts**. Attempt 2 should move *opposite* the failure (if a
tree was too deep, go shallower — not halfway back); if attempt 2 is still no
better, let it fail rather than spend attempt 3 on a near-duplicate.

## Model: a recipe (preferred) or a script (escape hatch)

Specify the model one of two ways. Both flow through the same framework
units/calibration stage, so **you never write exposure conversion or calibration
yourself**. Prefer a recipe for any method the registry covers; drop to a script
only for a model the recipe vocabulary cannot express.

### Option A — declarative recipe (`model.recipe`), preferred

A validated object interpreted by trusted code. Set `model_family = "recipe"`:
```json
{"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
 "encoding": "native_categorical", "params": {"num_leaves": 63}, "early_stopping": 50}
```
Champion follow-up (do not repeat the full recipe):
```json
{"recipe_ref":"champion","recipe_overrides":{"params":{"num_leaves":31}}}
```
The controller expands it before validation; feature selectors remain sibling fields.
Frequency × severity nests two stages (only when the dataset has a claim-count column):
`{"structure":"frequency_severity","stages":{"frequency":{"estimator":"lightgbm","objective":"poisson"},"severity":{"estimator":"lightgbm","objective":"gamma"}}}`

Structures: `direct`, `frequency_severity`. Estimators (only these obj × enc combos are legal —
invalid ones are rejected before running):
- **constant** — obj ['gamma', 'poisson', 'squared_error', 'tweedie']; enc ['native_categorical', 'one_hot', 'ordinal']
- **elasticnet** — obj ['squared_error']; enc ['one_hot']
- **hist_gbm** — obj ['gamma', 'poisson', 'squared_error']; enc ['one_hot', 'ordinal'] (early-stop)
- **lightgbm** — obj ['gamma', 'poisson', 'squared_error', 'tweedie']; enc ['native_categorical', 'one_hot', 'ordinal'] (early-stop)
- **tweedie_glm** — obj ['gamma', 'poisson', 'tweedie']; enc ['one_hot']
- **xgboost** — obj ['gamma', 'poisson', 'squared_error', 'tweedie']; enc ['one_hot', 'ordinal'] (early-stop)
- **tabpfn** — obj ['squared_error']; enc ['one_hot', 'ordinal'] (requires `bootstrap-track --enable-foundation-models` + the `[foundation]` extra)
- **tabfm** — obj ['squared_error']; enc ['one_hot', 'ordinal'] (requires `bootstrap-track --enable-foundation-models` + the `[foundation-modal]` extra (Modal GPU; non-commercial weights))

Target *shape* → objective (the handoff names the active target's shape):
**has zeros** (pure premium / incidence) → tweedie/poisson/squared_error;
**counts** → poisson/tweedie/squared_error; **strictly positive** (severity) →
gamma/squared_error. Features default to all eligible predictors; restrict with
`model.feature_inclusions/exclusions` using names from the handoff.

### Option B — run-local script (escape hatch, for novel models)

Expose `fit_predict`; return a `Prediction` (the framework converts unit→total
and calibrates for you) **or** a raw `np.ndarray` of TOTALS (you own calibration):
```python
from autoresearch.models.prediction import Prediction

def fit_predict(train, score, *, feature_inclusions=None,
                feature_exclusions=None, **hyperparameters):
    ...  # fit on `train`
    return Prediction(values=pred_rates, unit="rate"), notes
```
If you return a raw array instead: multiply rates by the weight column (`score[EXPOSURE]`,
the name the handoff prints); gamma/log losses need `y > 0` (split freq×sev or use Tweedie); encode categoricals
(`'B12'`): lightgbm `category` dtype, xgboost/sklearn ordinal/one-hot; early-stop
on a train-internal split only; and calibration is mandatory —
`apply_training_calibration(pred_score, pred_train, actual_train_cost)` from
`autoresearch.models.calibration` (factor = Σactual/Σpred). Feature names come
from the handoff — **build features only from that named list** (never sweep
"all remaining columns" into the model; the framework strips target and id
columns from script frames, and `score` carries no targets at all). Column
constants (`from autoresearch.models.dispatcher import`):
- `EXPOSURE` — weight/offset column; weights + rate->total only, never a feature
- `CLAIM_COST` — training-target total column (when present)
- `CLAIM_COUNT` — claim-count target (freq/freq-sev; when present)
- `RECORD_ID` — row identifier

## Compute budget

Per-experiment wall-clock budget: **10 min for the first 5 experiments,
+5 min every 5 thereafter** — `budget_minutes = 10 + 5 × (N // 5)`.
The challenger is refit ~5× per comparison (1 fit + 4 CV folds), so budget your
single fit at ~1/5 of that. A close call (win rate in [0.50, 0.75]) escalates to
more folds — up to ~13× the single fit, *outside* the budget alarm — so leave
headroom. Watch `n_estimators × (1/learning_rate)`, `num_leaves`/`max_depth`,
and row count.

## Decision policy

Comparisons run under `gate_mode = cv_bootstrap` and stop at `pending_llm`; the
mechanical gates are **advisory**, the verdict is yours. Before deciding, review
`gini_weighted`, `rank_gini_weighted`, `asym_pricing_loss` (lower is better;
penalises under-pricing 4×), and the calibration ratio. Then:
- **promote** — clean win; replaces the global champion + fires holdout eval.
- **local_promote** — useful progress for its research line, not a champion.
- **reject** — insufficient/contradictory evidence; keep it as a learning.

Record a structured `--reason-code` with the free-text rationale:
`clear_win`, `line_progress`, `noise`, `inferior`, `artifact_suspected`,
`calibration`, or `other`. This keeps cross-run memory queryable without
discarding the scientific explanation.

### Adaptive search — design experiments from results, not in advance

Treat the cycle count as a **budget, not a to-do list**. Submit **one**
experiment, see its result, and let that result shape the next hypothesis. Do
**not** pre-commit to a fixed slate of N experiments at the start of a run.
After each cycle, before proposing the next, state what the last result changed
about your thinking — a surprising, degenerate, or near-miss result is a signal
to run a quick **diagnostic** on *why* (e.g. a model that scores no better than
the flat baseline is probably broken, not merely weak — investigate before
moving on), not a cue to advance to the next pre-decided idea. You **may** commit
to a short, explicit sequence when you deliberately want comprehensive coverage
of a defined set (e.g. a feature-engineering sweep, or a head-to-head of a few
model families) — name the sequence and why, still read results between steps,
and abandon it early if a result makes it moot.

**Search policy:** let breadth **emerge** from these adaptive choices rather than
from a coverage checklist written up front. Prefer breadth over depth: rotate
exploration axis after 2 same-axis experiments and cap same-model-family tuning
at 3. A plateau forces a **structural** change (new family or target framing),
not more tuning — re-tuning at a plateau is provably below the gate's noise floor.

## Proposal contract (what you must supply)

Supply only the fields that encode your scientific choice:

`experiment_name`, `rationale`, `change_summary`, `expected_benefit`, `key_risk`, `exploration_axis`, `approach_family`, `target_framing`, `feature_representation`, `expected_learning`

plus an `experiment_config` with `model_family`, `target_strategy`, and a model
implementation — **either** `model.recipe` (preferred; set `model_family =
"recipe"`) **or** `model.script_path` (point it at your `model_<name>.py` for a
novel method).

The controller derives everything else from the champion, the recommended tree
action, and the research-line registry — you do **not** need to send:
`proposal_id`, `parent_experiment_id`, `tree_action`, `parent_rationale`, `selected_tree_action_id`, `research_line_action`, `research_line_id`, `research_line_label`, `research_line_hypothesis`, `line_membership_rationale`, `parent_branch_id`, `branch_action`, the fixed
`preprocessing` block, or the duplicate `experiment_config.experiment_name` /
`experiment_config.parent_experiment_id`. To deviate from a default (e.g. a
different tree parent or research line), set that field explicitly; when you
diverge from the recommended tree action, also include
`tree_policy_override_rationale`. The handoff embeds a ready-to-fill template.
