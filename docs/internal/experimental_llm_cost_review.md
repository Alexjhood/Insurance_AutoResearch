# Experimental LLM Cost and Efficiency Review

Date: 2026-06-07

## Scope

This review covers the full external-agent research loop:

1. Console launch and seed prompt.
2. Agent onboarding and run bootstrap.
3. Handoff/context construction.
4. Hypothesis and proposal generation.
5. Run-local model implementation.
6. Preflight, tests, experiment execution, screening, and comparison.
7. LLM promotion decisions, research logging, and the next cycle.

The goal is to reduce LLM input, cached-input misses, reasoning, output, tool
calls, retries, and unnecessary experiment work without reducing the process's
ability to discover strong solutions.

## Executive Conclusion

The largest near-zero-risk savings are not in model choice. They are in the
protocol around the model:

- The runtime manual is 47 KB and contains duplicated and stale workflows.
- Dynamic run data appears before the large stable manual in the conversation,
  which prevents most of that manual from benefiting from cross-run prefix
  caching.
- The full context grows to 83-86 KB after roughly 20 proposals, mostly because
  research-tree nodes retain verbose metadata.
- The proposal contract requires about 20 prose fields, even though many can be
  derived by the controller.
- Every experiment attempt runs the full 256-test suite. It took 57.4 seconds
  locally, while most model fits in the reviewed runs took 1-12 seconds.
- Run-local scripts repeatedly restate 60-120 lines of model scaffolding. The
  manual tells agents to use `champion_template.py`, but no such file is
  generated.
- The same preventable implementation errors recur across independent runs.
- CLI cycle output contains detailed nested metric panels instead of a compact
  agent response.
- Token and cache telemetry is incomplete, so the system cannot currently
  optimize the quantity it is trying to reduce.

The recommended architecture is:

- A compact, stable runtime contract placed before dynamic context.
- Orchestrator-owned bootstrap and state transitions.
- A short structured `next_action` payload with drill-down on demand.
- A declarative model recipe/SDK for common experiments, with arbitrary Python
  retained as an escape hatch.
- Framework-owned prediction units, exposure conversion, and calibration.
- Test results cached by relevant code hash.
- High-capability reasoning reserved for hypothesis selection and anomalies;
  routine execution uses low effort or deterministic code.

These changes should reduce cost substantially while improving reliability.

## Evidence From Existing Runs

### Instruction and context size

- `AGENT.md`: 47,247 bytes, 763 lines.
- Typical handoff: 7-8.5 KB.
- Proposal template: about 1.8-1.9 KB.
- Full context:
  - 28.8 KB after 3 proposals.
  - 41.3 KB after 6 proposals.
  - 82.7-85.5 KB after about 20 proposals.
- In the 20-proposal Claude run, `research_tree` alone was 49.4 KB.
- The same context contained only about 2.7 KB of recent comparison summaries.

The context builder calls its output "bounded", but includes up to 25 research
nodes with verbose `tree_metadata`, prose fields, metrics, and guidance. Most
of the tree payload is historical narration rather than information needed for
the next choice.

### Generated output size

In the reviewed runs:

- Proposal JSON files were commonly 1.9-3.1 KB each.
- Model scripts were commonly 2.4-5.0 KB each.
- One 20-experiment run retained about 90.8 KB of proposal Python and 66.8 KB
  of proposal JSON in iteration artifacts.
- Many scripts differ mainly in estimator choice or a few hyperparameters.

This is paid LLM output despite much of it being boilerplate.

### Fixed test cost

The complete local suite contains 256 tests and took:

```text
real 57.38 seconds
256 passed
```

`run_experiment()` invokes the full suite for every experiment attempt. In the
reviewed runs, most successful model fits took 1-12 seconds. Consequently, a
one-second model can pay almost a minute of fixed test cost before preflight,
screening, or CV.

Run-local proposal scripts live under artifacts and generally are not directly
covered by the repository test suite. Their meaningful validation is the
script preflight and output checks, so rerunning all repository tests for every
script revision has low marginal value.

### Preventable retries

Observed repair causes included:

- LightGBM Gamma objective receiving zero labels.
- A severity subset becoming empty in the random 5,000-row preflight sample.
- XGBoost `best_iteration` accessed when early stopping was not active.
- Repeated Gamma-on-zero-label failures in independent runs.

These are stable API/domain constraints. They should be encoded in helpers and
validators rather than repeatedly solved by the experimental LLM.

### Wasted state-transition calls

One OpenCode run recorded three `waiting_for_proposal` cycle invocations. The
command was called before a proposal was available, causing context rewrites
and another agent/tool round trip without advancing the experiment.

`run-session-cycles N` also stops at the first proposal, repair, or decision
boundary. The launch UI and seed prompt describe N cycles, but the agent must
manually drive each boundary.

### Stale and conflicting guidance

The operating manual contains both the current supervised workflow and a
legacy direct workflow:

- The current workflow says comparisons stop at `pending_llm` and require
  `record-decision`.
- Later sections say `compare-to-champion` automatically promotes.
- Later sections use legacy `artifacts/experiments/` and
  `artifacts/comparisons/` paths instead of run-scoped iteration paths.
- The quick-start says to use the proposal inbox and `run-session-cycle`;
  later steps say to run a baseline config directly.
- The manual requires `champion_template.py`, but the framework does not create
  it.

Conflicting instructions increase reasoning, file discovery, and incorrect
tool use even when the agent eventually recovers.

### Search behavior

The first reviewed Claude run found its durable global champion by cycle 2,
then ran many useful but mostly confirmatory experiments. Another OpenCode run
found a stronger two-stage model around cycle 10. This means a simple early
stop after the first plateau would be unsafe.

The correct cost policy is therefore not "stop early". It is:

- Guarantee broad coverage of distinct family/framing cells.
- Limit repeated tuning inside a weak or exhausted line.
- Spend high reasoning effort on structural choices and anomalies.
- Use cheap deterministic execution for the rest.

## Recommendations

## 1. Add Complete Cost Telemetry First

Priority: P0

Normalize and persist, per turn:

- input tokens
- cache-read input tokens
- cache-creation input tokens
- uncached input tokens
- output tokens
- reasoning/thinking tokens
- provider-reported cost, when available
- model and effort
- tool call count, failures, and duration
- command stdout/stderr bytes
- retries and repair cause

The Codex adapter preserves an opaque `usage` object. The Claude adapter drops
usage/cost fields from result events, and the OpenCode adapter records no usage.
Without normalized telemetry, changes cannot be ranked by actual savings.

Add run-level derived measures:

- cost per valid experiment
- cost per promotion
- cost to best-so-far score
- tokens per proposal and decision
- cache hit ratio
- failed-tool-call rate
- duplicate/stale proposal rate

## 2. Replace `AGENT.md` at Runtime With a Compact Contract

Priority: P0

Keep the long manual for humans, but generate a 3-6 KB agent runtime contract
containing only:

- objective and hard safety constraints
- exact current workflow
- compact model interface
- exact next action
- decision policy
- pointers for optional drill-down

Remove tutorials, legacy commands, duplicated schema, repository layout,
historical rationale, and long metric descriptions from the runtime path.

Generate the runtime contract from code/config so it cannot drift from the CLI.
Add a test that every command shown in it exists.

Expected effect: roughly 8-12K fewer input tokens at startup, before considering
the additional files the manual currently tells the agent to read.

## 3. Reorder Prompts for Prefix Caching

Priority: P0

The current seed contains dynamic track, run ID, cycle count, model identity,
and guidance before the agent reads the large stable manual. Once a dynamic
token differs, later manual/tool output cannot share a cross-run prefix cache.

Change launch order:

1. Orchestrator bootstraps the run before launching the LLM.
2. Send an identical stable runtime contract first.
3. Append a compact dynamic state block last.
4. Pass track/run paths through environment or structured fields, not prose
   repeated throughout the prompt.

Keep field ordering and serialization deterministic. Avoid timestamps and
absolute paths in the stable prefix.

For resumed sessions, append only the state delta rather than regenerating the
full handoff.

## 4. Make the Handoff the Only Default Context

Priority: P0

The handoff is already close to a useful compact view. Make it authoritative
and stop instructing the agent to read the full context, proposal template,
research log, source files, and SQLite state by default.

Return a compact structure such as:

```json
{
  "state": "needs_proposal",
  "champion": {"id": "...", "score": 0.3690},
  "budget": {"remaining_cycles": 7},
  "last_result": {"outcome": "clear_loser", "lift": -0.1131},
  "coverage": {"untried_cells": ["bagged/direct", "two_stage/boosted"]},
  "recommended_actions": ["open_underexplored_axis"],
  "next_action": "submit_proposal"
}
```

Expose detailed comparisons, diagnostics, and tree history through explicit
queries only when the planner asks for them.

Compact research nodes to:

- node ID
- parent ID
- axis/family/framing enums
- outcome enum
- score/lift
- one short learning sentence

Do not repeat all proposal prose in `tree_metadata`.

## 5. Shrink the Proposal Contract

Priority: P0

The controller can derive:

- proposal ID
- parent experiment ID
- parent branch
- experiment name duplication
- fixed preprocessing
- branch action
- default/recommended tree action
- line label/hypothesis already stored for an existing line
- script path

Require only information that changes the scientific choice:

```json
{
  "hypothesis": "...",
  "change": "...",
  "axis": "target_framing",
  "approach": "two_stage",
  "line": "freq_sev",
  "expected_learning": "...",
  "risk": "...",
  "recipe": {...}
}
```

Allow an optional override block for unusual tree or line decisions.

This should reduce proposal output by roughly 60-80% and remove many ingestion
failures without reducing the search space.

## 6. Introduce Declarative Model Recipes With a Python Escape Hatch

Priority: P0

Most experiments use a small set of repeatable operations:

- numeric/categorical feature selection
- one-hot or ordinal encoding
- estimator construction
- target-rate construction
- exposure weighting
- train-internal validation
- early stopping
- prediction-unit conversion
- aggregate calibration

Represent these as a recipe interpreted by trusted framework code. Example:

```json
{
  "estimator": "lightgbm",
  "objective": "tweedie",
  "target": "pure_premium_rate",
  "encoding": "native_categorical",
  "params": {"num_leaves": 63, "learning_rate": 0.05},
  "early_stopping": 50
}
```

Retain arbitrary run-local Python for genuinely novel models. This preserves
solution strength while making common experiments cheaper and more reliable.

## 7. Move Units, Exposure Conversion, and Calibration Into the Framework

Priority: P0

The LLM should not repeatedly implement the most safety-critical bookkeeping.

Change the model contract to return a structured prediction:

```python
Prediction(values=pred, unit="rate")
```

or:

```python
Prediction(values=pred, unit="target_total")
```

The framework should:

- convert rates to totals using exposure
- apply training-total calibration
- record native bias
- reject inconsistent units

This would have prevented the exposure-ranking artifact found in the long
Claude run and removes repeated calibration/exposure code from every script.

## 8. Generate the Promised Champion Template

Priority: P0

Either generate `proposal_inbox/champion_template.py` or remove the instruction.
Prefer generating it after every promotion.

For small follow-up experiments, support a patch/override file:

```python
PARAM_OVERRIDES = {"num_leaves": 31, "min_child_samples": 500}
```

Do not require the LLM to rewrite 60-120 lines when changing one parameter.

## 9. Cache the Test Gate by Relevant Code Hash

Priority: P0

Run the full suite:

- once at bootstrap
- after tracked source/config/test changes
- after dependency changes

Reuse the result while the hash of relevant files is unchanged.

For artifact-only model scripts, run:

- syntax/import check
- holdout/integrity scan
- interface check
- domain-aware preflight
- prediction validation

This retains safety and can remove about 57 seconds from most attempts in the
current environment.

## 10. Make Preflight Domain-Aware

Priority: P0

The random 5,000-row sample can omit rare positive outcomes, creating false
failures for severity models.

Use a deterministic stratified sample containing:

- positive and zero claim counts
- positive and zero costs
- all major categorical levels where feasible
- train and score rows

Add objective validators:

- Gamma labels must be strictly positive.
- Poisson labels must be non-negative.
- Severity training sets must be non-empty.
- `best_iteration` may only be read when early stopping ran.
- prediction length, finiteness, sign, and unit must be valid.

Keep repair requests concise: error class, failing line, domain rule, and next
action. Do not include a 4,000-character traceback unless requested.

## 11. Add a Compact Agent CLI Mode

Priority: P0

`run-session-cycle` currently prints the full state object, including nested
screening panels. Add `--agent-json` or make compact output the default for
agent-driven commands:

```json
{
  "state": "awaiting_decision",
  "proposal_id": "...",
  "comparison_id": "...",
  "summary": {"lift": 0.026, "win_rate": 0.81},
  "flags": ["ci_crosses_zero"],
  "next_action": "record_decision",
  "detail_path": "..."
}
```

The same applies to bootstrap, status, repair, and decision commands.

## 12. Collapse Mechanical Tool Calls

Priority: P1

Have the orchestrator perform:

- worktree creation
- bootstrap
- session creation
- context export
- inbox/state checks

before the LLM starts.

Provide one atomic submit/run operation rather than requiring the agent to:

- write files
- invoke a cycle
- inspect state
- invoke repair
- record a decision
- invoke another cycle

At minimum, reject a cycle invocation client-side when no proposal exists, so
`waiting_for_proposal` calls do not reach the agent process.

## 13. Split Planner and Executor Responsibilities

Priority: P1

Use the strongest model/effort only for:

- selecting the next hypothesis
- interpreting ambiguous comparisons
- investigating anomalies
- deciding when to rotate or stop a research line

Use low effort, a cheaper model, or deterministic code for:

- filling derived proposal fields
- rendering recipes
- writing logs
- routine promote/reject commands
- fixing known API-pattern errors

The current console fixes one model and effort for the whole run and defaults
to provider behavior. Make `low` explicit for routine execution, then escalate
on triggers such as:

- close comparison
- metric disagreement
- unexpected large lift
- calibration or unit anomaly
- two failed repairs
- plateau after broad coverage

## 14. Triage Promotion Decisions

Priority: P1

Do not remove LLM review entirely. The exposure artifact demonstrates why a
human-like anomaly review matters.

Instead:

- Auto-reject clear losers, as today.
- Auto-record routine inconclusive rejections when all metrics agree and no
  anomaly flag is present.
- Auto-promote only very conservative, fully clean wins.
- Require LLM review for large unexpected gains, metric disagreement,
  calibration issues, local promotion, and close calls.

This preserves the valuable part of reasoning while removing routine decision
turns.

## 15. Separate Process Memory From Research Memory

Priority: P1

Always-on process memory may include:

- API gotchas
- objective label domains
- unit/exposure rules
- framework interfaces
- known invalid code patterns

It should not include prior scores or successful hypotheses.

Keep scientific cross-run memory separately configurable as `none`, `own`, or
`all`. This prevents repeated implementation mistakes without contaminating an
independent research benchmark.

When research memory is enabled, inject a tiny ranked set of relevant verified
insights automatically. Do not force a separate memory-query tool call unless
the planner needs detail.

## 16. Use a Portfolio Search Budget, Not a Simple Early Stop

Priority: P1

To preserve discovery strength:

1. Reserve an initial breadth budget across distinct family x framing cells.
2. Allow at most two unguided tuning experiments per cell.
3. Permit additional tuning only when a diagnostic names a concrete error
   mode or estimated effect above the noise floor.
4. Park a line after repeated no-lift results.
5. Reserve a final budget for anomaly probes and combinations of the best
   distinct approaches.

This would reduce long sequences of confirmatory hyperparameter variants while
still allowing the two-stage discovery that appeared later in one run.

## 17. Add Progressive Fidelity Carefully

Priority: P2

Before a full-data fit and four-fold CV:

1. Interface/domain preflight.
2. Stratified small-data fit.
3. One full search-validation fit.
4. Full CV/bootstrap only for survivors.

The system already has steps 1, 3, and 4. Improve step 1 and optionally add step
2 with a generous pass threshold.

Do not use aggressive subsample rejection for model classes whose advantage may
appear only at scale. Treat the cheap stage as a clear-loser filter, not a
promotion gate.

## Suggested Implementation Order

### Phase 1: Low risk, immediate savings

1. Normalize token/cache/tool telemetry.
2. Remove stale runtime instructions and generate a compact contract.
3. Add compact CLI output.
4. Compact research-tree context.
5. Derive redundant proposal fields.
6. Cache full-suite test results.
7. Generate the champion template.
8. Add domain-aware preflight validators.

These changes should not reduce search capability.

### Phase 2: Reliability and output reduction

1. Add model recipes/SDK helpers.
2. Move units, exposure conversion, and calibration into the framework.
3. Add process memory.
4. Collapse mechanical tool calls.

These changes should improve search capability by reducing implementation
errors and freeing budget for more hypotheses.

### Phase 3: Adaptive intelligence

1. Planner/executor model split.
2. Adaptive effort escalation.
3. Decision triage.
4. Portfolio budget enforcement.
5. Progressive fidelity.

These should be A/B tested because they can change research behavior.

## Quality Guardrails for A/B Testing

Do not judge an optimization only by token reduction. Compare paired runs using
the same data, model, starting state, and experiment budget.

Primary quality measures:

- best valid score achieved by cycle N
- probability of finding the best known structural family/framing
- area under the best-so-far score curve
- number of distinct family x framing cells explored
- anomaly-detection rate
- invalid promotion rate

Efficiency measures:

- total and uncached input tokens
- cache hit ratio
- reasoning and output tokens
- tool calls and failed tool calls
- retries per valid experiment
- wall time per valid experiment
- full comparisons per promotion

A change should be accepted when it materially reduces cost with no meaningful
drop in paired-run discovery quality or safety.

## Highest-Value First Experiment

Implement and test this bundle first:

- orchestrator-owned bootstrap
- stable compact runtime contract
- compact `next_action` JSON
- proposal fields reduced to scientific intent
- pytest cache keyed by source/test/dependency hash
- generated champion template
- explicit low effort for routine turns

This bundle attacks input, caching, output, tool calls, and wall time without
pruning the scientific search space.
