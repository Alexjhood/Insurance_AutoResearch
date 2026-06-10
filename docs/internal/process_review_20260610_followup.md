# Experimental Process Review — Follow-up to the Cost Review

Date: 2026-06-10
Reviewer: Claude (analyst mode)
Evidence base: full repo read + the one available sequence, `codex/20260610T063646Z`
(5 cycles, 6 experiments incl. baseline, 2 promotions, gpt-5.4 @ low effort, ~17 min wall).

This review deliberately does **not** restate `experimental_llm_cost_review.md`
items 1–17. Items 1–9 are treated as done; where a "done" item has a residual
gap that shows up in the run evidence, it is called out explicitly. Everything
else here is new.

## Run snapshot (from `telemetry.sqlite`, the run's own ledger)

| Measure | Value |
|---|---:|
| Model calls | 61 |
| Input tokens | 3,828,595 |
| Cached input | 3,556,736 (92.9%) |
| Uncached input | ~271,859 |
| Output tokens | 16,424 |
| Reasoning tokens | 1,563 |
| Tool-call rows | 78 (0 recorded failures) |
| Repairs | 0 |
| Wall time | 06:35:47 → 06:52:40 (~17 min) |

Per-call average input is ~63K tokens and grows roughly linearly with cycle
count (the Codex harness re-sends the whole conversation every call). This
means **total run cost grows roughly quadratically with cycle count** — every
byte of conversation (tool output, doc reads, CLI dumps) is re-paid in every
subsequent model call. The most valuable levers are therefore (a) fewer model
calls and (b) less conversation growth per cycle, ahead of anything that only
trims a single payload once.

What worked well in this run, for the record: recipes were used for all five
experiments (zero model scripts written), `recipe_ref: "champion"` was used for
the follow-up, screening auto-rejected three clear losers without a decision
turn, there were no repairs and no failed tool calls, and cache hit climbed to
~93%.

---

## Part A — Errors and inconsistencies

### A1. The end-of-run research-log enforcement loop consumed ~40% of the run (highest impact)

**Evidence.** The last experiment checkpoint is 06:48:35 with cumulative input
2,283,757. The run total is 3,828,595. So **18 of 61 model calls and ~1.54M
input tokens (~40% of the run) happened after the final experiment** — almost
all of it spent writing `RESEARCH_LOG.md` at the very end. The
`research-log-guard.log` shows why it was so expensive:

```
06:50:15  missing ... Cycle 1..5 fields: Hypothesis, Changes, Outcome, Metrics, Interpretation, Next
06:51:03  missing ... (same — first rewrite still didn't match)
06:51:22  missing ... (same)
06:52:40  research log ok
```

The agent deferred all logging to the end (despite AGENT.md step 5 saying "log
each cycle"), then hit the Codex `Stop` hook, then needed **three attempts** to
match the hook's required format, re-reading comparison artifacts (`sed`/`rg`
calls at 06:51) to reconstruct metrics it had already seen.

**Root causes, each independently fixable:**

1. **AGENT.md and the hook disagree about the contract.** AGENT.md says "Log
   each cycle's hypothesis, outcome, and learning" (3 fields, no format).
   The hook requires six bold fields (`**Hypothesis**:`, `**Changes**:`,
   `**Outcome**:`, `**Metrics**:`, `**Interpretation**:`, `**Next**:`) under a
   `## Cycle N` heading. The agent cannot comply on the first try because the
   contract never states the real requirement. At minimum, the generated
   AGENT.md should embed the exact entry template, and the hook's deny message
   should include it verbatim so a single retry suffices.
2. **Four of the six required fields are framework-known.** Hypothesis
   (= proposal `rationale`/`research_line_hypothesis`), Changes
   (= `change_summary`), Outcome (promotion/auto-reject + reason), and Metrics
   (screening + comparison summary) are all in the registry. Paying LLM output
   tokens to transcribe registry rows into Markdown — with transcription-error
   risk — is pure waste. The framework should append the skeleton entry per
   cycle automatically.
3. **The two genuinely-LLM fields arrive at the wrong moment.** Interpretation
   and Next are exactly what the agent is thinking at `record-decision` /
   auto-reject time, when the result is already in context. Add optional
   `--interpretation` and `--next` flags to `record-decision` (and accept a
   short note on the next proposal for auto-rejected cycles), have the
   framework write the complete entry, and the Stop hook reduces to a no-op
   for compliant runs — or can be deleted.

**Expected effect.** Removes ~15+ model calls and ~1.3–1.5M input tokens per
5-cycle run (worse for longer runs: more cycles to backfill against a bigger
context), removes the retry loops, and makes the log more accurate (no
transcription drift). This is the single largest measured saving in the run
and is not covered by review items 1–17.

### A2. LLM_USAGE.md silently understates the run by the same ~40%, and the Model column is broken

- `usage_report._backfill_checkpoints` creates checkpoints only from registry
  experiments. There is no end-of-session checkpoint, so all post-final-
  experiment usage (the entire A1 wrap-up) is invisible: the report's footer
  says 2,283,757 input while the actual total is 3,828,595. Add a synthetic
  final checkpoint ("session wrap-up") when the session reaches a terminal
  state, so the report sums to the true total.
- The Model column renders `—` because the report reads `DISTINCT model FROM
  llm_model_calls`, and the Codex importer never sets `model` on call rows
  (`_import_codex` → `upsert_model_call` has no `model` key; the model lives
  only on `llm_turns`/`llm_sessions`). Either copy the current turn's model
  onto each call row or make the report fall back to the turn/session model.
  This partially defeats review item #1's purpose (rank changes by measured
  savings, attribute usage per model/effort).

### A3. Three different model identities are recorded for one run, and nothing reconciles them

For this run: `run_manifest.json` says `openai/gpt-5` (operator-typed at
bootstrap), telemetry `llm_sessions.model` says `gpt-5.4` (what the harness
actually ran), turn 1 says `gpt-5.5` (the `.codex/config.toml` default), and
`.codex/config.toml` pins `gpt-5.5`. The cross-run memory store and the
"compare models" benchmarking attribute results by manifest identity — which
is the one value no machine verified. Fix: after the first telemetry sync,
reconcile `model_identity` in the manifest from `llm_sessions.model` (or at
least warn loudly on mismatch). Otherwise the benchmark's central comparison
axis (which model did the research) is unreliable.

### A4. Two unlabeled definitions of "lift"/"score" leak into the agent's planning context

The research-tree node metrics are built as
`{**_screening_metrics_summary(screening), **metrics_summary}`
(workflow.py:367–370). The screening dict contributes single-split `lift`;
the CV comparison contributes `mean_lift`. The context compactor then picks
`metrics.get("lift", metrics.get("mean_lift"))` — so the **single-split** lift
always wins when both exist, and `score` (looked up as `score`/`mean_score`)
is usually `None` because neither summary uses those key names.

Observed result in the live handoff: the *promoted current champion* is shown
as `lgbm_direct_onehot_v1 … outcome=promoted … lift=-0.000126` — a negative
lift on the winning model (its single-split screen) while the decisive paired
CV lift was +0.010. The previous champion's node shows `lift=0.353453`
(single-split) while the comparison reported mean lift 0.316847. Nothing tells
the agent these are different quantities. A planner taking these numbers at
face value will mis-rank lines, or burn reasoning tokens reconciling them.

Fix: name the keys distinctly (`split_lift` vs `cv_mean_lift`), prefer the CV
value for nodes that have one, and render the chosen definition in the handoff
line (e.g. `cv_lift=+0.0100 (split −0.0001)`).

### A5. The "Recipe reuse" section can show the same recipe as both promoted and rejected

The ledger dedupes on the full param fingerprint, but the rendered summary is
only `estimator/objective/encoding`. This run's handoff shows:

```
- `lightgbm/tweedie/native_categorical` — promoted (score 0.3734)
- `lightgbm/tweedie/one_hot` — promoted (score 0.3733)
- `lightgbm/tweedie/native_categorical` — rejected (score 0.3619)
```

Two problems: (1) the same visible label appears with contradictory outcomes
(the params that differ are hidden), making "do not re-run a recipe that
already lost" ambiguous at the only granularity the agent can see; (2) the
list is ranked by single-split score, so the **superseded** champion sits
above the **current** champion with a *higher* score — inviting a pointless
"revert to the better recipe" proposal. Fix: append a compact param diff to
each line (e.g. `(leaves=127, lr=0.03)`), mark the current champion, and rank
current-champion-first.

### A6. The console seed template asks for something the CLI rejects

`SEED_TEMPLATES` (console/backend/orchestrator/jobs.py) says *"bootstrap a new
run … with run id "{run_id}""* — but `cli.py:1085` errors on
`--new-run` together with `--run-id`, and `bootstrap-track` generates its own
timestamp. An agent following the seed literally gets a parser error, then has
to reason its way out (the orchestrator's pre-made `run_id` is then orphaned —
job rows, worktree naming, and telemetry import all key on a run id that never
exists on disk). The reviewed run was launched manually with the (correct)
RUN_WITH_CODEX.md wording, so this is latent, but it will bite the first
console launch. Related inconsistencies in the same family:

- `bootstrap-track`'s "Ready" hint prints `… run-session-cycles 10`, directly
  contradicting AGENT.md's one-cycle-at-a-time adaptive loop (and the manual's
  own correction at line 302). Print `run-session-cycles 1` (or the
  handoff's next command) instead.
- The seed says "run {cycles} cycles" but nothing passes `--max-cycles` to
  `start-session` (this run had `max_cycles: None`). The agent must count
  cycles itself; a miscount silently over- or under-spends the budget. Have
  the seed (or orchestrator) set `--max-cycles` so the session completes
  itself and the budget is framework-owned.
- The seed's "Use synthetic data" line conflicts with AGENT.md's "French Motor
  dataset (freMTPL2, ~678K policies)" framing. Pick one provenance statement
  and put it in AGENT.md, not the seed (it's also per-run noise in what should
  be a stable, cacheable prompt prefix — review #3).

### A7. The research-log Stop hook exists only for Codex

`.codex/hooks.json` wires `codex_research_log_guard.py` on Stop; the Claude
(`.claude/settings.json`) and OpenCode hook sets only finalize telemetry. So
one track is forced to produce (and pay for) a six-field per-cycle log while
the others are merely asked politely. For a cross-model benchmark this is a
fairness problem twice over: unequal token overhead and unequal artifact
quality. Whichever resolution of A1 is chosen, apply it uniformly across the
three tracks.

### A8. Tool-call success/size telemetry is blind for exactly the commands that matter

Every `autoresearch …` exec in this run is recorded with output ≈113 bytes
(the immediate ack) and `success=NULL`; the real output arrives via later
`write_stdin` poll rows that aren't joined back to the originating command.
Consequences: `LLM_USAGE.md` reports 0/0 tool failures no matter what
happened, the per-command stdout-byte numbers (the input to several planned
optimizations) are wrong, and the failed-tool-call rate from review #1 is
unmeasurable for the workflow commands. Joining the poll rows to the pending
exec (Codex emits them in strict sequence) or parsing the session transcript's
aggregated output would fix attribution.

### A9. Residual instruction-surface conflicts (small, cheap to fix)

- Handoff "Next command" says `run-session-cycle`; AGENT.md teaches
  `run-session-cycles 1`. Both exist and do the same thing for N=1 —
  standardize on one to avoid the agent wondering whether they differ.
- The champion-follow-up template hardcodes
  `"recipe_overrides": {"params": {"num_leaves": "<new_value>"}}` — a
  hyperparameter-tuning nudge that sits awkwardly beside the contract's
  "plateau forces a structural change, not more tuning". Cycle 4 of this run
  was exactly such a num_leaves/learning_rate retune (clear loser). Make the
  placeholder neutral (`{"params": {"<param>": "<value>"}}`).
- OPERATING_MANUAL.md still carries the legacy direct workflow with
  `artifacts/experiments/` paths and `compare-to-champion` examples
  (lines ~660–690), albeit behind a warning box. The reviewed agent *did*
  read manual chunks before bootstrapping (see A10). Moving the legacy
  appendix to a separate `docs/legacy.md` would remove the trap entirely.
- Handoff hypothesis lines truncate mid-word ("claim probabili..."); the
  research-line hypotheses are ≤120 chars but the cap costs little and reads
  badly. Cosmetic.
- A junk zero-byte file `1` sits at the repo root.
- `proposal_fingerprint` rounds floats only at the top level of `model`;
  nested `params` floats are not rounded, so numeric jitter inside params can
  evade duplicate detection. Minor today (params are hand-picked), but cheap
  to fix.
- `_compact_research_nodes` labels `change_summary` as `learning` — the
  handoff then presents "what was tried" under the banner of "what was
  learned". The actual learning (auto-reject reason / decision rationale) is
  available and would be more useful in that slot.

### A10. The agent grazed ~82KB of docs before bootstrapping, despite the compact contract

Tool telemetry shows ~37KB of `sed` reads at 06:36:19 (before any
`autoresearch` command), another ~20KB at 06:36:39, and ~24KB at 06:38:18 —
AGENT.md plus README/RUN_WITH/manual chunks. That ~20K tokens of one-time
reads then rides along in all ~55 subsequent calls (≈1M token-reads, mostly
cached but not free, and it inflates the conversation toward context limits on
longer runs). Two contributors: (1) AGENT.md's "for anything not covered here,
read docs/OPERATING_MANUAL.md" reads as an invitation at session start;
(2) the contract isn't injected natively, so the agent must find and read it —
and while it's exploring, it reads neighbors too. See B2 for the fix.

### A11. Two bootstrap-track invocations at run start

Tool log: `bootstrap-track` at 06:36:27 and again at 06:36:44 (the run id
matches the second). One bootstrap attempt was wasted — consistent with a
first-call validation error (the 6.9KB output right after looks like an
error/help dump). Whatever the precise cause, it supports review #12's
"orchestrator-owned bootstrap": the agent should never be the one assembling
a five-flag bootstrap command from prose instructions.

---

## Part B — Additional cost levers (beyond review items 1–17)

### B1. Framework-written research log; agent supplies only interpretation at decision time

Covered in A1 — listed here because it is also the top cost lever:
~1.3–1.5M input tokens and ~15 calls saved per 5-cycle run, growing
super-linearly with run length. Zero search-quality risk (the scientific
content — interpretation, next direction — is still authored by the LLM, just
at the moment it's already in context).

### B2. Inject the contract natively per harness (AGENTS.md / CLAUDE.md)

Codex auto-loads `AGENTS.md`; Claude Code auto-loads `CLAUDE.md`; OpenCode
reads `AGENTS.md`. The repo has neither — only `AGENT.md`, which every agent
must locate and `sed` (12.7KB read in this run), and which lands *after* the
dynamic seed prompt in the conversation. Generating `AGENTS.md` and `CLAUDE.md`
(symlinks or copies from the same generator) would:

- put the contract in the harness-owned system region, **ahead of all dynamic
  content** — the practical realization of review #3's cache-ordering goal
  with no orchestrator changes;
- remove the read call and the risk of partial reads;
- shrink the seed prompt to pure dynamics ("track X, N cycles, model identity
  flags").

One caveat to verify: keep the generated file under the harnesses' auto-load
size limits and confirm OpenCode's lookup name.

### B3. Stop paying for poll loops on long-running commands

Each `run-session-cycles` (~100s) was followed by 2–4 `write_stdin` polls at
~30s intervals — every poll is a full-context model call (~63K input). Across
the run that's roughly 12–15 of the 61 calls doing nothing but waiting.
Options, cheapest first:

1. AGENT.md: "run-session-cycles takes 1–5 minutes; invoke it with a tool
   timeout ≥300s instead of polling" (Codex's exec tool accepts a timeout).
2. Print a single up-front line: `expected duration: ~Ns` so the agent can
   size the timeout.
3. Longer term (pairs with review #12's atomic submit/run): a `--wait` mode
   that blocks silently and then emits one compact JSON line, so there is
   nothing to poll for.

Expected effect: ~20–25% fewer model calls per cycle; bigger on slower
experiments where today's poll count scales with runtime.

### B4. Make refresh handoffs delta-only

Of the 12.1KB handoff, roughly 7KB (quick-start, template JSON, escape hatch,
override block, constraints, tree-guidance bullets) is static text duplicated
from AGENT.md and re-emitted on every `show-latest-handoff`/refresh. Since the
contract is already in context (and would be system-injected under B2), the
per-cycle handoff only needs the dynamic state: champion, next command, active
lines, recommended actions, recent nodes/learnings, recipe ledger. A
`show-latest-handoff --delta` (or making the full version the `--full` opt-in)
cuts ~2K tokens of conversation growth per cycle — which, given the quadratic
re-pay dynamic, compounds on long runs.

### B5. Fold post-decision state into the `record-decision` output

After the first promotion the agent fired three parallel calls —
`show-latest-handoff`, `list-champion-history`, `session-status` — returning
~27KB combined, to learn things `record-decision` already knew (new champion,
next action). Extending the decision output with a compact
`{champion, state, next_action}` block (same spirit as review #11) makes those
calls unnecessary; AGENT.md can then say "the decision output is sufficient;
do not call status commands routinely."

### B6. Let the session own the cycle budget

Pass `--max-cycles` at `start-session` (see A6). Besides correctness, this
saves the agent the bookkeeping reasoning ("how many cycles have I run?") and
enables a clean framework-driven stop: the final cycle's output can say
"budget exhausted — finish the log and stop", bounding the wrap-up phase.

### B7. Plan for context-length decay on long runs: restart at decision boundaries

This 5-cycle run ended at ~85–90K tokens per call. A 20-cycle run on this
trajectory reaches 200K+ per call — context-compaction territory, where the
harness starts summarizing away exactly the history the agent needs, and
cache efficiency collapses. The process is already architected for this: the
handoff is authoritative and all state is externalized. The orchestrator (or
AGENT.md guidance) can therefore **end the agent session every K cycles and
resume fresh** — the new session reads AGENT.md (system-injected under B2) +
the current handoff and continues with a ~15K-token context instead of 150K.
This converts quadratic growth into linear, and it's an A/B-testable change
(it could plausibly *improve* search quality by forcing the agent back to the
written record instead of its own drifting conversation memory).

### B8. Stabilize and slim the seed prompt

Combine A6's fixes: the seed should contain only (a) a pointer to the
contract, (b) track + cycles + model-identity flags, (c) optional operator
guidance. Drop the run id, the data-provenance line, and the bootstrap recipe
(those belong to the contract/orchestrator). Keep wording byte-identical
across launches so the harness/system prefix caches across runs.

### B9. Close the telemetry gaps that block measurement

A2 + A3 + A8 together: final wrap-up checkpoint, model on call rows (or
fallback), reconcile manifest identity, join poll output to its exec. With
those fixed, `LLM_USAGE.md` becomes a trustworthy A/B instrument for the
remaining review items (13–17), which all *require* paired-run measurement.

### B10. Trim decision-turn token cost on clean blowouts (refinement of review #14)

Both promotions in this run were advisory-clean (`guardrail_passed: true`,
advisory `promote`). Cycle 1 was a blowout (win rate 1.0, lift +0.317 on a
flat baseline) — under review #14's "auto-promote only very conservative,
fully clean wins" rule, that decision turn (~2 calls, ~150K input) was
mechanical. Cycle 5 (win rate 0.81, flat single split) is exactly the close
call that *should* stay with the LLM. The run therefore gives one concrete
calibration point for #14's thresholds: auto-promote when advisory=promote ∧
guardrails clean ∧ win_rate ≥ ~0.95 ∧ no flat-single-split flag; otherwise
pause for the agent.

---

## Part D — Second-pass findings (deep read of the evaluation/model/repair core)

A second pass covered the subsystems the run evidence didn't exercise:
`comparison_runner.py`, `evaluation/resampling.py`, `evaluation/metrics.py`,
`evaluation/validation.py`, `models/dispatcher.py`, `models/prediction.py`,
`models/recipe/*`, the repair loop in `workflow.py`, `cv_cache.py`,
`champion_template.py`, `milestone.py`, and the Claude/OpenCode telemetry
importers. Most of it is in good shape — findings below, then a positive
verification list.

### D1. Every repair rerun re-fits the known-failing attempt first

`_run_validated_experiment_attempts` (workflow.py:801) always restarts its
loop at attempt 1, and `run_experiment` has no resume/skip. The repair
protocol is: attempt 1 fails → `ExperimentNeedsRepair` → agent writes
`recipe_attempt_2.json` → agent reruns the cycle → the loop runs **attempt 1
again from scratch** (preflight + full fit + validation + the failed-attempt
diagnostic comparison) before reaching attempt 2. With two repairs, attempt 1
runs three times and attempt 2 twice. No run evidence yet (this run had zero
repairs), but each redundant re-fit is ~1–2 minutes of wall time — which, via
the poll-loop dynamic (B3), also converts into extra full-context model calls.
Fix: start the loop at the highest attempt index for which an
`attempt_*`/`recipe_attempt_*` file or completed output exists, or persist the
last completed attempt number on the proposal record.

### D2. AGENT.md's repair-trigger description is stale relative to the code

The contract says a cycle stops in `needs_repair` when the model fails
"preflight, output validation, or the **positive-lift check**" — but the
validation rules are constructed with `require_positive_lift=False`
(workflow.py:960); negative-lift models proceed to screening instead, by
design. The default `error_type` label in `_write_repair_request` is likewise
`"positive_lift_failed"`. Cosmetic, but it's the *generated* contract being
wrong about its own mechanics — exactly what the generator was built to
prevent. Update the generator text and the default label.

### D3. The compute-budget guidance understates comparison cost when escalation fires

AGENT.md: "The challenger is refit ~5× per comparison (1 fit + 4 CV folds)".
True for the base path, but a close call (win rate in [0.40, 0.60]) escalates
to two more fold partitions — up to 8 more challenger fits *and* 8 champion
fits (champion partitions 1–2 are not yet in `cv_cache` the first time). Worst
case is ~13× the single fit, and the escalation runs *outside* the
per-experiment budget alarm, so a model that fits within budget can still
produce a very slow comparison (more wall time → more poll calls). Either
mention escalation in the budget paragraph or count escalated refits against
the budget.

### D4. Latent holdout-leakage channel through `milestone_reports/`, plus a string bug

`record-decision` on promote calls `evaluate_on_holdout` **inside the agent's
own CLI process**, and the console launcher copies the full operator
environment into the agent (`_build_env`: `env = {**os.environ}`). If the
operator's shell exports `AUTORESEARCH_MILESTONE_TOKEN`, every agent-triggered
promotion writes real holdout metrics into
`<run>/milestone_reports/*.md` — a directory the run-scope guard happily lets
the agent read (it only fences *other* runs). The agent is told not to
*reference* the holdout in code, and the integrity scanner enforces that for
code — but nothing stops it reading its own run's holdout report mid-run and
steering the search with it. In the reviewed run the token was absent (both
reports say "Skipped"), so no actual leak occurred. Fixes, any of: deny
`milestone_reports/` to research-scoped sessions in the guard; write the
reports outside the run dir; or strip `AUTORESEARCH_MILESTONE_TOKEN` from the
agent environment in `_build_env` (cheapest and most robust).

Also trivial: the skip-report writes a literal `{champion_id}` into the
Markdown ("Re-run with `autoresearch evaluate-milestone {champion_id}`…") —
the third string fragment in `milestone.py` lacks the `f` prefix. Visible in
this run's report files.

### D5. `gini_weighted` is not tie-aware — a constant model scores ≈ +0.02, not 0

`_gini_weighted` sorts by predicted rate with `np.argsort`; for tied
predictions the Lorenz curve follows arbitrary input order. The global-mean
baseline (all predictions identical) scored `gini_weighted = 0.019975` on the
eval split — the agent's research log even reasons about the baseline's
"0.0200" Gini as if it were signal. All models share the convention so paired
comparisons stay fair, but it misleads interpretation of near-flat models and
lets the `gini_above_zero` guardrail pass for a literally constant model. A
tie-aware Lorenz (average over tied groups) or a documented caveat would fix
it.

### D6. Track parity: OpenCode has neither the research-log hook nor a telemetry import hook

`.opencode/plugins/` contains only `run-scope-guard.js`. Codex gets telemetry
import on SessionStart/Stop and research-log enforcement on Stop; Claude gets
telemetry import only; OpenCode gets neither. Cross-track comparisons of cost
or discipline are not apples-to-apples until the three hook sets match
(extends A7).

### Verified clean (no action needed)

- `comparison_runner.record_decision`: idempotent re-record, guardrail-blocked
  promotion, correct local-promote/reject bookkeeping, best-effort report
  re-render that can't undo a decision.
- Bonferroni family counting includes the current comparison; bootstrap CI
  uses common random numbers across champion/challenger.
- `cv_cache`: correct keying on (experiment, dataset hash, partition seed,
  cap); champion folds reused across cycles as intended.
- `models/prediction.py` + dispatcher finalisation: unit conversion,
  single-scalar calibration on train rows, validators (gamma>0, poisson≥0,
  finite/non-negative/length) — all sound; recipe interpreter never touches
  exposure or calibration; severity stage correctly claim-weighted and
  gamma-filtered.
- `champion_template.py`: reserved-filename handling, stale-recipe cleanup on
  script champions.
- The Claude telemetry importer captures full usage (input/cache-read/
  cache-creation/output per request id) — the original review's "Claude
  adapter drops usage" appears fixed at the importer layer.
- Holdout vault access is token-gated with graceful skip; promotion flow is
  not interrupted when the vault is unavailable.

### Coverage note

Not reviewed in depth (judged low-risk for the cost/error goals): the web
console frontend, `dashboard/`, and the test suite's internals beyond the
contract-drift tests. Memory, anonymisation, and the gate/decision stack are
covered in Parts E–G below (added 2026-06-10, second session).

---

## Part E — Memory subsystem review

Files: `memory/__init__.py`, `store.py`, `harvester.py`, `insights.py`,
`playbook.py`, `query.py`, plus the recipe library's global ledger and the
checkpoint call in `session.py`.

The architecture is right-minded: the aggregator lives outside the working
tree, harvests are read-only with holdout-path guards
(`assert_no_holdout_columns`, `_guard_path`), insights are evidence-bound and
verified against the run's own registry, and the query tool gates on
`none/own/all`. But the implementation has one production-breaking failure and
several integrity gaps.

### E1. Memory writes silently fail on the Codex track — proven in the store (highest impact)

`~/.autoresearch/<project>/memory` is **outside the Codex sandbox**
(`.codex/config.toml`: `sandbox_mode = "workspace-write"`, `writable_roots =
[]`). Every write-side memory operation inside a Codex run — the cycle-5
`maybe_memory_checkpoint`, the global recipe ledger append, playbook rebuild,
`record-insight` — hits a permission error that `maybe_memory_checkpoint`
swallows into a logger nobody reads. Evidence:

- `memory.sqlite` contains every claude/opencode/set3 run but **not**
  `codex/20260610T063646Z`, whose cycle-5 checkpoint should have harvested it.
- The global `recipes.jsonl` contains only Claude-run entries, even though the
  codex run executed five recipes and wrote its run-local ledger fine.
- `pending_reflection.md` is absent from the codex run's handoffs — the
  harvest fails before `_write_reflection_prompt` is reached.
- Re-running the identical harvest outside the sandbox succeeds
  (7 experiments, 2 promotions, peak 0.3734).

So with memory enabled (`reuse_scope="memory"`, access own/all), Codex runs
would *read* memory but never *contribute*, and the leaderboard silently
excludes a whole track. Fixes: add the memory root to Codex's
`writable_roots` (and confirm the same for OpenCode's sandbox); and make
checkpoint failures visible — write the warning into the run dir (e.g.
`results/memory_checkpoint_error.json`) instead of a logger.

### E2. The leaderboard's best-ever score is the known exposure artifact

`runs.peak_gini` takes the max over `completed` experiments. The store's top
run is `claude/20260601T074524Z` at **0.4622** — experiment
`s12_hurdle`, with `s13_noexp_control` (0.4209) right behind it: the
exposure-ranking artifact and its control from the incident the cost review
documents. Both are `status="completed"`; the rejection happened at the
comparison/decision layer, which `peak_gini` ignores. Consequences when
memory is on: the playbook header would announce "Best observed gini:
0.4622", and `peak-gini-by-framing` / `biggest-single-jumps` /
`efficiency-by-model` all crown the artifact — actively steering future runs
toward a phantom. Fix: compute `peak_gini` from promoted champions (or
experiments whose comparison decision was promote/local_promote), or at
minimum exclude experiments whose comparison was rejected; keep raw maxima in
a separate, clearly-labelled column if wanted.

### E3. Access-gate gaps

- **Agent-writable fallback**: `resolve_memory_access` falls back to
  `run_manifest.json` when the env var is unset — and the manifest sits inside
  the agent's own writable run dir. The docstring's "the agent cannot set its
  own env var, so it cannot self-escalate" misses this path: on a manual
  launch without the env var, an agent could edit its manifest to
  `memory_access: "all"` and query everything. (The `.scope` files are also
  agent-writable — `artifacts/tracks/.scope/` is not a run folder, so the
  run-scope guard does not protect it.) Practical fix: have the operator
  surface always set the env var, and treat the manifest value as
  informational only; or verify manifest integrity (hash recorded at
  bootstrap outside the run dir).
- **`own` fails open to `all`**: in `query_insights`/`query_experiments`/
  `run_analysis`, `access == "own"` with an underivable `own_model_id`
  (missing/corrupt manifest identity) simply applies no filter — returning
  all models' data. Should fail closed. This interacts badly with the A3
  model-identity mismatch: `openai/gpt-5` (manifest) vs `gpt-5.4` (actual)
  silently fragments or mislabels per-model history.

### E4. Smaller observations

- The insights table is empty across all eight harvested runs — the
  reflection→insight→playbook pipeline has never fired in production (partly
  E1: the prompt file is never written on Codex; the Claude/OpenCode hook sets
  never surface it either). Until an end-to-end test exists, the playbook
  features are dead code in practice.
- Insight verification is shallow by design (cited IDs must exist; the
  optional delta is checked within 5% on the first two experiment ids). The
  claim *text* and its direction are never checked against the metrics — a
  verified insight can still assert the wrong sign. Acceptable if documented;
  the playbook's keyword bucketing ("plateau", "leverage") is similarly
  rough.
- `_categorise_insights` has a redundant double-append construction for
  `works` that can duplicate entries; harmless today (zero insights) but
  worth a tidy-up when touched.

---

## Part F — Anonymisation and data-provenance review

`anonymise.py` renames the fourteen freMTPL2 columns to semantic pseudonyms
and emits a private mapping + agent-facing schema. The mechanics are clean
(deterministic, role inference, private/public separation). The problem is
the concept: **it anonymises the column names and nothing else, and the rest
of the system then announces the dataset identity anyway.**

- The *values* are untouched: `region_cluster_j` literally contains
  "Rhone-Alpes" and "Picardie"; brands are the canonical `B1…B14`;
  bonus-malus runs 50–230; density 1–27,000. Any frontier model recognizes
  freMTPL2 from one `head()`.
- AGENT.md (the generated contract!) states "French Motor dataset (freMTPL2,
  ~678K policies)" in its first paragraph. Whatever the renames were meant to
  hide, the contract un-hides it.
- The console/RUN_WITH_* seed prompts say "Use synthetic data — …", but
  `data/raw/` holds the real Kaggle CSVs and the prepared parquet is the real
  data (542,412 search rows = 80% of 678K, real region names). The agent is
  told the data is synthetic when it is not — a factual error in the prompt
  (and the reviewed run's models were fit on real freMTPL2).

Why it matters: if the renames exist to stop the agent retrieving memorized
freMTPL2 results (published Tweedie/freq-sev benchmarks, known Gini ranges —
all in every model's training data), the current setup provides **false
comfort only**, and the benchmark's "research from scratch" premise is
contaminated by recall. The instant cycle-1 jump to Gini 0.373 with
textbook-perfect hyperparameters is consistent with exactly that. Pick a
lane:

1. **Embrace it**: real data, drop the renames' pretense, state in AGENT.md
   that the dataset is freMTPL2 and prior knowledge is fair game. Honest, and
   simplest; cross-model comparison still works since all models share the
   prior.
2. **Actually de-identify**: extend anonymisation to values (relabel regions/
   brands/areas via the private mapping, rescale density/bonus-malus,
   optionally jitter exposure), remove the dataset name from AGENT.md, and
   fix the seed wording. Imperfect (joint distributions still fingerprint the
   dataset) but raises the recall bar substantially.
3. **Go synthetic for benchmarking**: the generator already exists
   (`scripts/generate_synthetic_data.py`); a synthetic-data benchmark mode
   genuinely eliminates memorization, at the cost of realism.

Whichever is chosen, today's mixed state — renamed columns, real values,
named dataset, "synthetic" seed claim — is internally inconsistent and should
not survive as-is.

---

## Part G — Gates and decision-making: assessment

### The stack as built

Per proposal: (1) schema/recipe/holdout/feature-policy validation + cached
pytest + 5K-row preflight; (2) output validation with ≤3 repair attempts and
a noise-floor auto-abandon; (3) **single-split screen** vs the research
line's local incumbent — auto-reject below (−0.001 abs / −0.002 rel);
(4) **CV-bootstrap comparison** vs the official champion (4 folds × 20
bootstraps, common random numbers; close calls at win rate 0.40–0.60 escalate
to 3 partitions / 240 samples); (5) an **advisory** gate (relative lift ≥
0.0005, win rate ≥ 0.60, Bonferroni-adjusted bootstrap CI lower bound ≥ 0,
rank-gini/gini sign agreement, calibration drift ≤ 0.10); (6) **hard
guardrails** that can only block (gini > 0, pred/actual in [0.5, 2.0], lift
CI not entirely below zero); (7) the **LLM verdict**
(promote/local_promote/reject) which alone can promote, firing holdout
evaluation.

### Verdict: the architecture is sound; the calibration and statistics need work

The three-tier design — *advisory gates inform, hard guardrails veto, the LLM
decides* — is the right shape for this system, and the exposure-artifact
incident is a documented case where mechanical gates would have promoted and
LLM-style review caught it. Screening against the line-local incumbent is an
elegant way to let weak lines develop while killing clear losers without a
decision turn. Paired comparisons with common random numbers, champion fold
caching, Bonferroni over attempts-per-champion, and idempotent decision
recording are all better than typical for this kind of harness. Keep the
structure. The issues are in the details:

**G1. The bootstrap CI is anti-conservative: its 80 samples are not 80
observations.** The (4 folds × 20 bootstraps) samples share fold fits — 20
bootstraps within a fold are strongly correlated — yet `bootstrap_lift_summary`
resamples all 80 as if i.i.d., and the MDE divides by √80. The effective
sample size is closer to 4 (the folds). The reported CI and
"probability challenger outperforms" therefore overstate confidence, and the
advisory `bootstrap_lower_bound ≥ 0` check is weaker than its 90%/Bonferroni
label claims. Compensating layers (win rate, sign agreement, LLM review,
holdout) contain the damage, but the *numbers shown to the deciding LLM are
miscalibrated*. Fix without extra compute: block-bootstrap at fold level
(resample folds, then within-fold samples), or report a fold-level paired
summary (4 fold-mean deltas + their range) alongside. The escalation path
already buys 12 folds for close calls, which is where this matters most.

**G2. Make the single-split screen uncertainty-aware instead of
fixed-threshold.** The screen rejects on a point estimate (lift < −0.001).
Because predictions for both models already exist on identical rows, a paired
bootstrap of the screen delta costs milliseconds — no refits. Rejecting on
"CI upper bound < 0" (clearly worse, with stated confidence) instead of an
arbitrary constant converts the screen from a tuned magic number into a
defensible test, and protects against the borderline case observed in this
run (xgboost rejected at −0.0052 — probably genuinely worse given pairing,
but the framework can't currently say so).

**G3. Adaptive overfitting to the fixed fold partition is unmanaged.** All
comparisons in a run reuse partition 0 (champion folds cached on it). Twenty
cycles of select-the-best-on-the-same-partition is the textbook adaptive
data-analysis setting; Bonferroni only counts attempts against the *current*
champion (resets each promotion, capped at lookback 10), so run-level
selection pressure on partition 0 is uncorrected. The holdout exists
precisely to catch the consequence — but holdout evaluation only fires on
promotion, and its results are (rightly) invisible to the agent. Cheap
options: rotate `partition_index` every K cycles (costs champion-fold
rebuilds at rotation only), or surface to the operator a per-promotion
"search-vs-holdout drift" trend so a run that has overfitted its partition is
at least detected. Worth an explicit decision rather than the current
implicit one.

**G4. The local_promote verdict lacks the evidence it nominally adjudicates.**
The full comparison is always vs the *official champion*; the only
line-incumbent evidence is the single-split screen. An agent deciding
`local_promote` ("useful progress for its line") has no CV-grade lift vs the
line incumbent. With fold predictions cached, computing the line-incumbent
paired summary alongside the champion one is nearly free — add it to the
comparison payload when the line incumbent differs from the champion.

**G5. Decision-turn triage (refines review #14 with this run's evidence).**
Both decisions in the reviewed run followed the advisory verdict verbatim,
and 3 of 5 cycles needed no decision at all (screen auto-rejects). The LLM
turn added value zero times in this run — its value is reserved for the
anomaly case, which is exactly the argument for triage: auto-promote when
advisory=promote ∧ guardrails clean ∧ win rate ≥ ~0.95 ∧ no flags
(cycle 1's baseline-beating blowout); auto-record routine inconclusive
rejections when every metric agrees; require the LLM verdict for close calls
(cycle 5), metric disagreement, calibration drift, unexpectedly large lifts,
and any guardrail interaction. Related fast-path: the first comparison of
every run is against the flat global-mean baseline — promoting past a
constant model on clean guardrails does not need a reasoning turn.

**G6. Smaller calibration points.**
- The escalation band [0.40, 0.60] is symmetric around 0.5, but the decision
  threshold is 0.60 — a 0.61 win rate promotes un-escalated while being
  statistically marginal. Centre the band on the threshold (e.g. escalate in
  [0.50, 0.70]).
- `min_relative_lift = 0.0005` is so small that the binding constraints are
  win rate and the (miscalibrated, see G1) CI bound — fine, but be aware the
  lift thresholds are doing almost nothing.
- `gini_above_zero` passes a constant model due to the tie artifact (D5).
- `mean_lift_positive` is a misnomer (it checks ≥ 0.0, and equality passes).
- Reject rationales are free text; a small enum (`noise | inferior |
  artifact_suspected | calibration | other`) would make rejections
  machine-learnable across runs (feeds Part E).

### Is there a better way overall?

No redesign is warranted — the screen → CV → advisory → guardrail → verdict
ladder with line-local screening is a good shape, and better than either
extreme (pure-mechanical gates would have promoted the exposure artifact;
pure-LLM gating would be expensive and noisy). The highest-value changes are
G1 (honest uncertainty in what the decider sees), G2 (principled screen),
and G5 (spend LLM verdicts only where they have ever added value), with G3
decided consciously. Each is local; none changes the search space.

## Suggested order

1. **B1/A1** — framework-owned research log + `record-decision
   --interpretation/--next`; align hook + AGENT.md format; apply to all three
   tracks (A7). *Biggest measured saving; zero search risk.*
2. **A2/A3/A8 (B9)** — telemetry: final checkpoint, model attribution,
   identity reconciliation, poll-output joins. *Restores the measurement basis
   for everything else.*
3. **B2 + B8** — AGENTS.md/CLAUDE.md native injection + minimal stable seed;
   fix the console seed/`--new-run` conflict and the `run-session-cycles 10`
   hint (A6). *Cache + correctness, cheap.*
4. **B3** — timeout guidance / `--wait` to kill poll calls.
5. **A4/A5** — lift/score labeling and recipe-reuse rendering. *Prevents
   mis-planning; trivial code.*
6. **B4/B5/B6** — delta handoff, richer decision output, `--max-cycles`.
7. **B7** — session restart at decision boundaries for long runs (A/B test).
8. **B10** — feed this run's numbers into review #14's triage thresholds.
9. **D4** — strip the milestone token from agent environments (one-line fix,
   closes a leakage channel before it ever opens).
10. **D1/D3** — repair-loop resume + escalation-aware budget guidance (matters
    as soon as a run actually hits repairs/close calls).
11. **E1/E2** — fix Codex-sandbox memory writes and artifact-contaminated
    `peak_gini` *before* enabling memory for any benchmark run; both corrupt
    cross-run learning silently.
12. **F** — resolve the data-provenance contradiction (real freMTPL2 vs
    "synthetic" seed vs renamed columns) as an explicit benchmark-design
    decision.
13. **G1/G2/G5** — fold-level uncertainty in the comparison summary,
    uncertainty-aware screening, and decision-turn triage.

A note on magnitude: items 1–4 above would have cut this specific run from
~3.83M to roughly 1.6–1.9M input tokens (and 61 → ~40 calls) without touching
a single scientific decision the agent made.
