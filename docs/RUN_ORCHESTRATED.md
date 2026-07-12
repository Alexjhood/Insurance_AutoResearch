# Running an Orchestrated Campaign

A **campaign** splits research into two jobs. A high-intelligence *orchestrator*
conversation plans directions and reads results; cheap headless *sub-agents* do
the mechanics — filling proposal templates, running cycles, handling repairs,
making routine promote/reject calls inside a narrow brief.

Nothing about single-agent research changes. A sub-agent workspace is an ordinary
tracked run: same `AGENT.md` contract, same proposal inbox, same
`run-session-cycles`, same `record-decision`. Orchestration only pre-bootstraps
the run, adds an **Orchestration brief** block to its handoff, and generates a
report from its registry when it ends. See
[docs/RUN_WITH_CLAUDE_CODE.md](RUN_WITH_CLAUDE_CODE.md) if you want the ordinary
single-agent workflow instead.

## Prerequisites

- The dataset is prepared: `autoresearch list-datasets` shows `prepared=yes`.
- The sub-agent CLI you plan to spawn is installed and authenticated:
  `claude` (`npm install -g @anthropic-ai/claude-code`) or `codex`.
- `autoresearch orchestrate list-backends` prints at least one non-deprecated
  backend for that tool.
- **For foundation-model briefs (`foundation_models: true`)** only: the
  `[foundation]` extra is installed in the environment you orchestrate from
  (`pip install -e '.[foundation]'` — the api client, no torch), and, if you use
  the API backend (`AUTORESEARCH_TABPFN_BACKEND=api`, the practical choice on
  Apple Silicon), `TABPFN_TOKEN` is exported (Prior Labs API key from
  <https://ux.priorlabs.ai/account>, licence accepted). The child inherits your
  interpreter and environment, so both reach it; the spawn preflights them and
  fails fast, naming the fix, if either is missing. **Prior Labs credits are a
  finite daily budget** — a TabPFN fit is minutes and a comparison refits it ~5×,
  so cap TabPFN fits in the brief and keep cycle budgets small. Watch usage at
  <https://ux.priorlabs.ai/account/usage>.

## Launching the orchestrator session

The orchestrator is a normal interactive Claude Code or Codex conversation, bound
to the campaign rather than to a single run:

```bash
cd <repo>
AUTORESEARCH_SCOPE=orchestrator claude
```

The orchestrator session **spawns other CLIs as child processes**, so it needs
real process privileges: the spawned `claude`/`codex` must be able to write
their own home state (`~/.claude`, `~/.codex`). From a sandboxed harness that
means relaunching with the sandbox relaxed — e.g. Codex
`--sandbox danger-full-access` — or orchestrating from Claude Code. The spawn
preflight refuses with a clear message when this is not the case. Authenticate
the sub-agent CLIs once beforehand (`claude /login`; preflight cannot check
auth without a paid call).

Then, as the first prompt:

```
Read ORCHESTRATOR.md before running any command — you are an orchestrator,
not a research agent, so never run `bootstrap-track`. Run an orchestrated
campaign on <dataset> (target mode <mode>) with a total budget of <N> cycles.
Bootstrap it with `orchestrate new --model-provider <p> --model-name <m>`
(your own model identity). <any modelling guidance>
```

The "before running any command" phrasing matters: the auto-loaded research
contract tells an agent to bootstrap first, and an orchestrator that obeys it
creates an orphan run — and, once the guard binds it as research, loses access
to every `orchestrate` command.

`AUTORESEARCH_SCOPE=orchestrator` is optional: a session also auto-binds to a
campaign the first time it successfully runs `orchestrate new` or
`orchestrate spawn`. Setting it up front just makes the intent explicit. Once
bound, the run-scope guard lets the session read `artifacts/orchestrations/<oid>/`
and every child run in that campaign's manifest — and denies every other
orchestration, every foreign run, and `runs/` enumeration.

Sub-agents are untouched by this: they bind as ordinary `research` sessions from
the environment the spawner gives them, confined to their own workspace from the first
tool call.

## Creating a campaign

```bash
autoresearch orchestrate new --dataset porto_seguro --target-mode claim_incidence \
  --total-cycles 20 --model-provider anthropic --model-name claude-opus-4-8
# → orchestration_id: 20260712T090000Z
```

This creates `artifacts/orchestrations/20260712T090000Z/` and pins the total
cycle budget. The framework refuses any spawn that would exceed it. Omit
`--orchestration-id` on later commands to use the most recent campaign.

## Choosing a backend

```bash
autoresearch orchestrate list-backends
```

Each entry prints its tool, track, tier, cost hint, model attribution, the curated
`good_for`/`avoid_for`/`notes` guidance, and — once delegations exist — a
**scorecard** measured on this repo's workload. `configs/orchestration/backends.toml`
is the only list of spawnable backends and is human-owned. `[TRIAL]` marks a
backend earning its keep; `deprecated` entries are refused at spawn.

Rule of thumb (full procedure in ORCHESTRATOR.md §5): recipe tuning and
structured sweeps go to the cheapest matching `default` backend; novel scripts and
diagnostic probes go one tier up; distress means climb one rung — effort first,
then tier. Calibrate a second backend rung with one low-stakes delegation per
campaign. Scorecard distress before 2026-07-12 overcounts because seed replays
were incorrectly recorded as forfeited cycles.

## Writing a brief

`direction` and `cycle_budget` are required. Everything else is optional.

```json
{
  "direction": "Explore frequency×severity structures on LightGBM; the direct tweedie line plateaued at gini 0.31.",
  "cycle_budget": 4,
  "starting_knowledge": ["num_leaves > 63 overfits on this dataset (d01, cycle 3)."],
  "constraints": ["Stay within approach_family=gbm; do not spend cycles on GLMs.",
                  "If two consecutive cycles are rejected as noise, stop early and report."],
  "success_criteria": "Beat gini_weighted 0.32 on search-validation, or produce a clear negative learning."
}
```

Add `"foundation_models": true` to opt the child into the TabPFN recipe
estimator (default `false`; see Prerequisites for the token/extra it requires and
the credit budget). TabPFN suits sparse / severity-shaped directions and is weak
on dense feature sets, so reserve it for those and keep the cycle budget small.

Preview exactly what will be launched before spending anything:

```bash
autoresearch orchestrate spawn --brief briefs/freq_sev.json \
  --backend claude-sonnet-medium --dry-run
```

`--dry-run` prints the full argv, the child's `AUTORESEARCH_*` environment, and
the composed prompt. It launches nothing. This is the review surface for spawn
correctness.

## Sequential campaign (reflect between delegations)

`--wait` blocks until the sub-agent exits, then builds its report. Use it under a
generous command timeout when the next brief depends on this one's result —
which is most of the time. If the harness cuts it off, run `orchestrate status
--follow --until-terminal`; never build a manual re-polling loop.

```bash
autoresearch orchestrate spawn --brief briefs/d01_tweedie.json \
  --backend claude-sonnet-low --wait

autoresearch orchestrate collect
cat artifacts/orchestrations/20260712T090000Z/reports/d01.json

autoresearch orchestrate note --kind reflection --delegation d01 \
  --text "Direct tweedie tops out near 0.31 with clean calibration. Worth testing whether the plateau is the target framing rather than the estimator."

autoresearch orchestrate spawn --brief briefs/d02_freq_sev.json \
  --backend claude-sonnet-medium --wait
```

## Parallel campaign (independent directions)

`--no-wait` launches detached. Bootstrap is serialised under a manifest lock, so
concurrent spawns cannot race; children always pass an explicit `--run-id`, so
nothing depends on "latest".

```bash
autoresearch orchestrate spawn --brief briefs/gbm.json  --backend claude-sonnet-low --no-wait
autoresearch orchestrate spawn --brief briefs/glm.json  --backend codex-gpt-5-5-medium --no-wait
autoresearch orchestrate spawn --brief briefs/nn.json   --backend claude-sonnet-medium --no-wait

autoresearch orchestrate status     # liveness, cycles done, champion, gini, elapsed vs timeout
autoresearch orchestrate collect    # once they are terminal
```

Parallel children cannot see each other's learnings. Knowledge flows between them
only through your briefs' `starting_knowledge`, seeded champions, and — when you
pass `--memory-access own|all` — the memory aggregator.

Spawn in parallel only for genuinely independent directions. Two delegations
exploring the same axis will rediscover the same dead end.

An ensemble cycle costs about `constituents × 5` fits, and more on close-call
escalation; budget K=2–3 with an extended timeout. Keep a single-variant
refinement as a conditional follow-up in its parent brief. Near a plateau,
single-weight or single-hyperparameter deltas are below the gate noise floor;
bundle them or skip them.

## Distress handling

Every report carries mechanical distress flags computed from the child's
registry. `orchestrate status` shows liveness; the report shows the flags.

| flag | response |
|---|---|
| `crashed` | read `logs/<did>.stdout.log`, fix the brief, respawn |
| `repair_exhausted` | the model spec was over-ambitious — constrain it, respawn one rung up |
| `all_rejected` | on a sound direction this is a real negative learning; record it and move on |
| `champion_is_baseline` | the direction or the sub-agent failed — find out which first |
| `budget_overrun` | delegation timed out or overran its cycles; `kill` it if still live |
| `no_finish_delegation` | the report stands; only the testimony is missing |
| `calibration_anomaly` | suspect an artifact; probe with K=1 before building on it |
| `cycles_forfeited` | attempted work never reached a decision; recover a remaining orphan cycle lock |

`early_stop` and `auto_rejected` are informational, not distress. The former is
a clean brief stop; the latter is a framework screening decision and counts as
evidence.

For a confirmed orphan evaluator:

```bash
autoresearch orchestrate recover --orchestration-id <oid> --delegation <dNN>
```

A delegation that overruns its wall-clock allowance is marked `timed_out` but is
**never killed automatically**. Terminate it explicitly:

```bash
autoresearch orchestrate kill --delegation d02
```

## Respawn and seeding

Respawn launches a revised brief. By default it creates a fresh run on the source
delegation's backend:

```bash
autoresearch orchestrate respawn --delegation d02 --brief briefs/d02_revised.json \
  --backend claude-sonnet-medium
```

Two variants:

- `--continue-run` reuses the stopped child's run and champion with a revised
  brief and a fresh cycle budget. Use it when the run is healthy and just needs
  more budget or a corrected direction. The new delegation's report counts only
  its own cycles.
- `--seed-champion from:d01` initialises the new run's champion from d01's
  champion — the same replay mechanic the playoff uses — so the delegation does
  not spend its first cycles re-beating the flat baseline. Recipes replay from
  their config snapshot; script finalists have their script and originating
  proposal copied in and hashed.

Respawn refuses a source whose process is still alive, including a live
`timed_out` child. `kill` it first: two agents writing the same run is not a
recoverable state.

## Playoff

Each delegation's champion was promoted under its own run's Bonferroni history.
Picking a winner by comparing Gini values across runs is not a statistical
result. The playoff re-runs the finalists head-to-head under the standard gates
in a fresh consolidation run.

```bash
autoresearch orchestrate playoff                     # interactive (default)
autoresearch orchestrate playoff --include d01,d02,d04
autoresearch orchestrate playoff --auto-decide
```

Mechanics: delegations flagged `champion_is_baseline` are excluded; finalists are
ordered by ascending search-validation `gini_weighted`; the weakest seeds the
consolidation run; each stronger finalist is replayed as a challenger through the
normal `cv_bootstrap` gauntlet, so the strongest faces the toughest incumbent
last.

Interactive mode stops at each `pending_llm` and prints the exact
`record-decision` command. Run it, then run `playoff` again to resume — the
playoff is resumable and idempotent. `--auto-decide` promotes only when the
advisory decision is `promote`, every standard check is literally true, and the
hard guardrails pass; it is fail-closed on missing evidence.

The consolidation run's final champion is the **orchestration champion**, and its
promotion fires the ordinary protected holdout evaluation. Results land in
`playoff/playoff_report.json` and `.md` with per-pairing gate tables and champion
lineage back to the originating delegation.

## Takeover

When a delegation needs diagnosis rather than redirection, drive its run
yourself. The guard permits `--run-id` against any run in your manifest:

```bash
autoresearch orchestrate note --kind takeover --delegation d02 \
  --text "Taking over d02 to diagnose the calibration anomaly before spending more cycles."

autoresearch --track claude --run-id 20260712T091500Z show-latest-handoff
autoresearch --track claude --run-id 20260712T091500Z run-session-cycles 1
autoresearch --track claude --run-id 20260712T091500Z record-decision <cmp> \
  --decision reject --rationale "..." --reason-code artifact_suspected \
  --interpretation "..." --next "..."
```

Declare the takeover in the log first. Prefer takeover for *diagnosis* and
respawn for *continuation*: a takeover that turns into many cycles of your own
means the delegation grain was wrong.

Never hand-edit a child run's artifacts. Influence children only through briefs,
seeds, respawns, and takeover-via-CLI — that is what keeps the audit trail
complete.

## The campaign log and the final report

`orchestrate note` appends a timestamped entry to `notes.json` and regenerates
`ORCHESTRATION_LOG.md`. The Markdown is derived: framework facts (budget,
delegations, status) render in their own sections, your commentary in its own.
Never hand-edit it — edit through the command. When a planned stage is dropped,
add a follow-up note closing it out and stating why.

```bash
autoresearch orchestrate report          # prints the Markdown
autoresearch orchestrate report --json   # the JSON payload
```

This writes `campaign_report.json` + `CAMPAIGN_REPORT.md`, records the report path
in the manifest, and marks the campaign `completed` once every delegation is
terminal and no playoff is in flight. The report separates
**framework-computed** facts (cycles, decisions, promotions, mean promoted Gini
lift, distress counts, playoff result, champion lineage, wall clock, usage/cost)
from **agent testimony** (each sub-agent's `finish-delegation` summary, verbatim).

Cost is reported as *not reported* when no delegation exposed one; a campaign
whose tool does not surface spend is unmeasured, not free.

## Backend scorecard

```bash
autoresearch orchestrate backend-stats
autoresearch orchestrate backend-stats --json
```

Aggregates every collected delegation report across every campaign, per backend:
delegations, cycles, promotion rate, distress rate by flag, repair attempts per
cycle, mean promoted Gini lift, and cost per cycle / per promotion when reported.
The same numbers appear beneath each `list-backends` entry. Delegations without a
collected report contribute nothing and are counted as skipped — never as zeros.

When the scorecard contradicts the curated `notes` in `backends.toml` (a trial
backend outperforming a default; a default's distress rate creeping up after a
silent model revision), that is the signal to edit the registry.

## Where to look afterward

- `artifacts/orchestrations/<oid>/orchestration.json` — the manifest: which child
  runs belong to this campaign
- `.../ORCHESTRATION_LOG.md` — the generated campaign narrative
- `.../CAMPAIGN_REPORT.md`, `.../campaign_report.json` — the final report
- `.../briefs/`, `.../prompts/`, `.../logs/` — exactly what each sub-agent was
  asked and what it printed
- `.../reports/<did>.json` — per-delegation facts from the child's registry
- `.../playoff/playoff_report.md` — the consolidation gauntlet
- `artifacts/tracks/<track>/runs/<run-id>/RESEARCH_LOG.md` — each child's own log

## Smoke test without spending anything

The `stub`, `stub-no-finish`, and `stub-codex` backends drive the entire
spawn → run → report pipeline through the real CLI with zero LLM calls. They are
the fastest way to check that a campaign is wired correctly:

```bash
autoresearch orchestrate new --dataset french_motor --total-cycles 2 \
  --model-provider anthropic --model-name claude-opus-4-8
AUTORESEARCH_SKIP_PYTEST_GATE=1 autoresearch orchestrate spawn \
  --brief /tmp/brief.json --backend stub --wait
autoresearch orchestrate collect
autoresearch orchestrate report
```

`AUTORESEARCH_SKIP_PYTEST_GATE=1` skips the pytest gate each child bootstrap
would otherwise run. Use it only for stub runs, and only after the suite passed.
