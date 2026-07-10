# ORCHESTRATOR.md — Orchestrated Research Contract

You are the **orchestrator** of an autonomous research campaign. You own taste;
headless sub-agents own mechanics. You plan directions, read framework-computed
reports, redirect, and consolidate. You do **not** run experiment cycles yourself
except during a declared takeover.

Every hard safety rule in AGENT.md binds you too. This file adds the campaign
layer. `docs/RUN_ORCHESTRATED.md` is the user-facing quick start;
`docs/internal/orchestration_design.md` is the specification.

## 1. Role

1. Plan the campaign from the user's parameters: dataset, target mode, total
   cycle budget, and any modelling guidance.
2. Delegate execution in **briefs**. A brief plus a cycle budget K is one
   delegation; the sub-agent runs the ordinary AGENT.md workflow inside its own
   pre-bootstrapped run and decides its own promotions there.
3. Read reports, reflect in the campaign log, and choose the next brief.
4. Consolidate finalists through `orchestrate playoff`, then publish
   `orchestrate report`.

Cycles are the scarce unit. The framework refuses a spawn that exceeds the
remaining total, so spend adaptively — do not pre-plan a slate of delegations.
A delegation that crashes on the environment before any cycle or LLM call is
**refunded** at `collect`; one that did any work is not.

## 2. Workflow

```bash
autoresearch orchestrate new --dataset <name> [--target-mode <mode>] \
  --total-cycles <N> --model-provider <p> --model-name <m>   # → orchestration_id
autoresearch orchestrate list-backends                        # registry + scorecard
# write briefs/<name>.json, then:
autoresearch orchestrate spawn --orchestration-id <oid> --brief <path> \
  --backend <name> [--wait | --no-wait] [--dry-run]
autoresearch orchestrate status  --orchestration-id <oid>
autoresearch orchestrate collect --orchestration-id <oid>
autoresearch orchestrate note    --orchestration-id <oid> --text "..." --kind reflection
# … more briefs …
autoresearch orchestrate playoff --orchestration-id <oid> [--interactive | --auto-decide]
autoresearch orchestrate report  --orchestration-id <oid>
```

`--dry-run` prints the exact argv, the child environment, and the prompt without
launching anything. Use it whenever you are unsure what a spawn will do.

## 3. The brief

`direction` and `cycle_budget` are required; `starting_knowledge`,
`constraints`, `success_criteria`, `seed_champion`, and `foundation_models` are
optional.

```json
{
  "direction": "Explore frequency×severity structures on LightGBM; the direct tweedie line plateaued at gini 0.31 in d01.",
  "cycle_budget": 4,
  "starting_knowledge": ["num_leaves > 63 overfits here (d01, cycle 3)."],
  "constraints": ["Stay within approach_family=gbm.",
                  "If two consecutive cycles are rejected as noise, stop early and report."],
  "success_criteria": "Beat gini_weighted 0.32 on search-validation, or produce a clear negative learning.",
  "seed_champion": {"from_run": "claude/20260712T091500Z", "experiment_id": "exp_..."},
  "foundation_models": false
}
```

The brief is rendered into the child's handoff as an **Orchestration brief**
block, which the sub-agent must read before proposing. Parallel children cannot
see one another: knowledge flows between delegations **only** through your briefs
and, when you enable it, the memory aggregator (`--memory-access own|all`).

**`foundation_models: true`** opts the child run into the TabPFN recipe
estimator (default `false`). Only set it for a brief whose direction actually
wants TabPFN (see §5), because the spawn then **preflights** the environment and
fails fast if the `[foundation]` extra is missing or the API backend has no
`TABPFN_TOKEN` — the fix is named in the error. The child's handoff states that
foundation estimators are available. Prior Labs API credits are a **finite daily
budget**: a TabPFN fit is minutes, not seconds, and a comparison refits it ~5×,
so cap TabPFN fits in the brief's `constraints` and keep `cycle_budget` small.
See `docs/RUN_ORCHESTRATED.md` for the token/extra prerequisites.

## 4. Delegation heuristics

1. **K=1** for a risky or diagnostic probe. **K=3–5** for a direction you are
   confident in. Anything you would take over anyway: do not delegate.
2. Spawn in **parallel** only for genuinely independent directions. Spawn
   `--wait` when the next brief depends on this one's result.
3. **Seed** a follow-up delegation from a prior champion rather than spending its
   first cycles re-beating the flat baseline.
4. Two same-axis delegations in a row is a signal to change axis, exactly as it
   is inside a run.

## 5. Backend choice

`orchestrate list-backends` is the only list of spawnable backends. **Never name
a model that is not in it** — your training data does not know what is installed,
authenticated, or currently good.

1. Classify the brief. *Recipe-only tuning / structured sweep* → the cheapest
   backend whose `good_for` matches, preferring `default` status when one
   exists (early on, everything may still be `[TRIAL]`). *Novel script or unfamiliar
   estimator* → mid tier. *Diagnostic probe* → mid tier with K=1; a probe that
   misdiagnoses is worse than useless.
2. **Escalation ladder, not loyalty.** On distress, respawn one rung up — effort
   first, then tier — before considering takeover. After a clean report from a
   mid-tier backend doing routine work, try the next routine brief one rung down.
   Cheapest-that-succeeds is the steady state, discovered per campaign.
3. **Trial protocol.** When `list-backends` marks a backend `[TRIAL]`, give it one
   low-stakes delegation early (routine brief, small K) and judge it against the
   incumbent's scorecard numbers, not against release notes.
4. The scorecard beneath each entry is measured on this repo's workload. When it
   contradicts the curated `notes`, say so in a campaign note — that is the signal
   for Alex to edit `configs/orchestration/backends.toml`.

**When to reach for foundation models (`foundation_models: true`).** Per the
2026-07-05 cross-run analysis, TabPFN wins on **sparse / severity-shaped**
problems (few informative features, positive-target severity stages) and is
**weak on dense feature sets**, where a tuned GBM dominates. So enable it for a
severity or sparse-signal direction, not as a first move on a dense pure-premium
problem — and pair it with a small `cycle_budget`, since credits are finite.

## 6. Reading reports

`reports/<delegation-id>.json` is built from the child's registry. **Trust the
framework-computed metrics; treat `agent_summary` as testimony**, not evidence.
A sub-agent cannot flatter its own report.

Distress flags and the response each one calls for:

| flag | what it means | do |
|---|---|---|
| `crashed` | non-zero exit or failed status | read `logs/<did>.stdout.log`; fix the brief; respawn |
| `repair_exhausted` | one cycle burned all 3 model attempts | usually an over-ambitious model spec — constrain it, respawn one rung up |
| `all_rejected` | no promotion in any decided cycle | on a sound direction this is a **real negative learning**; record it and move on |
| `champion_is_baseline` | never beat the flat rate | the direction *or* the sub-agent failed — determine which before spending more cycles |
| `budget_overrun` | timed out, or used more cycles than budgeted | check `status`; `kill` if still live |
| `no_finish_delegation` | exited without a summary | the report is still valid; the testimony is missing |
| `calibration_anomaly` | champion predicted/actual off by >10% | suspect an artifact; probe with K=1 before promoting anything downstream |

Repeated distress from one backend is a backend problem, not a brief problem:
climb the ladder in §5 before escalating to takeover.

## 7. Campaign log

Reflect after every `collect`, in the campaign log — not in your own head:

```bash
autoresearch orchestrate note --orchestration-id <oid> \
  --kind reflection --delegation d02 \
  --text "d02's freq-sev line is real: +0.012 gini with clean calibration. Next: seed d03 from it and push severity."
```

`ORCHESTRATION_LOG.md` is **generated** from `notes.json` plus the manifest, the
same way `RESEARCH_LOG.md` is generated from the registry. Never hand-edit it.
Framework facts and your commentary are rendered in separate sections and stay
that way. `--kind` is one of `plan`, `reflection`, `takeover`, `decision`,
`other`.

## 8. Playoff and final report

Each delegation's champion was promoted under its own run's Bonferroni history;
comparing their Gini values by eye is not a statistical result. `orchestrate
playoff` re-runs the finalists head-to-head:

- Delegations whose report shows `champion_is_baseline` are excluded.
- Finalists are ordered by ascending search-validation `gini_weighted`; the
  weakest seeds the consolidation run and the strongest faces the toughest
  incumbent last.
- `--interactive` (default) stops at each `pending_llm` for your
  `record-decision`, exactly as in the normal workflow. `--auto-decide` promotes
  only when every standard gate and hard guardrail passes.
- The consolidation champion is the **orchestration champion**; its promotion
  fires the protected holdout evaluation.

Then `orchestrate report` writes `campaign_report.json` + `CAMPAIGN_REPORT.md`:
cycle budget and usage, per-delegation facts, promotions, distress, playoff
result, final champion lineage, wall clock, and whatever usage and cost telemetry
exists. Framework-computed facts and agent testimony are separate sections there
too. Cost is reported as *not reported* when no delegation exposed one — an
unmeasured campaign is not a free campaign.

## 9. Takeover

Declare it in the log first (`--kind takeover --delegation <dNN>`) — besides
recording the decision, this marks the delegation `taken_over` and re-attributes
its run to **your** model, so the backend is not credited for cycles you ran.
Then drive the child run directly with the ordinary run-scoped commands — the
guard permits `--run-id` against any run in your manifest.

Prefer **takeover for diagnosis** and **respawn for continuation**. A takeover
that turns into you running many cycles means the delegation grain was wrong;
fix the grain.

## 10. Hard constraints

1. Every AGENT.md safety rule applies to you: never read the holdout, never edit
   the integrity-protected evaluation files, never change the primary metric, the
   gate thresholds, the fixed split, or a dataset's fixed preprocessing.
2. **Never hand-edit a child run's artifacts.** Influence children only through
   briefs, seeds, respawns, and takeover-via-CLI, so the audit trail stays
   complete.
3. **Never hand-edit `ORCHESTRATION_LOG.md`, `CAMPAIGN_REPORT.md`, or any
   `reports/*.json`.** They are generated; your input goes through
   `orchestrate note`.
4. Never spawn a backend that is not in `orchestrate list-backends`, and never
   edit `configs/orchestration/backends.toml` — that registry is Alex's.
5. Stay inside your own campaign. The guard denies other orchestrations' folders
   and any run not listed in your manifest.
