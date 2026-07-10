# Orchestrated Research — Design Specification

**Date:** 2026-07-10
**Status:** Proposed (agreed decisions from Alex baked in; open questions at the end)
**Audience:** whoever builds this (Claude Code / Codex build session) + future maintainers

---

## 1. Problem and goal

Today a single conversation drives an entire run: one agent (often a cheap
model, to control cost) does the strategy *and* the mechanics — reading
handoffs, writing proposal JSON, babysitting `run-session-cycles`, recording
decisions. That couples two very different jobs:

- **Taste** — choosing directions, reading results scientifically, knowing when
  to pivot, when a result smells like an artifact, when to stop. High-value,
  benefits from an expensive model with a small, curated context.
- **Execution** — filling proposal templates, running cycles, handling repair
  requests, making routine promote/reject calls inside a narrow brief. Cheap
  models handle this fine when the direction is already chosen.

**Goal:** a two-tier workflow. A high-intelligence **orchestrator**
conversation plans the campaign, delegates experiment batches to cheap
headless **sub-agents**, reads their structured reports, reflects, redirects,
and consolidates winners — while the existing single-agent workflow keeps
working unchanged.

### Agreed shape (decisions already made)

| Decision | Choice |
|---|---|
| Where orchestration lives | In the framework: new `autoresearch orchestrate …` commands spawn headless CLI sub-agents. The orchestrator itself is a normal interactive Claude Code / Codex conversation. |
| Sub-agent authority | Full cycle **including** promote/local_promote/reject decisions, inside its own run. |
| Unit of delegation | Chosen per spawn: a brief plus a cycle budget K (1..N). K=1 gives tight orchestrator control; K=3–5 gives an autonomous mini-run. |
| Parallelism | Via independent tracked runs (no change to sequential comparison semantics inside a run). Consolidation via a framework **playoff** command that re-runs finalists head-to-head under the same gates in a fresh consolidation run. |
| Sub-agent tools v1 | Claude Code (`claude -p`) and Codex (`codex exec`), behind a tool-agnostic backend registry so OpenCode etc. can be added by config. |
| Intervention | Report-triggered: sub-agents run to completion/failure; distress flags in the end-of-run report drive respawn-with-better-brief or direct takeover by the orchestrator. |
| Compatibility | Additive. Single-agent mode (and its docs, guard behaviour, and CLI) unchanged. |

---

## 2. What already exists that we build on (do not rebuild)

1. **Run isolation.** Every run is already a self-contained world:
   `artifacts/tracks/<track>/runs/<run-id>/` with its own registry, inbox,
   handoff, research log. Parallel runs need no new isolation work.
2. **Cycle budget.** `bootstrap-track --cycles N` already pins a per-run
   budget and stops the run at N cycles. A sub-agent's K is exactly this.
3. **Env-based scope binding.** `scripts/run_scope_guard.py` already binds a
   session from the launch environment at `SessionStart`
   (`AUTORESEARCH_SCOPE=research` + `AUTORESEARCH_TRACK` +
   `AUTORESEARCH_RUN_ID`). Spawned sub-agents get pinned to their run with
   zero new guard machinery on the child side.
4. **Cross-run comparison.** `src/autoresearch/tracks.py::compare_tracks()`
   is a privileged pairwise champion comparison (paired resamples, bootstrap
   CI, gate simulation, no promotion). The playoff generalises this pattern.
5. **Telemetry.** `telemetry/` already records per-run LLM calls
   (`llm_model_calls`, `llm_turns`); orchestration adds a rollup, not a new
   collector.
6. **Protected files.** Nothing in this design touches the integrity-protected
   evaluation/comparison files. All new code lives in a new
   `src/autoresearch/orchestration/` package that *calls* existing public
   functions (`paired_comparison`, `promotion_decision`,
   `get_official_champion`, …) the same way `tracks.py` does.

---

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (interactive, high-intelligence model)             │
│ Claude Code / Codex conversation, scope = orchestrator          │
│                                                                 │
│  autoresearch orchestrate new / spawn / status / collect /      │
│                           respawn / playoff / report            │
└───────┬───────────────────┬───────────────────┬─────────────────┘
        │ spawn(brief A,K=4)│ spawn(brief B,K=4)│ spawn(brief C,K=1)
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ SUB-AGENT run │   │ SUB-AGENT run │   │ SUB-AGENT run │
│ claude -p     │   │ codex exec    │   │ claude -p     │
│ (haiku/sonnet)│   │ (cheap model) │   │ ...           │
│ track/run pre-│   │ own run, own  │   │               │
│ bootstrapped; │   │ champion, own │   │               │
│ existing AGENT│   │ decisions     │   │               │
│ .md workflow  │   │               │   │               │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │ orchestration_report.json (framework-generated)
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ artifacts/orchestrations/<orch-id>/   manifest, briefs, reports │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
              orchestrate playoff  →  consolidation run
              (finalists re-run through the normal gauntlet;
               winner = orchestration champion, holdout eval fires)
```

A sub-agent run **is a normal tracked run**. It uses the existing AGENT.md
contract, the existing proposal inbox, `run-session-cycles`,
`record-decision`, repair flow, research log — everything. Orchestrated mode
adds only: (a) the run is pre-bootstrapped by the spawner, (b) the handoff
carries an **orchestration brief** block, (c) a report is generated at the
end. This is what keeps single-agent mode untouched: there is one workflow,
and orchestration is a caller of it.

---

## 4. New package: `src/autoresearch/orchestration/`

```
orchestration/
├── __init__.py
├── manifest.py      # Orchestration + delegation records (JSON on disk)
├── backends.py      # Backend registry: configs/orchestration/backends.toml
├── spawner.py       # Pre-bootstrap child run, compose prompt, launch process
├── monitor.py       # Liveness/progress polling, wall-clock timeout enforcement
├── report.py        # Build orchestration_report.json from child registry state
└── playoff.py       # Consolidation: seed run, replay finalists, rank
```

CLI: a new `orchestrate` command group in `cli.py` (subcommands below). All
subcommands take `--orchestration-id <oid>` (auto-resolves to the latest when
omitted, mirroring run-id resolution).

### 4.1 Orchestration record (`manifest.py`)

`artifacts/orchestrations/<orch-id>/` (orch-id = same `YYYYMMDDTHHMMSSZ`
timestamp convention):

```
orchestration.json      # the manifest (below)
briefs/<delegation-id>.json
prompts/<delegation-id>.md      # exact prompt sent to the sub-agent (audit)
logs/<delegation-id>.stdout.log # child process stdout/stderr
reports/<delegation-id>.json    # collected orchestration reports
playoff/                # playoff outputs (section 4.6)
ORCHESTRATION_LOG.md    # framework-generated narrative (like RESEARCH_LOG.md)
```

`orchestration.json`:

```json
{
  "orchestration_id": "20260712T090000Z",
  "dataset": "porto_seguro",
  "target_mode": "claim_incidence",
  "created_at": "...",
  "status": "active | consolidating | completed | abandoned",
  "total_cycle_budget": 20,
  "cycles_committed": 12,
  "delegations": [
    {
      "delegation_id": "d01",
      "brief_path": "briefs/d01.json",
      "backend": "claude-haiku",
      "track": "claude",
      "run_id": "20260712T091500Z",
      "cycle_budget": 4,
      "status": "spawned | running | completed | failed | timed_out | killed",
      "pid": 12345,
      "spawned_at": "...", "ended_at": "...",
      "report_path": "reports/d01.json"
    }
  ],
  "consolidation": {"track": "claude", "run_id": null, "playoff_report": null}
}
```

The manifest is the single source of truth for **which child runs belong to
this orchestration** — the guard's orchestrator scope (section 5) reads it.
Child run manifests also get a back-pointer field
(`orchestration_id`, `delegation_id`) written at bootstrap time.

### 4.2 Backend registry (`backends.py` + `configs/orchestration/backends.toml`)

Tool-agnostic by construction: a backend is a named command template plus
metadata. V1 ships `claude` and `codex` entries; adding OpenCode later is a
config edit (plus, at most, an output-parsing shim).

A backend is a **model × thinking-effort combination**: effort/reasoning
level is part of the command template (e.g. Codex's reasoning-effort flag,
Claude's effort/thinking settings), so "Sonnet 5 low" and "Sonnet 5 medium"
are two distinct named backends, selectable and measurable independently.

Each entry also carries **selection metadata** — human-maintained guidance
the orchestrator reads when choosing (see 4.7):

```toml
[backends.claude-sonnet-low]
tool = "claude"                    # selects the adapter shim + the track
command = ["claude", "-p", "--model", "claude-sonnet-5",
           "--output-format", "stream-json", "--verbose",
           "--permission-mode", "acceptEdits",
           "--max-turns", "{max_turns}"]   # + pinned effort flag
prompt_via = "stdin"
track = "claude"
model_provider = "anthropic"       # forwarded to bootstrap-track for memory attribution
model_name = "claude-sonnet-5-low"
# ── selection metadata ──
tier = "cheap"                     # cheap | mid | frontier
status = "default"                 # default | trial | deprecated
cost_hint = "$"                    # coarse relative cost: $ | $$ | $$$
good_for = ["recipe_tuning", "structured_sweeps"]
avoid_for = ["novel_scripts"]
notes = "Reliable on recipe-only briefs; escalate effort for repair-heavy estimators."

[backends.claude-sonnet-medium]
tool = "claude"
command = ["claude", "-p", "--model", "claude-sonnet-5", ...]  # medium effort
track = "claude"
model_provider = "anthropic"
model_name = "claude-sonnet-5-medium"
tier = "mid"
status = "default"
cost_hint = "$$"
good_for = ["novel_scripts", "diagnostic_probes", "repair_prone"]

[backends.codex-luna-medium]
tool = "codex"
command = ["codex", "exec", "--sandbox", "workspace-write", "{prompt}"]  # + effort flag
prompt_via = "argv"
track = "codex"
model_provider = "openai"
model_name = "codex-5.6-luna-medium"
tier = "cheap"
status = "default"
cost_hint = "$"
good_for = ["recipe_tuning", "structured_sweeps"]
```

Notes:
- `track` comes from the backend, so guard's `ALLOWED_RESEARCH_TRACKS`
  (`claude`/`codex`/`opencode`) is honoured with no guard change on the child
  side.
- The adapter shim per tool is small: how to pass the prompt, how to detect a
  clean exit, and (best-effort) how to extract cost/usage from the tool's JSON
  output stream into the delegation record.
- Exact CLI flags (permission mode, sandbox flags, effort flags, max-turns
  defaults) are pinned during the build after testing each tool headless in
  this repo; the table above is the shape, not final flag values.
- `autoresearch orchestrate list-backends` prints the registry with metadata
  and (once available) scorecard stats appended — the one view the
  orchestrator consults when choosing.

### 4.3 Brief contract (`briefs/<delegation-id>.json`)

What the orchestrator supplies per spawn — this is the delegation API:

```json
{
  "direction": "Explore frequency×severity structures on LightGBM; the direct
                tweedie line plateaued at gini 0.31 in delegation d01.",
  "cycle_budget": 4,
  "starting_knowledge": [
    "Champion recipe from run 20260712T091500Z: {…recipe json…}",
    "num_leaves > 63 overfits on this dataset (d01, cycle 3)."
  ],
  "constraints": [
    "Stay within approach_family=gbm; do not spend cycles on GLMs.",
    "If two consecutive cycles are rejected as noise, stop early and report."
  ],
  "success_criteria": "Beat gini_weighted 0.32 on search-validation, or
                       produce a clear negative learning about freq-sev here.",
  "seed_champion": {"from_run": "claude/20260712T091500Z", "experiment_id": "..."}   // optional
}
```

- `direction`, `cycle_budget` are required; everything else optional.
- `seed_champion` (optional): the spawner initialises the child run's champion
  from a named prior experiment (recipe replayed via the normal experiment
  path, script + proposal copied in by the privileged spawner) instead of
  `global_mean`, so follow-up delegations don't waste cycles re-beating a flat
  baseline. This is the same replay mechanic the playoff uses (4.6).
- The brief is rendered into a new **"Orchestration brief"** block in the child
  run's handoff (a small addition to `controller/handoff.py`, rendered only
  when the run manifest carries an `orchestration_id`). Single-agent runs
  never see the block.

### 4.4 Spawner (`spawner.py`) — `orchestrate spawn`

```bash
autoresearch orchestrate spawn --orchestration-id <oid> \
  --brief briefs/my_brief.json --backend claude-haiku [--wait]
```

Steps (all framework-side, before the sub-agent model sees anything):

1. Validate the brief; check remaining `total_cycle_budget`.
2. **Pre-bootstrap the child run**: programmatically run the equivalent of
   `bootstrap-track --track <backend.track> --new-run --cycles <K>
   --model-provider … --model-name …`, write `orchestration_id`/`delegation_id`
   into the run manifest, apply `seed_champion` if present, export the handoff
   (which now embeds the brief block).
3. Compose the launch prompt (`prompts/<delegation-id>.md`). It is short by
   design — the real contract already lives in AGENT.md / AGENTS.md, which
   both tools auto-load from the repo root:

   > You are a research sub-agent working under an orchestrator. Your run is
   > **already bootstrapped**: track `<track>`, run id `<run-id>`, cycle budget
   > `<K>` — do **not** run `bootstrap-track` and do not start any other run.
   > Begin with `autoresearch --track <track> --run-id <run-id>
   > show-latest-handoff` and follow the standard workflow from step 2 of the
   > contract. Your handoff contains an **Orchestration brief** — treat its
   > direction, constraints, and stop conditions as binding, on par with the
   > Active dataset block. Spend your K cycles adaptively within the brief.
   > When your budget is exhausted (or a brief stop-condition fires), finish
   > with `autoresearch --track <track> --run-id <run-id> orchestrate
   > finish-delegation --summary "<3–6 sentence scientific summary: what you
   > learned, what you'd try next, anything that smelled artifactual>"` and
   > then stop.

4. Launch the backend command as a **detached child process**, environment:
   `AUTORESEARCH_SCOPE=research`, `AUTORESEARCH_TRACK=<track>`,
   `AUTORESEARCH_RUN_ID=<run-id>`, plus per-delegation
   `AUTORESEARCH_MEMORY_ACCESS` if the orchestrator requested it. The guard's
   existing `SessionStart` env-binding confines the child to its run from the
   first tool call. Stdout/stderr tee to `logs/<delegation-id>.stdout.log`.
5. Record pid + status in the manifest and return immediately (or block until
   exit with `--wait`, for the sequential-with-reflection style).

**Parallel spawns** are just repeated `spawn` calls. Bootstrap is done
serially inside the spawner (steps 1–2 under a manifest lock file) so
`latest_run.json` races can't happen; children always pass `--run-id`, so
nothing downstream depends on "latest".

**AGENT.md change (small):** add a short "Orchestrated mode" subsection: *if
your launch prompt names a pre-bootstrapped run, skip bootstrap, pin the given
ids, obey the Orchestration brief block, and end with `finish-delegation`.*
No other contract change; single-agent instructions stay as-is.

### 4.5 Monitoring + report (`monitor.py`, `report.py`)

```bash
autoresearch orchestrate status   --orchestration-id <oid>   # table of delegations
autoresearch orchestrate collect  --orchestration-id <oid> [--delegation d01]
autoresearch orchestrate kill     --orchestration-id <oid> --delegation d01
```

- `status`: per delegation — process liveness, cycles completed (from the
  child's session/registry state), current champion + `gini_weighted`,
  last-activity timestamp, and elapsed wall-clock vs. a per-delegation
  timeout (default: `cycle_budget × per-cycle budget × safety factor 2`; the
  monitor marks `timed_out` and can kill on request — it never auto-kills in
  v1).
- `finish-delegation` (run-scoped; callable by the sub-agent itself): stores
  the sub-agent's free-text summary, marks the delegation completing, and
  triggers report generation.
- `collect` / report generation (`report.py`) builds
  `reports/<delegation-id>.json` **from the child's registry — never from the
  sub-agent's claims**. Contents:

```json
{
  "delegation_id": "d01", "run_id": "...", "track": "claude",
  "status": "completed",
  "cycles": {"budget": 4, "used": 4},
  "champion": {"experiment_id": "...", "model_family": "recipe",
               "recipe": {...}, "gini_weighted": 0.317,
               "rank_gini_weighted": 0.31, "asym_pricing_loss": 0.042,
               "calibration_ratio": 1.01, "beat_seed_baseline": true},
  "experiments": [
    {"cycle": 1, "name": "...", "decision": "reject", "reason_code": "noise",
     "lift_vs_champion": -0.001, "interpretation": "..."}
  ],
  "agent_summary": "<the finish-delegation free text>",
  "distress": {
    "flags": ["repair_exhausted", "all_rejected", "budget_overrun",
              "crashed", "no_finish_delegation", "champion_is_baseline",
              "calibration_anomaly"],
    "active": ["all_rejected"],
    "detail": "4/4 cycles rejected; 2 with reason_code=artifact_suspected"
  },
  "cost": {"wall_clock_minutes": 38, "llm_usage": {...from telemetry/tool output...}}
}
```

  Distress flags are mechanical predicates on registry/telemetry state (each
  one cheap and unambiguous). The orchestrator's contract (section 6) tells it
  how to respond: refine brief and `respawn`, or take over directly.
- `orchestrate respawn --delegation d01 --brief <revised>`: convenience for
  the takeover-lite path — new delegation, optionally `--seed-champion
  from:d01` or `--continue-run` (reopen the same run with additional cycles
  via the existing session mechanics) for the case where the run itself is
  healthy and just needs more budget or a corrected direction.

### 4.6 Playoff / consolidation (`playoff.py`) — `orchestrate playoff`

Purpose: parallel runs each end with their own champion evaluated under
per-run Bonferroni histories; picking a winner by eyeballing metrics is not a
fresh statistical comparison. The playoff re-runs finalists head-to-head under
the standard gates.

```bash
autoresearch orchestrate playoff --orchestration-id <oid> \
  [--include d01,d02,d04] [--auto-decide | --interactive]
```

Mechanics:

1. Collect each included delegation's champion. Drop any whose report shows
   `champion_is_baseline` (nothing to consolidate).
2. **Rank finalists** by search-validation `gini_weighted` (all runs share the
   same fixed split pack, so scores are directly comparable for *ordering*).
3. Create a **consolidation run** (normal tracked run on the orchestrator's
   choice of track, flagged `consolidation=true` in its manifest; recorded in
   the orchestration manifest). Initialise its champion from the **lowest-
   ranked** finalist.
4. Replay the remaining finalists as experiments in ascending rank order,
   each through the standard comparison gauntlet (`cv_bootstrap` gates,
   `pending_llm` stop). Recipes replay natively (`model.recipe` is portable by
   design); script-based finalists have their script + proposal copied in by
   the privileged playoff code (same cross-run read privilege as
   `compare_tracks`). Ascending order means the strongest candidate faces the
   toughest incumbent — the final promotion is the meaningful one.
5. Decisions: `--interactive` (default) stops at each `pending_llm` for the
   orchestrator to `record-decision` exactly as in the normal workflow —
   keeping the "verdict is yours" principle at the orchestration level.
   `--auto-decide` promotes iff all mechanical gates pass (documented as a
   convenience for clean, well-separated playoffs).
6. The consolidation run's final champion is the **orchestration champion**;
   its promotion fires the normal protected holdout evaluation. Write
   `playoff/playoff_report.md` + `.json` (per-pairing gate tables in the
   `compare_tracks` report style, final ranking, champion lineage back to the
   originating delegation).

No protected files change: the playoff composes `paired_comparison` /
`promotion_decision` / the existing experiment and comparison runners through
their public entry points, exactly like `tracks.py` does today.

### 4.7 Backend selection — how model/effort choice is determined and stays current

The orchestrator chooses a backend per spawn. That choice must be right today
("Codex 5.6 Luna medium and Sonnet 5 low/medium are good sub-agents") and stay
right as models release and task difficulty varies. Three layers, from
slowest-moving to fastest:

**Layer 1 — the curated registry (human-owned ground truth).**
`backends.toml` is the *only* list of spawnable backends, and Alex owns it.
When a new model releases, adding it is a config edit with
`status = "trial"`; retiring one is `status = "deprecated"` (existing
delegations finish, new spawns refuse it). The metadata fields (`tier`,
`good_for`/`avoid_for`, `cost_hint`, `notes`) encode current human judgement —
the "I know Luna medium is a good sub-agent" knowledge lives here, in one
reviewable file, instead of in a prompt that goes stale. The orchestrator is
never asked to know model names from its own training data: it reads
`list-backends` and chooses from what's offered.

**Layer 2 — selection heuristics in ORCHESTRATOR.md (task-profile → tier).**
The contract gives the decision procedure, not a fixed model list:

- Classify the brief: *recipe-only tuning / structured sweep* → cheapest
  `default` backend whose `good_for` matches; *novel script or unfamiliar
  estimator* → mid tier; *diagnostic probe of a suspicious result* → mid tier
  with K=1 (a probe that itself misdiagnoses is worse than useless);
  *anything you'd take over anyway* → don't delegate.
- **Escalation ladder, not loyalty:** on distress (`repair_exhausted`,
  `champion_is_baseline`, incoherent `agent_summary`), respawn one rung up
  (effort first, then tier) before considering takeover. On a clean report
  from a mid-tier backend doing routine work, try the next brief one rung
  down. Cheapest-that-succeeds is the steady state, discovered empirically
  per campaign rather than assumed.
- **Trial protocol:** when `list-backends` shows a `trial` backend, allocate
  it one low-stakes delegation (routine brief, small K) early in the campaign
  and judge it by its report against the incumbent's scorecard numbers. This
  is how a new model earns `default` status — evidence from this repo's
  actual workload, not release notes.

**Layer 3 — the empirical scorecard (evidence that keeps layers 1–2 honest).**
Every delegation report already records backend, cycles, decisions, distress
flags, and cost (section 7). `orchestrate backend-stats` aggregates these
across campaigns (reading orchestration manifests; optionally rolled into the
cross-run memory aggregator, keyed on the same `model_provider`/`model_name`
attribution `bootstrap-track` already requires):

| per backend | e.g. |
|---|---|
| delegations / cycles run | 14 / 52 |
| promotion rate (promotions ÷ cycles) | 0.19 |
| distress rate, by flag | repair_exhausted 7%, all_rejected 21% |
| repair attempts per cycle | 0.3 |
| mean lift per promoted experiment | +0.004 gini |
| cost per cycle / per promotion | $0.40 / $2.10 |

`list-backends` appends these numbers to each entry, so the orchestrator's
choice is guided by measured performance *on this workload*; Alex reviews the
same table when editing `backends.toml` metadata, closing the loop. When the
scorecard contradicts the curated `notes` (a trial backend outperforming a
default, a default's distress rate creeping up after a silent model revision),
that's the signal to update Layer 1.

Deliberately **not** in scope: automatic backend selection or auto-promotion
of trial backends by the framework. Model choice is a taste decision with
cost consequences — exactly what the orchestrator (guided by the registry and
scorecard) and Alex (owning the registry) are for. Revisit only if the manual
loop proves tedious in practice.

---

## 5. Run-scope guard: a new `orchestrator` scope

The orchestrator must read many child runs (reports, logs, and — during
takeover — drive a child run via the CLI), which `research` scope forbids and
`analyst` scope over-grants (everything, including unrelated runs and other
orchestrations). Add a third mode to `scripts/run_scope_guard.py`:

- **Binding.** `AUTORESEARCH_SCOPE=orchestrator` (+ optional
  `AUTORESEARCH_ORCHESTRATION_ID`) at launch, or auto-bind on `PostToolUse`
  when the session successfully runs `autoresearch orchestrate new` /
  `orchestrate spawn` (mirroring today's bootstrap auto-bind). Scope file:
  `{"mode": "orchestrator", "orchestration_id": "<oid>"}`.
- **Policy.** Allow: everything an unbound session may touch, plus
  `artifacts/orchestrations/<oid>/`, plus any run folder listed in that
  orchestration's manifest (the guard resolves the manifest fresh on each
  check, so newly spawned children are covered), plus `autoresearch` commands
  whose `--run-id` targets a listed child run (this is the takeover path).
  Deny: other orchestrations' folders, runs not in the manifest, `runs/`
  directory enumeration outside listed children.
- Sub-agents are untouched — they bind as plain `research` sessions via the
  existing env path, and the existing denial message ("knowledge of other runs
  reaches you only through memory") continues to hold for them.
- Guard stays fail-open, decision logic stays pure/unit-tested; new tests in
  `tests/` cover orchestrator allow/deny cases.

---

## 6. The orchestrator contract: `ORCHESTRATOR.md`

A sibling to AGENT.md, loaded by the orchestrator conversation (the user
starts an orchestrator session by asking Claude Code / Codex to read it, or
via a repo skill/launcher — same convention as today's RUN_WITH_*.md docs).
Contents in brief:

1. **Role.** You own taste, not mechanics. Plan the campaign from the user's
   parameters (dataset, target, total cycle budget, guidance). Delegate
   execution; never run experiment cycles yourself except during a declared
   takeover.
2. **Workflow.** `orchestrate new --dataset … --target-mode … --total-cycles N`
   → write brief → `spawn` (choose backend + K; `--wait` for
   sequential-reflective, parallel spawns for breadth) → `status` / wait →
   `collect` → **reflect in the orchestration log** (`orchestrate note` writes
   a timestamped entry into `ORCHESTRATION_LOG.md`; hypothesis/outcome facts
   are framework-written like the research log) → next brief(s) → … →
   `playoff` → `orchestrate report` (final campaign report for the user).
3. **Delegation heuristics.** K=1 for risky/diagnostic probes; K=3–5 for
   confident directions; parallel spawns only for genuinely independent
   directions (they can't see each other's learnings mid-flight — knowledge
   flows between delegations only through *your* briefs and, when enabled, the
   memory aggregator). Budget arithmetic: cycles are the scarce unit; the
   framework refuses spawns that exceed the remaining total. **Backend
   choice**: read `list-backends` (registry metadata + scorecard), match the
   brief's task profile to the cheapest suitable `default` tier, escalate on
   distress, trial-protocol any `trial` backend — full procedure in
   section 4.7; never spawn a model name that isn't in the registry.
4. **Reading reports.** Trust the framework-computed metrics, treat
   `agent_summary` as testimony. Distress-response table:
   `repair_exhausted`/`crashed` → inspect logs, fix the brief (usually an
   over-ambitious model spec), respawn; `all_rejected` on a sound direction →
   probably a real negative learning, record it and move on;
   `champion_is_baseline` → the direction or the sub-agent failed — check
   which before burning more cycles; repeated distress from one backend →
   climb the escalation ladder (effort up, then tier up — section 4.7) before
   escalating to takeover.
5. **Takeover.** Declare it in the log, then drive the child run directly via
   the normal run-scoped commands (guard permits it). Prefer takeover for
   *diagnosis*, respawn for *continuation* — a takeover that turns into you
   running many cycles means the delegation grain was wrong.
6. **Hard constraints.** All AGENT.md safety rules apply to you too (holdout,
   protected files, fixed splits/metric). Additionally: never edit a child
   run's artifacts by hand; influence children only through briefs, seeds,
   respawns, and takeover-via-CLI, so the audit trail stays complete.

Docs to add/update alongside: `docs/RUN_ORCHESTRATED.md` (user-facing quick
start: how to launch an orchestrator session with `AUTORESEARCH_SCOPE=orchestrator`,
example campaign transcript), one paragraph in `docs/architecture.md`, CLI.md
entries, and a pointer in README. AGENT.md gets only the small "Orchestrated
mode" subsection (4.4).

---

## 7. Cost & telemetry rollup

- Each delegation records: wall-clock, child-tool usage (parsed best-effort
  from the tool's JSON output stream), and the run's existing telemetry DB
  contents.
- `orchestrate report` aggregates per-delegation and campaign totals:
  cycles used vs. budget, spend per promotion, distress counts per backend —
  giving the data to answer "is orchestration actually cheaper per unit of
  lift than single-agent?" (the whole point of the exercise).
- The memory aggregator's model attribution keeps working because the spawner
  forwards each backend's `model_provider`/`model_name` to `bootstrap-track`.
  The orchestrator session itself is attributed via `orchestrate new
  --model-provider/--model-name` (the orchestrator's own model), stored in the
  manifest.

---

## 8. Build plan (phased, each phase shippable + tested)

**Phase 1 — Orchestration core, sequential.**
`manifest.py`, `backends.py` (+ backends.toml with the Claude entries incl.
selection metadata, `list-backends`),
`spawner.py` with `--wait` only, brief block in `handoff.py`,
`finish-delegation`, `report.py` with the distress predicates, `orchestrate
new/spawn/collect`. AGENT.md orchestrated-mode subsection. Tests: manifest
round-trip, brief validation, prompt composition, report built from a fixture
registry, spawner dry-run mode (`--dry-run` prints the command + env instead
of launching — also the standard debugging tool). *Milestone: a real
sequential 2-delegation campaign on Porto with claude-haiku sub-agents.*

**Phase 2 — Orchestrator scope + parallelism.**
Guard `orchestrator` mode + tests; `monitor.py`, `status`, `kill`, timeouts;
detached (non-`--wait`) spawning with the manifest lock; `respawn`
(`--seed-champion`, `--continue-run`). *Milestone: 3 parallel delegations,
orchestrator reads all three reports, guard denies a foreign-run read.*

**Phase 3 — Codex backend.**
Codex adapter shim (prompt passing, exit detection, usage parsing),
AGENTS.md parity check for the orchestrated-mode subsection, headless flag
pinning. *Milestone: mixed campaign (claude + codex delegations).*

**Phase 4 — Playoff.**
`playoff.py`: finalist collection, consolidation-run seeding, recipe replay,
script copy-in, ascending-order gauntlet, interactive decisions, playoff
report; `--auto-decide`. Tests against fixture runs with known champions.
*Milestone: end-to-end campaign → playoff → orchestration champion → holdout
eval fired.*

**Phase 5 — Contract, docs, telemetry rollup.**
`ORCHESTRATOR.md`, `docs/RUN_ORCHESTRATED.md`, `orchestrate note` +
`ORCHESTRATION_LOG.md`, `orchestrate report` with cost rollup, `orchestrate
backend-stats` (cross-campaign scorecard, 4.7), architecture/CLI docs. *Milestone: a full campaign driven purely from the documented
orchestrator workflow by a fresh session, no folklore needed.*

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Headless CLI flags drift / differ across tool versions | Backends are config, not code; `--dry-run` for verification; pin flags per tool during Phase 1/3 testing; log the exact command in `prompts/`+`logs/` for every delegation. |
| Sub-agent ignores the brief or never calls `finish-delegation` | Report is framework-generated regardless; `no_finish_delegation` distress flag; wall-clock timeout marks the delegation `timed_out`; the brief is in the handoff (the one document the workflow forces the agent to read). |
| Cheap models make bad promote decisions inside their runs | Contained by design: a child promotion only affects that run's champion; the playoff re-tests every finalist under full gates before anything becomes the orchestration champion, and holdout eval fires only there. |
| Parallel children rediscover the same dead ends | Knowledge routing is explicitly the orchestrator's job (briefs + `starting_knowledge`); optionally enable `AUTORESEARCH_MEMORY_ACCESS=own/all` per spawn for aggregator-mediated sharing. |
| Playoff multiple-comparisons optimism | Consolidation run uses the standard Bonferroni lookback over its own gauntlet; finalist count is small (≤ number of delegations); report states the caveat explicitly. |
| Guard complexity creep | Orchestrator mode reuses the existing scope-file + pure-`decide()` structure; fail-open preserved; policy delta is ~“allow manifest-listed children”. |
| Zombie child processes | pids in the manifest; `status` checks liveness; `kill` for cleanup; spawner refuses to spawn when the same delegation id is still alive. |

## 10. Open questions (small; none block Phase 1)

1. **Backend model pins** — initial `backends.toml` entries per Alex's current
   known-good picks: Sonnet 5 at low and medium effort for Claude Code, Codex
   5.6 Luna at medium for Codex; exact CLI effort flags pinned during Phase 1/3
   testing. Whether a haiku-tier entry earns `default` status is a trial-
   protocol question (4.7), not a spec decision.
2. **Timeout defaults** — safety factor 2 over the per-cycle compute budget is
   a guess; tune after the first real campaigns.
3. **`--auto-decide` default for playoff** — spec says interactive by default;
   revisit once we've seen how often playoff gates are borderline.
4. **Orchestrator launch ergonomics** — env var + "read ORCHESTRATOR.md" is
   the v1 launcher; a `/orchestrate` skill or wrapper script can come later if
   friction shows up.
