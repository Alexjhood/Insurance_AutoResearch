# Flight Deck — Orchestration Report Application

**Spec version 1.0 — 2026-07-11. Master design document.**

Flight Deck is the interactive front end for AutoResearch *orchestrations* — the
campaigns in which an orchestrator LLM delegates experiment budgets to sub-agent
research runs, reflects between delegations, performs takeovers, and consolidates
finalists in a playoff. One orchestration produces a rich artifact bundle under
`artifacts/orchestrations/<orchestration_id>/`; Flight Deck turns that bundle into
a beautiful, deep, explorable report.

This document is the single source of truth for **what** to build. Two sibling
documents complete the package:

- [`DATA.md`](DATA.md) — snapshot schema (TypeScript contracts) and the exact
  mapping from every artifact/sqlite source to every snapshot field.
- [`PHASES.md`](PHASES.md) — the phased build plan. Each phase is executed in a
  **fresh AI-coding thread** (Opus / GPT / Fable); each phase brief tells the
  builder exactly which sections of this spec to read.

Design tokens (the visual system) live in [`design/tokens.css`](design/tokens.css)
and are **canonical** — builders consume them, never invent colors or type sizes.

---

## 1. Vision & principles

The report must let a user **follow the journey** an orchestration went on:
what the orchestrator decided and why, what every sub-agent did cycle by cycle,
how the champion evolved, where things went wrong (distress, takeovers, forfeits),
and what everything cost in tokens, tool calls, and wall-clock time.

Principles, in priority order:

1. **Narrative first, dashboard always available.** The campaign is a story with
   chapters (delegations), characters (orchestrator + sub-agents), conflict
   (distress, rejections, takeovers) and resolution (playoff). The flagship
   Journey view tells that story; the Overview and Telemetry views serve the
   analyst who wants numbers fast.
2. **Every claim clicks through to evidence.** A number on a chart drills to the
   experiment; the experiment drills to its proposal, comparison metrics, tool
   calls, and raw logs. Nothing is a dead end.
3. **Zero impact on the experimental process.** The app reads only from
   *snapshots* built by an offline ETL step. It never opens a live
   `registry.sqlite`/`telemetry.sqlite` during a campaign. (Live mode is a
   designed-for future, see §9.)
4. **Multi-orchestration from day one.** The snapshot index and every ID scheme
   assume many orchestrations; v1 ships a simple league table, and the schema is
   ready for the future cross-LLM benchmark use.
5. **Shareable.** Any orchestration's report can be exported as a self-contained
   static bundle (`flightdeck export`), no server required.
6. **Mission-control dark as a theme, not a mood.** Dark instrument-panel
   foundation, but exhibits are varied, colorful and data-dense — closer to a
   flight recorder + data-journalism hybrid than a monochrome admin panel.

Non-goals for v1: launching runs, authentication, editing anything, mobile-first
layouts (must be *usable* at tablet width, optimized for ≥1280px).

---

## 2. Architecture

```
flightdeck/
  etl/                 # Python package: snapshot builder
    __init__.py
    build.py           # CLI entry: python -m flightdeck.etl [--all | --orchestration <id>] [--force]
    readers/           # one module per source (orchestration_json, registry, telemetry, notes, playoff, usage)
    schema.py          # dataclasses mirroring DATA.md; single place field names live
    tests/
  server/              # FastAPI app (thin)
    main.py            # serves /api/* from snapshots dir + static SPA from app/dist
  app/                 # Vite + React 18 + TypeScript SPA
    src/
      lib/data/        # DataProvider abstraction (HttpProvider | EmbeddedProvider)
      lib/types.ts     # generated-by-hand from DATA.md (kept in lockstep)
      routes/          # one folder per page (§5)
      exhibits/        # flagship visualizations (§6), self-contained components
      components/      # shared primitives (StatTile, Badge, LogViewer, MarkdownPane, …)
      design/          # tokens.css import + theme plumbing
  snapshots/           # OUTPUT — gitignored. index.json + <orch_id>/…
  export/              # static export builder (Phase 5)
  SPEC.md  DATA.md  PHASES.md  design/tokens.css
```

### 2.1 Data flow

```
artifacts/orchestrations/<id>/**  ──(flightdeck.etl, offline)──►  flightdeck/snapshots/
                                                                    index.json
                                                                    <id>/snapshot.json        (~hundreds of KB)
                                                                    <id>/telemetry_<dNN>.json (lazy, per delegation)
                                                                    <id>/files/**             (verbatim prompts/briefs/logs/reports)
      ▲                                                                   │
      │  never read directly by server/app                                ▼
      └───────────────────────────────────────────────  FastAPI ── React SPA (or static export)
```

- **`index.json`** — one entry per built orchestration with headline stats.
  Powers the Hangar selector and the league table. Rebuilt incrementally.
- **`snapshot.json`** — everything the UI needs for Overview/Journey/Delegation
  pages *except* per-tool-call telemetry and raw file bodies.
- **`telemetry_<dNN>.json`** — full tool-call and model-call event lists for one
  delegation. Lazy-loaded when the user opens the Flight Recorder.
- **`files/`** — raw prompts, briefs, stdout logs, reports, RESEARCH_LOG.md,
  CAMPAIGN_REPORT.md copied verbatim (stdout logs truncated to last 2 MB with a
  truncation marker). Served for the file viewer; inlined on static export only
  for files < 512 KB.

ETL is **idempotent and incremental**: it skips an orchestration whose sources'
max-mtime is older than its snapshot unless `--force`. A partially-written
snapshot is never visible (write to `<id>.tmp/`, atomic rename).

### 2.2 Server (v1)

FastAPI, deliberately thin — it is a file server with an index, so that the SPA
works identically against HTTP and against an embedded export:

| Endpoint | Returns |
|---|---|
| `GET /api/index` | `snapshots/index.json` |
| `GET /api/orchestrations/{id}` | `snapshot.json` |
| `GET /api/orchestrations/{id}/telemetry/{delegation_id}` | `telemetry_<dNN>.json` |
| `GET /api/orchestrations/{id}/files/{path}` | raw file (content-type by extension) |
| `POST /api/etl/rebuild` | runs ETL in a subprocess, streams progress (SSE) — the only "action" in v1 |
| `GET /healthz` | `{status, snapshot_count, built_at}` |

CORS: localhost only. No auth. Port 8799 (console uses 8765; don't collide).
Launch config goes in `.claude/launch.json` so AI builders preview it properly.

### 2.3 Frontend stack (fixed — do not substitute)

- **Vite + React 18 + TypeScript strict.** No Next.js (static export must be a
  dumb folder of files; SSR buys nothing here).
- **react-router v6** (`/`, `/o/:orchId`, `/o/:orchId/journey`,
  `/o/:orchId/delegations/:dId`, `/o/:orchId/telemetry`, `/compare`,
  `/o/:orchId/files/*`).
- **@tanstack/react-query** for data fetching over the `DataProvider` interface.
- **D3 (d3-scale, d3-shape, d3-array, d3-zoom, d3-force)** for the flagship
  exhibits — **no chart framework** (Recharts/ECharts/Plotly are banned; the
  aesthetic control and the exhibit designs in §6 require bespoke SVG/Canvas).
- **@tanstack/react-virtual** for long lists (tool calls, log lines).
- **marked + DOMPurify** for markdown panes (research logs, reports, reflections).
- Styling: **plain CSS with custom properties** from `design/tokens.css` +
  CSS modules per component. No Tailwind, no CSS-in-JS runtime.
- `DataProvider` interface: `getIndex()`, `getSnapshot(id)`,
  `getTelemetry(id, dId)`, `getFileText(id, path)`, `getFileUrl(id, path)`.
  `HttpProvider` hits `/api`; `EmbeddedProvider` reads JSON embedded in
  `<script type="application/json">` tags (static export). **All components are
  provider-agnostic.**

---

## 3. Domain model (concepts the UI speaks)

Read `DATA.md` for exact fields. The conceptual entities:

- **Orchestration (campaign)** — id (`20260711T164959Z`), dataset, target mode,
  orchestrator model, cycle budget, status, delegations[], playoff, notes[].
- **Delegation** — `d01…dNN`: a sub-agent run with a **brief** (direction,
  constraints, seed champion, starting knowledge), backend (e.g.
  `codex-gpt-5-6-luna-medium`), cycle budget, lifecycle timestamps, exit info,
  **distress flags** (subset of: repair_exhausted, all_rejected, budget_overrun,
  crashed, no_finish_delegation, champion_is_baseline, calibration_anomaly,
  cycles_forfeited), takeover/respawn linkage, **report** (agent summary,
  champion metrics, per-experiment decisions, cost), and token usage.
- **Experiment / cycle** — proposal (hypothesis, change summary, expected
  benefit, key risk) → screening metrics → optional CV comparison → **decision**
  (promote / local_promote / reject + reason code + interpretation + next).
  Repairs (up to 3 attempts) attach to a cycle.
- **Champion timeline** — ordered champion changes across the whole campaign
  (per-run `champion_history` stitched across delegations via seed lineage).
- **Operator notes** — orchestrator's `reflection` / `takeover` / `decision`
  entries with timestamps (the campaign's narrative voice).
- **Playoff** — finalists (per-delegation champions replayed under one
  consolidation run), exclusions, final champion lineage.
- **Telemetry** — per delegation: model calls (tokens: input/cached/uncached/
  output/reasoning, per call) and tool calls (name, duration, status, bytes),
  plus per-experiment usage checkpoints (the `LLM_USAGE.md` ledger:
  incremental tokens/calls per experiment step).

Derived metrics the ETL computes once (UI never recomputes):
- per-delegation and campaign **cache hit rate**, tokens/experiment,
  tokens-per-decided-cycle, tool-failure counts;
- **lift ladder**: each experiment's `gini_weighted` and lift vs the
  then-champion, tagged baseline-relative vs incremental (never mix them in one
  aggregate — a known past reporting bug);
- **budget accounting** per delegation: committed / used / forfeited / refunded;
- campaign totals for the index entry (final gini, total tokens, wall-clock,
  distress count, takeover count).

---

## 4. Design system

Canonical tokens: [`design/tokens.css`](design/tokens.css). Summary of intent:

- **Theme**: mission-control dark. Near-black blue-tinted surfaces (`#0b0e14`
  family), high-legibility off-white text, one accent per *semantic role* —
  not per chart. Light theme ships as a secondary `[data-theme="light"]`
  override (used by static export's print stylesheet too).
- **Semantic colors** (fixed meanings across the whole app):
  - `--c-promote` green — promotions, wins, clean exits
  - `--c-localpromote` teal — local promotions / line progress
  - `--c-reject` warm gray — rejections (NOT red; rejection is normal science)
  - `--c-distress` red — distress flags, crashes, forfeits, failures
  - `--c-takeover` amber — takeovers, repairs, warnings
  - `--c-orchestrator` violet — the orchestrator's voice/lane
  - `--c-baseline` dashed slate — baselines/reference lines
  - Delegation identity palette `--c-d1 … --c-d8` (categorical, colorblind-safe,
    used consistently for the same delegation across ALL exhibits and pages)
- **Type**: Inter (UI) + JetBrains Mono (numbers, ids, logs). All metric values
  render in mono with tabular-nums. Type scale in tokens.
- **Texture**: subtle 1px grid lines on chart surfaces, soft glows on live/status
  dots, 8px radius cards, hairline `--border-subtle` separators. No skeuomorphs,
  no scanlines, no gratuitous neon — instrument-panel restraint.
- **Motion**: 150–250ms ease-out for hover/expand; journey view uses
  scroll-linked reveals (IntersectionObserver, translate+fade, `prefers-reduced-motion`
  respected). Charts animate on first mount only.
- **Numbers**: gini to 4dp (`0.3383`), lifts signed with explicit `+`/`−` to 4dp,
  tokens humanized (`7.81M`), durations as `17m 05s`, timestamps in local time
  with UTC on hover.

Accessibility: all decision/distress encodings pair color with a glyph
(▲ promote, ◭ local, ○ reject, ✕ distress, ⚑ takeover); focus-visible rings;
charts get an adjacent "view as table" toggle.

---

## 5. Pages (information architecture)

### 5.1 `/` — Hangar (orchestration selector + league)

- Grid of **campaign cards**: orchestration id (+ user-assignable alias, stored in
  a `flightdeck/snapshots/aliases.json` the ETL preserves), dataset/target chip,
  orchestrator model, status, sparkline of champion gini over the campaign,
  headline stats (final gini, delegations, cycles used/committed, total tokens,
  duration, distress count). Click → Overview.
- **League table** below (v1 of cross-comparison): sortable columns — final
  champion `gini_weighted`, dataset, orchestrator model, sub-agent backend(s),
  cycles used, total tokens, cache hit %, wall-clock, distress flags, takeovers.
  One row per orchestration. A small **cost-vs-performance scatter** (x: total
  tokens log-scale, y: final gini, point = orchestration, colored by dataset)
  sits beside it. Comparisons across datasets are labeled non-comparable
  (grouped, never ranked together).
- Rebuild button → `POST /api/etl/rebuild`, streaming progress toast.

### 5.2 `/o/:id` — Campaign Overview

The analyst's one-screen debrief:

- **Hero strip**: final champion (family, recipe one-liner, gini, vs-baseline
  lift), dataset/target, orchestrator model, duration, status; playoff verdict
  chip linking to the playoff panel.
- **Champion Ascent** exhibit (§6.1) — the centerpiece chart.
- **Budget ledger**: horizontal stacked bar per delegation — cycles used /
  forfeited / refunded / unused, with distress glyphs inline. Totals row.
- **Cost panel**: campaign token totals (stacked: cached vs uncached input,
  output, reasoning), cache-hit dial, tokens-per-decided-cycle, wall-clock split
  (delegation active time vs orchestrator gaps).
- **Distress board**: every raised flag as a card (delegation, flag, detail
  sentence, resolution — e.g. `d02 crashed → diagnosis-only takeover recovered
  the cycle`), linking into the Journey at that moment.
- **Playoff panel**: finalists table (delegation, experiment, replayed gini),
  exclusions with reasons, final lineage (source experiment → consolidation
  experiment), decision mode (auto/operator).

### 5.3 `/o/:id/journey` — The Journey (flagship)

A scrollytelling reconstruction of the campaign. Two synchronized parts:

- **Mission Timeline** (§6.2): a pinned swimlane map (orchestrator lane on top,
  one lane per delegation) that acts as scrubber/minimap — it highlights where
  you are as you scroll the narrative, and clicking any event scrolls to it.
- **Narrative column**: chapters in chronological order:
  1. **Launch** — campaign config, the orchestrator's opening plan (from the
     first prompt/brief), baseline seed.
  2. **One chapter per delegation**: brief card (direction, constraints as
     checklist chips, seed champion, starting knowledge); then **experiment
     cards** in sequence — each shows hypothesis → what changed
     (`change_summary`, recipe diff vs parent when derivable) → outcome metrics
     (gini, lift vs champion with win-rate, calibration, asym pricing loss) →
     decision badge + reason code + agent interpretation. Repair loops render as
     nested amber sub-cards (attempt 1 → 2 → 3 with failed checks). Distress
     flags interrupt the flow as red callouts.
  3. **Interludes between chapters**: the orchestrator's `reflection` note,
     rendered as a distinct violet "orchestrator voice" block — plus, when a
     takeover happened, a full-width amber **takeover scene** (what died, what
     the orchestrator did, what was recovered).
  4. **Finale**: playoff as a bracket-style panel; final champion card; the
     campaign's closing stats.
- Every card deep-links (`/o/:id/journey#d03-x2`) and cross-links to the
  delegation detail page and file viewer (prompt, log at the right offset).

### 5.4 `/o/:id/delegations/:dId` — Delegation Detail

The forensic view of one sub-agent run:

- Header: backend, resolved binary+version, run id, spawn→end times, exit code,
  clean-exit flag, budget summary, distress flags, agent summary (verbatim).
- **Cycle strip**: horizontal stepper of cycles (proposal → screening → CV →
  decision), each step colored by outcome; selecting a cycle filters the page.
- **Metrics panel** per experiment: gini_weighted, rank_gini_weighted,
  asym_pricing_loss, calibration ratio, screening win %, CV lift + fold win
  rate — challenger vs champion side by side, with the decision + rationale.
- **Flight Recorder** (§6.3): zoomable tool-call/model-call timeline for the
  whole delegation (lazy-loads `telemetry_<dNN>.json`).
- **Token burn** (§6.4 single-delegation mode): cumulative token area chart
  with experiment-checkpoint markers (from usage checkpoints), cached vs
  uncached split.
- Tabs for raw artifacts: prompt, brief, RESEARCH_LOG.md, LLM_USAGE.md, report
  JSON (pretty), stdout log (virtualized, searchable).

### 5.5 `/o/:id/telemetry` — Resource Analytics

Cross-delegation resource story:

- **Token Flow** (§6.4 campaign mode): where did 10M+ tokens go — by
  delegation, by experiment step, cached vs uncached vs output vs reasoning.
- **Tool-call mix**: horizontal bars of tool-call counts + total duration by
  tool name per delegation; failure overlays.
- **Turn economics**: model calls per decided cycle, cache-hit trend over
  campaign time (cache % typically climbs — show it), duration histograms.
- **Efficiency panel**: tokens per promoted lift point, wall-clock per cycle,
  screening-vs-CV compute split (from `fit_wall_seconds` where present).

### 5.6 `/compare` — League (routes to Hangar's table in v1)

v1: same league table + scatter as Hangar, standalone route so it can grow into
the benchmark view (per-dataset boards, orchestrator-model head-to-heads) later.

### 5.7 `/o/:id/files/*` — File viewer

Markdown rendered (with heading nav), JSON pretty-printed with collapsing,
logs virtualized with find + line permalinks. Breadcrumbs back to context.

---

## 6. Flagship exhibits (Fable-built — Phase 4)

Builders in other phases create the **data plumbing and layout slots** for
these; the exhibit internals are built by Fable against the contracts in
DATA.md. Each is a self-contained component in `app/src/exhibits/` taking typed
props, no fetching inside.

### 6.1 Champion Ascent — `<ChampionAscent/>`
The campaign in one chart. X: experiment sequence over campaign time (dual
axis: index + clock). Y: `gini_weighted`. Every experiment is a point (glyph =
decision, color = delegation); the **champion line** is a step function that
only moves on promotions; delegation extents render as soft background bands
with the delegation's identity color; takeovers/distress draw thin vertical
amber/red rules. Hover = rich tooltip (experiment, recipe summary, lift,
win-rate, decision rationale); click = navigate to Journey card. A y-zoom
toggle ("magnify the ascent") rescales to the champion range because lifts are
~0.001–0.005 on a 0–0.34 scale — the interesting drama is invisible unzoomed.

### 6.2 Mission Timeline — `<MissionTimeline/>`
The swimlane map used in Journey (pinned, ~180px tall, expandable to full
screen). Lanes: orchestrator (top, violet) + one per delegation. Time on X.
Delegation lifespans as rounded bars (identity color, striped when awaiting
takeover); inside each bar, cycle ticks colored by outcome. Orchestrator lane
carries note glyphs (reflection ◆, takeover ⚑, decision ▣) with connector arcs
dropping to the delegation they concern; spawn arrows from orchestrator lane to
each delegation start; a takeover draws an arc from the dead delegation back up.
Scroll-sync: current narrative chapter glows. This is the exhibit that makes
the orchestration *legible as a system of agents*.

### 6.3 Flight Recorder — `<FlightRecorder/>`
Zoomable (d3-zoom, canvas-rendered for thousands of events) Gantt of one
delegation's telemetry: rows grouped by kind (model calls / tool calls by tool
name / workflow commands), each event a bar (duration) or tick (instant),
colored by tool, red-outlined on failure; model-call rows encode output tokens
as bar height. Brush-zoom, wheel-zoom, minimap strip. Selecting an event shows
a detail drawer (bytes, duration, status, linked workflow command). Experiment
checkpoint markers segment the strip so you can see *which experiment consumed
which flurry of activity*.

### 6.4 Token Flow — `<TokenFlow/>`
Campaign mode: a **flow ribbon** (sankey-style, custom) from Campaign → 
delegations → experiment steps, ribbon width = total tokens, split-shaded
cached/uncached; hovering a ribbon shows the ledger row. Delegation mode: a
cumulative stacked area over the delegation's turns (cached input / uncached
input / output / reasoning) with checkpoint markers and cache-hit % line
overlay. Emphasize the striking fact this data shows: >90% of input is cache
hits and input dwarfs output ~500:1.

### 6.5 Research Tree — `<ResearchTree/>` (stretch, Phase 4 if budget allows)
Force/dagre layout of `research_nodes` across the campaign: nodes = experiments
(glyph/color as elsewhere), edges = parentage, lanes tinted by research line,
seed-lineage edges connecting delegations (d01 champion seeding d02…). Shows
the *shape of the search* — breadth vs depth at a glance.

---

## 7. Static export

`python -m flightdeck.export <orchestration_id> [--out <dir>]`:
builds the SPA (or reuses `app/dist`), emits a folder (and `.zip`) containing
`index.html` with the snapshot + telemetry JSON embedded as
`<script type="application/json" id="fd-embedded-…">` tags, assets inlined or
relative, `EmbeddedProvider` auto-detected at runtime. Files < 512 KB inlined;
larger files listed but marked "not included in export". The export shows a
"static export — built <date>" ribbon and hides Rebuild/live-only affordances.
Must open correctly from `file://`.

## 8. Conventions & guardrails for builders

- Python ≥3.11, stdlib + `fastapi`/`uvicorn` only for server; ETL uses stdlib
  `sqlite3`/`json` (no pandas). Type hints everywhere; `pytest` tests colocated
  under `flightdeck/etl/tests/` run with the repo venv.
- **Never** import from `autoresearch` internals into flightdeck (the ETL reads
  artifact files/sqlite directly) — flightdeck must not couple to research-code
  refactors, and must never trigger the integrity/holdout machinery.
- **Never touch** `artifacts/**` except reads; never write inside orchestration
  folders. Snapshots and exports are the only outputs.
- Repo hygiene: add `flightdeck/snapshots/`, `app/node_modules/`, `app/dist/`,
  `export/out/` to `.gitignore`. Commit `package-lock.json`.
- **Session scope guard**: a PreToolUse hook (`scripts/run_scope_guard.py`)
  blocks AI sessions from reading `artifacts/orchestrations/**` until bound.
  Builder sessions that need to inspect real artifacts must first write
  `{"mode":"analyst","source":"user-request"}` to
  `artifacts/tracks/.scope/<session-id>.json` (session id from the scratchpad
  path or `artifacts/tracks/.scope/guard.log`). Phase briefs repeat this.
- TypeScript types in `app/src/lib/types.ts` mirror DATA.md exactly; when a
  phase changes the schema it updates DATA.md, schema.py, types.ts together and
  bumps `snapshot_schema_version`.
- Verification: every phase ends with the checks listed in its PHASES.md brief,
  run against the real orchestration `20260711T164959Z` (the reference fixture;
  the ETL test suite also carries a synthetic mini-fixture so tests don't
  depend on local artifacts).

## 9. Future roadmap (design for, don't build)

- **Live watch**: server gains `/api/live/orchestrations` reading
  `orchestration.json` + `logs/*.exit.json` mtimes (cheap, read-only) with SSE
  push; Mission Timeline gets a "now" cursor and delegations get status lights.
  The snapshot/live split means Journey depth stays post-hoc; live mode is a
  status layer, not a full report.
- **Launch & monitor**: reuse/absorb `console/backend` job-runner patterns to
  spawn `autoresearch orchestrate` under Claude Code/Codex; strictly additive
  server routes.
- **Benchmark mode**: `/compare` grows per-dataset leaderboards, orchestrator
  head-to-heads, cost-adjusted scores; static-export a whole league site. The
  index schema already carries the fields needed.
