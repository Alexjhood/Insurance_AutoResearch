# Flight Deck — Phased Build Plan

Each phase runs in a **fresh AI-coding thread** (Opus 4.8 / GPT-Sol; Phase 4 is
Fable). Start every thread by pasting the phase's **kickoff prompt** verbatim.
Phases are strictly ordered; a phase must end with its acceptance checks green
and a commit on branch `flightdeck` (branch off `foundation-orchestration`,
or `main` once that merges).

**Read-first rule for every phase**: `flightdeck/SPEC.md` §§1–4 + 8, then this
file's own phase section, then the SPEC/DATA sections the phase names. Do not
read the whole spec set for narrow phases — the briefs say what's needed.

**Scope-guard note (all phases)**: reading `artifacts/orchestrations/**` from a
Claude Code/Codex session is blocked by `scripts/run_scope_guard.py` until you
write `{"mode":"analyst","source":"user-request"}` to
`artifacts/tracks/.scope/<session-id>.json` (session id: see the scratchpad
path in your system prompt, or `artifacts/tracks/.scope/guard.log`). Do this
first if a read is denied.

Reference fixture for all verification: orchestration `20260711T164959Z`.

---

## Phase 1 — ETL & snapshots (Python) — Opus/GPT

**Reads**: SPEC §2.1, §3, §8; DATA.md in full.
**Builds**: `flightdeck/etl/` per SPEC layout — readers, `schema.py`
dataclasses mirroring DATA.md, `build.py` CLI, incremental/atomic build,
aliases.json preservation, files/ copying with log tail-cap, warnings channel.
Plus: `.gitignore` entries; a synthetic **mini-fixture** under
`flightdeck/etl/tests/fixtures/` (tiny hand-built orchestration folder incl.
minimal sqlite DBs created by a fixture script) so tests run anywhere.

**Acceptance**:
1. `python -m flightdeck.etl --orchestration 20260711T164959Z` succeeds; emits
   `snapshots/index.json`, `snapshot.json`, `telemetry_d01..d05.json`, `files/`.
2. Spot-check parity (write as a pytest that runs only when the real fixture
   exists): index final_gini == 0.3373153637090053 (playoff final champion);
   d01 tokens input == 7_801_150; d02 distress active includes `all_rejected`
   and `cycles_forfeited`; 5 delegations; ≥1 takeover note; champion_timeline
   non-empty and chronological; every Experiment with a comparison has
   `decision` populated; seed experiments flagged `is_seed`.
3. Full pytest suite (mini-fixture) green: schema round-trip, quirks 1–7 from
   DATA.md §4 each covered, idempotent rebuild skips unchanged, `--force`
   rebuilds, atomicity (no partial dir on injected failure).
4. Re-running against a *copy* of the fixture with `reports/d03.json` deleted
   still builds with warnings (defensive-degradation test).
5. `python -m flightdeck.etl --all` builds every orchestration present without
   error.

## Phase 2 — Server + app shell + Hangar + file viewer — Opus/GPT

**Reads**: SPEC §2.2–2.3, §4, §5.1, §5.7, §8; DATA.md §1, §2.7.
**Builds**: FastAPI server (all §2.2 endpoints incl. SSE rebuild), Vite app
scaffold with router/react-query/DataProvider (HttpProvider + EmbeddedProvider
stub), theme plumbing consuming `design/tokens.css` (dark default, light
override, toggle persisted), app chrome (top bar: orchestration switcher,
theme toggle, rebuild button; left nav within an orchestration), **Hangar**
page (cards + league table + cost/perf scatter — the scatter may be a simple
placeholder SVG scatter, Fable restyles later), **file viewer** route
(markdown/JSON/virtualized log with search + line permalinks), shared
primitives (`StatTile`, `Badge` with the §4 glyph vocabulary, `MetricNumber`,
`MarkdownPane`, `Drawer`). Adds `.claude/launch.json` entries: `flightdeck-api`
(uvicorn :8799) and `flightdeck-app` (vite :5199, proxying /api → 8799).

**Acceptance**:
1. `npm run build` + `tsc --noEmit` clean; server tests (httpx) for every
   endpoint incl. 404s and path-traversal rejection on `/files/`.
2. With snapshots built: Hangar lists the fixture with correct headline stats;
   league table sorts; card click routes to `/o/<id>` (placeholder page OK).
3. File viewer renders CAMPAIGN_REPORT.md (markdown), d01 report (JSON tree),
   d01 stdout log (virtualized, find works, `#L123` permalinks scroll).
4. Rebuild button streams progress and refreshes the index.
5. Browser check via launch.json preview: no console errors; dark + light
   themes both render; screenshot of Hangar attached to the final message.

## Phase 3 — Overview, Delegation detail, Telemetry, Journey (structure) — Opus/GPT

**Reads**: SPEC §3, §4, §5.2–5.6; DATA.md §2–3.
**Builds**: all remaining routes with **complete data plumbing and layout, and
placeholder exhibit slots**: Campaign Overview (hero, budget ledger bars,
cost panel, distress board, playoff panel — all real; `<ChampionAscent/>` slot
renders a basic line/scatter fallback), Delegation detail (header, cycle strip,
metrics panels, artifact tabs; `<FlightRecorder/>` slot = simple virtualized
tool-call table; token burn slot = simple stacked area), Telemetry page
(tool-mix bars, turn economics, efficiency panel — simple but real SVG charts),
Journey page (full narrative column per SPEC §5.3 with chapter/experiment/
interlude/takeover cards, deep links, cross-links; `<MissionTimeline/>` slot =
static lane sketch). `/compare` route aliasing the league table. Fallback
charts must already use tokens.css variables and §4 semantic colors/glyphs.

**Acceptance**:
1. Every SPEC §5 page renders the fixture with real data end-to-end; no
   dead-end elements (every metric/card links somewhere per SPEC principle 2).
2. Journey shows: 5 chapters, d02 takeover scene, orchestrator reflections
   between chapters, playoff finale; experiment cards show decision badge +
   reason code + interpretation; d02's undecided experiment renders as
   forfeited (quirk 1).
3. Delegation d01 page: 4-5 experiments on cycle strip, LLM_USAGE-backed token
   burn with checkpoint markers, tool-call table loads lazily from telemetry
   endpoint.
4. `tsc --noEmit` + build clean; exhibit slots receive exactly the typed props
   defined in DATA.md-derived types (Fable will swap internals only).
5. Browser pass with screenshots of all five pages, both themes.

## Phase 4 — Flagship exhibits — **Fable**

**Reads**: SPEC §4, §6; DATA.md §2.3–2.6, §3; `design/concept.html` (the
approved visual concept — open it in a browser; it sets the target look for
ChampionAscent, MissionTimeline, and the card/scene language).
**Builds**: `ChampionAscent`, `MissionTimeline` (with journey scroll-sync),
`FlightRecorder` (canvas + zoom), `TokenFlow` (both modes); `ResearchTree` if
budget allows. Replaces Phase-3 fallbacks in place (same props). Polish pass
over motion/microinteractions and any visual debt from Phases 2–3.

**Acceptance**: each exhibit matches its §6 spec against the fixture; tooltips,
zoom, click-through navigation work; reduced-motion respected; view-as-table
toggles present; screenshots/GIFs of each exhibit; both themes.

## Phase 5 — Static export + league polish + hardening — Opus/GPT

**Reads**: SPEC §5.1, §5.6, §7, §8.
**Builds**: `flightdeck/export/` builder (embedded-provider bundle per SPEC §7,
zip output, export ribbon, file:// compatibility), EmbeddedProvider completion,
league table polish (per-dataset grouping, non-comparable labeling),
empty/error states everywhere, README.md for flightdeck (setup, build, export,
troubleshooting), performance pass (snapshot payload budget: snapshot.json
< 2 MB for the fixture; telemetry lazy; route-level code splitting).

**Acceptance**:
1. `python -m flightdeck.export 20260711T164959Z` → open `index.html` from
   `file://`: all pages incl. exhibits work, no network requests fired.
2. Export of a second (even partial/crashed) orchestration works.
3. Lighthouse-style sanity: initial route JS < 400 KB gz; no console errors.
4. Full pytest + tsc + build green; README verified by following it verbatim.

## Phase 6+ (roadmap, not scheduled)

Live watch (SSE status layer) → launch & monitor (absorb console job-runner)
→ benchmark mode. See SPEC §9. Do not build ahead of need.

---

## Kickoff prompts (paste verbatim into the fresh thread)

> **Phase 1**: You are building Phase 1 of Flight Deck. Read
> `flightdeck/PHASES.md` (Phase 1 + the scope-guard note), `flightdeck/SPEC.md`
> §§1–4, 2.1, 3, 8, and `flightdeck/DATA.md` in full. Then implement exactly
> the Phase 1 scope and run all Phase 1 acceptance checks against orchestration
> `20260711T164959Z`. Work on branch `flightdeck`. Do not modify anything
> outside `flightdeck/` except `.gitignore`. Commit when green.

> **Phase 2**: You are building Phase 2 of Flight Deck. Phase 1 (ETL) is done —
> build snapshots first if `flightdeck/snapshots/` is empty. Read
> `flightdeck/PHASES.md` (Phase 2), `flightdeck/SPEC.md` §§1–4, 2.2, 2.3, 5.1,
> 5.7, 8, `flightdeck/DATA.md` §1 + §2.7, and `flightdeck/design/tokens.css`.
> Implement exactly the Phase 2 scope; verify in the browser via the
> launch.json preview; run all acceptance checks. Branch `flightdeck`; only
> touch `flightdeck/` and `.claude/launch.json`. Commit when green.

> **Phase 3**: You are building Phase 3 of Flight Deck. Phases 1–2 are done.
> Read `flightdeck/PHASES.md` (Phase 3), `flightdeck/SPEC.md` §§3–6 (treat §6
> as *slot contracts only* — you build fallbacks, not the exhibits), and
> `flightdeck/DATA.md` §§2–3. Implement exactly the Phase 3 scope; verify all
> pages in the browser against orchestration `20260711T164959Z`; run all
> acceptance checks. Branch `flightdeck`; only touch `flightdeck/`. Commit
> when green.

> **Phase 4** (Fable): Build the flagship exhibits per `flightdeck/SPEC.md` §6
> and the Phase 4 brief in `flightdeck/PHASES.md`, replacing the Phase-3
> fallback components without changing their props.

> **Phase 5**: You are building Phase 5 of Flight Deck. Phases 1–4 are done.
> Read `flightdeck/PHASES.md` (Phase 5) and `flightdeck/SPEC.md` §§5.1, 5.6, 7,
> 8. Implement exactly the Phase 5 scope and run all acceptance checks. Branch
> `flightdeck`; only touch `flightdeck/`. Commit when green.

## Working agreements for builder threads

- Follow SPEC §8 conventions strictly (stack is fixed; no chart frameworks, no
  Tailwind, no autoresearch imports, no writes under `artifacts/`).
- When the spec is ambiguous, choose the simplest interpretation consistent
  with SPEC §1 principles and record the choice in a `flightdeck/DECISIONS.md`
  appendix line — do not redesign.
- If a schema change is genuinely needed, update DATA.md + `etl/schema.py` +
  `app/src/lib/types.ts` together and bump `snapshot_schema_version`.
- End every phase by attaching proof (test output + screenshots) and a short
  handoff note in the commit message.
