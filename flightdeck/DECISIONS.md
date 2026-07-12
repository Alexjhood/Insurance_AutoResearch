# Flight Deck — Builder Decisions Log

Per `PHASES.md` ("Working agreements"), when the spec is ambiguous the builder
chooses the simplest interpretation consistent with SPEC §1 and records it here.
One line per decision, newest phase last.

## Phase 1 — ETL & snapshots

- **`Experiment.metrics` source resolution.** DATA.md §2.3 says parse
  `experiments.metrics_path` "if within run dir; else research_nodes.metrics_json".
  In the real fixture `metrics_path` is an absolute path into the canonical
  `artifacts/tracks/…` run (outside the orchestration folder), and
  `research_nodes.metrics_json` carries CV/split scalars, not the full metric
  panel. Chosen chain: (1) read the metrics.json rebased onto the delegation run
  dir via its `iterations/…` suffix (keeps the read inside the orchestration
  folder), (2) fall back to `research_nodes.screening_json.challenger_metric_panel`,
  (3) fall back to a `champion_metric_panel` for baselines that were never a
  challenger. All three agree numerically in the fixture.
- **`calibration_ratio` field.** Mapped from the metric panel's
  `predicted_to_actual_ratio` (the panel has no key literally named
  `calibration_ratio`).
- **`Comparison.fold_win_rate`.** DATA.md says "parse bootstrap_summary JSON:
  mean lift + challenger_win_rate", but `bootstrap_summary` in the fixture has no
  win-rate key. Primary source is therefore `research_nodes.metrics_json.cv_win_rate`
  (the true per-fold rate), falling back to any
  `challenger_win_rate`/`fold_win_rate`/`win_rate` key in `bootstrap_summary`.
  `cv_mean_lift` uses `bootstrap_summary.mean_lift` (falling back to
  `metrics_json.cv_mean_lift`).
- **`lift.vs_then_champion` / `lift.kind`.** Taken from `comparison.cv_mean_lift`
  when a CV comparison exists, else from `screening.lift`. `kind` is
  `baseline_relative` when the then-champion (`comparison.champion_id`) is a
  baseline experiment, else `incremental`; screening-only rejects (no comparison)
  are treated as `incremental` since in practice their champion is always a real
  model, never global-mean. Seed and baseline experiments get all-null lift
  (excluded from aggregates, quirk 2).
- **Campaign-wide experiment dedup.** `global_mean_baseline` shares one
  experiment id across every delegation registry; experiments are deduped by
  `experiment_id` across delegations-then-consolidation, first (chronological)
  occurrence winning and owning the `delegation_id`. This also implements quirk 3
  (an id present in a delegation *and* the consolidation registry).
- **`champion_timeline` filtering.** `champion_history` repeats
  `initialised`/`retained` rows in every run. `retained` (no-op) events are
  dropped (SPEC §3 calls this "champion *changes*"), and `initialised` is kept
  only once (first occurrence). Remaining events are sorted chronologically by
  `at`.
- **`ExperimentUsage` source.** `experiment_usage_checkpoints` in the fixture has
  no token columns, so per-experiment usage is parsed from the `LLM_USAGE.md`
  table (DATA.md §2.6's documented fallback). The in-DB token-column path is
  still implemented (and covered by the mini-fixture's d02) for orchestrations
  whose framework version records tokens there.
- **`index.final_gini`.** The finalist gini matching
  `playoff.final_champion_lineage.source.delegation_id` (the playoff-selected
  champion, which need not be the highest-gini finalist), else the best
  delegation champion gini.
- **`Playoff.finalists.replay_experiment_id`.** Matched to the consolidation
  experiment whose id contains the finalist's `_dNN_` delegation tag; null if
  none match.
- **mtime for incremental skip.** `-wal`, `-shm` and `.DS_Store` files are
  excluded from the max-source-mtime scan — opening a WAL sqlite read-only bumps
  the sidecar mtimes, which would otherwise defeat the skip.
- **Defensive metrics-file reads stay inside the orchestration folder.** The ETL
  never follows the recorded absolute `metrics_path` outside the delegation run
  dir; it only rebases the `iterations/…` suffix onto the in-folder run dir. This
  keeps snapshots portable and avoids coupling to `artifacts/tracks/` layout
  (the one deliberate exception is the consolidation registry, which DATA.md §2.3
  explicitly resolves from `artifacts/tracks/<track>/runs/<run_id>`).

## Phase 2 — Server + app shell + Hangar + file viewer

- **SSE wire format.** The rebuild endpoint emits named `progress`, `complete`,
  and `error` events with one JSON object per event. The app consumes the POST
  response stream directly because browser `EventSource` only supports GET.
- **Markdown heading navigation.** Native heading anchors and browser find are
  retained for Phase 2; a dedicated generated table of contents is deferred
  because DATA.md has no heading metadata and `marked` renders the semantic
  heading structure directly.
- **Phase 2 scatter identity.** The placeholder scatter uses the first canonical
  delegation color for every point. Dataset-specific color assignment is left
  to the Phase 5 league polish so colors remain token-only and stable.

## Phase 4 — Flagship exhibits

- **Champion Ascent point color follows decision, not delegation.** SPEC §6.1
  says "glyph = decision, color = delegation", but the approved concept
  (`design/concept.html`) colors glyphs by decision semantics and conveys
  delegation identity through the background bands. The concept sets the target
  look, so the concept wins; delegation identity remains visible via bands and
  band labels.
- **"Consolidation champion" annotation.** The champion timeline's final event
  is the consolidation-run seed at gini 0.3401 (the same model the playoff
  report scores at 0.3373 on the delegation split). To avoid contradicting the
  hero's playoff-sourced 0.3373, the ascent annotates the step line's terminal
  value as "consolidation champion" when the last event is a consolidation seed
  transfer.
- **Distress rules on the ascent are limited to crashes/unclean exits.** Every
  fixture delegation raises at least one distress flag (usually
  `cycles_forfeited`), so drawing a red rule per flag would be noise. Red
  vertical rules render only for delegations with `clean_exit == false` or a
  `crashed` flag; takeover notes always draw an amber rule. The distress board
  and journey callouts still surface every flag.
- **Programmatic scrolls are instant with a target flash.** `behavior:"smooth"`
  scrollIntoView is a silent no-op in the embedded verification browser, so
  timeline clicks and `#hash` deep links scroll instantly and highlight the
  landing card with a 1.6s outline fade (suppressed under
  `prefers-reduced-motion`).
- **Flight Recorder selection uses `click`, not `mouseup`.** d3-zoom installs a
  capture-phase window `mouseup` handler that stops propagation, which silently
  swallows React `onMouseUp`. Event selection binds to `click` (unsuppressed for
  no-drag gestures); shift-drag brush completion stays on `mouseup`.
- **Token Flow campaign mode's second stage is token classes.** SPEC §6.4 shows
  Campaign → delegations → experiment steps, but campaign mode's typed props
  (`TelemetryByDelegation[]`) carry no per-experiment usage. The right-hand
  stage groups into cached/uncached/output/reasoning instead, which also carries
  the §6.4 headline fact (>90% cache hits, input ≫ output); per-experiment burn
  lives in delegation mode where `ExperimentUsage` exists.

## Post-build review (2026-07-12)

- **Journey interludes render after their chapter.** Orchestrator
  reflection/takeover notes describe a delegation's outcome, so they now render
  as interludes following that delegation's chapter (chronological within the
  delegation); previously they rendered before it, d01's reflection was dropped
  by a `di>0` guard, and the campaign-level playoff decision note never
  rendered (it now appears in the finale).
- **Forfeited detection is data-driven.** An undecided experiment shows
  `forfeited` when its delegation has ended (`ended_at` set), `pending`
  otherwise — replacing a hardcoded `d02` special case.
- **Champion descriptor.** The Overview hero and playoff finalists derive a
  recipe one-liner (`estimator · objective · encoding`). Seeds carry no recipe,
  so it resolves via the longest recipe-bearing experiment whose name suffixes
  the seed's name (seed/consolidation names embed the source experiment name).
- **Static export is a single inlined file.** Browsers block external module
  scripts and dynamic `import()` from `file://`, so the export uses a dedicated
  single-chunk build (`npm run build:export` → `dist-export/`) and inlines JS +
  CSS into `index.html`. Payload injection anchors on the LAST `</head>`
  because the inlined bundle contains that literal string (DOMPurify).
