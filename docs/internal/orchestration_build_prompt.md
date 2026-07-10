# Build instructions: Orchestrated Research

You are the build agent for the orchestration feature. Your job is to
implement `docs/internal/orchestration_design.md` — read it in full before
writing any code. This file adds execution constraints and acceptance
criteria; where it and the design conflict, stop and ask.

Your work will be independently reviewed against the design doc afterwards,
so build notes and faithful deviation-logging matter as much as the code.

## Ground rules

1. **Branch.** Create branch `orchestration` from `MultiDataset`. If
   `git status` shows uncommitted changes when you start, stop and ask Alex
   whether to commit, stash, or include them — do not silently absorb them.
2. **Protected files.** Never edit the integrity-protected files listed in
   CLAUDE.md ("Hard safety constraints" §2). The design is explicitly built so
   you never need to: all new code lives in `src/autoresearch/orchestration/`
   and calls existing public functions the way `src/autoresearch/tracks.py`
   does. If you believe a protected file must change, stop and ask — that is a
   design flaw to surface, not a manifest update to request.
3. **Single-agent mode stays untouched.** The existing workflow
   (bootstrap-track → handoff → propose → run-session-cycles →
   record-decision), its docs, and its tests must behave identically. Any
   change to shared code paths (`controller/handoff.py`, run manifests,
   `cli.py`, the guard) must be additive and gated on orchestration fields
   being present.
4. **Generated contract files.** `AGENT.md` / `AGENTS.md` / `CLAUDE.md` are
   produced by `scripts/generate_agent_contract.py`. The "Orchestrated mode"
   subsection (design §4.4) goes into the generator's source, never
   hand-edited into the outputs. `scripts/generate_agent_contract.py --check`
   must pass.
5. **No real LLM sub-agents during the build.** Never launch a real
   `claude -p` / `codex exec` sub-agent — that spends Alex's money and time.
   For integration testing, add a `stub` backend to `backends.toml`
   (`tool = "stub"`) whose command is a small script in `scripts/` or
   `tests/` that drives a scripted delegation through the real CLI: reads the
   handoff, drops a fixed trivial recipe proposal, runs its cycle, records a
   decision, calls `finish-delegation`. The stub exercises the entire spawn →
   run → report pipeline with zero LLM calls, and stays in the repo as the
   permanent orchestration smoke test. `--dry-run` (design §8 Phase 1) covers
   the real backends' command/env composition.
6. **Pytest always green.** Full suite passes at the end of every phase, not
   just at the end. New code gets tests in the existing style (pure decision
   logic unit-tested; fixture registries for report generation; guard cases in
   the existing guard test module's style).
7. **Match the codebase.** Frozen dataclasses, explicit public entry points,
   fail-loud validation, the established CLI subparser patterns, and the
   existing docs voice. Read `src/autoresearch/tracks.py`,
   `src/autoresearch/datasets.py`, and `scripts/run_scope_guard.py` as style
   references before starting.

## Build order and commits

Follow the design's five phases (§8) in order. One commit per phase minimum,
message style matching the repo's existing phase commits (e.g.
"Phase 1: orchestration core — manifest, backends, spawner, reports").
Each phase's milestone must actually be demonstrated before moving on — for
phases whose milestone involves real sub-agents (Phase 1's "real sequential
2-delegation campaign", Phase 3's mixed campaign), substitute the stub
backend and record in the build notes that the real-model campaign is
deferred to Alex's post-review validation.

Phase-specific notes:

- **Phase 1:** `--dry-run` output must show the exact command argv and the
  full child environment (with `AUTORESEARCH_*` vars) — this is the review
  surface for spawn correctness. Pin real backend CLI flags only as far as
  you can verify from local `claude --help` / `codex --help`; where a flag is
  uncertain, leave a `# TODO(pin): verify flag` comment and list it in the
  build notes rather than guessing silently.
- **Phase 2:** guard changes go in `scripts/run_scope_guard.py` keeping
  `decide()` pure and fail-open; add orchestrator allow/deny cases to the
  guard tests, including: orchestrator reads own child run (allow), foreign
  run (deny), other orchestration dir (deny), `autoresearch --run-id <child>`
  takeover command (allow), runs-dir enumeration (deny).
- **Phase 4:** the playoff must produce a correct result on a fixture
  orchestration with three finalists of known ordering, including the
  script-finalist copy-in path and the `champion_is_baseline` exclusion.
- **Phase 5:** `ORCHESTRATOR.md` follows the design §6 outline; keep it in
  the same register as AGENT.md (imperative, numbered, no filler). If the
  contract generator should own it too, propose that in the build notes
  rather than deciding unilaterally.

## Acceptance criteria (the review will check these)

1. Full pytest green; `generate_agent_contract.py --check` passes.
2. Stub-backend end-to-end campaign works from a single documented command
   sequence: `orchestrate new` → 2 × `spawn --wait` (stub) → `collect` →
   reports exist with correct cycle counts, champion facts pulled from the
   child registry, and at least one deliberately-provoked distress flag
   (e.g. a stub variant that never calls `finish-delegation` →
   `no_finish_delegation`).
3. Parallel: 2 stub delegations spawned detached, `status` shows both,
   manifest lock prevents bootstrap races (test may simulate concurrency).
4. Guard: all Phase 2 cases above pass; existing guard tests unchanged.
5. Playoff fixture (Phase 4 note) passes; consolidation run's promotion path
   fires the standard holdout-eval hook (assert the hook is invoked — do not
   read holdout data in any test).
6. `list-backends` prints registry + metadata; `backend-stats` aggregates a
   fixture manifest correctly.
7. A single-agent smoke test (existing test or a targeted new one) proves a
   non-orchestrated run's handoff contains no Orchestration-brief block and
   no behaviour change.
8. Build notes exist (below).

## Build notes (review entry point)

Maintain `docs/internal/orchestration_build_notes.md` as you go, in the exact
format of `docs/internal/multi_dataset_build_notes.md`: acceptance-criteria
status table, phase-by-phase summary of what was built, **every deviation
from the design doc with rationale**, open choices you made and why, exact
smoke-test commands with key output, and what was deliberately left undone
(including the deferred real-model campaigns and any unpinned CLI flags).
The reviewer starts from this file; anything not recorded there will be
treated as an unexplained deviation.

## When to stop and ask

Stop and ask Alex rather than improvising when: a protected file seems to
need editing; the design contradicts observed behaviour of the existing code;
a real-tool CLI flag can't be verified locally; or a design ambiguity
materially changes an interface (new CLI flags, manifest schema fields,
report fields). Small internal choices — module layout, helper naming, test
fixtures — decide yourself and log in the build notes.
