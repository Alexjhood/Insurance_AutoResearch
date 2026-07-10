# Build instructions: environment-independent contracts + orchestrated foundation models

You are the build agent for two related pieces of work:

1. **Kill the "tabpfn drift"** — the generated agent contracts must stop
   depending on which optional extras are installed in the generating
   environment.
2. **Make foundation models orchestratable** — an orchestrator campaign must be
   able to delegate TabPFN modelling (API backend) and, as a later phase, TabFM
   modelling (Modal serverless GPU) to sub-agents.

Read these before writing any code, in order:
- `docs/internal/orchestration_design.md` + `docs/internal/orchestration_build_notes.md`
  (the orchestration system you are extending; the build notes' "Pre-flight"
  section describes the drift you are fixing)
- `docs/internal/foundation_models_integration_plan.md` (TabPFN as it exists)
- `docs/internal/tabfm_integration_plan.md` + `docs/internal/tabfm_spike_report.md`
- `src/autoresearch/models/recipe/foundation.py`, `scripts/tabfm_modal_app.py`,
  `scripts/generate_agent_contract.py`

The ground rules of `docs/internal/orchestration_build_prompt.md` all apply
verbatim: branch from the current `Orchestrator` head (name the branch
`foundation-orchestration`), never touch integrity-protected files, single-agent
mode unchanged, generated contracts edited only via the generator, **no real
LLM sub-agents and no paid TabPFN/Modal calls during the build** (stub/fixture
everything; the api/Modal execution paths get real validation from Alex
afterwards), pytest green per phase, build notes in the established format at
`docs/internal/foundation_orchestration_build_notes.md`.

## The drift, precisely

`generate_agent_contract.py` calls `enable_foundation_models()` +
`recipe_menu()`, so the estimator table only contains `tabpfn` when the
`[foundation]` extra is importable. This repo's `.venv` lacks the extra; the
operator's system Python has it. Every environment regenerates the contracts
its own way, and the pytest gate (which re-derives the contract in-process)
flip-flops the working tree between the two. This bit the first real campaign:
its very first spawn failed the gate on contracts another environment had
generated.

## Phase 1 — static foundation declarations (the drift fix)

- Give the recipe package a **static declaration** of every foundation
  estimator — name, legal objective × encoding combos, short caveat line —
  importable with **no** optional dependency (e.g. `FOUNDATION_ESTIMATOR_SPECS`
  in a module `foundation.py` re-exports from). Implementation registration
  (`register_foundation_estimators`) keeps requiring the packages; the
  *metadata* must not.
- The generator builds its estimator table from builtin registry + static
  foundation specs, annotating foundation rows (e.g. "requires
  `bootstrap-track --enable-foundation-models` + the `[foundation]` extra").
  Output must be byte-identical whether or not the extra is installed —
  that is the acceptance test, and it must be enforced by a unit test that
  fakes both conditions.
- Recipe validation for a foundation estimator must fail with a *specific*
  message on (a) a run that did not opt in, and (b) an opted-in run whose
  environment lacks the package — today (b) surfaces as "unknown estimator".
- Install the `[foundation]` extra into `.venv` (api client only — do not pull
  torch/local weights unless the extra already does) so this repo's gate
  environment matches the operator's. Regenerate contracts once; the tabpfn
  row now appears *by declaration*, identically everywhere. Note the stale
  `stash@{0}` in the build notes as superseded.

## Phase 2 — orchestrated TabPFN (API backend)

- **Brief field** `foundation_models: true` (optional, default false):
  `spawner._bootstrap_child_run` forwards it as
  `bootstrap_track(..., enable_foundation_models=True)`; the handoff's
  Orchestration-brief block states that foundation estimators are enabled.
- **Preflight** (extend `backends.preflight_backend` or a sibling called when
  the brief opts in): fail the spawn fast when `TABPFN_TOKEN` is unset while
  `AUTORESEARCH_TABPFN_BACKEND` resolves to `api`, or when the extra is not
  importable. Message names the fix (token env var / pip extra). Remember the
  child inherits the parent env, so an exported `TABPFN_TOKEN` already reaches
  it — do not add it to the stripped-variable list.
- **Playoff/replay**: a consolidation run must be able to replay a tabpfn
  finalist — enable foundation models on the consolidation run iff any
  finalist's source run had the manifest flag; `replay_experiment` should fail
  with a clear message rather than "unknown estimator" when the destination
  cannot support it.
- **Docs**: ORCHESTRATOR.md brief section + one line in the backend-choice
  heuristics (TabPFN's regime per the 2026-07-05 cross-run analysis: sparse /
  severity-shaped problems, weak on dense feature sets); RUN_ORCHESTRATED.md
  prerequisites (token, extra). Mention the Prior Labs API credit budget is
  finite — briefs should cap TabPFN fits, and the sub-agent contract already
  carries the compute-budget rules.
- **Tests**: brief validation, spawner forwarding (fixture, no real bootstrap
  needed beyond the existing test patterns), preflight cases, replay gating.

## Phase 3 — TabFM via Modal (stop-and-ask first)

`scripts/tabfm_modal_app.py` is a spike, and the integration plan records two
**user setup steps it is blocked on**. Before writing any Phase 3 code, verify
with Alex that the Modal account/secret setup is done; if not, stop after
Phase 2 and record Phase 3 as blocked in the build notes.

If unblocked: register `tabfm` as a foundation estimator per the same static-
declaration pattern (declared always; executable only with Modal configured),
with the fit/predict path delegating to the Modal app per
`tabfm_integration_plan.md`. Wall-clock spent inside Modal counts against the
experiment compute budget (the clock does not stop because the GPU is remote).
Preflight: `modal` importable + token configured. No Modal call in any test —
fixture the transport boundary.

## Acceptance criteria

1. Full pytest green; `generate_agent_contract.py --check` passes **in both
   environments** (`.venv` and the operator's system Python) — state the
   command outputs for both in the build notes.
2. Unit test proving contract bytes are identical with and without the extra.
3. Foundation recipe on a non-opted run and on an extra-less environment both
   fail with their specific messages (tested).
4. A stub-backend campaign whose brief sets `foundation_models: true` produces
   a child run manifest with `foundation_models: true` and a handoff noting it.
5. Preflight failure cases for missing token / missing extra (tested, no paid
   calls).
6. Playoff replay of a foundation finalist into a consolidation run is gated
   correctly (fixture test).
7. Build notes exist with every deviation logged; Phase 3 either implemented
   to the same standard or explicitly recorded as blocked and why.
