# Foundation-orchestration build notes

**Branch:** `foundation-orchestration` (from the `Orchestrator` head,
`23b7b9c`).
**Status:** Phases 1–3 complete. Real Modal/TabPFN endpoint validation deferred
to Alex (build ran no paid calls, per the rules).
**Spec:** `docs/internal/foundation_orchestration_build_prompt.md`, which extends
`docs/internal/orchestration_design.md` (built in
`docs/internal/orchestration_build_notes.md`). All ground rules of
`docs/internal/orchestration_build_prompt.md` apply verbatim: never touch
integrity-protected files, single-agent mode unchanged, generated contracts
edited only via the generator, **no real LLM sub-agents and no paid
TabPFN/Modal calls during the build** (stub/fixture everything), pytest green per
phase.

This file is the review entry point. It is factual: what was built per phase,
every deviation with rationale, exact commands with key output, and what was
deliberately left undone. One commit per phase.

---

## Two problems this build solves

1. **The "tabpfn drift".** The generated agent contract's content depended on
   which optional extras happened to be importable in the *generating*
   environment. `generate_agent_contract.py` called `enable_foundation_models()`
   + `recipe_menu()`, so the estimator table listed `tabpfn` only when the
   `[foundation]` extra was importable. This repo's `.venv` lacked it; the
   operator's system Python had it. Every environment regenerated the contract
   its own way, and the pytest gate (which re-derives the contract in-process)
   flip-flopped the working tree between the two. This bit the first real
   campaign — its very first spawn failed the gate on contracts another
   environment had generated (see `orchestration_build_notes.md`
   "Post-first-real-campaign fixes", last paragraph). **Fixed in Phase 1.**
2. **Foundation models are not orchestratable.** An orchestrator campaign cannot
   yet delegate TabPFN (API) or TabFM (Modal GPU) modelling to sub-agents.
   **Phases 2–3.**

---

## Acceptance criteria — status

| # | Criterion | Status |
|---|---|---|
| 1 | Full pytest green; `generate_agent_contract.py --check` passes in both environments | ✅ Phase 1 — outputs below |
| 2 | Unit test proving contract bytes identical with and without the extra | ✅ `test_contract_bytes_are_foundation_extra_independent` |
| 3 | Foundation recipe on a non-opted run and on an extra-less env both fail with their specific messages (tested) | ✅ two tests in `test_foundation_estimator.py` |
| 4 | Stub campaign brief `foundation_models: true` → child manifest flag + handoff note | ✅ Phase 2 — clean spawn (see note on guard) + forwarding/brief-block tests |
| 5 | Preflight failure cases for missing token / missing extra (tested, no paid calls) | ✅ Phase 2 — 4 preflight tests |
| 6 | Playoff replay of a foundation finalist gated correctly (fixture test) | ✅ Phase 2 — replay-gating tests |
| 7 | Build notes with every deviation logged; Phase 3 implemented or recorded blocked | ✅ Phase 3 implemented (Alex confirmed Modal setup done) |

---

## Phase 1 — static foundation declarations (the drift fix)

**Commit:** *(this phase's commit)*. Suite: **571 passed, 4 skipped**;
`--check` in sync in both `.venv` and system Python 3.13. **No protected file
touched.**

### What was built

- **`src/autoresearch/models/recipe/foundation_specs.py`** (new) — the single,
  **dependency-free** declaration of every foundation estimator. Imports only
  `importlib.util`, `sys`, `dataclasses` — no `tabpfn`/`tabpfn_client`, no
  `numpy`. Contents:
  - `FoundationEstimatorSpec` — a frozen metadata record: `name`, `objectives`,
    `encodings`, `default_encoding`, `supports_early_stopping`,
    `native_categorical`, `required_packages`, `caveat`, `description`.
  - `TABPFN_SPEC` — the tabpfn declaration (obj `{squared_error}`, enc
    `{ordinal, one_hot}`, caveat "requires `bootstrap-track
    --enable-foundation-models` + the `[foundation]` extra").
  - `FOUNDATION_ESTIMATOR_SPECS` / `FOUNDATION_ESTIMATOR_NAMES` — the collection
    and its name set.
  - `foundation_spec(name)`, `foundation_packages_available(spec)` — lookups a
    validator/generator uses without importing the heavy module.
- **`foundation.py`** — now *re-exports* the static declarations and builds its
  runtime `EstimatorSpec` (`_TABPFN_SPEC`) **from** `TABPFN_SPEC` (name,
  objectives, encodings, default_encoding, early-stop, native flags, description
  all sourced from the static record; only `fit=_fit_tabpfn` and `allowed_params`
  are added). This makes the legal obj×enc matrix in the contract *provably* the
  one the registered estimator enforces — they cannot drift
  (`test_runtime_spec_matches_static_declaration`). `tabpfn_available()` now
  routes through `foundation_packages_available(_TABPFN_STATIC)`. Removed the
  now-unused local `_importable`, `importlib`, and `sys` imports.
  Implementation registration (`register_foundation_estimators`) is unchanged —
  it still requires the package. *Metadata is static; execution stays gated.*
- **`scripts/generate_agent_contract.py`** — the estimator table is now built
  from the **always-registered builtins** (menu rows minus foundation names)
  **plus the static foundation specs** (always appended, annotated with their
  caveat). It no longer calls `enable_foundation_models()` and no longer reads
  foundation rows from the live menu, so the output is byte-identical regardless
  of which extras are importable or whether `AUTORESEARCH_FOUNDATION_MODELS` is
  set. Removed the dead `foundation_note` machinery.
- **`schema.py`** — recipe validation now gives a **specific** message when a
  *known* foundation estimator is named but unregistered
  (`_foundation_unavailable_message`), replacing the misleading generic "unknown
  estimator" both causes used to produce:
  - extra not importable → "…needs the optional [foundation] extra, which is not
    installed… `pip install -e '.[foundation]'`… enable it with
    `bootstrap-track --enable-foundation-models`."
  - extra importable but not registered → "…is not enabled for this run… start
    the run with `bootstrap-track --enable-foundation-models`."
  A name that is not a foundation estimator falls through to the original
  "unknown estimator" error unchanged.
- **`.venv` now has the `[foundation]` extra** (`tabpfn-client 0.3.3`, api client
  only — no torch/local weights), matching the operator's system Python so the
  gate environment matches. See "Environment note" below.
- **Contract regenerated once.** The `tabpfn` row now appears **by declaration**,
  identically everywhere: `- **tabpfn** — obj ['squared_error']; enc ['one_hot',
  'ordinal'] (requires \`bootstrap-track --enable-foundation-models\` + the
  \`[foundation]\` extra)`. Contract size 15,968 → **17,107 bytes** (ceiling
  17,500).

### Tests added

- `tests/test_agent_contract.py::test_contract_bytes_are_foundation_extra_independent`
  — renders with tabpfn *unregistered* and *registered* and asserts byte-equal
  output (criterion 2). This is the regression guard for the drift itself.
- `tests/test_foundation_estimator.py`:
  - `test_runtime_spec_matches_static_declaration` — the registered estimator's
    obj/enc matrix equals the static declaration's.
  - `test_non_opted_run_gets_specific_foundation_message` — extra present, run
    not opted in → the opt-in message, not "unknown estimator".
  - `test_extra_missing_run_gets_specific_foundation_message` — extra absent →
    the extra message, not "unknown estimator".
  Both message tests monkeypatch `schema.foundation_packages_available` and
  control the registry directly, so they are independent of the ambient
  environment (they pass whether or not `.venv` has the extra).

### Environment note — the superseded stash

`orchestration_build_notes.md` "Pre-flight" recorded a `stash@{0}` ("tabpfn
contract regen (needs [foundation] extra installed to pass --check)") holding
contract output an extra-having environment produced under the **old**
menu-driven generator. It is **now superseded** and should be dropped: its
tabpfn row is the un-annotated menu form `- **tabpfn** — obj ['squared_error'];
enc ['one_hot', 'ordinal']` placed alphabetically between lightgbm and
tweedie_glm, and it bumped `SIZE_CEILING_BYTES` to 16,100. Phase 1's contract
instead emits the caveat-annotated row grouped after the builtins and keeps the
17,500 ceiling (already set by the orchestration branch). Regenerating from
either environment now yields Phase 1's bytes, so the stash's contents can no
longer be produced by the generator and carry no information to recover.

### Exact commands and output

```bash
# Baseline (Orchestrator head, before any change): 569 passed, 2 skipped; --check in sync.

# After Phase 1, in .venv (now has tabpfn-client):
.venv/bin/python -m pytest -q
# 571 passed, 4 skipped in 52.12s
.venv/bin/python scripts/generate_agent_contract.py --check
# AGENT.md (and harness mirrors) in sync.

# In the operator's system Python 3.13 (has tabpfn_client):
python3 scripts/generate_agent_contract.py --check
# AGENT.md (and harness mirrors) in sync.   exit 0
```

The two skips that appeared (569→571 passed, 2→4 skipped, net after +4 new tests)
are the two pre-existing gating tests (`test_not_registered_by_default`,
`test_enable_is_noop_without_package`) that skip when the extra is installed —
which it now is in `.venv`. Their coverage of the extra-absent path is preserved
in CI and any extra-less checkout; the extra-present paths are covered by the
stub fixtures and the new message tests.

### Deviations from the build prompt (Phase 1)

1. **Two modules, not one.** The prompt says the static declaration lives "in a
   module `foundation.py` re-exports from" — i.e. it explicitly anticipates a
   separate source. That module is `foundation_specs.py`; `foundation.py`
   re-exports `FOUNDATION_ESTIMATOR_SPECS`/`FOUNDATION_ESTIMATOR_NAMES`/
   `TABPFN_SPEC`. Keeping the metadata in a dependency-free file is what lets
   `schema.py` and the generator read it without importing `numpy` or risking
   the heavy module's lazy optional-dep paths.
2. **Foundation rows grouped after the builtins**, not interleaved
   alphabetically. Deterministic and byte-stable either way; grouping keeps the
   caveat-bearing rows together and visually separated from the always-available
   builtins. (The superseded stash placed tabpfn alphabetically; the change is
   cosmetic and intentional.)
3. **`schema` distinguishes the two failure causes by package availability**,
   the only deterministic signal `validate_recipe` has without run-manifest
   context: extra importable but unregistered ⇒ "not opted in"; extra not
   importable ⇒ "install the extra". This maps cleanly onto the prompt's cases
   (a) and (b). A run that is both un-opted *and* extra-less gets the extra
   message (installing the extra is the blocking prerequisite either way).

### Deliberately left undone (Phase 1)

- No change to `register_foundation_estimators` gating, the bootstrap
  `--enable-foundation-models` flag, or any runtime fit/predict path — Phase 1 is
  *metadata* only.
- The `--permission-mode acceptEdits` `TODO(pin)` inherited from the
  orchestration build is untouched (needs a real spawn; out of scope).

---

## Phase 2 — orchestrated TabPFN (API backend)

**Commit:** *(this phase's commit)*. Suite: **589 passed, 4 skipped**;
`--check` in sync. **No protected file touched.** All new code is additive.

### What was built

- **`brief.py`** — new optional brief field **`foundation_models: bool`**
  (default `false`). `validate_brief` accepts it, rejects a non-bool, and it
  round-trips through `to_dict`/`from_dict`. When true, `render_brief_block` adds
  a **"Foundation estimators enabled"** line to the child's handoff block naming
  TabPFN's regime (sparse / severity-shaped, weak on dense sets) and the finite
  Prior Labs credit budget, so the sub-agent reads the caution before proposing.
- **`spawner.py`** —
  - `_bootstrap_child_run` forwards the flag as
    `bootstrap_track(..., enable_foundation_models=brief.foundation_models)`,
    which sets the child run manifest's `foundation_models` flag and registers
    the estimators. The child's own `cli.main` re-applies the gate from that
    manifest flag on every command.
  - `child_environment` now **strips `AUTORESEARCH_FOUNDATION_MODELS`** from the
    child. Without this, a non-opted child would inherit the orchestrator's
    in-process env flag (set when the orchestrator bootstrapped an *earlier*
    foundation child) and silently self-register foundation estimators. Per-run
    enablement is driven solely by the child's manifest flag now.
    `TABPFN_TOKEN` is deliberately **not** stripped — an opted-in child inherits
    the exported token to reach the api backend (build-prompt instruction).
  - `spawn()` and the `respawn --continue-run` path call the new
    `preflight_foundation_models()` **only when the brief opted in** (skipped on
    `--dry-run`).
- **`backends.py`** — new `preflight_foundation_models()` sibling of
  `preflight_backend`. Fails the spawn fast, each message naming its fix, when
  (a) the `[foundation]` extra is not importable, or (b)
  `AUTORESEARCH_TABPFN_BACKEND` resolves to `api` while `TABPFN_TOKEN` is unset.
  A `local`-backend brief needs no token. Cannot verify token *validity* without
  a paid call; an invalid token still surfaces via the report's `crashed` flag.
- **`playoff.py`** — foundation-aware replay:
  - `_any_finalist_needs_foundation(finalists)` reads each finalist's **source
    run manifest** `foundation_models` flag; `_create_consolidation_run` passes
    `enable_foundation_models=` that result to `bootstrap_track`, so the
    consolidation run enables foundation models iff it will replay a foundation
    finalist.
  - `_ensure_foundation_support(model, source, destination)` (called inside
    `replay_experiment`) detects a foundation estimator in the portable recipe
    (`_recipe_foundation_estimators`, covering `direct` and `frequency_severity`
    stages). If the extra is missing it raises a **clear message** naming the
    fix rather than letting the fit fail deep inside `run_experiment` as a
    misleading "unknown estimator"; if present it registers the estimators so
    the fresh fit can interpret the recipe. Non-foundation recipes never even
    probe availability.
- **Docs (hand-authored, per prior deviation 24 — not generator-owned):**
  - `ORCHESTRATOR.md` §3 documents the `foundation_models` field, the spawn
    preflight, and the credit caution; §5 adds a "when to reach for foundation
    models" paragraph pinning TabPFN's regime to the 2026-07-05 cross-run
    analysis (sparse/severity strong, dense weak).
  - `docs/RUN_ORCHESTRATED.md` Prerequisites gains a foundation-only bullet
    (extra + token + finite credit budget + usage dashboard); "Writing a brief"
    notes the field and regime.

### Tests added (`tests/test_foundation_orchestration.py`, 18 tests)

Brief field (default, round-trip, non-bool rejection, block-renders-note,
block-omits-when-off); spawner forwarding (parametrized true/false, asserts the
`enable_foundation_models` kwarg reaches `bootstrap_track`); child-env hygiene
(strips the flag, keeps the token); preflight (missing extra, api-without-token,
api-with-token, local-no-token, and that `spawn()` invokes the preflight *only*
when opted in); replay gating (estimator extraction incl. freq-sev, clear message
when extra missing with no "unknown estimator", registers when available, no-op
for a GBM recipe, and `_any_finalist_needs_foundation` reading source manifests).
Also updated the existing playoff fixture stub of `_create_consolidation_run` to
accept the new keyword.

### Phase 2 milestone (stub, no paid call)

Per build-prompt rule (no real sub-agents / no paid TabPFN calls), the milestone
uses the zero-cost `stub` backend with a `foundation_models: true` brief. In
`.venv` (which now has the extra) the foundation preflight passes — extra
importable, default `local` backend needs no token — and the spawn runs the full
bootstrap → handoff → cycle → report path:

```bash
autoresearch orchestrate new --dataset french_motor --total-cycles 2 \
  --model-provider anthropic --model-name claude-opus-4-8
# → 20260710T191217Z
AUTORESEARCH_SKIP_PYTEST_GATE=1 autoresearch orchestrate spawn \
  --orchestration-id 20260710T191217Z --brief p2_foundation_brief.json \
  --backend stub --wait
# → {"status": "completed", "clean_exit": true, "exit_code": 0,
#    "delegation_id": "d01", "run_id": "20260710T191225Z", ...}
```

The spawn completing `clean_exit: true` is itself evidence the child's handoff
carried the Orchestration-brief block: `_export_handoff_with_brief` **raises**
unless `## Orchestration brief` is present, and it runs on every spawn. The
foundation line inside that block is covered by
`test_brief_block_states_foundation_enabled`, and the manifest flag by
`test_bootstrap_child_run_forwards_foundation_flag`.

**Reviewer note — direct child-run inspection is guard-blocked.** As in the
orchestration build, an *unbound* session cannot read
`artifacts/tracks/claude/runs/20260710T191225Z/run_manifest.json` or the child
handoff, and cannot read `artifacts/orchestrations/20260710T191217Z/` either.
The sanctioned path is analyst/orchestrator scope at `SessionStart`; I did not
self-elevate by writing a `.scope` file (the harness's auto-mode classifier
correctly denied that as weakening an access control). Criterion 4's manifest +
handoff facts are therefore evidenced by the clean spawn's internal assertion
plus the two unit tests above, not by a direct read from this session.

### Deviations from the build prompt (Phase 2)

4. **Foundation preflight is a sibling function, not an extension of
   `preflight_backend`.** The build prompt allows either ("extend … or a sibling
   called when the brief opts in"). `preflight_backend(backend)` takes only a
   `Backend`; foundation enablement is a *brief* property, so a sibling called
   conditionally on `brief.foundation_models` keeps each preflight's inputs
   honest and lets the backend preflight stay unconditional.
5. **`child_environment` strips `AUTORESEARCH_FOUNDATION_MODELS`.** Not named in
   the build prompt, but required for correctness: the orchestrator process sets
   that env var in-process when it bootstraps a foundation child
   (`apply_foundation_models_gate`), and `child_environment` copies
   `os.environ`, so a *later* non-opted child would inherit it and self-register
   foundation estimators. The manifest flag is the single source of per-run
   truth; the env var must not leak across children. (`TABPFN_TOKEN` is
   correctly *not* stripped, per the build prompt.)
6. **Replay registers foundation via `enable_foundation_models()` directly**
   when a foundation recipe is replayed and the extra is present, in addition to
   the consolidation run's manifest flag. The manifest flag alone drives later
   CLI processes (interactive decisions), but the replay's own `run_experiment`
   executes in the current process, which may not have re-read the manifest —
   so the estimator is registered explicitly there. Both together mean every
   process that touches a foundation replay can interpret the recipe.

## Phase 3 — TabFM via Modal (stop-and-ask resolved: PROCEED)

**Stop-and-ask gate.** `tabfm_integration_plan.md` §2 records two blocking
operator setup steps: (1) Modal account + `modal setup` (writes `~/.modal.toml`),
and (2) HF licence accepted + `modal secret create huggingface`. Local signals
were partial — `~/.modal.toml` present (dated the Jul 5 spike), `.venv-api`
present, but `modal` not importable in `.venv`/system Python and the HF secret
unverifiable without a paid Modal call. **I asked Alex; he confirmed the setup is
done and to proceed.** Per build-prompt rule 5 the build still made **no real
Modal call** — the transport boundary is fixtured in every test.

**Commit:** *(this phase's commit)*. Suite: **600 passed, 4 skipped**; `--check`
byte-identical in both `.venv` and system Python 3.13. **No protected file
touched.**

### What was built

- **`foundation_specs.py`** — added `TABFM_SPEC` to `FOUNDATION_ESTIMATOR_SPECS`
  by the same static-declaration pattern as TabPFN: obj `{squared_error}`, enc
  `{ordinal, one_hot}`, `required_packages=("modal", "tabfm")`, a caveat naming
  the `[foundation-modal]` extra + Modal GPU + non-commercial weights, and a
  description carrying the non-commercial licence. The estimator is therefore
  **declared everywhere** (contract, validation messages) regardless of whether
  `modal` is installed; only its *execution* is gated.
- **`foundation.py`** — the TabFM estimator:
  - `_TABFM_SPEC` built from `TABFM_STATIC` (obj/enc matrix cannot drift from the
    declaration), `fit=_fit_tabfm`, `allowed_params` = `backend`,
    `max_context_rows`, `subsample_strategy`, `random_state`,
    `predict_batch_size`, `n_estimators`, `gpu` (modal-only), `device`
    (local-only).
  - `_fit_tabfm` reuses the shared `subsample_context` (exposure-weighted context
    cap) and returns a `_TabFMRemoteModel` whose `predict` ships the context +
    score frame to the backend and clips to ≥ 0. Zero-shot means fit+predict
    happen together remotely, so the context is re-sent each `predict` (payload
    is small at the capped context).
  - **Transport boundary** `_tabfm_fit_predict(backend, …)` → `modal` calls the
    deployed app (`modal.Cls.from_name("tabfm-inference", "TabFMRunner")` →
    `.fit_predict.remote(...)`, matching `scripts/tabfm_modal_app.py`); `local`
    runs in-process `tabfm[pytorch]` (LightGBM-first libomp guard, `_select_device`).
    **This function is what every test fixtures — no Modal client is ever
    constructed in the suite.**
  - `_authenticate_modal()` is the estimator-level **preflight** the build asks
    for: `modal` importable **and** a token configured (`~/.modal.toml` or
    `MODAL_TOKEN_ID`/`_SECRET`), each failure naming its fix. No network call —
    the TabFM analogue of TabPFN's `_authenticate_api`.
  - Defaults from the spike (`tabfm_integration_plan.md` §6): 20k context, 4
    ensemble members (library default 32 times out), L4 GPU, 20k predict batch.
  - `tabfm_available()` + `register_foundation_estimators` now also registers
    `tabfm` when `modal`/`tabfm` is importable.
- **Compute budget** — no special handling needed and none added: a `modal`
  `.remote()` call **blocks locally** until the GPU returns, so the remote
  wall-clock is charged to the experiment's compute budget automatically (the
  clock does not stop because the GPU is remote). Documented in the OPERATING
  MANUAL and the transport docstring.
- **`pyproject.toml`** — new `foundation-modal = ["modal"]` extra (the client
  only; torch/weights stay remote, so it is safe next to `tabpfn-client`).
- **`backends.py`** — `preflight_foundation_models` broadened: the extra check
  now passes if **TabPFN _or_ TabFM** is importable (a TabFM-only environment is
  no longer wrongly told to install the TabPFN extra). TabFM's deeper readiness
  (deployed app + `~/.modal.toml`) surfaces at fit time via `_authenticate_modal`
  — a Modal deploy is not cheaply verifiable at spawn.
- **Contract** — regenerated once; the `tabfm` row now appears **by
  declaration**, byte-identical everywhere (15,968 → 17,296 bytes, ceiling
  17,500).
- **Docs** — `docs/OPERATING_MANUAL.md` gains a TabFM subsection (Modal setup,
  recipe usage, params, the remote-clock budget note, the regime, and a
  **NON-COMMERCIAL licence** warning); the section heading now reads
  "(TabPFN, TabFM)".

### Tests added (`tests/test_foundation_estimator.py`, 10 TabFM tests)

Registration gating (not registered without `modal`/`tabfm`); runtime spec
matches the static declaration; unknown-backend rejection; `_authenticate_modal`
failure on missing client and on missing token (fixtured, no network);
`_TabFMRemoteModel` clip; registered-and-validates and invalid-objective through
the recipe validator; end-to-end `dispatch_model` through a **stub `modal`
module + fixtured transport** asserting the context was capped and shipped with
`gpu=L4`, and notes carry `estimator=tabfm`/`backend=modal`; and that a `modal`
recipe setting the local-only `device` param is rejected.

### Deviations from the build prompt (Phase 3)

7. **`_TabFMRemoteModel` is a bespoke wrapper, not `_BatchedRegressor`.** TabPFN
   uses `_BatchedRegressor` to batch scoring locally; TabFM batches *remotely*
   (the `predict_batch` arg to the Modal function), and re-sending the context
   per local chunk would be wasteful. The wrapper ships the whole score frame in
   one `.remote()` call and applies the same `_densify` + clip-to-≥0 contract.
6. **Spawn preflight broadened to "TabPFN or TabFM".** See `backends.py` above —
   a small correctness fix so a TabFM-only orchestrator is not told to install
   the wrong extra. Existing Phase 2 preflight tests updated accordingly (they
   now stub both availability checks).

### Deliberately left undone / deferred to Alex (Phase 3)

- **No real Modal call** was made (build-prompt rule 5): the deployed-app call
  path (`modal.Cls.from_name(...).fit_predict.remote(...)`), the exact
  `with_options(gpu=...)` behaviour on a *deployed* app, and the local
  `tabfm[pytorch]` load are all validated by Alex's post-review campaign. The
  spike (`scripts/tabfm_modal_app.py`) proves the remote science; this phase
  wires it into the recipe framework behind the fixtured boundary.
- **`modal` is not installed** in `.venv` (the extra pulls a large client and
  the build makes no Modal calls), so `tabfm` is *unregistered* in the gate
  environment — exactly the extra-less case the static declaration exists to make
  reproducible. `test_tabfm_not_registered_without_modal_or_tabfm` asserts it.
- **Fan-out scoring** (the spike's recommended Phase-2 speed path — split the
  score frame across N Modal containers) is **not** implemented: single-container
  `.remote()` is the correctness-first integration. Wiring `.map()`/`.starmap()`
  is a follow-up once Alex's real campaign confirms the accuracy is worth it.
  Documented in the OPERATING MANUAL's budget note (scoring is slow single-container).
