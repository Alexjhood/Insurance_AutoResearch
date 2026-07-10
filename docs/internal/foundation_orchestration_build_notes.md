# Foundation-orchestration build notes

**Branch:** `foundation-orchestration` (from the `Orchestrator` head,
`23b7b9c`).
**Status:** Phase 1 complete. Phases 2–3 pending.
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
| 4 | Stub campaign brief `foundation_models: true` → child manifest flag + handoff note | ⏳ Phase 2 |
| 5 | Preflight failure cases for missing token / missing extra (tested, no paid calls) | ⏳ Phase 2 |
| 6 | Playoff replay of a foundation finalist gated correctly (fixture test) | ⏳ Phase 2 |
| 7 | Build notes with every deviation logged; Phase 3 implemented or recorded blocked | ⏳ Phase 3 gate |

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

⏳ Not started.

## Phase 3 — TabFM via Modal (stop-and-ask first)

⏳ Not started. Gated on verifying with Alex that the Modal account/secret setup
(the two user setup steps `tabfm_integration_plan.md` records as blocking) is
done. If not, Phase 3 is recorded here as blocked and the build stops after
Phase 2.
