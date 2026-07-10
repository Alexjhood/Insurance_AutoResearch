# Orchestration Build Notes

**Branch:** `orchestration` (from `MultiDataset`)
**Status:** Phase 1 complete. Full suite green (490 passed, 2 skipped);
`generate_agent_contract.py --check` passes. Phases 2–5 not started.

This is the review entry point for the implementation of
`docs/internal/orchestration_design.md`, per the execution constraints in
`docs/internal/orchestration_build_prompt.md`. It is factual: what was built per
phase, every deviation from the design with rationale, the open-choice decisions,
the exact smoke commands with key output, and what was deliberately left undone.

---

## Pre-flight: the dirty working tree

`git status` at the start showed four uncommitted files (`AGENT.md`, `AGENTS.md`,
`CLAUDE.md`, `tests/test_agent_contract.py`) adding a `tabpfn` estimator line and
bumping the contract size ceiling to 16,100. Per ground rule 1 I stopped and
asked rather than absorbing them. Established first:

| State | `generate_agent_contract.py --check` | `tests/test_agent_contract.py` |
|---|---|---|
| Committed `MultiDataset` HEAD | passes | 5 passed |
| With those four files | **fails** | **2 failed**, 3 passed |

They are generator output from an environment where the `[foundation]` extra is
installed; `tabpfn`/`tabpfn_client` are **not** importable in this repo's `.venv`,
so the generator here does not emit that line. Alex chose **stash and build from
clean HEAD**. They are preserved in
`stash@{0}` ("tabpfn contract regen (needs [foundation] extra installed to pass
--check)"), untouched.

> **Latent issue worth surfacing, unrelated to orchestration:** the generated
> contract's content depends on which optional extras happen to be installed in
> the generating environment (`generate_agent_contract.py` calls
> `enable_foundation_models()` then `recipe_menu()`). That makes `--check`
> non-reproducible across machines and is the root cause of those files being
> dirty. Not fixed here — out of scope.

Baseline before any work: **433 passed, 2 skipped**.

---

## Acceptance criteria — status

| # | Criterion | Status |
|---|---|---|
| 1 | Full pytest green; `generate_agent_contract.py --check` passes | ✅ 490 passed / 2 skipped; check in sync |
| 2 | Stub end-to-end campaign: `new` → 2 × `spawn --wait` → `collect`; correct cycle counts, champion facts from child registry, ≥1 provoked distress flag | ✅ (`no_finish_delegation` provoked via `stub-no-finish`) |
| 3 | Parallel: 2 detached stub delegations, `status` shows both, manifest lock prevents bootstrap races | ⬜ Phase 2 (lock built + unit-tested now; `--no-wait` wired but `status`/`monitor.py` are Phase 2) |
| 4 | Guard: orchestrator allow/deny cases; existing guard tests unchanged | ⬜ Phase 2 |
| 5 | Playoff fixture passes; consolidation promotion fires the holdout-eval hook | ⬜ Phase 4 |
| 6 | `list-backends` prints registry + metadata; `backend-stats` aggregates a fixture manifest | 🟡 `list-backends` ✅ (scorecard column renders "—"); `backend-stats` is Phase 5 |
| 7 | Single-agent smoke test proves no Orchestration-brief block and no behaviour change | ✅ `test_single_agent_handoff_has_no_orchestration_brief` |
| 8 | Build notes exist | ✅ this file |

---

## Phase-by-phase summary

### Phase 1 — orchestration core, sequential

New package `src/autoresearch/orchestration/`, all of it a *caller* of existing
public entry points. **No protected file was touched.**

- **`manifest.py`** — frozen `Orchestration` / `Delegation` / `Consolidation`
  records with fail-loud `__post_init__` validation (timestamp-shaped
  orchestration ids, `dNN` delegation ids, positive budgets, closed status sets,
  no duplicate delegation ids). JSON round-trip via `to_dict`/`from_dict`.
  `cycles_committed`/`cycles_remaining` are derived, not stored twice.
  `manifest_lock()` is an `O_CREAT|O_EXCL` lock with a 120 s staleness break, so
  a crashed spawner cannot wedge a campaign. `write_run_backpointer` /
  `read_run_backpointer` / `find_orchestration_for_run` implement the child-run
  back-pointer that keys everything orchestration-aware.
- **`brief.py`** — the delegation API. `Brief` + `SeedChampion` frozen records,
  `validate_brief` (rejects unknown fields, non-positive/bool budgets, unqualified
  `from_run`), `load_brief`, and `render_brief_block` producing the handoff block.
- **`backends.py` + `configs/orchestration/backends.toml`** — the curated
  registry (design §4.7 Layer 1). Validation refuses a `track` the guard would
  not bind, a `prompt_via='argv'` template without `{prompt}`, a
  `prompt_via='stdin'` template *with* one, unknown placeholders, and
  `{max_turns}` without a configured value. `get_backend` refuses `deprecated`
  entries. `format_backend_table` renders the registry + scorecard slot.
- **`spawner.py`** — `plan_spawn` (pure) → `SpawnPlan.render()` is the `--dry-run`
  review surface (exact argv + full `AUTORESEARCH_*` child env + prompt).
  `compose_prompt` (pure), `child_environment` (pure), `compute_timeout_minutes`
  (pure). `spawn()` bootstraps the child under the manifest lock, archives the
  brief, records the delegation, exports the handoff, launches detached with
  `start_new_session=True`, tees stdout/stderr to `logs/<did>.stdout.log`, and on
  `--wait` finalises + collects the report. `finish_delegation()` is the
  run-scoped child-side call.
- **`report.py`** — `assess_distress` is a **pure** function over plain values
  (all seven flags), unit-tested exhaustively. `build_report` reads the child's
  registry and scores the champion's predictions artifact through
  `full_metric_panel`, the same privileged read `tracks.py` uses. The sub-agent's
  `agent_summary` is stored verbatim and never feeds a computed number.
- **`cli.py`** — `orchestrate` command group (`new`, `spawn`, `collect`,
  `list-backends`, `finish-delegation`) following the existing `telemetry`/`memory`
  nested-subparser pattern. Additive: `COMMANDS` gains one key.
- **`controller/handoff.py`** — `_render_orchestration_brief(config)` returns `[]`
  unless the run manifest carries an `orchestration_id`. Spliced into
  `current_state_lines` so it appears in both full and `--delta` handoffs (the
  brief is binding on every mid-run refresh, so it must survive `--delta`).
- **`scripts/generate_agent_contract.py`** — new "Orchestrated mode" subsection
  (design §4.4). Hand-edited into the *generator*, never the outputs.
- **`scripts/stub_subagent.py`** + two `backends.toml` entries (`stub`,
  `stub-no-finish`) — the permanent zero-LLM smoke test (build-prompt rule 5).
- **`tests/test_orchestration.py`** — 57 tests.

---

## Two real bugs the stub backend caught

Worth recording, because they are the argument for the stub existing.

1. **The child's handoff carried no Orchestration brief.** `spawn()` exported the
   child's context bundle inside `_bootstrap_child_run`, i.e. *before* the brief
   file was archived and *before* the delegation was appended to the manifest.
   The renderer therefore found nothing and — by design — degraded silently. The
   unit test passed because it wrote the brief directly. Only the live stub run
   revealed it, via its own `WARNING — handoff carries no Orchestration brief`
   check.

   **Fix:** `_bootstrap_child_run` no longer exports; it returns the child config.
   `spawn()` archives the brief → saves the delegation → *then* calls the new
   `_export_handoff_with_brief`, which **asserts** the block is present and raises
   otherwise. Graceful degradation is right for a mid-run refresh; refusing to
   launch a paid sub-agent that cannot read its own direction is right for a
   spawn. Regression tests:
   `test_export_handoff_with_brief_refuses_a_child_that_cannot_see_its_brief`,
   `test_export_handoff_with_brief_passes_once_brief_and_delegation_exist`.

2. **The archived brief could not be re-validated.** `_store_brief` stamps
   `_source_path` onto its copy for audit; `validate_brief` rejects unknown
   fields, so re-loading the archived brief raised and the block vanished — a
   second, independent cause of the same symptom. **Fix:** underscore-prefixed
   keys are treated as framework metadata (a typo'd field name never starts with
   `_`, so the typo guard is intact). Regression tests:
   `test_validate_brief_allows_underscore_metadata`,
   `test_stored_brief_round_trips_through_validation`.

---

## Deviations from the design (with rationale)

1. **Extra module `brief.py`.** The design's §4 file list names six modules;
   brief validation would otherwise live in `spawner.py`. A brief is a record on
   disk consumed by the spawner, the handoff renderer, and (later) the playoff, so
   it earns its own module. Build-prompt rule: module layout is mine to decide,
   logged here.
2. **`seed_champion` accepted by the brief schema but refused at spawn.** Design
   §4.3 defines it and §4.4 step 2 applies it, but §4.3 itself says it is "the
   same replay mechanic the playoff uses (4.6)", and the build plan lands replay
   in Phase 4 (`respawn --seed-champion` is Phase 2). Phase 1 validates it, then
   raises `NotImplementedError` **before** creating anything, rather than
   half-bootstrapping a child. Test:
   `test_spawn_rejects_seed_champion_before_creating_anything`.
3. **`Delegation` carries fields the design's JSON sketch omits:** `exit_code`,
   `prompt_path`, `log_path`, `command`, `timeout_minutes`, `agent_summary`. The
   design's own §4.5/§9 require every one of them (report generation, the
   `no_finish_delegation` predicate, timeout enforcement, "log the exact command
   for every delegation"). Additive to the sketch, not a change of meaning.
4. **`Orchestration` carries `model_provider`/`model_name`.** Required by §7
   ("the orchestrator session itself is attributed via `orchestrate new
   --model-provider/--model-name` … stored in the manifest"); absent from the §4.1
   JSON sketch.
5. **`cycles_committed` is derived, not a stored field.** The §4.1 sketch shows it
   as a manifest key. It is emitted in `to_dict()` (so the on-disk shape matches)
   but computed as `sum(d.cycle_budget)` rather than maintained independently — two
   sources of truth for the same number is how budgets drift. A `failed`
   delegation still counts as committed: the campaign paid for the attempt.
6. **`build_report` derives the run dir from `config.artifacts_dir`** rather than
   `Delegation.run_dir()` (which rebuilds it from `PROJECT_ROOT`). Same path;
   makes report generation testable against a fixture registry.
7. **Contract size ceiling raised 16,000 → 17,500 bytes.** The "Orchestrated mode"
   subsection took AGENT.md from 15,968 → 16,795 bytes; there were 32 bytes of
   headroom. Note this collides with the stashed tabpfn change, which bumped the
   same constant to 16,100 — whoever unstashes it should keep 17,500.
8. **`stub` backends use `track = "claude"`.** The guard's `ALLOWED_RESEARCH_TRACKS`
   is `{claude, codex, opencode}`; a `stub` track would be refused. Stub runs
   therefore land in the real `claude` track (under their own run ids, isolated as
   any run is).

---

## Open-choice decisions

- **`no_finish_delegation` predicate** fires on a missing *or whitespace-only*
  summary, regardless of exit status — a child that exits 0 without reporting is
  exactly the case the flag exists for.
- **`budget_overrun`** fires on `status == "timed_out"` **or** `cycles_used >
  cycle_budget`. `all_rejected` is suppressed when `cycles_used == 0` (that
  delegation failed some other way; don't double-flag).
- **`champion_is_baseline`** is detected via `model_family == "global_mean"`, the
  family `configs/experiments/global_mean.toml` declares.
- **`calibration_anomaly`** threshold: `|predicted/actual − 1| > 0.10`.
- **Timeout** integrates the *growing* per-cycle budget (`10 + 5×(N//5)` minutes)
  over the budgeted cycles rather than assuming cycle 1's allowance for all of
  them, then applies the design's safety factor 2. `K=1 → 20 min`, `K=5 → 100`,
  `K=6 → 130`.
- **Brief block placement:** immediately *above* the Active dataset block, in
  `current_state_lines`, so it survives `--delta`.
- **`child_environment` strips `AUTORESEARCH_ORCHESTRATION_ID` and any inherited
  `AUTORESEARCH_MEMORY_ACCESS`** from the child, or a child spawned by an
  orchestrator session would inherit the orchestrator's scope.

---

## Unpinned CLI flags (build-prompt Phase 1 note)

- **`claude` is not installed on this machine** (`which claude` → not found; no
  npm global). **Every flag in both `claude-*` backend entries is unverified.**
  `backends.toml` carries a `TODO(pin)` block naming them: `-p`,
  `--output-format`, `--verbose`, `--permission-mode`, `--max-turns`, and — most
  importantly — **no thinking-effort flag is pinned at all**, so
  `claude-sonnet-low` and `claude-sonnet-medium` currently differ *only* in
  `model_name` attribution, **not in actual effort**. Both are `status = "trial"`.
  Verify with `orchestrate spawn --dry-run` before any real-model campaign.
- **`codex` is installed** (`/opt/homebrew/bin/codex`). Verified from
  `codex exec --help`: `-m/--model`, `-s/--sandbox {read-only,workspace-write,
  danger-full-access}`, `--json`, `-C/--cd`, `--skip-git-repo-check`,
  `-o/--output-last-message`, prompt via argv or stdin. **There is no
  `--reasoning-effort` and no `--max-turns` on `codex exec`**; effort must go
  through the generic `-c model_reasoning_effort=…` config override, whose key
  name I could not verify locally. Codex entries land in Phase 3.

---

## Exact commands run (Phase 1 milestone)

The design's Phase 1 milestone is "a real sequential 2-delegation campaign on
Porto with claude-haiku sub-agents". Per build-prompt rule 5 / build-order note,
the **stub backend is substituted** and the real-model campaign is deferred to
Alex's post-review validation. `french_motor` was used instead of Porto: it is
already prepared, and the stub's science is meaningless by construction, so the
dataset only has to exist. `AUTORESEARCH_SKIP_PYTEST_GATE=1` avoids re-running
the suite inside each `bootstrap_track` (the suite was run immediately before).

```bash
autoresearch orchestrate list-backends
autoresearch orchestrate new --dataset french_motor --total-cycles 6 \
  --model-provider anthropic --model-name claude-opus-4-8      # → 20260710T094215Z
autoresearch orchestrate spawn --brief brief_d01.json --backend stub           --wait
autoresearch orchestrate spawn --brief brief_d02.json --backend stub-no-finish --wait
autoresearch orchestrate collect
```

`spawn --dry-run` (the spawn-correctness review surface):

```
Orchestration : 20260710T093515Z
Delegation    : d01
Backend       : stub (tool=stub, tier=cheap, status=trial)
Child run     : claude/<pending-bootstrap>
Cycle budget  : 2
Timeout       : 40 min
Prompt via    : stdin

Command argv:
  [0] python3
  [1] scripts/stub_subagent.py

Child environment (AUTORESEARCH_* overrides):
  AUTORESEARCH_RUN_ID=<pending-bootstrap>
  AUTORESEARCH_SCOPE=research
  AUTORESEARCH_TRACK=claude
```

Final campaign `20260710T094215Z` (after the two bug fixes above; d01's log
contains **zero** "no Orchestration brief" warnings):

```
d01: backend=stub            cycles=2/2 champion=global_mean gini=0.0000
     distress=['all_rejected', 'champion_is_baseline']
     summary=Scripted stub delegation. Every cycle proposed a c...
d02: backend=stub-no-finish  cycles=1/1 champion=global_mean gini=0.0000
     distress=['no_finish_delegation', 'all_rejected', 'champion_is_baseline']
     summary=<none>

manifest: total=6 committed=3
```

Both children ran the real workflow: `show-latest-handoff` → `start-session` →
`run-session-cycles 1` → `record-decision` (× K) → `orchestrate
finish-delegation`. The reports' `champion`, `cycles`, `experiments[].decision`
and `reason_code` are all reads of the child registry, never of the stub's claims.

**Note for the reviewer:** verifying the rendered brief block by reading the
child's handoff directly is *blocked by the run-scope guard* from an unbound
session — which is exactly the gap Phase 2's `orchestrator` scope closes. Evidence
that the block renders comes from the stub's own warning check (0 hits) and from
`test_orchestrated_handoff_carries_the_brief_block`.

---

## Deliberately left undone

- **Real-model campaigns (Phase 1 and Phase 3 milestones).** No `claude -p` /
  `codex exec` sub-agent was ever launched (build-prompt rule 5). Deferred to
  Alex's post-review validation.
- **Claude effort flags** — see "Unpinned CLI flags". `claude` is not installed
  here; guessing silently was explicitly forbidden.
- **`seed_champion`** — deviation 2 above.
- **Phases 2–5**: orchestrator guard scope, `monitor.py`/`status`/`kill`/timeout
  enforcement, `respawn`, the Codex backend, `playoff.py`, `ORCHESTRATOR.md`,
  `docs/RUN_ORCHESTRATED.md`, `orchestrate note`/`report`/`backend-stats`.
- **`--no-wait`** is wired through `spawn()` and records `pid`/`status=running`,
  but nothing yet observes the process (that is `monitor.py`, Phase 2). Do not
  rely on it before Phase 2.
- **Timeout is computed and stored** on the delegation, **not enforced**. Design
  §4.5 puts enforcement in `monitor.py` (Phase 2) and says the monitor "never
  auto-kills in v1".
