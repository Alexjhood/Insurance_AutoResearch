# Orchestration Build Notes

**Branch:** `orchestration` (from `MultiDataset`)
**Status:** Phases 1–4 complete. Full suite green (final Phase 4 count recorded
below); `generate_agent_contract.py --check` passes. Phase 5 not started.

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
| 1 | Full pytest green; `generate_agent_contract.py --check` passes | ✅ 539 passed / 2 skipped; check in sync |
| 2 | Stub end-to-end campaign: `new` → 2 × `spawn --wait` → `collect`; correct cycle counts, champion facts from child registry, ≥1 provoked distress flag | ✅ (`no_finish_delegation` provoked via `stub-no-finish`) |
| 3 | Parallel: 2 detached stub delegations, `status` shows both, manifest lock prevents bootstrap races | ✅ 3 detached stubs live concurrently; lock unit test + concurrent smoke |
| 4 | Guard: orchestrator allow/deny cases; existing guard tests unchanged | ✅ all five required cases; pre-existing tests unchanged |
| 5 | Playoff fixture passes; consolidation promotion fires the holdout-eval hook | ✅ Three eligible finalists + one baseline exclusion; protected hook stub called exactly once for the clean final promotion |
| 6 | `list-backends` prints registry + metadata; `backend-stats` aggregates a fixture manifest | 🟡 `list-backends` ✅ (scorecard column renders "—"); `backend-stats` is Phase 5 |
| 7 | Single-agent smoke test proves no Orchestration-brief block and no behaviour change | ✅ `test_single_agent_handoff_has_no_orchestration_brief` |
| 8 | Build notes exist | ✅ this file |
| 9 | Phase 3 mixed Claude/Codex campaign using zero-cost substitutes | ✅ `stub` on `claude` + `stub-codex` on `codex`; both 1/1 cycles, clean exits; Codex usage parsed |

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
- **`tests/test_orchestration.py`** — 64 tests.

### Phase 2 — orchestrator scope + parallelism

- **`scripts/run_scope_guard.py`** — third `orchestrator` mode. The scope file
  stores only the orchestration id; `resolve_scope()` hydrates the current child
  list from `orchestration.json` on every hook invocation, then passes that plain
  data to the still-pure `decide()`. The orchestrator may read its own campaign
  folder and manifest-listed child runs and may issue explicit takeover commands;
  foreign runs, other orchestration folders, implicit run selection, and `runs/`
  enumeration are denied. Successful non-dry-run `orchestrate new` / `spawn`
  commands auto-bind the session. Existing research/analyst cases were not
  changed; the five required cases and binding paths were appended as new tests.
- **`monitor.py`** — refreshes process state, registry progress, champion/gini,
  last activity, and elapsed/timeout values for `orchestrate status`. A live
  process crossing its stored allowance is marked `timed_out` but never signalled.
  `orchestrate kill` sends SIGTERM to the detached process group and records
  `killed` under the manifest lock.
- **Detached observation** — `scripts/run_orchestration_child.py` is a zero-LLM
  wrapper around the exact backend argv. It writes `logs/dNN.exit.json`
  atomically when the backend exits, allowing a later CLI process to distinguish
  clean completion from a crash. The backend argv, not the wrapper argv, remains
  the command recorded in the manifest/audit surface.
- **Parallel spawning** — launch + PID/status persistence now happen inside the
  same manifest lock as bootstrap and delegation creation. This closes the
  read-modify-write window where concurrent detached spawners could lose one
  another's records. The lock has both an exclusivity test and a two-thread
  serialization test.
- **`orchestrate respawn`** — inherits the source backend by default and creates
  a new run, or `--continue-run` reuses the stopped child's run/champion with a
  revised brief and fresh cycle budget. Continuations record `respawn_of`,
  `continue_run`, and `cycles_at_start`, so reports contain only the new cycles.
  Respawn refuses a still-running process, including a live `timed_out` child;
  the operator must wait or call `kill` first.
- **Report collection** — `collect` refreshes detached state first and persists
  each generated `report_path`. Cycle accounting now sums sessions in a run and
  subtracts `cycles_at_start` for same-run continuation.
- **Tests** — detached wrapper output/exit capture, timeout-without-kill, explicit
  process-group kill, same-run continuation lineage, stopped-process enforcement,
  report accounting, lock concurrency, guard policy, and orchestrator auto-bind.

### Phase 3 — Codex backend

- **`backends.toml`** — added `codex-gpt-5-5-medium`, pinned to the locally
  configured and locally advertised `gpt-5.5` model at medium reasoning. The
  entry carries the full design §4.2/§4.7 selection metadata and uses
  `codex -a never exec --json --sandbox workspace-write ... -` for a headless,
  sandboxed, stdin-prompted run. Added `stub-codex`, a zero-cost Codex-shaped
  backend on the `codex` track for integration testing.
- **`adapters.py`** — tool-specific exit interpretation. Generic backends retain
  the Phase 1/2 process-exit behaviour. Codex additionally requires a terminal
  `turn.completed` JSONL event for a clean zero exit and aggregates numeric
  usage leaves across completed turns, tolerating non-JSON stderr lines.
- **Detached exit sidecar + manifest** — the wrapper now receives the backend
  tool name, inspects the completed log, and atomically records `clean_exit` and
  best-effort `usage` beside the exit code. `Delegation` persists those as
  backward-compatible optional/additive `clean_exit` and `tool_usage` fields;
  both `--wait` finalisation and detached `status` consume the same sidecar.
- **Reports** — existing run-telemetry keys are unchanged; when backend usage is
  available it is appended at `cost.llm_usage.backend`.
- **Codex-shaped smoke script** — `scripts/stub_codex_subagent.py` forwards the
  launch prompt over stdin to the existing scripted research stub, then emits
  the documented Codex JSONL lifecycle with zero usage. It exercises the real
  bootstrap → handoff → cycle → decision → finish → report path without an LLM.
- **Contract parity** — no generated contract edit was needed. The existing
  generator already emits byte-identical `AGENT.md`, `AGENTS.md`, and
  `CLAUDE.md` files with the Orchestrated mode subsection. The existing full-file
  parity test remains, plus a focused assertion that Codex's `AGENTS.md` carries
  the same subsection.
- **Tests** — exact command rendering and metadata, Codex template validation,
  stdin prompt transport through the detached wrapper, clean/incomplete/nonzero
  exits, nested multi-turn usage aggregation, manifest round-trip, detached
  monitor propagation, report exposure, and contract parity.

### Phase 4 — playoff and consolidation

- **`playoff.py`** — finalist collection from delegation reports, mechanical
  baseline exclusion, ascending `gini_weighted` ordering, fresh consolidation
  bootstrap, weakest-finalist seeding, and resumable challenger replay through
  the public `compare_experiments` / `record_decision` path. Interactive mode
  writes a partial report and stops at every `pending_llm`; `--auto-decide`
  promotes only when the advisory decision is `promote`, every standard check
  is literally true, and hard guardrails pass.
- **One privileged replay implementation** — `ReplaySource`,
  `replay_experiment`, and `seed_champion_from_source` are shared by playoff,
  `brief.seed_champion`, and `respawn --seed-champion from:dNN`. Recipes are
  reconstructed from the source config snapshot. Script finalists require and
  copy both the exact run-local script and originating proposal into the
  destination's `orchestration_replay/<label>/` folder. SHA-256s, source run,
  source experiment, source delegation, destination experiment, and copied
  paths are recorded in `replay_manifest.json`; the artifact registry links the
  audit files to the replayed experiment. Completed imports are idempotently
  recoverable after interruption and lineage insertion is deduplicated.
- **Lineage** — every replay appends `orchestration_replay/lineage.json`; the
  final report resolves the consolidation champion back to its originating
  orchestration delegation/run/experiment. Seed installation is recorded as a
  champion-history `seeded` action, distinct from a statistical promotion.
- **Reports and manifest** — partial and final
  `playoff/playoff_report.json` / `.md` contain ordered finalists, exclusions,
  every pairing's comparison summary, gate table, decision/rationale, and final
  champion lineage. The campaign transitions `active -> consolidating ->
  completed`; the existing consolidation `track`, `run_id`, and
  `playoff_report` fields are populated at initialization.
- **Target and guard continuity** — orchestration child/consolidation run
  manifests now persist the campaign target mode. `load_config` reads that field
  only when an `orchestration_id` is present, so a fresh CLI process making an
  interactive decision cannot silently fall back to the dataset default and
  single-agent loading is unchanged. Orchestrator scope hydration adds the
  manifest-listed consolidation run, allowing the printed `record-decision`
  takeover command while retaining all foreign-run denials.
- **Fixed preprocessing** — replay refuses a source snapshot whose effective
  claim-capping enablement or threshold conflicts with the destination dataset's
  fixed policy.
- **Tests** — new `tests/test_orchestration_playoff.py` uses registry/report
  fixtures and deterministic comparison substitutes. It covers recipe replay,
  script + proposal copy-in, replay recovery, brief seeding, three known-order
  eligible finalists, a fourth `champion_is_baseline` exclusion, `--include`,
  ascending gauntlet order, interactive stops/resume, strict auto decisions,
  final lineage, both reports, manifest transitions, CLI modes, and invocation
  of the existing protected promotion hook. No child process or model runs.

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
9. **`max_budget_usd` added as a registry field + command placeholder.** The design's
   §4.2 template shows only `{prompt}` and `{max_turns}`; the real Claude CLI has
   no `--max-turns` but does have `--max-budget-usd`. Both placeholders are now
   supported so the registry stays tool-agnostic. See "CLI flag pinning".
10. **`_TOOL_FLAG_ENUMS` validates enumerated flag values in `backends.py`.** Not in
    the design. Justified because Claude *warns and silently defaults* on a bad
    `--effort`, which would re-create the exact defect (two "different" backends
    that are secretly the same) that the scorecard cannot detect. Fails at load
    time; scoped per `tool`.
11. **`respawn --seed-champion` is deferred to Phase 4; Phase 2 exposes no
    placeholder flag.** Alex delegated the phase-ordering choice to the build
    agent. Pulling replay forward would duplicate the playoff's most sensitive
    mechanic before its fixture gauntlet exists; accepting a flag that raises at
    runtime would be a dishonest interface. `brief.seed_champion` continues to
    fail before creating anything, and Phase 4 will land replay and the respawn
    flag together. Phase 2 ships default new-run respawn plus `--continue-run`.
12. **Detached backends run through a stdlib exit-status wrapper.** The design
    says the spawner launches the backend command directly, but a later CLI
    process cannot `waitpid()` a child it did not parent. PID liveness alone
    cannot distinguish exit 0 from a crash. The wrapper preserves the exact
    backend argv as the audit command and adds only an atomic exit sidecar.
13. **Continuation lineage adds three optional delegation fields:** `respawn_of`,
    `continue_run`, and `cycles_at_start`. The design's sketch does not include
    them, but without a start offset a continued run's second report would claim
    the first delegation's cycles and experiments as its own. Ordinary
    delegations serialize the backward-compatible defaults (`null`, `false`, 0).
14. **Manifest lock wait/staleness increased from 30 s / 120 s to 300 s / 900 s.**
    The first three-way concurrent smoke correctly serialized bootstrap, but a
    cold matplotlib/font-cache build held the lock beyond 30 seconds and one
    caller timed out. The original stale threshold was also unsafe for any valid
    bootstrap longer than two minutes. Explicit-test timeouts remain configurable.
15. **`finish-delegation` records testimony; terminal report generation happens
    after backend exit (`--wait`) or on `collect`.** The design says finish
    triggers report generation, but at that instant the detached backend is still
    live and its exit status is unknowable. Generating then would create a stale
    report with `status=running`. `collect` refreshes the exit sidecar first and
    produces the authoritative report.
16. **The configured Codex backend is `gpt-5.5` medium, not the design's
    illustrative `codex-5.6-luna-medium`.** `codex debug models` on the installed
    CLI advertises `gpt-5.5`, confirms medium reasoning support, and the repo's
    `.codex/config.toml` plus console catalog both pin `gpt-5.5`. No local
    non-executing source recognises a “5.6 Luna” identifier. Using it would make
    the human-owned registry claim an unverified model; real endpoint validation
    remains deliberately deferred.
17. **Codex prompt transport uses stdin with an explicit final `-`, not the
    design sketch's `{prompt}` argv placeholder.** `codex exec --help` explicitly
    supports both. Stdin avoids command-line length limits and keeps the full
    orchestration prompt out of process listings; the existing generic prompt
    channel already supports it.
18. **`Delegation` adds `clean_exit` and `tool_usage`.** Design §4.2/§7 requires
    adapter clean-exit detection and tool-output usage in delegation/campaign
    telemetry but the §4.1 JSON sketch omits their storage. Both fields are
    additive, nullable/empty for old manifests, and populated from the detached
    exit sidecar so `--wait` and later `status` observe identical facts.
19. **A separate `stub-codex` backend and script simulate Codex JSONL.** Reusing
    the plain stub with `track = "codex"` would prove track selection but would
    bypass the new adapter. The shaped stub tests prompt, lifecycle, and usage
    parsing at zero cost and stays in the registry as a permanent smoke surface.
20. **Consolidation uses the lowest-ranked finalist's track.** Design §4.6 says
    the consolidation run is on the orchestrator's chosen track, but its designed
    CLI exposes only `--include`, `--interactive`, and `--auto-decide`; adding a
    new `--track` interface would contradict the Phase 4 request. Deterministically
    using the seed finalist's track creates a normal allowed tracked run without
    inventing an unreviewed flag. The choice is recorded in the manifest/report.
21. **Auto decisions still serialize `decided_by="llm"` in the protected
    comparison registry.** The existing public `record_decision` API owns the
    only complete promotion transaction (champion, history, report, recipe hooks,
    protected evaluation hook) and hard-codes that attribution. Phase 4 does not
    edit the protected comparison runner merely to rename it. The playoff report
    unambiguously records `decision_mode="auto"`, strict gate evidence, and the
    mechanical rationale.
22. **The source proposal is copied as an audit artifact, not enqueued into the
    consolidation proposal queue.** Enqueuing would add ordinary research-tree,
    screening, and reflection semantics that are not part of design §4.6 and can
    auto-reject before the required fresh comparison. `run_experiment` creates a
    native destination experiment; `compare_experiments` then executes the full
    standard comparison gauntlet. Both are existing public paths.
23. **Orchestration target mode is persisted and recovered from run manifests.**
    This additive shared-path change is gated on `orchestration_id`; manifests
    from single-agent runs ignore the new reader and retain prior behaviour. It
    is required because interactive playoff decisions happen in later CLI
    processes, after the in-memory target override used during bootstrap is gone.

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
- **Timeout enforcement means state, not signalling.** `status` marks an overdue
  live child `timed_out`; only the explicit `kill` command sends SIGTERM. This is
  the design's "never auto-kills in v1" rule applied literally.
- **Respawn source must be stopped.** A `timed_out` delegation can still be live,
  so it is not eligible until killed. This prevents two agents from writing the
  same continued run and prevents unnoticed spend from an old process while a
  replacement starts.
- **Script replay requires a source proposal.** A script without its proposal is
  not auditable enough for privileged copy-in; recipe replay can proceed from its
  portable config snapshot alone.
- **One eligible finalist is valid.** The playoff creates a consolidation run,
  seeds it from that finalist, writes final lineage, and completes with no pairings.
- **Missing or incomplete delegation reports are explicit exclusions.** They are
  written into the playoff report (`report_missing` or
  `champion_or_gini_missing`) rather than guessed from child claims.
- **Auto promotion is fail-closed.** A non-empty check map, all checks exactly
  true, advisory `promote`, and hard-guardrail `passed=true` are all required;
  absent evidence rejects.

---

## CLI flag pinning

Initially the `claude` binary was **not installed** (only `/Applications/Claude.app`;
this session runs Claude Code via the desktop app's local agent mode). Every
Claude flag was therefore an unverified guess, and no thinking-effort flag was
pinned at all — `claude-sonnet-low` and `claude-sonnet-medium` differed only in
`model_name` attribution. Alex asked for it to be installed
(`npm install -g @anthropic-ai/claude-code` → **2.1.206**), and the flags are now
pinned against real `claude --help` output. **No real sub-agent was ever launched
(build-prompt rule 5): only `--help` and argument-parsing probes.**

**Claude Code 2.1.206 — verified:**

| Flag | Status |
|---|---|
| `-p` / `--print` | ✅ exists |
| `--model <full-name\|alias>` | ✅ exists (aliases `opus`/`sonnet`/`fable`) |
| `--effort low\|medium\|high\|xhigh\|max` | ✅ **exists** — the real effort flag |
| `--output-format text\|json\|stream-json` | ✅ exists (requires `--print`) |
| `--verbose` | ✅ exists |
| `--permission-mode acceptEdits\|auto\|bypassPermissions\|manual\|dontAsk\|plan` | ✅ exists |
| `--max-budget-usd <amount>` | ✅ exists (requires `--print`) — hard spend ceiling |
| `--max-turns` | ❌ **does not exist**; my guess. Removed. |

Consequences, all applied:

1. `--max-turns` removed from both Claude entries. `--max-budget-usd` (5 USD for
   `low`, 10 for `medium`) replaces it — a spend ceiling is the guarantee we
   actually wanted from a turn cap. `max_budget_usd` is now a first-class
   registry field and command placeholder alongside `max_turns` (which stays for
   tools that do have it).
2. `--effort low` / `--effort medium` pinned, so the two backends now differ in
   substance, not just in name. New test
   `test_distinct_backends_render_distinct_commands` fails if any two spawnable
   backends ever render identical argv again — that is the invariant the original
   defect broke.
3. **Claude only *warns* on an invalid `--effort` and silently falls back to the
   default effort** (verified: `claude --effort bogus --version` →
   `Warning: Unknown --effort value 'bogus' — ignoring it…`). That would quietly
   re-collapse two backends into one while the scorecard compared them as rivals.
   So `backends.py` now validates enumerated flag values itself, per tool
   (`_TOOL_FLAG_ENUMS`), and a bad value fails at **load** time. Scoped to
   `tool == "claude"`; another tool may use `--effort` with its own vocabulary.

Remaining `TODO(pin)` — one, and it needs a real spawn:
`--permission-mode acceptEdits` is a *valid value*, but whether it *suffices* for
a headless sub-agent's Bash tool calls (the `autoresearch` commands) is unverified;
confirming it costs one real API call, which the build was not allowed to make. If
a sub-agent stalls on a permission prompt, the candidates are
`--permission-mode bypassPermissions` or an explicit
`--allowed-tools "Bash Edit Read Write"`.

**Codex CLI 0.137.0 — verified without a model invocation:**

| Flag / behaviour | Status |
|---|---|
| `exec` | ✅ non-interactive subcommand |
| `-a/--ask-for-approval never` | ✅ global flag; prevents an impossible headless approval wait |
| `--json` | ✅ JSONL output; local console adapter documents `turn.completed.usage` |
| `-s/--sandbox read-only\|workspace-write\|danger-full-access` | ✅ entry pins `workspace-write` |
| `-C/--cd <dir>` | ✅ entry pins repo working root |
| `--skip-git-repo-check` | ✅ exists |
| `-m/--model <model>` | ✅ exists; `codex debug models` advertises `gpt-5.5` |
| `-c key=value` | ✅ generic TOML config override |
| `model_reasoning_effort="medium"` | ✅ existing console adapter uses this key; model catalog confirms `gpt-5.5` supports medium |
| `--color never` | ✅ valid enum value |
| final prompt `-` / omitted prompt | ✅ stdin prompt transport |
| `--max-turns` | ❌ absent |
| dedicated `--reasoning-effort` | ❌ absent; use `-c model_reasoning_effort=…` |

Argument-only probes rejected an unknown flag and an invalid sandbox value;
the complete pinned argv parsed under `--help`/`--version`. `--version` exits
before validating config keys, so the effort-key evidence comes from the repo's
already-tested console adapter and the non-executing local model catalog, not a
paid call. `backends.py` independently validates the pinned enums, forbids
dangerous bypass flags and nonexistent `--max-turns`, requires JSONL plus the
workspace sandbox, and rejects unsupported reasoning values at registry load.

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

## Exact commands run (Phase 2 milestone)

Full suite and generated-contract check were run before using the documented
pytest-gate skip for the stub campaign:

```bash
.venv/bin/python -m pytest
.venv/bin/python scripts/generate_agent_contract.py --check
autoresearch orchestrate new --dataset french_motor --total-cycles 6 \
  --model-provider openai --model-name codex
# orchestration_id: 20260710T103721Z

# These three commands were issued concurrently (Promise.all at the harness):
AUTORESEARCH_SKIP_PYTEST_GATE=1 autoresearch orchestrate spawn \
  --orchestration-id 20260710T103721Z --brief /private/tmp/phase2_brief_d01.json \
  --backend stub --no-wait
AUTORESEARCH_SKIP_PYTEST_GATE=1 autoresearch orchestrate spawn \
  --orchestration-id 20260710T103721Z --brief /private/tmp/phase2_brief_d02.json \
  --backend stub --no-wait
AUTORESEARCH_SKIP_PYTEST_GATE=1 autoresearch orchestrate spawn \
  --orchestration-id 20260710T103721Z --brief /private/tmp/phase2_brief_d03.json \
  --backend stub --no-wait

autoresearch orchestrate status --orchestration-id 20260710T103721Z
autoresearch orchestrate collect --orchestration-id 20260710T103721Z
.venv/bin/python -m pytest \
  tests/test_run_scope_guard.py::test_orchestrator_denies_foreign_run -q
```

The first concurrent attempt proved exclusion but exposed the old wait timeout:
two spawns succeeded as `d01`/`d02`; one caller received `TimeoutError` after
30 seconds while cold bootstrap/cache work legitimately held the lock. After
raising wait/staleness to 300/900 seconds, the remaining two-cycle brief was
spawned as `d03`. This did not exceed budget: the failed caller never appended a
delegation, so the final campaign is exactly 6/6 cycles.

The key live status (all three processes alive concurrently):

```
Orchestration 20260710T103721Z: active cycles=6/6
d01  stub  running  true  1/2  gini=0.0000  claude/20260710T103729Z
d02  stub  running  true  0/2  gini=0.0000  claude/20260710T103748Z
d03  stub  running  true  0/2  gini=0.0000  claude/20260710T103857Z
```

Final status showed all three `completed`, `process_alive=false`, and `2/2`
cycles. `collect` wrote all three reports, which were then read directly by the
orchestrator scope. Registry-backed key facts were identical by construction:
`status=completed`, `cycles=2/2`, `model_family=global_mean`,
`gini_weighted=0.0`, summary present, distress
`[all_rejected, champion_is_baseline]`. The direct pure-decision guard probe
passed (`1 passed`); no live foreign-path tool call was attempted.

---

## Exact commands run (Phase 3 milestone)

The full suite and contract check passed before applying the documented
pytest-gate skip to smoke spawns:

```bash
.venv/bin/python -m pytest
# 528 passed, 2 skipped
.venv/bin/python scripts/generate_agent_contract.py --check
# AGENT.md (and harness mirrors) in sync.

.venv/bin/autoresearch orchestrate new --dataset french_motor --total-cycles 2 \
  --model-provider openai --model-name gpt-5.5
# orchestration_id: 20260710T110354Z

AUTORESEARCH_SKIP_PYTEST_GATE=1 .venv/bin/autoresearch orchestrate spawn \
  --orchestration-id 20260710T110354Z \
  --brief /private/tmp/phase3_brief_claude.json --backend stub --wait

AUTORESEARCH_SKIP_PYTEST_GATE=1 .venv/bin/autoresearch orchestrate spawn \
  --orchestration-id 20260710T110354Z \
  --brief /private/tmp/phase3_brief_codex.json --backend stub-codex --wait

.venv/bin/autoresearch orchestrate collect \
  --orchestration-id 20260710T110354Z
.venv/bin/autoresearch orchestrate status \
  --orchestration-id 20260710T110354Z
```

Key terminal state:

```text
Orchestration 20260710T110354Z: active cycles=2/2
d01  stub        completed  false  1/1  gini=0.0000  claude/20260710T110401Z
d02  stub-codex  completed  false  1/1  gini=0.0000  codex/20260710T110527Z
```

Both reports derive `cycles.used=1`, the `global_mean` champion and
`gini_weighted=0.0` from their child registries, with expected distress
`[all_rejected, champion_is_baseline]`. Both manifest records have
`exit_code=0` and `clean_exit=true`. `d02.tool_usage` and
`d02.cost.llm_usage.backend` are
`{input_tokens: 0, cached_input_tokens: 0, output_tokens: 0,
completed_turns: 1}`, proving that the Codex JSONL adapter ran. The plain stub's
usage remains empty. **No real `claude -p` or `codex exec` process was launched.**

---

## Exact commands run (Phase 4 milestone)

Phase 4's milestone is intentionally fixture-only. It creates no research
track, starts no subprocess, uses no model endpoint, and does not need
`AUTORESEARCH_SKIP_PYTEST_GATE`.

```bash
.venv/bin/python -m pytest \
  tests/test_orchestration.py \
  tests/test_orchestration_playoff.py \
  tests/test_run_scope_guard.py -q
# 143 passed

.venv/bin/python -m pytest \
  tests/test_orchestration_playoff.py::test_auto_playoff_orders_finalists_reports_lineage_and_calls_promotion_hook -q
# 1 passed

.venv/bin/python -m pytest
# 539 passed, 2 skipped, 1 warning in 48.87s

.venv/bin/python scripts/generate_agent_contract.py --check
# AGENT.md (and harness mirrors) in sync.
```

The deterministic milestone fixture has three eligible finalists with Gini
`d01=0.10`, `d02=0.20`, `d03=0.30`, plus `d04` carrying
`champion_is_baseline`. Key asserted terminal state:

```text
excluded: d04 (champion_is_baseline)
order: seed:d01 -> challenger:d02 -> challenger:d03
d02: standard gates fail -> auto reject
d03: all standard gates + hard guardrails pass -> auto promote
protected promotion hook calls: [(replayed_d03, cmp_replayed_d03)]
final lineage: consolidation/replayed_d03 -> d03/strong
manifest: completed; consolidation=claude/20260710T150000Z
reports: playoff_report.json + playoff_report.md
```

The interactive fixture separately proves the first call stops on `d02`, an
external normal `record-decision` is observed on resume, and only then is `d03`
replayed. The script fixture copies and hashes `model_replay.py` and
`source_proposal.json`; the recipe fixture replays the portable model object and
proves an interrupted completed import recovers without a second fit.

---

## Open questions for later phases (raise with Alex; do not improvise)

1. **`--permission-mode acceptEdits` sufficiency** — the last `TODO(pin)`; needs
   one real spawn. See "CLI flag pinning".

The seed/replay ordering question was resolved for Phase 2 after Alex said to use
best judgement: defer the flag and replay together to Phase 4 (deviation 11).

There is no open Phase 3 interface question. Real `gpt-5.5` availability,
authentication, sandbox/approval behaviour during tool calls, and live JSONL
usage shape still require the deliberately deferred paid validation campaign.

---

## Notes for whoever continues this build

- `AUTORESEARCH_SKIP_PYTEST_GATE=1` skips the pytest gate that `bootstrap_track`
  runs on every spawn. Use it for stub smoke runs (after running the suite), or a
  2-delegation campaign re-runs the whole suite twice.
- `tests/test_orchestration.py` redirects `artifacts/orchestrations/` into
  `tmp_path` by monkeypatching `manifest.ORCHESTRATIONS_DIR` (the
  `orchestrations_root` fixture). Module-global path constants are resolved at
  call time, so this works for the whole package.
- The `stub` / `stub-no-finish` / `stub-codex` backends are the zero-cost ways to
  exercise spawn → run → report. `stub-codex` additionally tests the Codex track,
  JSONL completion event and usage parser. Reach for them before a real model.
- **The run-scope guard will block you** (an unbound session) from reading any
  `artifacts/tracks/*/runs/*` folder — including a child's handoff. That is the
  gap Phase 2's `orchestrator` scope closes. Until then, verify child-run state
  through the stub's own log output and the delegation reports.
- **There is no `python` on PATH.** Use `.venv/bin/python` for pytest and ad-hoc
  scripts. The `autoresearch` console script on PATH resolves to the *system*
  3.13 framework install, not `.venv`; both work, and the stub backend's
  `python3 scripts/stub_subagent.py` only needs stdlib plus `autoresearch` on
  PATH.
- **The guard runs under all three harnesses** (`.claude/hooks`, `.codex/hooks.json`,
  `.opencode`). Phase 2 edits `scripts/run_scope_guard.py` *while the guard is
  policing the editing session's own tool calls*. `decide()` is fail-open on
  exceptions, but a **logic** bug that returns `(False, …)` will block your own
  Bash/Edit calls. Unit-test `decide()` directly rather than probing it with live
  tool calls, and if you wedge yourself, delete the session's scope file under
  `artifacts/tracks/.scope/`.
- **Phase 2's orchestrator auto-bind will bind the *build* session.** Design §5
  auto-binds on a successful `orchestrate new` / `orchestrate spawn` at
  PostToolUse. The moment a build session runs a smoke campaign it becomes an
  orchestrator scoped to that campaign, and the guard then denies it other
  orchestrations' folders. Expect it; it is the feature working. Clear the scope
  file to unbind.

---

## Deliberately left undone

- **Real-model campaigns (Phase 1 and Phase 3 milestones).** No `claude -p` /
  `codex exec` sub-agent was ever launched (build-prompt rule 5). Deferred to
  Alex's post-review validation.
- **Whether `--permission-mode acceptEdits` suffices for a headless sub-agent's
  Bash calls** — the one remaining `TODO(pin)`; see "CLI flag pinning". Needs one
  real spawn to settle.
- **Phase 5**: `ORCHESTRATOR.md`,
  `docs/RUN_ORCHESTRATED.md`, `orchestrate note`/`report`/`backend-stats`.
