# Insurance AutoResearch — Refactor Specification

This document specifies all changes agreed from the deep review. It is intended
to be executed by a smaller coding model. Every section is self-contained: file
paths are absolute from the repo root, function/symbol names are exact, and the
expected post-conditions are spelled out. Work sections in the order given —
later sections assume earlier ones have landed.

When in doubt: **do not invent behavior**. Keep semantics identical to the
current code unless a section explicitly says to change them. Run
`pytest --tb=short -q` after each major section and fix any regressions before
moving on.

---

## Conventions

- "Delete" means remove the file/function entirely, including imports and
  references throughout the repo.
- "Drop key X from JSON Y" means do not write key X any more; downstream
  consumers must tolerate its absence.
- All new code must keep type hints consistent with the surrounding file.
- Do not add docstrings beyond one short line. Do not add comments unless the
  WHY is non-obvious.
- Do not change `configs/default.toml` fields except where this spec says so.
- Do not change behavior of `evaluation/metrics.py`, `evaluation/resampling.py`,
  `data/holdout_vault.py`, or `experiment_registry/registry.py` schemas —
  those are protected. (Pure deletions/refactors inside `registry.py` that
  preserve the public function signatures are fine, but will require running
  `autoresearch update-integrity-manifest` afterward.)

---

## Section 1 — Fix milestone holdout for scripted models (BUG F)

**File:** `src/autoresearch/milestone.py`

### Problem
`_dispatch_for_milestone` calls `_call_model(...)` without `model_script_path`,
so any promoted challenger whose `model_family` is a script label
(e.g. `scripted_tweedie_glm`) raises `ModuleNotFoundError`, which
`evaluate_on_holdout` silently swallows and writes "skipped".

### Required changes

1. In `_run_evaluation`:
   - After loading `snapshot = read_json(config_snapshot_path)`, also read
     `snapshot.get("model_script_path")`. If non-empty, convert to `Path`. If
     the path does not exist on disk, raise `FileNotFoundError` with a clear
     message (do not silently skip — see step 3).
   - Pass that path through to `_dispatch_for_milestone` as a new
     `model_script_path: Path | None = None` keyword argument.

2. In `_dispatch_for_milestone`:
   - Add `model_script_path: Path | None = None` to the signature.
   - Forward it to `_call_model(..., model_script_path=model_script_path)`.

3. In `evaluate_on_holdout`:
   - Replace the blanket `except Exception` swallow with structured handling:
     - If the holdout vault is unavailable (token missing, file absent), keep
       the existing "skipped" behavior — that is a legitimate environmental
       gap.
     - For any other exception (script missing, dispatch error, metric
       computation failure), still write the JSON/markdown warning **but
       additionally re-raise** so the promotion flow surfaces the failure.
   - Concretely: detect the vault-absent case by inspecting the exception
     message for `"AUTORESEARCH_MILESTONE_TOKEN"` or by attempting
     `load_holdout_dataset` defensively at the top of `_run_evaluation` and
     catching only that specific failure. Everything else propagates.

4. Add a test `tests/test_milestone_scripted.py`:
   - Build a tiny in-memory champion experiment whose `model_family` is
     `"scripted_demo"` and whose `model_script_path` points at a stub
     `fit_predict` returning a constant prediction.
   - Stub `load_holdout_dataset` to return a small DataFrame.
   - Call `evaluate_on_holdout` and assert the JSON report has
     `status == "completed"` and `holdout_metrics` populated.
   - Add a negative test: same setup but with `model_script_path` pointing at
     a non-existent file; assert `evaluate_on_holdout` returns a report whose
     `status != "completed"` AND that the exception was raised (use
     `pytest.raises`).

### Post-conditions
- Promoted scripted challengers produce a real holdout report.
- Vault-absent runs still degrade gracefully.
- Any other holdout-evaluation failure is loud, not silent.

---

## Section 2 — Fix `current_champion_id` polarity (BUG G)

**File:** `src/autoresearch/comparison_runner.py`

### Problem
Lines 281–287: `min(rows, key=lambda row: row["mean_score"])` is "lower is
better", but the primary metric is now `gini_weighted` (higher is better). The
fallback picks the worst experiment.

### Required changes

1. Delete the `current_champion_id` function entirely.
2. In `compare_against_current_champion`, remove the `else current_champion_id(config)`
   branch. If `get_official_champion(config.registry_path)` returns `None`,
   raise `ValueError("Official champion is not initialised. Run "
   "init-official-champion first.")` — matching the message used in
   `workflow._require_champion`.
3. Remove all imports of `current_champion_id` elsewhere in the repo
   (`grep -rn current_champion_id src/ tests/`).

### Post-conditions
- No silent worst-model selection is possible.
- Behavior when champion is missing is the same as in
  `run_next_queued_proposal`.

---

## Section 3 — Fix cross-track report mislabel (BUG H)

**File:** `src/autoresearch/tracks.py`

### Required changes

1. Replace the hard-coded `"Mean Tweedie deviance (p=1.5)"` row in
   `_REPORT_TEMPLATE` (line ~45) with `"Mean {metric_label}"` and add a
   `metric_label` placeholder.
2. In `_run_comparison`, set `metric_label = config_a.primary_metric` and pass
   it into the `_REPORT_TEMPLATE.format(...)` call.
3. Keep the sign convention as-is — the existing
   `if mean_lift > 0: winner = config_b.track_id` block is correct for both
   higher-is-better and lower-is-better because `paired_comparison` already
   normalises lift so that positive == challenger wins. Add a one-line comment
   ABOVE that block stating this invariant (this is one of the rare WHY
   comments worth keeping).

### Post-conditions
- The cross-track Markdown report names the actual metric in use.

---

## Section 4 — Fix `_to_toml` None handling (BUG I)

**File:** `src/autoresearch/controller/workflow.py`

### Required changes

In `_to_toml` (~line 462), change `_sanitise` to **drop** keys whose value is
`None` instead of converting them to `""`. For lists, drop None items
(`tomli_w` rejects them at top level but lists of strings/numbers may be
empty). Concretely:

```python
def _sanitise(obj):
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_sanitise(i) for i in obj if i is not None]
    return obj
```

### Post-conditions
- `experiment_config_attempt_*.toml` files never contain `key = ""` placeholders
  that originated as `None`.
- Existing tests for the workflow continue to pass.

---

## Section 5 — Tighten integrity-scan whitelist (BUG J)

**File:** `src/autoresearch/utils/integrity.py`

### Required changes

1. Replace `_SCAN_WHITELIST` with a frozenset of **exact path-relative-to-`src/autoresearch/`** strings:
   ```python
   _SCAN_WHITELIST = frozenset({
       "data/holdout_vault.py",
       "milestone.py",
       "models/dispatcher.py",
       "utils/integrity.py",
   })
   ```
   Note: drop `baselines.py` (it is being deleted in Section 8) and the
   `test_*` patterns (those files live under `tests/`, not under the scan
   directories `src/autoresearch/models/` or `src/autoresearch/features/`).

2. Rewrite `_file_is_whitelisted(path: Path) -> bool` to compare the relative
   path from `src/autoresearch/` (compute via
   `path.resolve().relative_to((root / 'src' / 'autoresearch').resolve())`).
   If the path is not under that root (defensive), return False.

3. `scan_file_for_holdout_access` is called from `experiment_runner.py` on the
   model script path which lives under `artifacts/.../proposal/model_attempt_1.py`.
   That path is **not** under `src/autoresearch/`, so the whitelist check must
   not short-circuit it. Update the function so the whitelist only applies
   when the file IS under `src/autoresearch/`; otherwise always scan.

4. Add a test `tests/test_integrity_whitelist.py`:
   - Asserts a file at `src/autoresearch/models/holdout_vault_evil.py`
     containing the marker is NOT whitelisted (would have been under the old
     substring rule).
   - Asserts the real `data/holdout_vault.py` IS whitelisted.
   - Asserts a model script outside `src/autoresearch/` is always scanned
     regardless of name.

### Post-conditions
- Whitelist cannot be bypassed by naming a model script with a whitelisted
  substring.

---

## Section 6 — One log format in `docs/RESEARCH_LOG.md` (Finding K)

**File:** `src/autoresearch/comparison_runner.py`

### Required changes

1. Delete the entire `_append_research_log` function and its call site inside
   `compare_experiments`.
2. Leave `_read_research_log_tail` consumption alone for now (it is removed in
   Section 11).
3. Update `AGENT.md`'s "Step 6 — Update the research log" section: it is
   already correct (agent writes free-form sections); no change required.

### Post-conditions
- `docs/RESEARCH_LOG.md` only contains entries the agent wrote itself, in one
  consistent format.

---

## Section 7 — Delete API proposers + mock references (Finding L)

### Files affected
- `src/autoresearch/controller/proposer.py`
- `src/autoresearch/controller/workflow.py`
- `src/autoresearch/cli.py`
- `configs/default.toml`
- `src/autoresearch/config.py`
- `README.md`
- `docs/architecture.md`
- `tests/` — any test referencing the removed classes/commands

### Required changes

1. In `controller/proposer.py`:
   - Delete `OpenAIProposer`, `AnthropicProposer`, `_extract_openai_text`,
     `_parse_json`, the `time`, `os`, `urllib.error`, `urllib.request` imports
     they need, and the corresponding branches in `proposer_from_config`.
   - Delete `FileHandoffProposer` (a no-op wrapper).
   - Delete `build_prompt` entirely — it is unreachable once API proposers are
     gone.
   - `proposer_from_config` reduces to: return `FileProposer(config.llm_proposal_file)`
     for any provider value, OR raise if `provider != "file_handoff"`. Use the
     second form (strict) to keep the contract clear.

2. In `controller/workflow.py`:
   - Delete `generate_and_enqueue_proposal`, `run_one_cycle`, `run_n_cycles`.
   - Delete the `build_prompt` and `proposer_from_config` imports if no
     remaining function uses them. (They likely become unused.)
   - Delete `_materialise_embedded_model_script` if it is only used by
     `generate_and_enqueue_proposal` — verify via grep first. The file-handoff
     path uses `_materialise_referenced_model_script`.

3. In `cli.py`:
   - Remove subparsers: `generate-proposal`, `run-cycle`, `run-cycles`.
   - Remove the corresponding `if args.command == ...` blocks in `main`.
   - Remove the now-unused imports from `controller.workflow`.

4. In `configs/default.toml`:
   - Delete the entire `[llm]` section. The file-handoff inbox path is now
     hard-coded into the run layout via `config.py` (it already is: see
     `llm_proposal_file = handoff_proposal_inbox_dir / "manual_proposals.jsonl"`
     for tracked runs).

5. In `config.py`:
   - Remove `llm_provider`, `llm_model`, `llm_temperature`, `llm_proposal_file`
     from `ProjectConfig`.
   - Remove the corresponding parsing of `raw["llm"]` from `load_config`.
   - For the default-track path (no `--track`), set the inbox file to
     `handoff_proposal_inbox_dir / "manual_proposals.jsonl"` and expose it as
     a property/attribute named `proposal_inbox_file` if any caller still
     needs it. Update `FileProposer` callers (`proposer_from_config`,
     `controller/handoff.py` if it referenced `config.llm_proposal_file`) to
     use the new attribute.

6. In `controller/handoff.py`:
   - Replace any `config.llm_provider` / `config.llm_model` references in
     `inbox_status` and `record_proposal` calls. For `record_proposal`, use the
     literal string `"file_handoff"` for `llm_provider` and `None` for
     `llm_model`. For the `inbox_status` payload, drop the `"provider"` and
     `"mode"` keys (they are now constant).

7. In `README.md`:
   - Delete the MockProposer bullet (line 22).
   - Delete the "Optional Runtime Proposers" section entirely (lines ~362–384).
   - Replace the test-count claims ("64 tests", "52 tests") with "Tests cover
     ..." — no number.
   - Fix the "official champion ... starts as the Tweedie GLM baseline" line:
     it now starts as the `global_mean` baseline (Section 9 also covers this).

8. In `docs/architecture.md`:
   - Replace the `MockProposer, OpenAIProposer, AnthropicProposer` comment on
     line 28 with `FileProposer (file-handoff inbox)`.

9. Delete any test (`tests/test_*.py`) that imports the removed classes or
   exercises the removed CLI commands. `grep -l "OpenAIProposer\|AnthropicProposer\|MockProposer\|generate-proposal\|run-cycle\|run-cycles" tests/`
   to find them. Do not silently disable tests — delete them outright if the
   feature is gone.

### Post-conditions
- `proposer.py` is ~50 lines of just `FileProposer`.
- CLI has no `generate-proposal`, `run-cycle`, `run-cycles`.
- `configs/default.toml` has no `[llm]` block.
- `ProjectConfig` has no `llm_*` fields.

---

## Section 8 — Delete dead built-in models + hyperparameter validators (Findings A, B, M)

### Files affected
- `src/autoresearch/models/baselines.py` — DELETE
- `src/autoresearch/models/glm.py` — DELETE
- `src/autoresearch/models/gbm.py` — DELETE
- `src/autoresearch/models/dispatcher.py` — slim
- `src/autoresearch/controller/proposal_schema.py` — slim
- `src/autoresearch/experiment_runner.py` — update imports
- `src/autoresearch/milestone.py` — update imports
- `configs/experiments/examples/` — DELETE the entire directory
- `configs/default.toml` — slim search space

### Required changes

1. Move the shared column constants into `dispatcher.py` (they already exist
   there: `RECORD_ID`, `EXPOSURE`, `CLAIM_COUNT`, `CLAIM_EVENTS`, `CLAIM_COST`,
   `RAW_CLAIM_COST`). Update `experiment_runner.py:17` and `milestone.py:33` to
   import `RAW_CLAIM_COST` from `autoresearch.models.dispatcher` instead of
   `autoresearch.models.baselines`.

2. In `dispatcher.py`:
   - Delete the entire `_call_model` body except (a) the `model_script_path is
     not None` branch (call `_call_script_model`), (b) the `model_family ==
     "global_mean"` branch which dispatches to
     `autoresearch.models.global_mean.fit_predict`, and (c) the open-registry
     fallback `importlib.import_module(f"autoresearch.models.{model_family}")`.
   - Remove the `tweedie_glm`, `frequency_severity_glm`, `tweedie_gbm`,
     `regularized_linear` branches.
   - Remove the unused imports for those families.

3. In `controller/proposal_schema.py`:
   - Delete `_validate_model_hyperparameters` entirely.
   - Remove its call site in `validate_proposal`.
   - Delete the `_SUPPORTED_FAMILIES` constant (unused after removal).
   - In `allowed_search_space`, remove the `for family in families: ...
     space[f"{family}_params"] = family_cfg` block. The agent's prompt no
     longer needs those bounds.

4. In `configs/default.toml`:
   - Replace `model_families = [...]` with `model_families = ["global_mean"]`.
     (Note: with `allow_open_model_families = true` the agent can still use
     any string; this just stops advertising the dead names.)
   - Delete the `[search_space.tweedie_glm]`, `[search_space.frequency_severity_glm]`,
     `[search_space.tweedie_gbm]`, `[search_space.regularized_linear]` blocks
     entirely.

5. Delete the directory `configs/experiments/examples/` and every file under
   it.

6. Delete any test that exercises the removed model families directly. Re-run
   `pytest --tb=short -q` and fix any test that imported the deleted modules.
   Tests covering `global_mean`, `dispatcher` scripted-path, `proposal_schema`
   validation (minus the hyperparameter validator), CV, metrics, holdout
   separation must continue to pass.

7. Update `AGENT.md`:
   - In the "Dataset schema" section keep the column table as-is.
   - In the "Column constants" subsection (around line 138), it already says
     "Import from `autoresearch.models.dispatcher`" — good.
   - Remove any mention of "built-in model families" elsewhere in the file
     (search for `tweedie_glm`, `tweedie_gbm`, `regularized_linear` and update
     prose accordingly).

### Post-conditions
- `src/autoresearch/models/` contains only `__init__.py`, `dispatcher.py`,
  `global_mean.py`.
- The agent's allowed-search-space JSON no longer advertises per-family
  hyperparameter bounds.
- All experiments run via either the global_mean built-in or a run-local
  script — no third path.

---

## Section 9 — README and docs accuracy (Findings D, E, N)

### Files affected
- `README.md`
- `docs/architecture.md`
- `configs/experiments/examples/` (already deleted in Section 8)

### Required changes

1. In `README.md`:
   - "Features" bullet list (top of file): remove mock proposer bullets, remove
     hard-coded test counts, replace the "actuarially-correct model layer"
     bullet (currently advertises Tweedie GLM, freq×sev, GBM, regularized
     linear) with: **"Open model surface — every non-global_mean experiment
     supplies a run-local Python `fit_predict` script. The built-in
     `global_mean` baseline is the no-model starting champion for every run."**
   - Fix the "Official Champion" paragraph (around line 261): it now starts as
     the `global_mean` baseline.
   - In the "Primary Metric" section, no change required.
   - In the "Baseline Experiments" section, remove the references to
     `direct_pure_premium.toml` and `frequency_severity.toml` (deleted). Keep
     only `global_mean.toml`.
   - Search/replace any remaining hard-coded test counts ("64 tests",
     "52 tests") with "the test suite".

2. In `docs/architecture.md`:
   - Update the `proposer.py` line to reflect `FileProposer` only.
   - If the document references "built-in model families" or the
     hyperparameter validator, update those paragraphs.

3. The examples directory is already gone (Section 8). Verify no other doc
   references files from it (`grep -rn "examples/" docs/ README.md`).

### Post-conditions
- No README claim conflicts with the code.
- No reference to deleted files anywhere in docs.

---

## Section 10 — Refactor milestone dispatch (Finding O)

**Files:** `src/autoresearch/models/dispatcher.py`, `src/autoresearch/milestone.py`

### Required changes

1. In `dispatcher.py`, add a parameter `allow_holdout_split: bool = False` to
   `dispatch_model`. When True, skip the
   `(data["split"] == "milestone_holdout").any()` guard. Keep the default
   False so the experiment runner is unchanged.

2. In `milestone.py`, delete `_dispatch_for_milestone` and call
   `dispatch_model(...)` directly:
   - Pass `allow_holdout_split=True`.
   - Pass `train_split="train_full"`, `score_splits=("milestone_holdout",)`.
   - Pass the scripted `model_script_path` resolved in Section 1.
   - The function returns a `ModelResult`; use `result.predictions` exactly as
     `_dispatch_for_milestone` returned the prediction frame.

3. Remove the now-unused private import `from autoresearch.models.dispatcher
   import _call_model` from `milestone.py`.

### Post-conditions
- No code duplication between the experiment runner's dispatch and the
  milestone evaluator.
- The Section 1 fix and this refactor compose cleanly.

---

## Section 11 — Slim the LLM context bundle (Findings P, V, W, X, Y, Z)

This is the largest behavioral change but it is purely additive to the agent
flow — it just removes noise. Test by running the existing session smoke tests
and inspecting the resulting `latest_context.json` size.

### Files affected
- `src/autoresearch/controller/context.py`
- `src/autoresearch/controller/handoff.py`
- `src/autoresearch/controller/session.py`

### Required changes

#### 11a — Trim `build_llm_context`

In `controller/context.py`:

1. **Drop these keys entirely from the returned dict:**
   - `default_capping_diagnostics` (static after `prepare-data`; agent doesn't
     need it per cycle)
   - `latest_session_summary` (redundant with `latest_cycle_result`)
   - `recent_sessions` (operational state, not modelling context)
   - `research_log_tail` (agent has the file path; saves ~1 KB)
   - `champion_history` (the official_champion record already carries the
     active branch + reason; full history is on disk if needed)

2. **Replace `agent_schema` with a compact summary.** Add a private helper:
   ```python
   def _compact_agent_schema(schema):
       if not schema:
           return None
       return {
           "row_count": schema.get("row_count"),
           "columns": [
               {"name": c["name"], "role": c["role"]}
               for c in schema.get("columns", [])
           ],
       }
   ```
   Use this in place of the full `agent_schema` object. The per-column
   `dtype`, `unique_count`, `missing_count` fields are dropped — AGENT.md
   already documents them in prose.

3. **Flatten `latest_cycle_result`:** keep only
   `{completed_at, proposal_id, experiment_id, comparison_id, decision}` (drop
   the nested `official_champion` — it's already at the top level).

4. **Compact `recent_proposals`:** the helper `_compact_proposals` is fine; do
   not include the full `change_summary` if it is longer than 200 chars —
   truncate with `"…"`.

#### 11b — Slim `export_context_bundle`

In `controller/handoff.py`:

1. Stop writing `current_champion_summary.json`, `recent_comparisons_summary.json`,
   `recent_branch_summary.json`. The agent reads `latest_context.json`; these
   three side files duplicate its contents. Remove the `write_json` calls and
   their entries in the returned dict.

2. Stop writing `context_<stamp>.json` (the timestamped snapshot). Keep only
   `latest_context.json`. Rationale: the timestamped snapshots accumulate
   under `artifacts/.../context/` (the user's repo already has many) without
   being read by anything. If history is wanted, the git log of
   `latest_context.json` provides it.

3. The function's returned dict shrinks to:
   ```python
   {"latest_context_json": latest_context, "latest_handoff_markdown": latest_handoff}
   ```
   Update all callers (`cli.py`, `bootstrap.py`, `session.py`, `workflow.py`)
   that index into the old keys.

#### 11c — Shrink the handoff Markdown

In `controller/handoff.py`, rewrite `render_handoff_markdown` so it is ≤30
lines:

```python
def render_handoff_markdown(config, context):
    champion = context.get("official_champion") or {}
    return "\n".join([
        "# Auto-Research Handoff",
        "",
        "Read `AGENT.md` for the full operating manual.",
        "",
        f"- Current champion: `{champion.get('champion_id')}` "
        f"(branch `{champion.get('branch_id')}`)",
        f"- Context JSON: `{config.handoff_context_dir / 'latest_context.json'}`",
        f"- Proposal schema: `{config.handoff_handoffs_dir / 'proposal_schema.json'}`",
        f"- Proposal template: `{config.handoff_handoffs_dir / 'proposal_template.json'}`",
        f"- Write proposal JSON + model script to: `{config.handoff_proposal_inbox_dir}`",
        "",
        "Then run `autoresearch run-session-cycle`.",
    ]) + "\n"
```

All exploration-philosophy and constraints text is removed — it lives in
`AGENT.md` only.

#### 11d — Delete `proposal_instructions.md`

In `controller/handoff.py`:

1. Delete the `render_proposal_instructions` function.
2. Remove `instructions_path.write_text(...)` from `write_proposal_template`.
3. Remove `"proposal_instructions": instructions_path` from the returned dict
   and from all callers that index into it.
4. Delete any existing `proposal_instructions.md` files under
   `artifacts/auto_research/handoffs/` and `artifacts/tracks/*/handoffs/`
   (one-off cleanup; `git rm` what's tracked, `rm` what isn't).

#### 11e — Export context once per session cycle

In `controller/session.py`:

1. In `run_session_cycle`, remove every call to `export_context_bundle(config)`
   inside the function body **except the final one** at the very end (just
   before `return state`).
2. In `pause_session`, `resume_session`, `stop_session`,
   `create_session`, `_record_waiting`: keep one `export_context_bundle` call
   each — those are state-boundary events where re-exporting is cheap and
   semantically meaningful.
3. In `run_next_queued_proposal` (in `workflow.py`), the existing flow does
   not export — leave as is; the session wrapper handles it.

#### 11f — Verify and document the new context shape

1. Add a test `tests/test_context_shape.py`:
   - Builds a `ProjectConfig` with a small fixture registry.
   - Calls `build_llm_context(config)` and asserts the top-level keys are
     exactly:
     ```
     {"project_goal", "official_champion", "recent_experiments",
      "recent_comparisons", "recent_proposals", "proposal_count",
      "latest_cycle_result", "latest_nonpromotion_summary",
      "agent_schema", "allowed_search_space", "evaluation_rules"}
     ```
   - Asserts `len(json.dumps(context)) < 6000` for a fresh registry (the
     real number after the trim should be well under 6 KB).

2. Update `AGENT.md` "Session start" section: remove the
   `cat artifacts/auto_research/handoffs/proposal_instructions.md` step (the
   file is gone). Keep the `latest_context.json` and `RESEARCH_LOG.md` reads.

### Post-conditions
- `latest_context.json` is ≤ 6 KB on a fresh track, ≤ 8 KB after many
  experiments.
- `latest_handoff.md` is a short pointer file.
- `proposal_instructions.md` no longer exists anywhere in the project.
- Per cycle, the bundle is written exactly once during steady-state runs.

---

## Section 12 — Refactor `cli.py` to dict dispatch (Finding Q)

**File:** `src/autoresearch/cli.py`

### Required changes

1. Move each `if args.command == "X": ... return 0` block into a small
   module-level function named `_cmd_X(config, args) -> int` in `cli.py`
   (do not scatter them across other modules — keep one file). Each function
   returns the exit code.

2. Build a dispatch table at module scope:
   ```python
   COMMANDS = {
       "prepare-data": _cmd_prepare_data,
       "bootstrap-track": _cmd_bootstrap_track,
       # ...
   }
   ```

3. `main` becomes:
   ```python
   def main(argv=None):
       parser = build_parser()
       args = parser.parse_args(argv)
       config = load_config(args.config, track_id=args.track, run_id=args.run_id)
       handler = COMMANDS.get(args.command)
       if handler is None:
           parser.error(f"Unknown command: {args.command}")
           return 2
       return handler(config, args)
       ```

4. Remove every `import json` that was inlined inside a branch — put one
   `import json` at the top of the file.

5. Keep the user-visible CLI surface identical (subcommands, flags, output
   format) — only the internal structure changes.

6. Make sure all subcommands removed in Section 7 (`generate-proposal`,
   `run-cycle`, `run-cycles`) are also absent from `build_parser` and
   `COMMANDS`.

### Post-conditions
- `cli.py` is shorter and each subcommand is independently testable.
- `pytest -q` still passes; CLI integration tests (if any) still pass.

---

## Section 13 — Split `registry.py` per-table (Finding R)

**File:** `src/autoresearch/experiment_registry/registry.py` →
new package `src/autoresearch/experiment_registry/`.

⚠ This file is on the protected-integrity manifest. After Section 13 lands,
the user must run `autoresearch update-integrity-manifest` once. Do NOT
silently regenerate the manifest from inside the refactor — leave that as a
manual user step and call it out in the section's PR description.

### Required changes

1. Create the following modules under `src/autoresearch/experiment_registry/`:
   - `schema.py` — `CREATE TABLE` SQL strings and `init_registry`,
     `registry_counts`.
   - `experiments.py` — `record_experiment`, `get_experiment`,
     `list_experiments`, `list_artifacts`, `record_experiment_artifacts`.
   - `proposals.py` — `record_proposal`, `update_proposal_status`,
     `next_queued_proposal`, `list_proposals`, `get_proposal`.
   - `branches.py` — `upsert_branch`, `list_branches`.
   - `champions.py` — `set_official_champion`, `get_official_champion`,
     `list_champion_history`.
   - `comparisons.py` — `record_comparison`, `list_comparisons`.
   - `sessions.py` — `upsert_session`, `list_sessions`, `record_session_event`,
     `list_session_events`.

2. The package's `__init__.py` re-exports every function that was previously
   importable from `autoresearch.experiment_registry.registry`. Specifically:
   ```python
   from autoresearch.experiment_registry.experiments import (
       record_experiment, get_experiment, list_experiments,
       list_artifacts, record_experiment_artifacts,
   )
   # ...etc
   from autoresearch.experiment_registry.schema import (
       init_registry, registry_counts,
   )
   ```
   And ALSO keep a shim module `experiment_registry/registry.py` that
   re-exports the same symbols, so existing
   `from autoresearch.experiment_registry.registry import X` imports continue
   to work without code changes elsewhere.

3. Each split module owns only the SQL and Python for its table(s). Shared
   helpers (`_connect`, `_row_to_dict`, JSON serialisation) live in a private
   `_common.py`.

4. Function signatures, argument names, and return shapes must be byte-for-byte
   identical. The only allowed change is internal organisation.

5. Run `pytest --tb=short -q`. If any test fails, the refactor broke a public
   signature — fix the refactor, not the test.

### Post-conditions
- No file in `experiment_registry/` exceeds ~250 lines.
- All existing import paths still work.
- The user is told (in the PR notes / commit message) to run
  `autoresearch update-integrity-manifest`.

---

## Section 14 — Use Parquet for prediction artifacts (Finding U)

### Files affected
- `src/autoresearch/experiment_runner.py`
- `src/autoresearch/comparison_runner.py`
- `src/autoresearch/milestone.py`
- `src/autoresearch/controller/workflow.py` (which reads `predictions.csv` in
  `_validate_attempt_outputs`)
- Any test that reads `predictions.csv`

### Required changes

1. In `experiment_runner.run_experiment`:
   - Change `predictions_path = run_dir / "predictions.csv"` →
     `predictions_path = run_dir / "predictions.parquet"`.
   - Replace `result.predictions.to_csv(predictions_path, index=False)` with
     `result.predictions.to_parquet(predictions_path, index=False)`.
   - The artifact-registry key stays `"predictions"`; downstream code looks up
     by key, not extension.

2. Everywhere a `predictions.csv` artifact path is read with `pd.read_csv`,
   switch to `pd.read_parquet`:
   - `comparison_runner.compare_experiments`
   - `comparison_runner.run_repeated_evaluation`
   - `milestone._run_evaluation`
   - `controller/workflow._validate_attempt_outputs`
   - `tracks._load_predictions`

3. Keep `split_metrics.csv` as CSV (it's tiny and human-inspectable).

4. Verify `pyproject.toml` already pulls in `pyarrow` (the data pipeline uses
   parquet for the agent dataset, so it should). If not, add it to the
   `[project] dependencies` array.

5. Update any test fixture or assertion that hard-codes `"predictions.csv"`.

### Post-conditions
- Per-experiment prediction artifacts are ~10× smaller on disk.
- All readers use `read_parquet`.
- No `predictions.csv` file is written anywhere by the framework.

---

## Section 15 — Cleanup imports and dead code sweep

After Sections 1–14 are merged, do one cleanup pass:

1. Run `grep -rn "current_champion_id" src/ tests/` — should be empty.
2. Run `grep -rn "MockProposer\|OpenAIProposer\|AnthropicProposer\|FileHandoffProposer" src/ tests/ docs/ README.md` — should be empty.
3. Run `grep -rn "generate-proposal\|run-cycle\b\|run-cycles" src/ tests/ docs/ README.md` — should only match `run-session-cycle` and `run-session-cycles`.
4. Run `grep -rn "tweedie_glm\|tweedie_gbm\|frequency_severity_glm\|regularized_linear" src/ configs/ tests/` — should only match strings inside test fixtures that exercise the open-registry fallback (if any) and possibly proposal-validation tests. No production code path should reference them.
5. Run `grep -rn "predictions.csv" src/ tests/` — should be empty.
6. Run `python -c "import autoresearch.cli; autoresearch.cli.build_parser().parse_args(['--help'])"` — should exit cleanly.
7. Run `pytest --tb=short -q` — must pass.
8. Run `autoresearch --track sanitycheck --run-id S1 bootstrap-track` end-to-end on a fresh checkout (in a scratch directory) and verify:
   - `latest_context.json` ≤ 6 KB.
   - `latest_handoff.md` ≤ 1 KB.
   - No `proposal_instructions.md` file is written.
   - Bootstrap completes with the global_mean baseline as champion.

---

## Section 16 — Priority recap

Implement strictly in this order. Each section is a separate commit
(or PR) so the user can review incrementally:

1. Section 1 — Bug F (milestone scripted models)
2. Section 2 — Bug G (champion polarity)
3. Section 3 — Bug H (cross-track label)
4. Section 4 — Bug I (None→"" in TOML)
5. Section 5 — Bug J (whitelist tightening)
6. Section 6 — Single research-log format
7. Section 7 — Delete API proposers (largest deletion; do before Section 8 so
   that proposal validator simplifications in Section 8 don't conflict)
8. Section 8 — Delete dead model families
9. Section 9 — README / docs accuracy
10. Section 10 — Milestone dispatch refactor (depends on Sections 1 + 8)
11. Section 11 — Context bundle slim-down
12. Section 12 — CLI dict dispatch
13. Section 13 — Registry split (requires manifest update afterward)
14. Section 14 — Parquet for predictions
15. Section 15 — Cleanup sweep

After Section 13 the user must run
`autoresearch update-integrity-manifest`. Surface this in the commit message
and PR description.

---

## Out of scope

These items from the original review are intentionally NOT in this spec
(the user is keeping them as-is):

- **Section S** of the review: HTML report refactor in
  `src/autoresearch/reporting/comparison.py`. Leave the 799-line file alone.

Everything else in the original review is covered above.
