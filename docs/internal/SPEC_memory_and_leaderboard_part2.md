# Specification (Part 2) — Memory: P1 fixes + P2–P5 to completion

**Audience:** the implementing engineer (Claude Sonnet) continuing work in the
`Insurance_AutoResearch` repo.
**Repo root:** `/Users/alexhood/Documents/Insurance_AutoResearch`
**Prerequisite reading:** `docs/SPEC_memory_and_leaderboard.md` (the original
five-phase spec) — this document does **not** restate the architecture, the hard
constraints (§0), or the locked decisions. Re-read §0 of the original; every
constraint there still binds (run isolation sacrosanct, search-split-only,
read-only harvester, no holdout leakage, no new heavyweight deps, no emojis, wire
all CLI into `docs/CLI.md`).
**Status of P1:** landed in commit `417d68e` (branch `feature/memory-p1`).
Reviewed. Two correctness bugs found (Part A below). P2–P5 remain unbuilt.

This spec has two parts:
- **Part A** — fix the two P1 bugs + finish the loose ends found in review.
- **Part B** — implement P2, P3, P4, P5 to fully complete the functionality.

Land Part A first as its own commit; it changes already-shipped behaviour.

---

# Part A — P1 fixes (do these first)

## A1. Train-split contamination in harvested `gini_weighted` (must fix)

**Symptom.** `harvester._read_metrics` (`src/autoresearch/memory/harvester.py`,
~lines 38–68) averages `gini_weighted` over every split *except* those literally
named `milestone_holdout`/`holdout`. Real `metrics.json` files also contain a
**`train`** split, and it is currently averaged in. Verified on a real run:

```
per-split gini:    {'search_validation': 0.343, 'train': 0.457}
harvester reports: 0.400      <- (0.343 + 0.457) / 2   WRONG
true search gini:  0.343
```

This inflates every `gini_weighted`, every `peak_gini`, and the whole leaderboard
with in-sample training performance — the exact optimism the harness exists to
prevent. It violates the original spec's "search-split metrics only" rule (via
train leakage rather than holdout leakage).

**Fix.** `metrics.json` carries a top-level `"ordinary_eval_splits"` list (e.g.
`["search_validation"]`). Restrict the average to those splits:

```python
eval_splits = set(data.get("ordinary_eval_splits") or [])
search_ginis = [
    sm["gini_weighted"]
    for sm in split_metrics
    if "gini_weighted" in sm
    and (sm.get("split") in eval_splits if eval_splits
         else sm.get("split") not in ("milestone_holdout", "holdout", "train"))
]
```

Keep the holdout/train names as a defensive fallback for the (rare) case where
`ordinary_eval_splits` is absent.

**Test.** Extend `tests/test_memory_harvester.py::test_metrics_read_skips_holdout_split`
(or add a sibling) so the fixture `split_metrics` includes **all three** of
`search_validation`, `train`, and `milestone_holdout`, with distinct ginis, and
asserts the harvested `gini_weighted` equals the `search_validation` value
exactly — proving both train and holdout are excluded.

## A2. Deviance-as-gini fallback (should fix)

**Symptom.** Same function: when no eval gini is found, `gini_weighted` falls
back to `mean_score`. But `mean_score` is a Tweedie deviance (lower-is-better),
not a gini — wrong scale and wrong polarity. It pollutes `peak_gini`.

**Fix.** Return `None` for `gini_weighted` when no eligible split gini exists. Do
not substitute `mean_score`. Add a test asserting `None` in that case.

## A3. Wire the every-5-cycle checkpoint (must fix — it is currently dead code)

`maybe_memory_checkpoint` is defined in `src/autoresearch/memory/__init__.py` but
**never called**, so auto-harvest does not happen. Wire it into
`src/autoresearch/controller/session.py`, in `run_session_cycle()`, right after
`state["current_cycle"] += 1` (~line 169) and before the existing
`export_context_bundle(config)` near the end of the function (~line 200):

```python
if state["current_cycle"] % 5 == 0:
    from autoresearch.memory import maybe_memory_checkpoint
    maybe_memory_checkpoint(config, state)
```

It is already wrapped no-op-safe (catches and logs). In Part B this same wrapper
gains the reflection-prompt and playbook-regen steps, so all three fire on one
cadence. Add `tests/test_memory_checkpoint.py`: the checkpoint fires on cycle
multiples of 5, never raises into the loop even if the aggregator path is
unwritable, and is a no-op when the run manifest lacks identity.

## A4. Loose ends from review (should fix)

- **`assert_no_holdout_columns` is never exercised.** Add a test in
  `tests/test_memory_store.py` that builds a store and calls it, asserting it
  passes on the real schema (and fails on a deliberately poisoned table).
- **`peak_gini` ignores status.** Filter the `peak_gini` computation in
  `harvest_run` to experiments with `status='completed'` so a failed experiment
  with a stray metric cannot top the leaderboard.
- **`n_promotions` proxy.** Prefer counting from `champion_history` (which the
  harvester already opens via `_fetch_final_champion`) rather than comparison
  `decision='promote'` rows, so reverted promotions are not double-counted. If
  this is awkward, leave as-is but add a code comment noting the proxy.

**Part A acceptance:** the new/updated tests pass; re-running `memory harvest
--all` on real runs yields `peak_gini` equal to the true search-validation gini
(≈0.343 for the codex run above, not 0.400); full `pytest` stays green.

---

# Part B — Complete the functionality (P2–P5)

Build in order. Each phase is its own commit. Re-read the corresponding section
of the original spec for intent; the notes below give the concrete integration
points discovered during P1 review.

## P2 — Leaderboard + score-trace visuals (dashboard only, read-only)

Add a **"Memory & Leaderboard"** page to `src/autoresearch/dashboard/app.py`.
Follow the existing pattern there: a new `render_memory()` function plus a nav
entry (the file already uses `st.title`/`st.subheader`/`st.dataframe`; charts via
`st.line_chart` or the bundled `altair` — no new dependency). All reads come from
`artifacts/memory/memory.sqlite`. **No access gate** — this is the operator's
view, not the agent's.

1. **Score-trace chart.** x = `cycle_index`, y = running max of `gini_weighted`,
   one line per `model_id` (overlay all). Compute the running max in pandas from
   the `experiments` table grouped by `run_uid`, then aggregate to `model_id`.
   Per-model show/hide toggle.
2. **Four ranked tables** (the locked leaderboard dimensions):
   - **Peak quality** — `max(gini_weighted)` per model across its runs.
   - **Efficiency** — `peak_gini / n_experiments`; also a `/sum(fit_wall_seconds)`
     variant. Higher is better.
   - **Time-to-structural-insight** — first `cycle_index` where the model's
     running-max gini crossed a configurable threshold (default **0.37**, the
     rate-Tweedie plateau-escape band). Lower is better; "never reached" sorts
     last. Put the threshold in `[memory]` config (see note below) so it is not a
     magic number.
   - **Decision quality** — fraction of promotions not later reverted (from
     `champion_history`) and sub-noise-thrash rate (rejected comparisons where
     `abs(mean_lift) < 2 * std_lift`, both already in the `comparisons` table).
   Each row: model, provider, n_runs, n_experiments, the metric.

   Add a `[memory]` section to `configs/default.toml` (e.g.
   `structural_gini_threshold = 0.37`, `memory_store_relpath = "artifacts/memory/memory.sqlite"`)
   and surface it on `ProjectConfig` for the dashboard and P5 to read.

**P2 acceptance:** the page renders all four boards and the overlaid score-trace
from real harvested data after `memory harvest --all`.

## P3 — Evidence-bound self-reflection insights

New module `src/autoresearch/memory/insights.py`:

- `validate_insight(run_registry_path, insight_dict) -> (verified: bool, note: str)`
  — opens the run registry **read-only** (reuse the `mode=ro` pattern and the
  `_guard_path` guard from `harvester.py`). For every id in
  `evidence_json.experiment_ids` / `comparison_ids`, confirm it exists in that
  registry; for the cited `metric`/`delta`, confirm it matches the registry value
  within tolerance (e.g. `abs(claimed - actual) <= max(1e-6, 0.05*abs(actual))`).
  Return `(False, note)` on any mismatch, naming the failing id/field.
- `record_insight(memory_path, run_uid, model_identity, insight_dict)` — validates
  then upserts into the `insights` table (schema already exists in `store.py`).
  Sets `verified` from the validator and stores `verification_note`. Computes
  `insight_id` (stable hash of run_uid + claim + sorted evidence ids).

CLI: `autoresearch memory record-insight --file <path.json>` and
`autoresearch memory list-insights [--verified-only] [--run <run_uid>]`. The
insight JSON schema:

```json
{
  "claim": "rate-based Tweedie GBMs plateau ~0.33; total-target trees reach ~0.40",
  "scope": "general",
  "confidence": 0.8,
  "evidence": {
    "experiment_ids": ["..."],
    "comparison_ids": ["..."],
    "metric": "gini_weighted",
    "delta": 0.07
  },
  "supersedes": null,
  "contradicts": null
}
```

`record-insight` reads `model_identity` and `run_uid` from the current run's
`run_manifest.json` (same pattern as `_run_checkpoint`).

**Capture point.** Extend `maybe_memory_checkpoint` (the wrapper wired in A3) so
that on each checkpoint it ALSO writes a `pending_reflection.md` into the run's
handoff dir (`config.handoff_handoffs_dir`) prompting the agent to record 0–3
evidence-bound insights via `record-insight`. Do **not** block the loop on it —
capture is best-effort; the validator is what gates trust. Queries default to
`verified=1` unless `--include-unverified` is passed.

**P3 acceptance:** an insight with valid evidence lands `verified=1`; one with a
fabricated `delta` lands `verified=0` with a descriptive note; both appear in
`memory list-insights`; a reflection prompt file is produced at the checkpoint.

## P4 — Access gate + query/analysis tool

**Gate.** Read `AUTORESEARCH_MEMORY_ACCESS` env var: `none` (default) | `own` |
`all`. Resolve it once at run launch (in `bootstrap_track` and `start-session`)
and record the resolved value into `run_manifest.json` (`memory_access` key) so it
is auditable. Provide a helper `resolve_memory_access(config) -> str` in
`src/autoresearch/memory/__init__.py` that reads the env var, falling back to the
manifest value, defaulting to `none`. The agent cannot set its own env var, so it
cannot self-escalate — this mirrors the holdout-token model.

**Query tool.** `autoresearch memory query` with two modes (both respect the gate;
`none` -> refuse with a clear message; `own` -> filter to the run's own
`model_id`; `all` -> all models, fully attributed):

- **Retrieval:** `--insights [--family X] [--target-strategy Y] [--model Z]
  [--verified-only|--include-unverified]`, and `--experiments --filter ...`.
  Returns compact JSON/markdown rows.
- **Analytical:** named canned analyses run as SQL aggregations over the
  aggregator, each returning a small table:
  `--analysis peak-gini-by-framing` (group by target_strategy/model_family),
  `--analysis plateau-families` (families whose max gini stays under the
  structural threshold), `--analysis biggest-single-jumps` (largest
  cycle-over-cycle running-max gini increases), `--analysis efficiency-by-model`.

**Context integration.** In `build_llm_context()`
(`src/autoresearch/controller/context.py:18`): when `resolve_memory_access != none`,
add a `memory_access` block describing scope (`own`/`all`), how to invoke the
query tool, and a pointer to the playbook (P5). Do **NOT** auto-dump raw insights
into context — the agent pulls on demand. **Critical regression test:** with the
env unset (`none`), the dict returned by `build_llm_context()` and the exported
`latest_context.json` must be byte-for-byte identical to today (assert against a
saved baseline). This is the isolation guarantee.

**P4 acceptance:** env unset -> `query` refuses and context is unchanged from
baseline; `own` -> only own-model rows; `all` -> attributed cross-model rows;
analytical queries return correct aggregations.

## P5 — Regenerated dynamic playbook

`autoresearch memory build-playbook` compiles **verified** insights +
leaderboard-derived facts into `artifacts/memory/playbook/latest.md`, plus a
timestamped copy under `artifacts/memory/playbook/`. Sections: "What works",
"What plateaus / known ceilings", "Highest-leverage moves" — each bullet citing
its evidence (experiment/comparison ids) and source `model_id` (attribution is
on, per the locked decision). Only `verified=1` insights are included.

**Regeneration.** Extend `maybe_memory_checkpoint` once more: after harvest +
reflection prompt, if any new verified insight landed since the last build,
regenerate the playbook.

**Injection.** When `resolve_memory_access` is `own` or `all`, the handoff
bundle (`render_handoff_markdown` / `export_context_bundle` in
`controller/handoff.py`) links the playbook. For `own`, generate and link a
filtered own-model variant; for `all`, the full attributed playbook. When access
is `none`, the handoff is unchanged (assert in a test).

**P5 acceptance:** playbook regenerates from the store, contains only verified
attributed insights, and is referenced in the handoff only when access is granted.

---

# Cross-cutting

- **Testing.** Each phase adds its own `tests/test_memory_*.py`. The full suite
  (`pytest`) must stay green at every commit; the existing integrity tests
  (`test_integrity_whitelist`, `test_holdout_vault`, `test_context_shape`) are
  the canaries for isolation/leak regressions — never let them go red.
- **Do not touch** the promotion gate, metrics, resampling, or holdout mechanics.
  Prefer new modules over editing integrity-protected files; if a protected file's
  SHA changes intentionally, run `update-integrity-manifest`.
- **docs/CLI.md** must list every new subcommand (`record-insight`,
  `list-insights`, `query`, `build-playbook`) with examples.
- **Commit discipline.** One commit for Part A, then one per phase
  (P2, P3, P4, P5). Report after each: what changed, new CLI surface, test
  results, and any judgement calls or spec ambiguities.
