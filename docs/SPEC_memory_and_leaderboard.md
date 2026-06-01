# Specification — Cross-Run Memory, Leaderboard & Dynamic Playbook

**Audience:** an implementing engineer (Claude Sonnet) working in the
`Insurance_AutoResearch` repo.
**Repo root:** `/Users/alexhood/Documents/Insurance_AutoResearch`
**Status:** ready to implement. Five phases (P1–P5); each lands independently
and is independently valuable. P1–P2 add zero agent-facing surface.

---

## Background / context

This is an autonomous insurance burning-cost modelling loop on the French Motor
dataset (~678K policies). An LLM agent proposes model scripts; the framework
fits them, evaluates with a `cv_bootstrap` promotion gate, and the agent records
a promote/reject decision. Multiple agent runs execute the same loop, each fully
isolated under `artifacts/tracks/<track>/runs/<run-id>/registry.sqlite`. An
agent's entire worldview is built **only** from its own run's registry via
`build_llm_context()` (`src/autoresearch/controller/context.py`), exported to
`latest_context.json` + `latest_handoff.md`. There is currently **no** cross-run
or cross-model knowledge flow — that isolation is deliberate.

We want to accumulate experiment results, metadata and learnings across runs so
that (a) we can analyse which research strategies work, (b) future agents can —
**only when explicitly granted** — learn from prior runs, and (c) we can
visualise score-improvement traces and rank models by research ability.

### Empirical motivation (from prior runs)

- Rate-based Tweedie GBMs plateau at Gini ~0.33; tree ensembles on the
  claim-cost total reach ~0.40 on the same data/metric. Models repeatedly
  mistook this **paradigm ceiling for a data ceiling** and burned 50–70
  post-plateau experiments. A queryable memory of "what framing breaks which
  ceiling" is the highest-leverage thing we can give a future agent.
- `model_identity` is currently unreliable: many proposals are recorded as
  `manual_file` with no model field, so we cannot cleanly attribute results to a
  model. This spec makes identity mandatory.

---

## 0. Hard constraints (apply to every section)

- **Run isolation is sacrosanct.** Nothing in this spec may let an agent read
  another run's registry directly, or change what a non-granted run sees. With
  no grant, the exported context bundle MUST be byte-for-byte identical to today.
- **No holdout leakage.** The aggregator and everything an agent can query MUST
  contain **search-split metrics only**. Any field derived from the holdout
  vault or milestone reports (`data/holdout_vault/**`,
  `artifacts/milestone_reports/**`, anything gated by
  `AUTORESEARCH_MILESTONE_TOKEN`) MUST be excluded at harvest time. Add a test
  that asserts the aggregator schema has no holdout-derived columns and that the
  harvester never opens vault/milestone paths.
- **The harvester is read-only** with respect to per-run registries — it opens
  them `mode=ro` (`sqlite3.connect("file:...?mode=ro", uri=True)`) and never
  writes to them.
- **Do not modify the integrity-protected files** (`metrics.py`,
  `resampling.py`, `holdout_vault.py`, `registry.py` core) in ways that change
  their SHA256 unless intended; if you must, run `update-integrity-manifest`.
  Prefer adding new modules over editing protected ones.
- **No new heavyweight dependencies.** Use the stack already present (`pandas`,
  `sqlite3`, `streamlit`; charts via `st.line_chart`/`altair` already shipped
  with streamlit). No new ORM.
- **No emojis** in any new file.
- All new commands must be wired into `src/autoresearch/cli.py` and listed in
  `docs/CLI.md`.

---

## Decisions already made (do not re-litigate)

| Dimension | Decision |
|---|---|
| Identity unit | Actual LLM model (`provider/model/version`), mandatory at run start + backfill |
| Captured | Auto experiment metadata + agent-self-authored, evidence-bound insights |
| Insight author | Research agent (self-reflection), validated against the registry |
| Access surface | Queryable tool (retrieval + analytical), gated |
| Default | Hidden. Grants: `own` (this model's history) or `all` (everyone, fully attributed) |
| Leaderboard | Four boards: peak quality, efficiency, time-to-insight, decision quality |
| Storage | Separate aggregator store under `artifacts/memory/` |
| Population | Automatic harvest + insight prompt **every 5 cycles** |
| Grant enforcement | Env var, mirroring the holdout-vault token |
| Anti-leak | Search-split only |
| Attribution | Fully attributed across models |
| Dynamic skill | Regenerated playbook artifact, injected only when access granted |

---

## Architecture

```
 Per-run registries (ISOLATED, unchanged)
 artifacts/tracks/<track>/runs/<run-id>/registry.sqlite
        |  (read-only, mode=ro)
        v
 +-------------------------+   every 5 cycles, from the session loop
 |  Harvester (read-only)  | <-- autoresearch memory harvest
 +-------------------------+
        | upsert (search-split metrics only; holdout stripped)
        v
 +-------------------------------------------------+
 |  Aggregator store   artifacts/memory/           |
 |   memory.sqlite  (models, runs, experiments,    |
 |                   comparisons, insights)        |
 |   insights/*.md  (evidence-bound, attributed)   |
 |   playbook/latest.md  (regenerated digest)      |
 +-------------------------------------------------+
        |                               |
   env-gated read                   dashboard (read-only)
        v                               v
 Query/analysis tool             Leaderboards + score-trace visuals
 (exposed only if granted)
```

---

## P1 — Identity + aggregator substrate (no agent-facing change)

### 1.1 Model identity capture

Add a required model-identity input at run start.

- New CLI args on `bootstrap-track` and `start-session`:
  `--model-provider`, `--model-name`, `--model-version` (version optional),
  plus optional `--harness` (e.g. `claude-code`, `codex`, `opencode`).
- `bootstrap_track()` (`src/autoresearch/bootstrap.py`) MUST refuse to proceed
  if provider+name are absent (raise a clear `ValueError` instructing the user).
- Persist identity into `run_manifest.json` (written in
  `src/autoresearch/config.py` around line 270 where the manifest is created).
  Add a `model_identity` object: `{provider, name, version, harness}`.
- Compute a stable `model_id` slug = `f"{provider}/{name}"` (lowercased,
  version excluded from the slug so the leaderboard groups versions under one
  model unless you opt to include it; keep version as a column).

### 1.2 Aggregator schema (`artifacts/memory/memory.sqlite`)

New module `src/autoresearch/memory/store.py` with an `init_memory_store(path)`
mirroring the style of `experiment_registry/schema.py`. Tables:

```sql
CREATE TABLE models (
    model_id TEXT PRIMARY KEY,         -- "provider/name"
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    first_seen TEXT, last_seen TEXT
);

CREATE TABLE runs (
    run_uid TEXT PRIMARY KEY,          -- f"{track_id}/{run_id}"
    model_id TEXT NOT NULL,
    track_id TEXT, run_id TEXT,
    version TEXT, harness TEXT,
    started_at TEXT, last_harvested_at TEXT,
    n_experiments INTEGER, n_promotions INTEGER,
    peak_gini REAL,                    -- search-split only
    final_champion_id TEXT,
    FOREIGN KEY (model_id) REFERENCES models(model_id)
);

CREATE TABLE experiments (
    experiment_uid TEXT PRIMARY KEY,   -- f"{run_uid}/{experiment_id}"
    run_uid TEXT NOT NULL,
    experiment_id TEXT,
    cycle_index INTEGER,               -- order within the run
    model_family TEXT, target_strategy TEXT, target_mode TEXT,
    features_json TEXT, hyperparameters_json TEXT,
    mean_score REAL, std_score REAL, gini_weighted REAL,  -- SEARCH SPLIT ONLY
    fit_wall_seconds REAL, compute_budget_seconds REAL, timed_out INTEGER,
    status TEXT,
    FOREIGN KEY (run_uid) REFERENCES runs(run_uid)
);

CREATE TABLE comparisons (
    comparison_uid TEXT PRIMARY KEY,   -- f"{run_uid}/{comparison_id}"
    run_uid TEXT NOT NULL,
    champion_id TEXT, challenger_id TEXT,
    mean_lift REAL, challenger_win_rate REAL,
    std_lift REAL,                     -- noise floor, if present in paired_summary
    decision TEXT, guardrail_status TEXT,
    created_at TEXT,
    FOREIGN KEY (run_uid) REFERENCES runs(run_uid)
);

CREATE TABLE insights (
    insight_id TEXT PRIMARY KEY,
    run_uid TEXT NOT NULL, model_id TEXT NOT NULL,
    created_at TEXT,
    claim TEXT NOT NULL,
    scope TEXT NOT NULL,               -- 'own_model' | 'general'
    confidence REAL,
    evidence_json TEXT NOT NULL,       -- {experiment_ids:[], comparison_ids:[], metric, delta}
    verified INTEGER NOT NULL DEFAULT 0,
    verification_note TEXT,
    supersedes TEXT, contradicts TEXT,
    FOREIGN KEY (run_uid) REFERENCES runs(run_uid)
);
```

Use `INSERT ... ON CONFLICT(...) DO UPDATE` (upsert) so harvest is idempotent.

### 1.3 Harvester

New module `src/autoresearch/memory/harvester.py`:

- `harvest_run(memory_path, run_registry_path, model_identity)` — opens the run
  registry **read-only**, reads `experiments`, `comparisons`, `champion_history`,
  derives `cycle_index` (order experiments by `created_at`), computes
  `peak_gini`/`n_promotions`, and upserts into the aggregator.
- `harvest_all(memory_path)` — discovers every
  `artifacts/tracks/**/runs/**/registry.sqlite`, reads each run's
  `run_manifest.json` for `model_identity` (skip + warn if missing), and calls
  `harvest_run`. This doubles as **backfill**.
- **Holdout guard:** the harvester must never open paths under
  `data/holdout_vault/` or `artifacts/milestone_reports/`, and must only select
  the search-split metric columns. Add a unit test enforcing this.

### 1.4 Backfill command

`autoresearch memory backfill-identity` — interactive/argument-driven helper to
write a `model_identity` block into existing `run_manifest.json` files for the
handful of historical runs (`deepseek_v4_flash`, `nemotron-3-super`,
`opencode/runs/20260531T093227Z` = Mimo, `claude`, `codex`, …). Then
`autoresearch memory harvest --all` populates the store from history.

### 1.5 CLI

Add a `memory` subcommand group: `harvest [--all]`, `backfill-identity`,
`status` (counts per table). Wire into `cli.py` and document in `docs/CLI.md`.

**P1 acceptance:** `memory harvest --all` builds `artifacts/memory/memory.sqlite`
with one `models` row per distinct model and correct per-run `peak_gini`;
holdout-exclusion test passes; no change to any agent-facing output.

---

## P2 — Leaderboard + score-trace visuals (dashboard only, read-only)

Add a **"Memory & Leaderboard"** page to `src/autoresearch/dashboard/app.py`
(new render function + nav entry). All reads come from `memory.sqlite`. No access
gate — this is the operator's view, not the agent's.

### 2.1 Score-trace chart

Line chart: x = `cycle_index`, y = running peak `gini_weighted`, one line per
`model_id` (overlay all models). Use the champion progression (monotone running
max) so each line shows how that model climbed. Add a per-model toggle.

### 2.2 Four leaderboards (tabs or sections)

Compute from the aggregator; show as ranked tables:

1. **Peak quality** — `max(gini_weighted)` per model (best across its runs).
2. **Efficiency** — `peak_gini / n_experiments` (and a `/compute_seconds`
   variant from `fit_wall_seconds`). Rewards getting there fast.
3. **Time-to-structural-insight** — cycle index at which the model first
   exceeded a configurable structural threshold (default: first crossing of
   Gini 0.37, i.e. escaping the rate-Tweedie plateau band). Lower is better;
   "never" sorts last.
4. **Decision quality** — promotion-gate correctness proxy: fraction of
   promotions that were not later reverted + rate of sub-noise thrash
   (rejected comparisons where `abs(mean_lift) < 2*std_lift`). Lower thrash and
   higher correctness rank higher.

Each board shows model, provider, n_runs, n_experiments, and the metric.

**P2 acceptance:** dashboard page renders all four boards and the overlaid
score-trace from real harvested data.

---

## P3 — Evidence-bound self-reflection insights

### 3.1 Insight schema + validator

New module `src/autoresearch/memory/insights.py`:

- `validate_insight(memory_path, run_registry_path, insight_dict)` — checks that
  every `experiment_ids`/`comparison_ids` in `evidence_json` exists in the run's
  registry and that the cited `metric`/`delta` matches the registry value within
  tolerance. Returns `(verified: bool, note: str)`. Insights that fail are still
  stored but with `verified=0` and a note; queries default to `verified=1`.
- `record_insight(memory_path, run_uid, model_identity, insight_dict)` — validates
  then upserts into `insights`.

### 3.2 Capture point in the loop

Wire an insight prompt into the **every-5-cycle** checkpoint (see P1.3 / §6).
At cycle indices that are multiples of 5 (and at session end), the session loop:

1. runs `memory harvest` for the current run, and
2. writes a `pending_reflection.md` into the run's handoff dir prompting the
   agent to record 0–3 evidence-bound insights via a new
   `autoresearch memory record-insight --file <json>` command.

The reflection is part of the agent's normal turn (it already pauses in
`awaiting_decision` each cycle); do **not** block the loop on it — capture is
best-effort, validation gates trust.

**P3 acceptance:** an agent-recorded insight with valid evidence lands as
`verified=1`; one with a fabricated delta lands as `verified=0` with a note;
both visible in `memory status`.

---

## P4 — Access gate + query/analysis tool

### 4.1 Gate

Read `AUTORESEARCH_MEMORY_ACCESS` env var at run launch:
`none` (default) | `own` | `all`.

- Record the resolved value into `run_manifest.json` at bootstrap.
- The query tool and any context injection consult this value. With `none`, the
  query subcommands refuse (clear message) and `build_llm_context()` adds
  nothing. With `own`, results are filtered to the run's own `model_id`. With
  `all`, all models are visible, **fully attributed**.
- Enforcement parallels the holdout token: the agent cannot set its own env var,
  so it cannot self-escalate.

### 4.2 Query/analysis tool

`autoresearch memory query` with two modes:

- **Retrieval:** `--insights [--family X] [--target-strategy Y] [--model Z]
  [--verified-only]`, `--experiments --filter ...` — returns matching rows as
  JSON/markdown.
- **Analytical:** named canned analyses, e.g.
  `--analysis peak-gini-by-framing`, `--analysis plateau-families`,
  `--analysis biggest-single-jumps`, `--analysis efficiency-by-model`. Each runs
  a SQL aggregation over the aggregator and returns a small table. This is the
  "run analysis to draw conclusions" path; keep outputs compact.

All query output respects the gate (own vs all). Always exclude `verified=0`
insights unless `--include-unverified`.

### 4.3 Context integration

In `build_llm_context()`, when access != `none`, add a `memory_access` block
describing what's available and how to query it (tool usage hint), and — for
`own`/`all` — a short pointer to the playbook (P5). Do **not** auto-dump raw
insights into context; the agent pulls on demand. Add a test asserting that with
`none` the context dict is unchanged from baseline.

**P4 acceptance:** with the env unset, query refuses and context is unchanged;
with `own`, only own-model rows return; with `all`, attributed cross-model rows
return.

---

## P5 — Regenerated dynamic playbook

`autoresearch memory build-playbook` compiles **verified** insights +
leaderboard-derived facts into `artifacts/memory/playbook/latest.md` (versioned;
keep timestamped copies). Structure: "What works", "What plateaus / known
ceilings", "Highest-leverage moves", each bullet citing the evidence and source
model. Regenerate at the every-5-cycle checkpoint when any new verified insight
landed.

Injection: when `AUTORESEARCH_MEMORY_ACCESS` is `own` or `all`, the handoff
bundle links the playbook (and for `own`, a filtered own-model variant). This is
the "dynamic skill for future agents."

**P5 acceptance:** playbook regenerates from the store, contains only verified
attributed insights, and is referenced in the handoff only when access is granted.

---

## 6. The every-5-cycle checkpoint (shared hook)

Single integration point in `src/autoresearch/controller/session.py`, in
`run_session_cycle()` right after `state["current_cycle"] += 1` (~line 169) and
before `export_context_bundle(config)` (~line 200):

```python
if state["current_cycle"] % 5 == 0:
    maybe_memory_checkpoint(config, state)   # harvest + reflection prompt + playbook
```

`maybe_memory_checkpoint` lives in `src/autoresearch/memory/__init__.py`, is a
no-op-safe wrapper (never raises into the loop — wrap in try/except, log to
session_events), and does: harvest current run -> drop reflection prompt ->
(P5) regenerate playbook. Keep it cheap; harvest of one run is small.

---

## 7. Testing checklist

- `tests/test_memory_store.py` — schema init, upsert idempotency.
- `tests/test_memory_harvester.py` — harvest a fixture run; **assert no
  holdout/milestone path is ever opened** and no holdout column exists.
- `tests/test_memory_identity.py` — bootstrap refuses without identity; manifest
  contains `model_identity`.
- `tests/test_memory_insights.py` — valid evidence -> verified=1; bad delta ->
  verified=0.
- `tests/test_memory_access_gate.py` — `none` leaves context unchanged and query
  refuses; `own` filters; `all` attributes.
- `tests/test_memory_checkpoint.py` — checkpoint fires on multiples of 5 and
  never raises into the loop.

Run the full suite (`pytest`) before declaring any phase done; the existing
integrity tests must stay green.

---

## 8. File summary (new/changed)

New:
- `src/autoresearch/memory/__init__.py` (checkpoint wrapper)
- `src/autoresearch/memory/store.py`, `harvester.py`, `insights.py`,
  `query.py`, `playbook.py`
- `tests/test_memory_*.py`
- dashboard page (in existing `app.py`)

Changed:
- `src/autoresearch/cli.py` (memory subcommands + identity args)
- `src/autoresearch/bootstrap.py` (require identity)
- `src/autoresearch/config.py` (manifest: model_identity, memory_access)
- `src/autoresearch/controller/session.py` (checkpoint hook)
- `src/autoresearch/controller/context.py` (gated memory_access block)
- `docs/CLI.md`

Out of scope: changing the promotion gate, metrics, resampling, or holdout
mechanics in any way.
