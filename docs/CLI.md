# CLI Reference

All commands are invoked as `autoresearch [--track NAME] [--run-id ID] [--dataset NAME] <command>`.
Global flags `--track` and `--run-id` scope all artifact paths and the registry to
`artifacts/tracks/<NAME>/runs/<ID>/`. Use `--new-run` instead of `--run-id` when
starting a fresh timestamped run in a track.

`--dataset NAME` selects the registered dataset (`configs/datasets/*.toml`; see
`list-datasets`). It defaults to the dataset pinned in the run's manifest, or
`french_motor` for new work; passing a value that contradicts an existing run's
pinned dataset is an error. `--target-mode MODE` overrides the evaluation
target; it must be one of the active dataset's modes (e.g. `french_motor`:
`burning_cost` (default) / `frequency` / `severity`) and must be passed on every
command of a non-default-mode run.

---

## Data Preparation

### `prepare-data`

Build the data artifacts for the active dataset from its raw files in
`data/datasets/<name>/raw/`.

```bash
autoresearch prepare-data                          # french_motor (default)
autoresearch --dataset porto_seguro prepare-data
```

Writes under `data/datasets/<name>/`: `processed/agent_dataset_search.parquet`,
`holdout_vault/agent_dataset_holdout.parquet`, metadata (schema, profile,
capping diagnostics and — for sampled datasets — `sample_manifest.json`), and
the deterministic split pack under `splits/`.

### `list-datasets`

List registered datasets with display name, default target mode, available
modes, weight column, prepared status, and row count. The active dataset is
starred.

```bash
autoresearch list-datasets
```

---

## Registry & Bootstrap

### `bootstrap-track`

Idempotently prepare data, registry, the global-mean starting baseline, champion,
templates, and context for a named track. Bootstrap always runs exactly
`configs/experiments/global_mean.toml`; other experiment configs are ignored.
Safe to run at the start of every new session.

**Model identity is required.** Pass `--model-provider` and `--model-name` so results can be attributed in the cross-run memory aggregator.

```bash
autoresearch --track demo --new-run bootstrap-track \
  --model-provider anthropic \
  --model-name claude-sonnet-4-6 \
  --model-version 20251101 \
  --harness claude-code
```

Flags:

| Flag | Required | Description |
|---|---|---|
| `--model-provider` | Yes | LLM provider (e.g. `anthropic`, `openai`, `deepseek`) |
| `--model-name` | Yes | Model identifier (e.g. `claude-sonnet-4-6`, `gpt-4o`) |
| `--model-version` | No | Version string stored for reference |
| `--harness` | No | Agent harness name (e.g. `claude-code`, `codex`, `opencode`) |
| `--dataset` | No | Registered dataset for the run (default `french_motor`); pinned in `run_manifest.json` so later commands on the run resolve it automatically |
| `--skip-data` | No | Skip `prepare-data` even if shared data is missing |
| `--force-data` | No | Rebuild shared data artifacts before bootstrapping |
| `--skip-baselines` | No | Do not run the global-mean starting baseline if the registry is empty |
| `--cycles` | No | Pin the run's cycle budget: sessions opened without `--max-cycles` default to this, so the framework stops the run at N cycles instead of trusting the agent to count |
| `--enable-foundation-models` | No | Opt the run into foundation tabular estimators (e.g. `tabpfn`). Written to `run_manifest.json`; the estimator appears in the recipe menu/contract only when this is set **and** the `[foundation]` extra is installed (`pip install -e '.[foundation]'`; then `python scripts/setup_foundation_models.py`). See the Operating Manual "Foundation tabular models" section. |

Writes the operator-declared `model_identity` into `run_manifest.json`; registry,
the global-mean starting experiment, official champion, proposal templates, and
handoff context under the run directory. The first telemetry sync verifies this
identity against the model reported by the harness. A mismatch preserves the
original value as `model_identity_declared` and makes the observed identity
canonical. Multiple observed identities create `model_identity_conflict` and
block memory harvesting until the run is investigated.

### `init-registry`

Create the SQLite experiment registry for the current track and run.

```bash
autoresearch --track demo --run-id quickstart init-registry
```

Writes: `artifacts/tracks/demo/runs/quickstart/registry.sqlite`

### `list-tracks`

List all tracks that have a registry under `artifacts/tracks/`.

```bash
autoresearch list-tracks
```

---

## Baselines

### `run-baseline`

Run one deterministic baseline experiment from a TOML config.

This is a direct diagnostic command. It does not register a proposal, create a
champion comparison, or pause for a research decision. Use the proposal
workflow for ordinary research experiments.

```bash
autoresearch --track demo --run-id quickstart run-baseline configs/experiments/global_mean.toml
```

Writes: experiment artifacts under the run's `iterations/` directory.

### `run-all-baselines`

Run every TOML config under `configs/experiments/` directly. This command is
intended for diagnostics and compatibility checks, not autonomous research
cycles; it bypasses proposals, comparisons, and decisions.

Run all baseline configs under `configs/experiments/`.

```bash
autoresearch --track demo --run-id quickstart run-all-baselines
```

### `list-experiments`

Print a summary of all registered experiments for the current run.

```bash
autoresearch --track demo --run-id quickstart list-experiments
```

### `init-official-champion`

Initialise the official champion as the `global_mean` baseline. The official champion only changes through the promotion gate.

```bash
autoresearch --track demo --run-id quickstart init-official-champion
```

---

## Comparison & Promotion

### `run-repeated-evaluation`

Resample one experiment's search-time predictions for variance estimation.

```bash
autoresearch --track demo --run-id quickstart run-repeated-evaluation <experiment-id>
```

### `compare-experiments`

Compare a champion and challenger experiment directly.

```bash
autoresearch --track demo --run-id quickstart compare-experiments <champion-id> <challenger-id>
```

Writes: `comparison_report.html` and `promotion_report.json` under the run's `iterations/` directory.

### `compare-to-champion`

Compare a challenger against the current official champion.

```bash
autoresearch --track demo --run-id quickstart compare-to-champion <challenger-id>
```

Writes: comparison report and promotion decision to the run's `iterations/` directory.

### `record-decision`

Record the supervised verdict, structured outcome category, interpretation,
and next research direction for a pending comparison.

```bash
autoresearch --track demo --run-id quickstart record-decision <comparison-id> \
  --decision reject \
  --reason-code noise \
  --rationale "The apparent lift is not stable across folds." \
  --interpretation "This variant does not add reliable ranking signal." \
  --next "Rotate to a different target framing."
```

Reason codes: `clear_win`, `line_progress`, `noise`, `inferior`,
`artifact_suspected`, `calibration`, `other`.

### `list-promotions`

Print all volatility-aware comparison and promotion decisions.

```bash
autoresearch --track demo --run-id quickstart list-promotions
```

### `list-champion-history`

Print the official champion history for the current run.

```bash
autoresearch --track demo --run-id quickstart list-champion-history
```

### `list-branches`

Print branch lineage records for the current run.

```bash
autoresearch --track demo --run-id quickstart list-branches
```

---

## File-Handoff Workflow

### `export-context`

Export the file-based handoff context bundle for the current champion state.

```bash
autoresearch --track demo --run-id quickstart export-context
```

Writes: `context/latest_context.json` and related summaries under the run directory.

### `write-proposal-template`

Write the proposal template and schema files to the handoff directory.

```bash
autoresearch --track demo --run-id quickstart write-proposal-template
```

### `show-latest-handoff`

Print the latest handoff Markdown summary to stdout.

```bash
autoresearch --track demo --run-id quickstart show-latest-handoff
```

### `show-proposal-inbox-status`

Print the current inbox and processed-folder status as JSON.

```bash
autoresearch --track demo --run-id quickstart show-proposal-inbox-status
```

### `ingest-proposals`

Validate proposal files from the handoff inbox and enqueue valid ones.

```bash
autoresearch --track demo --run-id quickstart ingest-proposals
```

### `enqueue-proposal`

Validate and enqueue a specific proposal JSON file.

```bash
autoresearch --track demo --run-id quickstart enqueue-proposal path/to/proposal.json
```

### `run-next-proposal`

Run the next queued proposal through experiment, comparison, and promotion gate.

```bash
autoresearch --track demo --run-id quickstart run-next-proposal
```

### `run-latest-proposal-cycle`

Ingest the newest inbox proposal and run one complete gated cycle.

```bash
autoresearch --track demo --run-id quickstart run-latest-proposal-cycle
```

### `list-proposals`

Print the proposal queue status for the current run.

```bash
autoresearch --track demo --run-id quickstart list-proposals
```

### `inspect-proposal`

Print a single proposal record as JSON.

```bash
autoresearch --track demo --run-id quickstart inspect-proposal <proposal-id>
```

---

## Supervised Sessions

### `start-session`

Create a named supervised autonomous research session. If model identity was not written at `bootstrap-track` time, you can pass it here and it will be patched into `run_manifest.json`.

```bash
autoresearch --track demo --run-id quickstart start-session my-session --max-cycles 10
```

Optional identity flags (same as `bootstrap-track`): `--model-provider`, `--model-name`, `--model-version`, `--harness`.

### `session-status`

Inspect the current session state.

```bash
autoresearch --track demo --run-id quickstart session-status
```

### `pause-session`

Pause the current or specified session.

```bash
autoresearch --track demo --run-id quickstart pause-session
```

### `resume-session`

Resume a paused session.

```bash
autoresearch --track demo --run-id quickstart resume-session
```

### `stop-session`

Stop the current or specified session cleanly.

```bash
autoresearch --track demo --run-id quickstart stop-session
```

### `run-session-cycle`

Run one local-side cycle for the current session.

```bash
autoresearch --track demo --run-id quickstart run-session-cycle
```

If no proposal is available, the session moves to `waiting_for_proposal` and refreshes the handoff files.

### `run-session-cycles`

Run up to N local-side session cycles.

```bash
autoresearch --track demo --run-id quickstart run-session-cycles 3
```

### `record-cycle-reflection`

Complete the interpretation and next-direction fields after an auto-rejected
cycle when there is no next proposal to carry `previous_cycle_reflection`.

```bash
autoresearch --track demo --run-id quickstart record-cycle-reflection \
  --interpretation "The simpler model removed useful segmentation." \
  --next "Stop at the current champion."
```

---

## Milestone / Integrity

### `evaluate-milestone`

Run holdout evaluation for an experiment (requires `AUTORESEARCH_MILESTONE_TOKEN`). Human-only operation.

```bash
autoresearch --track demo --run-id quickstart evaluate-milestone <experiment-id>
```

### `update-integrity-manifest`

Recompute the integrity manifest after intentionally changing a protected file.

```bash
autoresearch update-integrity-manifest
```

---

## Cross-Track (Human-Only)

### `compare-tracks`

Compare the official champions of two research tracks without promoting either.

```bash
autoresearch compare-tracks claude codex
```

Writes: a full comparison report to `artifacts/cross_track/<timestamp>/comparison_report.md`. No promotion is performed.

---

## Orchestration

All `orchestrate` subcommands act on a **campaign** in
`artifacts/orchestrations/<orchestration-id>/`. `--orchestration-id` resolves to
the most recent campaign when omitted, mirroring run-id resolution. The
orchestrator's contract is [ORCHESTRATOR.md](../ORCHESTRATOR.md); the user-facing
walkthrough is [docs/RUN_ORCHESTRATED.md](RUN_ORCHESTRATED.md).

`orchestrate finish-delegation` is the one **child-side** subcommand: it is run by
a sub-agent inside its own run and takes the ordinary `--track`/`--run-id` global
flags. Everything else is orchestrator-side.

### `orchestrate new`

Create a campaign and pin its total cycle budget. Spawns that would exceed the
budget are refused.

```bash
autoresearch orchestrate new --dataset porto_seguro --target-mode claim_incidence \
  --total-cycles 20 --model-provider anthropic --model-name claude-opus-4-8
```

`--model-provider`/`--model-name` attribute the **orchestrator's own** model, so
the memory aggregator can tell it apart from its sub-agents. Prints the manifest
and the new orchestration id.

### `orchestrate list-backends`

Print the spawnable backend registry (`configs/orchestration/backends.toml`) with
its curated selection metadata, and the measured scorecard beneath each entry.

```bash
autoresearch orchestrate list-backends
```

The registry is the only source of spawnable backends; the scorecard is appended
as evidence and never adds or removes one. A backend with no collected
delegations renders `scorecard: —` rather than being hidden.

### `orchestrate spawn`

Pre-bootstrap a child run and launch a sub-agent against a brief.

```bash
autoresearch orchestrate spawn --brief briefs/freq_sev.json --backend claude-sonnet-medium --wait
autoresearch orchestrate spawn --brief briefs/glm.json --backend codex-gpt-5-5-medium --no-wait
autoresearch orchestrate spawn --brief briefs/probe.json --backend stub --dry-run
```

- `--wait` (default) blocks until the sub-agent exits, then builds its report.
- `--no-wait` launches detached, for parallel delegations. Bootstrap happens under
  a manifest lock so concurrent spawns cannot race.
- `--dry-run` prints the exact command argv, the child's `AUTORESEARCH_*`
  environment, and the composed prompt; it launches nothing.
- `--memory-access own|all` grants the sub-agent aggregator-mediated cross-run
  memory access.

### `orchestrate status`

Refresh detached delegation state and print a table: process liveness, cycles
completed, current champion and `gini_weighted`, last activity, and elapsed
wall-clock against the per-delegation timeout. A live delegation past its
allowance is marked `timed_out`; it is never killed automatically.

### `orchestrate kill`

Terminate one detached delegation's process group and record `killed`.

```bash
autoresearch orchestrate kill --delegation d02
```

### `orchestrate collect`

Rebuild delegation reports from child-run registry state and record each
`report_path` in the manifest.

```bash
autoresearch orchestrate collect
autoresearch orchestrate collect --delegation d01
```

Reports are built from the child's registry and its predictions artifact, never
from the sub-agent's claims. The sub-agent's `finish-delegation` summary is stored
verbatim as `agent_summary` and never feeds a computed number.

### `orchestrate respawn`

Launch a revised brief after a delegation stops. Defaults to the source
delegation's backend and a fresh run.

```bash
autoresearch orchestrate respawn --delegation d02 --brief briefs/d02_revised.json
autoresearch orchestrate respawn --delegation d02 --brief briefs/more.json --continue-run
autoresearch orchestrate respawn --delegation d02 --brief briefs/next.json --seed-champion from:d01
```

`--continue-run` reuses the stopped child's run and champion with a fresh cycle
budget; its report counts only the new cycles. `--seed-champion from:dNN`
initialises the new run's champion by replaying a prior delegation's champion.
Respawn refuses a source whose process is still alive.

### `orchestrate playoff`

Consolidate delegation finalists through an ascending gauntlet in a fresh
consolidation run.

```bash
autoresearch orchestrate playoff                      # interactive (default)
autoresearch orchestrate playoff --include d01,d02,d04
autoresearch orchestrate playoff --auto-decide
```

Delegations flagged `champion_is_baseline` are excluded. Finalists are ordered by
ascending search-validation `gini_weighted`; the weakest seeds the consolidation
run, and each stronger finalist is replayed as a challenger through the standard
`cv_bootstrap` gauntlet. `--interactive` stops at each `pending_llm` and prints
the exact `record-decision` command; re-running `playoff` resumes. `--auto-decide`
promotes only when every standard gate and hard guardrail passes.

Writes `playoff/playoff_report.json` and `.md`. The consolidation champion is the
orchestration champion, and its promotion fires the standard holdout evaluation.

### `orchestrate note`

Append a timestamped operator reflection and regenerate `ORCHESTRATION_LOG.md`.

```bash
autoresearch orchestrate note --kind reflection --delegation d02 \
  --text "The freq-sev line is real: +0.012 gini with clean calibration."
```

`--kind` is one of `plan`, `reflection`, `takeover`, `decision`, `other`.
Structured entries live in `notes.json` and are the source of truth;
`ORCHESTRATION_LOG.md` is generated from them plus the manifest, and framework
facts render in separate sections from operator commentary. Do not hand-edit the
Markdown.

### `orchestrate report`

Write the final campaign report and finalize the campaign's status.

```bash
autoresearch orchestrate report
autoresearch orchestrate report --json
```

Writes `campaign_report.json` + `CAMPAIGN_REPORT.md`, stores the report path in
the manifest, and marks the campaign `completed` once every delegation is terminal
and no playoff is in flight (a `consolidating` campaign stays that way — the
playoff owns that transition). The payload separates `framework_computed` (cycle
budget and usage, per-delegation facts, experiments by decision, promotions and
mean promoted Gini lift, distress counts, playoff result, final champion lineage,
wall clock, usage and cost) from `agent_testimony`. Delegations without a
collected report are listed and excluded from every aggregate. Cost is `null` when
no delegation reported one.

### `orchestrate backend-stats`

Aggregate collected delegation reports across all campaigns into the backend
scorecard (design §4.7 Layer 3).

```bash
autoresearch orchestrate backend-stats
autoresearch orchestrate backend-stats --json
autoresearch orchestrate backend-stats --orchestrations-root /path/to/fixtures
```

Per backend: campaigns, delegations, cycles, promotions, promotion rate
(promotions ÷ cycles), distress rate overall and by flag, repair attempts per
cycle, mean promoted Gini lift, and total/per-cycle/per-promotion cost. A rate
over zero observations prints `—`, not `0`. A delegation without a collected
report contributes nothing and is counted as skipped.

### `orchestrate finish-delegation`

**Child-side.** Run by the sub-agent in its own run when its budget is exhausted
or a brief stop-condition fires.

```bash
autoresearch --track claude --run-id 20260712T091500Z orchestrate finish-delegation \
  --summary "3–6 sentences: what you learned, what you'd try next, anything artifactual."
```

The summary is testimony. The report is generated from the registry either way; a
sub-agent that never calls this raises the `no_finish_delegation` distress flag.

---

## Cross-Run Memory Aggregator

All `memory` subcommands operate on the cross-run aggregator, which lives **outside the repo working tree** by default (`~/.autoresearch/<project>/memory/memory.sqlite`, overridable with the `AUTORESEARCH_MEMORY_DIR` environment variable). The aggregator contains **search-split metrics only** — no holdout data. None of these commands change per-run registries.

### `memory harvest`

Harvest the current run into the aggregator.

```bash
autoresearch --track claude --run-id 20260531T221638Z memory harvest
```

Harvest all discovered runs at once (backfill mode):

```bash
autoresearch memory harvest --all
```

`--all` discovers every `artifacts/tracks/**/runs/**/registry.sqlite`, reads its `run_manifest.json` for `model_identity`, and upserts. Runs whose manifest lacks `model_identity` are skipped with a warning — use `memory backfill-identity` first.

### `memory backfill-identity`

Write `model_identity` into existing `run_manifest.json` files that lack it. Use this for historical runs created before identity capture was added.

Patch the current run:

```bash
autoresearch --track opencode --run-id 20260531T093227Z memory backfill-identity \
  --provider deepseek \
  --name deepseek-v3 \
  --harness opencode
```

Patch all manifests missing identity:

```bash
autoresearch memory backfill-identity \
  --provider anthropic \
  --name claude-sonnet-4-6 \
  --harness claude-code \
  --all-missing
```

Target a specific directory:

```bash
autoresearch memory backfill-identity \
  --provider openai --name codex-mini-latest --harness codex \
  --run-dir artifacts/tracks/codex/runs/20260531T173106Z
```

Add `--force` to overwrite an existing `model_identity` entry.

### `memory status`

Print row counts for every table in the aggregator.

```bash
autoresearch memory status
```

Example output:

```json
{
  "memory_path": "~/.autoresearch/<project>/memory/memory.sqlite",
  "counts": {
    "models": 4,
    "runs": 7,
    "experiments": 312,
    "comparisons": 98,
    "insights": 0
  }
}
```

### Dashboard: Memory & Leaderboard page

The Streamlit dashboard (`autoresearch dashboard`) includes a **Memory & Leaderboard** page that reads from the cross-run aggregator (`~/.autoresearch/<project>/memory/memory.sqlite` by default; `AUTORESEARCH_MEMORY_DIR` overrides). It shows:

- **Score-trace chart** — running peak `gini_weighted` per `model_id` across cycles (toggle models on/off).
- **Peak Quality** tab — `max(gini_weighted)` per model across all its runs.
- **Efficiency** tab — `peak_gini / n_experiments` (how quickly each model improves).
- **Time-to-Structural-Insight** tab — first cycle where the running peak crossed the structural threshold (configurable in `[memory] structural_gini_threshold`, default 0.37).
- **Decision Quality** tab — sub-noise thrash rate: fraction of comparisons where `|mean_lift| < 2 * std_lift`.

The page is operator-facing (no access gate). Populate it first with `autoresearch memory harvest --all`.

#### Config: `[memory]` section in `configs/default.toml`

```toml
[memory]
# The aggregator lives outside the repo working tree by default
# (~/.autoresearch/<project>/memory); override with AUTORESEARCH_MEMORY_DIR.
structural_gini_threshold = 0.37
```

`structural_gini_threshold` controls the Time-to-Structural-Insight leaderboard threshold. It is also available on `ProjectConfig.structural_gini_threshold`.

### `memory record-insight`

Record an evidence-bound insight from a JSON file into the aggregator. The insight is validated against the run's own registry (read-only): experiment/comparison IDs must exist and the cited `delta` must match the registry values within tolerance. Insights with fabricated evidence are stored with `verified=0` and excluded from the playbook and default queries.

```bash
autoresearch --track claude --run-id 20260531T221638Z memory record-insight \
  --file /path/to/my_insight.json
```

Insight JSON schema:

```json
{
  "claim": "rate-based Tweedie GBMs plateau ~0.33; total-target trees reach ~0.40",
  "scope": "general",
  "confidence": 0.8,
  "evidence": {
    "experiment_ids": ["exp_abc123", "exp_def456"],
    "comparison_ids": ["cmp_xyz789"],
    "metric": "gini_weighted",
    "delta": 0.07
  },
  "supersedes": null,
  "contradicts": null
}
```

### `memory list-insights`

List insights stored in the aggregator. Defaults to verified-only.

```bash
# Verified insights only (default)
autoresearch memory list-insights

# Include unverified insights
autoresearch memory list-insights --include-unverified

# Filter to a specific run
autoresearch memory list-insights --run claude/20260531T221638Z
```

### `memory query`

Query the aggregator. Requires `AUTORESEARCH_MEMORY_ACCESS` to be set to `own` or `all`; with the default (`none`) the command refuses with a clear message.

```bash
export AUTORESEARCH_MEMORY_ACCESS=own   # or 'all'

# Retrieve verified insights (own-model only when access=own)
autoresearch memory query --insights

# Include unverified insights
autoresearch memory query --insights --include-unverified

# Filter by model
autoresearch memory query --insights --model anthropic/claude-sonnet-4-6

# Retrieve experiments
autoresearch memory query --experiments

# Canned analytical queries
autoresearch memory query --analysis peak-gini-by-framing
autoresearch memory query --analysis plateau-families
autoresearch memory query --analysis biggest-single-jumps
autoresearch memory query --analysis efficiency-by-model
```

**Access levels:**
- `none` (default) — query refuses; `build_llm_context()` is byte-for-byte unchanged from today (run isolation guarantee).
- `own` — results filtered to the current run's `model_id`.
- `all` — all models returned, fully attributed.

The resolved access level is recorded in `run_manifest.json` (`memory_access` key) at bootstrap for auditability.

### `memory build-playbook`

Compile verified insights and leaderboard-derived facts into a dynamic playbook at `<memory_dir>/playbook/latest.md` (default `~/.autoresearch/<project>/memory/playbook/latest.md`; `AUTORESEARCH_MEMORY_DIR` overrides). A timestamped copy is also saved. Only `verified=1` insights are included; each bullet cites evidence IDs and source `model_id`.

```bash
# Build the full attributed playbook
autoresearch memory build-playbook

# Build an own-model filtered variant
autoresearch memory build-playbook --model-filter anthropic/claude-sonnet-4-6
```

The playbook is automatically regenerated at every 5-cycle checkpoint when new verified insights have landed. When `AUTORESEARCH_MEMORY_ACCESS` is `own` or `all`, the handoff bundle links the playbook (or the filtered own-model variant). When access is `none`, the handoff is unchanged.

## Desktop LLM telemetry

Claude Code Desktop and Codex Desktop sessions are automatically imported at
the end of each turn once the native session is bound to a research run. Stop
hooks launch a short deferred import so provider completion records written
after the hook returns are included. The
run-scoped `telemetry.sqlite` contains normalized token, cache, reasoning,
tool-call, timing, error, and workflow-attribution records. Raw prompts and
full tool output are not copied into the run.

Each recorded experiment and finalized desktop turn updates `LLM_USAGE.md` in
the run directory. The table reports incremental input, cached, uncached,
output, reasoning, and total tokens at experiment checkpoints and at every
user-visible breakpoint. This keeps the ledger truthful when a user requests X
experiments, later continues with Y, and then continues with Z. Usage that has
been imported but does not yet belong to a settled checkpoint is shown
explicitly as between-cycle or wrap-up overhead. Model attribution falls back
from the provider call to its turn and native session, so historical Codex
records without a call-level model remain attributable.

```bash
# Human-readable JSON report
autoresearch --track codex --run-id 20260607T081407Z telemetry report

# Manual recovery/backfill. Transcript discovery is automatic by session id.
autoresearch --track codex --run-id 20260607T081407Z telemetry sync \
  --surface codex --session-id <codex-thread-id> --finalize-turn

autoresearch --track claude --run-id 20260604T064305Z telemetry sync \
  --surface claude --session-id <claude-session-id> --finalize-turn
```

Imports are incremental and idempotent. Re-running `telemetry sync` reads only
new complete JSONL records and does not double-count provider requests or tool
calls. Workflow attribution is rebuilt deterministically on every import.
Telemetry sync also reconciles `run_manifest.json` model identity with the
harness-observed model. The command result reports `verified`, `conflict`,
`no_observed_model`, or `manifest_missing` for that reconciliation.
Provider-reported dollar cost remains `null` when the desktop product
does not expose it; the system does not guess a cost.

Use `--rebuild` after upgrading an importer if an existing run should be
reparsed from byte zero. Stable provider request and tool IDs keep the rebuild
idempotent.
