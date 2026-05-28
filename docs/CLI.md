# CLI Reference

All commands are invoked as `autoresearch [--track NAME] [--run-id ID] <command>`.
Global flags `--track` and `--run-id` scope all artifact paths and the registry to
`artifacts/tracks/<NAME>/runs/<ID>/`. Use `--new-run` instead of `--run-id` when
starting a fresh timestamped run in a track.

---

## Data Preparation

### `prepare-data`

Build the Phase 1 data artifacts from raw freMTPL2 files in `data/raw/`.

```bash
autoresearch prepare-data
```

Writes: `data/processed/agent_dataset_search.parquet`, `data/holdout_vault/agent_dataset_holdout.parquet`, metadata files under `data/metadata/`, and the deterministic split pack under `data/splits/`.

---

## Registry & Bootstrap

### `bootstrap-track`

Idempotently prepare data, registry, baselines, champion, templates, and context for a named track. Safe to run at the start of every new session.

```bash
autoresearch --track demo --run-id quickstart bootstrap-track
```

For new agent runs, prefer `--new-run` so the run folder includes date and time:

```bash
autoresearch --track demo --new-run bootstrap-track
```

Writes: registry, baseline experiments, official champion, proposal templates, and handoff context under the run directory.

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

```bash
autoresearch --track demo --run-id quickstart run-baseline configs/experiments/global_mean.toml
```

Writes: experiment artifacts under the run's `iterations/` directory.

### `run-all-baselines`

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

Create a named supervised autonomous research session.

```bash
autoresearch --track demo --run-id quickstart start-session my-session --max-cycles 10
```

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
