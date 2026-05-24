# Insurance AutoResearch

Local Python backbone for reproducible insurance burning-cost experimentation on freMTPL2-style local data.

This repository currently implements Phases 0-4 plus the file-based handoff proposal workflow:

- clean `src/` Python package layout
- config loading
- SQLite experiment registry skeleton
- Streamlit dashboard skeleton
- raw data loading from `data/raw/`
- deterministic anonymised dataset creation
- dataset profile and metadata outputs
- reproducible split-pack generation with a protected milestone holdout
- configurable claim capping with visible diagnostics
- deterministic direct and frequency-severity baseline experiments
- registry-backed experiment comparison
- repeated search-time resampling
- paired champion/challenger comparison
- bootstrap uncertainty summaries
- explicit promotion decisions
- official champion state and champion history
- controlled LLM-guided proposal queue
- branch lineage for proposed experiments
- bounded propose -> run -> compare -> promote workflows
- file-based Codex / Claude Code handoff artifacts
- proposal inbox and processed proposal flow
- focused tests for the Phase 1 data layer

Codex or Claude Code should normally act as the external experiment-design agent by reading handoff files and writing structured proposal JSON into the inbox. The Python framework remains responsible for validation, queueing, deterministic execution, comparison, promotion, registry persistence, and dashboard/reporting.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Prepare Data

Raw files are expected under `data/raw/`. The current loader discovers the freMTPL2 frequency and severity files recursively by filename.

```bash
autoresearch prepare-data
```

This writes:

- `data/processed/agent_dataset.parquet`
- `data/metadata/private_column_mapping.json`
- `data/metadata/agent_schema.json`
- `data/metadata/dataset_profile.json`
- `data/metadata/capping_diagnostics.json`
- `data/splits/split_pack.csv`
- `data/splits/split_pack_manifest.json`

The default split protocol reserves 20% of the full dataset as `milestone_holdout`.
Ordinary baseline evaluation fits on `train` and scores on `search_validation` only.

The default claim cap is configured in `configs/default.toml`:

```toml
[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000
```

## Experiment Registry

```bash
autoresearch init-registry
```

The default registry lives at `artifacts/experiment_registry.sqlite`.

## Dashboard

```bash
streamlit run src/autoresearch/dashboard/app.py
```

The dashboard shows project status, split metadata, capping diagnostics, baseline experiment tables, official champion state, comparison views, proposal queue, branch lineage, and file-handoff status.

## Baseline Experiments

Run one named baseline:

```bash
autoresearch run-baseline configs/experiments/direct_pure_premium.toml
autoresearch run-baseline configs/experiments/frequency_severity.toml
```

Run all checked-in baselines:

```bash
autoresearch run-all-baselines
```

Inspect registered runs:

```bash
autoresearch list-experiments
```

## Volatility-Aware Comparison

Run repeated search-time evaluation for one experiment:

```bash
autoresearch run-repeated-evaluation EXPERIMENT_ID
```

Compare a champion and challenger on the same repeated resamples:

```bash
autoresearch compare-experiments CHAMPION_ID CHALLENGER_ID
```

Compare a challenger against the current point-estimate champion:

```bash
autoresearch compare-to-champion CHALLENGER_ID
```

List promotion decisions:

```bash
autoresearch list-promotions
```

Default Phase 3 settings live in `configs/default.toml`:

```toml
[resampling]
repeated_resamples = 30
bootstrap_iterations = 1000
resample_fraction = 1.0
random_seed = 20260524

[promotion]
minimum_mean_lift = 0.0
minimum_win_rate = 0.55
bootstrap_lower_bound = 0.0
confidence_level = 0.90
```

Positive lift means the challenger lowered the primary score (`rmse_pure_premium`) versus the champion. Promotion requires positive mean lift, at least 55% challenger wins across paired resamples, and a positive lower bootstrap interval bound. Otherwise the result is recorded as inconclusive.

## File-Based Codex / Claude Code Workflow

Initialise the official champion as the direct pure premium baseline:

```bash
autoresearch init-official-champion
```

The official champion is intentionally distinct from the best point-estimate experiment. It starts as the direct pure premium baseline by product decision and changes only through the promotion gate.

Export context and handoff instructions for Codex or Claude Code:

```bash
autoresearch export-context
autoresearch write-proposal-template
autoresearch show-latest-handoff
autoresearch show-proposal-inbox-status
```

External agent workflow:

1. Read `artifacts/auto_research/handoffs/latest_handoff.md`
2. Read `artifacts/auto_research/context/latest_context.json`
3. Read `artifacts/auto_research/handoffs/proposal_instructions.md`
4. Use `artifacts/auto_research/handoffs/proposal_template.json` as the shape
5. Write one proposal JSON file to `artifacts/auto_research/proposals/inbox/`

Ingest inbox proposals and enqueue valid ones:

```bash
autoresearch ingest-proposals
autoresearch enqueue-ingested-proposals
```

Run the next queued proposal through deterministic execution, comparison, and promotion:

```bash
autoresearch run-next-proposal
```

Run a full cycle from the latest inbox proposal:

```bash
autoresearch run-latest-proposal-cycle
```

The file-handoff workflow writes inspectable artifacts under:

- `artifacts/auto_research/context/latest_context.json`
- `artifacts/auto_research/context/current_champion_summary.json`
- `artifacts/auto_research/context/recent_comparisons_summary.json`
- `artifacts/auto_research/context/recent_branch_summary.json`
- `artifacts/auto_research/handoffs/latest_handoff.md`
- `artifacts/auto_research/handoffs/proposal_template.json`
- `artifacts/auto_research/handoffs/proposal_schema.json`
- `artifacts/auto_research/handoffs/proposal_instructions.md`
- `artifacts/auto_research/proposals/inbox/`
- `artifacts/auto_research/proposals/processed/valid/`
- `artifacts/auto_research/proposals/processed/invalid/`
- `artifacts/auto_research/results/latest_cycle_result.md`

Inspect queue, champion, and lineage state:

```bash
autoresearch list-proposals
autoresearch inspect-proposal PROPOSAL_ID
autoresearch list-champion-history
autoresearch list-branches
```

## Supervised Autonomous Sessions

Create a named session that Codex or Claude Code can continue:

```bash
autoresearch start-session local-research --max-cycles 10
autoresearch session-status
```

Run one local-side cycle. If no proposal is available, the session moves to `waiting_for_proposal` and refreshes the handoff files for the external agent:

```bash
autoresearch run-session-cycle
```

Run up to N cycles, stopping automatically if the next proposal is needed:

```bash
autoresearch run-session-cycles 3
```

Pause, resume, or stop:

```bash
autoresearch pause-session
autoresearch resume-session
autoresearch stop-session
```

Session logs and summaries are written under:

- `artifacts/auto_research/sessions/`
- `artifacts/auto_research/results/latest_session_summary.md`
- `artifacts/auto_research/results/latest_cycle_result.md`
- `artifacts/auto_research/results/non_promoted/`

See `docs/autonomous_session_workflow.md` for the full operating model.

## Optional Runtime Proposers

The primary workflow above does not require API keys. Runtime API-backed proposers are still available for experiments, but they are secondary.

Default handoff settings are configured in `configs/default.toml`:

```toml
[llm]
provider = "file_handoff"
model = "external-codex-or-claude-code"
temperature = 0.2
proposal_file = "artifacts/auto_research/proposals/inbox/manual_proposals.jsonl"
```

Supported provider values are `file_handoff`, `mock`, `file`, `openai`, and `anthropic`. `openai` uses `OPENAI_API_KEY`; `anthropic` uses `ANTHROPIC_API_KEY`. All providers must return the same structured proposal schema, which is validated before anything is run.

Legacy direct proposer commands remain available:

```bash
autoresearch generate-proposal
autoresearch enqueue-proposal path/to/proposal.json
autoresearch run-cycle
autoresearch run-cycles 3
```

Each run writes a folder under `artifacts/experiments/` containing:

- `config_snapshot.json`
- `metrics.json`
- `split_metrics.csv`
- `predictions.csv`
- `capping_diagnostics.json`

## Tests

```bash
pytest
```

## Phase Boundaries

Next steps should make the file-handoff loop smoother over multiple cycles: proposal deduplication, richer branch analytics, automatic handoff refresh after every run, and better segment-level diagnostics. The milestone holdout split is for checkpoint evaluation only and is not used by ordinary baseline, proposal, or promotion workflows.
