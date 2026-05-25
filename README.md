# Insurance AutoResearch

Local Python backbone for reproducible insurance burning-cost experimentation on freMTPL2-style local data.

This repository implements a full autonomous experimentation loop: data preparation → model training → actuarially-rigorous evaluation → promotion-gated champion management → LLM-guided proposal generation.

## Features

- Clean `src/` Python package layout with a frozen `ProjectConfig` dataclass
- SQLite experiment registry with session, proposal, branch, and champion state tables
- Streamlit dashboard: experiment tables, promotion decisions, calibration diagnostics, proposal queue, branch lineage
- Raw data loading from `data/raw/` with deterministic anonymisation and column renaming
- Configurable claim capping with visible per-decile diagnostics
- **Architecturally-separated milestone holdout vault** — `agent_dataset_search.parquet` (never contains holdout rows) stored separately from `data/holdout_vault/agent_dataset_holdout.parquet` (token-gated)
- 5-fold deterministic CV with variance decomposition (between-fold vs. within-fold)
- **Actuarially-correct model layer**: Tweedie GLM, frequency×severity Poisson/Gamma GLM, Tweedie GBM (HistGradientBoosting), regularized linear baseline
- **Full actuarial metric panel**: Tweedie deviance (power=1.5) as primary metric, exposure-weighted Gini, double-lift slope, predicted-to-actual ratio, Poisson deviance
- Calibration diagnostics by predicted decile and exposure band, PSI, segment loss ratios
- **Hardened promotion gate**: relative lift floor (0.5%), MDE estimation, Bonferroni adjustment for multiple comparisons, calibration check
- Reproducibility environment manifest (git SHA, pip freeze, file SHA256s) written per experiment
- Controlled LLM-guided proposal queue with per-family hyperparameter validation
- MockProposer with a 5-entry rotating pool (prevents autonomous loop deadlock)
- Retry logic with exponential backoff for OpenAI/Anthropic proposers
- Branch lineage for proposed experiments; deduplication by proposal fingerprint
- 52 tests covering statistical claims (false-positive rate, true-positive rate, variance decomposition, holdout separation)

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Prepare Data

Raw files are expected under `data/raw/`. The loader discovers freMTPL2 frequency and severity files recursively by filename.

```bash
autoresearch prepare-data
```

This writes:

- `data/processed/agent_dataset_search.parquet` — train + search_validation rows only (no holdout)
- `data/holdout_vault/agent_dataset_holdout.parquet` — milestone holdout (token-gated)
- `data/metadata/private_column_mapping.json`
- `data/metadata/agent_schema.json`
- `data/metadata/dataset_profile.json`
- `data/metadata/capping_diagnostics.json`
- `data/splits/split_pack.csv`
- `data/splits/split_pack_manifest.json`
- `data/splits/split_pack_folds.parquet` — 5-fold CV assignments

The default split reserves 20% of the full dataset as `milestone_holdout`. Ordinary baseline evaluation fits on `train` and scores on `search_validation` only. The holdout rows are never visible to experiment runners; access requires the `AUTORESEARCH_MILESTONE_TOKEN` environment variable.

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

The dashboard shows project status, split metadata, capping diagnostics, baseline experiment tables, official champion state, comparison views, calibration diagnostics, proposal queue, branch lineage, and file-handoff status.

## Model Families

Four model families are supported. Choose via `model_family` in an experiment config:

| `model_family`           | Description |
|--------------------------|-------------|
| `tweedie_glm`            | `TweedieRegressor` with log link and exposure weights. Hyperparameters: `alpha`, `power`. |
| `frequency_severity_glm` | `PoissonRegressor` for frequency × `GammaRegressor` for severity per claim. Hyperparameters: `freq_alpha`, `sev_alpha`. |
| `tweedie_gbm`            | `HistGradientBoostingRegressor` with Poisson loss. Hyperparameters: `max_iter`, `max_depth`, `learning_rate`, `min_samples_leaf`, `l2_regularization`. |
| `regularized_linear`     | Ridge regression on log1p pure premium (legacy baseline). Hyperparameter: `alpha`. |

## Primary Metric

The primary metric is **Tweedie deviance at power=1.5** (`tweedie_deviance_p15`). Lower is better. This is the industry-standard proper scoring rule for insurance burning-cost under a compound Poisson-Gamma loss model.

Additional panel metrics (not used for promotion decisions):
- `gini_weighted` — exposure-weighted Gini coefficient (rank discrimination)
- `double_lift_slope` — OLS slope of actual on predicted pure premium by decile (calibration linearity)
- `predicted_to_actual_ratio` — aggregate calibration (should be ≈ 1.0)
- `poisson_deviance` — frequency model calibration
- `weighted_mae_claim_cost`, `weighted_rmse_claim_cost` — scale-dependent error metrics

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

Each run writes a folder under `artifacts/experiments/` containing:

- `config_snapshot.json`
- `metrics.json`
- `split_metrics.csv`
- `predictions.csv`
- `diagnostics.json` — calibration decile table, PSI, segment loss ratios
- `environment_manifest.json` — git SHA, dirty flag, pip freeze, file SHA256s
- `capping_diagnostics.json`

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

Promotion requires all of the following checks to pass:

- Mean lift positive (challenger has lower Tweedie deviance than champion)
- Relative lift ≥ 0.5% of champion score (prevents noise promotions)
- Challenger win rate ≥ 60% across paired resamples
- Bootstrap 90% lower bound ≥ 0 (Bonferroni-adjusted for multiple comparisons)
- Calibration check: all predicted decile ratios in [0.3, 3.0]

Default promotion settings in `configs/default.toml`:

```toml
[promotion]
minimum_mean_lift = 0.0
min_relative_lift = 0.005
min_absolute_lift = 0.0
minimum_win_rate = 0.60
bootstrap_lower_bound = 0.0
bootstrap_lower_bound_relative = 0.0
confidence_level = 0.90
max_predicted_to_actual_drift = 0.05
require_diagnostics = true
bonferroni_lookback = 10
```

## K-Fold Cross-Validation

When `use_cv = true` in `[evaluation]`, experiments use 5-fold CV instead of the single search_validation split. CV results include variance decomposition: `between_fold_variance`, `within_fold_variance`, `total_variance`, and a warning flag when between-fold variance dominates (indicating data leakage risk or unstable feature engineering).

```toml
[evaluation]
use_cv = false
cv_folds = 5
cv_n_repeats = 1
```

## File-Based Codex / Claude Code Workflow

Initialise the official champion as the direct pure premium baseline:

```bash
autoresearch init-official-champion
```

The official champion is intentionally distinct from the best point-estimate experiment. It starts as the Tweedie GLM baseline by product decision and changes only through the promotion gate.

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

The primary workflow does not require API keys. Runtime API-backed proposers are available as an alternative to the file-handoff workflow.

```toml
[llm]
provider = "file_handoff"
model = "claude-opus-4-7"
temperature = 0.2
proposal_file = "artifacts/auto_research/proposals/inbox/manual_proposals.jsonl"
```

Supported `provider` values: `file_handoff`, `mock`, `file`, `openai`, `anthropic`. `openai` uses `OPENAI_API_KEY`; `anthropic` uses `ANTHROPIC_API_KEY`. The `mock` provider cycles through a pool of 5 diverse pre-defined proposals (Tweedie GLM ×2, freq×sev GLM, Tweedie GBM ×2) to prevent autonomous loop deadlock.

All providers return the same structured proposal schema, which is validated before anything is run. Per-family hyperparameter bounds are enforced (e.g. GBM proposals cannot set `power`; alpha values are checked against the configured search space ranges).

Legacy direct proposer commands:

```bash
autoresearch generate-proposal
autoresearch enqueue-proposal path/to/proposal.json
autoresearch run-cycle
autoresearch run-cycles 3
```

## Tests

```bash
pytest
```

52 tests cover data pipeline, model dispatch, metric panel statistical properties, CV variance decomposition, holdout separation, calibration diagnostics, promotion gate false-positive rate (≤ 20% under H0), and promotion gate true-positive rate (≥ 70% for a 10% improvement).
