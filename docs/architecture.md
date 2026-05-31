# Architecture

## Module Map

```
src/autoresearch/
├── config.py                  # ProjectConfig frozen dataclass (all settings)
├── experiment_runner.py       # Load data → dispatch model → evaluate → persist
├── comparison_runner.py       # Paired comparison → promotion decision
├── data/
│   ├── pipeline.py            # prepare-data: clean, cap, split, fold assignments
│   ├── splits.py              # generate_fold_assignments() for deterministic CV
│   └── holdout_vault.py       # write_vault / load_search_dataset / load_holdout_dataset
├── models/
│   ├── dispatcher.py          # dispatch_model() → ModelResult
│   └── global_mean.py         # built-in global-mean baseline
├── evaluation/
│   ├── metrics.py             # full_metric_panel(), evaluate_predictions()
│   ├── diagnostics.py         # compute_diagnostics() — calibration, PSI, segments
│   └── resampling.py          # paired_comparison(), promotion_decision(),
│                              #   bootstrap_lift_summary(), cv_repeated_scores()
├── experiment_registry/
│   └── registry.py            # SQLite: record_experiment, set_official_champion, …
├── controller/
│   ├── proposal_schema.py     # allowed_search_space(), validate_proposal()
│   ├── proposer.py            # FileProposer (file-handoff inbox)
│   ├── workflow.py            # enqueue_proposal_from_file(), run_next_queued_proposal()
│   ├── champion.py            # initialise_official_champion()
│   └── handoff.py             # export_context(), write_proposal_template()
├── utils/
│   └── environment.py         # capture_environment() — git SHA, pip freeze, SHA256s
└── dashboard/
    └── app.py                 # Streamlit dashboard
```

## High-Level Flow

```
prepare-data
  → load freMTPL2 → anonymise → compute capping diagnostics → create split pack (train/sv/holdout)
  → write agent_dataset_search.parquet (no holdout rows; raw uncapped target)
  → write holdout_vault/agent_dataset_holdout.parquet (token-gated)
  → write split_pack_folds.parquet (5-fold CV assignments)
  (the fixed claim cap is applied uniformly at scoring time, not baked into the
   persisted artifacts, so search/holdout stay a single canonical source)

run-baseline / run-next-proposal
  → load_search_dataset()  ← only train + search_validation rows visible
  → dispatch_model()       ← routes to global_mean built-in or a run-local script
  → evaluate_predictions() ← target-aware metric panel + Gini + double-lift
  → compute_diagnostics()  ← calibration by pred decile, PSI, segment ratios
  → capture_environment()  ← git SHA, pip freeze, file SHA256s
  → record_experiment()    ← SQLite registry

compare-experiments / compare-to-champion
  → paired_comparison()    ← bootstrap lift distribution, Bonferroni-adjusted CI
  → promotion_decision()   ← 8-check gate: relative lift, calibration, win rate, …
  → set_official_champion() if promoted
```

## Holdout Vault

The milestone holdout is architecturally separated from the search partition. `experiment_runner.py` calls `load_search_dataset()` which reads only `agent_dataset_search.parquet` — a file that never contains holdout record IDs. The holdout file lives in a separate directory (`data/holdout_vault/`) with a `.locked` sentinel. Reading it requires the `AUTORESEARCH_MILESTONE_TOKEN` environment variable. This makes accidental holdout contamination fail loudly rather than silently.

## Model Layer

Autonomous experiments use run-local modelling scripts. A proposal points
`experiment_config.model.script_path` at a Python file stored beside the
proposal. The file must expose:

```python
def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters):
    return predicted_target_array, notes_dict
```

The framework still owns data loading, split application, capping, evaluation,
comparison, and registry writes. The script owns the modelling choice for that
single run. This means GLMs, GBMs, GAMs, ensembles, or simpler hand-built rules
are all possible research directions, but the implementation is an auditable
per-run artifact instead of a pre-selected method imported from `src/models`.

The `global_mean` no-model baseline is the built-in bootstrap starting point.
All other experiments must supply a run-local `fit_predict` script via
`experiment_config.model.script_path`.

`dispatch_model()` in `models/dispatcher.py` handles script loading, prediction
DataFrame construction, and a row-count assertion to catch silent drops.

The active target is controlled by `evaluation.target_mode` or the CLI
`--target-mode` override. `burning_cost` is the default and interprets model
outputs as predicted claim costs. `frequency` interprets model outputs as
expected claim counts. In both modes scripts return target totals, not rates.

## Output Validation

After each queued proposal run, the controller writes `validation_report.json`
before any promotion comparison. The report checks that predictions are finite,
non-negative, non-empty, have sensible aggregate scale, produce a finite primary
metric, and show positive lift against the current champion. Failed checks write
`repair_request_2.json` or `repair_request_3.json`; the agent can revise the
next script attempt up to three total attempts before the proposal is marked as
failed.

## Evaluation and Promotion

**Primary metric**: configured in `[evaluation]` and currently defaults to
`gini_weighted`. Burning-cost runs also record Tweedie deviance on pure premium;
frequency runs record Poisson deviance on claim frequency.

**Promotion gate** (all 8 checks must pass):
1. Mean lift > 0 (challenger improves the configured primary metric)
2. Relative lift ≥ `min_relative_lift` (default 0.5%) — prevents noise promotions
3. Absolute lift ≥ `min_absolute_lift`
4. Challenger win rate ≥ `minimum_win_rate` across paired resamples
5. Bootstrap 90% CI lower bound ≥ `bootstrap_lower_bound`
6. Bootstrap 90% CI relative lower bound ≥ `bootstrap_lower_bound_relative`
7. Calibration OK (all predicted decile ratios in [0.3, 3.0])
8. Diagnostics present (when `require_diagnostics = true`)

The CI is Bonferroni-adjusted using `bonferroni_lookback` to account for multiple comparisons across the autonomous search history.

## Reproducibility

Every experiment artifacts folder includes `environment_manifest.json` capturing: Python version, platform, git SHA, git dirty flag, pip freeze output, key dependency versions (numpy, pandas, sklearn, pyarrow), and SHA256 hashes of the input data files. This makes any result reproducible given the captured environment — the manifest records the exact resolved versions even though `pyproject.toml` only lower-bounds them (there is no lockfile), so a faithful rerun means recreating the environment the manifest describes.

## K-Fold CV

5-fold fold assignments are generated deterministically from `record_id` via a stable hash (`stable_unit()`). When `use_cv = true`, `cv_repeated_scores()` runs `n_folds` train/val splits and returns a variance decomposition: `between_fold_variance` measures how much the score varies across folds, `within_fold_variance` measures residual noise within each fold, and `total_variance` is their sum. A `warning_between_dominates` flag fires when between-fold variance exceeds 70% of total — this indicates potential leakage, unstable feature engineering, or severe distributional shift across folds.

## Constraints

- Local Python project; no distributed compute
- Streamlit dashboard for interactive inspection
- No holdout access during ordinary search (enforced architecturally)
- Every experiment is fully resumable and reproducible from its `environment_manifest.json`
- Promotion decisions are non-reversible in the registry (append-only history)
