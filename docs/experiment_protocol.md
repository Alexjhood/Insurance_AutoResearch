# Experiment Protocol

## Evaluation Layers

### Search Evaluation (ordinary experiments)

Fast single-pass evaluation on `search_validation`. Input data comes exclusively from `agent_dataset_search.parquet` — holdout rows are never visible.

**Primary metric**: `tweedie_deviance_p15` (Tweedie deviance at power=1.5). This is a proper scoring rule for compound Poisson-Gamma loss and is the industry standard for motor insurance pure premium modelling.

Additional panel metrics are recorded for diagnostic purposes but not used for promotion decisions:
- `gini_weighted` — exposure-weighted Gini (rank discrimination, scale-invariant)
- `double_lift_slope` — OLS slope of actual on predicted pure premium by decile
- `predicted_to_actual_ratio` — aggregate calibration (target: ≈ 1.0)
- `poisson_deviance`, `weighted_mae_claim_cost`, `weighted_rmse_claim_cost`

Calibration diagnostics (`diagnostics.json`) are written alongside every experiment:
- Predicted-to-actual ratio by predicted decile and by exposure band
- Loss ratio by segment (risk_score_index_e, vehicle_age_band_c, driver_age_band_d, territory_band_h)
- Population Stability Index (PSI) between train and search_validation distributions

### Promotion Evaluation (champion/challenger comparison)

A challenger is compared to the champion via `compare-experiments` or `compare-to-champion`. This runs paired bootstrap resampling over `search_validation` and applies the promotion gate.

**All 8 gate checks must pass for promotion**:
1. Mean lift > 0 (challenger Tweedie deviance < champion)
2. Relative lift ≥ 0.5% of champion score (prevents promotion on numerical noise)
3. Absolute lift ≥ `min_absolute_lift` (default 0)
4. Challenger win rate ≥ 60% across paired resamples
5. Bootstrap 90% CI lower bound ≥ 0
6. Bootstrap 90% CI relative lower bound ≥ `bootstrap_lower_bound_relative`
7. Calibration OK (all predicted decile ratios in [0.3, 3.0])
8. Calibration diagnostics present

The bootstrap CI is Bonferroni-adjusted using `bonferroni_lookback = 10` (the number of prior comparisons in the search history) to control the family-wise error rate.

The `promotion_decision` result also records:
- `mde_relative` — estimated minimum detectable effect given the resampling noise
- `power_note` — whether the detected effect is above or below the MDE
- Per-check pass/fail flags for auditability

### Milestone Evaluation (frozen holdout)

The milestone holdout is reserved for checkpoint comparisons (e.g. quarterly or before a production deployment). It must not be used for ordinary search or promotion decisions.

To access:
```bash
export AUTORESEARCH_MILESTONE_TOKEN=<secret>
autoresearch evaluate-on-holdout EXPERIMENT_ID
```

The token prevents accidental reads. The holdout file lives in `data/holdout_vault/` which is never written by the experiment runner.

## Model Families

The following families are supported within the allowed search space:

- `tweedie_glm`
- `frequency_severity_glm`
- `tweedie_gbm`
- `regularized_linear`

## Required Artifacts Per Experiment

Every experiment folder under `artifacts/experiments/<experiment_id>/` must contain:

| File | Description |
|------|-------------|
| `config_snapshot.json` | Full experiment config at run time |
| `metrics.json` | Tweedie deviance panel, aggregate and per-split |
| `split_metrics.csv` | Per-split metric rows |
| `predictions.csv` | Actual/predicted/exposure per row |
| `diagnostics.json` | Calibration decile table, PSI, segment loss ratios |
| `environment_manifest.json` | Git SHA, dirty flag, pip freeze, file SHA256s |
| `capping_diagnostics.json` | Claim cap threshold and affected rows |
| `validation_report.json` | Autonomous proposal output sanity/lift checks, when run through the proposal controller |
| `model_attempt_N.py` | Run-local modelling script used by autonomous proposal attempt N |

## Proposal Constraints

Autonomous proposals must provide a run-local script for every non-`global_mean`
experiment via `experiment_config.model.script_path`. The script is copied into
the proposal iteration directory, scanned for holdout markers, and executed
through the `fit_predict()` contract. Built-in GLM/GBM ideas are allowed, but
the implementation must live in the run-local script rather than relying on a
pre-existing module in `src/autoresearch/models`.

The proposal search space is validated before any experiment is run. Per-family bounds:

- `regularized_linear`: alpha in [0.01, 100.0]
- `tweedie_glm`: alpha in [0.001, 10.0]; power in {1.1, 1.3, 1.5, 1.7, 1.9}
- `frequency_severity_glm`: freq_alpha and sev_alpha each in [0.001, 10.0]
- `tweedie_gbm`: max_iter in [100, 2000]; max_depth in {3, 5, 7, 9}; learning_rate in [0.01, 0.2]; min_samples_leaf in {50, 200, 500, 1000}. Proposals must not include a `power` key.

Proposals referencing `milestone_holdout` as an eval split are rejected.

## Reproducibility Protocol

1. Every experiment run captures a full `environment_manifest.json` (git SHA + dirty flag, pip freeze, input file hashes). Running the same experiment config on the same data with the same environment must produce bit-for-bit identical metrics.

2. All random operations use `random_seed` from `ProjectConfig`. No global NumPy or Python random state is set by library code.

3. Fold assignments (`split_pack_folds.parquet`) are computed once from `record_id` hashes and reused across all CV runs.

4. The SQLite registry is append-only. No records are updated or deleted after creation.
