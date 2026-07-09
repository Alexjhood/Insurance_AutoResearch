# TabFM Spike — Full Comparison Report

**Dataset:** search split, 542,412 rows × 9 features, exposure 286,773 yrs, burning-cost target (`min(ClaimAmount, 100k)`), 19,957 claim rows (3.68%; 4.8% of exposure).
**Protocol:** exposure-weighted context subsample (seed 42), zero-shot fit, batched scoring, multiplicative recalibration, `full_metric_panel` from the protected evaluation code. Holdout never touched.
**Dates:** 2026-07-04 (design) / 2026-07-05 (all runs). **Hardware:** Modal serverless — T4 $0.59/hr · L4 $0.80/hr · A100-40GB ~$2.10/hr · H100 ~$3.95/hr.

## Reference points

| Model | gini_weighted | Notes |
|---|---|---|
| Global mean baseline | 0.000 | run starting champion |
| TabPFN-3, best (64k ctx) | 0.344 | 2026-07-03 benchmark; ~0.20 @ 1k ctx |
| **LightGBM Tweedie champion** | **0.369** | current global champion |

## All successful runs (headline metrics)

| # | Config | GPU | Context | Score rows | fit s | predict | VRAM | gini_w | rank_gini_w | spearman | APL | pred/actual |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | base, 4 members, batch 10k | L4 | 5k | 542k | 0.1 | 51.6 m | 10.3 GB | 0.2858 | 0.3630 | 0.0707 | 624.83 | 0.853 |
| 2 | = run 1 (GPU swap only) | A100 | 5k | 542k | 0.1 | 19.7 m | 10.3 GB | 0.2858 | 0.3630 | 0.0707 | 624.83 | 0.853 |
| 3 | base, 4 members, batch 10k | A100 | 20k | 542k | 0.2 | 58.9 m | 14.4 GB | 0.3149 | 0.3700 | 0.0782 | 626.19 | 0.869 |
| 4 | = run 3 + 2-stack, batch 20k | A100 | 20k | 542k | 0.2 | 48.7 m | 29.5 GB | 0.3149 | 0.3700 | 0.0782 | 626.19 | 0.869 |
| 5 | base, 4 members, 2-stack, batch 10k | A100 | **32k ↑4×** | 542k | 0.3 | 106.0 m | 30.7 GB | **0.3550** | 0.3714 | **0.0834** | 627.03 | 0.842 |
| 6 | **Ensemble-8** (crosses+SVD+NNLS), 2-stack, batch 16k | H100 | 32k ↑4× | 100k | 94.5 | 21.3 m | 44.1 GB | 0.3636 | 0.3710 | 0.0850 | 632.30 | 0.905 |
| 7 | = run 6, full frame | H100 | 32k ↑4× | 542k | 72.9 | 81.3 m | 44.1 GB | **0.3571** | **0.3725** | 0.0820 | 627.03 | 0.843 |

`↑4×` = stratified positive upsampling: 6,173 claim rows / 32,000 (19.3% vs 4.8% baseline), positives swapped in for negatives, recalibrated on an unbiased exposure-weighted 50k sample. Runs 2, 4, 7 confirm bit-exact determinism across GPUs and batching (identical metrics wherever the science config matched).

## Complete metric panel — best fair run (run 7, ensemble-8, full frame)

| Metric | Value | | Metric | Value |
|---|---|---|---|---|
| gini_weighted | **0.35706** | | asym_pricing_loss | 627.034 |
| rank_gini_weighted | 0.37248 | | apl_under_cost | 129.702 |
| spearman_rho | 0.08205 | | apl_over_cost | 108.227 |
| kendall_tau | 0.06975 | | apl_under_over_ratio | 1.1984 |
| decile_lift_monotonicity | 1.0000 | | double_lift_slope | 1.7888 |
| tweedie_deviance_p15 | 73.200 | | predicted_to_actual_ratio | 0.8434 |
| poisson_deviance | 1133.48 | | total_actual_target | 39,329,195 |
| weighted_mae_target | 160.22 | | total_predicted_target | 33,170,621 |
| weighted_rmse_target | 1151.39 | | mean_actual_rate | 137.14 |
| rmse_rate | 13,603.3 | | mean_predicted_rate | 115.67 |
| mae_rate | 406.78 | | exposure_sum | 286,772.9 |

(Older runs pre-date the full-panel dump; their retained metrics are the headline table above.)

## Failed / diagnostic runs (all intentional, cumulative cost ~$3)

| Attempt | Outcome | Lesson |
|---|---|---|
| pip `tabfm` 1.0.0 load on any GPU | crash-loop: `pytorch_model.bin` missing | HF repo went safetensors 2026-07-03; convert once into volume + `checkpoint_path` |
| T4, 5k ctx | OOM (needed >14.6 GB) | 16 GB GPUs unusable; L4 24 GB is the floor |
| L4, 5k ctx, 50k batch | >60 m timeout (vs 51.6 m @ 10k) | row attention ~quadratic in (ctx+batch): optimal outer batch ≈ context size |
| L4, 20k ctx, stacking 2–4 | OOM 22.6–27 GB | L4 = sequential members only at 20k+ |
| A100, 20k ctx, 4-stack, 30k batch | OOM >42 GB | 4-stack needs H100 class |
| H100, 32k ctx, 4-stack, 32k batch | OOM ~97 GB (22 GB single spike) | allocation spikes super-linear; H100 envelope ≈ 2-stack @ 48k rows/forward |

## Findings

1. **Accuracy:** context size and positive upsampling are the levers — 0.286 (5k) → 0.315 (20k) → 0.355 (32k↑4×). The 4× upsample + 32k jump (+0.040) is the largest single gain. TabFM-Ensemble machinery adds only **+0.0021** on a fair full-frame comparison (the 100k subsample had flattered it by +0.0065) — not worth 2× compute.
2. **Rank quality is champion-grade:** decile monotonicity 1.0, rank-gini 0.372, spearman 0.082–0.085 (GBM ~0.074). Remaining gap to 0.369 is **spread, not ordering**: double-lift slope 1.79 means predictions are ~1.8× too compressed across risk deciles, and the level under-predicts (0.843) even after recalibration.
3. **Compute:** scoring is FLOPs-bound (~50–110 min single-GPU per full fit+score at useful contexts; fit itself ≤ ~95 s). Batching/stacking bought ≤17%; GPU price/perf is flat (A100 = 2.6× L4 speed at 2.6× cost). The Phase-2 path is **fan-out scoring** over 6–8 containers (independent batches, weights volume-cached): ~10 min wall, ~$3–4/comparison.
4. **Spend:** ≈ $28 of the $30/month Modal free credits (resets monthly). No cash outlay.

## Recommendation

Integrate as the `tabfm` recipe estimator (slot already reserved in the contract) with: fan-out scoring, upsampled-context sampling as a recipe param (`upsample_positive`, exact stratified draw + unbiased recalibration), defaults 32k ctx / 4× / 4 members / no ensemble extras, `modal` backend beside TabPFN's `api` backend in `foundation.py`. Cheap accuracy probes still on the table: 6–8× upsampling, 48–64k context. **Licence: weights are non-commercial** — research use only.

*Full design & decision log: `docs/internal/tabfm_integration_plan.md`. Runner: `scripts/tabfm_modal_app.py`.*
