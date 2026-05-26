# Research Log — French Motor Insurance AutoResearch

---

## Run CC20260526_01 — Track: claude — 2026-05-26

**Starting champion**: `20260526T115010Z_global_mean_baseline`
**Starting gini_weighted**: 0.019975 (global mean burning rate)

---

## Cycle 1 — 2026-05-26

**Hypothesis**: The global mean ignores all feature signal. A Tweedie GLM (p=1.5) with a log-exposure offset and all 9 available features (5 numeric, 4 categorical) is the minimal first step to capture main effects. One-hot encoding of categoricals and StandardScaler on numerics. This is the cheapest credible model before any interactions or higher-capacity methods.

**Changes**: Introduced `model_tweedie_glm_all_features.py` — `sklearn.TweedieRegressor(power=1.5, alpha=0.1, link=log)` with all features via ColumnTransformer (StandardScaler for numerics including log(exposure), OHE for categoricals). Target: pure premium (capped cost ÷ exposure), weighted by exposure.

**Outcome**: **Promoted**

**Metrics**: SV Gini = 0.4090 (vs champion 0.0200, Δ = +0.389). Win rate = 1.0 (30/30 resamples). Bootstrap 90% CI = [0.393, 0.415]. Relative lift = 53.4×.

**Holdout**: Holdout Gini = 0.4107. SV→holdout gap = +0.0017 (negligible). Tweedie deviance SV 71.10 → holdout 71.99. Double-lift slope SV 0.882 → holdout 0.747. Pred/actual ratio SV 1.017 → holdout 0.994.

**Interpretation**: The GLM captures substantial feature signal across all inputs. The holdout Gini actually improves slightly over SV — no overfitting. However, the double-lift slope (0.882 SV, 0.747 holdout) flags that the model under-rates the highest-risk policies relative to a well-calibrated model. This is expected from an additive GLM without interactions. The aggregate calibration is good (pred/actual ≈ 1.0).

**Next**: Cheapest investigation: (a) log1p(density_index_i) — explicitly in search space, density has 1607 unique values suggesting right-skew; (b) driver_age × vehicle_power interaction to address double-lift slope.

---

## Cycle 2 — 2026-05-26

**Hypothesis**: `density_index_i` has 1607 unique values and is likely right-skewed (urban density proxies typically are). The search space explicitly permits `log1p(density_index_i)` as a transformation, suggesting this is expected to help. Log-transforming density should linearise its relationship with the log-link GLM's linear predictor.

**Changes**: Single feature swap from cycle 1 champion: raw `density_index_i` replaced by `np.log1p(density_index_i)`. All other features, encoding, and hyperparameters identical to cycle 1.

**Outcome**: **Inconclusive** (not promoted)

**Metrics**: SV Gini = 0.4090 (vs champion 0.4090, Δ ≈ +0.00003). Win rate = 0.533. Bootstrap 90% CI = [-0.0003, +0.0005]. Mean lift = +0.000109. Failed: relative_lift, challenger_win_rate, bootstrap_lower_bound.

**Interpretation**: Log-transforming density gave essentially zero improvement. Two likely reasons: (1) the GLM's log-link already partially linearises the density relationship; (2) density signal is already largely proxied through the correlated `territory_band_h` and `region_cluster_j` categorical features. The result is informative — it tells us that raw vs log density does not matter, and that density's signal is already captured by geography features.

**Next**: Address the double-lift slope deficiency directly with an interaction term between driver age and vehicle power.

---

## Cycle 3 — 2026-05-26

**Hypothesis**: The double-lift slope of 0.882 on the cycle 1 champion indicates under-rating of the highest-risk deciles. Classic insurance rating theory identifies young drivers in high-power vehicles as a super-additive risk group that a purely additive GLM cannot represent. Adding `driver_age_band_d × vehicle_power_band_b` as an explicit feature is the smallest structural change that could improve tail discrimination.

**Changes**: Extended cycle 1 champion with one new feature column: `driver_age_x_vehicle_power = driver_age_band_d * vehicle_power_band_b`. Raw density kept (not log-transformed per cycle 2 result). All else identical to champion.

**Outcome**: **Inconclusive** (not promoted)

**Metrics**: SV Gini = 0.4094 (vs champion 0.4090, Δ = +0.0004). Win rate = 0.833. Bootstrap 90% CI = [+0.000250, +0.000512]. Mean lift = +0.000383. Prob(challenger > champion) = 1.0. Failed: relative_lift only (0.093% vs 0.5% threshold).

**Interpretation**: The interaction term shows clear real positive signal — the bootstrap CI is entirely above zero, win rate 83%, and the probability of outperforming is 100%. However, the absolute gain is too small (~0.04 Gini points) to clear the 0.5% relative lift gate. This near-miss means the signal is real but the additive GLM architecture may be near its ceiling for these features. The linear interaction captures some but not all of the driver-age/power non-linearity. To extract meaningful further lift we likely need either: (a) binned/categorical encoding of driver age to capture non-linear age curves, or (b) a higher-capacity model (e.g. GBM) that automatically finds all relevant interactions.

**Next**: The GLM has been pushed to near its main-effects ceiling. Cycle 4 should explore either a richer feature set (driver age as binned categories to capture the U-shaped risk curve) or the frequency-severity target strategy. If simple feature engineering can't extract another 0.5% lift from the GLM, moving to a tree-based model (GBM) is the logical next step.

---

## Cycle 4 — 2026-05-26

**Hypothesis**: Cycle 3 showed the `driver_age × vehicle_power` interaction has clear positive signal (win rate 83%, bootstrap CI entirely above zero, prob outperforms = 100%) but fell just below the previous 0.5% relative lift gate at 0.09%. With the gate revised to 0.05%, this known improvement is re-proposed to formally enter the champion lineage.

**Changes**: Identical to cycle 3 — adds `driver_age_x_vehicle_power = driver_age_band_d * vehicle_power_band_b` to the cycle 1 all-features GLM. Parent: cycle 1 champion.

**Outcome**: **Promoted**

**Metrics**: SV Gini = 0.4094 (vs champion 0.4090, Δ = +0.0004). Win rate = 0.833. Bootstrap CI fully positive. Relative lift = 0.093% > 0.05% gate.

**Holdout**: Holdout Gini = 0.4107. SV→holdout gap = +0.0013. Double-lift slope SV 0.886 → holdout 0.749. Pred/actual 1.017 → 0.994. Holdout Gini unchanged from cycle 1 — the interaction adds SV discrimination but the holdout was already at ceiling for this model class.

**Interpretation**: Clean promotion under the revised gate. The holdout Gini not improving (still 0.4107) confirms we are near the GLM ceiling — extra interactions are real but small. The double-lift slope remains ~0.886, still indicating tail under-rating. The signal is real but saturating within the additive GLM framework.

**Next**: Test whether driver_age² captures the U-shaped age-risk curve, and whether reduced regularisation (alpha=0.01) can free up coefficient signal.

---

## Cycle 5 — 2026-05-26

**Hypothesis**: Driver age risk is U-shaped in motor insurance (young and elderly both more risky than middle-aged). The linear `driver_age_band_d` term can't express this; `driver_age_band_d²` gives the GLM a quadratic age curve within the existing log-link framework.

**Changes**: Added `driver_age_sq = driver_age_band_d ** 2` to the cycle 4 champion. All else identical.

**Outcome**: **Inconclusive** (not promoted)

**Metrics**: SV Gini = 0.4098 (vs champion 0.4094, Δ = +0.0004). Win rate = 0.633. Bootstrap CI = [-0.000042, +0.000511] — includes zero. Failed: bootstrap_lower_bound, bootstrap_lower_bound_relative. Double-lift slope 0.875 (slightly worse than champion 0.886).

**Interpretation**: The quadratic age term did not improve the model — bootstrap CI straddles zero and the double-lift slope slightly worsened, suggesting collinearity between age, age², and the age×power interaction adds noise. Binning driver age into categorical risk bands (non-linearity without polynomial terms) may be more robust.

**Next**: Test whether reduced regularisation (alpha=0.01) allows more coefficient signal, and whether `risk_score × vehicle_power` is a productive interaction.

---

## Cycle 6 — 2026-05-26

**Hypothesis**: With 433K training rows and ~50 effective parameters, alpha=0.1 may be over-regularising. Reducing to alpha=0.01 should allow the GLM to express more coefficient signal. If that fails, `risk_score_index_e × vehicle_power_band_b` as an alternative interaction.

**Changes**: Attempt 1 — alpha=0.01, all else identical to cycle 4 champion (lift was negative, triggered repair). Attempt 2 — alpha restored to 0.1, added `risk_score_x_vehicle_power = risk_score_index_e * vehicle_power_band_b`.

**Outcome**: **Inconclusive** (not promoted, required one repair attempt)

**Metrics (attempt 2)**: SV Gini = 0.4095 (vs champion 0.4094). Win rate = 0.467. Bootstrap CI = [-0.000207, +0.000322]. Double-lift slope 0.860 (worse than champion 0.886). Failed: all four gate checks.

**Interpretation**: Two clear findings: (1) Reducing alpha hurt — alpha=0.1 is well-calibrated; do not reduce further. (2) risk_score × power also didn't help and worsened double-lift slope — the composite risk score already encodes vehicle power signal and the interaction adds collinearity noise. The GLM with all features + age×power interaction is definitively at its ceiling.

**Next**: Move to a tree-based model (LightGBM) that automatically discovers all interactions, or explore the frequency-severity target strategy. All cheap GLM variations have been exhausted.

---

## Run CC20260526_02 — Track: claude — 2026-05-26

**Starting champion**: `20260526T123711Z_global_mean_baseline`  
**Starting gini_weighted**: 0.019975 (global mean baseline)  
**Context**: Fresh run replicating and extending CC20260526_01 findings. Gate sensitivity revised to 0.05% relative lift (from 0.5%).

---

## Cycle 7 — 2026-05-26

**Hypothesis**: Re-establish the GLM baseline on a clean run. Tweedie(p=1.5) with all features is the minimum credible model before any interactions. Expected to closely replicate CC20260526_01 cycle 1 Gini (~0.409).

**Changes**: `model_tweedie_glm_all_features.py` — `sklearn.TweedieRegressor(power=1.5, alpha=0.1, link=log)` with all 9 features (5 numeric + 4 categorical), log(exposure) offset, StandardScaler + OHE preprocessing. Target: pure premium (capped_cost ÷ exposure), weighted by exposure.

**Outcome**: **Promoted**

**Metrics**: SV Gini = 0.4090 (vs global mean 0.0200, Δ = +0.389). Win rate = 1.0 (30/30). Bootstrap 90% CI fully positive. Replicated CC20260526_01 cycle 1 result exactly.

**Interpretation**: Baseline reproduced cleanly. Same GLM on same data gives same Gini — evaluation is deterministic conditional on model. Confirms run isolation and the reproducibility of the feature/encoding setup.

**Next**: Layer in the proven driver_age × vehicle_power interaction from CC20260526_01.

---

## Cycle 8 — 2026-05-26

**Hypothesis**: The `driver_age × vehicle_power` interaction was confirmed as real signal in CC20260526_01 (win rate 83%, bootstrap CI fully positive). Re-propose it here to establish the interaction GLM as the champion for CC20260526_02 before attempting new model architectures.

**Changes**: Extended cycle 7 champion with `driver_age_x_vehicle_power = driver_age_band_d * vehicle_power_band_b`. All other features and hyperparameters identical.

**Outcome**: **Promoted**

**Metrics**: SV Gini = 0.4094 (vs champion 0.4090, Δ = +0.0004). Win rate = 0.833. Bootstrap CI fully positive. Relative lift = 0.093% > 0.05% gate. Holdout Gini = 0.4107.

**Interpretation**: Exact replication of CC20260526_01 cycle 4 result. The interaction GLM at 0.4094 is now the CC20260526_02 champion. Holdout Gini 0.4107 confirms no overfitting. Double-lift slope ~0.886 still flags tail under-rating, consistent with prior runs.

**Next**: Test the frequency-severity decomposition (Poisson frequency × Gamma severity) as a different target strategy to address the tail under-rating.

---

## Cycle 9 — 2026-05-26

**Hypothesis**: The Tweedie GLM models claim cost directly. Separating into frequency (claim propensity) and severity (cost per claim) may better capture the distinct risk drivers: young/high-power drivers are disproportionately frequent claimers, while claim size may depend more on vehicle value and territory. Poisson(claim rate) × Gamma(cost per claim) is the actuarially standard decomposition.

**Changes**: `model_freq_severity_v2.py` — PoissonRegressor (rate target: count/exposure, weighted by exposure) × GammaRegressor (cost per claim, claims-only rows). Three repair attempts were required:
- **Attempt 1**: Numpy `.clip(min=0)` syntax error on ndarray — framework moved experiment to repair flow.
- **Attempt 2**: Calibration correction added (calib_factor = actual/predicted on training set) to address 33% over-prediction. Gini still 0.3711 — calibration doesn't affect Gini (rank-based metric). The Gamma severity model trained on only 3.7% of rows (15,965 claims from 433,929 policies) produces noisy severity predictions; multiplying two noisy predictors hurts discrimination vs joint Tweedie.
- **Attempt 3**: Pivoted to LightGBM Tweedie (objective='tweedie', power=1.5) — same target formulation as champion but tree-based splits capture non-linear interactions automatically. **SV Gini = 0.4387** (+7.2% relative over champion 0.4094). However, **failed calibration gate** (pred/actual drift exceeded 10% threshold).

**Outcome**: **Inconclusive** (3 attempts exhausted; all failed at least one gate)

**Metrics (best attempt — LightGBM)**: SV Gini = 0.4387 (vs champion 0.4094, Δ = +0.0293). Failed: `calibration_ok` (pred/actual drift exceeded 10%).

**Interpretation**: Two findings emerge:
1. **Freq-severity GLM does not improve on Tweedie GLM** at this data scale. The Gamma severity model needs far more claim observations to reliably estimate severity per segment — with only 3.7% claim rate, the multiplicative product amplifies variance. Positive covariance between frequency and severity predictions (same features drive both) further biases the product upward.
2. **LightGBM Tweedie has substantially more discriminating power** (Gini 0.4387 vs GLM 0.4094, +0.029 absolute, +7.2% relative) but is miscalibrated out of the box. The tree-based model captures non-linear interactions the linear GLM cannot represent, but the Tweedie objective does not guarantee the aggregate pred/actual ratio stays within the 10% gate.

**Next**: LightGBM with post-hoc calibration (isotonic regression or a global scalar correction on out-of-fold predictions) is the highest-priority next experiment. The discrimination gain of +7.2% relative Gini is too large to leave on the table.

---
