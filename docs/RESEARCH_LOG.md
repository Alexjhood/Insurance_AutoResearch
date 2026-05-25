# Research Log — Insurance AutoResearch

Append-only. One entry per research cycle. The framework auto-appends a one-line comparison result table row; the agent should add a full entry before and after each cycle.

**Format for a cycle entry:**

```
## Cycle N — YYYY-MM-DD
**Hypothesis**: ...
**Changes**: ...
**Outcome**: promoted / inconclusive / failed
**Metrics**: SV Tweedie deviance = X.XXXXX (vs champion Y.YYYYY, Δ = ...)
**Holdout**: (if promoted) Tweedie deviance = X.XXXXX, SV→holdout gap = ...
**Interpretation**: ...
**Next**: ...
```

---

## Auto-comparison log

The framework appends one row here after every comparison:

| Timestamp | Challenger | Decision | Lift | Win-rate | Rationale |
|-----------|-----------|----------|------|----------|-----------|

---

## Cycle 0 — 2026-05-24 (baselines)

**Hypothesis**: Establish baselines before autonomous search begins.

**Changes**: Ran two baseline experiments — regularised Ridge (direct pure premium) and Tweedie GLM + freq×sev GLM (both via `run-all-baselines`).

**Outcome**: Baselines registered. Official champion initialised as the direct pure premium baseline.

**Metrics (search_validation)**:
- Ridge baseline: Tweedie deviance ≈ (see registry) — note mean_predicted_pure_premium ≈ 0.44 vs actual ≈ 122, indicating the Ridge log1p target retransformation is not exposure-corrected. This is a known artefact of the regularized_linear legacy family.
- Tweedie GLM: primary model family going forward.

**Interpretation**: The regularized_linear family has a calibration bug (predicted PP ≈ 0 relative to actual ≈ 122). The LLM should focus on `tweedie_glm`, `frequency_severity_glm`, and `tweedie_gbm` families, and consider adding better-calibrated alternatives (LightGBM, XGBoost, monotone GBM).

**Next**: Run a proper Tweedie GLM baseline, then explore:
1. Feature engineering — interactions (age × power, region × density), log transforms, derived variables.
2. GBM with Poisson loss — LightGBM native Tweedie objective with monotone constraints.
3. Frequency×severity with richer severity model (Gamma vs log-normal).
4. Calibration overlay — isotonic regression on top of the GLM.

## Cycle 1 — 2026-05-25

**Hypothesis**: A first cheap Tweedie GLM with lower regularisation could capture segment signal beyond the flat global mean.
**Changes**: Ran `mock_20260525T155848085752Z_0_tweedie_glm`, Tweedie GLM with `alpha=0.3`, `power=1.5`, fixed claim cap 100000.
**Outcome**: inconclusive; not promoted.
**Metrics**: SV Gini = -0.24636 (vs champion -0.24636, delta = +0.00000); comparison lift = +0.0000, win-rate = 0.00.
**Holdout**: no milestone report because there was no promotion.
**Interpretation**: The candidate reproduced the global-mean ranking and retained the same 16.4% aggregate overprediction on search-validation, so it failed relative lift, win-rate, and calibration checks.
**Next**: Try a small GLM variant that changes tail handling before moving to more capacity.

## Cycle 2 — 2026-05-25

**Hypothesis**: A slightly more tail-heavy Tweedie GLM could improve heavy-claim calibration while remaining a simple model-form change.
**Changes**: Ran `mock_20260525T155909724236Z_1_tweedie_glm`, Tweedie GLM with `alpha=3.0`, `power=1.7`, fixed claim cap 100000.
**Outcome**: inconclusive; not promoted.
**Metrics**: SV Gini = -0.24636 (vs champion -0.24636, delta = +0.00000); comparison lift = +0.0000, win-rate = 0.00.
**Holdout**: no milestone report because there was no promotion.
**Interpretation**: Changing GLM power/regularisation did not create rank lift and left calibration outside the promotion drift threshold.
**Next**: Test whether separating claim frequency and severity adds signal that the direct GLM misses.

## Cycle 3 — 2026-05-25

**Hypothesis**: A frequency x severity GLM may expose actuarial signal that direct pure-premium GLMs are not using.
**Changes**: Ran `mock_20260525T155931917353Z_2_frequency_severity_glm`, Poisson frequency GLM x severity GLM with `freq_alpha=1.0`, `sev_alpha=0.5`, fixed claim cap 100000.
**Outcome**: inconclusive; not promoted.
**Metrics**: SV Gini = -0.24636 (vs champion -0.24636, delta = +0.00000); comparison lift = +0.0000, win-rate = 0.00.
**Holdout**: no milestone report because there was no promotion.
**Interpretation**: The split model produced no rank lift and materially worsened aggregate calibration, with predicted-to-actual ratio 1.584 on search-validation.
**Next**: Calibration must be addressed before a frequency/severity path is viable; if continuing automated search, constrain proposals toward cheap feature transforms or explicit recalibration.

## Cycle 4 — 2026-05-25

**Hypothesis**: A modest Tweedie GBM could capture nonlinear feature interactions that the simple GLM candidates missed.
**Changes**: Ran `mock_20260525T155955608360Z_3_tweedie_gbm`, Tweedie GBM with `max_iter=300`, `max_depth=5`, `learning_rate=0.05`, `min_samples_leaf=200`, fixed claim cap 100000.
**Outcome**: inconclusive; not promoted.
**Metrics**: SV Gini = 0.22319 (vs champion -0.24636, delta = +0.46955); comparison lift = +0.4720, win-rate = 1.00.
**Holdout**: no milestone report because there was no promotion.
**Interpretation**: The GBM delivered strong rank lift and lower Tweedie deviance, but failed promotion on calibration because search-validation predicted-to-actual ratio was 1.160, outside the 5% drift threshold.
**Next**: The rank signal is real, but any follow-up should make one calibration-focused change rather than adding more capacity.

## Cycle 5 — 2026-05-25

**Hypothesis**: A deeper GBM with a lower learning rate and larger leaves might preserve GBM rank lift while stabilising predictions.
**Changes**: Ran `mock_20260525T160020756615Z_4_tweedie_gbm`, Tweedie GBM with `max_iter=800`, `max_depth=7`, `learning_rate=0.03`, `min_samples_leaf=500`, fixed claim cap 100000.
**Outcome**: inconclusive; not promoted.
**Metrics**: SV Gini = 0.23727 (vs champion -0.24636, delta = +0.48363); comparison lift = +0.4834, win-rate = 1.00.
**Holdout**: no milestone report because there was no promotion.
**Interpretation**: This was the best ranker of the five cycles and slightly improved aggregate calibration over the prior GBM, but predicted-to-actual ratio remained 1.150, still outside the promotion gate.
**Next**: Use the GBM signal diagnostically, then try an explicit exposure-weighted calibration scalar or return to cheap feature engineering with calibration constraints.
