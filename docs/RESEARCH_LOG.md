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
