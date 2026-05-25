"""Statistical tests for the promotion gate: false-positive and true-positive rates.

These tests verify that the gate's statistical claims hold, not just that the
plumbing works.
"""

import numpy as np
import pandas as pd
import pytest

from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    paired_comparison,
    promotion_decision,
)


def _make_predictions(rng, n: int = 200, noise_scale: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic predictions for champion and challenger.

    ``noise_scale=0`` → identical models (null hypothesis).
    Positive ``noise_scale`` → challenger adds IID Gaussian noise (still null,
    but with random perturbations that should not trigger promotion).
    """
    exposure = np.ones(n)
    actual = rng.exponential(100.0, size=n)
    pred_champion = actual * (1 + rng.normal(0, 0.3, size=n))
    pred_challenger = pred_champion + rng.normal(0, noise_scale, size=n) if noise_scale > 0 else pred_champion.copy()

    def _df(pred):
        return pd.DataFrame({
            "record_id": np.arange(n),
            "split": ["search_validation"] * n,
            "actual_claim_cost": actual,
            "predicted_claim_cost": np.clip(pred, 1e-6, None),
            "exposure": exposure,
        })

    return _df(pred_champion), _df(pred_challenger)


def _tight_rules() -> PromotionRules:
    return PromotionRules(
        minimum_mean_lift=0.0,
        min_relative_lift=0.005,
        min_absolute_lift=0.0,
        minimum_win_rate=0.60,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.90,
        max_predicted_to_actual_drift=0.5,
        require_diagnostics=False,
        bonferroni_lookback=10,
    )


def test_null_false_positive_rate_below_threshold() -> None:
    """Under H0 (identical models), the gate should rarely promote.

    With N=30 resamples and confidence=0.90, the expected FPR is ~10%.
    We test over 100 random seeds and allow a generous 20% upper bound.
    """
    n_trials = 100
    rng_master = np.random.default_rng(42)
    promotions = 0
    rules = _tight_rules()

    for trial in range(n_trials):
        seed = int(rng_master.integers(0, 10000))
        rng = np.random.default_rng(seed)
        champion_preds, challenger_preds = _make_predictions(rng, n=400, noise_scale=0.0)
        # Add tiny IID noise to challenger (pure null)
        challenger_preds = challenger_preds.copy()
        challenger_preds["predicted_claim_cost"] *= (1 + rng.normal(0, 0.001, size=len(challenger_preds)))
        challenger_preds["predicted_claim_cost"] = challenger_preds["predicted_claim_cost"].clip(lower=1e-6)

        per_resample, comp_summary = paired_comparison(
            champion_preds, challenger_preds,
            champion_id="champ", challenger_id="chal",
            eval_split="search_validation", n_resamples=30, seed=seed,
        )
        boot = bootstrap_lift_summary(
            per_resample["lift"], iterations=200, seed=seed, confidence_level=0.90,
        )
        decision = promotion_decision(comp_summary, boot, rules)
        if decision["promoted"]:
            promotions += 1

    fpr = promotions / n_trials
    assert fpr <= 0.20, f"False positive rate {fpr:.2%} exceeds 20% — gate is too permissive"


def test_true_positive_rate_for_clear_improvement() -> None:
    """A challenger with a 10% uniform improvement should usually promote."""

    n_trials = 50
    rng_master = np.random.default_rng(99)
    promotions = 0
    rules = PromotionRules(
        minimum_mean_lift=0.0,
        min_relative_lift=0.0,   # relaxed for detectable-effect test
        min_absolute_lift=0.0,
        minimum_win_rate=0.55,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.90,
        max_predicted_to_actual_drift=0.5,
        require_diagnostics=False,
        bonferroni_lookback=1,
    )

    for trial in range(n_trials):
        seed = int(rng_master.integers(0, 10000))
        rng = np.random.default_rng(seed)
        n = 400
        exposure = np.ones(n)
        actual = rng.exponential(100.0, size=n)
        pred_champ = np.clip(actual * (1 + rng.normal(0, 0.5, size=n)), 1e-6, None)
        # Challenger is 10% closer to actuals
        pred_chal = np.clip(pred_champ * 0.9 + actual * 0.1, 1e-6, None)

        def _df(pred):
            return pd.DataFrame({
                "record_id": np.arange(n),
                "split": ["search_validation"] * n,
                "actual_claim_cost": actual,
                "predicted_claim_cost": pred,
                "exposure": exposure,
            })

        per_resample, comp_summary = paired_comparison(
            _df(pred_champ), _df(pred_chal),
            champion_id="champ", challenger_id="chal",
            eval_split="search_validation", n_resamples=30, seed=seed,
        )
        boot = bootstrap_lift_summary(
            per_resample["lift"], iterations=200, seed=seed, confidence_level=0.90,
        )
        decision = promotion_decision(comp_summary, boot, rules)
        if decision["promoted"]:
            promotions += 1

    tpr = promotions / n_trials
    assert tpr >= 0.70, f"True positive rate {tpr:.2%} below 70% — gate is too conservative for clear improvements"
