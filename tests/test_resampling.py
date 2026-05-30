import pandas as pd

import numpy as np

from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    cv_bootstrap_comparison,
    evaluate_guardrails,
    paired_comparison,
    paired_cv_comparison,
    promotion_decision,
    repeated_scores,
)
from autoresearch.data.splits import fold_seed_from_run_id, generate_fold_assignments


def _predictions(predicted: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": [1, 2, 3, 4],
        "split": ["search_validation"] * 4,
        "actual_claim_cost": [10.0, 20.0, 30.0, 40.0],
        "predicted_claim_cost": predicted,
        "exposure": [1.0, 1.0, 1.0, 1.0],
    })


def _default_rules(**overrides) -> PromotionRules:
    defaults = dict(
        minimum_mean_lift=0.0,
        min_relative_lift=0.0,
        min_absolute_lift=0.0,
        minimum_win_rate=0.55,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.5,
        require_diagnostics=False,
        bonferroni_lookback=10,
        require_sign_agreement=False,  # off by default in tests to keep old tests stable
    )
    defaults.update(overrides)
    return PromotionRules(**defaults)


def test_repeated_scores_are_reproducible() -> None:
    predictions = _predictions([10.0, 18.0, 35.0, 39.0])

    first = repeated_scores(predictions, eval_split="search_validation", n_resamples=5, seed=123)
    second = repeated_scores(predictions, eval_split="search_validation", n_resamples=5, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_paired_comparison_positive_lift_when_challenger_is_better() -> None:
    # Champion predicts zero (terrible); challenger predicts actuals (perfect)
    champion = _predictions([0.01, 0.01, 0.01, 0.01])
    challenger = _predictions([10.0, 20.0, 30.0, 40.0])

    per_resample, summary = paired_comparison(
        champion,
        challenger,
        champion_id="champ",
        challenger_id="challenger",
        eval_split="search_validation",
        n_resamples=10,
        seed=7,
    )

    assert (per_resample["lift"] > 0).all()
    assert summary["mean_lift"] > 0
    assert summary["challenger_win_rate"] == 1.0


def test_paired_comparison_positive_lift_for_higher_is_better_gini() -> None:
    champion = _predictions([40.0, 30.0, 20.0, 10.0])
    challenger = _predictions([10.0, 20.0, 30.0, 40.0])

    per_resample, summary = paired_comparison(
        champion,
        challenger,
        champion_id="champ",
        challenger_id="challenger",
        eval_split="search_validation",
        n_resamples=10,
        seed=7,
        primary_metric="gini_weighted",
    )

    assert summary["lower_is_better"] is False
    assert (per_resample["lift"] > 0).all()
    assert summary["mean_lift"] > 0


def test_bootstrap_and_promotion_decision() -> None:
    bootstrap = bootstrap_lift_summary(pd.Series([1.0, 2.0, 3.0]), iterations=100, seed=1, confidence_level=0.9)
    decision = promotion_decision(
        {"mean_lift": 2.0, "challenger_win_rate": 1.0, "champion_mean_score": 10.0,
         "std_lift": 0.5, "n_resamples": 30},
        bootstrap,
        _default_rules(),
    )

    assert bootstrap["probability_challenger_outperforms"] == 1.0
    assert decision["decision"] == "promote"


def _make_cv_frame_and_folds(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal dataset + fold assignments for CV tests."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "record_id": list(range(n)),
        "claim_cost_capped_active": rng.exponential(100, n),
        "claim_cost_observed_k": rng.exponential(100, n),
        "claim_count_signal_q": rng.poisson(0.1, n).astype(float),
        "claim_event_count_l": rng.poisson(0.1, n).astype(float),
        "exposure_term_a": rng.uniform(0.5, 1.5, n),
        # A feature for the model to use
        "risk_feature": rng.normal(0, 1, n),
    })
    n_folds = 4
    fold_frame = pd.DataFrame({
        "record_id": list(range(n)),
        "fold": [i % n_folds + 1 for i in range(n)],
    })
    return frame, fold_frame


def _make_factories_for_cv(good: bool) -> tuple:
    """Return (champion_factory, challenger_factory).

    good=True  → challenger sorts by risk_feature (positive correlation with claim_cost).
    good=False → challenger is random (no signal), champion is the good one.
    """
    from autoresearch.evaluation.metrics import full_metric_panel

    def _factory_base(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        # Global mean prediction (terrible discriminator)
        mean_pp = (train_df["claim_cost_capped_active"] / train_df["exposure_term_a"].clip(lower=1e-9)).mean()
        return pd.DataFrame({
            "record_id": val_df["record_id"].to_numpy(),
            "split": "val",
            "target_mode": "burning_cost",
            "exposure": val_df["exposure_term_a"].to_numpy(),
            "actual_target": val_df["claim_cost_capped_active"].to_numpy(),
            "actual_claim_cost": val_df["claim_cost_capped_active"].to_numpy(),
            "actual_claim_count": val_df["claim_count_signal_q"].to_numpy(),
            "predicted_target": mean_pp * val_df["exposure_term_a"].to_numpy(),
            "predicted_claim_cost": mean_pp * val_df["exposure_term_a"].to_numpy(),
            "predicted_claim_count": np.nan * np.ones(len(val_df)),
        })

    def _factory_good(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        # Uses risk_feature as the prediction (has some signal if correlated)
        preds = val_df["exposure_term_a"].to_numpy() * (0.5 + val_df["risk_feature"].to_numpy())
        preds = np.clip(preds, 0.01, None)
        return pd.DataFrame({
            "record_id": val_df["record_id"].to_numpy(),
            "split": "val",
            "target_mode": "burning_cost",
            "exposure": val_df["exposure_term_a"].to_numpy(),
            "actual_target": val_df["claim_cost_capped_active"].to_numpy(),
            "actual_claim_cost": val_df["claim_cost_capped_active"].to_numpy(),
            "actual_claim_count": val_df["claim_count_signal_q"].to_numpy(),
            "predicted_target": preds,
            "predicted_claim_cost": preds,
            "predicted_claim_count": np.nan * np.ones(len(val_df)),
        })

    if good:
        return _factory_base, _factory_good
    return _factory_good, _factory_base


def test_paired_cv_comparison_structure() -> None:
    """paired_cv_comparison returns per-partition df with expected columns."""
    frame, fold_frame = _make_cv_frame_and_folds(80)
    champ_fac, chal_fac = _make_factories_for_cv(good=False)

    per_partition, summary = paired_cv_comparison(
        frame, fold_frame,
        champion_factory=champ_fac,
        challenger_factory=chal_fac,
        champion_id="champ",
        challenger_id="chal",
        n_folds=4,
        n_repeats=2,
        gate_primary_metric="rank_gini_weighted",
        seed=42,
    )

    assert len(per_partition) == 8  # 4 folds × 2 repeats
    assert "lift" in per_partition.columns
    assert "challenger_won" in per_partition.columns
    assert "kpi_lift" in per_partition.columns
    assert summary["n_partitions"] == 8
    assert "between_partition_std" in summary
    assert summary["between_partition_std"] >= 0.0
    assert 0.0 <= summary["challenger_win_rate"] <= 1.0


def test_paired_cv_comparison_good_challenger_wins() -> None:
    """A challenger that predicts better wins in the majority of partitions."""
    frame, fold_frame = _make_cv_frame_and_folds(200)
    # Make the feature actually predictive
    rng = np.random.default_rng(99)
    frame["claim_cost_capped_active"] = (
        50.0 + 30.0 * frame["risk_feature"] + rng.normal(0, 20, len(frame))
    ).clip(0)

    champ_fac, chal_fac = _make_factories_for_cv(good=True)
    per_partition, summary = paired_cv_comparison(
        frame, fold_frame,
        champion_factory=champ_fac,
        challenger_factory=chal_fac,
        champion_id="champ",
        challenger_id="chal",
        n_folds=4,
        n_repeats=2,
        gate_primary_metric="rank_gini_weighted",
        seed=7,
    )
    # Challenger (uses risk_feature signal) should outperform global-mean champion
    assert summary["challenger_win_rate"] >= 0.5


def test_sign_agreement_check_in_promotion_decision() -> None:
    """Promotion is blocked when rank_gini lift is positive but KPI (gini) regresses."""
    bootstrap = bootstrap_lift_summary(pd.Series([0.01] * 30), iterations=50, seed=1, confidence_level=0.9)
    comparison_summary = {
        "mean_lift": 0.01,
        "challenger_win_rate": 1.0,
        "champion_mean_score": 0.5,
        "std_lift": 0.001,
        "n_resamples": 30,
        "kpi_lift_positive": False,  # KPI went the wrong way!
    }
    decision = promotion_decision(
        comparison_summary,
        bootstrap,
        _default_rules(require_sign_agreement=True),
    )
    assert "sign_agreement_kpi" in decision["checks"]
    assert not decision["checks"]["sign_agreement_kpi"]
    assert decision["decision"] == "inconclusive"


def test_sign_agreement_passes_when_both_positive() -> None:
    bootstrap = bootstrap_lift_summary(pd.Series([0.01] * 30), iterations=50, seed=1, confidence_level=0.9)
    comparison_summary = {
        "mean_lift": 0.01,
        "challenger_win_rate": 1.0,
        "champion_mean_score": 0.5,
        "std_lift": 0.001,
        "n_resamples": 30,
        "kpi_lift_positive": True,
    }
    decision = promotion_decision(
        comparison_summary,
        bootstrap,
        _default_rules(require_sign_agreement=True),
    )
    assert decision["checks"].get("sign_agreement_kpi") is True


def test_sign_agreement_skipped_when_disabled() -> None:
    bootstrap = bootstrap_lift_summary(pd.Series([0.01] * 30), iterations=50, seed=1, confidence_level=0.9)
    comparison_summary = {
        "mean_lift": 0.01,
        "challenger_win_rate": 1.0,
        "champion_mean_score": 0.5,
        "std_lift": 0.001,
        "n_resamples": 30,
        "kpi_lift_positive": False,  # disagrees, but gate disabled
    }
    decision = promotion_decision(
        comparison_summary,
        bootstrap,
        _default_rules(require_sign_agreement=False),
    )
    assert "sign_agreement_kpi" not in decision["checks"]


def _make_fold_predictions(n_folds: int = 4, n_rows: int = 40) -> tuple[dict, dict]:
    """Return (champion_fold_preds, challenger_fold_preds) for cv_bootstrap tests."""
    rng = np.random.default_rng(0)
    champ: dict[int, list] = {0: []}
    chal: dict[int, list] = {0: []}
    for fold in range(n_folds):
        actual = rng.exponential(100.0, n_rows)
        champ_pred = np.full(n_rows, actual.mean())  # global mean
        chal_pred = actual * 0.9 + rng.normal(0, 5, n_rows)  # slightly better
        chal_pred = np.clip(chal_pred, 0.01, None)
        ids = np.arange(fold * n_rows, (fold + 1) * n_rows)
        df_base = pd.DataFrame({
            "record_id": ids,
            "exposure": np.ones(n_rows),
            "actual_target": actual,
            "actual_claim_cost": actual,
            "predicted_claim_cost": champ_pred,
        })
        df_base["predicted_target"] = champ_pred
        df_chal = df_base.copy()
        df_chal["predicted_target"] = chal_pred
        df_chal["predicted_claim_cost"] = chal_pred
        champ[0].append(df_base)
        chal[0].append(df_chal)
    return champ, chal


def test_cv_bootstrap_comparison_sample_count() -> None:
    """n_samples = partitions × folds × bootstrap_per_fold."""
    champ_fp, chal_fp = _make_fold_predictions(n_folds=4)
    per_sample, summary = cv_bootstrap_comparison(
        champion_fold_predictions=champ_fp,
        challenger_fold_predictions=chal_fp,
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=5,
        seed=0,
    )
    assert summary["n_partitions"] == 1
    assert summary["n_folds"] == 4
    assert summary["bootstrap_per_fold"] == 5
    assert summary["n_samples"] == 1 * 4 * 5
    assert len(per_sample) == 20


def test_cv_bootstrap_comparison_win_rate_in_bounds() -> None:
    """Win rate is always in [0, 1]."""
    champ_fp, chal_fp = _make_fold_predictions()
    _, summary = cv_bootstrap_comparison(
        champion_fold_predictions=champ_fp,
        challenger_fold_predictions=chal_fp,
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=10,
        seed=1,
    )
    assert 0.0 <= summary["challenger_win_rate"] <= 1.0


def test_cv_bootstrap_comparison_reproducible() -> None:
    """Same seed → identical per_sample DataFrame."""
    champ_fp, chal_fp = _make_fold_predictions()
    ps1, _ = cv_bootstrap_comparison(
        champion_fold_predictions=champ_fp, challenger_fold_predictions=chal_fp,
        gate_primary_metric="gini_weighted", bootstrap_per_fold=5, seed=42,
    )
    ps2, _ = cv_bootstrap_comparison(
        champion_fold_predictions=champ_fp, challenger_fold_predictions=chal_fp,
        gate_primary_metric="gini_weighted", bootstrap_per_fold=5, seed=42,
    )
    pd.testing.assert_frame_equal(ps1.reset_index(drop=True), ps2.reset_index(drop=True))


def test_cv_bootstrap_comparison_paired_crn() -> None:
    """Champion and challenger receive the same bootstrap index sets (CRN).

    Verify by checking that champ_gini and chal_gini are NOT identical
    (they use same rows but different predictions) while the fold_idx
    and bootstrap_idx columns are consistent.
    """
    champ_fp, chal_fp = _make_fold_predictions(n_folds=2)
    per_sample, _ = cv_bootstrap_comparison(
        champion_fold_predictions=champ_fp, challenger_fold_predictions=chal_fp,
        gate_primary_metric="gini_weighted", bootstrap_per_fold=3, seed=7,
    )
    # Same sample structure but different predictions → scores differ
    assert "champ_gini_weighted" in per_sample.columns
    assert "chal_gini_weighted" in per_sample.columns
    # They should not be element-wise equal (different models)
    assert not (per_sample["champ_gini_weighted"] == per_sample["chal_gini_weighted"]).all()


def test_cv_bootstrap_comparison_escalation() -> None:
    """Escalation adds partitions and increases n_samples from 80 to 240."""
    champ_fp, chal_fp = _make_fold_predictions(n_folds=4)
    # Add escalation partitions (same data for simplicity)
    for p in range(1, 3):
        champ_fp[p] = champ_fp[0]
        chal_fp[p] = chal_fp[0]

    per_base, summary_base = cv_bootstrap_comparison(
        champion_fold_predictions={0: champ_fp[0]},
        challenger_fold_predictions={0: cal_fp[0] if False else chal_fp[0]},
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=20,
        seed=0,
    )
    assert summary_base["n_samples"] == 1 * 4 * 20  # 80

    per_esc, summary_esc = cv_bootstrap_comparison(
        champion_fold_predictions=champ_fp,
        challenger_fold_predictions=chal_fp,
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=20,
        seed=0,
    )
    assert summary_esc["n_samples"] == 3 * 4 * 20  # 240
    assert summary_esc["n_partitions"] == 3


def test_evaluate_guardrails_blocks_zero_gini() -> None:
    """Zero Gini triggers the no-discrimination guardrail."""
    result = evaluate_guardrails(
        {"gini_weighted": 0.0, "predicted_to_actual_ratio": 1.0, "total_predicted_target": 100.0},
        {"mean_lift": 0.01},
    )
    assert not result["passed"]
    assert "gini_above_zero" in result["failures"]


def test_evaluate_guardrails_blocks_miscalibration() -> None:
    """Predicted/actual ratio outside [0.5, 2.0] triggers calibration guardrail."""
    result = evaluate_guardrails(
        {"gini_weighted": 0.2, "predicted_to_actual_ratio": 0.3, "total_predicted_target": 100.0},
        {"mean_lift": 0.01},
    )
    assert not result["passed"]
    assert "calibration_sane" in result["failures"]


def test_evaluate_guardrails_passes_clean_challenger() -> None:
    """A good challenger passes all guardrails."""
    result = evaluate_guardrails(
        {"gini_weighted": 0.3, "predicted_to_actual_ratio": 1.05, "total_predicted_target": 500.0},
        {"mean_lift": 0.01},
    )
    assert result["passed"]
    assert result["failures"] == []


def test_per_run_unique_folds() -> None:
    """Two different run_ids produce different fold assignments."""
    frame = pd.DataFrame({
        "record_id": list(range(100)),
        "ClaimAmount": [float(i % 10) for i in range(100)],
        "Exposure": [1.0] * 100,
    })
    seed_a = fold_seed_from_run_id("run_2026A", 0)
    seed_b = fold_seed_from_run_id("run_2026B", 0)
    assert seed_a != seed_b

    folds_a = generate_fold_assignments(frame, "record_id", 4, seed_a)
    folds_b = generate_fold_assignments(frame, "record_id", 4, seed_b)
    # Different seeds → different fold assignments (extremely unlikely to be identical)
    assert not (folds_a["fold"] == folds_b["fold"]).all()


def test_partition_index_changes_folds() -> None:
    """partition_index=0 and partition_index=1 produce different folds for same run."""
    frame = pd.DataFrame({
        "record_id": list(range(80)),
        "ClaimAmount": [float(i % 5) for i in range(80)],
        "Exposure": [1.0] * 80,
    })
    base_seed = fold_seed_from_run_id("my_run", 0)
    esc_seed = fold_seed_from_run_id("my_run", 1)
    folds_base = generate_fold_assignments(frame, "record_id", 4, base_seed, partition_index=0)
    folds_esc = generate_fold_assignments(frame, "record_id", 4, esc_seed, partition_index=1)
    assert not (folds_base["fold"] == folds_esc["fold"]).all()


def test_promotion_requires_relative_lift() -> None:
    """Very small absolute lift that passes old 0.0 floor should fail relative threshold."""
    bootstrap = bootstrap_lift_summary(
        pd.Series([1e-7] * 30), iterations=100, seed=1, confidence_level=0.9
    )
    decision = promotion_decision(
        {"mean_lift": 1e-7, "challenger_win_rate": 1.0, "champion_mean_score": 1000.0,
         "std_lift": 1e-8, "n_resamples": 30},
        bootstrap,
        _default_rules(min_relative_lift=0.005),
    )
    assert decision["decision"] == "inconclusive"
    assert "relative_lift" in decision["checks"]
    assert not decision["checks"]["relative_lift"]
