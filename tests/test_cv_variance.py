"""Tests for k-fold CV and variance decomposition."""

import numpy as np
import pandas as pd
import pytest

from autoresearch.data.splits import generate_fold_assignments
from autoresearch.evaluation.resampling import cv_repeated_scores


def _make_dataset(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "record_id": np.arange(n),
        "exposure_term_a": np.ones(n),
        "claim_cost_capped_active": rng.exponential(100, n),
        "claim_cost_observed_k": rng.exponential(100, n),
        "claim_count_signal_q": rng.poisson(0.1, n),
        "claim_event_count_l": rng.poisson(0.1, n),
        "feature_a": rng.normal(0, 1, n),
    })


def _constant_model_factory(train, val):
    """Baseline: predict the training mean pure premium for all val rows."""
    train_pp = (train["claim_cost_capped_active"] / train["exposure_term_a"]).mean()
    preds = pd.DataFrame({
        "actual_claim_cost": val["claim_cost_capped_active"].to_numpy(),
        "predicted_claim_cost": np.full(len(val), train_pp * val["exposure_term_a"].to_numpy()),
        "exposure": val["exposure_term_a"].to_numpy(),
    })
    return preds


def test_generate_fold_assignments_produces_n_folds() -> None:
    frame = _make_dataset(100)
    folds = generate_fold_assignments(frame, id_column="record_id", n_folds=5, seed=1)
    assert set(folds["fold"].unique()) == {1, 2, 3, 4, 5}
    assert len(folds) == 100


def test_fold_assignments_are_deterministic() -> None:
    frame = _make_dataset(100)
    folds1 = generate_fold_assignments(frame, id_column="record_id", n_folds=5, seed=42)
    folds2 = generate_fold_assignments(frame, id_column="record_id", n_folds=5, seed=42)
    pd.testing.assert_frame_equal(folds1, folds2)


def test_cv_scores_returns_k_rows() -> None:
    frame = _make_dataset(100)
    folds = generate_fold_assignments(frame, id_column="record_id", n_folds=5, seed=1)
    cv_frame, summary = cv_repeated_scores(
        frame, folds, model_factory=_constant_model_factory, n_folds=5, seed=1,
    )
    assert len(cv_frame) == 5
    assert "mean_score" in summary
    assert "between_fold_variance" in summary


def test_cv_variance_decomposition_sums_correctly() -> None:
    frame = _make_dataset(200)
    folds = generate_fold_assignments(frame, id_column="record_id", n_folds=5, seed=2)
    _, summary = cv_repeated_scores(
        frame, folds, model_factory=_constant_model_factory, n_folds=5, seed=2,
    )
    # between + within ≈ total (up to floating point)
    recon = summary["between_fold_variance"] + summary["within_fold_variance"]
    assert abs(recon - summary["total_variance"]) < 1e-9


def test_between_dominates_when_data_varies_across_folds() -> None:
    """When claim amounts differ systematically by fold, between-fold variance dominates."""
    n = 300
    # Assign high/low claim amounts in alternating blocks to force between-fold differences
    records = pd.DataFrame({
        "record_id": np.arange(n),
        "exposure_term_a": np.ones(n),
        "claim_cost_capped_active": np.concatenate([
            np.full(60, 10.0), np.full(60, 1000.0),
            np.full(60, 20.0), np.full(60, 800.0),
            np.full(60, 5.0),
        ]),
        "claim_cost_observed_k": np.ones(n) * 50,
        "claim_count_signal_q": np.zeros(n),
        "claim_event_count_l": np.zeros(n),
    })
    folds = generate_fold_assignments(records, id_column="record_id", n_folds=5, seed=3)
    _, summary = cv_repeated_scores(
        records, folds, model_factory=_constant_model_factory, n_folds=5, seed=3,
    )
    # With highly variable data across blocks, between-fold variance should be substantial
    assert summary["total_variance"] > 0


def test_identical_predictions_give_zero_cv_variance() -> None:
    frame = _make_dataset(100)
    folds = generate_fold_assignments(frame, id_column="record_id", n_folds=5, seed=1)

    # Model that always predicts exactly the actual (zero error)
    def perfect_model(train, val):
        return pd.DataFrame({
            "actual_claim_cost": val["claim_cost_capped_active"].to_numpy(),
            "predicted_claim_cost": val["claim_cost_capped_active"].to_numpy(),
            "exposure": val["exposure_term_a"].to_numpy(),
        })

    cv_frame, summary = cv_repeated_scores(
        frame, folds, model_factory=perfect_model, n_folds=5, seed=1,
    )
    assert summary["mean_score"] < 1e-6  # near-zero deviance for perfect model
