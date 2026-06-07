"""Tests for autoresearch.models.interpretation — automatic pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from autoresearch.models.interpretation import compute_automatic_interpretation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Minimal eval frame with all columns compute_automatic_interpretation expects."""
    rng = np.random.default_rng(seed)
    exposure = rng.uniform(0.5, 1.5, n)
    age = rng.integers(20, 65, n).astype(float)
    vehicle_age = rng.integers(0, 15, n).astype(float)
    region = rng.choice(["North", "South", "East"], n)
    # Predicted rate = some function of features
    pred_rate = 100.0 + age * 2.0 + vehicle_age * 5.0 + rng.normal(0, 10, n)
    actual_rate = pred_rate * rng.uniform(0.8, 1.2, n)
    return pd.DataFrame({
        "record_id": np.arange(n),
        "split": ["search_validation"] * n,
        "exposure": exposure,
        "actual_claim_cost": actual_rate * exposure,
        "actual_claim_cost_uncapped": actual_rate * exposure,
        "actual_claim_count": np.zeros(n),
        "actual_claim_event_count": np.zeros(n),
        "actual_target": actual_rate * exposure,
        "predicted_target": pred_rate * exposure,
        "predicted_claim_cost": pred_rate * exposure,
        "predicted_claim_count": np.full(n, np.nan),
        "actual_pure_premium": actual_rate,
        "predicted_pure_premium": pred_rate,
        "actual_frequency": np.zeros(n),
        "predicted_frequency": np.full(n, np.nan),
        # feature columns
        "age": age,
        "vehicle_age": vehicle_age,
        "region": region,
    })


# ---------------------------------------------------------------------------
# TestComputeAutomaticInterpretation
# ---------------------------------------------------------------------------

class TestComputeAutomaticInterpretation:

    def test_returns_dict(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age", "vehicle_age", "region"])
        assert isinstance(result, dict)

    def test_pdp_data_present(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age", "vehicle_age"])
        assert "pdp_data" in result
        assert len(result["pdp_data"]) == 2

    def test_pdp_data_schema(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age"])
        entry = result["pdp_data"][0]
        assert entry["feature"] == "age"
        assert "x" in entry
        assert "y_pred" in entry
        assert "y_actual" in entry
        assert "ae_ratio" in entry
        assert "exposure" in entry
        assert entry["x_is_numeric"] is True
        assert entry["source"] == "one_way"

    def test_one_way_lengths_consistent(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age"])
        entry = result["pdp_data"][0]
        n = len(entry["x"])
        assert len(entry["y_pred"]) == n
        assert len(entry["y_actual"]) == n
        assert len(entry["ae_ratio"]) == n
        assert len(entry["exposure"]) == n

    def test_categorical_feature_x_is_numeric_false(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["region"])
        assert "pdp_data" in result
        entry = result["pdp_data"][0]
        assert entry["x_is_numeric"] is False

    def test_feature_importance_present(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age", "vehicle_age"])
        assert "feature_importance" in result
        fi = result["feature_importance"]
        assert isinstance(fi, list)
        assert len(fi) == 2
        for item in fi:
            assert "feature" in item
            assert "importance" in item
            assert "importance_type" in item

    def test_feature_importance_sorted_descending(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age", "vehicle_age", "region"])
        fi = result["feature_importance"]
        importances = [d["importance"] for d in fi]
        assert importances == sorted(importances, reverse=True)

    def test_empty_feature_cols_returns_empty(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, [])
        assert result == {}

    def test_nonexistent_feature_cols_skipped(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["nonexistent_col"])
        # Should return empty or partial — not raise
        assert isinstance(result, dict)

    def test_output_is_json_serialisable(self):
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age", "vehicle_age", "region"])
        serialised = json.dumps(result)
        assert len(serialised) > 10

    def test_ae_ratio_approximately_one_when_actual_equals_pred(self):
        """When actual rate == predicted rate, A/E should be ~1.0 everywhere."""
        df = _make_eval_df()
        # Override actual to equal predicted exactly
        df["actual_target"] = df["predicted_target"]
        df["actual_claim_cost"] = df["predicted_claim_cost"]
        df["actual_pure_premium"] = df["predicted_pure_premium"]
        result = compute_automatic_interpretation(df, ["age"])
        entry = result["pdp_data"][0]
        for ae in entry["ae_ratio"]:
            assert abs(ae - 1.0) < 0.05, f"Expected A/E ~1.0 but got {ae}"

    def test_with_interpret_fn_returns_pdp_marginal(self):
        """When interpret_fn is supplied, pdp_data source should be 'pdp_marginal'."""
        df = _make_eval_df()

        def interpret_fn(feat_df: pd.DataFrame) -> np.ndarray:
            return 100.0 + feat_df["age"].values * 2.0 + feat_df["vehicle_age"].values * 5.0

        result = compute_automatic_interpretation(
            df, ["age", "vehicle_age"],
            interpret_fn=interpret_fn,
            n_pdp_sample=200,
        )
        assert "pdp_data" in result
        sources = {e["source"] for e in result["pdp_data"]}
        assert "pdp_marginal" in sources

    def test_with_interpret_fn_feature_importance_type_permutation(self):
        """With interpret_fn, permutation importance should be used."""
        df = _make_eval_df()

        def interpret_fn(feat_df: pd.DataFrame) -> np.ndarray:
            return 100.0 + feat_df["age"].values * 2.0

        result = compute_automatic_interpretation(
            df, ["age", "vehicle_age"],
            interpret_fn=interpret_fn,
        )
        fi = result.get("feature_importance", [])
        assert len(fi) > 0
        # With interpret_fn, permutation importance is returned
        types = {d["importance_type"] for d in fi}
        assert "permutation" in types

    def test_shap_summary_structure_when_present(self):
        """If shap_summary is returned, it must have the expected keys."""
        df = _make_eval_df()
        result = compute_automatic_interpretation(df, ["age", "vehicle_age", "region"])
        if "shap_summary" not in result:
            pytest.skip("shap or lightgbm not installed in this environment")
        shap = result["shap_summary"]
        assert "features" in shap
        assert "mean_abs_shap" in shap
        assert len(shap["features"]) == len(shap["mean_abs_shap"])
        assert shap.get("source") == "surrogate"

    def test_result_stable_across_repeated_calls(self):
        """Same inputs should produce identical JSON output (random_state fixed)."""
        df = _make_eval_df()
        r1 = compute_automatic_interpretation(df, ["age", "vehicle_age"], random_state=7)
        r2 = compute_automatic_interpretation(df, ["age", "vehicle_age"], random_state=7)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
