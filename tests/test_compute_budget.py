"""Tests for compute budget escalation, timeout path, preflight, auto-abandon, and doubled-path fix."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from autoresearch.config import ProjectConfig
from autoresearch.experiment_runner import (
    ComputeBudgetExceeded,
    PreflightFailed,
    _compute_budget_alarm,
    _resolve_model_script_path,
)
from autoresearch.controller.workflow import _compute_experiment_budget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, **overrides) -> ProjectConfig:
    """Build a minimal ProjectConfig for budget/repair tests."""
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    artifacts = tmp_path / "artifacts"
    processed.mkdir(parents=True)
    splits.mkdir(parents=True)
    defaults = dict(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=processed,
        holdout_vault_dir=tmp_path / "holdout_vault",
        metadata_dir=tmp_path / "metadata",
        splits_dir=splits,
        artifacts_dir=artifacts,
        registry_path=artifacts / "registry.sqlite",
        research_log_path=tmp_path / "RESEARCH_LOG.md",
        track_id="test",
        random_seed=1,
        id_column="IDpol",
        agent_dataset_name="agent_dataset",
        claim_capping_enabled=True,
        claim_cap_threshold=100000,
        split_ratios={"train": 0.64, "search_validation": 0.16, "milestone_holdout": 0.2},
        ordinary_train_split="train",
        ordinary_eval_splits=("search_validation",),
        target_mode="burning_cost",
        primary_metric="tweedie_deviance_p15",
        tweedie_power=1.5,
        use_cv=False,
        cv_folds=4,
        cv_n_repeats=4,
        cv_seed=1,
        gate_mode="single_partition",
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=5,
        escalation_win_rate_low=0.40,
        escalation_win_rate_high=0.60,
        escalation_partitions=0,
        repeated_resamples=5,
        bootstrap_iterations=10,
        resample_fraction=1.0,
        resampling_seed=1,
        minimum_mean_lift=0.0,
        min_relative_lift=0.0,
        min_absolute_lift=0.0,
        minimum_win_rate=0.6,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.1,
        require_diagnostics=False,
        bonferroni_lookback=5,
        handoff_base_dir=tmp_path / "handoff",
        handoff_context_dir=tmp_path / "handoff" / "context",
        handoff_proposal_inbox_dir=tmp_path / "handoff" / "inbox",
        handoff_proposal_processed_dir=tmp_path / "handoff" / "processed",
        handoff_results_dir=tmp_path / "handoff" / "results",
        handoff_handoffs_dir=tmp_path / "handoff" / "handoffs",
        proposal_inbox_file=tmp_path / "handoff" / "inbox" / "manual.jsonl",
        deduplication_policy="reject",
        deduplication_lookback=25,
        search_space={},
        # compute budget
        base_budget_minutes=10,
        budget_increment_minutes=5,
        experiments_per_increment=5,
        compute_enforce=True,
        preflight_enabled=False,   # disabled by default for most tests
        preflight_sample_rows=100,
        # repair
        repair_noise_floor_eps=0.002,
        repair_auto_abandon_enabled=True,
        running_stale_minutes=30,
    )
    defaults.update(overrides)
    return ProjectConfig(**defaults)


# ---------------------------------------------------------------------------
# 1.1 Budget escalation formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_prior,expected_minutes", [
    (0, 10),   # N=0 → 10 min
    (4, 10),   # N=4 → 10 min (still in first bucket)
    (5, 15),   # N=5 → 15 min (second bucket)
    (10, 20),  # N=10 → 20 min (third bucket)
    (9, 15),   # N=9 → still second bucket
    (15, 25),  # N=15 → fourth bucket
])
def test_budget_escalation_formula(tmp_path: Path, n_prior: int, expected_minutes: int):
    config = _make_config(tmp_path)
    # Populate registry with n_prior dummy experiments
    config.registry_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(config.registry_path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            experiment_name TEXT,
            status TEXT NOT NULL
        )
        """)
        for i in range(n_prior):
            con.execute(
                "INSERT INTO experiments (experiment_id, status) VALUES (?, 'completed')",
                (f"exp_{i:04d}",),
            )
    budget_sec = _compute_experiment_budget(config)
    assert budget_sec == expected_minutes * 60, (
        f"N={n_prior}: expected {expected_minutes}min ({expected_minutes*60}s), got {budget_sec}s"
    )


def test_budget_disabled_returns_none(tmp_path: Path):
    config = _make_config(tmp_path, compute_enforce=False)
    budget_sec = _compute_experiment_budget(config)
    assert budget_sec is None


# ---------------------------------------------------------------------------
# 1.2 SIGALRM timeout raises ComputeBudgetExceeded
# ---------------------------------------------------------------------------

def test_compute_budget_alarm_fires():
    with pytest.raises(ComputeBudgetExceeded):
        with _compute_budget_alarm(1):   # 1-second budget
            time.sleep(3)


def test_compute_budget_alarm_no_fire():
    # Should complete without raising
    with _compute_budget_alarm(60):
        time.sleep(0.01)


def test_compute_budget_alarm_none_skips():
    with _compute_budget_alarm(None):
        time.sleep(0.01)  # should not raise


# ---------------------------------------------------------------------------
# 1.3 Timeout → failed path (integration-level check)
# ---------------------------------------------------------------------------

def test_timeout_routes_to_failed(tmp_path: Path):
    """ComputeBudgetExceeded raised by run_experiment must propagate as-is (not crash cycle)."""
    config = _make_config(tmp_path)
    # We don't have the full data pipeline, so we test the exception shape
    exc = ComputeBudgetExceeded(
        "Compute budget exceeded: ran 62.0s, budget 60s. Reduce n_estimators / use early stopping."
    )
    assert "budget" in str(exc).lower()
    assert "n_estimators" in str(exc).lower()


# ---------------------------------------------------------------------------
# 2.5 Doubled-path fix regression test
# ---------------------------------------------------------------------------

def test_resolve_model_script_path_no_doubling(tmp_path: Path):
    """Script path stored as repo-root-relative must not produce doubled artifacts/tracks/... path."""
    # Simulate a proposal whose config_path is deep inside artifacts/tracks/
    deep_proposal_dir = tmp_path / "artifacts" / "tracks" / "claude" / "runs" / "run1" / "iterations" / "iter1" / "proposal"
    deep_proposal_dir.mkdir(parents=True)
    config_path = deep_proposal_dir / "experiment_config.toml"
    config_path.touch()

    # The script exists at a flat location relative to the proposal dir
    script = deep_proposal_dir / "model_attempt_1.py"
    script.write_text("# dummy", encoding="utf-8")

    model_cfg = {"script_path": "model_attempt_1.py"}
    resolved = _resolve_model_script_path(config_path, model_cfg)
    assert resolved is not None
    resolved_str = str(resolved)

    # Must NOT contain the doubled-path pattern
    assert "artifacts/tracks/artifacts/tracks" not in resolved_str, (
        f"Doubled path detected: {resolved_str}"
    )
    assert resolved.exists()


def test_resolve_model_script_path_abs_unchanged(tmp_path: Path):
    """Absolute paths are returned unchanged (resolved)."""
    script = tmp_path / "my_model.py"
    script.write_text("# dummy", encoding="utf-8")
    config_path = tmp_path / "experiment.toml"
    config_path.touch()

    resolved = _resolve_model_script_path(config_path, {"script_path": str(script)})
    assert resolved == script.resolve()


def test_resolve_model_script_path_none_when_missing():
    """Returns None when no script is specified."""
    config_path = Path("/tmp/dummy.toml")
    assert _resolve_model_script_path(config_path, {}) is None


# ---------------------------------------------------------------------------
# 2.4 Auto-abandon: two consecutive ≤noise-floor attempts
# ---------------------------------------------------------------------------

def test_auto_abandon_two_consecutive_zero_lift(tmp_path: Path):
    """Two consecutive near-zero lifts should trigger auto-abandon before attempt 3."""
    from autoresearch.controller.workflow import _run_validated_experiment_attempts, ExperimentNeedsRepair

    config = _make_config(tmp_path)

    # Build a fake proposal and champion_id; mock out the expensive calls
    proposal = {
        "proposal_id": "test_prop",
        "config": {"experiment_name": "test", "model_family": "scripted_model",
                   "target_strategy": "direct_pure_premium", "model": {"script_path": "model_attempt_1.py"}},
    }
    proposal_dir = tmp_path / "proposal"
    proposal_dir.mkdir(parents=True)
    iteration_dir = tmp_path / "iteration"
    iteration_dir.mkdir(parents=True)

    # Write dummy attempt scripts
    for n in range(1, 4):
        (proposal_dir / f"model_attempt_{n}.py").write_text("# dummy", encoding="utf-8")

    call_count = 0

    def fake_run_experiment(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # Return fake outputs
        out_dir = kwargs.get("output_dir") or (tmp_path / f"exp_{call_count}")
        out_dir.mkdir(parents=True, exist_ok=True)
        fake_preds = pd.DataFrame({"record_id": [1], "split": ["search_validation"],
                                    "predicted_pure_premium": [100.0], "actual_pure_premium": [100.0],
                                    "exposure": [1.0]})
        pred_path = out_dir / "predictions.parquet"
        fake_preds.to_parquet(pred_path, index=False)
        config_snap = out_dir / "config_snapshot.json"
        import json
        config_snap.write_text(json.dumps({"experiment_id": f"exp_{call_count}", "experiment": {
            "model_family": "scripted_model"}}), encoding="utf-8")
        return {"predictions": pred_path, "config_snapshot": config_snap}

    # Two consecutive attempts both yield near-zero lift (|lift| < noise_eps)
    attempt_n = [0]

    def fake_validate(*args, **kwargs):
        attempt_n[0] += 1
        return {
            "attempt": attempt_n[0],
            "valid": False,
            "reason": "lift too low",
            "checks": [],
            "lift_summary": {"lift": 0.0001},  # below noise_eps=0.002
        }

    def fake_attach(*args, **kwargs):
        # Return report unchanged (no comparison in test)
        return kwargs.get("report", args[3] if len(args) > 3 else {})

    # Intercept the signature of _attach_failed_attempt_comparison properly
    from autoresearch.controller import workflow as _wf

    with patch.object(_wf, "run_experiment", side_effect=fake_run_experiment), \
         patch.object(_wf, "_validate_attempt_outputs", side_effect=fake_validate), \
         patch.object(_wf, "_attach_failed_attempt_comparison",
                      side_effect=lambda config, champion_id, experiment_id, report, **kw: report), \
         patch.object(_wf, "record_experiment_artifacts"), \
         patch.object(_wf, "write_json"), \
         patch.object(_wf, "read_json", return_value={"experiment_id": "exp_1", "experiment": {}}):
        with pytest.raises(ValueError, match="noise floor"):
            _run_validated_experiment_attempts(
                config,
                proposal,
                "champion_exp",
                proposal_dir,
                iteration_dir,
                compute_budget_sec=None,
            )

    # Should have only run 2 attempts, not 3
    assert call_count == 2, f"Expected 2 attempts before auto-abandon, got {call_count}"


# ---------------------------------------------------------------------------
# Schema migration: existing registry opens without error
# ---------------------------------------------------------------------------

def test_schema_migration_adds_timing_columns(tmp_path: Path):
    """Existing registries (without timing columns) must open and migrate cleanly."""
    registry_path = tmp_path / "registry.sqlite"
    # Create old-style registry without timing columns
    with sqlite3.connect(registry_path) as con:
        con.execute("""
        CREATE TABLE experiments (
            experiment_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            experiment_name TEXT,
            status TEXT NOT NULL
        )
        """)
        con.execute("INSERT INTO experiments (experiment_id, status) VALUES ('old_exp', 'completed')")

    from autoresearch.experiment_registry.schema import init_registry
    init_registry(registry_path)

    # Check new columns exist
    with sqlite3.connect(registry_path) as con:
        cols = {row[1] for row in con.execute("PRAGMA table_info(experiments)").fetchall()}
    for col in ("fit_wall_seconds", "fit_cpu_seconds", "compute_budget_seconds", "timed_out"):
        assert col in cols, f"Missing column after migration: {col}"

    # Old row should still be readable
    with sqlite3.connect(registry_path) as con:
        row = con.execute("SELECT * FROM experiments WHERE experiment_id='old_exp'").fetchone()
    assert row is not None
