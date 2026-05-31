"""Tests for milestone holdout evaluation with scripted model families (Bug F fix)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from autoresearch.config import ProjectConfig
from autoresearch.milestone import evaluate_on_holdout
from autoresearch.utils.io import write_json


_N_SEARCH = 20
_N_HOLDOUT = 10


def _make_config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        holdout_vault_dir=tmp_path / "vault",
        metadata_dir=tmp_path / "metadata",
        splits_dir=tmp_path / "splits",
        artifacts_dir=tmp_path / "artifacts",
        registry_path=tmp_path / "artifacts" / "registry.sqlite",
        research_log_path=tmp_path / "RESEARCH_LOG.md",
        track_id="test",
        random_seed=1,
        id_column="IDpol",
        agent_dataset_name="agent_dataset",
        claim_capping_enabled=False,
        claim_cap_threshold=100_000.0,
        split_ratios={"train": 0.64, "search_validation": 0.16, "milestone_holdout": 0.2},
        ordinary_train_split="train",
        ordinary_eval_splits=("search_validation",),
        target_mode="burning_cost",
        primary_metric="gini_weighted",
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
        escalation_partitions=1,
        repeated_resamples=5,
        bootstrap_iterations=20,
        resample_fraction=1.0,
        resampling_seed=1,
        minimum_mean_lift=0.0,
        min_relative_lift=0.005,
        min_absolute_lift=0.0,
        minimum_win_rate=0.60,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.05,
        require_diagnostics=True,
        bonferroni_lookback=10,
        handoff_base_dir=tmp_path / "auto_research",
        handoff_context_dir=tmp_path / "auto_research" / "context",
        handoff_proposal_inbox_dir=tmp_path / "auto_research" / "proposals" / "inbox",
        handoff_proposal_processed_dir=tmp_path / "auto_research" / "proposals" / "processed",
        handoff_results_dir=tmp_path / "auto_research" / "results",
        handoff_handoffs_dir=tmp_path / "auto_research" / "handoffs",
        proposal_inbox_file=tmp_path / "auto_research" / "proposals" / "inbox" / "manual_proposals.jsonl",
        deduplication_policy="reject",
        deduplication_lookback=25,
        search_space={},
    )


def _make_row_frame(n: int, id_offset: int = 0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "record_id": np.arange(id_offset, id_offset + n),
        "exposure_term_a": np.ones(n),
        "claim_cost_capped_active": rng.exponential(100, n).clip(1.0),
        "claim_cost_observed_k": rng.exponential(100, n).clip(1.0),
        "claim_count_signal_q": rng.integers(0, 2, n),
        "claim_event_count_l": rng.integers(0, 2, n),
    })


def _write_fixtures(config: ProjectConfig, script_path: Path) -> tuple[Path, Path]:
    """Write search parquet, holdout parquet, split pack, config snapshot, and SV predictions."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.holdout_vault_dir.mkdir(parents=True, exist_ok=True)
    config.splits_dir.mkdir(parents=True, exist_ok=True)
    (config.artifacts_dir / "milestone_reports").mkdir(parents=True, exist_ok=True)

    search_frame = _make_row_frame(_N_SEARCH, id_offset=0)
    holdout_frame = _make_row_frame(_N_HOLDOUT, id_offset=_N_SEARCH, seed=42)

    search_frame.to_parquet(config.processed_dir / "agent_dataset_search.parquet", index=False)
    holdout_frame.to_parquet(config.holdout_vault_dir / "agent_dataset_holdout.parquet", index=False)

    all_ids = list(range(_N_SEARCH + _N_HOLDOUT))
    split_pack = pd.DataFrame({
        "record_id": all_ids,
        "split_unit": np.linspace(0.0, 1.0, _N_SEARCH + _N_HOLDOUT),
        "split": (
            ["train"] * 14
            + ["search_validation"] * (_N_SEARCH - 14)
            + ["milestone_holdout"] * _N_HOLDOUT
        ),
    })
    split_pack.to_csv(config.splits_dir / "split_pack.csv", index=False)

    # Config snapshot for the champion experiment
    snapshot_path = config.artifacts_dir / "config_snapshot.json"
    write_json(snapshot_path, {
        "experiment": {
            "model_family": "scripted_demo",
            "target_strategy": "direct_pure_premium",
            "model": {},
            "preprocessing": {"claim_capping_enabled": False, "claim_cap_threshold": 100000},
        },
        "model_script_path": str(script_path),
    })

    # SV predictions (needed for the gap calculation)
    sv_ids = list(split_pack.loc[split_pack["split"] == "search_validation", "record_id"])
    sv_rows = search_frame[search_frame["record_id"].isin(sv_ids)].copy()
    sv_preds = pd.DataFrame({
        "record_id": sv_rows["record_id"].to_numpy(),
        "split": "search_validation",
        "exposure": sv_rows["exposure_term_a"].to_numpy(),
        "actual_claim_cost": sv_rows["claim_cost_capped_active"].to_numpy(),
        "actual_claim_cost_uncapped": sv_rows["claim_cost_observed_k"].to_numpy(),
        "predicted_claim_cost": np.full(len(sv_rows), 80.0),
    })
    sv_preds_path = config.artifacts_dir / "predictions.parquet"
    sv_preds.to_parquet(sv_preds_path, index=False)

    return snapshot_path, sv_preds_path


def _write_stub_script(path: Path) -> None:
    path.write_text(
        textwrap.dedent("""\
            import numpy as np

            def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **kw):
                predicted = score["exposure_term_a"].to_numpy() * 80.0
                return predicted, {"model": "stub_constant"}
        """),
        encoding="utf-8",
    )


def _artifact_path_side_effect(config, champion_id, artifact_type, *, snapshot_p, preds_p):
    if artifact_type == "config_snapshot":
        return snapshot_p
    if artifact_type == "predictions":
        return preds_p
    raise FileNotFoundError(f"No fixture for artifact_type={artifact_type!r}")


# ── Positive test ──────────────────────────────────────────────────────────────


def test_scripted_champion_produces_completed_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "milestone")
    config = _make_config(tmp_path)
    script_path = tmp_path / "model_stub.py"
    _write_stub_script(script_path)
    snapshot_p, preds_p = _write_fixtures(config, script_path)

    with patch(
        "autoresearch.milestone._artifact_path",
        side_effect=lambda cfg, cid, atype: _artifact_path_side_effect(
            cfg, cid, atype, snapshot_p=snapshot_p, preds_p=preds_p
        ),
    ):
        result = evaluate_on_holdout(config, "champ_exp_001", "promo_001")

    assert result is not None
    assert result["status"] == "completed", result.get("reason")
    assert "holdout_metrics" in result
    assert result["holdout_metrics"]["gini_weighted"] is not None


# ── Negative test: missing script ─────────────────────────────────────────────


def test_missing_script_raises_and_report_status_not_completed(tmp_path: Path, monkeypatch) -> None:
    # Token must be set so evaluation gets past the holdout gate and reaches the
    # missing-script error path under test (rather than skipping as vault-absent).
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "milestone")
    config = _make_config(tmp_path)
    # Point to a script that does not exist
    missing_script = tmp_path / "nonexistent_model.py"
    snapshot_p, preds_p = _write_fixtures(config, missing_script)

    with patch(
        "autoresearch.milestone._artifact_path",
        side_effect=lambda cfg, cid, atype: _artifact_path_side_effect(
            cfg, cid, atype, snapshot_p=snapshot_p, preds_p=preds_p
        ),
    ):
        with pytest.raises(FileNotFoundError):
            evaluate_on_holdout(config, "champ_exp_002", "promo_002")

    # Report JSON should have been written with a non-completed status
    json_path = config.artifacts_dir / "milestone_reports" / "promo_002.json"
    assert json_path.exists(), "Report JSON should be written even on failure"
    import json
    report = json.loads(json_path.read_text())
    assert report["status"] != "completed"
