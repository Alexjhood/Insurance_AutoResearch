from pathlib import Path

import pandas as pd
import pytest

from autoresearch.config import ProjectConfig
from autoresearch.experiment_registry.registry import list_experiments
from autoresearch.experiment_runner import run_experiment


def _make_config(tmp_path: Path) -> ProjectConfig:
    """Build a minimal ProjectConfig for runner tests."""
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    artifacts = tmp_path / "artifacts"
    processed.mkdir(parents=True)
    splits.mkdir(parents=True)
    return ProjectConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=processed,
        holdout_vault_dir=tmp_path / "holdout_vault",
        metadata_dir=tmp_path / "metadata",
        splits_dir=splits,
        artifacts_dir=artifacts,
        registry_path=artifacts / "registry.sqlite",
        random_seed=1,
        id_column="IDpol",
        agent_dataset_name="agent_dataset",
        claim_capping_enabled=True,
        claim_cap_threshold=100000,
        split_ratios={"train": 0.64, "search_validation": 0.16, "milestone_holdout": 0.2},
        ordinary_train_split="train",
        ordinary_eval_splits=("search_validation",),
        primary_metric="tweedie_deviance_p15",
        tweedie_power=1.5,
        use_cv=False,
        cv_folds=5,
        cv_n_repeats=1,
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
        llm_provider="mock",
        llm_model="mock",
        llm_temperature=0.2,
        llm_proposal_file=tmp_path / "proposals.jsonl",
        handoff_base_dir=tmp_path / "auto_research",
        handoff_context_dir=tmp_path / "auto_research" / "context",
        handoff_proposal_inbox_dir=tmp_path / "auto_research" / "proposals" / "inbox",
        handoff_proposal_processed_dir=tmp_path / "auto_research" / "proposals" / "processed",
        handoff_results_dir=tmp_path / "auto_research" / "results",
        handoff_handoffs_dir=tmp_path / "auto_research" / "handoffs",
        deduplication_policy="reject",
        deduplication_lookback=25,
        search_space={
            "model_families": ["regularized_linear"],
            "target_strategies": ["direct_pure_premium", "frequency_severity"],
            "regularized_linear": {"min_alpha": 0.01, "max_alpha": 100.0},
            "preprocessing": {"claim_cap_thresholds": [100000], "allow_disable_claim_capping": False},
        },
    )


def _write_fixtures(config: ProjectConfig) -> None:
    """Write minimal parquet + split_pack for runner tests."""
    frame = pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "claim_count_signal_q": [0, 1, 0, 1, 0, 1],
        "exposure_term_a": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "vehicle_power_band_b": [4, 5, 4, 5, 4, 5],
        "region_cluster_j": ["a", "b", "a", "b", "a", "b"],
        "claim_cost_observed_k": [0.0, 100.0, 0.0, 200.0, 0.0, 150.0],
        "claim_event_count_l": [0, 1, 0, 1, 0, 1],
    })
    # Write both legacy path and new search path
    frame.to_parquet(config.processed_dir / "agent_dataset.parquet", index=False)
    frame.to_parquet(config.processed_dir / "agent_dataset_search.parquet", index=False)
    pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "split_unit": [0.1, 0.2, 0.5, 0.6, 0.7, 0.9],
        "split": ["train", "train", "train", "train", "search_validation", "search_validation"],
    }).to_csv(config.splits_dir / "split_pack.csv", index=False)


def test_run_experiment_writes_registry_and_artifacts(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)

    exp_config = tmp_path / "experiment.toml"
    exp_config.write_text(
        """
experiment_name = "test_direct"
model_family = "regularized_linear"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
alpha = 1.0
""".strip(),
        encoding="utf-8",
    )

    outputs = run_experiment(config, exp_config)
    rows = list_experiments(config.registry_path)

    assert outputs["metrics"].exists()
    assert outputs["diagnostics"].exists()
    assert outputs["environment_manifest"].exists()
    assert rows[0]["experiment_name"] == "test_direct"
    assert rows[0]["claim_cap_threshold"] == 100000


def test_run_experiment_tweedie_glm(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)

    exp_config = tmp_path / "experiment_glm.toml"
    exp_config.write_text(
        """
experiment_name = "test_tweedie_glm"
model_family = "tweedie_glm"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
alpha = 1.0
power = 1.5
""".strip(),
        encoding="utf-8",
    )

    outputs = run_experiment(config, exp_config)
    rows = list_experiments(config.registry_path)
    assert outputs["metrics"].exists()
    assert any(r["model_family"] == "tweedie_glm" for r in rows)


def test_metrics_include_tweedie_deviance(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)

    exp_config = tmp_path / "experiment.toml"
    exp_config.write_text(
        """
experiment_name = "test_metric_panel"
model_family = "tweedie_glm"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
alpha = 0.1
power = 1.5
""".strip(),
        encoding="utf-8",
    )

    import json
    outputs = run_experiment(config, exp_config)
    metrics = json.loads(outputs["metrics"].read_text())
    panel = {m["split"]: m for m in metrics["split_metrics"]}
    assert "tweedie_deviance_p15" in panel.get("search_validation", panel.get("train", {}))
    assert "predicted_to_actual_ratio" in panel.get("search_validation", panel.get("train", {}))
