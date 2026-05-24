from pathlib import Path

import pandas as pd

from autoresearch.config import ProjectConfig
from autoresearch.experiment_registry.registry import list_experiments
from autoresearch.experiment_runner import run_experiment


def test_run_experiment_writes_registry_and_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    artifacts = tmp_path / "artifacts"
    processed.mkdir()
    splits.mkdir()

    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "claim_count_signal_q": [0, 1, 0, 1],
            "exposure_term_a": [1.0, 1.0, 1.0, 1.0],
            "vehicle_power_band_b": [4, 5, 4, 5],
            "region_cluster_j": ["a", "b", "a", "b"],
            "claim_cost_observed_k": [0.0, 100.0, 0.0, 200000.0],
            "claim_event_count_l": [0, 1, 0, 1],
        }
    )
    frame.to_parquet(processed / "agent_dataset.parquet", index=False)
    pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "split_unit": [0.1, 0.2, 0.7, 0.9],
            "split": ["train", "train", "search_validation", "milestone_holdout"],
        }
    ).to_csv(splits / "split_pack.csv", index=False)

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
    config = ProjectConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=processed,
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
        repeated_resamples=5,
        bootstrap_iterations=20,
        resample_fraction=1.0,
        resampling_seed=1,
        minimum_mean_lift=0.0,
        minimum_win_rate=0.55,
        bootstrap_lower_bound=0.0,
        confidence_level=0.9,
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
            "min_alpha": 0.01,
            "max_alpha": 100.0,
            "claim_cap_thresholds": [100000],
            "allow_disable_claim_capping": False,
        },
    )

    outputs = run_experiment(config, exp_config)
    rows = list_experiments(config.registry_path)

    assert outputs["metrics"].exists()
    assert rows[0]["experiment_name"] == "test_direct"
    assert rows[0]["claim_cap_threshold"] == 100000
