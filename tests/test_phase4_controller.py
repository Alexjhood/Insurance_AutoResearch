from pathlib import Path
from dataclasses import replace

from autoresearch.config import ProjectConfig
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.proposal_schema import allowed_search_space, validate_proposal
from autoresearch.controller.workflow import generate_and_enqueue_proposal
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    init_registry,
    list_champion_history,
    list_proposals,
    record_experiment,
    set_official_champion,
)


def _config(tmp_path: Path) -> ProjectConfig:
    return ProjectConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        holdout_vault_dir=tmp_path / "holdout_vault",
        metadata_dir=tmp_path / "metadata",
        splits_dir=tmp_path / "splits",
        artifacts_dir=tmp_path / "artifacts",
        registry_path=tmp_path / "artifacts" / "registry.sqlite",
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
        min_relative_lift=0.0,
        min_absolute_lift=0.0,
        minimum_win_rate=0.55,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.5,
        require_diagnostics=False,
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
            "model_families": ["tweedie_glm", "frequency_severity_glm", "tweedie_gbm", "regularized_linear"],
            "target_strategies": ["direct_pure_premium", "frequency_severity"],
            "allow_legacy_baselines": False,
            "regularized_linear": {"min_alpha": 0.01, "max_alpha": 100.0},
            "tweedie_glm": {"min_alpha": 0.001, "max_alpha": 10.0, "power_choices": [1.1, 1.3, 1.5, 1.7, 1.9]},
            "frequency_severity_glm": {"min_freq_alpha": 0.001, "max_freq_alpha": 10.0, "min_sev_alpha": 0.001, "max_sev_alpha": 10.0},
            "tweedie_gbm": {"min_max_iter": 100, "max_max_iter": 2000, "max_depth_choices": [3, 5, 7, 9], "min_learning_rate": 0.01, "max_learning_rate": 0.2, "min_samples_leaf_choices": [50, 200, 500, 1000]},
            "preprocessing": {"claim_cap_thresholds": [100000], "allow_disable_claim_capping": False},
        },
    )


def _record_direct(config: ProjectConfig, experiment_id: str = "direct") -> None:
    init_registry(config.registry_path)
    path = config.artifacts_dir / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    metrics = path / "metrics.json"
    metrics.write_text('{"aggregate": {"mean_score": 10.0, "std_score": 0.0}}', encoding="utf-8")
    record_experiment(
        config.registry_path,
        experiment_id=experiment_id,
        experiment_name="direct_pure_premium_baseline",
        model_family="regularized_linear",
        target_strategy="direct_pure_premium",
        preprocessing_summary={"claim_capping_enabled": True, "claim_cap_threshold": 100000},
        claim_cap_threshold=100000,
        status="completed",
        parent_experiment_id=None,
        config_snapshot_path=path / "config.json",
        metrics_path=metrics,
        artifacts={"metrics": metrics},
    )


def test_validate_proposal_rejects_milestone_and_bad_alpha(tmp_path: Path) -> None:
    config = _config(tmp_path)
    space = allowed_search_space(config, {"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]})
    proposal = {
        "proposal_id": "bad",
        "parent_experiment_id": "direct",
        "experiment_name": "bad_exp",
        "rationale": "bad",
        "change_summary": "uses milestone_holdout",
        "expected_benefit": "none",
        "key_risk": "invalid",
        "experiment_config": {
            "experiment_name": "bad_exp",
            "model_family": "regularized_linear",
            "target_strategy": "direct_pure_premium",
            "parent_experiment_id": "direct",
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": {"alpha": 1000.0},
        },
    }

    errors = validate_proposal(proposal, space)

    assert any("alpha" in error for error in errors)
    assert any("milestone_holdout" in error for error in errors)


def test_initialise_champion_records_history_and_branch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)

    state = initialise_official_champion(config)
    history = list_champion_history(config.registry_path)

    assert state["champion_id"] == "direct"
    assert get_official_champion(config.registry_path)["branch_id"] == "main"
    assert history[0]["action"] == "initialised"


def test_mock_proposer_generates_validated_queue_item(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )

    result = generate_and_enqueue_proposal(config)
    proposals = list_proposals(config.registry_path)

    assert result["status"] == "validated"
    assert proposals[0]["status"] == "validated"
    assert proposals[0]["branch_id"] == proposals[0]["proposal_id"]


def test_invalid_file_proposer_output_is_recorded_as_failed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    proposal_file = tmp_path / "bad.jsonl"
    proposal_file.write_text("{not-json}", encoding="utf-8")
    config = replace(config, llm_provider="file", llm_proposal_file=proposal_file)
    _record_direct(config)
    initialise_official_champion(config)

    result = generate_and_enqueue_proposal(config)
    proposals = list_proposals(config.registry_path)

    assert result["status"] == "failed"
    assert proposals[0]["status"] == "failed"
    assert proposals[0]["validation_errors"]


def test_promoted_champion_replacement_is_persisted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config, "direct")
    _record_direct(config, "challenger")
    initialise_official_champion(config, "direct")

    set_official_champion(
        config.registry_path,
        champion_id="challenger",
        branch_id="branch_1",
        reason="passed promotion gate",
        action="promoted",
        comparison_id="comparison_1",
        proposal_id="proposal_1",
    )
    state = get_official_champion(config.registry_path)
    history = list_champion_history(config.registry_path)

    assert state["champion_id"] == "challenger"
    assert history[0]["action"] == "promoted"
    assert history[0]["previous_champion_id"] == "direct"
