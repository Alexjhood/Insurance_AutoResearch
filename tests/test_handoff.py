import json
from dataclasses import replace
from pathlib import Path

from autoresearch.config import load_config
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import export_context_bundle, inbox_status, ingest_proposals, write_proposal_template
from autoresearch.controller.workflow import ExperimentNeedsRepair, run_next_queued_proposal
from autoresearch.experiment_registry.registry import init_registry, list_proposals, record_experiment
from autoresearch.experiment_runner import run_experiment
from tests.test_phase4_controller import _config
from tests.test_runner import _write_fixtures


def _record_direct(config, experiment_id: str = "direct") -> None:
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


def _valid_proposal(parent_id: str = "direct") -> dict:
    return {
        "proposal_id": "handoff_valid_1",
        "parent_experiment_id": parent_id,
        "parent_branch_id": "main",
        "branch_action": "new_branch",
        "experiment_name": "handoff_alpha_2",
        "rationale": "Try modestly stronger regularisation.",
        "change_summary": "Set alpha to 2.0.",
        "expected_benefit": "Reduce variance.",
        "key_risk": "Underfitting.",
        "experiment_config": {
            "experiment_name": "handoff_alpha_2",
            "model_family": "regularized_linear",
            "target_strategy": "direct_pure_premium",
            "parent_experiment_id": parent_id,
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": {"alpha": 2.0, "feature_exclusions": []},
        },
    }


def test_default_config_uses_file_handoff() -> None:
    config = load_config()

    assert config.llm_provider == "file_handoff"
    assert "auto_research" in str(config.handoff_base_dir)


def test_tracked_config_scopes_artifacts_to_run() -> None:
    config = load_config(track_id="codex", run_id="CodexTimeX")

    assert config.track_id == "codex"
    assert config.run_id == "CodexTimeX"
    assert config.artifacts_dir.name == "CodexTimeX"
    assert config.artifacts_dir.parent.name == "runs"
    assert config.registry_path == config.artifacts_dir / "registry.sqlite"
    assert config.handoff_context_dir == config.artifacts_dir / "context"


def test_export_context_and_template(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        llm_provider="file_handoff",
        llm_model="external",
        llm_proposal_file=config.handoff_proposal_inbox_dir / "manual.jsonl",
    )
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )

    context_outputs = export_context_bundle(config)
    template_outputs = write_proposal_template(config)

    assert context_outputs["latest_context_json"].exists()
    assert context_outputs["latest_handoff_markdown"].exists()
    assert template_outputs["proposal_template"].exists()
    assert template_outputs["proposal_schema"].exists()


def test_ingest_proposals_moves_valid_and_invalid_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(config, llm_provider="file_handoff")
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "valid.json").write_text(
        json.dumps(_valid_proposal()),
        encoding="utf-8",
    )
    (config.handoff_proposal_inbox_dir / "invalid.json").write_text("{bad-json}", encoding="utf-8")

    summary = ingest_proposals(config)
    proposals = list_proposals(config.registry_path)
    status = inbox_status(config)

    assert summary["valid_count"] == 1
    assert summary["invalid_count"] == 1
    assert any(item["status"] == "validated" for item in proposals)
    assert status["processed_valid_count"] == 1
    assert status["processed_invalid_count"] == 1


def test_validation_failure_writes_repair_request(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.search_space["requires_model_script"] = True
    config.search_space["allow_open_model_families"] = True
    config.processed_dir.mkdir(parents=True)
    config.splits_dir.mkdir(parents=True)
    _write_fixtures(config)
    champion_config = tmp_path / "global_mean.toml"
    champion_config.write_text(
        """
experiment_name = "global_mean_test"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )
    champion_outputs = run_experiment(config, champion_config)
    champion_id = json.loads(champion_outputs["config_snapshot"].read_text(encoding="utf-8"))["experiment_id"]
    initialise_official_champion(config, champion_id)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "bad_model.py").write_text(
        """
import numpy as np


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hp):
    return np.zeros(len(score)), {"intent": "bad output"}
""".strip(),
        encoding="utf-8",
    )
    proposal = _valid_proposal(champion_id)
    proposal["proposal_id"] = "needs_repair_1"
    proposal["parent_branch_id"] = "main"
    proposal["experiment_name"] = "needs_repair_exp"
    proposal["experiment_config"]["experiment_name"] = "needs_repair_exp"
    proposal["experiment_config"]["model_family"] = "scripted_bad"
    proposal["experiment_config"]["model"] = {"script_path": "bad_model.py"}
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    ingest_proposals(config)

    try:
        run_next_queued_proposal(config)
    except ExperimentNeedsRepair:
        pass
    else:
        raise AssertionError("Expected validation to request repair")

    record = list_proposals(config.registry_path)[0]
    proposal_dir = Path(record["proposal_path"]).parent
    assert record["status"] == "needs_repair"
    assert (proposal_dir / "repair_request_2.json").exists()
