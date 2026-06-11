import json
import re
from pathlib import Path

import pytest

from autoresearch.config import ProjectConfig, _resolve_run_id, ensure_project_dirs, load_config
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import (
    _render_recipe_reuse,
    _render_tree_node_lines,
    export_context_bundle,
    inbox_status,
    ingest_proposals,
    render_handoff_markdown,
    write_proposal_template,
)
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.proposal_schema import allowed_search_space, validate_proposal
from autoresearch.controller.workflow import ExperimentNeedsRepair, run_next_queued_proposal
from autoresearch.experiment_registry.registry import (
    init_registry,
    clear_research_line_champion,
    list_proposals,
    list_research_lines,
    list_research_nodes,
    record_proposal,
    record_experiment,
    set_official_champion,
    update_proposal_status,
    upsert_research_line,
    upsert_research_node,
)
from autoresearch.experiment_runner import run_experiment
from tests.test_runner import _make_config as _config, _write_fixtures


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
        model_family="global_mean",
        target_strategy="direct_pure_premium",
        target_mode=config.target_mode,
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
        "research_line_action": "create_line",
        "research_line_id": "line_handoff_valid",
        "research_line_label": "Handoff validation line",
        "research_line_hypothesis": "Validate the proposal queue on a simple baseline line.",
        "line_membership_rationale": "This proposal starts the handoff validation line.",
        "tree_action": "new_root",
        "research_parent_node_id": None,
        "selected_tree_action_id": "start_first_root",
        "parent_rationale": "This starts a first active-run hypothesis.",
        "exploration_axis": "model_family",
        "approach_family": "baseline check",
        "target_framing": "direct_pure_premium",
        "feature_representation": "raw",
        "expected_learning": "Verify the queue and evaluation loop with a simple baseline.",
        "branch_action": "new_branch",
        "experiment_name": "handoff_global_mean_2",
        "rationale": "Run a second global-mean baseline.",
        "change_summary": "Identical global-mean config for comparison.",
        "expected_benefit": "Verify reproducibility.",
        "key_risk": "None.",
        "experiment_config": {
            "experiment_name": "handoff_global_mean_2",
            "model_family": "global_mean",
            "target_strategy": "direct_pure_premium",
            "parent_experiment_id": parent_id,
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": {"feature_exclusions": []},
        },
    }


def test_default_config_uses_file_handoff() -> None:
    config = load_config()

    assert "auto_research" in str(config.handoff_base_dir)
    assert config.proposal_inbox_file.name == "manual_proposals.jsonl"


def test_tracked_config_scopes_artifacts_to_run() -> None:
    config = load_config(track_id="codex", run_id="20260604T081500Z")

    assert config.track_id == "codex"
    assert config.run_id == "20260604T081500Z"
    assert config.artifacts_dir.name == "20260604T081500Z"
    assert config.artifacts_dir.parent.name == "runs"
    assert config.registry_path == config.artifacts_dir / "registry.sqlite"
    assert config.handoff_context_dir == config.artifacts_dir / "context"


def test_agent_track_rejects_non_timestamp_run_id() -> None:
    try:
        load_config(track_id="codex", run_id="CodexTimeX")
    except ValueError as exc:
        assert "must be a UTC timestamp" in str(exc)
    else:
        raise AssertionError("Expected non-timestamp agent run_id to be rejected")


def test_new_run_id_is_timestamp_even_when_latest_exists(tmp_path: Path) -> None:
    track_base = tmp_path / "artifacts" / "tracks" / "codex"
    track_base.mkdir(parents=True)
    (track_base / "latest_run.json").write_text('{"run_id": "CC20260526_01"}', encoding="utf-8")

    run_id = _resolve_run_id(track_base, None, new_run=True)

    assert run_id != "CC20260526_01"
    assert re.fullmatch(r"\d{8}T\d{6}Z", run_id)


def test_explicit_run_id_does_not_move_latest_pointer(tmp_path: Path) -> None:
    track_base = tmp_path / "artifacts" / "tracks" / "codex"
    track_base.mkdir(parents=True)
    latest = track_base / "latest_run.json"
    latest.write_text(json.dumps({"track_id": "codex", "run_id": "newer"}), encoding="utf-8")

    config = ProjectConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        holdout_vault_dir=tmp_path / "holdout_vault",
        metadata_dir=tmp_path / "metadata",
        splits_dir=tmp_path / "splits",
        artifacts_dir=track_base / "runs" / "older",
        registry_path=track_base / "runs" / "older" / "registry.sqlite",
        research_log_path=track_base / "runs" / "older" / "RESEARCH_LOG.md",
        track_id="codex",
        random_seed=1,
        id_column="record_id",
        agent_dataset_name="agent_dataset",
        claim_capping_enabled=True,
        claim_cap_threshold=100000,
        split_ratios={"train": 0.64, "search_validation": 0.16, "milestone_holdout": 0.2},
        ordinary_train_split="train",
        ordinary_eval_splits=("search_validation",),
        target_mode="burning_cost",
        primary_metric="gini_weighted",
        tweedie_power=1.5,
        use_cv=False,
        cv_folds=4,
        cv_n_repeats=1,
        cv_seed=1,
        gate_mode="cv_bootstrap",
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=20,
        escalation_win_rate_low=0.4,
        escalation_win_rate_high=0.6,
        escalation_partitions=2,
        repeated_resamples=30,
        bootstrap_iterations=100,
        resample_fraction=1.0,
        resampling_seed=1,
        minimum_mean_lift=0.0,
        min_relative_lift=0.0,
        min_absolute_lift=0.0,
        minimum_win_rate=0.5,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.05,
        require_diagnostics=False,
        bonferroni_lookback=10,
        handoff_base_dir=track_base / "runs" / "older",
        handoff_context_dir=track_base / "runs" / "older" / "context",
        handoff_proposal_inbox_dir=track_base / "runs" / "older" / "proposal_inbox",
        handoff_proposal_processed_dir=track_base / "runs" / "older" / "proposal_processed",
        handoff_results_dir=track_base / "runs" / "older" / "results",
        handoff_handoffs_dir=track_base / "runs" / "older" / "handoffs",
        proposal_inbox_file=track_base / "runs" / "older" / "proposal_inbox" / "manual_proposals.jsonl",
        deduplication_policy="hash",
        deduplication_lookback=10,
        search_space={},
        run_id="older",
        track_base_dir=track_base,
        update_latest_run=False,
    )

    ensure_project_dirs(config)

    assert json.loads(latest.read_text(encoding="utf-8"))["run_id"] == "newer"


def test_export_context_and_template(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        json.dumps(
            {
                "columns": [
                    {"name": "exposure_term_a", "role": "exposure_offset"},
                    {"name": "driver_age_band_d", "role": "numeric_feature"},
                ]
            }
        ),
        encoding="utf-8",
    )

    context_outputs = export_context_bundle(config)
    template_outputs = write_proposal_template(config)
    context = json.loads(context_outputs["latest_context_json"].read_text(encoding="utf-8"))
    handoff = context_outputs["latest_handoff_markdown"].read_text(encoding="utf-8")

    assert context_outputs["latest_context_json"].exists()
    assert context_outputs["latest_handoff_markdown"].exists()
    assert template_outputs["proposal_template"].exists()
    assert template_outputs["proposal_schema"].exists()
    refreshed_template = json.loads(context_outputs["proposal_template"].read_text(encoding="utf-8"))
    # The template is the minimal scientific contract: it carries the scientific
    # fields + the experiment_config machine block, and omits controller-derived
    # parentage/tree-walk fields entirely.
    assert refreshed_template["experiment_name"]
    assert refreshed_template["exploration_axis"]
    assert "parent_experiment_id" not in refreshed_template
    assert "tree_action" not in refreshed_template
    assert "research_line_id" not in refreshed_template
    assert set(refreshed_template["experiment_config"]) >= {"model_family", "target_strategy", "model"}
    assert "parent_experiment_id" not in refreshed_template["experiment_config"]
    assert context["allowed_search_space"]["feature_columns"] == ["driver_age_band_d"]
    assert "exposure_term_a` is not a predictive feature" in handoff

    # The schema document marks parentage as controller-derived rather than required.
    schema = json.loads(template_outputs["proposal_schema"].read_text(encoding="utf-8"))
    assert "experiment_name" in schema["required"]
    assert "parent_experiment_id" not in schema["required"]
    assert "parent_experiment_id" in schema["controller_derived"]


def test_delta_handoff_drops_static_blocks_keeps_dynamic_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    context = build_llm_context(config)

    full = render_handoff_markdown(config, context)
    delta = render_handoff_markdown(config, context, delta=True)

    # Static blocks duplicated from the contract are present in full, gone in delta.
    assert "## Proposal quick-start" in full
    assert "## Key constraints" in full
    assert "Escape hatch (novel models only)" in full
    assert "## Proposal quick-start" not in delta
    assert "## Key constraints" not in delta
    assert "Escape hatch (novel models only)" not in delta

    # Dynamic state the agent needs every cycle is retained in both.
    for body in (full, delta):
        assert "## Current state" in body
        assert "**Champion**" in body
        assert "**Next command**" in body

    assert "(delta)" in delta
    # The delta view is materially shorter.
    assert len(delta) < len(full)


def _minimal_proposal() -> dict:
    """A proposal carrying only the scientific fields the agent must supply.

    Everything else (ids, parentage, tree-walk, research line, fixed
    preprocessing) is left for the controller to hydrate.
    """

    return {
        "experiment_name": "minimal_global_mean",
        "rationale": "Run a minimal baseline to exercise hydration.",
        "change_summary": "Identical global-mean config for comparison.",
        "expected_benefit": "Verify reproducibility.",
        "key_risk": "None.",
        "exploration_axis": "model_family",
        "approach_family": "baseline check",
        "target_framing": "direct_pure_premium",
        "feature_representation": "raw",
        "expected_learning": "Verify the queue with the minimal proposal contract.",
        "experiment_config": {
            "model_family": "global_mean",
            "target_strategy": "direct_pure_premium",
            "model": {"feature_exclusions": []},
        },
    }


def _ingest_single(config, proposal: dict) -> dict:
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(
        json.dumps(proposal), encoding="utf-8"
    )
    return ingest_proposals(config)


def test_minimal_scientific_proposal_hydrates_and_validates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)

    summary = _ingest_single(config, _minimal_proposal())

    assert summary["valid_count"] == 1, summary
    proposals = list_proposals(config.registry_path)
    validated = [p for p in proposals if p["status"] == "validated"]
    assert len(validated) == 1
    stored = validated[0]
    # proposal_id is derived from the experiment_name, not labelled invalid_*.
    assert not stored["proposal_id"].startswith("invalid_proposal")
    assert stored["proposal_id"].startswith("minimal_global_mean")
    # Parentage derived from the official champion.
    assert stored["parent_experiment_id"] == "direct"
    config_snapshot = stored.get("config") or {}
    # experiment_config mirrors the single experiment_name and gets fixed prep.
    assert config_snapshot["experiment_name"] == "minimal_global_mean"
    assert config_snapshot["parent_experiment_id"] == "direct"
    assert config_snapshot["preprocessing"]["claim_cap_threshold"] == 100000
    # A research line was opened automatically (none existed).
    lines = list_research_lines(config.registry_path)
    assert len(lines) == 1
    # The research node carries the derived tree-walk metadata.
    nodes = list_research_nodes(config.registry_path)
    assert nodes and nodes[0]["tree_metadata"]["tree_action"] == "new_root"


def test_create_line_without_id_mints_new_line_when_one_exists(tmp_path: Path) -> None:
    # Regression: an agent that sets research_line_action=create_line but omits
    # research_line_id must get a brand-new line, never be bound to the existing
    # one (which would fail "create_line requires a new research_line_id").
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    upsert_research_line(
        config.registry_path,
        line_id="line_existing",
        label="Existing line",
        status="active",
        root_node_id="line_existing",
        hypothesis="A prior line.",
    )

    proposal = _minimal_proposal()
    proposal["research_line_action"] = "create_line"
    proposal["research_line_label"] = "Brand new line"
    proposal["research_line_hypothesis"] = "A genuinely new direction."
    # Deliberately no research_line_id.

    summary = _ingest_single(config, proposal)

    assert summary["valid_count"] == 1, summary
    line_ids = {line["line_id"] for line in list_research_lines(config.registry_path)}
    assert "line_existing" in line_ids
    assert len(line_ids) == 2  # a new line was minted, not the existing one reused


def test_hydration_does_not_override_agent_supplied_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)

    proposal = _minimal_proposal()
    proposal["research_line_id"] = "agent_chosen_line"
    proposal["research_line_label"] = "Agent chosen label"
    proposal["line_membership_rationale"] = "Deliberate line membership note."

    summary = _ingest_single(config, proposal)

    assert summary["valid_count"] == 1, summary
    line = list_research_lines(config.registry_path)[0]
    assert line["line_id"] == "agent_chosen_line"
    assert line["label"] == "Agent chosen label"


def test_exposure_is_rejected_as_model_feature(tmp_path: Path) -> None:
    config = _config(tmp_path)
    search_space = allowed_search_space(
        config=config,
        agent_schema={
            "columns": [
                {"name": "exposure_term_a", "role": "exposure_offset"},
                {"name": "driver_age_band_d", "role": "numeric_feature"},
            ]
        },
    )
    proposal = _valid_proposal()
    proposal["experiment_config"]["model"] = {
        "script_path": "model.py",
        "feature_inclusions": ["exposure_term_a", "driver_age_band_d"],
    }

    errors = validate_proposal(proposal, search_space)

    assert any("non-predictive columns" in error for error in errors)


def test_ingest_proposals_moves_valid_and_invalid_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
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
    assert any(node["proposal_id"] == "handoff_valid_1" for node in list_research_nodes(config.registry_path))
    assert list_research_lines(config.registry_path)[0]["line_id"] == "line_handoff_valid"
    assert status["processed_valid_count"] == 1
    assert status["processed_invalid_count"] == 1


def test_batch_ingest_defers_second_valid_until_context_refresh(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    first = _valid_proposal()
    second = _valid_proposal()
    second["proposal_id"] = "handoff_valid_2"
    second["experiment_name"] = "handoff_global_mean_3"
    second["experiment_config"]["experiment_name"] = "handoff_global_mean_3"
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "first.json").write_text(json.dumps(first), encoding="utf-8")
    (config.handoff_proposal_inbox_dir / "second.json").write_text(json.dumps(second), encoding="utf-8")

    summary = ingest_proposals(config)
    proposals = list_proposals(config.registry_path)

    assert summary["valid_count"] == 1
    assert summary["deferred_count"] == 1
    assert "agent_warning" in summary
    assert len([item for item in proposals if item["status"] == "validated"]) == 1
    assert (config.handoff_proposal_inbox_dir / "second.json").exists()
    handoff = (config.handoff_handoffs_dir / "latest_handoff.md").read_text(encoding="utf-8")
    assert "Deferred proposal warning" in handoff


def test_research_tree_metadata_survives_status_only_update(tmp_path: Path) -> None:
    config = _config(tmp_path)
    upsert_research_node(
        config.registry_path,
        node_id="node_1",
        proposal_id="node_1",
        status="validated",
        tree_metadata={"exploration_axis": "model_family", "tree_action": "new_root"},
    )
    upsert_research_node(
        config.registry_path,
        node_id="node_1",
        proposal_id="node_1",
        status="running",
        tree_metadata={},
    )

    node = list_research_nodes(config.registry_path)[0]

    assert node["status"] == "running"
    assert node["tree_metadata"]["exploration_axis"] == "model_family"


def test_tree_node_handoff_labels_cv_and_split_lift() -> None:
    lines = _render_tree_node_lines(
        [
            {
                "node_id": "winner",
                "status": "promoted",
                "outcome_type": "promoted",
                "cv_lift": 0.01,
                "split_lift": -0.000126,
                "learning": "One-hot encoding won under paired CV.",
            }
        ]
    )

    assert "cv_lift=+0.010000 (split_lift=-0.000126)" in lines[1]
    assert ", lift=" not in lines[1]


def test_recipe_reuse_handoff_marks_current_champion_and_metric_types(tmp_path: Path) -> None:
    from autoresearch.models.recipe_library import record_recipe

    config = _config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    recipe = {
        "structure": "direct",
        "estimator": "lightgbm",
        "objective": "tweedie",
        "params": {"num_leaves": 31},
    }
    record_recipe(
        config,
        recipe,
        experiment_id="champion",
        outcome="promoted",
        score=0.37,
        comparison_score=0.38,
        comparison_lift=0.01,
    )
    set_official_champion(
        config.registry_path,
        champion_id="champion",
        branch_id="main",
        reason="test",
        action="promoted",
    )

    text = "\n".join(_render_recipe_reuse(config))

    assert "CURRENT CHAMPION" in text
    assert "CV score 0.3800" in text
    assert "CV lift +0.0100" in text
    assert "split score 0.3700" in text
    assert "leaves=31" in text


def test_create_line_at_cap_requires_and_applies_parking(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    for idx in range(5):
        upsert_research_line(
            config.registry_path,
            line_id=f"line_{idx}",
            label=f"Line {idx}",
            status="active",
            hypothesis="Existing line.",
        )

    proposal = _valid_proposal()
    proposal["proposal_id"] = "new_line_no_park"
    proposal["research_line_id"] = "line_new"
    proposal["research_line_label"] = "New line"
    proposal["experiment_name"] = "new_line_no_park"
    proposal["experiment_config"]["experiment_name"] = "new_line_no_park"
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")

    summary = ingest_proposals(config)

    assert summary["invalid_count"] == 1
    assert any("At most 5 active research lines" in err for err in summary["results"][0]["validation_errors"])

    proposal["proposal_id"] = "new_line_with_park"
    proposal["park_research_line_id"] = "line_0"
    proposal["park_research_line_rationale"] = "Line 0 has no useful next step."
    proposal["tree_policy_override_rationale"] = "Test isolates line parking after an intentionally invalid proposal."
    proposal["experiment_name"] = "new_line_with_park"
    proposal["experiment_config"]["experiment_name"] = "new_line_with_park"
    (config.handoff_proposal_inbox_dir / "proposal2.json").write_text(json.dumps(proposal), encoding="utf-8")

    summary = ingest_proposals(config)
    lines = {line["line_id"]: line for line in list_research_lines(config.registry_path)}

    assert summary["valid_count"] == 1
    assert lines["line_0"]["status"] == "parked"
    assert lines["line_new"]["status"] == "active"


def test_clear_research_line_champion_keeps_line_but_removes_incumbent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    upsert_research_line(
        config.registry_path,
        line_id="freq_sev",
        label="Frequency-severity",
        status="active",
        hypothesis="Test decompositions.",
        current_node_id="s12",
        current_experiment_id="exp_s12",
        best_node_id="s12",
        best_experiment_id="exp_s12",
    )

    clear_research_line_champion(
        config.registry_path,
        line_id="freq_sev",
        reason="s12 was an exposure artefact.",
    )
    line = next(item for item in list_research_lines(config.registry_path) if item["line_id"] == "freq_sev")

    assert line["status"] == "active"
    assert line["current_experiment_id"] is None
    assert line["best_experiment_id"] is None


def test_tree_action_requires_parent_for_non_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    proposal = _valid_proposal()
    proposal["tree_action"] = "extend_node"
    proposal["selected_tree_action_id"] = "start_first_root"
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")

    summary = ingest_proposals(config)

    assert summary["invalid_count"] == 1
    assert any("requires research_parent_node_id" in err for err in summary["results"][0]["validation_errors"])


def test_stale_parent_is_auto_rejected_before_execution(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(
        json.dumps(_valid_proposal()),
        encoding="utf-8",
    )
    ingest_proposals(config)
    set_official_champion(
        config.registry_path,
        champion_id="new_champion",
        branch_id="main",
        reason="test champion moved",
        action="promote",
    )

    result = run_next_queued_proposal(config)
    proposal = list_proposals(config.registry_path)[0]
    nodes = list_research_nodes(config.registry_path)

    assert result["decision"] == "auto_reject"
    assert proposal["status"] == "stale_parent"
    assert any(node["node_id"] == "handoff_valid_1" and node["outcome_type"] == "stale_parent" for node in nodes)


def test_validation_failure_writes_repair_request(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.search_space["requires_model_script"] = True
    config.search_space["allow_open_model_families"] = True
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.splits_dir.mkdir(parents=True, exist_ok=True)
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


def test_cycle_pauses_for_decision_then_record_decision_promotes(tmp_path: Path) -> None:
    """A successful cycle leaves the comparison pending; record_decision finalises it."""
    from dataclasses import replace
    from autoresearch.config import PROJECT_ROOT
    from autoresearch.comparison_runner import record_decision
    from autoresearch.experiment_registry.registry import get_official_champion

    # root=PROJECT_ROOT so the protected-file integrity check (run inside
    # compare_experiments) can locate the real metrics/resampling modules.
    config = replace(_config(tmp_path), root=PROJECT_ROOT)
    config.search_space["requires_model_script"] = True
    config.search_space["allow_open_model_families"] = True
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.splits_dir.mkdir(parents=True, exist_ok=True)
    _write_fixtures(config)
    champion_config = tmp_path / "global_mean.toml"
    champion_config.write_text(
        """
experiment_name = "gm_champion"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )
    champion_id = json.loads(
        run_experiment(config, champion_config)["config_snapshot"].read_text(encoding="utf-8")
    )["experiment_id"]
    initialise_official_champion(config, champion_id)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"},'
        ' {"name": "vehicle_power_band_b", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )

    # Discriminative challenger: predicts in proportion to vehicle power band, so it
    # ranks the high-claim policy above the low-claim one → positive lift vs the
    # constant global-mean champion (passes the positive-lift validation gate).
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "power_model.py").write_text(
        """
import numpy as np

EXPOSURE = "exposure_term_a"


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hp):
    base = float(train["claim_cost_capped_active"].sum() / train[EXPOSURE].sum())
    rate = base * score["vehicle_power_band_b"].astype(float).to_numpy() / 4.0
    return rate * score[EXPOSURE].astype(float).to_numpy(), {"intent": "power-banded"}
""".strip(),
        encoding="utf-8",
    )
    proposal = _valid_proposal(champion_id)
    proposal["experiment_config"]["model_family"] = "scripted_power"
    proposal["experiment_config"]["model"] = {"script_path": "power_model.py"}
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    ingest_proposals(config)

    # Run the cycle — comparison must be left pending, champion unchanged.
    result = run_next_queued_proposal(config)

    assert result["decision"] == "pending_llm"
    comparison_id = result["comparison_id"]
    challenger_id = result["experiment_id"]
    record = next(p for p in list_proposals(config.registry_path) if p["proposal_id"] == proposal["proposal_id"])
    assert record["status"] == "awaiting_decision"
    assert get_official_champion(config.registry_path)["champion_id"] == champion_id  # unchanged

    # Agent records the decision — this finalises promotion + proposal status.
    res = record_decision(
        config,
        comparison_id,
        decision="promote",
        rationale="Improves the panel.",
        interpretation="The challenger improves the metric panel consistently.",
        next_step="Use the promoted model as the next parent.",
    )
    assert res["decision"] == "promote"
    assert get_official_champion(config.registry_path)["champion_id"] == challenger_id
    record = next(p for p in list_proposals(config.registry_path) if p["proposal_id"] == proposal["proposal_id"])
    assert record["status"] == "promoted"


def test_run_next_blocks_when_comparison_awaits_decision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config, "champion")
    initialise_official_champion(config, "champion")
    record_proposal(
        config.registry_path,
        proposal_id="pending_decision",
        status="awaiting_decision",
        parent_experiment_id="champion",
        parent_branch_id="main",
        branch_id="main",
        experiment_name="pending",
        rationale="pending",
        change_summary="pending",
        expected_benefit="pending",
        key_risk="pending",
        config={},
        validation_errors=[],
        llm_provider="test",
        llm_model=None,
        prompt_path=None,
        response_path=None,
        proposal_path=None,
        notes="Awaiting decision.",
    )
    update_proposal_status(
        config.registry_path,
        "pending_decision",
        "awaiting_decision",
        experiment_id="challenger",
        comparison_id="cmp_pending",
    )

    with pytest.raises(ValueError, match="record-decision"):
        run_next_queued_proposal(config)


def test_clear_loser_is_auto_rejected_after_single_split_screen(tmp_path: Path) -> None:
    """A structurally valid but clearly worse challenger skips expensive comparison."""
    from dataclasses import replace
    from autoresearch.config import PROJECT_ROOT

    config = replace(
        _config(tmp_path),
        root=PROJECT_ROOT,
        primary_metric="gini_weighted",
        screening_min_absolute_lift=-0.001,
        screening_min_relative_lift=-0.002,
    )
    config.search_space["requires_model_script"] = True
    config.search_space["allow_open_model_families"] = True
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.splits_dir.mkdir(parents=True, exist_ok=True)
    _write_fixtures(config)

    champion_config = tmp_path / "global_mean.toml"
    champion_config.write_text(
        """
experiment_name = "gm_screen_champion"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )
    champion_id = json.loads(
        run_experiment(config, champion_config)["config_snapshot"].read_text(encoding="utf-8")
    )["experiment_id"]
    initialise_official_champion(config, champion_id)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"},'
        ' {"name": "vehicle_power_band_b", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )

    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "inverse_model.py").write_text(
        """
import numpy as np

EXPOSURE = "exposure_term_a"


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hp):
    base = float(train["claim_cost_capped_active"].sum() / train[EXPOSURE].sum())
    inverse_power = 6.0 - score["vehicle_power_band_b"].astype(float).to_numpy()
    return base * inverse_power * score[EXPOSURE].astype(float).to_numpy(), {"intent": "inverse-power"}
""".strip(),
        encoding="utf-8",
    )
    proposal = _valid_proposal(champion_id)
    proposal["proposal_id"] = "clear_loser_1"
    proposal["experiment_name"] = "clear_loser_exp"
    proposal["experiment_config"]["experiment_name"] = "clear_loser_exp"
    proposal["experiment_config"]["model_family"] = "scripted_inverse_power"
    proposal["experiment_config"]["model"] = {"script_path": "inverse_model.py"}
    (config.handoff_proposal_inbox_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
    ingest_proposals(config)

    result = run_next_queued_proposal(config)
    proposals = list_proposals(config.registry_path)
    nodes = list_research_nodes(config.registry_path)

    assert result["decision"] == "auto_reject"
    assert result["comparison_id"] is None
    assert Path(result["comparison_report"]).exists()
    diagnostic_report = json.loads(Path(result["diagnostic_comparison_report"]).read_text(encoding="utf-8"))
    assert diagnostic_report["gate_mode"] == "single_partition"
    assert diagnostic_report["comparison_summary"]["n_resamples"] == 1
    assert diagnostic_report["bootstrap_summary"]["bootstrap_iterations"] == 1
    assert proposals[0]["status"] == "rejected"
    assert any(node["node_id"] == "clear_loser_1" and node["outcome_type"] == "clear_loser" for node in nodes)
    assert (config.handoff_results_dir / "latest_nonpromotion_summary.md").exists()
