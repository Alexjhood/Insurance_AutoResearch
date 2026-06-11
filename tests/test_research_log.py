import json
from pathlib import Path

from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import export_context_bundle
from autoresearch.controller.session import (
    _complete_pending_reflection_from_queued_proposal,
    create_session,
    record_cycle_reflection,
    record_session_decision,
)
from autoresearch.controller.workflow import enqueue_proposal_from_file
from autoresearch.experiment_registry.registry import (
    get_research_log_entry,
    list_research_log_entries,
    upsert_research_log_entry,
)
from autoresearch.research_log import render_research_log
from autoresearch.utils.io import read_json, write_json
from tests.test_handoff import _record_direct, _valid_proposal
from tests.test_runner import _make_config as _config


def _ready_config(tmp_path: Path):
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True)
    (config.metadata_dir / "agent_schema.json").write_text(
        '{"columns": [{"name": "exposure_term_a", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    return config


def test_research_log_render_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    upsert_research_log_entry(
        config.registry_path,
        session_id="session_a",
        cycle=1,
        proposal_id=None,
        experiment_id="experiment_a",
        comparison_id=None,
        hypothesis="Try a simpler model.",
        changes="Reduce model capacity.",
        outcome="auto_reject: lower weighted Gini",
        metrics={"lift": -0.01, "passed": False},
        interpretation="The removed capacity was useful.",
        next_step="Return to the incumbent and rotate model family.",
        completed_at="2026-06-10T12:00:00Z",
    )

    first = render_research_log(config).read_text(encoding="utf-8")
    second = render_research_log(config).read_text(encoding="utf-8")

    assert first == second
    assert first.count("## Cycle 1") == 1
    assert "**Hypothesis**: Try a simpler model." in first
    assert "`lift=-0.01`" in first
    assert "Pending agent" not in first


def test_decision_reflection_completes_final_session_cycle(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    session = create_session(config, "main", max_cycles=1)
    state_path = config.handoff_base_dir / "sessions" / session["session_id"] / "state.json"
    state = read_json(state_path)
    state["current_cycle"] = 1
    state["state"] = "awaiting_decision"
    state["latest_cycle_result"] = {
        "proposal_id": "proposal_a",
        "experiment_id": "experiment_a",
        "comparison_id": "comparison_a",
        "decision": "pending_llm",
    }
    write_json(state_path, state)
    upsert_research_log_entry(
        config.registry_path,
        session_id=session["session_id"],
        cycle=1,
        proposal_id=None,
        experiment_id="experiment_a",
        comparison_id="comparison_a",
        hypothesis="Test a new family.",
        changes="Replace the incumbent estimator.",
        outcome="awaiting LLM decision",
        metrics={"mean_lift": 0.02},
    )

    result = record_session_decision(
        config,
        comparison_id="comparison_a",
        decision="promote",
        details={
            "rationale": "Clear win.",
            "interpretation": "The new family captured useful structure.",
            "next_step": "Explore a different feature representation.",
        },
    )

    assert result is not None
    assert result["state"] == "completed"
    entry = get_research_log_entry(
        config.registry_path,
        session_id=session["session_id"],
        cycle=1,
    )
    assert entry["outcome"] == "promote: Clear win."
    assert entry["interpretation"] == "The new family captured useful structure."
    assert "**Next**: Explore a different feature representation." in config.research_log_path.read_text()


def test_final_auto_reject_uses_record_cycle_reflection(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    session = create_session(config, "main", max_cycles=1)
    state_path = config.handoff_base_dir / "sessions" / session["session_id"] / "state.json"
    state = read_json(state_path)
    state["current_cycle"] = 1
    state["state"] = "awaiting_reflection"
    write_json(state_path, state)
    upsert_research_log_entry(
        config.registry_path,
        session_id=session["session_id"],
        cycle=1,
        proposal_id=None,
        experiment_id="experiment_a",
        comparison_id=None,
        hypothesis="Try a smaller tree.",
        changes="Reduce leaves.",
        outcome="auto_reject: failed screen",
        metrics={"lift": -0.02},
    )

    result = record_cycle_reflection(
        config,
        interpretation="The smaller tree underfit.",
        next_step="Stop and retain the current champion.",
    )

    assert result["state"] == "completed"
    entries = list_research_log_entries(config.registry_path, session["session_id"])
    assert entries[0]["next_step"] == "Stop and retain the current champion."


def test_next_proposal_must_carry_auto_reject_reflection(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    session = create_session(config, "main", max_cycles=2)
    upsert_research_log_entry(
        config.registry_path,
        session_id=session["session_id"],
        cycle=1,
        proposal_id=None,
        experiment_id="experiment_a",
        comparison_id=None,
        hypothesis="Try a smaller tree.",
        changes="Reduce leaves.",
        outcome="auto_reject: failed screen",
        metrics={"lift": -0.02},
    )
    outputs = export_context_bundle(config)
    template = json.loads(outputs["proposal_template"].read_text(encoding="utf-8"))
    assert template["previous_cycle_reflection"]["cycle"] == 1
    assert "previous_cycle_reflection" in outputs["latest_handoff_markdown"].read_text(
        encoding="utf-8"
    )

    missing_path = tmp_path / "missing_reflection.json"
    missing_path.write_text(json.dumps(_valid_proposal()), encoding="utf-8")
    missing = enqueue_proposal_from_file(config, missing_path)
    assert missing["status"] == "failed"
    assert any("previous_cycle_reflection is required" in error for error in missing["validation_errors"])

    proposal = _valid_proposal()
    proposal["proposal_id"] = "handoff_valid_2"
    proposal["research_line_id"] = "line_handoff_valid_2"
    proposal["research_line_label"] = "Second handoff validation line"
    proposal["tree_policy_override_rationale"] = "Test isolates reflection validation."
    proposal["experiment_name"] = "handoff_global_mean_3"
    proposal["experiment_config"]["experiment_name"] = "handoff_global_mean_3"
    proposal["previous_cycle_reflection"] = {
        "cycle": 1,
        "interpretation": "The smaller tree underfit.",
        "next": "Rotate to a different estimator family.",
    }
    complete_path = tmp_path / "complete_reflection.json"
    complete_path.write_text(json.dumps(proposal), encoding="utf-8")

    complete = enqueue_proposal_from_file(config, complete_path)

    assert complete["status"] == "validated", complete["validation_errors"]
    assert _complete_pending_reflection_from_queued_proposal(config) is True
    entry = get_research_log_entry(
        config.registry_path,
        session_id=session["session_id"],
        cycle=1,
    )
    assert entry["interpretation"] == "The smaller tree underfit."
