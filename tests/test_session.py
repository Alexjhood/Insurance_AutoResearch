import json
from pathlib import Path

from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import ingest_proposals
from autoresearch.controller.session import create_session, pause_session, resume_session, run_session_cycle, session_status, stop_session
from autoresearch.experiment_registry.registry import list_proposals, list_sessions, update_proposal_status
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


def test_session_wait_pause_resume_stop(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)

    session = create_session(config, "smoke", max_cycles=2)
    waiting = run_session_cycle(config, session["session_id"])
    paused = pause_session(config, session["session_id"])
    resumed = resume_session(config, session["session_id"])
    stopped = stop_session(config, session["session_id"])
    status = session_status(config, session["session_id"])

    assert waiting["state"] == "waiting_for_proposal"
    assert paused["state"] == "paused"
    assert resumed["state"] == "idle"
    assert stopped["state"] == "completed"
    assert status["state"]["stop_requested"] is True
    assert list_sessions(config.registry_path)[0]["state"] == "completed"


def test_session_clears_stale_error_on_new_cycle(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    session = create_session(config, "smoke")
    state_path = config.handoff_base_dir / "sessions" / session["session_id"] / "state.json"
    state = read_json(state_path)
    state["latest_error"] = "old repair error"
    write_json(state_path, state)

    result = run_session_cycle(config, session["session_id"])

    assert result["state"] == "waiting_for_proposal"
    assert "latest_error" not in result


def test_session_status_reports_running_proposals(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    session = create_session(config, "smoke")
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "first.json").write_text(json.dumps(_valid_proposal()), encoding="utf-8")
    ingest_proposals(config)
    update_proposal_status(config.registry_path, "handoff_valid_1", "running", notes="test running")

    status = session_status(config, session["session_id"])

    assert status["running_proposals"][0]["proposal_id"] == "handoff_valid_1"
    assert status["running_proposals"][0]["stale_after_seconds"] > 0


def test_second_proposal_is_deferred_while_one_is_queued(tmp_path: Path) -> None:
    config = _ready_config(tmp_path)
    config.handoff_proposal_inbox_dir.mkdir(parents=True)
    first = _valid_proposal()
    second = _valid_proposal()
    second["proposal_id"] = "handoff_valid_2"
    second["experiment_name"] = "handoff_alpha_2_repeat"
    second["experiment_config"]["experiment_name"] = "handoff_alpha_2_repeat"

    (config.handoff_proposal_inbox_dir / "first.json").write_text(json.dumps(first), encoding="utf-8")
    first_summary = ingest_proposals(config)
    (config.handoff_proposal_inbox_dir / "second.json").write_text(json.dumps(second), encoding="utf-8")
    second_summary = ingest_proposals(config)
    proposals = list_proposals(config.registry_path)

    assert first_summary["valid_count"] == 1
    assert second_summary["deferred_count"] == 1
    assert not any(item["status"] == "duplicate" for item in proposals)
    assert (config.handoff_proposal_inbox_dir / "second.json").exists()
