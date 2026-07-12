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
    (config.metadata_dir / "dataset_schema.json").write_text(
        '{"columns": [{"name": "Exposure", "role": "numeric_feature"}]}',
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


def test_session_inherits_pinned_cycle_budget_from_manifest(tmp_path: Path) -> None:
    # bootstrap-track --cycles N pins the budget; sessions opened without
    # --max-cycles inherit it (run 20260612T105643Z ran 16 cycles when asked
    # for 15 because max_cycles was never bound).
    config = _ready_config(tmp_path)
    manifest_path = config.artifacts_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest["default_max_cycles"] = 15
    write_json(manifest_path, manifest)

    inherited = create_session(config, "budgeted")
    explicit = create_session(config, "explicit", max_cycles=3)

    assert inherited["max_cycles"] == 15
    assert explicit["max_cycles"] == 3


# ── crash-safe cycles: in-flight marker + orphan recovery ────────────────────


def _dead_pid() -> int:
    import subprocess

    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def _write_marker(config, session_id: str, pid: int) -> Path:
    marker = (
        config.handoff_base_dir / "sessions" / session_id / "cycle_in_flight.json"
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, {"pid": pid, "session_id": session_id, "cycle": 1,
                        "started_at": "2026-07-11T00:00:00Z"})
    return marker


def _queue_proposal(config, status: str, experiment_id: str | None = None) -> str:
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "first.json").write_text(
        json.dumps(_valid_proposal()), encoding="utf-8"
    )
    ingest_proposals(config)
    update_proposal_status(
        config.registry_path, "handoff_valid_1", status, experiment_id=experiment_id
    )
    return "handoff_valid_1"


def test_inflight_marker_with_live_pid_blocks_concurrent_cycle(tmp_path: Path) -> None:
    import os

    from autoresearch.controller.session import inflight_cycle_status

    config = _ready_config(tmp_path)
    session = create_session(config, "busy")
    marker = _write_marker(config, session["session_id"], os.getpid())

    status = inflight_cycle_status(config, session["session_id"])
    state = run_session_cycle(config, session["session_id"])

    assert status is not None and status["alive"] is True
    # The busy marker must survive and the session must not start a new cycle.
    assert marker.exists()
    assert state["current_cycle"] == 0


def test_dead_marker_requeues_proposal_killed_mid_fit(tmp_path: Path, monkeypatch) -> None:
    from autoresearch.experiment_registry.registry import list_session_events

    config = _ready_config(tmp_path)
    session = create_session(config, "recover", max_cycles=3)
    proposal_id = _queue_proposal(config, "running")  # killed mid-fit
    marker = _write_marker(config, session["session_id"], _dead_pid())

    canned = {
        "proposal_id": proposal_id,
        "experiment_id": "exp_recovered",
        "comparison_id": "cmp_recovered",
        "decision": "pending_llm",
        "metrics_summary": {},
    }
    seen_status: list[str] = []

    def fake_run(config_arg):
        seen_status.append(
            next(
                item["status"]
                for item in list_proposals(config_arg.registry_path)
                if item["proposal_id"] == proposal_id
            )
        )
        return canned

    monkeypatch.setattr(
        "autoresearch.controller.session.run_next_queued_proposal", fake_run
    )

    state = run_session_cycle(config, session["session_id"])

    events = [e["event_type"] for e in list_session_events(config.registry_path, session["session_id"])]
    assert "orphan_detected" in events
    assert "orphan_requeued" in events
    # The killed proposal was requeued before the normal flow evaluated it.
    assert seen_status == ["validated"]
    assert state["state"] == "awaiting_decision"
    assert state["current_cycle"] == 1
    assert not marker.exists()


def test_dead_marker_resumes_screened_proposal(tmp_path: Path, monkeypatch) -> None:
    from autoresearch.experiment_registry.registry import list_session_events

    config = _ready_config(tmp_path)
    session = create_session(config, "resume", max_cycles=3)
    proposal_id = _queue_proposal(config, "screened", experiment_id="exp_orphan")
    marker = _write_marker(config, session["session_id"], _dead_pid())

    resumed: list[str] = []

    def fake_resume(config_arg, orphan):
        resumed.append(orphan["proposal_id"])
        return {
            "proposal_id": orphan["proposal_id"],
            "experiment_id": "exp_orphan",
            "comparison_id": "cmp_resumed",
            "decision": "pending_llm",
            "metrics_summary": {},
        }

    monkeypatch.setattr(
        "autoresearch.controller.session.resume_orphaned_proposal", fake_resume
    )

    state = run_session_cycle(config, session["session_id"])

    events = [e["event_type"] for e in list_session_events(config.registry_path, session["session_id"])]
    assert "orphan_detected" in events
    assert resumed == [proposal_id]
    # The recovered comparison counts as a real cycle and stops for a decision.
    assert state["state"] == "awaiting_decision"
    assert state["current_cycle"] == 1
    assert not marker.exists()


def test_marker_cleared_after_normal_cycle_and_after_failure(tmp_path: Path, monkeypatch) -> None:
    config = _ready_config(tmp_path)
    session = create_session(config, "cleanup", max_cycles=3)
    _queue_proposal(config, "validated")
    marker = (
        config.handoff_base_dir / "sessions" / session["session_id"] / "cycle_in_flight.json"
    )

    def boom(config_arg):
        assert marker.exists()  # written before the evaluating phase starts
        raise RuntimeError("simulated mid-cycle crash")

    monkeypatch.setattr(
        "autoresearch.controller.session.run_next_queued_proposal", boom
    )

    import pytest

    with pytest.raises(RuntimeError, match="simulated mid-cycle crash"):
        run_session_cycle(config, session["session_id"])

    # Handled failures must not leave a stale marker; only a hard kill does.
    assert not marker.exists()


def test_find_and_requeue_orphaned_proposals(tmp_path: Path) -> None:
    from autoresearch.controller.workflow import (
        find_orphaned_proposals,
        requeue_orphaned_proposal,
    )

    config = _ready_config(tmp_path)
    proposal_id = _queue_proposal(config, "comparing", experiment_id="exp_x")

    orphans = find_orphaned_proposals(config)
    assert [item["proposal_id"] for item in orphans] == [proposal_id]

    requeue_orphaned_proposal(config, orphans[0])
    statuses = {
        item["proposal_id"]: item["status"] for item in list_proposals(config.registry_path)
    }
    assert statuses[proposal_id] == "validated"
    assert find_orphaned_proposals(config) == []
