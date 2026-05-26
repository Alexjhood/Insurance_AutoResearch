"""Supervised autonomous session orchestration for file-handoff research."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.controller.handoff import (
    export_context_bundle,
    inbox_status,
    ingest_proposals,
    write_nonpromotion_summary,
)
from autoresearch.controller.workflow import ExperimentNeedsRepair, run_next_queued_proposal
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_proposals,
    list_session_events,
    list_sessions,
    record_session_event,
    upsert_session,
)
from autoresearch.utils.io import read_json, write_json


SESSION_STATES = {
    "idle",
    "running",
    "waiting_for_proposal",
    "ingesting",
    "evaluating",
    "comparing",
    "promoted",
    "rejected",
    "inconclusive",
    "waiting_for_repair",
    "paused",
    "failed",
    "completed",
}


def create_session(config: ProjectConfig, name: str, max_cycles: int | None = None) -> dict[str, Any]:
    """Create a named autonomous research session."""

    ensure_project_dirs(config)
    session_id = _session_id(name)
    state = _base_state(session_id, name, max_cycles)
    _persist_state(config, state, event_type="created", message="Session created.")
    export_context_bundle(config)
    return state


def latest_session(config: ProjectConfig) -> dict[str, Any] | None:
    """Load the latest session state, if any."""

    latest_path = _sessions_dir(config) / "latest_session_id.txt"
    if latest_path.exists():
        return load_session(config, latest_path.read_text(encoding="utf-8").strip())
    rows = list_sessions(config.registry_path)
    return load_session(config, rows[0]["session_id"]) if rows else None


def load_session(config: ProjectConfig, session_id: str) -> dict[str, Any]:
    """Load one session state from disk."""

    path = _session_dir(config, session_id) / "state.json"
    if not path.exists():
        raise ValueError(f"Unknown session id: {session_id}")
    return read_json(path)


def pause_session(config: ProjectConfig, session_id: str | None = None) -> dict[str, Any]:
    """Pause a session cleanly."""

    state = _require_session(config, session_id)
    state["state"] = "paused"
    state["paused_at"] = _now()
    _persist_state(config, state, event_type="paused", message="Session paused.")
    export_context_bundle(config)
    return state


def resume_session(config: ProjectConfig, session_id: str | None = None) -> dict[str, Any]:
    """Resume a paused or waiting session."""

    state = _require_session(config, session_id)
    state["state"] = "idle"
    state["stop_requested"] = False
    state["resumed_at"] = _now()
    _persist_state(config, state, event_type="resumed", message="Session resumed.")
    export_context_bundle(config)
    return state


def stop_session(config: ProjectConfig, session_id: str | None = None) -> dict[str, Any]:
    """Request a clean stop after current work."""

    state = _require_session(config, session_id)
    state["stop_requested"] = True
    state["state"] = "completed"
    state["completed_at"] = _now()
    _persist_state(config, state, event_type="stopped", message="Session stop requested/completed.")
    export_context_bundle(config)
    return state


def session_status(config: ProjectConfig, session_id: str | None = None) -> dict[str, Any]:
    """Return latest session status plus recent events."""

    state = _require_session(config, session_id)
    return {
        "state": state,
        "events": list_session_events(config.registry_path, state["session_id"], limit=20),
        "inbox": inbox_status(config),
    }


def run_session_cycle(config: ProjectConfig, session_id: str | None = None) -> dict[str, Any]:
    """Run one local side of the file-handoff cycle from current state."""

    state = _require_session(config, session_id)
    if state["state"] == "paused":
        return _record_waiting(config, state, "Session is paused.")
    if state.get("stop_requested") or state["state"] == "completed":
        return _record_waiting(config, state, "Session is stopped or completed.")

    state["state"] = "running"
    _persist_state(config, state, event_type="running", message="Cycle started.")

    inbox = inbox_status(config)
    queued_before = _queued_count(config)
    if inbox["inbox_json_count"] == 0 and queued_before == 0:
        state["state"] = "waiting_for_proposal"
        _persist_state(config, state, event_type="waiting_for_proposal", message="Waiting for an external proposal file.")
        export_context_bundle(config)
        return state

    state["state"] = "ingesting"
    _persist_state(config, state, event_type="ingesting", message="Ingesting inbox proposals.")
    ingest_summary = ingest_proposals(config)

    if _queued_count(config) == 0:
        state["state"] = "waiting_for_proposal"
        state["latest_ingest_summary"] = ingest_summary
        _persist_state(config, state, event_type="waiting_for_proposal", message="No validated proposal is queued.")
        export_context_bundle(config)
        return state

    state["state"] = "evaluating"
    _persist_state(config, state, event_type="evaluating", message="Running next queued proposal.")
    try:
        result = run_next_queued_proposal(config)
    except ExperimentNeedsRepair as exc:
        state["state"] = "waiting_for_repair"
        state["latest_error"] = str(exc)
        _persist_state(config, state, event_type="waiting_for_repair", message=str(exc))
        export_context_bundle(config)
        return state
    except Exception as exc:
        state["state"] = "failed"
        state["latest_error"] = str(exc)
        _persist_state(config, state, event_type="failed", message=str(exc))
        raise

    state["current_cycle"] += 1
    state["latest_cycle_result"] = result
    state["latest_ingest_summary"] = ingest_summary
    state["state"] = "promoted" if result.get("decision") == "promote" else "inconclusive"
    if result.get("decision") != "promote":
        write_nonpromotion_summary(
            config,
            proposal_id=result["proposal_id"],
            outcome_type=state["state"],
            reason="Proposal did not satisfy volatility-aware promotion thresholds.",
            quantitative_signal={"comparison_id": result.get("comparison_id")},
        )

    _persist_state(
        config,
        state,
        event_type="cycle_completed",
        proposal_id=result.get("proposal_id"),
        experiment_id=result.get("experiment_id"),
        comparison_id=result.get("comparison_id"),
        message=f"Cycle completed with decision {result.get('decision')}.",
        details=result,
    )
    _write_latest_cycle_summary(config, state)

    if state.get("max_cycles") is not None and state["current_cycle"] >= state["max_cycles"]:
        state["state"] = "completed"
        state["completed_at"] = _now()
        _persist_state(config, state, event_type="completed", message="Session reached max_cycles.")
    elif state.get("stop_requested"):
        state["state"] = "completed"
        state["completed_at"] = _now()
        _persist_state(config, state, event_type="completed", message="Session stopped after current cycle.")

    export_context_bundle(config)
    return state


def run_session_cycles(config: ProjectConfig, count: int, session_id: str | None = None) -> list[dict[str, Any]]:
    """Run up to N local-side cycles, stopping when a proposal is needed."""

    if count <= 0:
        raise ValueError("count must be positive")
    states = []
    for _ in range(count):
        state = run_session_cycle(config, session_id)
        states.append(state)
        if state["state"] in {"waiting_for_proposal", "waiting_for_repair", "paused", "failed", "completed"}:
            break
    return states


def _base_state(session_id: str, name: str, max_cycles: int | None) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "name": name,
        "state": "idle",
        "created_at": _now(),
        "updated_at": _now(),
        "current_cycle": 0,
        "max_cycles": max_cycles,
        "stop_requested": False,
        "official_champion": None,
    }


def _persist_state(
    config: ProjectConfig,
    state: dict[str, Any],
    *,
    event_type: str,
    message: str,
    proposal_id: str | None = None,
    experiment_id: str | None = None,
    comparison_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    state["updated_at"] = _now()
    state["official_champion"] = get_official_champion(config.registry_path)
    session_dir = _session_dir(config, state["session_id"])
    session_dir.mkdir(parents=True, exist_ok=True)
    state_path = session_dir / "state.json"
    summary_path = session_dir / "summary.md"
    log_path = session_dir / "events.jsonl"
    write_json(state_path, state)
    summary_path.write_text(_render_session_summary(state), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at": _now(), "event_type": event_type, "message": message, "details": details or {}}) + "\n")
    (_sessions_dir(config) / "latest_session_id.txt").write_text(state["session_id"], encoding="utf-8")
    write_json(config.handoff_results_dir / "latest_session_summary.json", state)
    (config.handoff_results_dir / "latest_session_summary.md").write_text(_render_session_summary(state), encoding="utf-8")
    upsert_session(
        config.registry_path,
        session_id=state["session_id"],
        name=state["name"],
        state=state["state"],
        current_cycle=state["current_cycle"],
        max_cycles=state.get("max_cycles"),
        stop_requested=bool(state.get("stop_requested")),
        state_path=state_path,
        summary_path=summary_path,
        notes=message,
    )
    record_session_event(
        config.registry_path,
        session_id=state["session_id"],
        event_type=event_type,
        state=state["state"],
        proposal_id=proposal_id,
        experiment_id=experiment_id,
        comparison_id=comparison_id,
        message=message,
        details=details,
    )


def _write_latest_cycle_summary(config: ProjectConfig, state: dict[str, Any]) -> None:
    result = state.get("latest_cycle_result", {})
    summary = {
        "completed_at": _now(),
        "session_id": state["session_id"],
        "cycle": state["current_cycle"],
        "cycle_result": result,
        "official_champion": get_official_champion(config.registry_path),
    }
    write_json(config.handoff_results_dir / "latest_cycle_result.json", summary)
    lines = [
        "# Latest Session Cycle Result",
        "",
        f"- session_id: `{state['session_id']}`",
        f"- cycle: {state['current_cycle']}",
        f"- state: `{state['state']}`",
        f"- proposal_id: `{result.get('proposal_id')}`",
        f"- experiment_id: `{result.get('experiment_id')}`",
        f"- comparison_id: `{result.get('comparison_id')}`",
        f"- decision: `{result.get('decision')}`",
    ]
    (config.handoff_results_dir / "latest_cycle_result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_waiting(config: ProjectConfig, state: dict[str, Any], message: str) -> dict[str, Any]:
    _persist_state(config, state, event_type=state["state"], message=message)
    export_context_bundle(config)
    return state


def _require_session(config: ProjectConfig, session_id: str | None) -> dict[str, Any]:
    state = load_session(config, session_id) if session_id else latest_session(config)
    if state is None:
        raise ValueError("No session exists. Run start-session first.")
    return state


def _queued_count(config: ProjectConfig) -> int:
    return sum(
        1
        for item in list_proposals(config.registry_path)
        if item["status"] in {"validated", "proposed", "needs_repair"}
    )


def _session_id(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_")
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe or 'session'}"


def _sessions_dir(config: ProjectConfig) -> Path:
    path = config.handoff_base_dir / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_dir(config: ProjectConfig, session_id: str) -> Path:
    return _sessions_dir(config) / session_id


def _render_session_summary(state: dict[str, Any]) -> str:
    champion = state.get("official_champion") or {}
    return "\n".join(
        [
            "# Auto-Research Session Summary",
            "",
            f"- session_id: `{state['session_id']}`",
            f"- name: {state['name']}",
            f"- state: `{state['state']}`",
            f"- current_cycle: {state['current_cycle']}",
            f"- max_cycles: {state.get('max_cycles')}",
            f"- stop_requested: {state.get('stop_requested')}",
            f"- official_champion: `{champion.get('champion_id')}`",
            f"- updated_at: {state['updated_at']}",
        ]
    ) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
