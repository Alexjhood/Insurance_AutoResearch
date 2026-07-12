"""One-command recovery for evaluator processes orphaned by a delegation."""

from __future__ import annotations

import os
import signal
import time
from dataclasses import replace
from typing import Any

from autoresearch.config import load_config
from autoresearch.controller.session import (
    inflight_cycle_status,
    run_session_cycles,
    session_status,
)
from autoresearch.orchestration.campaign_log import append_note
from autoresearch.orchestration.manifest import load_orchestration
from autoresearch.orchestration.monitor import process_is_alive


def recover_delegation(orchestration_id: str, delegation_id: str) -> dict[str, Any]:
    """Terminate a confirmed orphan lock holder and run its recovery cycle."""

    orch = load_orchestration(orchestration_id)
    delegation = orch.delegation(delegation_id)
    config = replace(
        load_config(track_id=delegation.track, run_id=delegation.run_id, dataset=orch.dataset),
        target_mode=orch.target_mode,
    )
    status = session_status(config)
    session_id = status["state"]["session_id"]
    inflight = inflight_cycle_status(config, session_id)
    if inflight is None:
        raise ValueError(f"Delegation {delegation_id} has no in-flight cycle lock to recover.")

    holder_pid = int(inflight["pid"])
    holder_alive = bool(inflight["alive"])
    parent_alive = process_is_alive(delegation.pid)
    if holder_alive:
        if parent_alive:
            raise ValueError(
                f"Cycle-lock holder pid {holder_pid} is alive and delegation pid "
                f"{delegation.pid} is still alive; this is active work, not a confirmed orphan."
            )
        os.kill(holder_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while process_is_alive(holder_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if process_is_alive(holder_pid):
            raise RuntimeError(
                f"Confirmed orphan pid {holder_pid} did not terminate after SIGTERM; "
                "terminate it manually, then rerun this command."
            )

    states = run_session_cycles(config, 1, session_id)
    final = states[-1] if states else session_status(config)["state"]
    pending = final.get("latest_cycle_result") or {}
    if final.get("state") != "awaiting_decision" or not pending.get("comparison_id"):
        raise RuntimeError(
            f"Recovery cycle ended in {final.get('state')!r}, not awaiting_decision; "
            "inspect session-status for the child run."
        )
    append_note(
        orchestration_id,
        kind="other",
        delegation_id=delegation_id,
        text=(
            f"Recovered orphan cycle-lock holder pid {holder_pid}; comparison "
            f"{pending['comparison_id']} is awaiting decision."
        ),
    )
    return {
        "orchestration_id": orchestration_id,
        "delegation_id": delegation_id,
        "lock_holder_pid": holder_pid,
        "terminated": holder_alive,
        "state": final.get("state"),
        "pending_decision": pending,
        "track": delegation.track,
        "run_id": delegation.run_id,
    }
