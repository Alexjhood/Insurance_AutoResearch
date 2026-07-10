"""Observe detached delegations, enforce timeout state, and stop processes."""

from __future__ import annotations

import errno
import json
import os
import signal
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch.config import PROJECT_ROOT
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    exit_status_path,
    load_orchestration,
    manifest_lock,
    save_orchestration,
    update_delegation,
    utc_stamp,
)


def process_is_alive(pid: int | None) -> bool:
    """Return whether *pid* exists; permission-denied still means alive."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        if exc.errno == errno.EPERM:
            return True
        raise
    return True


def refresh_orchestration(
    orchestration_id: str, *, now: datetime | None = None
) -> Orchestration:
    """Refresh process-backed statuses without ever killing a child.

    A live process beyond its allowance is marked ``timed_out``. The process is
    deliberately left running until an operator calls :func:`kill_delegation`.
    """

    current_time = now or datetime.now(timezone.utc)
    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        changed = False
        for delegation in orch.delegations:
            updated = _observed_delegation(
                orchestration_id, delegation, now=current_time
            )
            if updated != delegation:
                orch = update_delegation(orch, updated)
                changed = True
        if changed:
            save_orchestration(orch)
    return orch


def _observed_delegation(
    orchestration_id: str, delegation: Delegation, *, now: datetime
) -> Delegation:
    exit_record = _read_exit_record(orchestration_id, delegation.delegation_id)
    alive = process_is_alive(delegation.pid)

    if exit_record is not None:
        exit_code = int(exit_record["exit_code"])
        ended_at = str(exit_record.get("ended_at") or _format_time(now))
        status = delegation.status
        if status not in {"timed_out", "killed"}:
            status = "completed" if exit_code == 0 else "failed"
        return replace(
            delegation,
            status=status,
            exit_code=exit_code,
            ended_at=delegation.ended_at or ended_at,
        )

    if delegation.status in {"completed", "failed", "killed"}:
        return delegation
    if delegation.status == "timed_out":
        return delegation
    if alive:
        elapsed = elapsed_minutes(delegation, now=now)
        if (
            delegation.timeout_minutes is not None
            and elapsed is not None
            and elapsed >= delegation.timeout_minutes
        ):
            return replace(delegation, status="timed_out", ended_at=_format_time(now))
        if delegation.status != "running":
            return replace(delegation, status="running")
        return delegation
    if delegation.pid is not None:
        return replace(delegation, status="failed", ended_at=_format_time(now))
    return delegation


def status_rows(
    orchestration_id: str, *, now: datetime | None = None
) -> tuple[Orchestration, list[dict[str, Any]]]:
    """Return refreshed manifest state plus one progress row per delegation."""

    current_time = now or datetime.now(timezone.utc)
    orch = refresh_orchestration(orchestration_id, now=current_time)
    rows: list[dict[str, Any]] = []
    for delegation in orch.delegations:
        progress = _registry_progress(delegation)
        rows.append(
            {
                "delegation_id": delegation.delegation_id,
                "backend": delegation.backend,
                "track": delegation.track,
                "run_id": delegation.run_id,
                "status": delegation.status,
                "pid": delegation.pid,
                "process_alive": process_is_alive(delegation.pid),
                "cycles_completed": progress["cycles_completed"],
                "cycle_budget": delegation.cycle_budget,
                "champion_id": progress["champion_id"],
                "gini_weighted": progress["gini_weighted"],
                "last_activity": _last_activity(delegation),
                "elapsed_minutes": elapsed_minutes(delegation, now=current_time),
                "timeout_minutes": delegation.timeout_minutes,
            }
        )
    return orch, rows


def format_status_table(orch: Orchestration, rows: list[dict[str, Any]]) -> str:
    """Render a compact status table suitable for repeated polling."""

    lines = [
        f"Orchestration {orch.orchestration_id}: {orch.status} "
        f"cycles={orch.cycles_committed}/{orch.total_cycle_budget}",
        "delegation  backend         status      alive  cycles  gini      elapsed/timeout  run",
    ]
    for row in rows:
        gini = row["gini_weighted"]
        gini_text = "—" if gini is None else f"{gini:.4f}"
        elapsed = row["elapsed_minutes"]
        timeout = row["timeout_minutes"]
        elapsed_text = "—" if elapsed is None else f"{elapsed:.1f}m"
        timeout_text = "—" if timeout is None else f"{timeout:.0f}m"
        lines.append(
            f"{row['delegation_id']:<11} {row['backend']:<15} {row['status']:<11} "
            f"{str(row['process_alive']).lower():<6} "
            f"{row['cycles_completed']}/{row['cycle_budget']:<5} {gini_text:<9} "
            f"{elapsed_text}/{timeout_text:<8} {row['track']}/{row['run_id']}"
        )
    if not rows:
        lines.append("(no delegations)")
    return "\n".join(lines) + "\n"


def kill_delegation(orchestration_id: str, delegation_id: str) -> Delegation:
    """Terminate a detached delegation's process group and mark it killed."""

    orch = refresh_orchestration(orchestration_id)
    delegation = orch.delegation(delegation_id)
    if delegation.status in {"completed", "failed", "killed"}:
        raise ValueError(
            f"Delegation {delegation_id} is already {delegation.status}; nothing to kill."
        )
    if delegation.pid is None or not process_is_alive(delegation.pid):
        raise ValueError(f"Delegation {delegation_id} has no live process to kill.")

    try:
        os.killpg(delegation.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        os.kill(delegation.pid, signal.SIGTERM)

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        delegation = orch.delegation(delegation_id)
        delegation = replace(delegation, status="killed", ended_at=utc_stamp())
        save_orchestration(update_delegation(orch, delegation))
    return delegation


def elapsed_minutes(
    delegation: Delegation, *, now: datetime | None = None
) -> float | None:
    started = _parse_time(delegation.spawned_at)
    if started is None:
        return None
    end = _parse_time(delegation.ended_at) or now or datetime.now(timezone.utc)
    return max(0.0, (end - started).total_seconds() / 60.0)


def _read_exit_record(
    orchestration_id: str, delegation_id: str
) -> dict[str, Any] | None:
    path = exit_status_path(orchestration_id, delegation_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        int(payload["exit_code"])
        return payload
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _registry_progress(delegation: Delegation) -> dict[str, Any]:
    try:
        from autoresearch.orchestration.report import (
            _champion_facts,
            _child_config,
            _cycles_used,
        )

        config = _child_config(delegation)
        champion = _champion_facts(config)
        cycles = max(0, _cycles_used(config.registry_path) - delegation.cycles_at_start)
        return {
            "cycles_completed": cycles,
            "champion_id": champion.get("experiment_id"),
            "gini_weighted": champion.get("gini_weighted"),
        }
    except Exception:
        return {"cycles_completed": 0, "champion_id": None, "gini_weighted": None}


def _last_activity(delegation: Delegation) -> str | None:
    candidates = [delegation.run_dir() / "registry.sqlite"]
    if delegation.log_path:
        stored = Path(delegation.log_path)
        candidates.append(stored if stored.is_absolute() else PROJECT_ROOT / stored)
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    if not mtimes:
        return delegation.spawned_at
    return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
