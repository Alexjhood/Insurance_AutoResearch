"""Job lifecycle: launch, steer, pause, stop, reattach."""

from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any

from console.backend import config as cfg
from console.backend.orchestrator import db, telemetry
from console.backend.orchestrator.adapters.base import EventType, SteerResult
from console.backend.orchestrator.runner import get_runner

_runner = get_runner(cfg.REPO_ROOT, cfg.WORKTREES_DIR)

# In-process registry of running adapters keyed by job_id.
_LIVE: dict[str, Any] = {}

_AUTORESEARCH = cfg.AUTORESEARCH_BIN

SEED_TEMPLATES = {
    "claude": dedent("""\
        Read AGENT.md, then run `autoresearch --track {track} --run-id {run_id} bootstrap-track --model-provider {model_provider} --model-name {model_name}`, then `autoresearch --track {track} --run-id {run_id} start-session main --max-cycles {cycles}`, and run the adaptive cycle loop until the session reports its budget is exhausted. \
        The prepared dataset data/processed/agent_dataset_search.parquet already exists. \
        Do not use `--new-run`. \
        {guidance}
    """),
    "codex": dedent("""\
        Read AGENT.md, then run `autoresearch --track {track} --run-id {run_id} bootstrap-track --model-provider {model_provider} --model-name {model_name}`, then `autoresearch --track {track} --run-id {run_id} start-session main --max-cycles {cycles}`, and run the adaptive cycle loop until the session reports its budget is exhausted. \
        The prepared dataset data/processed/agent_dataset_search.parquet already exists. \
        Do not use `--new-run`. \
        {guidance}
    """),
    "opencode": dedent("""\
        Read AGENT.md, then run `autoresearch --track {track} --run-id {run_id} bootstrap-track --model-provider {model_provider} --model-name {model_name}`, then `autoresearch --track {track} --run-id {run_id} start-session main --max-cycles {cycles}`, and run the adaptive cycle loop until the session reports its budget is exhausted. \
        The prepared dataset data/processed/agent_dataset_search.parquet already exists. \
        Do not use `--new-run`. \
        {guidance}
    """),
}


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_env(job: dict) -> dict:
    env = {**os.environ}
    # Never propagate the holdout vault token into a research-scoped agent: the
    # agent triggers holdout evaluation on promotion in-process, and with the
    # token present it would write real holdout metrics into a run folder it is
    # allowed to read — a leakage channel the integrity scanner does not cover.
    env.pop("AUTORESEARCH_MILESTONE_TOKEN", None)
    env["AUTORESEARCH_SCOPE"] = job.get("scope", "research")
    env["AUTORESEARCH_MEMORY_ACCESS"] = job.get("memory_access", "none")
    env["AUTORESEARCH_TRACK"] = job["track"]
    env["AUTORESEARCH_RUN_ID"] = job["run_id"]
    if cfg.CLAUDE_BIN:
        env["CLAUDE_BIN"] = cfg.CLAUDE_BIN
    if cfg.CODEX_BIN:
        env["CODEX_BIN"] = cfg.CODEX_BIN
    if cfg.OPENCODE_BIN:
        env["OPENCODE_BIN"] = cfg.OPENCODE_BIN
    # Agent model + thinking/reasoning effort — each adapter translates these
    # to the right CLI flag (--model/--effort, -m/-c model_reasoning_effort, -m/--variant).
    if job.get("agent_model"):
        env["AGENT_MODEL"] = job["agent_model"]
    if job.get("agent_effort"):
        env["AGENT_EFFORT"] = job["agent_effort"]
    return env


def _get_adapter(surface: str):
    if surface == "claude":
        from console.backend.orchestrator.adapters.claude import ClaudeAdapter
        return ClaudeAdapter()
    if surface == "codex":
        from console.backend.orchestrator.adapters.codex import CodexAdapter
        return CodexAdapter()
    if surface == "opencode":
        from console.backend.orchestrator.adapters.opencode import OpenCodeAdapter
        return OpenCodeAdapter()
    raise ValueError(f"Unknown surface: {surface}")


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _run_autoresearch(job: dict, *args: str) -> None:
    """Run an autoresearch subcommand in the job's worktree."""
    worktree = job.get("worktree_path") or str(cfg.REPO_ROOT)
    subprocess.run(
        [_AUTORESEARCH, "--track", job["track"], "--run-id", job["run_id"], *args],
        cwd=worktree,
        check=True,
        capture_output=True,
    )


# ── Startup reattach ──────────────────────────────────────────────────────────

def reattach_running_jobs() -> None:
    """
    Called at orchestrator startup. For each job marked `running`:
    - If PID is alive: mark as `interrupted` (we lost the stdout pipe — the
      agent may still be running but we can no longer stream it. The job can
      be resumed via the resume endpoint if it has an agent_session_id).
    - If PID is dead: mark as `interrupted`.

    In both cases the job can be re-launched from the Launch page or resumed
    via the resume endpoint (which uses `--resume <agent_session_id>`).
    """
    for job in db.list_jobs():
        if job["status"] != "running":
            continue
        pid = job.get("pid")
        alive = _pid_alive(pid)
        db.update_job(job["id"], status="interrupted")
        db.record_event(job["id"], "system", {
            "msg": (
                f"Orchestrator restarted; pid {pid} {'still alive (lost pipe)' if alive else 'not found'}."
                " Job marked interrupted. Use Resume to reconnect via session_id."
                if job.get("agent_session_id") else
                f"Orchestrator restarted; pid {pid} {'still alive (lost pipe)' if alive else 'not found'}."
                " Job marked interrupted. Launch a new run to continue."
            ),
        })


# ── Internal drain helper ─────────────────────────────────────────────────────

def _start_drain(job_id: str, adapter: Any) -> None:
    def _drain():
        job = db.get_job(job_id) or {}
        turn_id: int | None = None
        output_chars = 0
        session_id = job.get("agent_session_id")
        model = job.get("agent_model")

        def ensure_turn() -> int:
            nonlocal turn_id
            if turn_id is None:
                turn_id = db.start_telemetry_turn(
                    job_id,
                    surface=job.get("surface") or "unknown",
                    session_id=session_id,
                    model=model,
                    effort=job.get("agent_effort"),
                )
            return turn_id

        def record_signals(payload: dict[str, Any]) -> None:
            for signal in telemetry.extract_signals(payload):
                db.record_telemetry_signal(job_id, turn_id, **signal)

        for event in adapter.stream():
            payload = event.payload

            # Capture agent_session_id as soon as the adapter emits it
            if event.type in (EventType.SYSTEM, EventType.TURN_END, EventType.AGENT_EXIT):
                sid = payload.get("session_id")
                if sid:
                    session_id = sid
                    current_job = db.get_job(job_id)
                    if current_job and not current_job.get("agent_session_id"):
                        db.update_job(job_id, agent_session_id=sid)

            db.record_event(job_id, event.type.value, payload)
            try:
                if event.type == EventType.SYSTEM:
                    if payload.get("model"):
                        model = payload["model"]
                    record_signals(payload)
                elif event.type == EventType.TOKEN:
                    ensure_turn()
                    output_chars += len(str(payload.get("text") or ""))
                elif event.type == EventType.TOOL_USE:
                    current_turn = ensure_turn()
                    normalized = telemetry.normalize_tool_payload(payload)
                    db.record_telemetry_tool_call(
                        job_id,
                        current_turn,
                        normalized=normalized,
                        raw_payload=payload,
                    )
                    record_signals(payload)
                elif event.type == EventType.TOOL_RESULT:
                    current_turn = ensure_turn()
                    normalized = telemetry.normalize_tool_payload(payload)
                    db.finish_telemetry_tool_call(
                        job_id,
                        current_turn,
                        normalized=normalized,
                        raw_payload=payload,
                    )
                    record_signals(payload)
                elif event.type == EventType.TURN_END:
                    current_turn = ensure_turn()
                    raw_usage = payload.get("usage")
                    usage = telemetry.normalize_usage(raw_usage, raw_result=payload)
                    db.finish_telemetry_turn(
                        current_turn,
                        session_id=session_id,
                        duration_ms=_optional_float(payload.get("duration_ms")),
                        is_error=payload.get("is_error")
                        if isinstance(payload.get("is_error"), bool)
                        else None,
                        output_chars=output_chars,
                        usage=usage,
                        raw_usage=raw_usage,
                        raw_result=payload,
                    )
                    record_signals(payload)
                    turn_id = None
                    output_chars = 0
                elif event.type == EventType.AGENT_EXIT and turn_id is not None:
                    returncode = payload.get("returncode")
                    db.finish_telemetry_turn(
                        turn_id,
                        session_id=session_id,
                        duration_ms=None,
                        is_error=returncode != 0 if isinstance(returncode, int) else None,
                        output_chars=output_chars,
                        usage=telemetry.normalize_usage(None),
                        raw_usage=None,
                        raw_result=payload,
                    )
                    turn_id = None
                    output_chars = 0
            except Exception as exc:
                # Telemetry must never break the agent stream or alter run status.
                db.record_event(job_id, "system", {
                    "msg": "Telemetry capture failed",
                    "event_type": event.type.value,
                    "error": str(exc),
                })

        # Finalize — but don't clobber a terminal status a user action already
        # set (stop/pause). Only move running → done when the stream ends on
        # its own.
        current = db.get_job(job_id)
        if current and current["status"] == "running":
            db.update_job(job_id, status="done")
            db.record_event(job_id, "system", {"msg": "Agent process exited"})
        _LIVE.pop(job_id, None)

    threading.Thread(target=_drain, daemon=True, name=f"drain-{job_id}").start()


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── Launch ────────────────────────────────────────────────────────────────────

def launch_job(
    track: str,
    surface: str,
    model_provider: str,
    model_name: str,
    cycles: int,
    memory_access: str,
    scope: str,
    guidance: str,
    agent_model: str = "",
    agent_effort: str = "",
) -> str:
    run_id = _make_run_id()
    seed_prompt = SEED_TEMPLATES[surface].format(
        track=track,
        run_id=run_id,
        cycles=cycles,
        model_provider=model_provider,
        model_name=model_name,
        guidance=guidance or "",
    ).strip()

    job_id = db.create_job(
        track=track,
        run_id=run_id,
        surface=surface,
        model_provider=model_provider,
        model_name=model_name,
        cycles=cycles,
        memory_access=memory_access,
        scope=scope,
        guidance=guidance,
        seed_prompt=seed_prompt,
        env={"AUTORESEARCH_SCOPE": scope, "AUTORESEARCH_MEMORY_ACCESS": memory_access},
        agent_model=agent_model,
        agent_effort=agent_effort,
    )

    try:
        worktree_path = _runner.create_worktree(track, run_id)
        db.update_job(job_id, worktree_path=str(worktree_path))
        db.record_event(job_id, "system", {"msg": f"Worktree created at {worktree_path}"})
    except subprocess.CalledProcessError as exc:
        db.update_job(job_id, status="failed")
        db.record_event(job_id, "system", {"error": exc.stderr.decode()})
        raise RuntimeError(f"Worktree creation failed: {exc.stderr.decode()}")

    adapter = _get_adapter(surface)
    full_env = _build_env(db.get_job(job_id))
    handle = adapter.launch(cwd=worktree_path, env=full_env, seed_prompt=seed_prompt)
    db.update_job(job_id, status="running", pid=handle.pid)
    db.record_event(job_id, "system", {"msg": "Agent launched", "pid": handle.pid})

    _start_drain(job_id, adapter)
    _LIVE[job_id] = adapter
    return job_id


# ── Steer ─────────────────────────────────────────────────────────────────────

def steer_job(job_id: str, message: str, interrupt: bool = True) -> SteerResult:
    adapter = _LIVE.get(job_id)
    if not adapter:
        db.enqueue_steer(job_id, message, interrupt)
        return SteerResult(mode="queued", ok=True, detail="Agent not live — queued for delivery")
    result = adapter.steer(message, interrupt=interrupt)
    event_type = "steer_sent" if result.ok and result.mode == "interrupt" else "steer_queued"
    db.record_event(job_id, event_type, {"message": message, "mode": result.mode})
    if result.mode == "queued":
        db.enqueue_steer(job_id, message, interrupt)
    return result


# ── Pause ─────────────────────────────────────────────────────────────────────

def pause_job(job_id: str) -> None:
    """
    Pause a running session. Calls `autoresearch pause-session` so the
    framework stops auto-promotion, then marks the job paused.
    The adapter process keeps running — we do NOT kill it here.
    """
    job = db.get_job(job_id)
    if not job:
        raise KeyError(f"Job {job_id} not found")
    _run_autoresearch(job, "pause-session")
    db.update_job(job_id, status="paused")
    db.record_event(job_id, "system", {"msg": "Session paused via autoresearch pause-session"})


# ── Resume ────────────────────────────────────────────────────────────────────

def resume_job(job_id: str) -> None:
    """
    Resume a paused or interrupted job.

    Paused (process still alive in _LIVE):
        Call autoresearch resume-session; mark running; deliver queued steers.

    Interrupted (process died, orchestrator restarted):
        If agent_session_id is available, re-spawn with `--resume <id>`.
        Otherwise mark running and let the next user prompt drive it.
    """
    job = db.get_job(job_id)
    if not job:
        raise KeyError(f"Job {job_id} not found")

    prev_status = job["status"]

    if prev_status == "paused":
        # Tell the autoresearch framework to allow the session to proceed
        try:
            _run_autoresearch(job, "resume-session")
        except subprocess.CalledProcessError:
            pass  # Best-effort; process may have already exited

        db.update_job(job_id, status="running")
        db.record_event(job_id, "system", {"msg": "Session resumed"})

        adapter = _LIVE.get(job_id)
        if adapter:
            for msg in _drain_steer_queue(job_id):
                adapter.steer(msg, interrupt=False)
                db.record_event(job_id, "steer_sent", {"message": msg, "mode": "resume-delivery"})

    elif prev_status in ("interrupted", "stopped"):
        session_id = job.get("agent_session_id")
        surface = job.get("surface")
        # All three adapters expose resume(session_id=...): Claude via --resume,
        # Codex via `exec resume <thread_id>`, OpenCode via `run -s <sessionID>`.
        if session_id and surface in ("claude", "codex", "opencode"):
            worktree = Path(job["worktree_path"]) if job.get("worktree_path") else cfg.REPO_ROOT
            adapter = _get_adapter(surface)
            full_env = _build_env(job)
            queued = _drain_steer_queue(job_id)
            extra = " ".join(queued) if queued else ""
            handle = adapter.resume(cwd=worktree, env=full_env,
                                    session_id=session_id, extra_prompt=extra)
            db.update_job(job_id, status="running", pid=handle.pid)
            db.record_event(job_id, "system", {
                "msg": f"Resumed {surface} session {session_id}", "pid": handle.pid
            })
            _start_drain(job_id, adapter)
            _LIVE[job_id] = adapter
        else:
            db.update_job(job_id, status="running")
            db.record_event(job_id, "system", {
                "msg": "Marked running (no session_id available for surface re-attach — "
                       "agent may have exited; use Launch to start fresh)"
            })
    else:
        raise ValueError(f"Job {job_id} has status={prev_status!r}; can only resume paused/interrupted jobs")


# ── Stop ──────────────────────────────────────────────────────────────────────

def stop_job(job_id: str) -> None:
    # Set the terminal status BEFORE ending the stream, so the drain thread
    # observes "stopped" and does not overwrite it with "done".
    db.update_job(job_id, status="stopped")
    db.record_event(job_id, "system", {"msg": "Job stopped by user"})
    adapter = _LIVE.pop(job_id, None)
    if adapter:
        adapter.stop()

    job = db.get_job(job_id)
    wt = job.get("worktree_path") if job else None
    if wt:
        _runner.remove_worktree(Path(wt))


# ── Steer queue ───────────────────────────────────────────────────────────────

def _drain_steer_queue(job_id: str) -> list[str]:
    conn = db._connect()
    rows = conn.execute(
        "SELECT id, message FROM steer_queue WHERE job_id=? AND status='pending' ORDER BY id ASC",
        (job_id,),
    ).fetchall()
    messages = [row["message"] for row in rows]
    for row in rows:
        conn.execute(
            "UPDATE steer_queue SET status='sent', sent_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), row["id"]),
        )
    conn.commit()
    conn.close()
    return messages
