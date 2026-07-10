"""Spawn a sub-agent: pre-bootstrap its run, compose its prompt, launch the tool.

Everything the framework can decide is decided *before* the sub-agent model sees
anything. By the time the child process starts, its run exists, its champion is
seeded, and its handoff already carries the Orchestration brief. The child then
follows the ordinary AGENT.md workflow from step 2.

The child is confined by the guard's existing ``SessionStart`` env binding
(``AUTORESEARCH_SCOPE=research`` + ``AUTORESEARCH_TRACK`` + ``AUTORESEARCH_RUN_ID``),
so no new guard machinery runs on the child side.

Bootstrap happens under :func:`manifest_lock` because it mutates the track's
``latest_run.json``; children always pass ``--run-id``, so nothing downstream
depends on "latest".
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from autoresearch.config import PROJECT_ROOT, load_config
from autoresearch.orchestration.backends import Backend, get_backend
from autoresearch.orchestration.brief import Brief, load_brief
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    add_delegation,
    exit_status_path,
    load_orchestration,
    log_path,
    manifest_lock,
    prompt_path,
    save_orchestration,
    update_delegation,
    utc_stamp,
    write_run_backpointer,
)
from autoresearch.utils.io import read_json, write_json


#: Per-cycle compute budget grows with experiment count (AGENT.md "Compute
#: budget"); the timeout applies a safety factor over the whole delegation.
#: Design §10 flags the factor as a guess to tune after the first real campaigns.
TIMEOUT_SAFETY_FACTOR = 2.0


@dataclass(frozen=True)
class SpawnPlan:
    """Everything a spawn will do, computed before anything is launched.

    ``--dry-run`` prints this instead of launching: the exact argv and the full
    child environment are the review surface for spawn correctness.
    """

    orchestration_id: str
    delegation_id: str
    backend: Backend
    track: str
    run_id: str
    cycle_budget: int
    command: tuple[str, ...]
    env: dict[str, str]
    prompt: str
    timeout_minutes: float

    def render(self) -> str:
        lines = [
            f"Orchestration : {self.orchestration_id}",
            f"Delegation    : {self.delegation_id}",
            f"Backend       : {self.backend.name} (tool={self.backend.tool}, "
            f"tier={self.backend.tier}, status={self.backend.status})",
            f"Child run     : {self.track}/{self.run_id}",
            f"Cycle budget  : {self.cycle_budget}",
            f"Timeout       : {self.timeout_minutes:.0f} min",
            f"Prompt via    : {self.backend.prompt_via}",
            "",
            "Command argv:",
        ]
        lines.extend(f"  [{i}] {arg}" for i, arg in enumerate(self.command))
        lines.extend(["", "Child environment (AUTORESEARCH_* overrides):"])
        lines.extend(f"  {key}={self.env[key]}" for key in sorted(self.env) if key.startswith("AUTORESEARCH_"))
        lines.extend(["", "Prompt:", "", self.prompt])
        return "\n".join(lines) + "\n"


def compute_timeout_minutes(cycle_budget: int) -> float:
    """Wall-clock allowance: ``cycle_budget × per-cycle budget × safety factor``.

    The per-cycle compute budget itself grows as a run accumulates experiments
    (``10 + 5 × (N // 5)`` minutes), so we integrate it over the budgeted cycles
    rather than assuming the first cycle's allowance for all of them.
    """

    per_cycle_total = sum(10 + 5 * (n // 5) for n in range(cycle_budget))
    return per_cycle_total * TIMEOUT_SAFETY_FACTOR


def compose_prompt(*, track: str, run_id: str, cycle_budget: int) -> str:
    """The launch prompt. Short by design — AGENT.md/AGENTS.md carry the contract.

    Both tools auto-load the contract from the repo root, so this only has to
    establish the three facts the contract cannot know: which run is already
    bootstrapped, that bootstrap must be skipped, and how to finish.
    """

    return (
        "You are a research sub-agent working under an orchestrator. Your run is "
        f"**already bootstrapped**: track `{track}`, run id `{run_id}`, cycle budget "
        f"`{cycle_budget}` — do **not** run `bootstrap-track` and do not start any "
        "other run.\n\n"
        f"Begin with `autoresearch --track {track} --run-id {run_id} show-latest-handoff` "
        "and follow the standard workflow from step 2 of the contract. Your handoff "
        "contains an **Orchestration brief** — treat its direction, constraints, and "
        "stop conditions as binding, on par with the Active dataset block. Spend your "
        f"{cycle_budget} cycles adaptively within the brief.\n\n"
        "When your budget is exhausted (or a brief stop-condition fires), finish with "
        f"`autoresearch --track {track} --run-id {run_id} orchestrate finish-delegation "
        '--summary "<3–6 sentence scientific summary: what you learned, what you\'d try '
        'next, anything that smelled artifactual>"` and then stop.\n'
    )


def child_environment(
    *, track: str, run_id: str, memory_access: str | None = None
) -> dict[str, str]:
    """The child's environment: inherited, plus the guard's scope binding."""

    env = dict(os.environ)
    env["AUTORESEARCH_SCOPE"] = "research"
    env["AUTORESEARCH_TRACK"] = track
    env["AUTORESEARCH_RUN_ID"] = run_id
    if memory_access:
        env["AUTORESEARCH_MEMORY_ACCESS"] = memory_access
    else:
        env.pop("AUTORESEARCH_MEMORY_ACCESS", None)
    # An orchestrator session exports these; a child must never inherit them or
    # it would bind as an orchestrator instead of a research agent.
    env.pop("AUTORESEARCH_ORCHESTRATION_ID", None)
    return env


def plan_spawn(
    orch: Orchestration,
    *,
    brief: Brief,
    backend_name: str,
    delegation_id: str | None = None,
    run_id: str = "<pending-bootstrap>",
    memory_access: str | None = None,
) -> SpawnPlan:
    """Compute the full spawn plan without touching the filesystem or processes."""

    backend = get_backend(backend_name)
    did = delegation_id or orch.next_delegation_id()
    prompt = compose_prompt(
        track=backend.track, run_id=run_id, cycle_budget=brief.cycle_budget
    )
    return SpawnPlan(
        orchestration_id=orch.orchestration_id,
        delegation_id=did,
        backend=backend,
        track=backend.track,
        run_id=run_id,
        cycle_budget=brief.cycle_budget,
        command=backend.render_command(prompt=prompt),
        env=child_environment(
            track=backend.track, run_id=run_id, memory_access=memory_access
        ),
        prompt=prompt,
        timeout_minutes=compute_timeout_minutes(brief.cycle_budget),
    )


def _check_budget(orch: Orchestration, brief: Brief) -> None:
    if brief.cycle_budget > orch.cycles_remaining:
        raise ValueError(
            f"Orchestration {orch.orchestration_id} has {orch.cycles_remaining} of "
            f"{orch.total_cycle_budget} cycles remaining; this brief asks for "
            f"{brief.cycle_budget}. Reduce the brief's cycle_budget or start a new "
            "orchestration."
        )


def _bootstrap_child_run(
    orch: Orchestration,
    *,
    backend: Backend,
    brief: Brief,
    delegation_id: str,
) -> Any:
    """Create the child run and stamp its orchestration back-pointer.

    Deliberately does **not** export the handoff: the brief block can only render
    once the brief file and the delegation record exist, so the caller exports
    afterwards via :func:`_export_handoff_with_brief`. Runs under the manifest
    lock (the caller holds it). Returns the child's ``ProjectConfig``.
    """

    from autoresearch.bootstrap import bootstrap_track

    config = load_config(
        track_id=backend.track,
        new_run=True,
        dataset=orch.dataset,
    )
    from dataclasses import replace as _replace

    config = _replace(
        config,
        model_provider=backend.model_provider,
        model_name=backend.model_name,
        model_harness=backend.tool,
        target_mode=orch.target_mode,
    )

    bootstrap_track(config, default_max_cycles=brief.cycle_budget)

    write_run_backpointer(
        config.artifacts_dir,
        orchestration_id=orch.orchestration_id,
        delegation_id=delegation_id,
    )
    return config


def _export_handoff_with_brief(config: Any, delegation_id: str) -> None:
    """Re-export the child's handoff and prove the brief block actually landed.

    The renderer degrades gracefully when a brief cannot be read — the right call
    for a mid-run refresh, but it would let a spawner-ordering bug ship a child
    that never sees its direction. We are about to spend a model's time and the
    user's money on this child, so verify here and fail loudly instead.
    """

    from autoresearch.controller.handoff import export_context_bundle

    outputs = export_context_bundle(config)
    handoff = outputs["latest_handoff_markdown"].read_text(encoding="utf-8")
    if "## Orchestration brief" not in handoff:
        raise RuntimeError(
            f"Child run {config.track_id}/{config.run_id} was bootstrapped but its "
            f"handoff carries no Orchestration brief block for {delegation_id}. "
            "Refusing to launch a sub-agent that cannot read its own direction."
        )


def _reject_unsupported_brief(brief: Brief) -> None:
    """Fail before anything is created, not after a child run exists."""

    if brief.seed_champion is not None:
        raise NotImplementedError(
            "brief.seed_champion is not implemented yet: it shares the playoff's "
            "cross-run replay path and lands with `respawn --seed-champion`. "
            "Omit seed_champion from the brief for now."
        )


def spawn(
    orchestration_id: str,
    *,
    brief_path: Path,
    backend_name: str,
    wait: bool = True,
    dry_run: bool = False,
    memory_access: str | None = None,
    respawn_of: str | None = None,
) -> dict[str, Any]:
    """Pre-bootstrap a child run and launch the sub-agent against it.

    With *dry_run* nothing is created or launched: the plan is returned for
    inspection. This is both the spawn-correctness review surface and the
    standard debugging tool when a backend's flags drift.
    """

    orch = load_orchestration(orchestration_id)
    brief = load_brief(brief_path)
    backend = get_backend(backend_name)
    _reject_unsupported_brief(brief)
    _check_budget(orch, brief)

    if dry_run:
        plan = plan_spawn(
            orch,
            brief=brief,
            backend_name=backend_name,
            memory_access=memory_access,
        )
        return {"status": "dry_run", "plan": plan}

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)  # re-read under the lock
        _check_budget(orch, brief)
        delegation_id = orch.next_delegation_id()
        child_config = _bootstrap_child_run(
            orch, backend=backend, brief=brief, delegation_id=delegation_id
        )
        run_id = child_config.run_id

        stored_brief = _store_brief(orch, delegation_id, brief, brief_path)
        plan = plan_spawn(
            orch,
            brief=brief,
            backend_name=backend_name,
            delegation_id=delegation_id,
            run_id=run_id,
            memory_access=memory_access,
        )
        prompt_file = prompt_path(orchestration_id, delegation_id)
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(plan.prompt, encoding="utf-8")

        delegation = Delegation(
            delegation_id=delegation_id,
            brief_path=str(stored_brief.relative_to(PROJECT_ROOT)),
            backend=backend.name,
            track=backend.track,
            run_id=run_id,
            cycle_budget=brief.cycle_budget,
            status="spawned",
            spawned_at=utc_stamp(),
            prompt_path=str(prompt_file.relative_to(PROJECT_ROOT)),
            log_path=str(log_path(orchestration_id, delegation_id).relative_to(PROJECT_ROOT)),
            command=plan.command,
            timeout_minutes=plan.timeout_minutes,
            respawn_of=respawn_of,
        )
        orch = add_delegation(orch, delegation)
        save_orchestration(orch)

        # Only now can the handoff render the brief: it reads the brief file and
        # the delegation record written immediately above.
        _export_handoff_with_brief(child_config, delegation_id)

        # Launch and record the PID while still holding the manifest lock. This
        # closes the detached-spawn race where two callers could overwrite each
        # other's read-modify-write after bootstrapping serially.
        process = _launch(plan, delegation)
        orch = update_delegation(
            orch, replace(delegation, pid=process.pid, status="running")
        )
        save_orchestration(orch)

    if not wait:
        return {
            "status": "running",
            "orchestration_id": orchestration_id,
            "delegation_id": delegation_id,
            "run_id": run_id,
            "pid": process.pid,
        }

    exit_code = process.wait()
    return _finalise_after_wait(orchestration_id, delegation_id, exit_code=exit_code)


def _store_brief(
    orch: Orchestration, delegation_id: str, brief: Brief, source: Path
) -> Path:
    """Copy the validated brief into the orchestration for audit."""

    from autoresearch.orchestration.manifest import brief_path as stored_brief_path

    target = stored_brief_path(orch.orchestration_id, delegation_id)
    payload = brief.to_dict()
    payload["_source_path"] = str(source)
    write_json(target, payload)
    return target


def _launch(plan: SpawnPlan, delegation: Delegation) -> subprocess.Popen:
    """Launch a detached wrapper that records the backend's eventual exit code."""

    log_file = PROJECT_ROOT / str(delegation.log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    status_file = exit_status_path(plan.orchestration_id, delegation.delegation_id)
    prompt_file = prompt_path(plan.orchestration_id, delegation.delegation_id)
    runner = PROJECT_ROOT / "scripts" / "run_orchestration_child.py"
    wrapper_command = [
        sys.executable,
        str(runner),
        "--status-path",
        str(status_file),
        "--log-path",
        str(log_file),
        "--prompt-path",
        str(prompt_file),
        "--prompt-via",
        plan.backend.prompt_via,
        "--tool",
        plan.backend.tool,
        "--",
        *plan.command,
    ]
    process = subprocess.Popen(  # noqa: S603 — argv comes from the curated registry
        wrapper_command,
        cwd=PROJECT_ROOT,
        env=plan.env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process


def _finalise_after_wait(
    orchestration_id: str, delegation_id: str, *, exit_code: int
) -> dict[str, Any]:
    """Record the child's exit, then generate its report from the registry."""

    from autoresearch.orchestration.report import collect_report

    exit_record: dict[str, Any] = {}
    try:
        payload = read_json(exit_status_path(orchestration_id, delegation_id))
        if isinstance(payload, dict):
            exit_record = payload
    except (OSError, TypeError, ValueError):
        pass
    recorded_exit_code = int(exit_record.get("exit_code", exit_code))
    clean_exit = bool(exit_record.get("clean_exit", recorded_exit_code == 0))
    tool_usage = exit_record.get("usage")
    if not isinstance(tool_usage, dict):
        tool_usage = {}

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        delegation = orch.delegation(delegation_id)
        # `finish-delegation` may already have marked it completed and stored the
        # summary; a non-zero exit still overrides that to `failed`.
        status = "completed" if clean_exit else "failed"
        delegation = replace(
            delegation,
            status=status,
            exit_code=recorded_exit_code,
            clean_exit=clean_exit,
            tool_usage=tool_usage,
            ended_at=str(exit_record.get("ended_at") or utc_stamp()),
        )
        orch = update_delegation(orch, delegation)
        save_orchestration(orch)

    orch = load_orchestration(orchestration_id)
    delegation = orch.delegation(delegation_id)
    report_file = collect_report(orch, delegation)

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        delegation = orch.delegation(delegation_id)
        orch = update_delegation(
            orch,
            replace(delegation, report_path=str(report_file.relative_to(PROJECT_ROOT))),
        )
        save_orchestration(orch)

    return {
        "status": delegation.status,
        "orchestration_id": orchestration_id,
        "delegation_id": delegation_id,
        "run_id": delegation.run_id,
        "exit_code": recorded_exit_code,
        "clean_exit": clean_exit,
        "report_path": str(report_file),
    }


def finish_delegation(run_dir: Path, *, summary: str) -> dict[str, Any]:
    """Record a sub-agent's own end-of-run summary (called by the child).

    Run-scoped: the child names only its own run, and the back-pointer in its run
    manifest identifies the orchestration. The summary is stored as testimony —
    it never feeds a computed number in the report.
    """

    from dataclasses import replace

    from autoresearch.orchestration.manifest import read_run_backpointer

    if not summary.strip():
        raise ValueError("finish-delegation requires a non-empty --summary")

    pointer = read_run_backpointer(run_dir)
    if pointer is None:
        raise ValueError(
            f"Run {run_dir} is not part of an orchestration (no orchestration_id in its "
            "run manifest). `finish-delegation` is only for spawned sub-agent runs."
        )
    orchestration_id, delegation_id = pointer

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        delegation = orch.delegation(delegation_id)
        orch = update_delegation(orch, replace(delegation, agent_summary=summary))
        save_orchestration(orch)

    return {
        "orchestration_id": orchestration_id,
        "delegation_id": delegation_id,
        "summary_chars": len(summary),
    }


def respawn(
    orchestration_id: str,
    *,
    delegation_id: str,
    brief_path: Path,
    backend_name: str | None = None,
    continue_run: bool = False,
    wait: bool = True,
    dry_run: bool = False,
    memory_access: str | None = None,
) -> dict[str, Any]:
    """Spawn a revised brief, optionally continuing the source child run."""

    from autoresearch.orchestration.monitor import refresh_orchestration
    from autoresearch.orchestration.monitor import process_is_alive

    orch = refresh_orchestration(orchestration_id)
    source = orch.delegation(delegation_id)
    selected_backend = backend_name or source.backend
    if not source.is_terminal or process_is_alive(source.pid):
        raise ValueError(
            f"Cannot respawn {delegation_id}: status is {source.status!r} and its "
            "process has not stopped. Wait for completion or kill it first."
        )
    if not continue_run:
        return spawn(
            orchestration_id,
            brief_path=brief_path,
            backend_name=selected_backend,
            wait=wait,
            dry_run=dry_run,
            memory_access=memory_access,
            respawn_of=delegation_id,
        )

    brief = load_brief(brief_path)
    _reject_unsupported_brief(brief)
    backend = get_backend(selected_backend)
    if backend.track != source.track:
        raise ValueError(
            f"Cannot continue {source.track}/{source.run_id} with backend "
            f"{selected_backend!r} on track {backend.track!r}. Choose a backend on "
            f"track {source.track!r}, or respawn without --continue-run."
        )
    _check_budget(orch, brief)

    if dry_run:
        plan = plan_spawn(
            orch,
            brief=brief,
            backend_name=selected_backend,
            run_id=source.run_id,
            memory_access=memory_access,
        )
        return {"status": "dry_run", "plan": plan}

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        source = orch.delegation(delegation_id)
        if not source.is_terminal:
            raise ValueError(
                f"Cannot continue {delegation_id}: status changed to {source.status!r}."
            )
        _check_budget(orch, brief)
        new_delegation_id = orch.next_delegation_id()
        stored_brief = _store_brief(orch, new_delegation_id, brief, brief_path)
        plan = plan_spawn(
            orch,
            brief=brief,
            backend_name=selected_backend,
            delegation_id=new_delegation_id,
            run_id=source.run_id,
            memory_access=memory_access,
        )
        prompt_file = prompt_path(orchestration_id, new_delegation_id)
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(plan.prompt, encoding="utf-8")

        from autoresearch.orchestration.report import _child_config, _cycles_used

        child_config = _child_config(source)
        delegation = Delegation(
            delegation_id=new_delegation_id,
            brief_path=str(stored_brief.relative_to(PROJECT_ROOT)),
            backend=backend.name,
            track=source.track,
            run_id=source.run_id,
            cycle_budget=brief.cycle_budget,
            status="spawned",
            spawned_at=utc_stamp(),
            prompt_path=str(prompt_file.relative_to(PROJECT_ROOT)),
            log_path=str(log_path(orchestration_id, new_delegation_id).relative_to(PROJECT_ROOT)),
            command=plan.command,
            timeout_minutes=plan.timeout_minutes,
            respawn_of=delegation_id,
            continue_run=True,
            cycles_at_start=_cycles_used(child_config.registry_path),
        )
        orch = add_delegation(orch, delegation)
        save_orchestration(orch)
        write_run_backpointer(
            child_config.artifacts_dir,
            orchestration_id=orchestration_id,
            delegation_id=new_delegation_id,
        )
        _export_handoff_with_brief(child_config, new_delegation_id)
        process = _launch(plan, delegation)
        orch = update_delegation(
            orch, replace(delegation, pid=process.pid, status="running")
        )
        save_orchestration(orch)

    if not wait:
        return {
            "status": "running",
            "orchestration_id": orchestration_id,
            "delegation_id": new_delegation_id,
            "run_id": source.run_id,
            "pid": process.pid,
            "respawn_of": delegation_id,
            "continue_run": True,
        }
    return _finalise_after_wait(
        orchestration_id, new_delegation_id, exit_code=process.wait()
    )
