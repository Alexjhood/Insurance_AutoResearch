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
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from autoresearch.config import PROJECT_ROOT, load_config
from autoresearch.orchestration.backends import Backend, get_backend
from autoresearch.orchestration.brief import Brief, SeedChampion, load_brief
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    add_delegation,
    baseline_registry_path,
    delegation_run_dir,
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
        "`run-session-cycles` runs a full experiment and can take many minutes. If "
        "your harness enforces a per-command timeout, raise it for that command or "
        "use `run-session-cycles 1 --background` and poll `session-status` until the "
        "state is `awaiting_decision` — never let the harness kill a running cycle.\n\n"
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
    # Foundation enablement is a per-run decision driven by the child's own run
    # manifest (set at bootstrap from the brief). Strip any inherited env flag so
    # a non-opted child cannot silently self-register foundation estimators just
    # because the orchestrator's process set AUTORESEARCH_FOUNDATION_MODELS when
    # it bootstrapped an earlier foundation child. (TABPFN_TOKEN is deliberately
    # NOT stripped — an opted-in child needs it to reach the api backend.)
    env.pop("AUTORESEARCH_FOUNDATION_MODELS", None)
    # Never forward holdout access or the pytest-gate skip: a sub-agent must not
    # be able to read the milestone vault or dodge hard constraint #4 just
    # because the operator's shell had these set.
    env.pop("AUTORESEARCH_MILESTONE_TOKEN", None)
    env.pop("AUTORESEARCH_SKIP_PYTEST_GATE", None)
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

    run_id = _new_child_run_id(backend.track)
    canonical_dir = delegation_run_dir(orch.orchestration_id, delegation_id)
    legacy_dir = PROJECT_ROOT / "artifacts" / "tracks" / backend.track / "runs" / run_id
    config = load_config(track_id=backend.track, run_id=run_id, dataset=orch.dataset)
    managed_layout = config.artifacts_dir == legacy_dir
    if managed_layout:
        canonical_dir.parent.mkdir(parents=True, exist_ok=True)
        if canonical_dir.exists() or legacy_dir.exists() or legacy_dir.is_symlink():
            raise FileExistsError(
                f"Refusing to overwrite orchestration child path: {canonical_dir} / {legacy_dir}"
            )
        canonical_dir.mkdir()
        legacy_dir.parent.mkdir(parents=True, exist_ok=True)
        legacy_dir.symlink_to(canonical_dir, target_is_directory=True)
    from dataclasses import replace as _replace

    config = _replace(
        config,
        model_provider=backend.model_provider,
        model_name=backend.model_name,
        model_harness=backend.tool,
        target_mode=orch.target_mode,
    )

    baseline_registry = _campaign_baseline_registry(orch)
    try:
        bootstrap_track(
            config,
            run_baselines=baseline_registry is None,
            baseline_registry_source=baseline_registry,
            default_max_cycles=brief.cycle_budget,
            enable_foundation_models=brief.foundation_models,
        )
        if baseline_registry is None and managed_layout:
            template = baseline_registry_path(orch.orchestration_id)
            template.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config.registry_path, template)
    except Exception:
        # Do not leave a path that looks like a valid run after failed bootstrap.
        if managed_layout:
            legacy_dir.unlink(missing_ok=True)
            shutil.rmtree(canonical_dir, ignore_errors=True)
        raise

    write_run_backpointer(
        config.artifacts_dir,
        orchestration_id=orch.orchestration_id,
        delegation_id=delegation_id,
        target_mode=orch.target_mode,
    )
    if brief.seed_champion is not None:
        from autoresearch.orchestration.playoff import ReplaySource, seed_champion_from_source

        seed_champion_from_source(
            config,
            ReplaySource(
                track=brief.seed_champion.track,
                run_id=brief.seed_champion.run_id,
                experiment_id=brief.seed_champion.experiment_id,
            ),
            label=f"delegation_seed_{delegation_id}",
        )
    return config


def _new_child_run_id(track: str) -> str:
    """Allocate a timestamp-shaped compatibility id without moving latest_run."""

    from datetime import datetime, timedelta, timezone

    runs_dir = PROJECT_ROOT / "artifacts" / "tracks" / track / "runs"
    candidate = datetime.now(timezone.utc).replace(microsecond=0)
    while (runs_dir / candidate.strftime("%Y%m%dT%H%M%SZ")).exists():
        candidate += timedelta(seconds=1)
    return candidate.strftime("%Y%m%dT%H%M%SZ")


def _campaign_baseline_registry(orch: Orchestration) -> Path | None:
    """Return the campaign's frozen baseline-only registry, if initialized."""

    delegations = getattr(orch, "delegations", ())
    if not delegations:
        return None
    source = baseline_registry_path(orch.orchestration_id)
    if not source.is_file():
        raise FileNotFoundError(
            "The campaign's frozen baseline registry is missing: "
            f"{source}"
        )
    return source


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


def spawn(
    orchestration_id: str,
    *,
    brief_path: Path,
    backend_name: str,
    wait: bool = True,
    dry_run: bool = False,
    memory_access: str | None = None,
    respawn_of: str | None = None,
    seed_champion_override: SeedChampion | None = None,
) -> dict[str, Any]:
    """Pre-bootstrap a child run and launch the sub-agent against it.

    With *dry_run* nothing is created or launched: the plan is returned for
    inspection. This is both the spawn-correctness review surface and the
    standard debugging tool when a backend's flags drift.
    """

    orch = load_orchestration(orchestration_id)
    brief = load_brief(brief_path)
    if respawn_of is not None:
        source = orch.delegation(respawn_of)
        if not source.is_terminal:
            raise ValueError(
                f"Cannot mark spawn as retry of {respawn_of}: source status is "
                f"{source.status!r}, not terminal."
            )
    if seed_champion_override is not None:
        if brief.seed_champion is not None:
            raise ValueError(
                "Seed champion was supplied both in the brief and via --seed-champion"
            )
        brief = replace(brief, seed_champion=seed_champion_override)
    backend = get_backend(backend_name)
    _check_budget(orch, brief)

    resolved_backend: dict[str, str | None] = {}
    if not dry_run:
        from autoresearch.orchestration.backends import (
            preflight_backend,
            preflight_foundation_models,
        )

        resolved_backend = preflight_backend(backend) or {}
        if resolved_backend.get("executable"):
            backend = replace(
                backend,
                command=(str(resolved_backend["executable"]), *backend.command[1:]),
            )
        if brief.foundation_models:
            preflight_foundation_models()

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
            run_path=str(child_config.artifacts_dir.resolve().relative_to(PROJECT_ROOT)),
            status="spawned",
            spawned_at=utc_stamp(),
            prompt_path=str(prompt_file.relative_to(PROJECT_ROOT)),
            log_path=str(log_path(orchestration_id, delegation_id).relative_to(PROJECT_ROOT)),
            command=plan.command,
            timeout_minutes=plan.timeout_minutes,
            respawn_of=respawn_of,
            resolved_executable=resolved_backend.get("executable"),
            resolved_version=resolved_backend.get("version"),
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

    from autoresearch.orchestration.report import should_refund_budget

    orch = load_orchestration(orchestration_id)
    delegation = orch.delegation(delegation_id)
    report_file = collect_report(orch, delegation)
    report_payload = read_json(report_file)
    refunded = should_refund_budget(report_payload)
    # A finished delegation returns its committed-but-unused cycles to the pool
    # explicitly, instead of silently down-scoping whatever spawns next. A full
    # refund supersedes this; a takeover keeps the budget with the orchestrator.
    forfeited = 0
    if not refunded and not delegation.taken_over and delegation.status == "completed":
        # Only for clean completions: a failed/timed-out delegation still paid
        # its full commitment (the attempt), matching the refund policy above.
        completed_cycles = int((report_payload.get("cycles") or {}).get("completed") or 0)
        forfeited = max(0, delegation.cycle_budget - completed_cycles)

    with manifest_lock(orchestration_id):
        orch = load_orchestration(orchestration_id)
        delegation = orch.delegation(delegation_id)
        orch = update_delegation(
            orch,
            replace(
                delegation,
                report_path=str(report_file.relative_to(PROJECT_ROOT)),
                budget_refunded=refunded,
                cycles_forfeited=forfeited,
            ),
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

    # A headless agent can issue tool calls concurrently. Refuse to let it end
    # the delegation while a framework command is still evaluating or while a
    # proposal still needs a decision/repair. Otherwise the CLI process exit can
    # terminate that work and leave screening-only artifacts behind.
    from autoresearch.config import load_config
    from autoresearch.controller.session import inflight_cycle_status, latest_session
    from autoresearch.controller.workflow import INFLIGHT_PROPOSAL_STATUSES
    from autoresearch.experiment_registry.registry import list_proposals

    child_config = load_config(
        track_id=str(read_json(run_dir / "run_manifest.json").get("track_id") or "codex"),
        run_id=run_dir.name,
    )
    session = latest_session(child_config)
    active_states = {"running", "ingesting", "evaluating", "comparing"}
    decision_states = {"awaiting_decision", "awaiting_reflection", "waiting_for_repair"}
    if session is not None and session.get("state") in active_states | decision_states:
        session_state = str(session.get("state"))
        inflight = inflight_cycle_status(child_config, session.get("session_id"))
        if session_state in active_states and (inflight is None or not inflight["alive"]):
            # Wedged, not busy: the process that owned this state died (e.g. a
            # harness command timeout killed `run-session-cycles`). Blocking
            # unconditionally would deadlock the delegation — point at recovery.
            raise ValueError(
                f"Cannot finish delegation: the session is stuck in {session_state!r} "
                "but no process is working on it — the command that owned this cycle "
                "was killed. Run `run-session-cycles 1` to recover the orphaned "
                "cycle, complete any resulting decision, then finish."
            )
        if session_state in active_states:
            raise ValueError(
                "Cannot finish delegation while a process "
                f"(pid {inflight['pid']}) is still evaluating the current cycle. "
                "Wait for it to finish and complete the resulting decision first."
            )
        raise ValueError(
            "Cannot finish delegation while the supervised session is "
            f"{session_state!r}. Complete the required decision, reflection, "
            "or repair first."
        )
    nonterminal_proposals = [
        f"{item.get('proposal_id')} ({item.get('status')})"
        for item in list_proposals(child_config.registry_path)
        if item.get("status")
        in {"proposed", "validated", "queued", "needs_repair", "awaiting_decision"}
        | set(INFLIGHT_PROPOSAL_STATUSES)
    ]
    if nonterminal_proposals:
        raise ValueError(
            "Cannot finish delegation with nonterminal proposals: "
            + ", ".join(nonterminal_proposals[:5])
            + ". Run `run-session-cycles 1` — it recovers proposals orphaned by a "
            "killed cycle — and complete the outcome first."
        )

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
    seed_champion: str | None = None,
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
    brief: Brief | None = None
    seed_override: SeedChampion | None = None
    if seed_champion is not None:
        brief = load_brief(brief_path)
        if continue_run:
            raise ValueError("--seed-champion cannot be combined with --continue-run")
        if not seed_champion.startswith("from:"):
            raise ValueError("--seed-champion must use the form from:<delegation-id>")
        seed_delegation_id = seed_champion.removeprefix("from:").strip()
        if not seed_delegation_id:
            raise ValueError("--seed-champion must name a delegation after 'from:'")
        from autoresearch.orchestration.playoff import replay_source_for_delegation

        replay_source = replay_source_for_delegation(orch, seed_delegation_id)
        seed_override = SeedChampion(
            from_run=replay_source.run_ref,
            experiment_id=replay_source.experiment_id,
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
            seed_champion_override=seed_override,
        )

    brief = brief or load_brief(brief_path)
    if brief.seed_champion is not None:
        raise ValueError(
            "A brief with seed_champion cannot be used with --continue-run; "
            "the continued run already retains its champion"
        )
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

    from autoresearch.orchestration.backends import (
        preflight_backend,
        preflight_foundation_models,
    )

    resolved_backend = preflight_backend(backend) or {}
    if resolved_backend.get("executable"):
        backend = replace(
            backend,
            command=(str(resolved_backend["executable"]), *backend.command[1:]),
        )
    if brief.foundation_models:
        preflight_foundation_models()

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

        from autoresearch.bootstrap import _pin_default_max_cycles
        from autoresearch.orchestration.report import _child_config, _cycles_used

        child_config = _child_config(source)
        # The run manifest still pins the source delegation's cycle budget; the
        # continuation's new session inherits that pin, so re-pin it to the new
        # brief's budget or a larger continuation would stall at the old cap.
        _pin_default_max_cycles(child_config, brief.cycle_budget)
        delegation = Delegation(
            delegation_id=new_delegation_id,
            brief_path=str(stored_brief.relative_to(PROJECT_ROOT)),
            backend=backend.name,
            track=source.track,
            run_id=source.run_id,
            cycle_budget=brief.cycle_budget,
            run_path=source.run_path,
            status="spawned",
            spawned_at=utc_stamp(),
            prompt_path=str(prompt_file.relative_to(PROJECT_ROOT)),
            log_path=str(log_path(orchestration_id, new_delegation_id).relative_to(PROJECT_ROOT)),
            command=plan.command,
            timeout_minutes=plan.timeout_minutes,
            respawn_of=delegation_id,
            continue_run=True,
            cycles_at_start=_cycles_used(child_config.registry_path),
            resolved_executable=resolved_backend.get("executable"),
            resolved_version=resolved_backend.get("version"),
        )
        orch = add_delegation(orch, delegation)
        save_orchestration(orch)
        write_run_backpointer(
            child_config.artifacts_dir,
            orchestration_id=orchestration_id,
            delegation_id=new_delegation_id,
            target_mode=orch.target_mode,
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
