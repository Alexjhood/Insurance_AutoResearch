"""Orchestration manifest — the on-disk record of a campaign and its delegations.

``artifacts/orchestrations/<orchestration-id>/orchestration.json`` is the single
source of truth for **which child runs belong to this orchestration**. The
run-scope guard reads it to decide what an orchestrator session may touch, so it
is written atomically and always under :func:`manifest_lock` when mutated.

The records are frozen dataclasses; every mutation returns a new object and is
persisted explicitly with :func:`save_orchestration`.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from autoresearch.config import PROJECT_ROOT
from autoresearch.utils.io import read_json, write_json


ORCHESTRATIONS_DIR = PROJECT_ROOT / "artifacts" / "orchestrations"

ORCHESTRATION_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
DELEGATION_ID_RE = re.compile(r"^d\d{2,}$")

ORCHESTRATION_STATUSES = frozenset(
    {"active", "consolidating", "completed", "abandoned"}
)
DELEGATION_STATUSES = frozenset(
    {"spawned", "running", "completed", "failed", "timed_out", "killed"}
)
#: Delegation statuses that mean the child process is no longer expected to run.
TERMINAL_DELEGATION_STATUSES = frozenset(
    {"completed", "failed", "timed_out", "killed"}
)

_LOCK_STALE_SECONDS = 900.0
_LOCK_POLL_SECONDS = 0.05
_LOCK_WAIT_SECONDS = 300.0


def utc_stamp() -> str:
    """Return the ``YYYY-MM-DDTHH:MM:SSZ`` timestamp used across the framework."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_orchestration_id() -> str:
    """Return a fresh ``YYYYMMDDTHHMMSSZ`` orchestration id (run-id convention)."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── records ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Delegation:
    """One spawned sub-agent run within an orchestration."""

    delegation_id: str
    brief_path: str
    backend: str
    track: str
    run_id: str
    cycle_budget: int
    status: str = "spawned"
    pid: int | None = None
    spawned_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    report_path: str | None = None
    prompt_path: str | None = None
    log_path: str | None = None
    command: tuple[str, ...] = ()
    timeout_minutes: float | None = None
    agent_summary: str | None = None
    respawn_of: str | None = None
    continue_run: bool = False
    cycles_at_start: int = 0

    def __post_init__(self) -> None:
        if not DELEGATION_ID_RE.fullmatch(self.delegation_id):
            raise ValueError(
                f"delegation_id must look like 'd01'; got {self.delegation_id!r}"
            )
        if self.status not in DELEGATION_STATUSES:
            raise ValueError(
                f"Unknown delegation status {self.status!r}; "
                f"expected one of {sorted(DELEGATION_STATUSES)}"
            )
        if self.cycle_budget <= 0:
            raise ValueError(
                f"delegation {self.delegation_id}: cycle_budget must be positive, "
                f"got {self.cycle_budget}"
            )
        if self.cycles_at_start < 0:
            raise ValueError("cycles_at_start must be non-negative")
        if self.continue_run and not self.respawn_of:
            raise ValueError("continue_run delegations must name respawn_of")

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_DELEGATION_STATUSES

    def run_dir(self) -> Path:
        """Absolute path of this delegation's child run folder."""

        return PROJECT_ROOT / "artifacts" / "tracks" / self.track / "runs" / self.run_id

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "delegation_id": self.delegation_id,
            "brief_path": self.brief_path,
            "backend": self.backend,
            "track": self.track,
            "run_id": self.run_id,
            "cycle_budget": self.cycle_budget,
            "status": self.status,
            "pid": self.pid,
            "spawned_at": self.spawned_at,
            "ended_at": self.ended_at,
            "exit_code": self.exit_code,
            "report_path": self.report_path,
            "prompt_path": self.prompt_path,
            "log_path": self.log_path,
            "command": list(self.command),
            "timeout_minutes": self.timeout_minutes,
            "agent_summary": self.agent_summary,
            "respawn_of": self.respawn_of,
            "continue_run": self.continue_run,
            "cycles_at_start": self.cycles_at_start,
        }
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Delegation":
        return cls(
            delegation_id=str(raw["delegation_id"]),
            brief_path=str(raw["brief_path"]),
            backend=str(raw["backend"]),
            track=str(raw["track"]),
            run_id=str(raw["run_id"]),
            cycle_budget=int(raw["cycle_budget"]),
            status=str(raw.get("status", "spawned")),
            pid=(int(raw["pid"]) if raw.get("pid") is not None else None),
            spawned_at=raw.get("spawned_at"),
            ended_at=raw.get("ended_at"),
            exit_code=(int(raw["exit_code"]) if raw.get("exit_code") is not None else None),
            report_path=raw.get("report_path"),
            prompt_path=raw.get("prompt_path"),
            log_path=raw.get("log_path"),
            command=tuple(str(c) for c in raw.get("command") or ()),
            timeout_minutes=(
                float(raw["timeout_minutes"]) if raw.get("timeout_minutes") is not None else None
            ),
            agent_summary=raw.get("agent_summary"),
            respawn_of=raw.get("respawn_of"),
            continue_run=bool(raw.get("continue_run", False)),
            cycles_at_start=int(raw.get("cycles_at_start") or 0),
        )


@dataclass(frozen=True)
class Consolidation:
    """Where the playoff's consolidation run lives (populated in Phase 4)."""

    track: str | None = None
    run_id: str | None = None
    playoff_report: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "run_id": self.run_id,
            "playoff_report": self.playoff_report,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Consolidation":
        raw = raw or {}
        return cls(
            track=raw.get("track"),
            run_id=raw.get("run_id"),
            playoff_report=raw.get("playoff_report"),
        )


@dataclass(frozen=True)
class Orchestration:
    """A campaign: a dataset, a total cycle budget, and its delegations."""

    orchestration_id: str
    dataset: str
    target_mode: str
    created_at: str
    total_cycle_budget: int
    status: str = "active"
    model_provider: str | None = None
    model_name: str | None = None
    delegations: tuple[Delegation, ...] = ()
    consolidation: Consolidation = Consolidation()

    def __post_init__(self) -> None:
        if not ORCHESTRATION_ID_RE.fullmatch(self.orchestration_id):
            raise ValueError(
                "orchestration_id must be a UTC timestamp in YYYYMMDDTHHMMSSZ form; "
                f"got {self.orchestration_id!r}"
            )
        if self.status not in ORCHESTRATION_STATUSES:
            raise ValueError(
                f"Unknown orchestration status {self.status!r}; "
                f"expected one of {sorted(ORCHESTRATION_STATUSES)}"
            )
        if self.total_cycle_budget <= 0:
            raise ValueError(
                f"total_cycle_budget must be positive, got {self.total_cycle_budget}"
            )
        seen = [d.delegation_id for d in self.delegations]
        if len(set(seen)) != len(seen):
            raise ValueError(f"duplicate delegation ids in orchestration: {seen}")

    # -- derived views --------------------------------------------------

    @property
    def cycles_committed(self) -> int:
        """Cycles handed to delegations that were actually spawned.

        A ``failed`` delegation still consumed its budget as far as the campaign
        is concerned — the orchestrator paid for the attempt. Only delegations
        that never spawned are absent from this list entirely.
        """

        return sum(d.cycle_budget for d in self.delegations)

    @property
    def cycles_remaining(self) -> int:
        return self.total_cycle_budget - self.cycles_committed

    def delegation(self, delegation_id: str) -> Delegation:
        for d in self.delegations:
            if d.delegation_id == delegation_id:
                return d
        known = ", ".join(d.delegation_id for d in self.delegations) or "(none)"
        raise KeyError(
            f"Orchestration {self.orchestration_id} has no delegation "
            f"{delegation_id!r}; known: {known}"
        )

    def next_delegation_id(self) -> str:
        return f"d{len(self.delegations) + 1:02d}"

    def child_run_dirs(self) -> tuple[Path, ...]:
        """Every child run folder this orchestration owns (guard reads this)."""

        return tuple(d.run_dir() for d in self.delegations)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "orchestration_id": self.orchestration_id,
            "dataset": self.dataset,
            "target_mode": self.target_mode,
            "created_at": self.created_at,
            "status": self.status,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "total_cycle_budget": self.total_cycle_budget,
            "cycles_committed": self.cycles_committed,
            "delegations": [d.to_dict() for d in self.delegations],
            "consolidation": self.consolidation.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Orchestration":
        return cls(
            orchestration_id=str(raw["orchestration_id"]),
            dataset=str(raw["dataset"]),
            target_mode=str(raw["target_mode"]),
            created_at=str(raw["created_at"]),
            total_cycle_budget=int(raw["total_cycle_budget"]),
            status=str(raw.get("status", "active")),
            model_provider=raw.get("model_provider"),
            model_name=raw.get("model_name"),
            delegations=tuple(Delegation.from_dict(d) for d in raw.get("delegations") or ()),
            consolidation=Consolidation.from_dict(raw.get("consolidation")),
        )


# ── paths ────────────────────────────────────────────────────────────────────


def orchestration_dir(orchestration_id: str) -> Path:
    return ORCHESTRATIONS_DIR / orchestration_id


def manifest_path(orchestration_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "orchestration.json"


def brief_path(orchestration_id: str, delegation_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "briefs" / f"{delegation_id}.json"


def prompt_path(orchestration_id: str, delegation_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "prompts" / f"{delegation_id}.md"


def log_path(orchestration_id: str, delegation_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "logs" / f"{delegation_id}.stdout.log"


def exit_status_path(orchestration_id: str, delegation_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "logs" / f"{delegation_id}.exit.json"


def report_path(orchestration_id: str, delegation_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "reports" / f"{delegation_id}.json"


def playoff_dir(orchestration_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "playoff"


def log_markdown_path(orchestration_id: str) -> Path:
    return orchestration_dir(orchestration_id) / "ORCHESTRATION_LOG.md"


def _relative(path: Path) -> str:
    """Store paths relative to the orchestration dir where possible (portable)."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ── persistence ──────────────────────────────────────────────────────────────


def create_orchestration(
    *,
    dataset: str,
    target_mode: str,
    total_cycle_budget: int,
    model_provider: str | None = None,
    model_name: str | None = None,
    orchestration_id: str | None = None,
) -> Orchestration:
    """Create the campaign folder skeleton and write the initial manifest."""

    oid = orchestration_id or new_orchestration_id()
    if manifest_path(oid).exists():
        raise FileExistsError(
            f"Orchestration {oid} already exists at {orchestration_dir(oid)}"
        )
    base = orchestration_dir(oid)
    for sub in ("briefs", "prompts", "logs", "reports", "playoff"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    orch = Orchestration(
        orchestration_id=oid,
        dataset=dataset,
        target_mode=target_mode,
        created_at=utc_stamp(),
        total_cycle_budget=int(total_cycle_budget),
        model_provider=model_provider,
        model_name=model_name,
    )
    save_orchestration(orch)
    return orch


def save_orchestration(orch: Orchestration) -> Path:
    path = manifest_path(orch.orchestration_id)
    write_json(path, orch.to_dict())
    return path


def load_orchestration(orchestration_id: str) -> Orchestration:
    path = manifest_path(orchestration_id)
    if not path.exists():
        available = ", ".join(list_orchestration_ids()) or "(none)"
        raise FileNotFoundError(
            f"Unknown orchestration {orchestration_id!r}: no manifest at {path}. "
            f"Existing orchestrations: {available}"
        )
    return Orchestration.from_dict(read_json(path))


def list_orchestration_ids() -> list[str]:
    """Return every orchestration id on disk, oldest first."""

    if not ORCHESTRATIONS_DIR.exists():
        return []
    return sorted(
        p.name
        for p in ORCHESTRATIONS_DIR.iterdir()
        if p.is_dir() and (p / "orchestration.json").exists()
    )


def resolve_orchestration_id(orchestration_id: str | None) -> str:
    """Resolve an explicit id, or the latest one when omitted (mirrors run-id)."""

    if orchestration_id:
        return orchestration_id
    known = list_orchestration_ids()
    if not known:
        raise FileNotFoundError(
            "No orchestrations exist yet. Create one with `autoresearch orchestrate new`."
        )
    return known[-1]


def add_delegation(orch: Orchestration, delegation: Delegation) -> Orchestration:
    """Return a copy of *orch* with *delegation* appended (does not save)."""

    return replace(orch, delegations=(*orch.delegations, delegation))


def update_delegation(orch: Orchestration, updated: Delegation) -> Orchestration:
    """Return a copy of *orch* with the matching delegation replaced (does not save)."""

    orch.delegation(updated.delegation_id)  # fail loudly on unknown id
    return replace(
        orch,
        delegations=tuple(
            updated if d.delegation_id == updated.delegation_id else d
            for d in orch.delegations
        ),
    )


# ── locking ──────────────────────────────────────────────────────────────────


@contextmanager
def manifest_lock(
    orchestration_id: str, *, timeout: float = _LOCK_WAIT_SECONDS
) -> Iterator[None]:
    """Exclusive lock around manifest read-modify-write and child bootstrap.

    Bootstrapping a child run mutates ``latest_run.json`` for its track, so two
    concurrent ``spawn`` calls must serialise through here. The lock is a
    ``O_CREAT|O_EXCL`` file holding the owner pid; a lock older than
    ``_LOCK_STALE_SECONDS`` is treated as abandoned and broken, so a crashed
    spawner cannot wedge the campaign forever.
    """

    lock_file = orchestration_dir(orchestration_id) / "orchestration.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if _lock_is_stale(lock_file):
                _break_stale_lock(lock_file)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout:.0f}s waiting for the orchestration "
                    f"lock at {lock_file}. Another `orchestrate` command is running; "
                    "delete the lock file if you are sure it is stale."
                )
            time.sleep(_LOCK_POLL_SECONDS)
    try:
        os.write(fd, f'{{"pid": {os.getpid()}, "acquired_at": "{utc_stamp()}"}}\n'.encode())
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_file.unlink(missing_ok=True)


def _lock_is_stale(lock_file: Path) -> bool:
    try:
        age = time.time() - lock_file.stat().st_mtime
    except FileNotFoundError:
        return False
    return age > _LOCK_STALE_SECONDS


def _break_stale_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink()
    except FileNotFoundError:
        pass


# ── child run back-pointer ───────────────────────────────────────────────────


def write_run_backpointer(
    run_dir: Path, *, orchestration_id: str, delegation_id: str
) -> None:
    """Stamp ``orchestration_id``/``delegation_id`` into the child's run manifest.

    The handoff renderer keys the "Orchestration brief" block off this, so a
    single-agent run — which never gets these fields — renders exactly as before.
    """

    path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest["orchestration_id"] = orchestration_id
    manifest["delegation_id"] = delegation_id
    write_json(path, manifest)


def read_run_backpointer(run_dir: Path) -> tuple[str, str] | None:
    """Return ``(orchestration_id, delegation_id)`` for a child run, else ``None``."""

    path = run_dir / "run_manifest.json"
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    oid = manifest.get("orchestration_id")
    did = manifest.get("delegation_id")
    if oid and did:
        return str(oid), str(did)
    return None


def find_orchestration_for_run(run_dir: Path) -> Orchestration | None:
    """Load the orchestration owning *run_dir*, or ``None`` for a single-agent run."""

    pointer = read_run_backpointer(run_dir)
    if pointer is None:
        return None
    try:
        return load_orchestration(pointer[0])
    except FileNotFoundError:
        return None
