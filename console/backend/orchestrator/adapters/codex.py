"""
Codex adapter — `codex exec --json` (verified against codex-cli 0.137.0).

Real event format (NDJSON on stdout):
  {"type":"thread.started","thread_id":"<uuid>"}
  {"type":"turn.started"}
  {"type":"item.started","item":{"id","type","status",...}}
  {"type":"item.completed","item":{"id","type":"agent_message","text":"..."}}
  {"type":"item.completed","item":{"id","type":"command_execution",
        "command":"...","aggregated_output":"...","exit_code":0,"status":"completed"}}
  {"type":"turn.completed","usage":{...}}

Session id = thread_id. Resume: `codex exec resume <thread_id> <prompt>`.
Steering is between-turn only (Codex exec has no live stdin injection).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator

from console.backend.config import CODEX_BIN
from .base import AgentEvent, EventType, SessionHandle, SteerResult

# `--sandbox workspace-write` confines writes to the working tree (matches the
# repo's .codex/config.toml). We do NOT use --dangerously-bypass-* which would
# disable the sandbox entirely.
_BASE_FLAGS = [
    "exec",
    "--json",
    "--sandbox", "workspace-write",
    "--skip-git-repo-check",
]


class CodexAdapter:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._cwd: Path | None = None
        self._env: dict | None = None
        self._thread_id: str | None = None
        self._pending_steer: list[str] = []
        self._event_queue: queue.Queue[AgentEvent | None] = queue.Queue()

    @property
    def session_id(self) -> str | None:
        return self._thread_id

    def launch(self, *, cwd: Path, env: dict, seed_prompt: str) -> SessionHandle:
        return self._spawn(cwd=cwd, env=env, args=[seed_prompt])

    def resume(self, *, cwd: Path, env: dict, session_id: str, extra_prompt: str = "") -> SessionHandle:
        args = ["resume", session_id]
        if extra_prompt:
            args.append(extra_prompt)
        # resume is a subcommand of exec; build full args minus the leading "exec"
        return self._spawn(cwd=cwd, env=env, args=args, resume=True)

    def stream(self) -> Iterator[AgentEvent]:
        while True:
            event = self._event_queue.get()
            if event is None:
                return
            yield event

    def steer(self, message: str, *, interrupt: bool = True) -> SteerResult:
        # Codex exec has no live stdin injection — queue for the next turn.
        self._pending_steer.append(message)
        return SteerResult(
            mode="queued", ok=True,
            detail="Codex is between-turn only — queued for next cycle",
        )

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._event_queue.put(None)

    def continue_with_steer(self, message: str) -> SessionHandle:
        """Launch a follow-up `codex exec resume` incorporating queued steers."""
        pending = list(self._pending_steer)
        self._pending_steer.clear()
        prompt = "\n\n".join([*pending, message]) if pending else message
        assert self._cwd is not None and self._thread_id is not None
        return self.resume(cwd=self._cwd, env=self._env or {},
                           session_id=self._thread_id, extra_prompt=prompt)

    # ── Internal ──────────────────────────────────────────────────────────

    def _spawn(self, *, cwd: Path, env: dict, args: list[str], resume: bool = False) -> SessionHandle:
        bin_path = env.get("CODEX_BIN") or CODEX_BIN
        if not bin_path:
            raise RuntimeError(
                "codex binary not found. Install: npm i -g @openai/codex (then `codex login`). "
                "Set CODEX_BIN to override."
            )
        self._cwd = cwd
        self._env = {**os.environ, **env}

        if resume:
            # codex exec resume <id> [prompt]
            cmd = [bin_path, "exec", "--json", "--sandbox", "workspace-write",
                   "--skip-git-repo-check", *args]
        else:
            cmd = [bin_path, *_BASE_FLAGS, *args]

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=self._env,
            stdin=subprocess.DEVNULL,   # avoid "Reading additional input from stdin" hang
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stdout, daemon=True,
                         name=f"codex-drain-{self._proc.pid}").start()
        return SessionHandle(session_id=self._thread_id or f"codex-{self._proc.pid}",
                             pid=self._proc.pid)

    def _drain_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload={"raw": line}))
                continue

            etype = obj.get("type", "")

            if etype == "thread.started":
                self._thread_id = obj.get("thread_id")
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload={
                    "msg": f"Codex thread started: {self._thread_id}",
                    "session_id": self._thread_id,
                }))

            elif etype == "item.completed":
                item = obj.get("item", {})
                itype = item.get("type")
                if itype == "agent_message":
                    text = item.get("text", "")
                    if text:
                        self._event_queue.put(AgentEvent(
                            type=EventType.TOKEN, payload={"text": text + "\n"}))
                elif itype == "command_execution":
                    self._event_queue.put(AgentEvent(type=EventType.TOOL_USE, payload={
                        "name": "shell",
                        "input": {"command": item.get("command")},
                        "exit_code": item.get("exit_code"),
                    }))
                elif itype in ("file_change", "patch", "mcp_tool_call"):
                    self._event_queue.put(AgentEvent(type=EventType.TOOL_USE, payload={
                        "name": itype, "input": item,
                    }))
                # reasoning / other item types are ignored for display

            elif etype == "turn.completed":
                self._event_queue.put(AgentEvent(type=EventType.TURN_END, payload={
                    "session_id": self._thread_id, "usage": obj.get("usage"),
                }))

            elif etype == "turn.started" or etype == "item.started":
                pass  # lifecycle noise

            else:
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload=obj))

        self._proc.wait()
        self._event_queue.put(AgentEvent(
            type=EventType.AGENT_EXIT,
            payload={"returncode": self._proc.returncode, "session_id": self._thread_id},
        ))
        self._event_queue.put(None)
