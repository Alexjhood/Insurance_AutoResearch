"""
OpenCode adapter — `opencode run --format json` (verified against opencode 1.16.0).

Real event format (NDJSON on stdout):
  {"type":"step_start","sessionID":"ses_...","part":{"type":"step-start",...}}
  {"type":"...","sessionID":"ses_...","part":{"type":"text","text":"..."}}
  {"type":"...","part":{"type":"tool",...}}
  {"type":"step_finish",...}

Session id = sessionID. Resume: `opencode run -s <sessionID> [message]`.
Steering is between-turn (a fresh `run -s` continues the session).

Note: some models emit very sparse JSON; the adapter tolerates that and relies
on process exit for turn completion.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator

from console.backend.config import OPENCODE_BIN
from .base import AgentEvent, EventType, SessionHandle, SteerResult


class OpenCodeAdapter:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._cwd: Path | None = None
        self._env: dict | None = None
        self._session_id: str | None = None
        self._model: str | None = None
        self._variant: str = ""
        self._pending_steer: list[str] = []
        self._event_queue: queue.Queue[AgentEvent | None] = queue.Queue()

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def launch(self, *, cwd: Path, env: dict, seed_prompt: str) -> SessionHandle:
        # Model as `provider/model` string; AGENT_MODEL is the unified key.
        self._model = env.get("AGENT_MODEL") or env.get("OPENCODE_MODEL")
        self._variant = env.get("AGENT_EFFORT", "").strip()
        return self._spawn(cwd=cwd, env=env, message=seed_prompt, session_id=None)

    def resume(self, *, cwd: Path, env: dict, session_id: str, extra_prompt: str = "") -> SessionHandle:
        self._model = env.get("AGENT_MODEL") or env.get("OPENCODE_MODEL") or self._model
        self._variant = env.get("AGENT_EFFORT", "").strip() or getattr(self, "_variant", "")
        return self._spawn(cwd=cwd, env=env, message=extra_prompt or "continue",
                           session_id=session_id)

    def stream(self) -> Iterator[AgentEvent]:
        while True:
            event = self._event_queue.get()
            if event is None:
                return
            yield event

    def steer(self, message: str, *, interrupt: bool = True) -> SteerResult:
        # opencode run is one-shot per process; steering continues the session.
        self._pending_steer.append(message)
        return SteerResult(
            mode="queued", ok=True,
            detail="OpenCode is between-turn — queued; continues session via -s",
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
        pending = list(self._pending_steer)
        self._pending_steer.clear()
        msg = "\n\n".join([*pending, message]) if pending else message
        assert self._cwd is not None and self._session_id is not None
        return self.resume(cwd=self._cwd, env=self._env or {},
                           session_id=self._session_id, extra_prompt=msg)

    # ── Internal ──────────────────────────────────────────────────────────

    def _spawn(self, *, cwd: Path, env: dict, message: str, session_id: str | None) -> SessionHandle:
        bin_path = env.get("OPENCODE_BIN") or OPENCODE_BIN
        if not bin_path:
            raise RuntimeError(
                "opencode binary not found. Install: npm i -g opencode-ai "
                "(then `opencode auth login`). Set OPENCODE_BIN to override."
            )
        self._cwd = cwd
        self._env = {**os.environ, **env}

        cmd = [bin_path, "run", "--format", "json"]
        if self._model:
            cmd += ["-m", self._model]
        if self._variant:
            cmd += ["--variant", self._variant]
        if session_id:
            cmd += ["-s", session_id]
        cmd.append(message)

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=self._env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_stdout, daemon=True,
                         name=f"opencode-drain-{self._proc.pid}").start()
        return SessionHandle(session_id=self._session_id or f"opencode-{self._proc.pid}",
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

            # sessionID appears on every event
            sid = obj.get("sessionID")
            if sid and not self._session_id:
                self._session_id = sid
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload={
                    "msg": f"OpenCode session: {sid}", "session_id": sid,
                }))

            part = obj.get("part", {}) if isinstance(obj.get("part"), dict) else {}
            ptype = part.get("type", "")

            if ptype == "text":
                text = part.get("text", "")
                if text:
                    self._event_queue.put(AgentEvent(type=EventType.TOKEN, payload={"text": text}))
            elif ptype in ("tool", "tool-invocation", "tool_use") or "tool" in str(ptype):
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                status = state.get("status") or part.get("status")
                completed = str(status or "").lower() in {
                    "completed", "success", "succeeded", "done",
                    "failed", "error", "cancelled", "canceled",
                }
                tool_input = state.get("input") or part.get("input")
                tool_output = state.get("output") or part.get("output")
                self._event_queue.put(AgentEvent(
                    type=EventType.TOOL_RESULT if completed else EventType.TOOL_USE,
                    payload={
                    "provider_call_id": (
                        part.get("callID") or part.get("callId")
                        or part.get("id") or state.get("id")
                    ),
                    "name": part.get("tool") or part.get("name") or "tool",
                    "input": _preview(tool_input),
                    "input_bytes": _byte_size(tool_input),
                    "output": _preview(tool_output),
                    "output_bytes": _byte_size(tool_output),
                    "status": status,
                    "duration_ms": state.get("duration") or part.get("duration"),
                    "error": state.get("error") or part.get("error"),
                }))
            elif obj.get("type") in ("step_finish", "step-finish"):
                usage = obj.get("usage") or part.get("usage") or obj.get("tokens") or part.get("tokens")
                self._event_queue.put(AgentEvent(type=EventType.TURN_END, payload={
                    "session_id": self._session_id,
                    "usage": usage,
                    "cost": obj.get("cost") or part.get("cost"),
                    "duration_ms": obj.get("duration_ms") or part.get("duration_ms"),
                    "is_error": obj.get("is_error") or part.get("is_error"),
                }))
            # step_start and other lifecycle events are ignored for display

        self._proc.wait()
        self._event_queue.put(AgentEvent(
            type=EventType.AGENT_EXIT,
            payload={"returncode": self._proc.returncode, "session_id": self._session_id},
        ))
        self._event_queue.put(None)


def _byte_size(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(json.dumps(value, ensure_ascii=True, default=str).encode("utf-8"))


def _preview(value: object, limit: int = 4000) -> object:
    if _byte_size(value) <= limit:
        return value
    if isinstance(value, str):
        return value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    encoded = json.dumps(value, ensure_ascii=True, default=str)
    return encoded[:limit]
