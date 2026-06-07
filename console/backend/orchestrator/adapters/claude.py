"""Claude Code adapter — headless via --print --output-format stream-json."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator

from console.backend.config import CLAUDE_BIN
from .base import AgentAdapter, AgentEvent, EventType, SessionHandle, SteerResult

# Required flags for headless research runs.
# --verbose                    emits system/init event (session_id) + all events
# --include-partial-messages   streaming token deltas during generation
# --dangerously-skip-permissions  bypass interactive tool-use prompts
_BASE_FLAGS = [
    "--print",
    "--output-format", "stream-json",
    "--verbose",
    "--include-partial-messages",
    "--dangerously-skip-permissions",
]


class ClaudeAdapter:
    """
    Drives Claude Code in headless stream-json mode.

    Real event format (from --verbose --include-partial-messages):
      system / subtype=init      → session_id, tools, model (first event)
      system / subtype=hook_*    → hook lifecycle (ignored for display)
      assistant                  → message.content[].type = text | tool_use
      user                       → message.content[].type = tool_result
                                   (partial messages arrive mid-generation)
      result                     → turn complete; session_id, duration_ms, is_error

    Steering:
      interrupt=True  → write new prompt to stdin using --input-format stream-json
      interrupt=False → queue for between-turn delivery (drained on each result event)

    Resume after restart:
      Capture session_id from the init event. On reattach, launch with
      `--resume <session_id>` so Claude Code continues the same conversation thread.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._session_id: str | None = None
        self._event_queue: queue.Queue[AgentEvent | None] = queue.Queue()
        self._pending_between_turn: list[str] = []
        self._lock = threading.Lock()
        self._cwd: Path | None = None
        self._env: dict | None = None
        # True once we've streamed token deltas in the current turn, so we
        # don't re-emit the same text from the final complete assistant message.
        self._turn_had_stream = False
        self._latest_usage: dict | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def launch(self, *, cwd: Path, env: dict, seed_prompt: str) -> SessionHandle:
        return self._spawn(cwd=cwd, env=env, prompt=seed_prompt, resume_id=None)

    def resume(self, *, cwd: Path, env: dict, session_id: str, extra_prompt: str = "") -> SessionHandle:
        """Re-attach to a previous Claude Code session by session_id."""
        return self._spawn(cwd=cwd, env=env, prompt=extra_prompt or None, resume_id=session_id)

    def stream(self) -> Iterator[AgentEvent]:
        while True:
            event = self._event_queue.get()
            if event is None:
                return
            yield event

    def steer(self, message: str, *, interrupt: bool = True) -> SteerResult:
        if self._proc is None or self._proc.poll() is not None:
            return SteerResult(mode="unsupported", ok=False, detail="Process not running")

        if interrupt:
            try:
                self._write_message(message)
                return SteerResult(mode="interrupt", ok=True)
            except (BrokenPipeError, OSError) as exc:
                with self._lock:
                    self._pending_between_turn.append(message)
                return SteerResult(mode="queued", ok=True,
                                   detail=f"stdin write failed ({exc}); queued")
        with self._lock:
            self._pending_between_turn.append(message)
        return SteerResult(mode="queued", ok=True, detail="Queued for next turn")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._event_queue.put(None)

    # ── Internal ──────────────────────────────────────────────────────────

    def _spawn(self, *, cwd: Path, env: dict, prompt: str | None, resume_id: str | None) -> SessionHandle:
        bin_path = env.get("CLAUDE_BIN") or CLAUDE_BIN
        if not bin_path:
            raise RuntimeError(
                "claude binary not found. Set CLAUDE_BIN env var or install Claude Code."
            )

        self._cwd = cwd
        self._env = {**os.environ, **env}

        cmd = [bin_path] + _BASE_FLAGS

        # Agent model + thinking effort (from the launch form)
        model = env.get("AGENT_MODEL", "").strip()
        effort = env.get("AGENT_EFFORT", "").strip()
        if model:
            cmd += ["--model", model]
        if effort:
            cmd += ["--effort", effort]

        # Multi-turn mode when steering is needed
        if prompt and not resume_id:
            # Pass prompt as positional arg (text input mode)
            cmd += [prompt]
        elif resume_id:
            cmd += ["--resume", resume_id]
            if prompt:
                cmd += [prompt]
        elif prompt:
            cmd += [prompt]

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=self._env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        threading.Thread(target=self._drain_stdout, daemon=True,
                         name=f"claude-drain-{self._proc.pid}").start()

        return SessionHandle(
            session_id=self._session_id or f"claude-{self._proc.pid}",
            pid=self._proc.pid,
        )

    def _write_message(self, text: str) -> None:
        """Write a user message to stdin (stream-json input format)."""
        assert self._proc and self._proc.stdin
        payload = json.dumps({"type": "user", "message": text})
        self._proc.stdin.write(payload + "\n")
        self._proc.stdin.flush()

    def _drain_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload={"raw": line}))
                continue

            # Capture session_id from any event that carries it
            if "session_id" in obj and not self._session_id:
                self._session_id = obj["session_id"]

            etype = obj.get("type", "")
            subtype = obj.get("subtype", "")

            if etype == "system" and subtype == "init":
                # Emit as SYSTEM so the drain thread records model/tools info
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload={
                    "msg": f"Session started: {obj.get('session_id')} model={obj.get('model')}",
                    "session_id": obj.get("session_id"),
                    "model": obj.get("model"),
                }))

            elif etype == "system" and subtype in ("hook_started", "hook_response"):
                # Skip hook lifecycle noise
                pass

            elif etype == "stream_event":
                # Real-time partial output (from --include-partial-messages).
                # Format: {"type":"stream_event","event":{"type":"content_block_delta",
                #          "delta":{"type":"text_delta","text":"..."}}}
                inner = obj.get("event", {})
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            self._turn_had_stream = True
                            self._event_queue.put(AgentEvent(
                                type=EventType.TOKEN, payload={"text": text},
                            ))

            elif etype == "assistant":
                # The complete assistant message for the turn. If we already
                # streamed the text via stream_event deltas, emit ONLY tool_use
                # blocks here to avoid duplicating the text. If no deltas were
                # seen (partial messages unavailable), emit the text too.
                msg = obj.get("message", {})
                if isinstance(msg.get("usage"), dict):
                    self._latest_usage = msg["usage"]
                for block in msg.get("content", []):
                    btype = block.get("type")
                    if btype == "text" and not self._turn_had_stream:
                        text = block.get("text", "")
                        if text:
                            self._event_queue.put(AgentEvent(
                                type=EventType.TOKEN, payload={"text": text + "\n"},
                            ))
                    elif btype == "tool_use":
                        self._event_queue.put(AgentEvent(
                            type=EventType.TOOL_USE,
                            payload={
                                "provider_call_id": block.get("id"),
                                "name": block.get("name"),
                                "input": _preview(block.get("input")),
                                "input_bytes": _byte_size(block.get("input")),
                                "status": "started",
                            },
                        ))
                    elif btype == "tool_result":
                        content = block.get("content")
                        self._event_queue.put(AgentEvent(
                            type=EventType.TOOL_RESULT,
                            payload={
                                "provider_call_id": block.get("tool_use_id"),
                                "output": _preview(content),
                                "output_bytes": _byte_size(content),
                                "is_error": block.get("is_error"),
                                "status": "error" if block.get("is_error") else "completed",
                            },
                        ))

            elif etype == "user":
                msg = obj.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    self._event_queue.put(AgentEvent(
                        type=EventType.TOOL_RESULT,
                        payload={
                            "provider_call_id": block.get("tool_use_id"),
                            "output": _preview(block.get("content")),
                            "output_bytes": _byte_size(block.get("content")),
                            "is_error": block.get("is_error"),
                            "status": "error" if block.get("is_error") else "completed",
                        },
                    ))

            elif etype == "result":
                self._turn_had_stream = False
                usage = obj.get("usage")
                if not isinstance(usage, dict):
                    usage = self._latest_usage
                self._event_queue.put(AgentEvent(
                    type=EventType.TURN_END,
                    payload={
                        "session_id": obj.get("session_id"),
                        "duration_ms": obj.get("duration_ms"),
                        "is_error": obj.get("is_error"),
                        "result": obj.get("result", "")[:200],
                        "num_turns": obj.get("num_turns"),
                        "usage": usage,
                        "total_cost_usd": obj.get("total_cost_usd"),
                        "model": obj.get("model"),
                    },
                ))
                self._latest_usage = None
                # Deliver queued between-turn steers
                with self._lock:
                    pending = list(self._pending_between_turn)
                    self._pending_between_turn.clear()
                for msg_text in pending:
                    try:
                        self._write_message(msg_text)
                        self._event_queue.put(AgentEvent(
                            type=EventType.SYSTEM,
                            payload={"msg": f"Between-turn steer delivered: {msg_text[:80]}"},
                        ))
                    except OSError:
                        pass

            else:
                # Emit everything else as SYSTEM for visibility
                self._event_queue.put(AgentEvent(type=EventType.SYSTEM, payload=obj))

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
