"""Agent adapter protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


class EventType(str, Enum):
    TOKEN = "token"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    TURN_END = "turn_end"
    AGENT_EXIT = "agent_exit"
    SYSTEM = "system"


@dataclass
class AgentEvent:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionHandle:
    session_id: str
    pid: int | None = None


@dataclass
class SteerResult:
    mode: str   # "interrupt" | "queued" | "unsupported"
    ok: bool
    detail: str = ""


@runtime_checkable
class AgentAdapter(Protocol):
    def launch(self, *, cwd: Path, env: dict, seed_prompt: str) -> SessionHandle: ...
    def stream(self) -> Iterator[AgentEvent]: ...
    def steer(self, message: str, *, interrupt: bool = True) -> SteerResult: ...
    def stop(self) -> None: ...
