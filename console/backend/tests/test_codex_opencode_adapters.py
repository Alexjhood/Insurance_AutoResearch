"""
Parser tests for the Codex and OpenCode adapters, using the REAL NDJSON event
formats captured from codex-cli 0.137.0 and opencode 1.16.0.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from console.backend.orchestrator.adapters.base import EventType
from console.backend.orchestrator.adapters.codex import CodexAdapter
from console.backend.orchestrator.adapters.opencode import OpenCodeAdapter


class _FakeProc:
    def __init__(self, lines):
        self.stdout = io.StringIO("".join(l + "\n" for l in lines))
        self.returncode = 0
        self.pid = 999

    def wait(self):
        return 0

    def poll(self):
        return 0


def _drain(adapter, lines):
    adapter._proc = _FakeProc(lines)
    adapter._drain_stdout()
    out = []
    while not adapter._event_queue.empty():
        ev = adapter._event_queue.get_nowait()
        if ev is None:
            break
        out.append(ev)
    return out


# ── Codex (real format) ───────────────────────────────────────────────────────

CODEX_LINES = [
    json.dumps({"type": "thread.started", "thread_id": "019e976f-3dab-79e1-819d-01f7bf1684f6"}),
    json.dumps({"type": "turn.started"}),
    json.dumps({"type": "item.started", "item": {"id": "item_0", "type": "command_execution",
                "command": "/bin/zsh -lc 'echo hello'", "status": "in_progress"}}),
    json.dumps({"type": "item.completed", "item": {"id": "item_0", "type": "command_execution",
                "command": "/bin/zsh -lc 'echo hello'", "aggregated_output": "hello\n",
                "exit_code": 0, "status": "completed"}}),
    json.dumps({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message",
                "text": "DONE"}}),
    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 56064}}),
]


def test_codex_captures_thread_id():
    adapter = CodexAdapter()
    _drain(adapter, CODEX_LINES)
    assert adapter._thread_id == "019e976f-3dab-79e1-819d-01f7bf1684f6"


def test_codex_agent_message_is_token():
    adapter = CodexAdapter()
    events = _drain(adapter, CODEX_LINES)
    tokens = "".join(e.payload["text"] for e in events if e.type == EventType.TOKEN)
    assert "DONE" in tokens


def test_codex_command_execution_is_tool():
    adapter = CodexAdapter()
    events = _drain(adapter, CODEX_LINES)
    tools = [e for e in events if e.type == EventType.TOOL_USE]
    assert len(tools) == 1
    assert tools[0].payload["name"] == "shell"
    assert "echo hello" in tools[0].payload["input"]["command"]
    results = [e for e in events if e.type == EventType.TOOL_RESULT]
    assert len(results) == 1
    assert results[0].payload["provider_call_id"] == "item_0"
    assert results[0].payload["output_bytes"] == 6


def test_codex_turn_and_exit():
    adapter = CodexAdapter()
    events = _drain(adapter, CODEX_LINES)
    assert any(e.type == EventType.TURN_END for e in events)
    assert any(e.type == EventType.AGENT_EXIT for e in events)
    # session_id propagated on exit
    exit_ev = [e for e in events if e.type == EventType.AGENT_EXIT][0]
    assert exit_ev.payload["session_id"] == "019e976f-3dab-79e1-819d-01f7bf1684f6"
    turn = [e for e in events if e.type == EventType.TURN_END][0]
    assert turn.payload["usage"]["input_tokens"] == 56064


# ── OpenCode (real format) ──────────────────────────────────────────────────────

OPENCODE_LINES = [
    json.dumps({"type": "step_start", "timestamp": 1780657219368,
                "sessionID": "ses_1688f2113ffeyxoff1X0WE4ueV",
                "part": {"type": "step-start", "sessionID": "ses_1688f2113ffeyxoff1X0WE4ueV"}}),
    json.dumps({"type": "message", "sessionID": "ses_1688f2113ffeyxoff1X0WE4ueV",
                "part": {"type": "text", "text": "PONG"}}),
    json.dumps({"type": "message", "sessionID": "ses_1688f2113ffeyxoff1X0WE4ueV",
                "part": {"type": "tool", "tool": "bash", "state": {"command": "ls"}}}),
    json.dumps({"type": "step_finish", "sessionID": "ses_1688f2113ffeyxoff1X0WE4ueV",
                "tokens": {"input": 20, "output": 5, "reasoning": 3,
                           "cache": {"read": 10, "write": 2}},
                "cost": 0.002}),
]


def test_opencode_captures_session_id():
    adapter = OpenCodeAdapter()
    _drain(adapter, OPENCODE_LINES)
    assert adapter._session_id == "ses_1688f2113ffeyxoff1X0WE4ueV"


def test_opencode_text_is_token():
    adapter = OpenCodeAdapter()
    events = _drain(adapter, OPENCODE_LINES)
    tokens = "".join(e.payload["text"] for e in events if e.type == EventType.TOKEN)
    assert "PONG" in tokens


def test_opencode_tool_part_is_tool_use():
    adapter = OpenCodeAdapter()
    events = _drain(adapter, OPENCODE_LINES)
    tools = [e for e in events if e.type == EventType.TOOL_USE]
    assert len(tools) == 1
    assert tools[0].payload["name"] == "bash"


def test_opencode_step_finish_is_turn_end():
    adapter = OpenCodeAdapter()
    events = _drain(adapter, OPENCODE_LINES)
    assert any(e.type == EventType.TURN_END for e in events)
    assert any(e.type == EventType.AGENT_EXIT for e in events)
    turn = [e for e in events if e.type == EventType.TURN_END][0]
    assert turn.payload["usage"]["cache"]["read"] == 10
    assert turn.payload["cost"] == 0.002


def test_opencode_sparse_output_does_not_crash():
    """Some models emit only step_start — must still exit cleanly."""
    adapter = OpenCodeAdapter()
    events = _drain(adapter, [OPENCODE_LINES[0]])
    assert any(e.type == EventType.AGENT_EXIT for e in events)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
