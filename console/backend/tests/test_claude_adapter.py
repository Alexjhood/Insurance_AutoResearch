"""
Unit tests for the Claude adapter's stream-json parser.

We feed synthetic stream-json lines (the real format Claude Code emits with
--output-format stream-json --verbose --include-partial-messages) through the
adapter's drain loop and assert the normalized AgentEvents it produces. This
covers the riskiest untested code without needing an authenticated CLI.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Make `console` importable
_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from console.backend.orchestrator.adapters.base import EventType
from console.backend.orchestrator.adapters.claude import ClaudeAdapter


class _FakeProc:
    """Minimal stand-in for subprocess.Popen with a scripted stdout."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = io.StringIO("".join(l + "\n" for l in lines))
        self.stdin = io.StringIO()
        self.returncode = 0
        self.pid = 4242

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def _drain_lines(lines: list[str]) -> list:
    adapter = ClaudeAdapter()
    adapter._proc = _FakeProc(lines)  # type: ignore[assignment]
    adapter._drain_stdout()
    events = []
    while True:
        ev = adapter._event_queue.get_nowait() if not adapter._event_queue.empty() else None
        if ev is None:
            break
        events.append(ev)
    return events, adapter


def test_init_event_captures_session_id():
    lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "abc-123", "model": "claude-opus-4-8"}),
        json.dumps({"type": "result", "subtype": "success",
                    "session_id": "abc-123", "is_error": False, "result": "done"}),
    ]
    events, adapter = _drain_lines(lines)
    assert adapter._session_id == "abc-123"
    sys_events = [e for e in events if e.type == EventType.SYSTEM]
    assert any("Session started" in e.payload.get("msg", "") for e in sys_events)


def test_streaming_deltas_become_tokens():
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello "}}}),
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "world"}}}),
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1"}),
    ]
    events, _ = _drain_lines(lines)
    tokens = [e.payload["text"] for e in events if e.type == EventType.TOKEN]
    assert "".join(tokens) == "Hello world"


def test_no_duplicate_text_when_streamed_then_full_message():
    """If deltas streamed the text, the final assistant message must NOT re-emit it."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello world"}}}),
        # The complete assistant message arrives after streaming
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Hello world"}]}}),
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1"}),
    ]
    events, _ = _drain_lines(lines)
    tokens = [e.payload["text"] for e in events if e.type == EventType.TOKEN]
    # Should appear once, not twice
    assert "".join(tokens) == "Hello world", f"got {tokens!r}"


def test_full_message_emits_when_no_streaming():
    """If partial messages were unavailable, the complete assistant text is emitted."""
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "text", "text": "No streaming here"}]}}),
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1"}),
    ]
    events, _ = _drain_lines(lines)
    tokens = "".join(e.payload["text"] for e in events if e.type == EventType.TOKEN)
    assert "No streaming here" in tokens


def test_tool_use_event():
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "autoresearch bootstrap-track"}}]}}),
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1"}),
    ]
    events, _ = _drain_lines(lines)
    tools = [e for e in events if e.type == EventType.TOOL_USE]
    assert len(tools) == 1
    assert tools[0].payload["name"] == "Bash"


def test_hook_noise_is_skipped():
    lines = [
        json.dumps({"type": "system", "subtype": "hook_started", "session_id": "s1"}),
        json.dumps({"type": "system", "subtype": "hook_response", "session_id": "s1"}),
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1"}),
    ]
    events, _ = _drain_lines(lines)
    # Hook events should not appear as SYSTEM msgs
    msgs = [e.payload.get("msg", "") for e in events if e.type == EventType.SYSTEM]
    assert not any("hook" in m.lower() for m in msgs)


def test_turn_end_and_agent_exit_emitted():
    lines = [
        json.dumps({"type": "system", "subtype": "init", "session_id": "s1"}),
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1",
                    "duration_ms": 1234, "is_error": False, "result": "ok"}),
    ]
    events, _ = _drain_lines(lines)
    assert any(e.type == EventType.TURN_END for e in events)
    assert any(e.type == EventType.AGENT_EXIT for e in events)


def test_malformed_line_does_not_crash():
    lines = [
        "this is not json",
        json.dumps({"type": "result", "subtype": "success", "session_id": "s1"}),
    ]
    events, _ = _drain_lines(lines)
    # Malformed line surfaces as a SYSTEM raw event, parser keeps going
    assert any(e.payload.get("raw") == "this is not json"
               for e in events if e.type == EventType.SYSTEM)
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
