"""
Tests that the selected agent model + thinking/reasoning effort are translated
into the correct CLI flags for each surface, and that the surfaces catalog +
attribution behave correctly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from console.backend import surfaces as sc
from console.backend.orchestrator.adapters.claude import ClaudeAdapter
from console.backend.orchestrator.adapters.codex import CodexAdapter
from console.backend.orchestrator.adapters.opencode import OpenCodeAdapter


class _CapturePopen:
    last_cmd: list[str] = []

    def __init__(self, cmd, **kw):
        _CapturePopen.last_cmd = cmd
        self.pid = 1
        self.stdout = iter([])
        self.stdin = None
        self.returncode = 0

    def wait(self):
        return 0

    def poll(self):
        return 0


def _launch(adapter, env):
    with patch.object(subprocess, "Popen", _CapturePopen):
        adapter.launch(cwd=Path("/tmp"), env=env, seed_prompt="hi")
    return _CapturePopen.last_cmd


def test_claude_model_and_effort_flags():
    cmd = _launch(ClaudeAdapter(),
                  {"CLAUDE_BIN": "/bin/claude", "AGENT_MODEL": "opus", "AGENT_EFFORT": "xhigh"})
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "opus"
    assert "--effort" in cmd and cmd[cmd.index("--effort") + 1] == "xhigh"


def test_codex_model_and_effort_flags():
    cmd = _launch(CodexAdapter(),
                  {"CODEX_BIN": "/bin/codex", "AGENT_MODEL": "gpt-5-codex", "AGENT_EFFORT": "high"})
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "gpt-5-codex"
    assert "-c" in cmd and 'model_reasoning_effort="high"' in cmd


def test_opencode_model_and_variant_flags():
    cmd = _launch(OpenCodeAdapter(),
                  {"OPENCODE_BIN": "/bin/opencode",
                   "AGENT_MODEL": "openrouter/openai/gpt-4o-mini", "AGENT_EFFORT": "high"})
    assert "-m" in cmd and cmd[cmd.index("-m") + 1] == "openrouter/openai/gpt-4o-mini"
    assert "--variant" in cmd and cmd[cmd.index("--variant") + 1] == "high"


def test_no_flags_when_unset():
    cmd = _launch(ClaudeAdapter(), {"CLAUDE_BIN": "/bin/claude"})
    assert "--model" not in cmd
    assert "--effort" not in cmd


def test_surface_catalog_shape():
    surfaces = sc.all_surfaces()
    names = {s["surface"] for s in surfaces}
    assert names == {"claude", "codex", "opencode"}
    for s in surfaces:
        assert "" in s["efforts"]          # default option present
        assert isinstance(s["models"], list)
    claude = next(s for s in surfaces if s["surface"] == "claude")
    assert "opus" in claude["models"]
    assert claude["allows_custom_model"] is True
    opencode = next(s for s in surfaces if s["surface"] == "opencode")
    assert opencode["allows_custom_model"] is False


def test_attribution_derivation():
    assert sc.attribution_for("claude", "opus") == ("anthropic", "opus")
    assert sc.attribution_for("codex", "gpt-5.5") == ("openai", "gpt-5.5")
    assert sc.attribution_for("opencode", "openrouter/openai/gpt-4o-mini") == (
        "openrouter", "openai/gpt-4o-mini")
    # Empty model falls back to a sensible default
    assert sc.attribution_for("claude", "") == ("anthropic", "sonnet")


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
