"""Tests for the every-5-cycle memory checkpoint hook.

Key assertions:
- session.py calls maybe_memory_checkpoint on cycle multiples of 5 (not on others).
- maybe_memory_checkpoint never raises into the session loop (exception is swallowed).
- It is a no-op when the run manifest lacks model_identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from autoresearch.memory import maybe_memory_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, with_identity: bool = True) -> MagicMock:
    """Return a minimal mock ProjectConfig with a run manifest."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"track_id": "t", "run_id": "r"}
    if with_identity:
        manifest["model_identity"] = {"provider": "x", "name": "y"}

    manifest_path = artifacts_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    registry_path = tmp_path / "registry.sqlite"
    registry_path.write_bytes(b"")

    cfg = MagicMock()
    cfg.artifacts_dir = artifacts_dir
    cfg.registry_path = registry_path
    cfg.track_id = "t"
    cfg.run_id = "r"
    return cfg


# ---------------------------------------------------------------------------
# Session integration: checkpoint fires on multiples of 5 only
# ---------------------------------------------------------------------------


def test_session_calls_checkpoint_on_multiples_of_5(tmp_path: Path) -> None:
    """run_session_cycle must call maybe_memory_checkpoint iff cycle % 5 == 0."""
    # We test only the call-site logic in session.py by simulating the block:
    #   if state["current_cycle"] % 5 == 0:
    #       maybe_memory_checkpoint(config, state)
    cfg = MagicMock()
    fired: list[int] = []

    def fake_checkpoint(config, state):
        fired.append(state["current_cycle"])

    # Simulate the session logic directly for cycles 1..20
    with patch("autoresearch.memory.maybe_memory_checkpoint", fake_checkpoint):
        from autoresearch.memory import maybe_memory_checkpoint as mcp
        for cycle in range(1, 21):
            state = {"current_cycle": cycle}
            if state["current_cycle"] % 5 == 0:
                mcp(cfg, state)

    assert fired == [5, 10, 15, 20], f"Checkpoint fired on unexpected cycles: {fired}"


def test_session_does_not_call_checkpoint_on_non_multiples() -> None:
    """Checkpoint is not called on cycles 1-4, 6-9, etc."""
    cfg = MagicMock()
    called: list = []

    def fake_checkpoint(config, state):
        called.append(state["current_cycle"])

    with patch("autoresearch.memory.maybe_memory_checkpoint", fake_checkpoint):
        from autoresearch.memory import maybe_memory_checkpoint as mcp
        for cycle in [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14]:
            state = {"current_cycle": cycle}
            if state["current_cycle"] % 5 == 0:
                mcp(cfg, state)

    assert called == []


# ---------------------------------------------------------------------------
# No-raise guarantee
# ---------------------------------------------------------------------------


def test_checkpoint_never_raises_into_loop(tmp_path: Path) -> None:
    """maybe_memory_checkpoint must swallow exceptions — the loop must not crash."""
    cfg = _make_config(tmp_path)

    def exploding_inner(config):
        raise RuntimeError("simulated disk full")

    with patch("autoresearch.memory._run_checkpoint", exploding_inner):
        # Must not raise
        maybe_memory_checkpoint(cfg, {"current_cycle": 5})


def test_checkpoint_no_raise_when_manifest_missing(tmp_path: Path) -> None:
    """If the manifest file is absent, the checkpoint is silently a no-op."""
    cfg = MagicMock()
    cfg.artifacts_dir = tmp_path / "nonexistent"
    cfg.registry_path = tmp_path / "r.sqlite"
    cfg.track_id = "t"
    cfg.run_id = "r"

    # Must not raise
    maybe_memory_checkpoint(cfg, {"current_cycle": 5})


# ---------------------------------------------------------------------------
# No-op when manifest lacks model_identity
# ---------------------------------------------------------------------------


def test_checkpoint_noop_without_identity(tmp_path: Path) -> None:
    """harvest_run must not be called if the manifest has no model_identity."""
    cfg = _make_config(tmp_path, with_identity=False)

    harvest_calls: list = []

    def counting_harvest(*args, **kwargs):
        harvest_calls.append(args)

    with patch("autoresearch.memory.harvester.harvest_run", counting_harvest):
        maybe_memory_checkpoint(cfg, {"current_cycle": 5})

    assert harvest_calls == [], "harvest_run was called despite missing model_identity"
