"""Idempotent + incremental + atomic build behaviour (SPEC §2.1)."""

from __future__ import annotations

import json
import os
import time

import pytest

import importlib

from flightdeck.etl.build import build, build_orchestration, snapshots_dir

# The package __init__ re-exports the `build` function, shadowing the submodule
# attribute — grab the real module object from sys.modules for monkeypatching.
build_mod = importlib.import_module("flightdeck.etl.build")
from flightdeck.etl.tests.fixtures.make_mini_fixture import ORCH_ID


def _snapshot_mtime(root) -> float:
    return (snapshots_dir(root) / ORCH_ID / "snapshot.json").stat().st_mtime


def test_rebuild_skips_when_unchanged(mini_repo_fresh):
    root = mini_repo_fresh
    build(root, [ORCH_ID], log=lambda *_: None)
    first = _snapshot_mtime(root)
    time.sleep(0.02)
    # Second build with no source change → skip (snapshot not rewritten).
    build(root, [ORCH_ID], log=lambda *_: None)
    assert _snapshot_mtime(root) == first


def test_force_rebuilds(mini_repo_fresh):
    root = mini_repo_fresh
    build(root, [ORCH_ID], log=lambda *_: None)
    first = _snapshot_mtime(root)
    time.sleep(0.02)
    build(root, [ORCH_ID], force=True, log=lambda *_: None)
    assert _snapshot_mtime(root) > first


def test_source_change_triggers_rebuild(mini_repo_fresh):
    root = mini_repo_fresh
    build(root, [ORCH_ID], log=lambda *_: None)
    first = _snapshot_mtime(root)
    # Touch a source file into the future so max source mtime advances.
    notes = build_mod.orchestrations_dir(root) / ORCH_ID / "notes.json"
    future = time.time() + 10
    os.utime(notes, (future, future))
    build(root, [ORCH_ID], log=lambda *_: None)
    assert _snapshot_mtime(root) > first


def test_atomicity_no_partial_dir_on_failure(mini_repo_fresh, monkeypatch):
    """An injected failure mid-build leaves no snapshot dir and no .tmp dir."""
    root = mini_repo_fresh

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(build_mod, "_do_build", boom)
    with pytest.raises(RuntimeError, match="injected failure"):
        build_orchestration(ORCH_ID, root, log=lambda *_: None)

    out = snapshots_dir(root) / ORCH_ID
    tmp = snapshots_dir(root) / f"{ORCH_ID}.tmp"
    assert not out.exists()
    assert not tmp.exists()


def test_atomicity_preserves_previous_on_failure(mini_repo_fresh, monkeypatch):
    """A failed rebuild must not destroy a previously-good snapshot."""
    root = mini_repo_fresh
    build(root, [ORCH_ID], log=lambda *_: None)
    good = json.loads((snapshots_dir(root) / ORCH_ID / "snapshot.json").read_text())

    def boom(*_a, **_k):
        raise RuntimeError("injected")

    monkeypatch.setattr(build_mod, "_do_build", boom)
    with pytest.raises(RuntimeError):
        build_orchestration(ORCH_ID, root, force=True, log=lambda *_: None)

    still = json.loads((snapshots_dir(root) / ORCH_ID / "snapshot.json").read_text())
    assert still == good
