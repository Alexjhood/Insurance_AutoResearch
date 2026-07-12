"""Shared pytest fixtures for the Flight Deck ETL tests.

The synthetic mini-fixture (``make_mini_fixture``) lets the whole suite run with
no dependency on local artifacts. Tests that assert against the real reference
orchestration (``20260711T164959Z``) are gated on its presence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flightdeck.etl.build import build, find_repo_root, snapshots_dir
from flightdeck.etl.tests.fixtures.make_mini_fixture import ORCH_ID, make_mini_fixture

REAL_FIXTURE_ID = "20260711T164959Z"

# Subtrees the ETL never reads — pruned when skeleton-copying the real fixture
# so the defensive-degradation test doesn't clone ~800 MB of run internals.
_HEAVY_DIRS = {
    "iterations", "cv_cache", "results", "sessions", "context", "handoffs",
    "milestone_reports", "proposal_inbox", "proposal_processed",
    "orchestration_replay", "baseline",
}


@pytest.fixture(scope="session")
def mini_repo(tmp_path_factory) -> Path:
    """A temp repo root containing only the built synthetic mini-fixture."""
    root = tmp_path_factory.mktemp("mini_repo")
    make_mini_fixture(root)
    build(root, [ORCH_ID], log=lambda *_: None)
    return root


@pytest.fixture()
def mini_repo_fresh(tmp_path) -> Path:
    """A per-test temp repo with the mini-fixture *sources* (not yet built)."""
    make_mini_fixture(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def mini_snapshot(mini_repo: Path) -> dict:
    path = snapshots_dir(mini_repo) / ORCH_ID / "snapshot.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def mini_index(mini_repo: Path) -> dict:
    return json.loads((snapshots_dir(mini_repo) / "index.json").read_text())


@pytest.fixture(scope="session")
def real_repo_root() -> Path:
    root = find_repo_root(Path(__file__))
    if not (root / "artifacts" / "orchestrations" / REAL_FIXTURE_ID).is_dir():
        pytest.skip("real reference fixture not present")
    return root


def skeleton_copy_orchestration(src_root: Path, dst_root: Path, orch_id: str,
                                consolidation: tuple[str, str] | None) -> None:
    """Copy just the files the ETL consumes for one orchestration (+ its
    consolidation run) into ``dst_root``, skipping heavy run internals."""
    src = src_root / "artifacts" / "orchestrations" / orch_id
    dst = dst_root / "artifacts" / "orchestrations" / orch_id
    _prune_copy(src, dst)
    if consolidation is not None:
        track, run_id = consolidation
        csrc = src_root / "artifacts" / "tracks" / track / "runs" / run_id
        cdst = dst_root / "artifacts" / "tracks" / track / "runs" / run_id
        if csrc.is_dir():
            _prune_copy(csrc, cdst)


def _prune_copy(src: Path, dst: Path) -> None:
    def ignore(dirpath: str, names: list[str]) -> set[str]:
        return {n for n in names if n in _HEAVY_DIRS}

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)
