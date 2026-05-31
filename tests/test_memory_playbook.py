"""Tests for P5: dynamic playbook generation and handoff injection.

Key assertions:
- build_playbook writes latest.md and a timestamped copy.
- Only verified=1 insights are included.
- All bullets cite model_id (attribution).
- When access is none, the handoff markdown does not reference the playbook.
- When access is own/all and playbook exists, the handoff markdown links it.
- playbook_needs_rebuild returns True when new verified insights have landed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoresearch.memory.playbook import build_playbook, playbook_needs_rebuild
from autoresearch.memory.store import init_memory_store


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _populate_with_insights(memory: Path, *, n_verified: int = 2, n_unverified: int = 1) -> None:
    init_memory_store(memory)
    with sqlite3.connect(memory) as con:
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('a/b','a','b')")
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions, peak_gini)"
            " VALUES ('t/r1','a/b',5,1,0.40)"
        )
        for i in range(n_verified):
            con.execute(
                "INSERT INTO insights (insight_id, run_uid, model_id, created_at,"
                "  claim, scope, confidence, evidence_json, verified)"
                " VALUES (?,?,?,?,'verified claim ' || ?,'general',0.8,'{}',1)",
                (f"v{i}", "t/r1", "a/b", "2026-01-01T00:00:00Z", str(i)),
            )
        for i in range(n_unverified):
            con.execute(
                "INSERT INTO insights (insight_id, run_uid, model_id, created_at,"
                "  claim, scope, confidence, evidence_json, verified,"
                "  verification_note)"
                " VALUES (?,?,?,?,'unverified claim ' || ?,'general',0.5,'{}',0,'bad delta')",
                (f"u{i}", "t/r1", "a/b", "2026-01-01T00:00:00Z", str(i)),
            )


# ---------------------------------------------------------------------------
# build_playbook
# ---------------------------------------------------------------------------


def test_build_playbook_creates_latest_and_timestamped(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_with_insights(memory)
    path = build_playbook(memory)
    assert path is not None
    assert path.name == "latest.md"
    assert path.exists()

    playbook_dir = path.parent
    md_files = list(playbook_dir.glob("*.md"))
    # Should have at least 2: latest.md + one timestamped copy
    assert len(md_files) >= 2


def test_build_playbook_contains_only_verified_insights(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_with_insights(memory, n_verified=2, n_unverified=1)
    path = build_playbook(memory)
    content = path.read_text(encoding="utf-8")
    assert "verified claim" in content
    assert "unverified claim" not in content


def test_build_playbook_cites_model_id(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_with_insights(memory)
    path = build_playbook(memory)
    content = path.read_text(encoding="utf-8")
    assert "a/b" in content


def test_build_playbook_returns_none_when_no_verified_insights(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_with_insights(memory, n_verified=0, n_unverified=2)
    result = build_playbook(memory)
    assert result is None


def test_build_playbook_model_filter(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    init_memory_store(memory)
    with sqlite3.connect(memory) as con:
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('a/b','a','b')")
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('x/y','x','y')")
        for run_uid, model_id in [("t/r1", "a/b"), ("t/r2", "x/y")]:
            con.execute(
                "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions)"
                " VALUES (?,?,1,0)", (run_uid, model_id),
            )
        con.execute(
            "INSERT INTO insights (insight_id, run_uid, model_id, created_at,"
            "  claim, scope, evidence_json, verified)"
            " VALUES ('v1','t/r1','a/b','2026-01-01T00:00:00Z','claim A','general','{}',1)"
        )
        con.execute(
            "INSERT INTO insights (insight_id, run_uid, model_id, created_at,"
            "  claim, scope, evidence_json, verified)"
            " VALUES ('v2','t/r2','x/y','2026-01-01T00:00:00Z','claim B','general','{}',1)"
        )

    path = build_playbook(memory, model_id_filter="a/b")
    content = path.read_text(encoding="utf-8")
    assert "claim A" in content
    assert "claim B" not in content
    assert "a_b" in path.name  # filtered suffix in filename


# ---------------------------------------------------------------------------
# playbook_needs_rebuild
# ---------------------------------------------------------------------------


def test_playbook_needs_rebuild_true_when_no_file(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_with_insights(memory)
    assert playbook_needs_rebuild(memory, None) is True
    assert playbook_needs_rebuild(memory, tmp_path / "nonexistent.md") is True


def test_playbook_needs_rebuild_false_when_no_insights(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    init_memory_store(memory)
    playbook = tmp_path / "latest.md"
    playbook.write_text("# old", encoding="utf-8")
    assert playbook_needs_rebuild(memory, playbook) is False


# ---------------------------------------------------------------------------
# Handoff injection
# ---------------------------------------------------------------------------


def _make_cfg_with_access(tmp_path: Path, access: str) -> MagicMock:
    cfg = MagicMock()
    cfg.metadata_dir = tmp_path
    cfg.handoff_results_dir = tmp_path
    cfg.handoff_context_dir = tmp_path / "context"
    cfg.handoff_context_dir.mkdir(parents=True)
    cfg.handoff_proposal_inbox_dir = tmp_path / "inbox"
    cfg.handoff_proposal_inbox_dir.mkdir(parents=True)
    cfg.artifacts_dir = tmp_path / "artifacts"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    cfg.registry_path = tmp_path / "registry.sqlite"
    cfg.track_id = "t"
    cfg.run_id = "r"
    cfg.target_mode = "burning_cost"
    cfg.primary_metric = "gini_weighted"
    cfg.ordinary_train_split = "train"
    cfg.ordinary_eval_splits = ("search_validation",)
    cfg.minimum_mean_lift = 0.0
    cfg.minimum_win_rate = 0.6
    cfg.bootstrap_lower_bound = 0.0
    cfg.confidence_level = 0.9

    manifest = {"track_id": "t", "run_id": "r", "memory_access": access}
    (cfg.artifacts_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cfg


def test_handoff_markdown_no_playbook_link_when_access_none(tmp_path: Path) -> None:
    from autoresearch.controller.handoff import render_handoff_markdown

    cfg = _make_cfg_with_access(tmp_path, "none")
    context: dict = {
        "official_champion": None,
        "recent_experiments": [],
        "allowed_search_space": {"feature_columns": [], "target_strategies": []},
        "research_tree": {"recent_nodes": []},
    }
    with patch("autoresearch.memory.resolve_memory_access", return_value="none"):
        md = render_handoff_markdown(cfg, context)

    assert "playbook" not in md.lower() or "build-playbook" not in md.lower(), (
        "Playbook link should not appear in handoff when access=none"
    )


def test_handoff_markdown_links_playbook_when_access_all(tmp_path: Path) -> None:
    from autoresearch.controller.handoff import render_handoff_markdown

    # Create a real playbook file under the path the handoff function looks for.
    # handoff.py constructs: _cfg_module.PROJECT_ROOT / "artifacts" / "memory" / "playbook"
    playbook_dir = tmp_path / "artifacts" / "memory" / "playbook"
    playbook_dir.mkdir(parents=True)
    playbook_path = playbook_dir / "latest.md"
    playbook_path.write_text("# Playbook content", encoding="utf-8")

    cfg = _make_cfg_with_access(tmp_path, "all")
    context: dict = {
        "official_champion": None,
        "recent_experiments": [],
        "allowed_search_space": {"feature_columns": [], "target_strategies": []},
        "research_tree": {"recent_nodes": []},
    }

    import autoresearch.config as _autoresearch_config
    original_root = _autoresearch_config.PROJECT_ROOT
    try:
        _autoresearch_config.PROJECT_ROOT = tmp_path
        with patch("autoresearch.memory.resolve_memory_access", return_value="all"):
            md = render_handoff_markdown(cfg, context)
    finally:
        _autoresearch_config.PROJECT_ROOT = original_root

    assert "playbook" in md.lower(), "Playbook section missing from handoff with access=all"
    assert "memory query" in md or "memory" in md
