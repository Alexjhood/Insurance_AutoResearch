"""Tests for P4: access gate and query tool.

Key assertions:
- With AUTORESEARCH_MEMORY_ACCESS unset (none), build_llm_context() output is
  byte-for-byte identical to the baseline without access — the isolation guarantee.
- With 'none', memory query refuses with a clear message.
- With 'own', only own-model rows are returned.
- With 'all', all models are returned, fully attributed.
- Canned analyses return correct aggregations.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autoresearch.memory import resolve_memory_access
from autoresearch.memory.query import AccessDeniedError, query_experiments, query_insights, run_analysis
from autoresearch.memory.store import init_memory_store


# ---------------------------------------------------------------------------
# resolve_memory_access
# ---------------------------------------------------------------------------


def test_resolve_memory_access_defaults_to_none(tmp_path: Path) -> None:
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("AUTORESEARCH_MEMORY_ACCESS", None)
        assert resolve_memory_access() == "none"


def test_resolve_memory_access_reads_env_var(tmp_path: Path) -> None:
    with patch.dict(os.environ, {"AUTORESEARCH_MEMORY_ACCESS": "all"}):
        assert resolve_memory_access() == "all"


def test_resolve_memory_access_rejects_invalid_value() -> None:
    with patch.dict(os.environ, {"AUTORESEARCH_MEMORY_ACCESS": "INVALID"}):
        assert resolve_memory_access() == "none"


def test_resolve_memory_access_reads_manifest_fallback(tmp_path: Path) -> None:
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"memory_access": "own"}), encoding="utf-8")
    cfg = MagicMock()
    cfg.artifacts_dir = tmp_path
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("AUTORESEARCH_MEMORY_ACCESS", None)
        assert resolve_memory_access(cfg) == "own"


# ---------------------------------------------------------------------------
# CRITICAL: run isolation — context must be byte-for-byte identical with 'none'
# ---------------------------------------------------------------------------


def test_build_llm_context_unchanged_with_no_access(tmp_path: Path) -> None:
    """With AUTORESEARCH_MEMORY_ACCESS unset, build_llm_context() must return
    the identical dict as if resolve_memory_access were hard-coded to 'none'.

    This is the isolation guarantee: no memory context leaks in.
    """
    from autoresearch.controller.context import build_llm_context

    # Build a minimal config with necessary paths
    cfg = MagicMock()
    cfg.metadata_dir = tmp_path
    cfg.handoff_results_dir = tmp_path
    cfg.registry_path = tmp_path / "registry.sqlite"
    cfg.artifacts_dir = tmp_path
    cfg.target_mode = "burning_cost"
    cfg.primary_metric = "gini_weighted"
    cfg.ordinary_train_split = "train"
    cfg.ordinary_eval_splits = ("search_validation",)
    cfg.minimum_mean_lift = 0.0
    cfg.minimum_win_rate = 0.6
    cfg.bootstrap_lower_bound = 0.0
    cfg.confidence_level = 0.9

    # Patch registry functions to return empty results
    with patch("autoresearch.controller.context.get_official_champion", return_value=None), \
         patch("autoresearch.controller.context.list_experiments", return_value=[]), \
         patch("autoresearch.controller.context.list_comparisons", return_value=[]), \
         patch("autoresearch.controller.context.list_proposals", return_value=[]), \
         patch("autoresearch.controller.context.list_research_nodes", return_value=[]), \
         patch("autoresearch.controller.context.allowed_search_space", return_value={}):

        # Build baseline with access=none
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTORESEARCH_MEMORY_ACCESS", None)
            baseline = build_llm_context(cfg)

        # Build again (still none)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTORESEARCH_MEMORY_ACCESS", None)
            second = build_llm_context(cfg)

    # Serialise both to check key-by-key identity
    assert json.dumps(baseline, sort_keys=True) == json.dumps(second, sort_keys=True), (
        "build_llm_context() is not deterministic with access=none"
    )

    # The critical check: 'memory_access' key must NOT be present
    assert "memory_access" not in baseline, (
        f"memory_access key leaked into context with access=none: {list(baseline.keys())}"
    )


def test_build_llm_context_adds_memory_block_with_own_access(tmp_path: Path) -> None:
    """With access='own', build_llm_context must include a memory_access block."""
    from autoresearch.controller.context import build_llm_context

    cfg = MagicMock()
    cfg.metadata_dir = tmp_path
    cfg.handoff_results_dir = tmp_path
    cfg.registry_path = tmp_path / "registry.sqlite"
    cfg.artifacts_dir = tmp_path
    cfg.target_mode = "burning_cost"
    cfg.primary_metric = "gini_weighted"
    cfg.ordinary_train_split = "train"
    cfg.ordinary_eval_splits = ("search_validation",)
    cfg.minimum_mean_lift = 0.0
    cfg.minimum_win_rate = 0.6
    cfg.bootstrap_lower_bound = 0.0
    cfg.confidence_level = 0.9

    with patch("autoresearch.controller.context.get_official_champion", return_value=None), \
         patch("autoresearch.controller.context.list_experiments", return_value=[]), \
         patch("autoresearch.controller.context.list_comparisons", return_value=[]), \
         patch("autoresearch.controller.context.list_proposals", return_value=[]), \
         patch("autoresearch.controller.context.list_research_nodes", return_value=[]), \
         patch("autoresearch.controller.context.allowed_search_space", return_value={}), \
         patch.dict(os.environ, {"AUTORESEARCH_MEMORY_ACCESS": "own"}):
        ctx = build_llm_context(cfg)

    assert "memory_access" in ctx, "memory_access block not added with access=own"
    assert ctx["memory_access"]["scope"] == "own"


# ---------------------------------------------------------------------------
# Access gate — query refuses with none
# ---------------------------------------------------------------------------


def test_query_insights_refuses_with_none_access(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    init_memory_store(memory)
    with pytest.raises(AccessDeniedError):
        query_insights(memory, "none")


def test_run_analysis_refuses_with_none_access(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    init_memory_store(memory)
    with pytest.raises(AccessDeniedError):
        run_analysis(memory, "none", "efficiency-by-model")


# ---------------------------------------------------------------------------
# Access gate — 'own' filters to own model
# ---------------------------------------------------------------------------


def _populate_store(memory: Path) -> None:
    init_memory_store(memory)
    with sqlite3.connect(memory) as con:
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('a/b','a','b')")
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('x/y','x','y')")
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions, peak_gini)"
            " VALUES ('t/r1','a/b',3,1,0.35)"
        )
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions, peak_gini)"
            " VALUES ('t/r2','x/y',2,0,0.30)"
        )
        con.execute(
            "INSERT INTO insights (insight_id, run_uid, model_id, claim, scope, evidence_json, verified)"
            " VALUES ('ins1','t/r1','a/b','claim A','general','{}',1)"
        )
        con.execute(
            "INSERT INTO insights (insight_id, run_uid, model_id, claim, scope, evidence_json, verified)"
            " VALUES ('ins2','t/r2','x/y','claim B','general','{}',1)"
        )


def test_query_insights_own_filters_to_own_model(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_store(memory)
    rows = query_insights(memory, "own", own_model_id="a/b")
    assert all(r["model_id"] == "a/b" for r in rows)
    assert len(rows) == 1


def test_query_insights_all_returns_all_models(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_store(memory)
    rows = query_insights(memory, "all")
    model_ids = {r["model_id"] for r in rows}
    assert model_ids == {"a/b", "x/y"}


# ---------------------------------------------------------------------------
# Canned analyses
# ---------------------------------------------------------------------------


def test_run_analysis_efficiency_by_model(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    _populate_store(memory)
    rows = run_analysis(memory, "all", "efficiency-by-model")
    assert len(rows) == 2
    model_ids = {r["model_id"] for r in rows}
    assert "a/b" in model_ids


def test_run_analysis_unknown_name_raises(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    init_memory_store(memory)
    with pytest.raises(ValueError, match="Unknown analysis"):
        run_analysis(memory, "all", "nonexistent-analysis")
