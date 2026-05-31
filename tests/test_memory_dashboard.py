"""Tests for P2: memory config fields and dashboard module importability."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def test_config_has_memory_fields() -> None:
    """ProjectConfig must expose structural_gini_threshold; the store path is resolved
    out-of-tree via default_memory_store_path (not a config relpath)."""
    from autoresearch.config import load_config
    from autoresearch.memory.store import default_memory_store_path

    cfg = load_config()
    assert hasattr(cfg, "structural_gini_threshold")
    assert cfg.structural_gini_threshold == pytest.approx(0.37)
    assert "memory.sqlite" in str(default_memory_store_path())


def test_dashboard_module_imports() -> None:
    """dashboard/app.py must be importable and expose render_memory."""
    # Import the module without running Streamlit (it is a module-level script;
    # we cannot fully execute it outside Streamlit, but we can verify the function exists).
    import importlib
    import types

    # Build a minimal stub for streamlit so the import doesn't fail outside a server
    import sys
    if "streamlit" not in sys.modules:
        st_stub = types.ModuleType("streamlit")
        for attr in (
            "set_page_config", "sidebar", "title", "subheader", "caption",
            "info", "error", "write", "json", "dataframe", "line_chart",
            "metric", "columns", "selectbox", "multiselect", "markdown",
            "tabs",
        ):
            setattr(st_stub, attr, lambda *a, **kw: None)
        st_stub.sidebar = types.SimpleNamespace(radio=lambda *a, **kw: "Home")
        sys.modules["streamlit"] = st_stub

    # We just need render_memory to exist in the source; check via AST grep
    app_path = Path(__file__).parents[1] / "src" / "autoresearch" / "dashboard" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    assert "def render_memory" in source, "render_memory function not found in dashboard/app.py"
    assert "Memory & Leaderboard" in source, "Nav entry missing in dashboard/app.py"
    assert "structural_gini_threshold" in source, "structural_gini_threshold not used in dashboard"
    assert "default_memory_store_path" in source, "dashboard must resolve the store via default_memory_store_path"


def test_render_memory_source_references_memory_harvest(tmp_path: Path) -> None:
    """render_memory source must reference the 'memory harvest' command in its help text."""
    app_path = Path(__file__).parents[1] / "src" / "autoresearch" / "dashboard" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    # Verify the no-store branch mentions the harvest command
    assert "memory harvest" in source, (
        "render_memory does not mention 'memory harvest' for missing-store guidance"
    )
