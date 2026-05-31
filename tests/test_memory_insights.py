"""Tests for P3: evidence-bound insight validation and storage.

Key assertions:
- validate_insight returns verified=True for evidence that matches the registry.
- validate_insight returns verified=False with a descriptive note for fabricated delta.
- record_insight stores both verified and unverified insights.
- list_insights defaults to verified-only; --include-unverified returns both.
- The reflection prompt is written at checkpoint.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from autoresearch.memory.insights import list_insights, record_insight, validate_insight
from autoresearch.memory.store import init_memory_store


# ---------------------------------------------------------------------------
# Registry fixture helpers
# ---------------------------------------------------------------------------


def _make_registry_with_metrics(
    registry_path: Path,
    experiments: list[dict],
    comparisons: list[dict] | None = None,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(registry_path) as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                metrics_path TEXT
            );
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                champion_id TEXT,
                challenger_id TEXT,
                decision TEXT
            );
            """
        )
        for exp in experiments:
            con.execute(
                "INSERT INTO experiments (experiment_id, status, metrics_path) VALUES (?,?,?)",
                (exp["experiment_id"], exp.get("status", "completed"), exp.get("metrics_path")),
            )
        for cmp in (comparisons or []):
            con.execute(
                "INSERT INTO comparisons (comparison_id, champion_id, challenger_id, decision)"
                " VALUES (?,?,?,?)",
                (
                    cmp["comparison_id"],
                    cmp.get("champion_id"),
                    cmp.get("challenger_id"),
                    cmp.get("decision", "reject"),
                ),
            )


def _write_metrics(path: Path, gini: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "aggregate": {"mean_score": gini, "std_score": 0.0},
            "ordinary_eval_splits": ["search_validation"],
            "split_metrics": [{"split": "search_validation", "gini_weighted": gini}],
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# validate_insight
# ---------------------------------------------------------------------------


def test_validate_insight_verified_when_evidence_matches(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    _write_metrics(m1, 0.33)
    _write_metrics(m2, 0.40)
    _make_registry_with_metrics(
        registry,
        [
            {"experiment_id": "exp1", "metrics_path": str(m1)},
            {"experiment_id": "exp2", "metrics_path": str(m2)},
        ],
    )

    insight = {
        "claim": "total-target trees reach ~0.40",
        "evidence": {
            "experiment_ids": ["exp1", "exp2"],
            "metric": "gini_weighted",
            "delta": 0.07,  # 0.40 - 0.33
        },
    }
    verified, note = validate_insight(registry, insight)
    assert verified is True, f"Expected verified=True, got note: {note}"


def test_validate_insight_fails_when_experiment_missing(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    _make_registry_with_metrics(registry, [{"experiment_id": "exp1"}])

    insight = {
        "claim": "something",
        "evidence": {"experiment_ids": ["exp1", "nonexistent_exp"]},
    }
    verified, note = validate_insight(registry, insight)
    assert verified is False
    assert "nonexistent_exp" in note


def test_validate_insight_fails_fabricated_delta(tmp_path: Path) -> None:
    """An insight with a delta that doesn't match the registry must land verified=0."""
    registry = tmp_path / "registry.sqlite"
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    _write_metrics(m1, 0.33)
    _write_metrics(m2, 0.40)
    _make_registry_with_metrics(
        registry,
        [
            {"experiment_id": "exp1", "metrics_path": str(m1)},
            {"experiment_id": "exp2", "metrics_path": str(m2)},
        ],
    )

    insight = {
        "claim": "fabricated claim",
        "evidence": {
            "experiment_ids": ["exp1", "exp2"],
            "metric": "gini_weighted",
            "delta": 0.50,  # actual delta is ~0.07, not 0.50
        },
    }
    verified, note = validate_insight(registry, insight)
    assert verified is False
    assert "delta" in note.lower() or "differ" in note.lower(), (
        f"Expected note about delta mismatch, got: {note}"
    )


def test_validate_insight_fails_when_comparison_missing(tmp_path: Path) -> None:
    registry = tmp_path / "registry.sqlite"
    _make_registry_with_metrics(
        registry,
        [],
        comparisons=[{"comparison_id": "cmp1"}],
    )
    insight = {
        "claim": "something",
        "evidence": {"comparison_ids": ["cmp1", "nonexistent_cmp"]},
    }
    verified, note = validate_insight(registry, insight)
    assert verified is False
    assert "nonexistent_cmp" in note


def test_validate_insight_refuses_holdout_path(tmp_path: Path) -> None:
    bogus = tmp_path / "holdout_vault" / "registry.sqlite"
    bogus.parent.mkdir(parents=True)
    bogus.write_bytes(b"")
    with pytest.raises(ValueError, match="holdout guard"):
        validate_insight(bogus, {"claim": "x", "evidence": {}})


# ---------------------------------------------------------------------------
# record_insight
# ---------------------------------------------------------------------------


def test_record_insight_verified_stored(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    _write_metrics(m1, 0.33)
    _write_metrics(m2, 0.40)
    _make_registry_with_metrics(
        registry,
        [
            {"experiment_id": "exp1", "metrics_path": str(m1)},
            {"experiment_id": "exp2", "metrics_path": str(m2)},
        ],
    )
    init_memory_store(memory)
    # We need a runs row or the FK will fail — insert directly
    with sqlite3.connect(memory) as con:
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('a/b','a','b')")
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions) VALUES ('t/r','a/b',2,0)"
        )

    insight = {
        "claim": "total-target trees reach ~0.40",
        "scope": "general",
        "confidence": 0.8,
        "evidence": {
            "experiment_ids": ["exp1", "exp2"],
            "metric": "gini_weighted",
            "delta": 0.07,
        },
    }
    result = record_insight(
        memory, "t/r", {"provider": "a", "name": "b"}, insight,
        run_registry_path=registry,
    )
    assert result["verified"] is True
    assert result["verification_note"] == "ok"

    rows = list_insights(memory, verified_only=True)
    assert len(rows) == 1
    assert rows[0]["insight_id"] == result["insight_id"]


def test_record_insight_unverified_stored(tmp_path: Path) -> None:
    """An insight with a fabricated delta must be stored with verified=0."""
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "registry.sqlite"
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    _write_metrics(m1, 0.33)
    _write_metrics(m2, 0.40)
    _make_registry_with_metrics(
        registry,
        [
            {"experiment_id": "exp1", "metrics_path": str(m1)},
            {"experiment_id": "exp2", "metrics_path": str(m2)},
        ],
    )
    init_memory_store(memory)
    with sqlite3.connect(memory) as con:
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('a/b','a','b')")
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions) VALUES ('t/r','a/b',2,0)"
        )

    insight = {
        "claim": "fabricated claim with wrong delta",
        "scope": "general",
        "confidence": 0.5,
        "evidence": {
            "experiment_ids": ["exp1", "exp2"],
            "metric": "gini_weighted",
            "delta": 0.99,  # wrong
        },
    }
    result = record_insight(
        memory, "t/r", {"provider": "a", "name": "b"}, insight,
        run_registry_path=registry,
    )
    assert result["verified"] is False

    # Verified-only query returns nothing
    verified_rows = list_insights(memory, verified_only=True)
    assert len(verified_rows) == 0

    # Include-unverified returns the row
    all_rows = list_insights(memory, verified_only=False)
    assert len(all_rows) == 1
    assert all_rows[0]["verified"] == 0


def test_list_insights_run_filter(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    init_memory_store(memory)
    with sqlite3.connect(memory) as con:
        con.execute("INSERT INTO models (model_id, provider, name) VALUES ('a/b','a','b')")
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions) VALUES ('t/r1','a/b',1,0)"
        )
        con.execute(
            "INSERT INTO runs (run_uid, model_id, n_experiments, n_promotions) VALUES ('t/r2','a/b',1,0)"
        )
        for run_uid in ["t/r1", "t/r2"]:
            con.execute(
                "INSERT INTO insights (insight_id, run_uid, model_id, claim, scope, evidence_json, verified)"
                " VALUES (?,?,?,'claim','general','{}',1)",
                (f"id_{run_uid.replace('/','_')}", run_uid, "a/b"),
            )

    rows_r1 = list_insights(memory, verified_only=True, run_uid="t/r1")
    assert len(rows_r1) == 1
    assert rows_r1[0]["run_uid"] == "t/r1"

    all_rows = list_insights(memory, verified_only=True)
    assert len(all_rows) == 2


# ---------------------------------------------------------------------------
# Reflection prompt is written at checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_writes_reflection_prompt(tmp_path: Path) -> None:
    """maybe_memory_checkpoint must write pending_reflection.md into handoff dir."""
    import json
    from unittest.mock import MagicMock
    from autoresearch.memory import maybe_memory_checkpoint

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    handoffs_dir = tmp_path / "handoffs"
    handoffs_dir.mkdir()

    manifest = {
        "track_id": "t", "run_id": "r",
        "model_identity": {"provider": "x", "name": "y"},
    }
    (artifacts_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = tmp_path / "registry.sqlite"
    registry.write_bytes(b"")

    cfg = MagicMock()
    cfg.artifacts_dir = artifacts_dir
    cfg.registry_path = registry
    cfg.track_id = "t"
    cfg.run_id = "r"
    cfg.handoff_handoffs_dir = handoffs_dir

    # Patch harvest_run so we don't need a real registry
    with patch("autoresearch.memory.harvester.harvest_run"):
        with patch("autoresearch.config.PROJECT_ROOT", tmp_path):
            maybe_memory_checkpoint(cfg, {"current_cycle": 5})

    prompt = handoffs_dir / "pending_reflection.md"
    assert prompt.exists(), "pending_reflection.md not written by checkpoint"
    content = prompt.read_text(encoding="utf-8")
    assert "record-insight" in content
    assert "insight.json" in content.lower() or "json" in content.lower()
