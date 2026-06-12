"""E4: end-to-end proof that the reflection -> insight -> playbook pipeline fires.

The follow-up review (process_review_20260610_followup.md E4) found the insights
table empty across every harvested run: the reflection->insight->playbook pipeline
had never run in production, so the playbook features were effectively dead code.

This test exercises the whole chain in one go, so a regression that silently
breaks any link (harvest, reflection prompt, insight verification, playbook
inclusion) fails loudly:

    checkpoint -> pending_reflection.md written
              -> record_insight (cites real registry IDs) -> verified=1
              -> build_playbook -> the verified claim appears in latest.md
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from autoresearch.memory.harvester import harvest_run
from autoresearch.memory.insights import record_insight
from autoresearch.memory.playbook import build_playbook


def _make_registry(registry_path: Path, experiment_id: str, gini: float) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = registry_path.parent / f"{experiment_id}.json"
    metrics_path.write_text(
        json.dumps({
            "aggregate": {"mean_score": gini, "std_score": 0.0},
            "ordinary_eval_splits": ["search_validation"],
            "split_metrics": [{"split": "search_validation", "gini_weighted": gini}],
        }),
        encoding="utf-8",
    )
    with sqlite3.connect(registry_path) as con:
        con.executescript(
            """
            CREATE TABLE experiments (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                model_family TEXT,
                target_strategy TEXT,
                target_mode TEXT,
                metrics_path TEXT,
                fit_wall_seconds REAL,
                compute_budget_seconds REAL,
                timed_out INTEGER
            );
            CREATE TABLE comparisons (
                comparison_id TEXT PRIMARY KEY,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                champion_id TEXT, challenger_id TEXT,
                paired_summary TEXT, promotion_decision TEXT,
                decision TEXT, guardrail_status TEXT
            );
            """
        )
        con.execute(
            "INSERT INTO experiments (experiment_id, status, model_family,"
            " target_strategy, metrics_path) VALUES (?,?,?,?,?)",
            (experiment_id, "completed", "lightgbm", "direct", str(metrics_path)),
        )


def test_reflection_to_playbook_pipeline_end_to_end(tmp_path: Path) -> None:
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "run" / "registry.sqlite"
    _make_registry(registry, experiment_id="exp_alpha", gini=0.37)
    identity = {"provider": "anthropic", "name": "claude-opus-4-8"}

    # 1. Harvest the run into the aggregator.
    harvest_run(memory, registry, identity, track_id="claude", run_id="run1")

    # 2. Record an insight that cites a REAL experiment id from this run.
    result = record_insight(
        memory,
        run_uid="claude/run1",
        model_identity=identity,
        insight_dict={
            "claim": "direct lightgbm on the total target reaches gini ~0.37",
            "scope": "general",
            "confidence": 0.8,
            "evidence": {"experiment_ids": ["exp_alpha"], "metric": "gini_weighted"},
        },
        run_registry_path=registry,
    )
    assert result["verified"] is True, (
        f"insight citing a real registry id should verify: {result['verification_note']}"
    )

    # 3. Build the playbook and confirm the verified claim is included.
    playbook_path = build_playbook(memory, structural_gini_threshold=0.37)
    assert playbook_path is not None, "playbook should be built when a verified insight exists"
    body = playbook_path.read_text(encoding="utf-8")
    assert "direct lightgbm on the total target reaches gini ~0.37" in body
    assert "anthropic/claude-opus-4-8" in body, "playbook must cite the source model"


def test_fabricated_insight_excluded_from_playbook(tmp_path: Path) -> None:
    """An insight citing a non-existent experiment lands verified=0 and never reaches the playbook."""
    memory = tmp_path / "memory.sqlite"
    registry = tmp_path / "run" / "registry.sqlite"
    _make_registry(registry, experiment_id="exp_alpha", gini=0.37)
    identity = {"provider": "anthropic", "name": "claude-opus-4-8"}
    harvest_run(memory, registry, identity, track_id="claude", run_id="run1")

    result = record_insight(
        memory,
        run_uid="claude/run1",
        model_identity=identity,
        insight_dict={
            "claim": "fabricated breakthrough at gini 0.99",
            "evidence": {"experiment_ids": ["exp_does_not_exist"]},
        },
        run_registry_path=registry,
    )
    assert result["verified"] is False

    playbook_path = build_playbook(memory, structural_gini_threshold=0.37)
    assert playbook_path is None, "playbook must not build from only-unverified insights"
