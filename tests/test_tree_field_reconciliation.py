"""Tree-field reconciliation + research-line cap auto-park.

Run 20260612T105643Z lost 5 cycles to validation errors on tree-walk fields the
controller could have derived, and 1 cycle to the 5-active-line cap. These tests
pin the forgiving behaviour: derive, don't fail.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import ingest_proposals
from autoresearch.controller.workflow import _reconcile_tree_fields
from autoresearch.experiment_registry.registry import (
    init_registry,
    list_proposals,
    list_research_lines,
    upsert_research_line,
    upsert_research_node,
)

from tests.test_handoff import _record_direct, _valid_proposal
from tests.test_runner import _make_config as _config

_RECOMMENDED = [
    {"action_id": "rotate_after_streak", "tree_action": "rotate_axis", "parent_node_id": "node_recent"},
    {"action_id": "extend_current_champion", "tree_action": "exploit_champion", "parent_node_id": "node_champ"},
]


def test_tree_action_alone_binds_matching_recommendation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    init_registry(config.registry_path)
    parsed = {"tree_action": "exploit_champion"}
    _reconcile_tree_fields(config, parsed, _RECOMMENDED)
    assert parsed["selected_tree_action_id"] == "extend_current_champion"
    assert parsed["research_parent_node_id"] == "node_champ"
    assert parsed["tree_field_reconciliations"]


def test_conflicting_tree_action_corrected_to_selected_recommendation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    init_registry(config.registry_path)
    parsed = {"tree_action": "exploit_champion", "selected_tree_action_id": "rotate_after_streak"}
    _reconcile_tree_fields(config, parsed, _RECOMMENDED)
    assert parsed["tree_action"] == "rotate_axis"
    assert parsed["research_parent_node_id"] == "node_recent"


def test_override_rationale_wins_over_reconciliation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    init_registry(config.registry_path)
    parsed = {
        "tree_action": "exploit_champion",
        "selected_tree_action_id": "rotate_after_streak",
        "tree_policy_override_rationale": "Deliberate divergence.",
    }
    _reconcile_tree_fields(config, parsed, _RECOMMENDED)
    assert parsed["tree_action"] == "exploit_champion"  # not corrected


def test_new_line_hypothesis_becomes_override_rationale(tmp_path: Path) -> None:
    config = _config(tmp_path)
    init_registry(config.registry_path)
    parsed = {
        "tree_action": "new_root",
        "research_line_action": "create_line",
        "research_line_id": "feature_engineering",
        "research_line_hypothesis": "Raw features are the ceiling.",
    }
    _reconcile_tree_fields(config, parsed, _RECOMMENDED)
    assert "Raw features are the ceiling." in parsed["tree_policy_override_rationale"]


def test_cross_line_parent_rebinds_research_line(tmp_path: Path) -> None:
    config = _config(tmp_path)
    init_registry(config.registry_path)
    upsert_research_line(config.registry_path, line_id="line_a", label="A")
    upsert_research_line(config.registry_path, line_id="line_b", label="B")
    upsert_research_node(config.registry_path, node_id="node_in_a", line_id="line_a")
    parsed = {
        "research_line_id": "line_b",
        "research_line_action": "extend_line",
        "research_parent_node_id": "node_in_a",
    }
    _reconcile_tree_fields(config, parsed, [])
    assert parsed["research_line_id"] == "line_a"


def test_create_line_at_cap_auto_parks_lru_line(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    (config.metadata_dir / "dataset_schema.json").write_text(
        '{"columns": [{"name": "Exposure", "role": "numeric_feature"}]}',
        encoding="utf-8",
    )
    for index in range(5):
        upsert_research_line(config.registry_path, line_id=f"line_{index}", label=f"Line {index}")
    proposal = _valid_proposal()
    proposal["research_line_id"] = "line_new"
    proposal["research_line_label"] = "New line at the cap"
    # No park_research_line_id supplied — previously a hard validation error.
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "at_cap.json").write_text(
        json.dumps(proposal), encoding="utf-8"
    )

    summary = ingest_proposals(config)

    assert summary["valid_count"] == 1, summary
    assert any(item["status"] == "validated" for item in list_proposals(config.registry_path))
    active = {line["line_id"] for line in list_research_lines(config.registry_path, status="active")}
    parked = {line["line_id"] for line in list_research_lines(config.registry_path, status="parked")}
    assert "line_new" in active
    assert len(active) <= 5
    assert len(parked) == 1
