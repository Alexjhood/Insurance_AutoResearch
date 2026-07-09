"""Phase 5 — dataset-neutral agent surfaces (search space, context, recipe legality)."""

from __future__ import annotations

from dataclasses import replace

from autoresearch.config import load_config
from autoresearch.controller.context import _build_active_dataset
from autoresearch.controller.proposal_schema import allowed_search_space, target_columns_for
from autoresearch.datasets import load_dataset_spec


def _config(dataset: str):
    return load_config(dataset=dataset)


def test_search_space_reflects_porto():
    cfg = _config("porto_seguro")
    space = allowed_search_space(cfg, dataset_schema=None)
    assert space["active_target_mode"] == "claim_incidence"
    assert space["target_modes"] == ["claim_incidence"]
    # No claim-count column → frequency_severity is not offered.
    assert "frequency_severity" not in space["target_strategies"]
    # Unit weight is the reserved non-predictive weight column.
    assert "unit_weight" in space["non_predictive_columns"]
    assert space["claim_cap_thresholds"] == [None]
    assert space["allow_disable_claim_capping"] is True


def test_search_space_reflects_french():
    cfg = _config("french_motor")
    space = allowed_search_space(cfg, dataset_schema=None)
    assert "frequency_severity" in space["target_strategies"]  # has ClaimNb
    assert space["claim_cap_thresholds"] == [100000]
    assert "Exposure" in space["non_predictive_columns"]


def test_target_columns_for_datasets():
    porto = target_columns_for(load_dataset_spec("porto_seguro"))
    assert {"record_id", "id", "target"} <= porto
    french = target_columns_for(load_dataset_spec("french_motor"))
    assert {"ClaimAmount", "ClaimAmountCapped", "ClaimNb", "ClaimAmountCount", "IDpol"} <= french


def test_active_dataset_block_porto():
    cfg = _config("porto_seguro")
    block = _build_active_dataset(cfg)
    assert block["name"] == "porto_seguro"
    assert block["target_source_column"] == "target"
    assert block["weight_is_unit"] is True
    assert block["capping"] == "no capping."
    assert block["frequency_severity_available"] is False
    assert any("-1" in c for c in block["cautions"])


def test_active_dataset_block_allstate_grouping_caution():
    cfg = _config("allstate")
    block = _build_active_dataset(cfg)
    assert any("Household_ID" in c for c in block["cautions"])


def test_frequency_severity_rejected_without_count_column():
    from autoresearch.controller.proposal_schema import validate_proposal

    cfg = _config("porto_seguro")
    space = allowed_search_space(cfg, dataset_schema=None)
    proposal = {
        "experiment_name": "x", "rationale": "r", "change_summary": "c",
        "expected_benefit": "b", "key_risk": "k", "exploration_axis": "model_family",
        "approach_family": "gbm", "target_framing": "t", "feature_representation": "f",
        "expected_learning": "l", "proposal_id": "abc", "parent_experiment_id": "p",
        "tree_action": "new_root", "parent_rationale": "pr", "selected_tree_action_id": "s",
        "research_line_action": "create_line", "research_line_id": "linex",
        "research_line_label": "lbl", "research_line_hypothesis": "hyp",
        "line_membership_rationale": "lmr",
        "experiment_config": {
            "experiment_name": "x", "parent_experiment_id": "p",
            "model_family": "recipe", "target_strategy": "frequency_severity",
            "preprocessing": {"claim_capping_enabled": False, "claim_cap_threshold": None},
            "model": {"recipe": {"structure": "frequency_severity", "stages": {
                "frequency": {"estimator": "lightgbm", "objective": "poisson"},
                "severity": {"estimator": "lightgbm", "objective": "gamma"}}}},
        },
    }
    errors = validate_proposal(proposal, space)
    assert any("target_strategy" in e for e in errors)
