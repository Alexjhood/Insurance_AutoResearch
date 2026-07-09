import json
from pathlib import Path

from tests.test_runner import _make_config, _write_fixtures
from autoresearch.controller.context import _compact_research_nodes, build_llm_context
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.experiment_runner import run_experiment


_EXPECTED_TOP_KEYS = {
    "project_goal",
    "active_dataset",
    "official_champion",
    "recent_experiments",
    "recent_comparisons",
    "recent_proposals",
    "research_tree",
    "research_lines",
    "proposal_count",
    "active_queue",
    "latest_cycle_result",
    "latest_nonpromotion_summary",
    "dataset_schema",
    "allowed_search_space",
    "evaluation_rules",
}


def test_context_top_level_keys(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)
    context = build_llm_context(config)
    assert set(context.keys()) == _EXPECTED_TOP_KEYS


def test_context_size_fresh_registry(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)
    context = build_llm_context(config)
    assert len(json.dumps(context)) < 6000


def test_research_node_context_keeps_cv_and_split_metrics_distinct() -> None:
    nodes = _compact_research_nodes(
        [
            {
                "node_id": "promoted",
                "status": "promoted",
                "metrics": {
                    "split_lift": -0.000126,
                    "split_challenger_score": 0.3733,
                    "cv_mean_lift": 0.01,
                    "cv_challenger_score": 0.3742,
                },
            }
        ]
    )

    assert nodes[0]["cv_lift"] == 0.01
    assert nodes[0]["split_lift"] == -0.000126
    assert nodes[0]["cv_score"] == 0.3742
    assert nodes[0]["split_score"] == 0.3733
    assert "lift" not in nodes[0]


def test_context_size_with_experiment(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)
    exp_config = tmp_path / "experiment.toml"
    exp_config.write_text(
        """
experiment_name = "ctx_test"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )
    run_experiment(config, exp_config)
    initialise_official_champion(config)
    context = build_llm_context(config)
    assert set(context.keys()) == _EXPECTED_TOP_KEYS
    assert len(json.dumps(context)) < 8000


def test_tree_policy_recommends_structural_move_at_plateau() -> None:
    from autoresearch.controller.context import _build_tree_policy

    def _rejected(i):
        return {"node_id": f"n{i}", "status": "rejected", "outcome_type": "llm_rejected",
                "exploration_axis": "hyperparameter", "experiment_id": f"e{i}"}

    nodes = [_rejected(i) for i in range(5)]  # newest first
    policy = _build_tree_policy(None, nodes)
    assert policy["non_promotion_streak"] == 5
    assert policy["recommended_actions"][0]["action_id"] == "break_plateau_structurally"

    # A promotion inside the window resets the streak: no plateau action.
    nodes_with_promotion = nodes[:2] + [
        {"node_id": "champ", "status": "promoted", "outcome_type": None,
         "exploration_axis": "model_family", "experiment_id": "echamp"},
    ] + nodes[2:]
    policy = _build_tree_policy(None, nodes_with_promotion)
    assert policy["non_promotion_streak"] == 2
    assert all(a["action_id"] != "break_plateau_structurally" for a in policy["recommended_actions"])

    # Invalid proposals are not evidence and do not break the streak.
    nodes_with_invalid = nodes[:3] + [
        {"node_id": "bad", "status": "failed", "outcome_type": "invalid",
         "exploration_axis": "hyperparameter"},
    ] + nodes[3:]
    policy = _build_tree_policy(None, nodes_with_invalid)
    assert policy["non_promotion_streak"] == 5
