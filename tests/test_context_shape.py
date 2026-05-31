import json
from pathlib import Path

from tests.test_runner import _make_config, _write_fixtures
from autoresearch.controller.context import build_llm_context
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.experiment_runner import run_experiment


_EXPECTED_TOP_KEYS = {
    "project_goal",
    "official_champion",
    "recent_experiments",
    "recent_comparisons",
    "recent_proposals",
    "research_tree",
    "proposal_count",
    "latest_cycle_result",
    "latest_nonpromotion_summary",
    "agent_schema",
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
