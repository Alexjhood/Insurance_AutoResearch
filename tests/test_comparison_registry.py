from pathlib import Path

from autoresearch.experiment_registry.registry import init_registry, list_comparisons, record_comparison


def test_record_comparison_round_trips(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"
    init_registry(registry_path)

    record_comparison(
        registry_path,
        comparison_id="cmp",
        champion_id="a",
        challenger_id="b",
        paired_summary={"mean_lift": 1.0, "challenger_win_rate": 0.7},
        bootstrap_summary={"interval_lower": 0.1, "interval_upper": 2.0},
        promotion_decision="promote",
        promotion_rationale="passed",
        artifacts={"comparison_summary": tmp_path / "summary.json"},
    )

    rows = list_comparisons(registry_path)

    assert rows[0]["comparison_id"] == "cmp"
    assert rows[0]["mean_lift"] == 1.0
    assert rows[0]["promotion_decision"] == "promote"
