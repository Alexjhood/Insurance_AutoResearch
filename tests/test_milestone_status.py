from autoresearch.controller.milestone_status import milestone_status
from autoresearch.utils.io import write_json
from tests.test_runner import _make_config


def _champion(comparison_id="cmp-1"):
    return {
        "champion_id": "challenger-1",
        "comparison_id": comparison_id,
        "branch_id": "main",
    }


def test_baseline_champion_has_no_milestone_requirement(tmp_path) -> None:
    config = _make_config(tmp_path)

    result = milestone_status(config, _champion(comparison_id=None))

    assert result == {
        "status": "not_applicable",
        "champion_id": "challenger-1",
        "operator_action": None,
    }


def test_skipped_or_missing_report_is_pending_operator(tmp_path) -> None:
    config = _make_config(tmp_path)
    report_dir = config.artifacts_dir / "milestone_reports"
    write_json(
        report_dir / "cmp-1.json",
        {
            "status": "skipped",
            "champion_id": "challenger-1",
            "reason": "token unavailable",
        },
    )

    result = milestone_status(config, _champion())

    assert result["status"] == "pending_operator"
    assert result["operator_action"].endswith(
        "evaluate-milestone challenger-1"
    )


def test_completed_manual_report_satisfies_milestone(tmp_path) -> None:
    config = _make_config(tmp_path)
    report_dir = config.artifacts_dir / "milestone_reports"
    write_json(
        report_dir / "manual_challenger-1.json",
        {
            "status": "completed",
            "champion_id": "challenger-1",
            "holdout_metrics": {"gini_weighted": 0.4},
        },
    )

    result = milestone_status(config, _champion())

    assert result == {
        "status": "completed",
        "champion_id": "challenger-1",
        "operator_action": None,
    }
