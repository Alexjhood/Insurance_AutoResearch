"""Too-good-to-be-true triage: implausible scores/lifts get a pre-decision alert."""

from __future__ import annotations

from types import SimpleNamespace

from autoresearch.controller.workflow import _implausible_result_alert

_CONFIG = SimpleNamespace(primary_metric="gini_weighted")


def test_leaked_score_triggers_alert() -> None:
    # The run 20260612T105643Z cycle-16 shape: CV gini 0.980 vs champion 0.330.
    alert = _implausible_result_alert(
        _CONFIG,
        {"cv_challenger_score": 0.980299, "cv_champion_score": 0.33026, "cv_mean_lift": 0.650039},
        None,
    )
    assert alert is not None and "artifact" in alert.lower()


def test_large_lift_over_established_champion_triggers_alert() -> None:
    alert = _implausible_result_alert(
        _CONFIG,
        {"cv_challenger_score": 0.50, "cv_champion_score": 0.32, "cv_mean_lift": 0.18},
        None,
    )
    assert alert is not None


def test_first_real_model_over_flat_baseline_is_not_flagged() -> None:
    # Cycle 1 of every run: huge lift over the global-mean baseline is expected.
    alert = _implausible_result_alert(
        _CONFIG,
        {"cv_challenger_score": 0.322994, "cv_champion_score": 0.000399, "cv_mean_lift": 0.322596},
        None,
    )
    assert alert is None


def test_ordinary_result_is_not_flagged() -> None:
    alert = _implausible_result_alert(
        _CONFIG,
        {"cv_challenger_score": 0.3269, "cv_champion_score": 0.3243, "cv_mean_lift": 0.0027},
        {"split_lift": 0.0114, "split_challenger_score": 0.3766},
    )
    assert alert is None


def test_screening_split_score_can_trigger_alert() -> None:
    alert = _implausible_result_alert(
        _CONFIG,
        {"cv_challenger_score": 0.40, "cv_champion_score": 0.32, "cv_mean_lift": 0.08},
        {"split_lift": 0.05, "split_challenger_score": 0.97},
    )
    assert alert is not None
