"""Tests for the repair-loop resume (cost-review D1).

On a repaired rerun the attempt loop must resume at the prepared attempt instead
of re-fitting the known-failing earlier attempts from scratch, and it must replay
the persisted per-attempt lifts so the two-consecutive-attempts auto-abandon
check survives the resume.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.controller.workflow import (
    _reconstruct_attempt_lifts,
    _resume_attempt_index,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fresh_cycle_starts_at_attempt_one(tmp_path: Path):
    """No repair requests yet -> start at attempt 1, no replayed history."""
    assert _resume_attempt_index(tmp_path) == 1
    assert _reconstruct_attempt_lifts(tmp_path, 1) == []


def test_resume_skips_known_failing_attempt(tmp_path: Path):
    """A pending repair_request_2 + a supplied recipe_attempt_2 resumes at 2."""
    _write(tmp_path / "repair_request_2.json", {"next_attempt": 2, "failed_attempt_lift": -0.01})
    _write(tmp_path / "recipe_attempt_2.json", {"estimator": "lightgbm"})

    assert _resume_attempt_index(tmp_path) == 2
    # Attempt 1's lift (persisted in repair_request_2) is replayed for auto-abandon.
    assert _reconstruct_attempt_lifts(tmp_path, 2) == [-0.01]


def test_resume_requires_the_prepared_input(tmp_path: Path):
    """A repair request without the corresponding attempt file does not resume."""
    _write(tmp_path / "repair_request_2.json", {"next_attempt": 2, "failed_attempt_lift": -0.01})
    # No recipe_attempt_2/model_attempt_2 written yet.
    assert _resume_attempt_index(tmp_path) == 1


def test_resume_prefers_highest_prepared_attempt(tmp_path: Path):
    """With both attempt 2 and 3 prepared, resume at the latest (3)."""
    _write(tmp_path / "repair_request_2.json", {"next_attempt": 2, "failed_attempt_lift": -0.02})
    _write(tmp_path / "recipe_attempt_2.json", {"estimator": "lightgbm"})
    _write(tmp_path / "repair_request_3.json", {"next_attempt": 3, "failed_attempt_lift": -0.015})
    _write(tmp_path / "model_attempt_3.py", "def fit_predict(*a, **k): ...")

    assert _resume_attempt_index(tmp_path) == 3
    # Attempts 1 and 2 lifts replayed in order for the auto-abandon window.
    assert _reconstruct_attempt_lifts(tmp_path, 3) == [-0.02, -0.015]


def test_missing_lift_reconstructs_as_none(tmp_path: Path):
    """A runtime/compute failure persists no lift -> replayed as None."""
    _write(tmp_path / "repair_request_2.json", {"next_attempt": 2})  # no failed_attempt_lift
    _write(tmp_path / "recipe_attempt_2.json", {"estimator": "lightgbm"})

    assert _reconstruct_attempt_lifts(tmp_path, 2) == [None]
