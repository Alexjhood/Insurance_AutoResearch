"""Script models must never see target columns for scored rows (leak hole).

A column-sweeping script in run 20260612T105643Z picked up the raw
``ClaimAmount`` of its scoring rows as a feature and posted CV gini 0.98.
The dispatcher now strips target-bearing columns from the score frame and
identifier duplicates (e.g. ``IDpol``) from both frames before any
run-local script sees them.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from autoresearch.models.dispatcher import (
    CLAIM_COST,
    CLAIM_COUNT,
    CLAIM_EVENTS,
    RAW_CLAIM_COST,
    _sanitize_script_frames,
    dispatch_model,
)

from tests.test_recipe import _frame


def test_sanitize_drops_targets_from_score_and_id_dupes_everywhere() -> None:
    frame, _ = _frame()
    frame["IDpol"] = frame["record_id"]  # duplicate identifier, freMTPL2-style
    train = frame.iloc[:300]
    score = frame.copy()

    safe_train, safe_score = _sanitize_script_frames(train, score)

    for col in (CLAIM_COST, RAW_CLAIM_COST, CLAIM_COUNT, CLAIM_EVENTS, "IDpol"):
        assert col not in safe_score.columns
    assert "IDpol" not in safe_train.columns
    assert RAW_CLAIM_COST not in safe_train.columns
    # Training targets stay available for fitting.
    assert CLAIM_COST in safe_train.columns
    assert CLAIM_COUNT in safe_train.columns
    # Features, exposure and the canonical id survive on both frames.
    for col in ("record_id", "Exposure", "VehPower", "Region"):
        assert col in safe_train.columns and col in safe_score.columns
    # Row counts unchanged — only columns are dropped.
    assert len(safe_train) == len(train) and len(safe_score) == len(score)
    # Originals untouched.
    assert RAW_CLAIM_COST in train.columns and CLAIM_COST in score.columns


def test_script_model_cannot_see_score_targets(tmp_path: Path) -> None:
    script = tmp_path / "model_probe.py"
    script.write_text(textwrap.dedent(
        f"""
        import numpy as np
        from autoresearch.models.prediction import Prediction

        FORBIDDEN_SCORE = {{"{CLAIM_COST}", "{RAW_CLAIM_COST}", "{CLAIM_COUNT}", "{CLAIM_EVENTS}", "IDpol"}}

        def fit_predict(train, score, *, feature_inclusions=None,
                        feature_exclusions=None, **hyperparameters):
            leaked = FORBIDDEN_SCORE & set(score.columns)
            assert not leaked, f"score frame leaked target columns: {{leaked}}"
            assert "IDpol" not in train.columns
            assert "{RAW_CLAIM_COST}" not in train.columns
            assert "{CLAIM_COST}" in train.columns  # training target still present
            rate = float(np.average(
                train["{CLAIM_COST}"] / train["Exposure"], weights=train["Exposure"]
            ))
            return Prediction(values=np.full(len(score), rate), unit="rate"), {{}}
        """
    ))

    frame, split = _frame()
    frame["IDpol"] = frame["record_id"]
    res = dispatch_model(
        frame, split, model_family="scripted_challenger",
        target_strategy="direct_pure_premium", train_split="train",
        score_splits=("search_validation",), hyperparameters={},
        model_script_path=script, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)
