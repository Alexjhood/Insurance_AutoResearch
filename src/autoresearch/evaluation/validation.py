"""Run-output validation before promotion assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.evaluation.metrics import full_metric_panel, lower_is_better


@dataclass(frozen=True)
class ValidationRules:
    """Conservative sanity and lift checks for experiment outputs."""

    min_predicted_to_actual_ratio: float = 0.2
    max_predicted_to_actual_ratio: float = 5.0
    min_prediction_cv: float = 1e-6
    max_prediction_to_actual_mean_ratio: float = 100.0
    require_positive_lift: bool = True


def validate_experiment_outputs(
    predictions: pd.DataFrame,
    *,
    eval_split: str,
    primary_metric: str,
    tweedie_power: float,
    champion_predictions: pd.DataFrame | None = None,
    rules: ValidationRules | None = None,
    allow_constant_predictions: bool = False,
) -> dict[str, Any]:
    """Return an audit payload describing whether outputs are credible.

    This is intentionally separate from the promotion gate. It catches likely
    coding mistakes (NaNs, impossible ranges, flat accidental outputs) and can
    also enforce a positive validation lift before a proposal is assessed.
    """

    rules = rules or ValidationRules()
    checks: list[dict[str, Any]] = []
    frame = predictions[predictions["split"] == eval_split].copy()
    _check(checks, "eval_rows_present", not frame.empty, f"No rows for eval split {eval_split!r}")

    if frame.empty:
        return _report(False, checks, "No evaluation rows available", None)

    required_columns = {"record_id", "actual_claim_cost", "predicted_claim_cost", "exposure"}
    missing = sorted(required_columns.difference(frame.columns))
    _check(checks, "required_columns_present", not missing, f"Missing prediction columns: {missing}")
    if missing:
        return _report(False, checks, "Prediction file is missing required columns", None)

    predicted = frame["predicted_claim_cost"].astype(float).to_numpy()
    actual = frame["actual_claim_cost"].astype(float).to_numpy()
    exposure = frame["exposure"].astype(float).to_numpy()

    finite = bool(np.isfinite(predicted).all())
    _check(checks, "predictions_finite", finite, "Predictions contain NaN or infinite values")
    non_negative = bool((predicted >= 0).all())
    _check(checks, "predictions_non_negative", non_negative, "Predictions contain negative values")
    exposure_positive = bool((exposure > 0).all())
    _check(checks, "exposure_positive", exposure_positive, "Exposure must be positive for all scored rows")

    if not finite or not exposure_positive:
        return _report(False, checks, "Prediction values are not numerically valid", None)

    pred_mean = float(np.mean(predicted))
    pred_std = float(np.std(predicted))
    pred_cv = pred_std / max(abs(pred_mean), 1e-12)
    actual_mean = float(np.mean(actual))
    mean_ratio = pred_mean / max(abs(actual_mean), 1e-12)
    constant_ok = allow_constant_predictions or pred_cv >= rules.min_prediction_cv
    _check(
        checks,
        "predictions_not_accidentally_constant",
        constant_ok,
        f"Prediction coefficient of variation {pred_cv:.6g} is too low",
    )
    _check(
        checks,
        "prediction_mean_reasonable",
        mean_ratio <= rules.max_prediction_to_actual_mean_ratio,
        f"Mean prediction / mean actual ratio {mean_ratio:.6g} is implausibly high",
    )

    panel = full_metric_panel(
        frame["actual_claim_cost"],
        frame["predicted_claim_cost"],
        frame["exposure"],
        tweedie_power=tweedie_power,
    )
    metric_value = panel.get(primary_metric)
    metric_finite = isinstance(metric_value, (int, float)) and np.isfinite(float(metric_value))
    _check(checks, "primary_metric_finite", metric_finite, f"Primary metric {primary_metric!r} is not finite")

    pred_to_actual = panel.get("predicted_to_actual_ratio")
    ratio_ok = (
        isinstance(pred_to_actual, (int, float))
        and np.isfinite(float(pred_to_actual))
        and rules.min_predicted_to_actual_ratio <= float(pred_to_actual) <= rules.max_predicted_to_actual_ratio
    )
    _check(
        checks,
        "predicted_to_actual_ratio_sensible",
        ratio_ok,
        (
            f"Predicted/actual ratio {pred_to_actual} outside "
            f"[{rules.min_predicted_to_actual_ratio}, {rules.max_predicted_to_actual_ratio}]"
        ),
    )

    lift_summary = None
    if champion_predictions is not None:
        lift_summary = _lift_summary(
            champion_predictions,
            predictions,
            eval_split=eval_split,
            primary_metric=primary_metric,
            tweedie_power=tweedie_power,
        )
        positive = bool(lift_summary["lift"] > 0)
        _check(
            checks,
            "positive_lift_vs_champion",
            positive or not rules.require_positive_lift,
            f"Lift vs champion is not positive: {lift_summary['lift']:.6g}",
        )

    valid = all(check["passed"] for check in checks)
    failed = [check["message"] for check in checks if not check["passed"]]
    return {
        "valid": valid,
        "reason": "passed" if valid else "; ".join(failed),
        "checks": checks,
        "metric_panel": panel,
        "lift_summary": lift_summary,
        "rules": {
            "min_predicted_to_actual_ratio": rules.min_predicted_to_actual_ratio,
            "max_predicted_to_actual_ratio": rules.max_predicted_to_actual_ratio,
            "min_prediction_cv": rules.min_prediction_cv,
            "max_prediction_to_actual_mean_ratio": rules.max_prediction_to_actual_mean_ratio,
            "require_positive_lift": rules.require_positive_lift,
        },
    }


def _lift_summary(
    champion_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    *,
    eval_split: str,
    primary_metric: str,
    tweedie_power: float,
) -> dict[str, float | str | bool]:
    champion = champion_predictions[champion_predictions["split"] == eval_split].copy()
    challenger = challenger_predictions[challenger_predictions["split"] == eval_split].copy()
    merged = champion[["record_id", "actual_claim_cost", "exposure", "predicted_claim_cost"]].merge(
        challenger[["record_id", "predicted_claim_cost"]],
        on="record_id",
        how="inner",
        suffixes=("_champion", "_challenger"),
    )
    if merged.empty:
        return {
            "primary_metric": primary_metric,
            "lower_is_better": lower_is_better(primary_metric),
            "champion_score": float("nan"),
            "challenger_score": float("nan"),
            "lift": float("nan"),
            "overlap_rows": 0,
        }
    champion_panel = full_metric_panel(
        merged["actual_claim_cost"],
        merged["predicted_claim_cost_champion"],
        merged["exposure"],
        tweedie_power=tweedie_power,
    )
    challenger_panel = full_metric_panel(
        merged["actual_claim_cost"],
        merged["predicted_claim_cost_challenger"],
        merged["exposure"],
        tweedie_power=tweedie_power,
    )
    champion_score = float(champion_panel[primary_metric])
    challenger_score = float(challenger_panel[primary_metric])
    lift = champion_score - challenger_score if lower_is_better(primary_metric) else challenger_score - champion_score
    return {
        "primary_metric": primary_metric,
        "lower_is_better": lower_is_better(primary_metric),
        "champion_score": champion_score,
        "challenger_score": challenger_score,
        "lift": float(lift),
        "overlap_rows": int(len(merged)),
    }


def _check(checks: list[dict[str, Any]], name: str, passed: bool, message: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "message": "passed" if passed else message})


def _report(valid: bool, checks: list[dict[str, Any]], reason: str, lift_summary: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "valid": valid,
        "reason": reason,
        "checks": checks,
        "metric_panel": None,
        "lift_summary": lift_summary,
        "rules": {},
    }
