"""Calibration and segment diagnostics for insurance target models.

These are mandatory per-run artifacts, not promotion gates themselves.
The promotion gate can check calibration via ``max_predicted_to_actual_drift``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoresearch.evaluation.metrics import infer_target_mode, prediction_target_columns
from autoresearch.targets import target_spec


_SEGMENT_COLS = [
    "risk_score_index_e",
    "vehicle_age_band_c",
    "driver_age_band_d",
    "territory_band_h",
]


def compute_diagnostics(
    predictions: pd.DataFrame,
    eval_split: str,
    *,
    n_deciles: int = 10,
    target_mode: str | None = None,
) -> dict[str, Any]:
    """Compute calibration and segment diagnostics from prediction rows.

    ``predictions`` must have: record_id, split, exposure, active target
    actual/predicted columns, and optionally the segment columns.
    """

    target_mode = infer_target_mode(predictions, target_mode)
    spec = target_spec(target_mode)
    frame = predictions[predictions["split"] == eval_split].copy()
    if frame.empty:
        return {"error": f"No rows for eval split {eval_split!r}"}

    decile_table = _decile_calibration(frame, n_deciles, target_mode=target_mode)
    exposure_bands = _exposure_band_calibration(frame, target_mode=target_mode)
    segment_loss_ratio = _segment_diagnostics(frame, target_mode=target_mode)
    psi = _psi_train_vs_eval(predictions, target_mode=target_mode)

    return {
        "eval_split": eval_split,
        "target_mode": target_mode,
        "rate_label": spec.rate_label,
        "row_count": int(len(frame)),
        "calibration_by_pred_decile": decile_table,
        "calibration_by_exposure_band": exposure_bands,
        "segment_loss_ratio": segment_loss_ratio,
        "psi_train_vs_eval": psi,
        "calibration_pass": _calibration_pass(decile_table),
    }


def _decile_calibration(frame: pd.DataFrame, n_deciles: int, *, target_mode: str) -> list[dict[str, Any]]:
    """Actual vs predicted by active target-rate decile.

    Within each decile, rate values are exposure-weighted
    (sum target / sum exposure), the standard actuarial computation.
    """

    rows = []
    frame = frame.copy()
    exp_safe = frame["exposure"].clip(lower=1e-12)
    actual_col, predicted_col = prediction_target_columns(frame, target_mode)
    spec = target_spec(target_mode)

    if spec.rate_predicted_column in frame.columns:
        sort_col = spec.rate_predicted_column
    else:
        frame["_pred_rate"] = frame[predicted_col] / exp_safe
        sort_col = "_pred_rate"

    try:
        frame["decile"] = pd.qcut(frame[sort_col], n_deciles, labels=False, duplicates="drop") + 1
    except Exception:
        return []

    for decile, grp in frame.groupby("decile", sort=True):
        exp_sum = float(grp["exposure"].clip(lower=1e-12).sum())
        actual_sum = float(grp[actual_col].sum())
        pred_sum = float(grp[predicted_col].sum())
        actual_rate = actual_sum / exp_sum
        pred_rate = pred_sum / exp_sum
        rows.append({
            "decile": int(decile),
            "n": int(len(grp)),
            "exposure": exp_sum,
            "actual_rate": actual_rate,
            "pred_rate": pred_rate,
            "actual_pp": actual_rate,
            "pred_pp": pred_rate,
            "ratio": float(actual_rate / pred_rate) if pred_rate > 0 else float("nan"),
        })
    return rows


def _exposure_band_calibration(frame: pd.DataFrame, *, target_mode: str) -> list[dict[str, Any]]:
    """Actual vs predicted by exposure quantile band."""

    rows = []
    actual_col, predicted_col = prediction_target_columns(frame, target_mode)
    try:
        frame = frame.copy()
        frame["exp_band"] = pd.qcut(frame["exposure"], 5, labels=False, duplicates="drop") + 1
    except Exception:
        return []

    for band, grp in frame.groupby("exp_band", sort=True):
        exp_sum = float(grp["exposure"].clip(lower=1e-12).sum())
        rows.append({
            "exposure_band": int(band),
            "n": int(len(grp)),
            "exposure": exp_sum,
            "actual_rate": float(grp[actual_col].sum() / exp_sum),
            "pred_rate": float(grp[predicted_col].sum() / exp_sum),
            "actual_pp": float(grp[actual_col].sum() / exp_sum),
            "pred_pp": float(grp[predicted_col].sum() / exp_sum),
        })
    return rows


def _segment_diagnostics(frame: pd.DataFrame, *, target_mode: str) -> dict[str, list[dict[str, Any]]]:
    """Loss ratio by known segment columns (if present in predictions)."""

    result = {}
    actual_col, predicted_col = prediction_target_columns(frame, target_mode)
    for col in _SEGMENT_COLS:
        if col not in frame.columns:
            continue
        rows = []
        for val, grp in frame.groupby(col, sort=True):
            exp_sum = float(grp["exposure"].clip(lower=1e-12).sum())
            actual_pp = float(grp[actual_col].sum() / exp_sum)
            pred_pp = float(grp[predicted_col].sum() / exp_sum)
            rows.append({
                "band": str(val),
                "n": int(len(grp)),
                "actual_pp": actual_pp,
                "pred_pp": pred_pp,
                "ratio": float(actual_pp / pred_pp) if pred_pp > 0 else float("nan"),
            })
        if rows:
            result[col] = rows
    return result


def _psi_train_vs_eval(predictions: pd.DataFrame, *, target_mode: str) -> dict[str, float]:
    """Population Stability Index between train and eval splits for key columns."""

    train = predictions[predictions["split"] == "train"]
    eval_rows = predictions[predictions["split"] != "train"]
    if train.empty or eval_rows.empty:
        return {}

    psi_results = {}
    _, predicted_col = prediction_target_columns(predictions, target_mode)
    for col in [predicted_col, "predicted_target", "exposure"]:
        if col not in predictions.columns:
            continue
        try:
            psi_results[col] = float(_psi(train[col].to_numpy(), eval_rows[col].to_numpy()))
        except Exception:
            pass
    return psi_results


def _psi(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index."""

    bins = np.quantile(train, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-9
    bins[-1] += 1e-9
    train_counts = np.histogram(train, bins=bins)[0].astype(float)
    test_counts = np.histogram(test, bins=bins)[0].astype(float)
    train_pct = np.clip(train_counts / train_counts.sum(), 1e-6, None)
    test_pct = np.clip(test_counts / test_counts.sum(), 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def _calibration_pass(decile_table: list[dict[str, Any]], lo: float = 0.3, hi: float = 3.0) -> bool:
    """True if all decile actual/predicted ratios are within [lo, hi]."""

    if not decile_table:
        return False
    return all(lo <= row["ratio"] <= hi for row in decile_table if not np.isnan(row["ratio"]))
