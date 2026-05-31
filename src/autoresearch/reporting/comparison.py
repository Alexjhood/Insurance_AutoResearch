"""HTML reporting for run-scoped experiment comparisons."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.config import ProjectConfig
from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns
from autoresearch.targets import target_spec
from autoresearch.experiment_registry.registry import (
    get_experiment,
    list_artifacts,
    list_champion_history,
    list_experiments,
)
from autoresearch.utils.io import read_json


METRIC_KEYS = [
    # ── Gate metric (rank-based, bounded influence) ──────────────────────────
    "rank_gini_weighted",
    # ── KPI metric (reported discrimination) ────────────────────────────────
    "gini_weighted",
    # ── Additional rank-based metrics ───────────────────────────────────────
    "spearman_rho",
    "kendall_tau",
    "decile_lift_monotonicity",
    # ── Asymmetric Pricing Loss (lower is better, 4:1 under:over penalty) ───
    "asym_pricing_loss",
    "apl_under_cost",
    "apl_over_cost",
    "apl_under_over_ratio",
    # ── Calibration & fit ───────────────────────────────────────────────────
    "predicted_to_actual_ratio",
    "double_lift_slope",
    "tweedie_deviance_p15",
    "poisson_deviance",
    # ── Error metrics (burning cost) ─────────────────────────────────────────
    "weighted_mae_claim_cost",
    "weighted_rmse_claim_cost",
    "rmse_pure_premium",
    "mae_pure_premium",
    "mean_actual_pure_premium",
    "mean_predicted_pure_premium",
    "total_actual_claim_cost",
    "total_predicted_claim_cost",
    # ── Error metrics (frequency) ────────────────────────────────────────────
    "total_actual_claim_count",
    "total_predicted_claim_count",
    "weighted_mae_claim_count",
    "weighted_rmse_claim_count",
    "mean_actual_frequency",
    "mean_predicted_frequency",
    "rmse_frequency",
    "mae_frequency",
]

_BAND_COUNTS = [5, 10, 20, 50]
_DEFAULT_BANDS = 10
_GINI_CURVE_POINTS = 400
_HIST_BINS = 40
# Scatter sample caps per percentage tier (1%, 5%, 25%, 100%)
_SCATTER_CAPS = {"1pct": 800, "5pct": 3000, "25pct": 12000, "100pct": 30000}
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_comparison_html_report(
    *,
    config: ProjectConfig,
    comparison_id: str,
    champion_id: str,
    challenger_id: str,
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    decision: dict[str, Any],
    metric_lift_table: list[dict[str, Any]] | None = None,
    per_partition: Any = None,
    output_path: Path,
) -> Path:
    """Write a self-contained interactive comparison report and return its path."""
    champion_predictions = pd.read_parquet(_artifact_path(config, champion_id, "predictions"))
    challenger_predictions = pd.read_parquet(_artifact_path(config, challenger_id, "predictions"))
    eval_split = config.ordinary_eval_splits[0]
    champion_eval = champion_predictions[champion_predictions["split"] == eval_split].copy()
    challenger_eval = challenger_predictions[challenger_predictions["split"] == eval_split].copy()
    target_mode = config.target_mode
    spec = target_spec(target_mode)
    champ_actual_col, champ_predicted_col = prediction_target_columns(champion_eval, target_mode)
    chall_actual_col, chall_predicted_col = prediction_target_columns(challenger_eval, target_mode)

    champion_metrics = full_metric_panel(
        champion_eval[champ_actual_col],
        champion_eval[champ_predicted_col],
        champion_eval["exposure"],
        tweedie_power=config.tweedie_power,
        target_mode=target_mode,
    )
    challenger_metrics = full_metric_panel(
        challenger_eval[chall_actual_col],
        challenger_eval[chall_predicted_col],
        challenger_eval["exposure"],
        tweedie_power=config.tweedie_power,
        target_mode=target_mode,
    )

    lift_data = {
        "champion": _compute_lift_data(champion_eval, _BAND_COUNTS, target_mode=target_mode),
        "challenger": _compute_lift_data(challenger_eval, _BAND_COUNTS, target_mode=target_mode),
    }
    double_lift_data = _compute_double_lift_data(champion_eval, challenger_eval, _BAND_COUNTS, target_mode=target_mode)
    gini_data = {
        "champion": _compute_gini_curve(champion_eval, _GINI_CURVE_POINTS, target_mode=target_mode),
        "challenger": _compute_gini_curve(challenger_eval, _GINI_CURVE_POINTS, target_mode=target_mode),
    }
    pred_hist_data = {
        "champion": _compute_pred_histogram(champion_eval, _HIST_BINS, target_mode=target_mode),
        "challenger": _compute_pred_histogram(challenger_eval, _HIST_BINS, target_mode=target_mode),
    }
    diff_data = _compute_diff_data(champion_eval, challenger_eval, target_mode=target_mode)

    # The comparison report is generated *before* set_official_champion is called, so
    # the challenger's promotion has not yet been written to champion_history.  Pass the
    # challenger ID as a pending promotion hint so the Gini Progression chart renders it
    # correctly on the first render (when the report is opened immediately).
    pending_promoted = challenger_id if decision.get("decision") == "promote" else None
    history_points = _all_experiment_history_points(config, eval_split, pending_promoted_id=pending_promoted)

    champion_details = _experiment_details(config, champion_id)
    challenger_details = _experiment_details(config, challenger_id)
    champion_timing = _load_timing(config, champion_id)
    challenger_timing = _load_timing(config, challenger_id)

    gate_metric = getattr(config, "gate_primary_metric", "rank_gini_weighted")
    gate_mode = comparison_summary.get("gate_mode", "single_partition")

    html = _render_html(
        comparison_id=comparison_id,
        eval_split=eval_split,
        primary_metric=config.primary_metric,
        gate_primary_metric=gate_metric,
        gate_mode=gate_mode,
        target_mode=target_mode,
        rate_label=spec.rate_label,
        decision=decision,
        comparison_summary=comparison_summary,
        bootstrap_summary=bootstrap_summary,
        champion_id=champion_id,
        challenger_id=challenger_id,
        champion_details=champion_details,
        challenger_details=challenger_details,
        champion_timing=champion_timing,
        challenger_timing=challenger_timing,
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        lift_data=lift_data,
        double_lift_data=double_lift_data,
        gini_data=gini_data,
        pred_hist_data=pred_hist_data,
        diff_data=diff_data,
        history_points=history_points,
        metric_lift_table=metric_lift_table or [],
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Registry / artifact helpers
# ---------------------------------------------------------------------------

def _artifact_path(config: ProjectConfig, experiment_id: str, artifact_type: str) -> Path:
    for artifact in list_artifacts(config.registry_path, experiment_id):
        if artifact["artifact_type"] == artifact_type:
            return Path(artifact["path"])
    raise FileNotFoundError(f"Experiment {experiment_id} has no {artifact_type!r} artifact")


def _load_proposal_for_experiment(config: ProjectConfig, experiment_id: str) -> dict | None:
    """Try to find the proposal.json for an experiment via its config_snapshot path."""
    try:
        snapshot_path = _artifact_path(config, experiment_id, "config_snapshot")
        # Path: iterations/XXX/experiment/attempt_N/config_snapshot.json → go up 3
        iteration_dir = Path(snapshot_path).parent.parent.parent
        proposal_path = iteration_dir / "proposal" / "proposal.json"
        if proposal_path.exists():
            return read_json(proposal_path)
    except Exception:
        pass
    return None


def _load_timing(config: ProjectConfig, experiment_id: str) -> dict[str, Any]:
    """Load the timing block from an experiment's metrics.json, or empty dict."""
    try:
        metrics_path = _artifact_path(config, experiment_id, "metrics")
        metrics = read_json(metrics_path)
        return dict(metrics.get("timing") or {})
    except Exception:
        return {}


def _experiment_details(config: ProjectConfig, experiment_id: str) -> dict[str, Any]:
    experiment = get_experiment(config.registry_path, experiment_id)
    snapshot = read_json(_artifact_path(config, experiment_id, "config_snapshot"))
    proposal = _load_proposal_for_experiment(config, experiment_id)
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment.get("experiment_name"),
        "model_family": experiment.get("model_family"),
        "target_strategy": experiment.get("target_strategy"),
        "target_mode": experiment.get("target_mode") or snapshot.get("target_mode"),
        "mean_score": experiment.get("mean_score"),
        "status": experiment.get("status"),
        "preprocessing": snapshot.get("effective_preprocessing"),
        "model": snapshot.get("experiment", {}).get("model"),
        "rationale": (proposal or {}).get("rationale"),
        "change_summary": (proposal or {}).get("change_summary"),
        "expected_benefit": (proposal or {}).get("expected_benefit"),
        "key_risk": (proposal or {}).get("key_risk"),
    }


def _all_experiment_history_points(
    config: ProjectConfig,
    eval_split: str,
    *,
    pending_promoted_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return all experiments in chronological order with promotion status and label.

    ``pending_promoted_id`` should be set to the challenger_id when the
    comparison report is generated for a promotion decision.  Because the report
    is written *before* ``set_official_champion`` is called, the challenger is
    not yet in ``champion_history``; this parameter lets the chart pre-mark it as
    promoted so the Gini Progression renders correctly on first open.
    """
    promoted_ids: set[str] = set()
    for row in list_champion_history(config.registry_path):
        action = row.get("action", "")
        if action in ("promoted", "initialised"):
            eid = row.get("new_champion_id")
            if eid:
                promoted_ids.add(eid)
    if pending_promoted_id:
        promoted_ids.add(pending_promoted_id)

    # All experiments in reverse-creation order (list_experiments returns newest first)
    all_exps = list(reversed(list_experiments(config.registry_path)))

    points: list[dict[str, Any]] = []
    for exp in all_exps:
        eid = exp.get("experiment_id")
        if not eid:
            continue
        try:
            metrics = read_json(_artifact_path(config, eid, "metrics"))
            split_m = next(
                (item for item in metrics["split_metrics"] if item["split"] == eval_split),
                None,
            )
            if split_m is None:
                continue
            gini = float(split_m["gini_weighted"])
            # Carry all float metrics for the progression dropdown
            all_metrics = {k: round(float(v), 6) for k, v in split_m.items()
                          if isinstance(v, (int, float)) and k not in ("row_count",)}
        except Exception:
            continue
        points.append({
            "step": len(points) + 1,
            "gini": round(gini, 6),
            "metrics": all_metrics,
            "label": exp.get("experiment_name") or eid[:32],
            "promoted": eid in promoted_ids,
            "experiment_id": eid,
        })

    return points or [{"step": 1, "gini": 0.0, "label": "baseline", "promoted": True,
                       "experiment_id": ""}]


# ---------------------------------------------------------------------------
# Chart data computation
# ---------------------------------------------------------------------------

def _equal_exposure_bins(exposure_sorted: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign bin labels 1..n_bins so each bin has ~equal total exposure."""
    if n_bins <= 0 or len(exposure_sorted) == 0:
        return np.ones(len(exposure_sorted), dtype=int)
    cum_exp = np.cumsum(exposure_sorted)
    total_exp = cum_exp[-1]
    if total_exp < 1e-12:
        n = len(exposure_sorted)
        return np.ceil(np.arange(1, n + 1) / n * n_bins).clip(1, n_bins).astype(int)
    return np.ceil(cum_exp / total_exp * n_bins).clip(1, n_bins).astype(int)


def _compute_lift_data(frame: pd.DataFrame, band_counts: list[int], *, target_mode: str) -> dict[str, list[dict]]:
    """
    Sort policies by predicted target rate (ascending = lowest risk first).
    Bin into equal-exposure bands. Per band: actual rate, predicted rate, A/E ratio, exposure.
    """
    exp = frame["exposure"].astype(float).values
    actual_col, predicted_col = prediction_target_columns(frame, target_mode)
    actual_cc = frame[actual_col].astype(float).values
    pred_cc = frame[predicted_col].astype(float).values

    spec = target_spec(target_mode)
    if spec.rate_predicted_column in frame.columns:
        sort_idx = frame[spec.rate_predicted_column].astype(float).argsort().values
    else:
        sort_idx = (frame[predicted_col].astype(float) / frame["exposure"].astype(float).clip(lower=1e-12)).argsort().values
    exp_s = exp[sort_idx]
    actual_s = actual_cc[sort_idx]
    pred_s = pred_cc[sort_idx]

    result: dict[str, list[dict]] = {}
    for n in band_counts:
        labels = _equal_exposure_bins(exp_s, n)
        bands: list[dict] = []
        for b in range(1, n + 1):
            mask = labels == b
            if not mask.any():
                continue
            exp_sum = float(exp_s[mask].sum())
            if exp_sum < 1e-12:
                continue
            actual_pp = float(actual_s[mask].sum()) / exp_sum
            pred_pp = float(pred_s[mask].sum()) / exp_sum
            bands.append({
                "band": b,
                "actual_pp": round(actual_pp, 4),
                "predicted_pp": round(pred_pp, 4),
                "ae_ratio": round(actual_pp / max(pred_pp, 1e-9), 4),
                "exposure": round(exp_sum, 2),
                "n_policies": int(mask.sum()),
            })
        result[str(n)] = bands
    return result


def _compute_double_lift_data(
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
    band_counts: list[int],
    *,
    target_mode: str,
) -> dict[str, dict[str, list[dict]]]:
    """
    Returns {"equal_exposure": {n: [bands]}, "equal_width": {n: [bands]}}.

    equal_exposure: bins by equal total exposure; x = band ordinal (1..N).
    equal_width: bins by equal-width ratio intervals; x = ratio midpoint.
    """
    actual_col, champion_predicted_col = prediction_target_columns(champion, target_mode)
    _, challenger_predicted_col = prediction_target_columns(challenger, target_mode)
    paired = champion[["record_id", actual_col, champion_predicted_col, "exposure"]].merge(
        challenger[["record_id", challenger_predicted_col]],
        on="record_id",
        suffixes=("_champ", "_chall"),
    ).copy()

    exp = paired["exposure"].astype(float).values
    actual_cc = paired[actual_col].astype(float).values
    champ_col = f"{champion_predicted_col}_champ" if champion_predicted_col == challenger_predicted_col else champion_predicted_col
    chall_col = f"{challenger_predicted_col}_chall" if champion_predicted_col == challenger_predicted_col else challenger_predicted_col
    champ_cc = paired[champ_col].astype(float).values
    chall_cc = paired[chall_col].astype(float).values

    champ_pp_pol = champ_cc / exp.clip(min=1e-12)
    chall_pp_pol = chall_cc / exp.clip(min=1e-12)
    ratio = chall_pp_pol / champ_pp_pol.clip(min=1e-9)

    # ── Equal-exposure ─────────────────────────────────────────────────────
    sort_idx = ratio.argsort()
    exp_s = exp[sort_idx]
    actual_s = actual_cc[sort_idx]
    champ_s = champ_cc[sort_idx]
    chall_s = chall_cc[sort_idx]
    ratio_s = ratio[sort_idx]

    ee_result: dict[str, list[dict]] = {}
    for n in band_counts:
        labels = _equal_exposure_bins(exp_s, n)
        bands: list[dict] = []
        for b in range(1, n + 1):
            mask = labels == b
            if not mask.any():
                continue
            exp_sum = float(exp_s[mask].sum())
            if exp_sum < 1e-12:
                continue
            actual_pp = float(actual_s[mask].sum()) / exp_sum
            champ_pp = float(champ_s[mask].sum()) / exp_sum
            chall_pp = float(chall_s[mask].sum()) / exp_sum
            rv = ratio_s[mask]
            bands.append({
                "band": b,
                "x": b,  # ordinal for equal-exposure x-axis
                "ratio_mean": round(float(rv.mean()), 4),
                "ratio_min": round(float(rv.min()), 4),
                "ratio_max": round(float(rv.max()), 4),
                "actual_pp": round(actual_pp, 4),
                "champion_pp": round(champ_pp, 4),
                "challenger_pp": round(chall_pp, 4),
                "exposure": round(exp_sum, 2),
                "n_policies": int(mask.sum()),
            })
        ee_result[str(n)] = bands

    # ── Equal-width ────────────────────────────────────────────────────────
    r_lo = float(np.percentile(ratio, 2))
    r_hi = float(np.percentile(ratio, 98))
    if r_hi <= r_lo:
        r_hi = r_lo + 0.01

    ew_result: dict[str, list[dict]] = {}
    for n in band_counts:
        edges = np.linspace(r_lo, r_hi, n + 1)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        bin_idx = np.digitize(ratio, edges) - 1
        bin_idx = bin_idx.clip(0, n - 1)
        bands = []
        for b in range(n):
            mask = bin_idx == b
            if not mask.any():
                continue
            exp_sum = float(exp[mask].sum())
            if exp_sum < 1e-12:
                continue
            actual_pp = float(actual_cc[mask].sum()) / exp_sum
            champ_pp = float(champ_cc[mask].sum()) / exp_sum
            chall_pp = float(chall_cc[mask].sum()) / exp_sum
            rv = ratio[mask]
            bands.append({
                "band": b + 1,
                "x": round(float(midpoints[b]), 4),  # actual ratio for x-axis
                "ratio_mean": round(float(rv.mean()), 4),
                "ratio_min": round(float(rv.min()), 4),
                "ratio_max": round(float(rv.max()), 4),
                "actual_pp": round(actual_pp, 4),
                "champion_pp": round(champ_pp, 4),
                "challenger_pp": round(chall_pp, 4),
                "exposure": round(exp_sum, 2),
                "n_policies": int(mask.sum()),
            })
        ew_result[str(n)] = bands

    return {"equal_exposure": ee_result, "equal_width": ew_result}


def _compute_gini_curve(frame: pd.DataFrame, n_points: int = 400, *, target_mode: str) -> dict:
    """Lorenz curve sorted by predicted target rate ascending."""
    f = frame.copy()
    actual_col, predicted_col = prediction_target_columns(f, target_mode)
    f["_pred_pp"] = f[predicted_col].astype(float) / f["exposure"].astype(float).clip(lower=1e-12)
    ordered = f.sort_values("_pred_pp", ascending=True)
    exposure = ordered["exposure"].astype(float).values
    actual_cost = ordered[actual_col].astype(float).values

    total_exp = float(exposure.sum())
    total_cost = float(actual_cost.sum())

    if total_exp < 1e-12 or total_cost < 1e-12:
        return {"x": [0.0, 1.0], "y": [0.0, 1.0], "gini": 0.0}

    cum_exp = np.concatenate([[0.0], np.cumsum(exposure) / total_exp])
    cum_cost = np.concatenate([[0.0], np.cumsum(actual_cost) / total_cost])

    lorenz_area = float(np.sum(0.5 * (cum_cost[1:] + cum_cost[:-1]) * np.diff(cum_exp)))
    gini = round(float(1.0 - 2.0 * lorenz_area), 6)

    if len(cum_exp) > n_points:
        idx = np.round(np.linspace(0, len(cum_exp) - 1, n_points)).astype(int)
        cum_exp, cum_cost = cum_exp[idx], cum_cost[idx]

    x = [round(float(v), 6) for v in cum_exp]
    y = [round(float(v), 6) for v in cum_cost]
    if x[0] != 0.0:
        x, y = [0.0] + x, [0.0] + y
    if x[-1] != 1.0:
        x, y = x + [1.0], y + [1.0]

    return {"x": x, "y": y, "gini": gini}


def _compute_pred_histogram(frame: pd.DataFrame, n_bins: int = 40, *, target_mode: str) -> dict:
    """Exposure-weighted histogram of predicted target rates (clipped at 1st/99th pct)."""
    spec = target_spec(target_mode)
    _, predicted_col = prediction_target_columns(frame, target_mode)
    if spec.rate_predicted_column in frame.columns:
        pp = frame[spec.rate_predicted_column].astype(float).values
    else:
        pp = (frame[predicted_col].astype(float) / frame["exposure"].astype(float).clip(lower=1e-12)).values
    exp = frame["exposure"].astype(float).values
    p1, p99 = float(np.percentile(pp, 1)), float(np.percentile(pp, 99))
    if p1 >= p99:
        p99 = p1 + 1.0
    edges = np.linspace(p1, p99, n_bins + 1)
    counts, _ = np.histogram(pp, bins=edges, weights=exp)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "x": [round(float(v), 4) for v in centers],
        "y": [round(float(v), 4) for v in counts],
    }


def _compute_diff_data(
    champion_eval: pd.DataFrame, challenger_eval: pd.DataFrame, *, target_mode: str
) -> dict:
    """
    Compute per-policy predicted target-rate differences between challenger and champion.
    Returns exposure-weighted % diff histogram and a sampled scatter.
    """
    _, champion_predicted_col = prediction_target_columns(champion_eval, target_mode)
    _, challenger_predicted_col = prediction_target_columns(challenger_eval, target_mode)
    paired = champion_eval[["record_id", champion_predicted_col, "exposure"]].merge(
        challenger_eval[["record_id", challenger_predicted_col]],
        on="record_id",
        suffixes=("_champ", "_chall"),
    )
    exp = paired["exposure"].astype(float).values
    champ_col = f"{champion_predicted_col}_champ" if champion_predicted_col == challenger_predicted_col else champion_predicted_col
    chall_col = f"{challenger_predicted_col}_chall" if champion_predicted_col == challenger_predicted_col else challenger_predicted_col
    champ_pp = paired[champ_col].astype(float).values / exp.clip(min=1e-12)
    chall_pp = paired[chall_col].astype(float).values / exp.clip(min=1e-12)

    pct_diff = (chall_pp - champ_pp) / champ_pp.clip(min=1e-9) * 100.0

    # Exposure-weighted % diff histogram
    p2, p98 = float(np.percentile(pct_diff, 2)), float(np.percentile(pct_diff, 98))
    if p2 >= p98:
        p98 = p2 + 1.0
    edges = np.linspace(p2, p98, _HIST_BINS + 1)
    counts, _ = np.histogram(pct_diff, bins=edges, weights=exp)
    centers = 0.5 * (edges[:-1] + edges[1:])
    pct_hist = {
        "x": [round(float(v), 4) for v in centers],
        "y": [round(float(v), 4) for v in counts],
    }

    # 99th-percentile axis bound (computed from full data, not a sample)
    p99 = float(np.percentile(np.concatenate([champ_pp, chall_pp]), 99))

    # Scatter: four sample sizes — deterministic subsampling from the same pool
    rng = np.random.default_rng(seed=42)
    n = len(paired)
    # Draw the largest pool first; smaller samples are prefixes of it
    largest_cap = max(_SCATTER_CAPS.values())
    full_pool_idx = rng.choice(n, min(n, largest_cap), replace=False)

    scatter_samples: dict[str, dict] = {}
    for label, cap in _SCATTER_CAPS.items():
        k = min(len(full_pool_idx), cap)
        idx = full_pool_idx[:k]
        scatter_samples[label] = {
            "x": [round(float(v), 2) for v in champ_pp[idx]],
            "y": [round(float(v), 2) for v in chall_pp[idx]],
            "n": k,
        }

    return {
        "pct_hist": pct_hist,
        "scatter": scatter_samples["25pct"],   # backward-compat default
        "scatter_samples": scatter_samples,
        "scatter_p99": round(p99, 2),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(
    *,
    comparison_id: str,
    eval_split: str,
    primary_metric: str,
    gate_primary_metric: str = "rank_gini_weighted",
    gate_mode: str = "single_partition",
    target_mode: str,
    rate_label: str,
    decision: dict[str, Any],
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    champion_id: str,
    challenger_id: str,
    champion_details: dict[str, Any],
    challenger_details: dict[str, Any],
    champion_timing: dict[str, Any] | None = None,
    challenger_timing: dict[str, Any] | None = None,
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    lift_data: dict,
    double_lift_data: dict,
    gini_data: dict,
    pred_hist_data: dict,
    diff_data: dict,
    history_points: list,
    metric_lift_table: list[dict[str, Any]] | None = None,
) -> str:
    decision_str = decision.get("decision", "?")
    # pending_llm → amber; promote/promoted → green; reject/rejected → red
    if decision_str in ("promote", "promoted"):
        decision_color, decision_bg = "#0a5c2e", "#e8f5e9"
    elif decision_str in ("reject", "rejected"):
        decision_color, decision_bg = "#7a1a1a", "#fce8e8"
    elif decision_str == "pending_llm":
        decision_color, decision_bg = "#5a3a00", "#fff8e1"
    else:
        decision_color, decision_bg = "#5a4500", "#fffde7"

    # Human-readable decision label
    decision_label = {
        "pending_llm": "AWAITING LLM DECISION",
        "promote": "PROMOTED",
        "promoted": "PROMOTED",
        "reject": "REJECTED",
        "rejected": "REJECTED",
        "inconclusive": "INCONCLUSIVE",
    }.get(decision_str, decision_str.upper())

    # Guardrail status
    guardrail_passed = decision.get("guardrail_passed", True)
    guardrail_failures = decision.get("guardrail_failures", [])
    advisory_str = decision.get("advisory_decision", "")
    decided_by = decision.get("decided_by", "")
    decided_at = decision.get("decided_at", "")

    champ_gini = gini_data["champion"]["gini"]
    chall_gini = gini_data["challenger"]["gini"]
    mean_lift = float(comparison_summary.get("mean_lift") or 0)
    win_rate = float(comparison_summary.get("challenger_win_rate") or 0)
    lift_color = "#0a5c2e" if mean_lift > 0 else "#7a1a1a"
    win_color = "#0a5c2e" if win_rate >= 0.6 else "#7a1a1a"

    # Label for how many partitions / samples the lift distribution is drawn from
    if gate_mode == "cv_bootstrap":
        n_parts = int(comparison_summary.get("n_partitions", 1))
        n_folds_cv = int(comparison_summary.get("n_folds", 4))
        bpf = int(comparison_summary.get("bootstrap_per_fold", 20))
        n_total = int(comparison_summary.get("n_samples", n_parts * n_folds_cv * bpf))
        escalated = comparison_summary.get("escalated", False)
        esc_label = " (escalated)" if escalated else ""
        n_partitions_label = (
            f"{n_parts} partition{'s' if n_parts > 1 else ''}{esc_label} × "
            f"{n_folds_cv} folds × {bpf} bootstrap = {n_total} samples"
        )
    elif gate_mode == "repeated_cv":
        n_folds = int(comparison_summary.get("n_folds", 4))
        n_repeats = int(comparison_summary.get("n_repeats", 4))
        n_partitions_label = f"{n_folds}×{n_repeats} = {n_folds * n_repeats} CV partitions"
    else:
        n_resamples = int(comparison_summary.get("n_resamples", 30))
        n_partitions_label = f"{n_resamples} bootstrap resamples"

    _metric_table = metric_lift_table or []
    data_script = (
        "const LIFT=" + json.dumps(lift_data, separators=(",", ":")) + ";"
        "const DL=" + json.dumps(double_lift_data, separators=(",", ":")) + ";"
        "const GINI=" + json.dumps(gini_data, separators=(",", ":")) + ";"
        "const HIST=" + json.dumps(pred_hist_data, separators=(",", ":")) + ";"
        "const DIFF=" + json.dumps(diff_data, separators=(",", ":")) + ";"
        "const HISTORY=" + json.dumps(history_points, separators=(",", ":")) + ";"
        "const METRIC_TABLE=" + json.dumps(_metric_table, separators=(",", ":")) + ";"
        f"const GATE_METRIC={json.dumps(gate_primary_metric)};"
        f"const GATE_MODE={json.dumps(gate_mode)};"
    )

    band_options = "\n".join(
        f'<option value="{n}"{" selected" if n == _DEFAULT_BANDS else ""}>{n} bands</option>'
        for n in _BAND_COUNTS
    )

    metrics_html = _metrics_table(champion_metrics, challenger_metrics)
    summary_html = _summary_table(comparison_summary, bootstrap_summary)
    gate_html = _gate_table(decision, comparison_summary, bootstrap_summary, challenger_metrics)
    champ_details_html = _details_card("Champion", champion_details)
    chall_details_html = _details_card("Challenger", challenger_details)
    diff_discussion_html = _experiment_diff_discussion(champion_details, challenger_details)
    native_calib_html = _native_calibration_warning(challenger_details)
    compute_html = _compute_section(champion_timing or {}, challenger_timing or {})

    js_code = r"""
const CHAMP_COLOR='#1f77b4',CHALL_COLOR='#d62728',ACTUAL_COLOR='#2ca02c';
const PROMO_COLOR='#0a5c2e',FAILED_COLOR='#d62728',EXP_BAR_COLOR='rgba(180,180,180,0.35)';
const CFG={displayModeBar:true,modeBarButtonsToRemove:['select2d','lasso2d','toggleSpikelines'],displaylogo:false,responsive:true};
const BASE_LAYOUT={
  margin:{l:68,r:68,t:52,b:68},
  font:{family:'-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',size:12},
  plot_bgcolor:'#fff',paper_bgcolor:'#fff',
  legend:{orientation:'h',yanchor:'bottom',y:1.02,xanchor:'right',x:1,
    bgcolor:'rgba(255,255,255,0.92)',bordercolor:'#dee2e6',borderwidth:1},
  xaxis:{gridcolor:'#f0f0f0',zerolinecolor:'#dee2e6',linecolor:'#adb5bd'},
  yaxis:{gridcolor:'#f0f0f0',zerolinecolor:'#dee2e6',linecolor:'#adb5bd'},
};

function mkLayout(ov){
  return Object.assign({},BASE_LAYOUT,ov,{
    xaxis:Object.assign({},BASE_LAYOUT.xaxis,ov.xaxis||{}),
    yaxis:Object.assign({},BASE_LAYOUT.yaxis,ov.yaxis||{}),
  });
}

function expBarTrace(bands, xField){
  return {
    name:'Exposure',x:bands.map(d=>d[xField]),y:bands.map(d=>d.exposure),
    type:'bar',yaxis:'y2',marker:{color:EXP_BAR_COLOR},
    hovertemplate:'%{x}<br>Exposure: %{y:,.1f}<extra>Exposure</extra>',
    showlegend:true,
  };
}

function expAxis(){
  return {
    title:'Exposure',overlaying:'y',side:'right',showgrid:false,
    tickformat:',.0f',color:'#999',zeroline:false,
  };
}

/* ── Lift curves ─────────────────────────────────────────────────────── */
function renderLift(){
  const n=document.getElementById('lift-bands').value;
  const mode=document.querySelector('input[name="lift-mode"]:checked').value;
  const champ=LIFT.champion[n]||[];
  const chall=LIFT.challenger[n]||[];
  let traces,layout;
  const xField='band';

  if(mode==='absolute'){
    traces=[
      expBarTrace(chall, xField),
      {name:'Champion — Predicted',
       x:champ.map(d=>d[xField]),y:champ.map(d=>d.predicted_pp),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
       customdata:champ.map(d=>[d.n_policies,d.exposure,d.ae_ratio]),
       hovertemplate:'Band %{x}<br>Predicted PP: £%{y:,.2f}<br>A/E: %{customdata[2]:.3f}<br>Policies: %{customdata[0]:.0f}<br>Exposure: %{customdata[1]:.1f}<extra>Champion Predicted</extra>'},
      {name:'Champion — Actual',
       x:champ.map(d=>d[xField]),y:champ.map(d=>d.actual_pp),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2,dash:'dash'},marker:{size:6,symbol:'circle-open'},
       hovertemplate:'Band %{x}<br>Actual PP: £%{y:,.2f}<extra>Champion Actual</extra>'},
      {name:'Challenger — Predicted',
       x:chall.map(d=>d[xField]),y:chall.map(d=>d.predicted_pp),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       customdata:chall.map(d=>[d.n_policies,d.exposure,d.ae_ratio]),
       hovertemplate:'Band %{x}<br>Predicted PP: £%{y:,.2f}<br>A/E: %{customdata[2]:.3f}<br>Policies: %{customdata[0]:.0f}<br>Exposure: %{customdata[1]:.1f}<extra>Challenger Predicted</extra>'},
      {name:'Challenger — Actual',
       x:chall.map(d=>d[xField]),y:chall.map(d=>d.actual_pp),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2,dash:'dash'},marker:{size:6,symbol:'circle-open'},
       hovertemplate:'Band %{x}<br>Actual PP: £%{y:,.2f}<extra>Challenger Actual</extra>'},
    ];
    layout=mkLayout({
      title:{text:'Actual vs Predicted Pure Premium by Risk Band',font:{size:14}},
      xaxis:{title:'Risk Band  (1 = lowest predicted risk → N = highest)',dtick:1},
      yaxis:{title:'Pure Premium (£)'},
      yaxis2:expAxis(),
    });
  } else if(mode==='ae'){
    const maxBand=Math.max(champ.length,chall.length,1);
    traces=[
      expBarTrace(chall, xField),
      {name:'Champion A/E',
       x:champ.map(d=>d[xField]),y:champ.map(d=>d.ae_ratio),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
       customdata:champ.map(d=>[d.actual_pp,d.predicted_pp,d.n_policies,d.exposure]),
       hovertemplate:'Band %{x}<br>A/E: %{y:.3f}<br>Actual PP: £%{customdata[0]:,.2f}<br>Predicted PP: £%{customdata[1]:,.2f}<extra>Champion</extra>'},
      {name:'Challenger A/E',
       x:chall.map(d=>d[xField]),y:chall.map(d=>d.ae_ratio),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       customdata:chall.map(d=>[d.actual_pp,d.predicted_pp,d.n_policies,d.exposure]),
       hovertemplate:'Band %{x}<br>A/E: %{y:.3f}<br>Actual PP: £%{customdata[0]:,.2f}<br>Predicted PP: £%{customdata[1]:,.2f}<extra>Challenger</extra>'},
      {name:'Perfect Calibration (1.0)',
       x:[1,maxBand],y:[1,1],mode:'lines',
       line:{color:'#868e96',width:1.5,dash:'dot'},hoverinfo:'skip'},
    ];
    layout=mkLayout({
      title:{text:'Actual / Expected (A/E) Ratio by Risk Band',font:{size:14}},
      xaxis:{title:'Risk Band  (1 = lowest predicted risk → N = highest)',dtick:1},
      yaxis:{title:'A/E Ratio  (Actual PP ÷ Predicted PP)'},
      yaxis2:expAxis(),
    });
  } else {
    // Rescaled: normalise all values by the exposure-weighted mean champion predicted PP.
    // Champion predicted line crosses 1.0 at the median band; all lines express
    // "multiples of average champion rate" — useful for comparing absolute scale.
    const champTotalExp=champ.reduce((s,d)=>s+d.exposure,0);
    const champMeanPred=champTotalExp>0
      ? champ.reduce((s,d)=>s+d.predicted_pp*d.exposure,0)/champTotalExp : 1.0;
    const sc=v=>v/Math.max(champMeanPred,1e-9);
    traces=[
      expBarTrace(chall, xField),
      {name:'Champion — Predicted',
       x:champ.map(d=>d[xField]),y:champ.map(d=>sc(d.predicted_pp)),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
       hovertemplate:'Band %{x}<br>Rel. PP: %{y:.3f}× avg<extra>Champion Predicted</extra>'},
      {name:'Champion — Actual',
       x:champ.map(d=>d[xField]),y:champ.map(d=>sc(d.actual_pp)),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2,dash:'dash'},marker:{size:6,symbol:'circle-open'},
       hovertemplate:'Band %{x}<br>Rel. PP: %{y:.3f}× avg<extra>Champion Actual</extra>'},
      {name:'Challenger — Predicted',
       x:chall.map(d=>d[xField]),y:chall.map(d=>sc(d.predicted_pp)),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       hovertemplate:'Band %{x}<br>Rel. PP: %{y:.3f}× avg<extra>Challenger Predicted</extra>'},
      {name:'Challenger — Actual',
       x:chall.map(d=>d[xField]),y:chall.map(d=>sc(d.actual_pp)),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2,dash:'dash'},marker:{size:6,symbol:'circle-open'},
       hovertemplate:'Band %{x}<br>Rel. PP: %{y:.3f}× avg<extra>Challenger Actual</extra>'},
      {name:'Champion mean (1.0)',
       x:[1,champ.length],y:[1,1],mode:'lines',
       line:{color:'#868e96',width:1.5,dash:'dot'},hoverinfo:'skip'},
    ];
    layout=mkLayout({
      title:{text:'Lift Curves — Rescaled to Champion Mean PP (1.0 = champion average)',font:{size:14}},
      xaxis:{title:'Risk Band  (1 = lowest predicted risk → N = highest)',dtick:1},
      yaxis:{title:'PP Relative to Champion Mean  (×)'},
      yaxis2:expAxis(),
    });
  }
  Plotly.react('lift-chart',traces,layout,CFG);
}

/* ── Double lift ─────────────────────────────────────────────────────── */
function renderDoubleLift(){
  const n=document.getElementById('dl-bands').value;
  const dlMode=document.querySelector('input[name="dl-mode"]:checked').value;
  const rescale=document.getElementById('dl-rescale').checked;
  const bands=(DL[dlMode]||DL.equal_exposure)[n]||[];
  const isEW=(dlMode==='equal_width');
  const xField='x';
  const xTitle=isEW?'Challenger / Champion Predicted PP Ratio':'Band  (1 = challenger predicts lowest relative to champion)';

  let traces,yTitle;
  if(!rescale){
    yTitle='Pure Premium (£)';
    traces=[
      expBarTrace(bands, xField),
      {name:'Actual',
       x:bands.map(d=>d[xField]),y:bands.map(d=>d.actual_pp),
       mode:'lines+markers',line:{color:ACTUAL_COLOR,width:2.5,dash:'dash'},
       marker:{size:8,symbol:'diamond'},
       customdata:bands.map(d=>[d.ratio_min,d.ratio_max,d.n_policies,d.exposure]),
       hovertemplate:(isEW?'Ratio %{x:.3f}×':'Band %{x}')+'<br>Actual PP: £%{y:,.2f}<br>Policies: %{customdata[2]:.0f}<extra>Actual</extra>'},
      {name:'Champion Predicted',
       x:bands.map(d=>d[xField]),y:bands.map(d=>d.champion_pp),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
       hovertemplate:(isEW?'Ratio %{x:.3f}×':'Band %{x}')+'<br>Champion PP: £%{y:,.2f}<extra>Champion</extra>'},
      {name:'Challenger Predicted',
       x:bands.map(d=>d[xField]),y:bands.map(d=>d.challenger_pp),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       hovertemplate:(isEW?'Ratio %{x:.3f}×':'Band %{x}')+'<br>Challenger PP: £%{y:,.2f}<extra>Challenger</extra>'},
    ];
  } else {
    // Rescaled: per-band divide by champion_pp → champion = 1.0 flat line.
    // Challenger shows challenger/champion ratio; actual shows actual/champion.
    yTitle='Relative to Champion Predicted  (1.0 = champion level)';
    const nBands=bands.length;
    traces=[
      expBarTrace(bands, xField),
      {name:'Actual ÷ Champion',
       x:bands.map(d=>d[xField]),y:bands.map(d=>d.actual_pp/Math.max(d.champion_pp,1e-9)),
       mode:'lines+markers',line:{color:ACTUAL_COLOR,width:2.5,dash:'dash'},
       marker:{size:8,symbol:'diamond'},
       customdata:bands.map(d=>[d.actual_pp,d.champion_pp,d.n_policies]),
       hovertemplate:(isEW?'Ratio %{x:.3f}×':'Band %{x}')+'<br>Actual/Champ: %{y:.3f}×<br>Actual PP: £%{customdata[0]:,.2f}<extra>Actual</extra>'},
      {name:'Champion (1.0)',
       x:bands.map(d=>d[xField]),y:bands.map(()=>1.0),
       mode:'lines',line:{color:CHAMP_COLOR,width:2,dash:'dot'},
       hoverinfo:'skip'},
      {name:'Challenger ÷ Champion',
       x:bands.map(d=>d[xField]),y:bands.map(d=>d.challenger_pp/Math.max(d.champion_pp,1e-9)),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       customdata:bands.map(d=>[d.challenger_pp,d.champion_pp]),
       hovertemplate:(isEW?'Ratio %{x:.3f}×':'Band %{x}')+'<br>Chall/Champ: %{y:.3f}×<br>Challenger PP: £%{customdata[0]:,.2f}<extra>Challenger</extra>'},
    ];
  }
  Plotly.react('double-lift-chart',traces,mkLayout({
    title:{text:'Double Lift Curve  (bands sorted by Challenger ÷ Champion ratio)',font:{size:14}},
    xaxis:{title:xTitle},
    yaxis:{title:yTitle},
    yaxis2:expAxis(),
  }),CFG);
}

/* ── Gini Lorenz curves ──────────────────────────────────────────────── */
function renderGini(){
  const c=GINI.champion,h=GINI.challenger;
  const traces=[
    {name:`Champion  (Gini = ${c.gini.toFixed(4)})`,
     x:c.x,y:c.y,mode:'lines',line:{color:CHAMP_COLOR,width:2.5},
     hovertemplate:'Cum. exposure: %{x:.1%}<br>Cum. claim cost: %{y:.1%}<extra>Champion</extra>'},
    {name:`Challenger  (Gini = ${h.gini.toFixed(4)})`,
     x:h.x,y:h.y,mode:'lines',line:{color:CHALL_COLOR,width:2.5},
     hovertemplate:'Cum. exposure: %{x:.1%}<br>Cum. claim cost: %{y:.1%}<extra>Challenger</extra>'},
    {name:'Random (Gini = 0)',
     x:[0,1],y:[0,1],mode:'lines',
     line:{color:'#adb5bd',width:1.5,dash:'dot'},hoverinfo:'skip'},
  ];
  Plotly.react('gini-chart',traces,mkLayout({
    title:{text:'Lorenz Curves — Exposure-Weighted Gini',font:{size:14}},
    xaxis:{title:'Cumulative Exposure Share  (policies ranked by predicted risk, lowest first)',
           tickformat:'.0%',range:[0,1]},
    yaxis:{title:'Cumulative Actual Claim Cost Share',tickformat:'.0%',range:[0,1]},
  }),CFG);
}

/* ── Prediction distribution ─────────────────────────────────────────── */
function renderHist(){
  const view=document.querySelector('input[name="hist-view"]:checked').value;
  if(view==='dist'){
    const c=HIST.champion,h=HIST.challenger;
    const traces=[
      {name:'Champion',x:c.x,y:c.y,type:'bar',
       marker:{color:CHAMP_COLOR,opacity:0.55},
       hovertemplate:'PP: £%{x:,.2f}<br>Exposure: %{y:,.1f}<extra>Champion</extra>'},
      {name:'Challenger',x:h.x,y:h.y,type:'bar',
       marker:{color:CHALL_COLOR,opacity:0.55},
       hovertemplate:'PP: £%{x:,.2f}<br>Exposure: %{y:,.1f}<extra>Challenger</extra>'},
    ];
    Plotly.react('hist-chart',traces,mkLayout({
      barmode:'overlay',
      title:{text:'Predicted Pure Premium Distribution  (exposure-weighted)',font:{size:14}},
      xaxis:{title:'Predicted Pure Premium (£)'},
      yaxis:{title:'Total Exposure'},
    }),CFG);
  } else if(view==='pctdiff'){
    const d=DIFF.pct_hist;
    const traces=[{
      name:'% Difference',x:d.x,y:d.y,type:'bar',
      marker:{color:'#9467bd',opacity:0.7},
      hovertemplate:'%{x:.1f}%<br>Exposure: %{y:,.1f}<extra>% Diff</extra>',
    }];
    Plotly.react('hist-chart',traces,mkLayout({
      title:{text:'Distribution of Predicted PP Differences  (Challenger − Champion) / Champion',font:{size:14}},
      xaxis:{title:'% Difference  (positive = challenger predicts higher)'},
      yaxis:{title:'Total Exposure'},
      shapes:[{type:'line',x0:0,x1:0,y0:0,y1:1,yref:'paper',
               line:{color:'#868e96',width:1.5,dash:'dot'}}],
    }),CFG);
  } else {
    const pctKey=document.querySelector('input[name="scatter-pct"]:checked').value;
    const s=DIFF.scatter_samples[pctKey]||DIFF.scatter;
    const axMax=(DIFF.scatter_p99||Math.max(...s.x,...s.y))*1.02;
    const traces=[
      {name:'Policy (n='+s.n+')',x:s.x,y:s.y,mode:'markers',type:'scatter',
       marker:{color:CHALL_COLOR,opacity:0.25,size:3},
       hovertemplate:'Champion PP: £%{x:,.2f}<br>Challenger PP: £%{y:,.2f}<extra></extra>'},
      {name:'Equal prediction',x:[0,axMax],y:[0,axMax],mode:'lines',
       line:{color:'#868e96',width:1.5,dash:'dot'},hoverinfo:'skip'},
    ];
    Plotly.react('hist-chart',traces,mkLayout({
      title:{text:'Challenger vs Champion Predicted Pure Premium  (per-policy)',font:{size:14}},
      xaxis:{title:'Champion Predicted PP (£)',range:[0,axMax]},
      yaxis:{title:'Challenger Predicted PP (£)',range:[0,axMax]},
    }),CFG);
  }
}

/* ── Multi-metric exhibit ────────────────────────────────────────────── */
function renderMetricExhibit(){
  const sel=document.getElementById('metric-select');
  if(!sel||METRIC_TABLE.length===0){
    if(document.getElementById('metric-exhibit-chart'))
      Plotly.react('metric-exhibit-chart',[],{title:{text:'No metric data available'}},{});
    return;
  }
  const chosen=sel.value;
  const row=METRIC_TABLE.find(r=>r.metric===chosen);
  if(!row){return;}

  const hib=row.higher_is_better;
  const liftSign=row.mean_lift>0?'▲':'▼';
  const liftColor=row.mean_lift>0?'#0a5c2e':'#7a1a1a';
  document.getElementById('metric-lift-display').innerHTML=
    `<span style="color:${liftColor};font-weight:700">${liftSign} ${row.mean_lift>0?'+':''}${row.mean_lift.toFixed(4)}</span>`+
    (row.lift_std!=null?` <span style="color:#6c757d;font-size:12px">(σ ${row.lift_std.toFixed(4)})</span>`:'');

  const traces=[];

  // ── Left subplot: champion vs challenger score bars ───────────────────
  const scores=[
    {label:'Champion',val:row.champion_score,color:CHAMP_COLOR},
    {label:'Challenger',val:row.challenger_score,color:CHALL_COLOR},
  ];
  traces.push({
    x:scores.map(d=>d.label),y:scores.map(d=>d.val),
    type:'bar',marker:{color:scores.map(d=>d.color)},
    xaxis:'x',yaxis:'y',name:'Score',
    hovertemplate:'%{x}: %{y:.5g}<extra></extra>',showlegend:false,
  });

  // ── Right subplot: lift distribution (box + strip if partitions available) ──
  const lifts=row.per_partition_lifts||[];
  if(lifts.length>0){
    traces.push({
      x:lifts.map(()=>'Lift across<br>partitions'),
      y:lifts,
      type:'box',
      boxpoints:'all',jitter:0.4,pointpos:0,
      marker:{color:liftColor,size:5,opacity:0.6},
      line:{color:liftColor},
      fillcolor:liftColor.replace(')',',0.15)').replace('rgb','rgba'),
      xaxis:'x2',yaxis:'y2',
      name:'Lift distribution',showlegend:false,
      hovertemplate:'Lift: %{y:.5g}<extra></extra>',
    });
    // Zero reference line
    traces.push({
      x:['Lift across<br>partitions','Lift across<br>partitions'],
      y:[0,0],type:'scatter',mode:'lines',
      line:{color:'#868e96',width:1.5,dash:'dot'},
      xaxis:'x2',yaxis:'y2',hoverinfo:'skip',showlegend:false,name:'',
    });
  }

  const dirLabel=hib?'↑ higher is better':'↓ lower is better';
  const layout=Object.assign({},BASE_LAYOUT,{
    title:{text:`<b>${chosen}</b>  <span style="font-size:11px;color:#6c757d">${dirLabel}</span>`,font:{size:13}},
    grid:{rows:1,columns:lifts.length>0?2:1,pattern:'independent'},
    xaxis:{domain:[0,lifts.length>0?0.44:1],title:'Model'},
    yaxis:{title:chosen,gridcolor:'#f0f0f0'},
    xaxis2:{domain:[0.56,1]},
    yaxis2:{title:'Lift (challenger − champion)',gridcolor:'#f0f0f0',zeroline:true,zerolinecolor:'#aaa'},
    margin:{l:68,r:40,t:60,b:68},
    showlegend:false,
  });
  Plotly.react('metric-exhibit-chart',traces,layout,CFG);
}

/* ── Champion metric progression ─────────────────────────────────────── */
function renderHistory(){
  const metricKey=document.getElementById('history-metric-select')
    ?document.getElementById('history-metric-select').value:'gini';
  const promoted=HISTORY.filter(d=>d.promoted);
  const failed=HISTORY.filter(d=>!d.promoted);

  const getVal=(d)=>{
    if(d.metrics&&d.metrics[metricKey]!=null)return d.metrics[metricKey];
    if(metricKey==='gini_weighted')return d.gini??null;
    return null;
  };

  const nonBase=HISTORY.slice(1);
  if(nonBase.length===0){
    Plotly.react('history-chart',[],mkLayout({title:{text:'Champion Metric Progression'}}),CFG);
    return;
  }
  const allVals=nonBase.map(getVal).filter(v=>v!=null);
  if(allVals.length===0){
    Plotly.react('history-chart',[],mkLayout({title:{text:'No data for '+metricKey}}),CFG);
    return;
  }
  const promVals=promoted.filter(d=>d.step>1).map(getVal).filter(v=>v!=null);
  const lowestProm=promVals.length>0?Math.min(...promVals):Math.min(...allVals);
  const yFloor=lowestProm*0.9;
  const yMin=Math.max(Math.min(...allVals),yFloor);
  const yMax=Math.max(...allVals,lowestProm)+(Math.max(...allVals)-yMin)*0.12;

  const prom_=promoted.filter(d=>getVal(d)!=null);
  const fail_=failed.filter(d=>getVal(d)!=null);
  const traces=[
    {name:'Promoted champion',
     x:prom_.map(d=>d.step),y:prom_.map(getVal),
     mode:'lines+markers+text',
     text:prom_.map(d=>d.label),
     textposition:'top center',
     textfont:{size:9,color:PROMO_COLOR},
     line:{color:PROMO_COLOR,width:2.5},
     marker:{size:9,color:PROMO_COLOR},
     hovertemplate:'Step %{x}<br>'+metricKey+': %{y:.4f}<br>%{text}<extra>Promoted</extra>'},
    {name:'Not promoted',
     x:fail_.map(d=>d.step),y:fail_.map(getVal),
     mode:'markers+text',
     text:fail_.map(d=>d.label),
     textposition:'top center',
     textfont:{size:9,color:FAILED_COLOR},
     marker:{size:8,color:FAILED_COLOR,symbol:'circle-open',opacity:0.8},
     hovertemplate:'Step %{x}<br>'+metricKey+': %{y:.4f}<br>%{text}<extra>Not promoted</extra>'},
  ];
  Plotly.react('history-chart',traces,mkLayout({
    title:{text:'Champion Metric Progression — '+metricKey,font:{size:14}},
    xaxis:{title:'Experiment Step',dtick:1},
    yaxis:{title:metricKey,range:[yMin,yMax]},
  }),CFG);
}

/* ── Init ────────────────────────────────────────────────────────────── */
document.getElementById('lift-bands').addEventListener('change',renderLift);
document.querySelectorAll('input[name="lift-mode"]').forEach(r=>r.addEventListener('change',renderLift));
document.getElementById('dl-bands').addEventListener('change',renderDoubleLift);
document.querySelectorAll('input[name="dl-mode"]').forEach(r=>r.addEventListener('change',renderDoubleLift));
document.getElementById('dl-rescale').addEventListener('change',renderDoubleLift);
document.querySelectorAll('input[name="hist-view"]').forEach(r=>r.addEventListener('change',renderHist));
document.querySelectorAll('input[name="scatter-pct"]').forEach(r=>r.addEventListener('change',renderHist));
const metricSel=document.getElementById('metric-select');
if(metricSel){
  // Populate options from METRIC_TABLE
  METRIC_TABLE.forEach(row=>{
    const opt=document.createElement('option');
    opt.value=row.metric;
    const badge=row.is_gate_metric?' 🔒 gate':row.is_kpi_metric?' ★ KPI':'';
    opt.text=row.metric+badge;
    if(row.is_gate_metric)opt.selected=true;
    metricSel.appendChild(opt);
  });
  metricSel.addEventListener('change',renderMetricExhibit);
  renderMetricExhibit();
}
const histMetricSel=document.getElementById('history-metric-select');
if(histMetricSel){
  const _hKeys=Array.from(new Set(HISTORY.flatMap(d=>d.metrics?Object.keys(d.metrics):[])));
  if(_hKeys.length===0)_hKeys.push('gini_weighted');
  _hKeys.sort().forEach(k=>{
    const o=document.createElement('option');
    o.value=k;o.textContent=k+(k==='gini_weighted'?' (default)':'');
    if(k==='gini_weighted')o.selected=true;
    histMetricSel.appendChild(o);
  });
  histMetricSel.addEventListener('change',renderHistory);
}
renderLift();renderDoubleLift();renderGini();renderHist();renderHistory();
"""
    if target_mode != "burning_cost":
        js_code = (
            js_code
            .replace("Pure Premium", "Claim Frequency")
            .replace("pure premium", "claim frequency")
            .replace("Predicted PP", "Predicted frequency")
            .replace("Actual PP", "Actual frequency")
            .replace("Rel. PP", "Rel. frequency")
            .replace("Mean PP", "Mean frequency")
            .replace("predicted PP", "predicted frequency")
            .replace("PP:", "Frequency:")
            .replace("PP ", "frequency ")
            .replace(" PP", " frequency")
            .replace("claim cost", "claim count")
            .replace("Claim Cost", "Claim Count")
            .replace("£", "")
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comparison Report — {escape(comparison_id)}</title>
  <script src="{_PLOTLY_CDN}" crossorigin="anonymous"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f8f9fa;color:#212529;line-height:1.5}}
    .page{{max-width:1280px;margin:0 auto;padding:28px 32px}}
    h1{{font-size:22px;font-weight:700;margin:0 0 4px}}
    h2{{font-size:16px;font-weight:600;margin:28px 0 10px;color:#2c3e50;border-bottom:1px solid #dee2e6;padding-bottom:6px}}
    h3{{font-size:13px;font-weight:600;margin:0 0 8px;color:#495057}}
    p.meta{{color:#6c757d;font-size:13px;margin:0 0 20px}}
    .banner{{padding:12px 18px;border-radius:6px;border-left:4px solid {decision_color};margin-bottom:24px;background:{decision_bg};color:{decision_color}}}
    .banner strong{{font-size:15px;display:block;margin-bottom:2px}}
    .banner span{{font-size:13px}}
    .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:28px}}
    .kpi{{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:14px 18px}}
    .kpi .label{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#6c757d;margin-bottom:4px}}
    .kpi .value{{font-size:24px;font-weight:700}}
    .kpi .sub{{font-size:11px;color:#6c757d;margin-top:3px}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px}}
    .card{{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:16px}}
    .discussion{{font-size:13px;color:#495057;margin:0 0 12px;line-height:1.6;background:#f8f9fa;border-radius:4px;padding:10px 14px;border-left:3px solid #6c9bc5}}
    .discussion strong{{color:#212529}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}
    th,td{{border:1px solid #dee2e6;padding:6px 10px;text-align:left}}
    th{{background:#f3f5f7;font-weight:600}}
    tr:nth-child(even) td{{background:#fafbfc}}
    .pass{{color:#0a5c2e;font-weight:600}}.fail{{color:#7a1a1a;font-weight:600}}
    .section{{margin-bottom:32px}}
    .controls{{display:flex;align-items:center;gap:18px;margin-bottom:10px;flex-wrap:wrap}}
    .controls label{{font-size:13px;color:#495057;display:flex;align-items:center;gap:6px;cursor:pointer}}
    select{{font-size:13px;padding:4px 8px;border:1px solid #ced4da;border-radius:4px;cursor:pointer;background:#fff}}
    .chart-wrap{{background:#fff;border:1px solid #dee2e6;border-radius:8px;padding:8px;margin-bottom:24px}}
    .chart-note{{font-size:12px;color:#6c757d;margin:0 0 10px;line-height:1.5}}
    @media(max-width:768px){{.kpi-grid{{grid-template-columns:repeat(2,1fr)}}.two-col{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<div class="page">

  <h1>Comparison Report</h1>
  <p class="meta">
    ID: <code>{escape(comparison_id)}</code>&nbsp;&nbsp;·&nbsp;&nbsp;
    Eval split: <code>{escape(eval_split)}</code>&nbsp;&nbsp;·&nbsp;&nbsp;
    Primary metric: <code>{escape(primary_metric)}</code>
  </p>

  <div class="banner">
    <strong>{escape(decision_label)}</strong>
    <span>{escape(decision.get("rationale", ""))}</span>
    {f'<br><span style="font-size:12px">Advisory: {escape(advisory_str)} · Decided by: {escape(decided_by)} at {escape(decided_at)}</span>' if decided_by else ''}
    {f'<br><span style="font-size:12px;color:#7a1a1a">⚠ Guardrail failures: {escape(", ".join(guardrail_failures))}</span>' if not guardrail_passed else ''}
  </div>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="label">Champion Gini</div>
      <div class="value" style="color:#1f77b4">{champ_gini:.4f}</div>
      <div class="sub">{escape(champion_id[:48])}</div>
    </div>
    <div class="kpi">
      <div class="label">Challenger Gini</div>
      <div class="value" style="color:#d62728">{chall_gini:.4f}</div>
      <div class="sub">{escape(challenger_id[:48])}</div>
    </div>
    <div class="kpi">
      <div class="label">Mean Lift</div>
      <div class="value" style="color:{lift_color}">{mean_lift:+.4f}</div>
      <div class="sub">Challenger − Champion</div>
    </div>
    <div class="kpi">
      <div class="label">Win Rate</div>
      <div class="value" style="color:{win_color}">{win_rate:.0%}</div>
      <div class="sub">Threshold ≥ 60%</div>
    </div>
  </div>

  <div class="section">
    <h2>Advisory Gates (non-binding)</h2>
    <p class="chart-note">Advisory gates inform but do not enforce promotion. Hard guardrails (discrimination, calibration, validity) are shown in the decision banner. The LLM reviews all metrics and records the final decision.</p>
    <div class="card" style="padding:12px">{gate_html}</div>
  </div>

  <div class="section">
    <h2>Statistical Summary</h2>
    <div class="card" style="padding:12px">{summary_html}</div>
  </div>

  <div class="section">
    <h2>Experiment Details</h2>
    {native_calib_html}
    {diff_discussion_html}
    <div class="two-col">
      <div class="card">{champ_details_html}</div>
      <div class="card">{chall_details_html}</div>
    </div>
  </div>

  <div class="section">
    <h2>Compute &amp; Timing</h2>
    <div class="card" style="padding:12px">{compute_html}</div>
  </div>

  <div class="section">
    <h2>Validation Metrics</h2>
    <div class="card" style="padding:12px">{metrics_html}</div>
  </div>

  <!-- ── LIFT CURVES ── -->
  <div class="section">
    <h2>Lift Curves — Actual vs Predicted by Risk Band</h2>
    <p class="chart-note">
      Policies sorted ascending by each model's own predicted pure premium (Band 1 = lowest risk,
      Band N = highest). Bands contain equal total exposure; grey bars show exposure per band (right axis).
      Solid lines = predicted PP; dashed = actual PP. A well-discriminating model shows a steep monotone
      rise with predicted and actual tracking closely.
    </p>
    <div class="controls">
      <label>Bands: <select id="lift-bands">{band_options}</select></label>
      <label><input type="radio" name="lift-mode" value="absolute" checked> Absolute PP (£)</label>
      <label><input type="radio" name="lift-mode" value="ae"> A/E Ratio</label>
      <label><input type="radio" name="lift-mode" value="rescaled"> Rescaled to champion mean</label>
    </div>
    <div class="chart-wrap"><div id="lift-chart" style="height:440px"></div></div>
  </div>

  <!-- ── DOUBLE LIFT ── -->
  <div class="section">
    <h2>Double Lift Curve</h2>
    <p class="chart-note">
      Policies sorted ascending by Challenger ÷ Champion predicted PP ratio.
      Grey bars show exposure per band (right axis).
      <strong>Equal exposure:</strong> N bands of equal total exposure; x-axis = band ordinal.
      <strong>Equal width:</strong> N bands of equal ratio-interval width (2nd–98th percentile range);
      x-axis = the actual ratio midpoint — spacing reflects where the models truly disagree.
    </p>
    <div class="controls">
      <label>Bands: <select id="dl-bands">{band_options}</select></label>
      <label><input type="radio" name="dl-mode" value="equal_exposure" checked> Equal exposure</label>
      <label><input type="radio" name="dl-mode" value="equal_width"> Equal width</label>
      <label><input type="checkbox" id="dl-rescale"> Rescale to champion (champion&nbsp;=&nbsp;1.0)</label>
    </div>
    <div class="chart-wrap"><div id="double-lift-chart" style="height:440px"></div></div>
  </div>

  <!-- ── GINI EXHIBIT ── -->
  <div class="section">
    <h2>Gini Lorenz Curves</h2>
    <p class="chart-note">
      Policies ranked by predicted pure premium <strong>ascending</strong>. X = cumulative exposure share;
      Y = cumulative actual claim cost share. A well-discriminating model curves <em>below</em> the diagonal.
      Gini values match the reported <code>gini_weighted</code> metric exactly.
    </p>
    <div class="chart-wrap"><div id="gini-chart" style="height:460px"></div></div>
  </div>

  <!-- ── PREDICTION DISTRIBUTION ── -->
  <div class="section">
    <h2>Predicted Pure Premium Distribution</h2>
    <p class="chart-note">
      Three views of the predicted pure premium spread.
      <strong>Distribution:</strong> exposure-weighted histogram per model (1st–99th pct).
      <strong>% Diff:</strong> exposure-weighted histogram of (Challenger − Champion) / Champion per policy.
      <strong>Scatter:</strong> champion vs challenger PP per policy; axes capped at 99th percentile.
      Use the sample size selector to control density.
    </p>
    <div class="controls">
      <label><input type="radio" name="hist-view" value="dist" checked> Distribution</label>
      <label><input type="radio" name="hist-view" value="pctdiff"> % Diff</label>
      <label><input type="radio" name="hist-view" value="scatter"> Scatter</label>
      <span style="color:#adb5bd;margin:0 4px">|</span>
      <label><input type="radio" name="scatter-pct" value="1pct"> 1%</label>
      <label><input type="radio" name="scatter-pct" value="5pct"> 5%</label>
      <label><input type="radio" name="scatter-pct" value="25pct" checked> 25%</label>
      <label><input type="radio" name="scatter-pct" value="100pct"> 100%</label>
    </div>
    <div class="chart-wrap"><div id="hist-chart" style="height:380px"></div></div>
  </div>

  <!-- ── MULTI-METRIC EXHIBIT ── -->
  <div class="section">
    <h2>Multi-Metric Comparison</h2>
    <p class="chart-note">
      Select any metric from the panel to compare champion vs challenger.
      <strong>🔒 gate</strong> = the primary gate metric (<code>{escape(gate_primary_metric)}</code>).
      <strong>★ KPI</strong> = the reported business KPI (<code>gini_weighted</code>).
      <strong>asym_pricing_loss</strong> penalises under-pricing 4× over-pricing (lower is better).
      Left panel: score bars. Right panel: lift distribution across
      {escape(n_partitions_label)} — box shows median/IQR, dots show individual samples.
      A metric that consistently favours the challenger is more reliable than one that varies widely.
    </p>
    <div class="controls">
      <label style="font-weight:600">Metric:&nbsp;<select id="metric-select" style="min-width:280px"></select></label>
      <span id="metric-lift-display" style="margin-left:16px;font-size:14px"></span>
    </div>
    <div class="chart-wrap"><div id="metric-exhibit-chart" style="height:400px"></div></div>
  </div>

  <!-- ── CHAMPION HISTORY ── -->
  <div class="section">
    <h2>Champion Metric Progression</h2>
    <p class="chart-note">
      Selected metric for every experiment in this run.
      <span style="color:#0a5c2e;font-weight:600">Green filled = promoted</span>;
      <span style="color:#d62728;font-weight:600">red open = not promoted</span>.
      The connected line traces only the champion lineage.
    </p>
    <div class="controls">
      <label style="font-weight:600">Metric:&nbsp;
        <select id="history-metric-select" style="min-width:220px"></select>
      </label>
    </div>
    <div class="chart-wrap"><div id="history-chart" style="height:360px"></div></div>
  </div>

</div>

<script>{data_script}</script>
<script>{js_code}</script>
</body>
</html>
"""
    if target_mode != "burning_cost":
        html = (
            html
            .replace("pure premium", "claim frequency")
            .replace("Pure Premium", "Claim Frequency")
            .replace("predicted PP", "predicted frequency")
            .replace("PP", "frequency")
            .replace("claim cost", "claim count")
            .replace("Claim Cost", "Claim Count")
            .replace("£", "")
        )
    return html


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

def _gate_table(
    decision: dict[str, Any],
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    challenger_metrics: dict[str, Any],
) -> str:
    checks = decision.get("checks", {})
    thresholds = decision.get("thresholds", {})

    mean_lift = float(comparison_summary.get("mean_lift") or 0)
    win_rate = float(comparison_summary.get("challenger_win_rate") or 0)
    champ_score = float(comparison_summary.get("champion_mean_score") or 1e-9)
    rel_lift_pct = mean_lift / max(abs(champ_score), 1e-9) * 100.0
    bs_lower = float(bootstrap_summary.get("interval_lower") or 0)
    bs_lower_rel = bs_lower / max(abs(champ_score), 1e-9) * 100.0
    pred_actual = float(challenger_metrics.get("predicted_to_actual_ratio") or 1.0)
    drift_pct = abs(pred_actual - 1.0) * 100.0
    max_drift_pct = float(thresholds.get("max_predicted_to_actual_drift", 0.1)) * 100.0
    min_rel_pct = float(thresholds.get("min_relative_lift", 0.005)) * 100.0
    min_win = float(thresholds.get("minimum_win_rate", 0.6)) * 100.0
    bs_lb = float(thresholds.get("bootstrap_lower_bound", 0.0))
    bs_lb_rel = float(thresholds.get("bootstrap_lower_bound_relative", 0.0)) * 100.0

    def _row(label: str, threshold: str, challenger_val: str, passed: bool | None) -> str:
        if passed is None:
            icon, cls = "—", ""
        elif passed:
            icon, cls = "✓", "pass"
        else:
            icon, cls = "✗", "fail"
        return (
            f"<tr><td>{escape(label)}</td>"
            f"<td>{escape(threshold)}</td>"
            f"<td>{escape(challenger_val)}</td>"
            f"<td class='{cls}'>{icon}</td></tr>"
        )

    gate_mode_str = comparison_summary.get("gate_mode", "single_partition")
    gate_metric_str = comparison_summary.get("gate_primary_metric") or comparison_summary.get("primary_metric", "gini_weighted")
    n_partitions = comparison_summary.get("n_partitions") or comparison_summary.get("n_resamples", 30)
    if gate_mode_str == "repeated_cv":
        context_label = f"repeated CV ({comparison_summary.get('n_folds', 4)}×{comparison_summary.get('n_repeats', 4)} partitions)"
    else:
        context_label = f"single partition ({n_partitions} bootstrap resamples)"

    rows = [
        "<table>",
        f"<tr><th colspan='4' style='background:#f0f4ff;color:#2c3e50;font-size:12px'>"
        f"Gate mode: <strong>{escape(context_label)}</strong> — "
        f"decision metric: <strong>{escape(gate_metric_str)}</strong></th></tr>",
        "<tr><th>Gate</th><th>Threshold</th><th>Challenger value</th><th>Result</th></tr>",
        _row(f"Win rate across {n_partitions} partitions/resamples", f"≥ {min_win:.0f}%", f"{win_rate*100:.1f}%",
             checks.get("challenger_win_rate")),
        _row("Mean lift > 0", "> 0.0000", f"{mean_lift:+.5f}",
             checks.get("mean_lift_positive")),
        _row("Absolute lift", f"≥ {thresholds.get('min_absolute_lift', 0.0):.4f}",
             f"{mean_lift:+.5f}", checks.get("absolute_lift")),
        _row("Relative lift", f"≥ {min_rel_pct:.3f}%", f"{rel_lift_pct:+.4f}%",
             checks.get("relative_lift")),
        _row("Bootstrap lower bound (absolute)", f"≥ {bs_lb:.4f}",
             f"{bs_lower:+.5f}", checks.get("bootstrap_lower_bound")),
        _row("Bootstrap lower bound (relative)", f"≥ {bs_lb_rel:.3f}%",
             f"{bs_lower_rel:+.4f}%", checks.get("bootstrap_lower_bound_relative")),
        _row("Calibration (pred/actual drift)", f"≤ ±{max_drift_pct:.0f}%",
             f"{pred_actual:.4f}  ({drift_pct:.1f}% drift)", checks.get("calibration_ok")),
        _row("Diagnostics present", "required",
             "present" if checks.get("diagnostics_present") else "absent",
             checks.get("diagnostics_present")),
    ]
    if "sign_agreement_kpi" in checks:
        kpi_lift = float(comparison_summary.get("mean_kpi_lift", 0) or 0)
        rows.append(_row(
            "Sign agreement (rank_gini ↑ and gini_weighted ↑)",
            "both positive",
            f"gini_weighted lift {kpi_lift:+.4f}",
            checks.get("sign_agreement_kpi"),
        ))
    rows.append("</table>")
    return "\n".join(rows)


def _experiment_diff_discussion(
    champion_details: dict[str, Any],
    challenger_details: dict[str, Any],
) -> str:
    """Generate a human-readable paragraph comparing champion and challenger."""
    champ_name = champion_details.get("experiment_name") or "champion"
    chall_name = challenger_details.get("experiment_name") or "challenger"
    champ_family = champion_details.get("model_family") or "?"
    chall_family = challenger_details.get("model_family") or "?"
    champ_model = champion_details.get("model") or {}
    chall_model = challenger_details.get("model") or {}
    champ_target = champion_details.get("target_strategy") or ""
    chall_target = challenger_details.get("target_strategy") or ""

    parts: list[str] = []

    # Model family
    if champ_family == chall_family:
        parts.append(f"Both experiments use <strong>{escape(champ_family)}</strong>.")
    else:
        parts.append(
            f"Champion uses <strong>{escape(champ_family)}</strong>; "
            f"challenger uses <strong>{escape(chall_family)}</strong>."
        )

    # Target strategy
    if champ_target and chall_target:
        if champ_target == chall_target:
            parts.append(f"Target strategy: <strong>{escape(champ_target)}</strong> (unchanged).")
        else:
            parts.append(
                f"Target strategy changed from <strong>{escape(champ_target)}</strong> to "
                f"<strong>{escape(chall_target)}</strong>."
            )

    # Regularisation
    try:
        ca = float(champ_model.get("alpha") or 0.1)
        ha = float(chall_model.get("alpha") or 0.1)
        if abs(ca - ha) > 1e-9:
            parts.append(f"Regularisation α: {ca} → {ha}.")
    except (TypeError, ValueError):
        pass

    # Feature exclusions
    champ_excl = set(champ_model.get("feature_exclusions") or [])
    chall_excl = set(chall_model.get("feature_exclusions") or [])
    added_excl = chall_excl - champ_excl
    removed_excl = champ_excl - chall_excl
    if added_excl:
        parts.append(f"Features excluded in challenger: {escape(', '.join(sorted(added_excl)))}.")
    if removed_excl:
        parts.append(f"Features re-included in challenger: {escape(', '.join(sorted(removed_excl)))}.")

    # Model script
    champ_script = (champ_model.get("script_path") or "").rsplit("/", 1)[-1]
    chall_script = (chall_model.get("script_path") or "").rsplit("/", 1)[-1]
    if chall_script and champ_script != chall_script:
        parts.append(f"New model script: <code>{escape(chall_script)}</code>.")

    if not parts:
        parts.append("No structural differences detected; challenger is a re-run of the champion configuration.")

    # Proposal narrative (when available)
    change_summary = challenger_details.get("change_summary") or ""
    expected_benefit = challenger_details.get("expected_benefit") or ""
    key_risk = challenger_details.get("key_risk") or ""
    rationale = challenger_details.get("rationale") or ""

    narrative_parts: list[str] = []
    if change_summary:
        narrative_parts.append(f"<strong>Change:</strong> {escape(change_summary)}")
    if expected_benefit:
        narrative_parts.append(f"<strong>Expected benefit:</strong> {escape(expected_benefit)}")
    if key_risk:
        narrative_parts.append(f"<strong>Key risk:</strong> {escape(key_risk)}")

    html = f'<div class="discussion"><strong>{escape(champ_name)}</strong> vs <strong>{escape(chall_name)}</strong> — '
    html += " ".join(parts)
    if rationale:
        html += f"<br><em>{escape(rationale)}</em>"
    if narrative_parts:
        html += "<br>" + " &nbsp;·&nbsp; ".join(narrative_parts)
    html += "</div>"
    return html


def _native_calibration_warning(challenger_details: dict[str, Any]) -> str:
    """
    Show a soft warning banner if the challenger's model notes record a
    native_pred_to_actual_ratio (i.e. ratio before calibration was applied).
    This preserves observability of the underlying model bias even after
    the calibration scalar has been applied and the gate passes.
    """
    model = challenger_details.get("model") or {}
    # model notes are stored flat in the snapshot under [model] → script hyperparams
    native_ratio = model.get("native_pred_to_actual_ratio")
    calib_factor = model.get("calib_factor")
    if native_ratio is None:
        return ""
    try:
        ratio_f = float(native_ratio)
        drift_pct = abs(ratio_f - 1.0) * 100.0
        if drift_pct < 5.0:
            return ""  # negligible — no need to surface
        color = "#7a1a1a" if drift_pct > 20.0 else "#5a4500"
        bg = "#fce8e8" if drift_pct > 20.0 else "#fffde7"
        calib_str = f"  Correction applied: ×{float(calib_factor):.4f}." if calib_factor else ""
        return (
            f'<div style="padding:8px 14px;border-radius:6px;border-left:4px solid {color};'
            f'background:{bg};color:{color};font-size:13px;margin-bottom:12px">'
            f"<strong>Native calibration note:</strong> Before the training-total calibration "
            f"scalar was applied, the challenger's predicted/actual ratio was "
            f"<strong>{ratio_f:.4f}</strong> ({drift_pct:.1f}% drift).{escape(calib_str)} "
            f"This indicates a structural bias in the model's link/regularisation; "
            f"the promoted predictions have been rescaled but the underlying model "
            f"should be reviewed.</div>"
        )
    except (TypeError, ValueError):
        return ""


def _compute_section(champion_timing: dict[str, Any], challenger_timing: dict[str, Any]) -> str:
    """Render a small Compute & Timing table for the comparison report."""
    if not champion_timing and not challenger_timing:
        return "<p style='color:#6c757d;font-size:13px'>No timing data available (experiments predating this feature).</p>"

    def _t(val: Any, unit: str = "s") -> str:
        if val is None:
            return "—"
        try:
            return f"{float(val):.1f}{unit}"
        except (TypeError, ValueError):
            return str(val)

    def _pct(val: Any) -> str:
        if val is None:
            return "—"
        try:
            pct = float(val) * 100
            color = "#7a1a1a" if pct > 90 else "#5a3a00" if pct > 70 else "#0a5c2e"
            return f'<span style="color:{color};font-weight:600">{pct:.1f}%</span>'
        except (TypeError, ValueError):
            return "—"

    rows = [
        "<table>",
        "<tr><th>Metric</th><th>Champion</th><th>Challenger</th></tr>",
    ]

    def _to_row(label: str, c_val: Any, h_val: Any) -> str:
        return f"<tr><td>{escape(label)}</td><td>{c_val}</td><td>{h_val}</td></tr>"

    rows.append(_to_row(
        "Fit wall-clock time",
        _t(champion_timing.get("fit_wall_seconds")),
        _t(challenger_timing.get("fit_wall_seconds")),
    ))
    rows.append(_to_row(
        "Fit CPU time",
        _t(champion_timing.get("fit_cpu_seconds")),
        _t(challenger_timing.get("fit_cpu_seconds")),
    ))
    rows.append(_to_row(
        "Budget (seconds)",
        _t(champion_timing.get("compute_budget_seconds")),
        _t(challenger_timing.get("compute_budget_seconds")),
    ))
    rows.append(_to_row(
        "Budget utilisation",
        _pct(champion_timing.get("budget_utilisation")),
        _pct(challenger_timing.get("budget_utilisation")),
    ))

    c_to = bool(champion_timing.get("timed_out"))
    h_to = bool(challenger_timing.get("timed_out"))
    c_to_str = '<span style="color:#7a1a1a;font-weight:700">⚠ YES</span>' if c_to else "No"
    h_to_str = '<span style="color:#7a1a1a;font-weight:700">⚠ YES</span>' if h_to else "No"
    rows.append(_to_row("Timed out", c_to_str, h_to_str))

    rows.append("</table>")
    return "\n".join(rows)


def _summary_table(comparison_summary: dict[str, Any], bootstrap_summary: dict[str, Any]) -> str:
    return _dict_table({
        "Mean lift": comparison_summary.get("mean_lift"),
        "Win rate": comparison_summary.get("challenger_win_rate"),
        "Champion mean score": comparison_summary.get("champion_mean_score"),
        "Challenger mean score": comparison_summary.get("challenger_mean_score"),
        "Bootstrap CI lower": bootstrap_summary.get("interval_lower"),
        "Bootstrap CI upper": bootstrap_summary.get("interval_upper"),
        "P(challenger > champion)": bootstrap_summary.get("probability_challenger_outperforms"),
        "Bonferroni-adjusted confidence": bootstrap_summary.get("adjusted_confidence_level"),
    })


def _details_card(title: str, details: dict[str, Any]) -> str:
    display = {
        k: v for k, v in details.items()
        if k not in ("rationale", "change_summary", "expected_benefit", "key_risk")
        and v is not None
    }
    return f"<h3>{escape(title)}</h3>{_dict_table(display)}"


def _metrics_table(champion: dict[str, Any], challenger: dict[str, Any]) -> str:
    rows = [
        "<table>",
        "<tr><th>Metric</th><th>Champion</th><th>Challenger</th><th>Delta</th></tr>",
    ]
    for key in METRIC_KEYS:
        c = champion.get(key)
        h = challenger.get(key)
        delta = (h - c) if isinstance(c, (int, float)) and isinstance(h, (int, float)) else ""
        delta_style = ""
        if isinstance(delta, float):
            delta_style = ' style="color:#0a5c2e"' if delta > 0 else ' style="color:#7a1a1a"' if delta < 0 else ""
        rows.append(
            f"<tr><td>{escape(key)}</td><td>{_fmt(c)}</td>"
            f"<td>{_fmt(h)}</td><td{delta_style}>{_fmt(delta)}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _dict_table(values: dict[str, Any]) -> str:
    rows = ["<table>"]
    for key, value in values.items():
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(_fmt(value))}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(_fmt(item) for item in value)
    return str(value)
