"""HTML reporting for run-scoped experiment comparisons."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.config import ProjectConfig
from autoresearch.evaluation.metrics import full_metric_panel
from autoresearch.experiment_registry.registry import (
    get_experiment,
    list_artifacts,
    list_champion_history,
)
from autoresearch.utils.io import read_json


METRIC_KEYS = [
    "gini_weighted",
    "tweedie_deviance_p15",
    "poisson_deviance",
    "predicted_to_actual_ratio",
    "weighted_mae_claim_cost",
    "weighted_rmse_claim_cost",
    "rmse_pure_premium",
    "mae_pure_premium",
    "double_lift_slope",
    "mean_actual_pure_premium",
    "mean_predicted_pure_premium",
    "total_actual_claim_cost",
    "total_predicted_claim_cost",
]

_BAND_COUNTS = [5, 10, 20, 50]
_DEFAULT_BANDS = 10
_GINI_CURVE_POINTS = 400
_HIST_BINS = 40
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
    output_path: Path,
) -> Path:
    """Write a self-contained interactive comparison report and return its path."""
    champion_predictions = pd.read_parquet(_artifact_path(config, champion_id, "predictions"))
    challenger_predictions = pd.read_parquet(_artifact_path(config, challenger_id, "predictions"))
    eval_split = config.ordinary_eval_splits[0]
    champion_eval = champion_predictions[champion_predictions["split"] == eval_split].copy()
    challenger_eval = challenger_predictions[challenger_predictions["split"] == eval_split].copy()

    champion_metrics = full_metric_panel(
        champion_eval["actual_claim_cost"],
        champion_eval["predicted_claim_cost"],
        champion_eval["exposure"],
        tweedie_power=config.tweedie_power,
    )
    challenger_metrics = full_metric_panel(
        challenger_eval["actual_claim_cost"],
        challenger_eval["predicted_claim_cost"],
        challenger_eval["exposure"],
        tweedie_power=config.tweedie_power,
    )

    lift_data = {
        "champion": _compute_lift_data(champion_eval, _BAND_COUNTS),
        "challenger": _compute_lift_data(challenger_eval, _BAND_COUNTS),
    }
    double_lift_data = _compute_double_lift_data(champion_eval, challenger_eval, _BAND_COUNTS)
    gini_data = {
        "champion": _compute_gini_curve(champion_eval, _GINI_CURVE_POINTS),
        "challenger": _compute_gini_curve(challenger_eval, _GINI_CURVE_POINTS),
    }
    pred_hist_data = {
        "champion": _compute_pred_histogram(champion_eval, _HIST_BINS),
        "challenger": _compute_pred_histogram(challenger_eval, _HIST_BINS),
    }
    history_points = _champion_history_points(config, eval_split)

    html = _render_html(
        comparison_id=comparison_id,
        eval_split=eval_split,
        primary_metric=config.primary_metric,
        decision=decision,
        comparison_summary=comparison_summary,
        bootstrap_summary=bootstrap_summary,
        champion_id=champion_id,
        challenger_id=challenger_id,
        champion_details=_experiment_details(config, champion_id),
        challenger_details=_experiment_details(config, challenger_id),
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        lift_data=lift_data,
        double_lift_data=double_lift_data,
        gini_data=gini_data,
        pred_hist_data=pred_hist_data,
        history_points=history_points,
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


def _experiment_details(config: ProjectConfig, experiment_id: str) -> dict[str, Any]:
    experiment = get_experiment(config.registry_path, experiment_id)
    snapshot = read_json(_artifact_path(config, experiment_id, "config_snapshot"))
    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment.get("experiment_name"),
        "model_family": experiment.get("model_family"),
        "target_strategy": experiment.get("target_strategy"),
        "mean_score": experiment.get("mean_score"),
        "status": experiment.get("status"),
        "preprocessing": snapshot.get("effective_preprocessing"),
        "model": snapshot.get("experiment", {}).get("model"),
    }


def _champion_history_points(config: ProjectConfig, eval_split: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    seen: set[str] = set()
    for row in reversed(list_champion_history(config.registry_path)):
        champion_id = row.get("new_champion_id")
        if not champion_id or champion_id in seen:
            continue
        seen.add(champion_id)
        try:
            metrics = read_json(_artifact_path(config, champion_id, "metrics"))
            split_metric = next(
                item for item in metrics["split_metrics"] if item["split"] == eval_split
            )
            points.append((float(len(points) + 1), float(split_metric["gini_weighted"])))
        except Exception:
            continue
    return points or [(1.0, 0.0)]


# ---------------------------------------------------------------------------
# Chart data computation
# ---------------------------------------------------------------------------

def _equal_exposure_bins(exposure_sorted: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign bin labels 1..n_bins to a sorted array so each bin has ~equal total exposure."""
    if n_bins <= 0 or len(exposure_sorted) == 0:
        return np.ones(len(exposure_sorted), dtype=int)
    cum_exp = np.cumsum(exposure_sorted)
    total_exp = cum_exp[-1]
    if total_exp < 1e-12:
        n = len(exposure_sorted)
        return np.ceil(np.arange(1, n + 1) / n * n_bins).clip(1, n_bins).astype(int)
    return np.ceil(cum_exp / total_exp * n_bins).clip(1, n_bins).astype(int)


def _compute_lift_data(frame: pd.DataFrame, band_counts: list[int]) -> dict[str, list[dict]]:
    """
    Sort policies by predicted pure premium (ascending = lowest risk first).
    Bin into equal-exposure bands. Per band: actual PP, predicted PP, A/E ratio.
    Returned as dict of str(n_bands) → list of band dicts.
    """
    exp = frame["exposure"].astype(float).values
    actual_cc = frame["actual_claim_cost"].astype(float).values
    pred_cc = frame["predicted_claim_cost"].astype(float).values

    sort_idx = frame["predicted_pure_premium"].astype(float).argsort().values
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
) -> dict[str, list[dict]]:
    """
    Sort policies by Challenger / Champion predicted PP ratio (ascending).
    Bin into equal-exposure bands. Per band: actual PP, champion PP, challenger PP.
    X-axis value = mean ratio for the band.
    """
    paired = champion[["record_id", "actual_claim_cost", "predicted_claim_cost", "exposure"]].merge(
        challenger[["record_id", "predicted_claim_cost"]],
        on="record_id",
        suffixes=("_champ", "_chall"),
    ).copy()

    exp = paired["exposure"].astype(float).values
    actual_cc = paired["actual_claim_cost"].astype(float).values
    champ_cc = paired["predicted_claim_cost_champ"].astype(float).values
    chall_cc = paired["predicted_claim_cost_chall"].astype(float).values

    # Ratio at policy level using per-policy pure premium
    champ_pp_pol = champ_cc / exp.clip(min=1e-12)
    chall_pp_pol = chall_cc / exp.clip(min=1e-12)
    ratio = chall_pp_pol / champ_pp_pol.clip(min=1e-9)

    sort_idx = ratio.argsort()
    exp_s = exp[sort_idx]
    actual_s = actual_cc[sort_idx]
    champ_s = champ_cc[sort_idx]
    chall_s = chall_cc[sort_idx]
    ratio_s = ratio[sort_idx]

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
            champ_pp = float(champ_s[mask].sum()) / exp_sum
            chall_pp = float(chall_s[mask].sum()) / exp_sum
            ratio_vals = ratio_s[mask]
            bands.append({
                "band": b,
                "ratio_mean": round(float(ratio_vals.mean()), 4),
                "ratio_min": round(float(ratio_vals.min()), 4),
                "ratio_max": round(float(ratio_vals.max()), 4),
                "actual_pp": round(actual_pp, 4),
                "champion_pp": round(champ_pp, 4),
                "challenger_pp": round(chall_pp, 4),
                "exposure": round(exp_sum, 2),
                "n_policies": int(mask.sum()),
            })
        result[str(n)] = bands
    return result


def _compute_gini_curve(frame: pd.DataFrame, n_points: int = 400) -> dict:
    """
    Lorenz curve: sort by predicted PURE PREMIUM ascending (lowest predicted risk first).
    X = cumulative exposure share, Y = cumulative actual claim cost share.
    A good model curves below the diagonal; Gini > 0 for a good model.
    """
    f = frame.copy()
    f["_pred_pp"] = f["predicted_claim_cost"].astype(float) / f["exposure"].astype(float).clip(lower=1e-12)
    ordered = f.sort_values("_pred_pp", ascending=True)
    exposure = ordered["exposure"].astype(float).values
    actual_cost = ordered["actual_claim_cost"].astype(float).values

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


def _compute_pred_histogram(frame: pd.DataFrame, n_bins: int = 40) -> dict:
    """Exposure-weighted histogram of predicted pure premiums (clipped at 1st/99th pct)."""
    pp = frame["predicted_pure_premium"].astype(float).values
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


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(
    *,
    comparison_id: str,
    eval_split: str,
    primary_metric: str,
    decision: dict[str, Any],
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    champion_id: str,
    challenger_id: str,
    champion_details: dict[str, Any],
    challenger_details: dict[str, Any],
    champion_metrics: dict[str, Any],
    challenger_metrics: dict[str, Any],
    lift_data: dict,
    double_lift_data: dict,
    gini_data: dict,
    pred_hist_data: dict,
    history_points: list,
) -> str:
    decision_str = decision.get("decision", "?")
    decision_color = (
        "#0a5c2e" if decision_str == "promoted"
        else "#7a1a1a" if decision_str == "rejected"
        else "#5a4500"
    )
    decision_bg = (
        "#e8f5e9" if decision_str == "promoted"
        else "#fce8e8" if decision_str == "rejected"
        else "#fffde7"
    )

    champ_gini = gini_data["champion"]["gini"]
    chall_gini = gini_data["challenger"]["gini"]
    mean_lift = float(comparison_summary.get("mean_lift") or 0)
    win_rate = float(comparison_summary.get("challenger_win_rate") or 0)
    lift_color = "#0a5c2e" if mean_lift > 0 else "#7a1a1a"
    win_color = "#0a5c2e" if win_rate >= 0.6 else "#7a1a1a"

    # Data injected into the page as JSON (no f-string escaping issues with JS braces below)
    data_script = (
        "const LIFT=" + json.dumps(lift_data, separators=(",", ":")) + ";"
        "const DL=" + json.dumps(double_lift_data, separators=(",", ":")) + ";"
        "const GINI=" + json.dumps(gini_data, separators=(",", ":")) + ";"
        "const HIST=" + json.dumps(pred_hist_data, separators=(",", ":")) + ";"
        "const HISTORY=" + json.dumps(history_points, separators=(",", ":")) + ";"
    )

    band_options = "\n".join(
        f'<option value="{n}"{" selected" if n == _DEFAULT_BANDS else ""}>{n} bands</option>'
        for n in _BAND_COUNTS
    )

    metrics_html = _metrics_table(champion_metrics, challenger_metrics)
    summary_html = _summary_table(comparison_summary, bootstrap_summary)
    champ_details_html = _details_table("Champion", champion_details)
    chall_details_html = _details_table("Challenger", challenger_details)

    # JavaScript — written as a plain string (no Python f-string processing)
    # so JS object braces and template literals are preserved as-is.
    js_code = r"""
const CHAMP_COLOR='#1f77b4',CHALL_COLOR='#d62728',ACTUAL_COLOR='#2ca02c';
const CFG={displayModeBar:true,modeBarButtonsToRemove:['select2d','lasso2d','toggleSpikelines'],displaylogo:false,responsive:true};
const BASE_LAYOUT={
  margin:{l:68,r:28,t:52,b:68},
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

/* ── Lift curves ─────────────────────────────────────────────────────── */
function renderLift(){
  const n=document.getElementById('lift-bands').value;
  const mode=document.querySelector('input[name="lift-mode"]:checked').value;
  const champ=LIFT.champion[n]||[];
  const chall=LIFT.challenger[n]||[];
  let traces,layout;

  if(mode==='absolute'){
    traces=[
      {name:'Champion — Predicted',
       x:champ.map(d=>d.band),y:champ.map(d=>d.predicted_pp),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
       customdata:champ.map(d=>[d.n_policies,d.exposure,d.ae_ratio]),
       hovertemplate:'Band %{x}<br>Predicted PP: £%{y:,.2f}<br>A/E: %{customdata[2]:.3f}<br>Policies: %{customdata[0]:.0f}<br>Exposure: %{customdata[1]:.1f}<extra>Champion Predicted</extra>'},
      {name:'Champion — Actual',
       x:champ.map(d=>d.band),y:champ.map(d=>d.actual_pp),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2,dash:'dash'},marker:{size:6,symbol:'circle-open'},
       customdata:champ.map(d=>[d.n_policies,d.exposure]),
       hovertemplate:'Band %{x}<br>Actual PP: £%{y:,.2f}<br>Policies: %{customdata[0]:.0f}<extra>Champion Actual</extra>'},
      {name:'Challenger — Predicted',
       x:chall.map(d=>d.band),y:chall.map(d=>d.predicted_pp),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       customdata:chall.map(d=>[d.n_policies,d.exposure,d.ae_ratio]),
       hovertemplate:'Band %{x}<br>Predicted PP: £%{y:,.2f}<br>A/E: %{customdata[2]:.3f}<br>Policies: %{customdata[0]:.0f}<br>Exposure: %{customdata[1]:.1f}<extra>Challenger Predicted</extra>'},
      {name:'Challenger — Actual',
       x:chall.map(d=>d.band),y:chall.map(d=>d.actual_pp),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2,dash:'dash'},marker:{size:6,symbol:'circle-open'},
       customdata:chall.map(d=>[d.n_policies,d.exposure]),
       hovertemplate:'Band %{x}<br>Actual PP: £%{y:,.2f}<br>Policies: %{customdata[0]:.0f}<extra>Challenger Actual</extra>'},
    ];
    layout=mkLayout({
      title:{text:'Actual vs Predicted Pure Premium by Risk Band',font:{size:14}},
      xaxis:{title:'Risk Band  (1 = lowest predicted risk → N = highest)',dtick:1},
      yaxis:{title:'Pure Premium (£)'},
    });
  } else {
    // A/E ratio mode
    const maxBand=Math.max(champ.length,chall.length,1);
    traces=[
      {name:'Champion A/E',
       x:champ.map(d=>d.band),y:champ.map(d=>d.ae_ratio),
       mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
       customdata:champ.map(d=>[d.actual_pp,d.predicted_pp,d.n_policies,d.exposure]),
       hovertemplate:'Band %{x}<br>A/E: %{y:.3f}<br>Actual PP: £%{customdata[0]:,.2f}<br>Predicted PP: £%{customdata[1]:,.2f}<br>Policies: %{customdata[2]:.0f}<extra>Champion</extra>'},
      {name:'Challenger A/E',
       x:chall.map(d=>d.band),y:chall.map(d=>d.ae_ratio),
       mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
       customdata:chall.map(d=>[d.actual_pp,d.predicted_pp,d.n_policies,d.exposure]),
       hovertemplate:'Band %{x}<br>A/E: %{y:.3f}<br>Actual PP: £%{customdata[0]:,.2f}<br>Predicted PP: £%{customdata[1]:,.2f}<br>Policies: %{customdata[2]:.0f}<extra>Challenger</extra>'},
      {name:'Perfect Calibration (1.0)',
       x:[1,maxBand],y:[1,1],mode:'lines',
       line:{color:'#868e96',width:1.5,dash:'dot'},hoverinfo:'skip'},
    ];
    layout=mkLayout({
      title:{text:'Actual / Expected (A/E) Ratio by Risk Band',font:{size:14}},
      xaxis:{title:'Risk Band  (1 = lowest predicted risk → N = highest)',dtick:1},
      yaxis:{title:'A/E Ratio  (Actual PP ÷ Predicted PP)'},
    });
  }
  Plotly.react('lift-chart',traces,layout,CFG);
}

/* ── Double lift ─────────────────────────────────────────────────────── */
function renderDoubleLift(){
  const n=document.getElementById('dl-bands').value;
  const bands=DL[n]||[];
  const traces=[
    {name:'Actual',
     x:bands.map(d=>d.ratio_mean),y:bands.map(d=>d.actual_pp),
     mode:'lines+markers',line:{color:ACTUAL_COLOR,width:2.5,dash:'dash'},
     marker:{size:8,symbol:'diamond'},
     customdata:bands.map(d=>[d.ratio_min,d.ratio_max,d.n_policies,d.exposure]),
     hovertemplate:'Ratio %{x:.3f}×<br>Actual PP: £%{y:,.2f}<br>Ratio range: %{customdata[0]:.3f}× – %{customdata[1]:.3f}×<br>Policies: %{customdata[2]:.0f}<br>Exposure: %{customdata[3]:.1f}<extra>Actual</extra>'},
    {name:'Champion Predicted',
     x:bands.map(d=>d.ratio_mean),y:bands.map(d=>d.champion_pp),
     mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:6},
     customdata:bands.map(d=>[d.ratio_min,d.ratio_max,d.n_policies]),
     hovertemplate:'Ratio %{x:.3f}×<br>Champion PP: £%{y:,.2f}<br>Ratio range: %{customdata[0]:.3f}× – %{customdata[1]:.3f}×<br>Policies: %{customdata[2]:.0f}<extra>Champion</extra>'},
    {name:'Challenger Predicted',
     x:bands.map(d=>d.ratio_mean),y:bands.map(d=>d.challenger_pp),
     mode:'lines+markers',line:{color:CHALL_COLOR,width:2},marker:{size:6},
     customdata:bands.map(d=>[d.ratio_min,d.ratio_max,d.n_policies]),
     hovertemplate:'Ratio %{x:.3f}×<br>Challenger PP: £%{y:,.2f}<br>Ratio range: %{customdata[0]:.3f}× – %{customdata[1]:.3f}×<br>Policies: %{customdata[2]:.0f}<extra>Challenger</extra>'},
  ];
  Plotly.react('double-lift-chart',traces,mkLayout({
    title:{text:'Double Lift Curve  (bands sorted by Challenger ÷ Champion ratio)',font:{size:14}},
    xaxis:{title:'Mean Challenger / Champion Predicted PP Ratio per Band'},
    yaxis:{title:'Pure Premium (£)'},
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
}

/* ── Champion history ────────────────────────────────────────────────── */
function renderHistory(){
  const traces=[{
    name:'Champion Gini',
    x:HISTORY.map(d=>d[0]),y:HISTORY.map(d=>d[1]),
    mode:'lines+markers',line:{color:CHAMP_COLOR,width:2},marker:{size:8},
    hovertemplate:'Step %{x}<br>Gini: %{y:.4f}<extra></extra>',
  }];
  Plotly.react('history-chart',traces,mkLayout({
    title:{text:'Champion Gini Progression on Validation Split',font:{size:14}},
    xaxis:{title:'Promotion Step',dtick:1},
    yaxis:{title:'Validation Gini  (higher = better discrimination)'},
  }),CFG);
}

/* ── Init ────────────────────────────────────────────────────────────── */
document.getElementById('lift-bands').addEventListener('change',renderLift);
document.querySelectorAll('input[name="lift-mode"]').forEach(r=>r.addEventListener('change',renderLift));
document.getElementById('dl-bands').addEventListener('change',renderDoubleLift);
renderLift();renderDoubleLift();renderGini();renderHist();renderHistory();
"""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Comparison Report — {escape(comparison_id)}</title>
  <script src="{_PLOTLY_CDN}" crossorigin="anonymous"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;background:#f8f9fa;color:#212529;line-height:1.5}}
    .page{{max-width:1200px;margin:0 auto;padding:28px 32px}}
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
    table{{border-collapse:collapse;width:100%;font-size:13px}}
    th,td{{border:1px solid #dee2e6;padding:6px 10px;text-align:left}}
    th{{background:#f3f5f7;font-weight:600}}
    tr:nth-child(even) td{{background:#fafbfc}}
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
    <strong>{escape(decision_str.upper())}</strong>
    <span>{escape(decision.get("rationale", ""))}</span>
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
    <h2>Statistical Summary</h2>
    <div class="card" style="padding:12px">{summary_html}</div>
  </div>

  <div class="section">
    <h2>Experiment Details</h2>
    <div class="two-col">
      <div class="card">{champ_details_html}</div>
      <div class="card">{chall_details_html}</div>
    </div>
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
      Band N = highest risk). Bands contain equal total exposure. Solid lines = predicted PP;
      dashed lines = actual PP observed in that band. A well-discriminating model shows a steep,
      monotone rise with predicted and actual tracking closely.
    </p>
    <div class="controls">
      <label>Bands: <select id="lift-bands">{band_options}</select></label>
      <label><input type="radio" name="lift-mode" value="absolute" checked> Absolute PP (£)</label>
      <label><input type="radio" name="lift-mode" value="ae"> A/E Ratio</label>
    </div>
    <div class="chart-wrap"><div id="lift-chart" style="height:440px"></div></div>
  </div>

  <!-- ── DOUBLE LIFT ── -->
  <div class="section">
    <h2>Double Lift Curve</h2>
    <p class="chart-note">
      Policies sorted ascending by Challenger ÷ Champion predicted PP ratio
      (Band 1 = bands where challenger predicts relatively lower than champion; Band N = higher).
      Bands contain equal total exposure. X-axis shows the mean ratio within each band.
      The chart reveals where the two models disagree and whether the challenger's higher-risk
      bands actually observe more claims.
    </p>
    <div class="controls">
      <label>Bands: <select id="dl-bands">{band_options}</select></label>
    </div>
    <div class="chart-wrap"><div id="double-lift-chart" style="height:440px"></div></div>
  </div>

  <!-- ── GINI EXHIBIT ── -->
  <div class="section">
    <h2>Gini Lorenz Curves</h2>
    <p class="chart-note">
      Policies ranked by predicted pure premium <strong>ascending</strong> (lowest risk first —
      matching the evaluation metric convention). X = cumulative exposure share; Y = cumulative
      actual claim cost share. A well-discriminating model curves <em>below</em> the diagonal
      (low-risk policies accumulate little actual cost) → Gini &gt; 0. The diagonal = uncorrelated
      (random) predictor. Gini values match the reported <code>gini_weighted</code> metric exactly.
    </p>
    <div class="chart-wrap"><div id="gini-chart" style="height:460px"></div></div>
  </div>

  <!-- ── PREDICTION DISTRIBUTION ── -->
  <div class="section">
    <h2>Predicted Pure Premium Distribution</h2>
    <p class="chart-note">
      Exposure-weighted histogram of predicted pure premiums (1st–99th percentile shown).
      A wider spread indicates better discrimination; a spike at the grand mean indicates
      the model is producing near-constant predictions.
    </p>
    <div class="chart-wrap"><div id="hist-chart" style="height:360px"></div></div>
  </div>

  <!-- ── CHAMPION HISTORY ── -->
  <div class="section">
    <h2>Champion Gini Progression</h2>
    <p class="chart-note">Validation Gini for each successive champion in this run.</p>
    <div class="chart-wrap"><div id="history-chart" style="height:320px"></div></div>
  </div>

</div>

<script>{data_script}</script>
<script>{js_code}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Table helpers (unchanged)
# ---------------------------------------------------------------------------

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


def _details_table(title: str, details: dict[str, Any]) -> str:
    return f"<h3>{escape(title)}</h3>{_dict_table(details)}"


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
