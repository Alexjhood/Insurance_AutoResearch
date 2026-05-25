"""HTML reporting for run-scoped experiment comparisons."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

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
    """Write a self-contained comparison report and return its path."""

    champion_predictions = pd.read_csv(_artifact_path(config, champion_id, "predictions"))
    challenger_predictions = pd.read_csv(_artifact_path(config, challenger_id, "predictions"))
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

    lift_svg = _line_chart(
        [
            ("Champion", _lift_curve(champion_eval)),
            ("Challenger", _lift_curve(challenger_eval)),
        ],
        "Cumulative Exposure Share",
        "Cumulative Actual Claim Share",
    )
    double_lift_svg = _line_chart(
        _double_lift_series(champion_eval, challenger_eval),
        "Decile",
        "Pure Premium",
    )
    history_svg = _line_chart(
        [("Champion Gini", _champion_history_points(config, eval_split))],
        "Champion Step",
        "Validation Gini",
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Comparison Report {escape(comparison_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #182026; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d4d9de; padding: 7px 9px; text-align: left; font-size: 13px; }}
    th {{ background: #f3f5f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .panel {{ margin: 18px 0 28px; }}
    .meta {{ color: #53606a; }}
    svg {{ width: 100%; height: auto; border: 1px solid #d4d9de; background: #fff; }}
    .decision {{ font-weight: 700; }}
  </style>
</head>
<body>
  <h1>Comparison Report</h1>
  <p class="meta">Comparison: <code>{escape(comparison_id)}</code><br>
  Evaluation split: <code>{escape(eval_split)}</code>; primary gate metric:
  <code>{escape(config.primary_metric)}</code>.</p>

  <h2>Decision</h2>
  <p class="decision">{escape(decision.get("decision", "?"))}: {escape(decision.get("rationale", ""))}</p>
  {_summary_table(comparison_summary, bootstrap_summary)}

  <h2>Experiment Details</h2>
  <div class="grid">
    <div>{_details_table("Champion", _experiment_details(config, champion_id))}</div>
    <div>{_details_table("Challenger", _experiment_details(config, challenger_id))}</div>
  </div>

  <h2>Validation Metrics</h2>
  {_metrics_table(champion_metrics, challenger_metrics)}

  <div class="panel">
    <h2>Lift Curves</h2>
    {lift_svg}
  </div>

  <div class="panel">
    <h2>Double Lift Curve</h2>
    {double_lift_svg}
  </div>

  <div class="panel">
    <h2>Champion History</h2>
    {history_svg}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


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


def _lift_curve(frame: pd.DataFrame) -> list[tuple[float, float]]:
    ordered = frame.sort_values("predicted_pure_premium", ascending=False)
    exposure = ordered["exposure"].astype(float)
    actual = ordered["actual_claim_cost"].astype(float)
    total_exposure = max(float(exposure.sum()), 1e-12)
    total_actual = max(float(actual.sum()), 1e-12)
    points = [(0.0, 0.0)]
    points.extend(zip((exposure.cumsum() / total_exposure).tolist(), (actual.cumsum() / total_actual).tolist()))
    return [(float(x), float(y)) for x, y in points]


def _double_lift_series(champion: pd.DataFrame, challenger: pd.DataFrame) -> list[tuple[str, list[tuple[float, float]]]]:
    paired = champion[["record_id", "actual_pure_premium", "predicted_pure_premium", "exposure"]].merge(
        challenger[["record_id", "predicted_pure_premium"]],
        on="record_id",
        suffixes=("_champion", "_challenger"),
    )
    paired["ratio"] = paired["predicted_pure_premium_challenger"] / paired["predicted_pure_premium_champion"].clip(lower=1e-9)
    paired = paired.sort_values("ratio")
    bin_count = min(10, max(len(paired), 1))
    ranks = paired["ratio"].rank(method="first")
    paired["decile"] = ((ranks - 1) / max(len(paired), 1) * bin_count).astype(int) + 1
    series = {"Actual": [], "Champion": [], "Challenger": []}
    for decile, group in paired.groupby("decile", sort=True):
        weights = group["exposure"].astype(float)
        x = float(decile)
        series["Actual"].append((x, float((group["actual_pure_premium"] * weights).sum() / weights.sum())))
        series["Champion"].append((x, float((group["predicted_pure_premium_champion"] * weights).sum() / weights.sum())))
        series["Challenger"].append((x, float((group["predicted_pure_premium_challenger"] * weights).sum() / weights.sum())))
    return list(series.items())


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
            split_metric = next(item for item in metrics["split_metrics"] if item["split"] == eval_split)
            points.append((float(len(points) + 1), float(split_metric["gini_weighted"])))
        except Exception:
            continue
    return points or [(1.0, 0.0)]


def _summary_table(comparison_summary: dict[str, Any], bootstrap_summary: dict[str, Any]) -> str:
    return _dict_table(
        {
            "Mean lift": comparison_summary.get("mean_lift"),
            "Win rate": comparison_summary.get("challenger_win_rate"),
            "Champion mean score": comparison_summary.get("champion_mean_score"),
            "Challenger mean score": comparison_summary.get("challenger_mean_score"),
            "Bootstrap lower": bootstrap_summary.get("interval_lower"),
            "Bootstrap upper": bootstrap_summary.get("interval_upper"),
        }
    )


def _details_table(title: str, details: dict[str, Any]) -> str:
    return f"<h3>{escape(title)}</h3>{_dict_table(details)}"


def _metrics_table(champion: dict[str, Any], challenger: dict[str, Any]) -> str:
    rows = ["<table><tr><th>Metric</th><th>Champion</th><th>Challenger</th><th>Delta</th></tr>"]
    for key in METRIC_KEYS:
        c = champion.get(key)
        h = challenger.get(key)
        delta = h - c if isinstance(c, (int, float)) and isinstance(h, (int, float)) else ""
        rows.append(f"<tr><td>{escape(key)}</td><td>{_fmt(c)}</td><td>{_fmt(h)}</td><td>{_fmt(delta)}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _dict_table(values: dict[str, Any]) -> str:
    rows = ["<table>"]
    for key, value in values.items():
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{escape(_fmt(value))}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _line_chart(series: list[tuple[str, list[tuple[float, float]]]], x_label: str, y_label: str) -> str:
    width, height, pad = 760, 320, 44
    all_points = [point for _, points in series for point in points] or [(0.0, 0.0), (1.0, 1.0)]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x = min_x + 1
    if min_y == max_y:
        max_y = min_y + 1

    def sx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - min_y) / (max_y - min_y) * (height - 2 * pad)

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img">']
    parts.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#6b737a"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#6b737a"/>')
    parts.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12">{escape(x_label)}</text>')
    parts.append(f'<text x="14" y="{height/2}" transform="rotate(-90 14 {height/2})" text-anchor="middle" font-size="12">{escape(y_label)}</text>')
    for idx, (name, points) in enumerate(series):
        color = colors[idx % len(colors)]
        polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{width-pad-120}" y="{pad + idx*18}" fill="{color}" font-size="12">{escape(name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={_fmt(val)}" for key, val in value.items())
    if isinstance(value, list):
        return ", ".join(_fmt(item) for item in value)
    return str(value)
