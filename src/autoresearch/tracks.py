"""Cross-track champion comparison.

Compares the official champions of two isolated research tracks (e.g. 'claude'
vs 'codex') without promoting either.  Each track has its own registry,
artifacts directory, and research log, so neither agent has ever seen the
other's experiment history.  This function is the privileged view that sits
above both tracks and produces an informational report only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.config import PROJECT_ROOT, ProjectConfig
from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    paired_comparison,
    promotion_decision,
)
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_artifacts,
)
from autoresearch.utils.io import write_json


_REPORT_TEMPLATE = """\
# Cross-Track Comparison Report

Comparison date : {date}
Track A         : `{track_a}` — champion `{champion_a_id}` ({model_a})
Track B         : `{track_b}` — champion `{champion_b_id}` ({model_b})

---

## Head-to-head on search-validation ({n_resamples} paired resamples)

| | Track A | Track B |
|-|--------:|--------:|
| Mean Tweedie deviance (p=1.5) | {score_a:.6f} | {score_b:.6f} |
| Gini (weighted) | {gini_a:.4f} | {gini_b:.4f} |
| Pred/actual ratio | {pa_a:.4f} | {pa_b:.4f} |

**Winner**: {winner}
**Mean lift** ({winner_label} over {loser_label}): {abs_lift:.6f} ({rel_lift:.2%} relative)
**Win rate**: {win_rate:.1%} of resamples
**Bootstrap 90% CI on lift**: [{boot_lower:+.6f}, {boot_upper:+.6f}]
**P({winner_label} outperforms {loser_label})**: {p_winner:.1%}

---

## Gate simulation — would either track's champion promote over the other?

### Would Track B beat Track A (B as challenger)?
{gate_b_beats_a}

### Would Track A beat Track B (A as challenger)?
{gate_a_beats_b}

---

## Per-resample lift distribution (Track A score − Track B score; negative = B wins)

| Statistic | Value |
|-----------|------:|
| Mean | {lift_mean:+.6f} |
| Median | {lift_median:+.6f} |
| Std dev | {lift_std:.6f} |
| Min | {lift_min:+.6f} |
| Max | {lift_max:+.6f} |

---

*No promotion performed.  This is an informational cross-track comparison.*
*Re-run with `autoresearch compare-tracks {track_a} {track_b}` at any time.*
"""


def compare_tracks(
    config_a: ProjectConfig,
    config_b: ProjectConfig,
) -> dict[str, Any]:
    """Compare the official champions of two research tracks.

    Parameters
    ----------
    config_a, config_b:
        ProjectConfig objects loaded with ``load_config(track_id=...)`` for
        each track.  The shared evaluation parameters (resampling counts,
        bootstrap iterations, thresholds) are taken from *config_a*; both
        configs should ordinarily come from the same default.toml.

    Returns
    -------
    dict with keys: comparison_id, track_a, track_b, champion_a_id,
    champion_b_id, winner, mean_lift, win_rate, report_path, json_path.
    Never raises — failures are captured and returned as {"status": "error"}.
    """

    try:
        return _run_comparison(config_a, config_b)
    except Exception as exc:
        import traceback
        return {
            "status": "error",
            "track_a": config_a.track_id,
            "track_b": config_b.track_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _run_comparison(config_a: ProjectConfig, config_b: ProjectConfig) -> dict[str, Any]:
    champion_a = _require_champion(config_a)
    champion_b = _require_champion(config_b)

    champion_a_id = champion_a["champion_id"]
    champion_b_id = champion_b["champion_id"]

    predictions_a = _load_predictions(config_a, champion_a_id)
    predictions_b = _load_predictions(config_b, champion_b_id)

    eval_split = config_a.ordinary_eval_splits[0]

    # Paired comparison — A as "base", B as "challenger".
    # lift = A_score - B_score; positive means B wins (lower deviance is better).
    per_resample, summary = paired_comparison(
        predictions_a,
        predictions_b,
        champion_id=f"{config_a.track_id}/{champion_a_id}",
        challenger_id=f"{config_b.track_id}/{champion_b_id}",
        eval_split=eval_split,
        n_resamples=config_a.repeated_resamples,
        seed=config_a.resampling_seed,
        resample_fraction=config_a.resample_fraction,
        tweedie_power=config_a.tweedie_power,
        primary_metric=config_a.primary_metric,
    )

    # Bootstrap CI on lift (no Bonferroni — this is a single planned comparison)
    bootstrap = bootstrap_lift_summary(
        per_resample["lift"],
        iterations=config_a.bootstrap_iterations,
        seed=config_a.resampling_seed + 1,
        confidence_level=config_a.confidence_level,
        n_comparisons=1,
    )

    rules = PromotionRules(
        minimum_mean_lift=config_a.minimum_mean_lift,
        min_relative_lift=config_a.min_relative_lift,
        min_absolute_lift=config_a.min_absolute_lift,
        minimum_win_rate=config_a.minimum_win_rate,
        bootstrap_lower_bound=config_a.bootstrap_lower_bound,
        bootstrap_lower_bound_relative=config_a.bootstrap_lower_bound_relative,
        confidence_level=config_a.confidence_level,
        max_predicted_to_actual_drift=config_a.max_predicted_to_actual_drift,
        require_diagnostics=False,
        bonferroni_lookback=1,
    )

    # Gate sim: B beats A (lift > 0 means B wins; summary already has B as challenger)
    decision_b_beats_a = promotion_decision(summary, bootstrap, rules, n_prior_comparisons=0)

    # Gate sim: A beats B — flip champion/challenger roles
    _, summary_flipped = paired_comparison(
        predictions_b,
        predictions_a,
        champion_id=f"{config_b.track_id}/{champion_b_id}",
        challenger_id=f"{config_a.track_id}/{champion_a_id}",
        eval_split=eval_split,
        n_resamples=config_a.repeated_resamples,
        seed=config_a.resampling_seed,
        resample_fraction=config_a.resample_fraction,
        tweedie_power=config_a.tweedie_power,
        primary_metric=config_a.primary_metric,
    )
    bootstrap_flipped = bootstrap_lift_summary(
        pd.Series(-per_resample["lift"].to_numpy()),
        iterations=config_a.bootstrap_iterations,
        seed=config_a.resampling_seed + 1,
        confidence_level=config_a.confidence_level,
        n_comparisons=1,
    )
    decision_a_beats_b = promotion_decision(summary_flipped, bootstrap_flipped, rules, n_prior_comparisons=0)

    # Derive scalar summary values
    score_a = float(summary["champion_mean_score"])
    score_b = float(summary["challenger_mean_score"])
    mean_lift = float(summary["mean_lift"])     # positive = B wins
    win_rate_b = float(summary["challenger_win_rate"])

    # Per-split scalar metrics for report
    gini_a, pa_a = _scalar_metrics(predictions_a, eval_split, config_a.tweedie_power)
    gini_b, pa_b = _scalar_metrics(predictions_b, eval_split, config_b.tweedie_power)

    if mean_lift > 0:
        winner, loser = config_b.track_id, config_a.track_id
        abs_lift = mean_lift
        rel_lift = abs_lift / score_a if score_a else 0.0
        win_rate = win_rate_b
        p_winner = float(bootstrap["probability_challenger_outperforms"])
    else:
        winner, loser = config_a.track_id, config_b.track_id
        abs_lift = -mean_lift
        rel_lift = abs_lift / score_b if score_b else 0.0
        win_rate = 1.0 - win_rate_b
        p_winner = 1.0 - float(bootstrap["probability_challenger_outperforms"])

    # Build output paths
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    comparison_id = f"{timestamp}_{config_a.track_id}_vs_{config_b.track_id}"
    out_dir = PROJECT_ROOT / "artifacts" / "cross_track" / comparison_id
    out_dir.mkdir(parents=True, exist_ok=True)

    per_resample_path = out_dir / "paired_resample_scores.csv"
    per_resample.to_csv(per_resample_path, index=False)

    gate_b_beats_a_lines = _format_gate_checks(decision_b_beats_a["checks"])
    gate_a_beats_b_lines = _format_gate_checks(decision_a_beats_b["checks"])

    lifts = per_resample["lift"]
    model_a = champion_a.get("model_family") or "unknown"
    model_b = champion_b.get("model_family") or "unknown"

    report_text = _REPORT_TEMPLATE.format(
        date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        track_a=config_a.track_id,
        track_b=config_b.track_id,
        champion_a_id=champion_a_id,
        champion_b_id=champion_b_id,
        model_a=model_a,
        model_b=model_b,
        n_resamples=config_a.repeated_resamples,
        score_a=score_a,
        score_b=score_b,
        gini_a=gini_a,
        gini_b=gini_b,
        pa_a=pa_a,
        pa_b=pa_b,
        winner=winner,
        loser=loser,
        winner_label=winner,
        loser_label=loser,
        abs_lift=abs_lift,
        rel_lift=rel_lift,
        win_rate=win_rate,
        boot_lower=float(bootstrap["interval_lower"]),
        boot_upper=float(bootstrap["interval_upper"]),
        p_winner=p_winner,
        gate_b_beats_a=gate_b_beats_a_lines,
        gate_a_beats_b=gate_a_beats_b_lines,
        lift_mean=float(lifts.mean()),
        lift_median=float(lifts.median()),
        lift_std=float(lifts.std(ddof=0)),
        lift_min=float(lifts.min()),
        lift_max=float(lifts.max()),
    )

    report_path = out_dir / "comparison_report.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary_payload = {
        "comparison_id": comparison_id,
        "track_a": config_a.track_id,
        "track_b": config_b.track_id,
        "champion_a_id": champion_a_id,
        "champion_b_id": champion_b_id,
        "model_a": model_a,
        "model_b": model_b,
        "score_a": score_a,
        "score_b": score_b,
        "mean_lift": mean_lift,
        "win_rate_b_over_a": win_rate_b,
        "bootstrap": bootstrap,
        "gate_b_beats_a": decision_b_beats_a,
        "gate_a_beats_b": decision_a_beats_b,
        "winner": winner,
        "status": "completed",
        "report_path": str(report_path),
    }
    json_path = out_dir / "comparison_summary.json"
    write_json(json_path, summary_payload)

    summary_payload["json_path"] = str(json_path)
    return summary_payload


# ── private helpers ───────────────────────────────────────────────────────────

def _require_champion(config: ProjectConfig) -> dict[str, Any]:
    champion = get_official_champion(config.registry_path)
    if champion is None:
        raise ValueError(
            f"Track '{config.track_id}' has no official champion. "
            "Run `autoresearch --track {track_id} init-official-champion` first."
        )
    # Enrich with model_family from experiment record
    try:
        from autoresearch.experiment_registry.registry import get_experiment
        exp = get_experiment(config.registry_path, champion["champion_id"])
        champion = dict(champion)
        champion["model_family"] = exp.get("model_family")
    except Exception:
        pass
    return champion


def _load_predictions(config: ProjectConfig, experiment_id: str) -> pd.DataFrame:
    artifacts = list_artifacts(config.registry_path, experiment_id)
    for artifact in artifacts:
        if artifact["artifact_type"] == "predictions":
            return pd.read_csv(artifact["path"])
    raise FileNotFoundError(
        f"No predictions artifact for experiment '{experiment_id}' in track '{config.track_id}'"
    )


def _scalar_metrics(
    predictions: pd.DataFrame,
    eval_split: str,
    tweedie_power: float,
) -> tuple[float, float]:
    """Return (gini_weighted, predicted_to_actual_ratio) for the eval split."""
    from autoresearch.evaluation.metrics import full_metric_panel
    frame = predictions[predictions["split"] == eval_split]
    if frame.empty:
        return float("nan"), float("nan")
    panel = full_metric_panel(
        frame["actual_claim_cost"],
        frame["predicted_claim_cost"],
        frame["exposure"],
        tweedie_power=tweedie_power,
    )
    return float(panel["gini_weighted"]), float(panel["predicted_to_actual_ratio"])


def _format_gate_checks(checks: dict[str, bool]) -> str:
    lines = ["| Check | Result |", "|-------|--------|"]
    for name, passed in checks.items():
        icon = "✓ pass" if passed else "✗ FAIL"
        lines.append(f"| `{name}` | {icon} |")
    overall = "**WOULD PROMOTE**" if all(checks.values()) else "**would NOT promote**"
    lines.append(f"\n{overall}")
    return "\n".join(lines)
