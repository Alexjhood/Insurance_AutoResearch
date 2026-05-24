"""Phase 3 volatility-aware repeated evaluation and comparison workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    paired_comparison,
    promotion_decision,
    repeated_scores,
)
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    get_experiment,
    init_registry,
    list_experiments,
    record_comparison,
    record_experiment_artifacts,
)
from autoresearch.utils.io import write_json


def run_repeated_evaluation(config: ProjectConfig, experiment_id: str) -> dict[str, Path]:
    """Create repeated search-time scores for one registered experiment."""

    ensure_project_dirs(config)
    init_registry(config.registry_path)
    experiment = get_experiment(config.registry_path, experiment_id)
    predictions_path = _artifact_path(config, experiment_id, "predictions")
    predictions = pd.read_csv(predictions_path)
    eval_split = config.ordinary_eval_splits[0]

    scores = repeated_scores(
        predictions,
        eval_split=eval_split,
        n_resamples=config.repeated_resamples,
        seed=config.resampling_seed,
        resample_fraction=config.resample_fraction,
    )
    summary = {
        "experiment_id": experiment_id,
        "experiment_name": experiment.get("experiment_name"),
        "eval_split": eval_split,
        "primary_metric": "rmse_pure_premium",
        "lower_is_better": True,
        "n_resamples": config.repeated_resamples,
        "resample_fraction": config.resample_fraction,
        "seed": config.resampling_seed,
        "mean_score": float(scores["score"].mean()),
        "median_score": float(scores["score"].median()),
        "std_score": float(scores["score"].std(ddof=0)),
    }

    out_dir = config.artifacts_dir / "experiments" / experiment_id
    score_path = out_dir / "repeated_scores.csv"
    summary_path = out_dir / "repeated_summary.json"
    scores.to_csv(score_path, index=False)
    write_json(summary_path, summary)
    record_experiment_artifacts(
        config.registry_path,
        experiment_id,
        {"repeated_scores": score_path, "repeated_summary": summary_path},
    )
    return {"repeated_scores": score_path, "repeated_summary": summary_path}


def compare_experiments(config: ProjectConfig, champion_id: str, challenger_id: str) -> dict[str, Path]:
    """Run a paired volatility-aware comparison and persist promotion evidence."""

    ensure_project_dirs(config)
    init_registry(config.registry_path)
    champion_predictions = pd.read_csv(_artifact_path(config, champion_id, "predictions"))
    challenger_predictions = pd.read_csv(_artifact_path(config, challenger_id, "predictions"))
    eval_split = config.ordinary_eval_splits[0]

    per_resample, comparison_summary = paired_comparison(
        champion_predictions,
        challenger_predictions,
        champion_id=champion_id,
        challenger_id=challenger_id,
        eval_split=eval_split,
        n_resamples=config.repeated_resamples,
        seed=config.resampling_seed,
        resample_fraction=config.resample_fraction,
    )
    bootstrap = bootstrap_lift_summary(
        per_resample["lift"],
        iterations=config.bootstrap_iterations,
        seed=config.resampling_seed + 1,
        confidence_level=config.confidence_level,
    )
    decision = promotion_decision(
        comparison_summary,
        bootstrap,
        PromotionRules(
            minimum_mean_lift=config.minimum_mean_lift,
            minimum_win_rate=config.minimum_win_rate,
            bootstrap_lower_bound=config.bootstrap_lower_bound,
            confidence_level=config.confidence_level,
        ),
    )

    comparison_id = _comparison_id(champion_id, challenger_id)
    out_dir = config.artifacts_dir / "comparisons" / comparison_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_resample_path = out_dir / "paired_resample_scores.csv"
    comparison_path = out_dir / "comparison_summary.json"
    bootstrap_path = out_dir / "bootstrap_summary.json"
    decision_path = out_dir / "promotion_decision.json"
    report_path = out_dir / "promotion_report.json"

    payload = {
        "comparison_id": comparison_id,
        "comparison_summary": comparison_summary,
        "bootstrap_summary": bootstrap,
        "promotion_decision": decision,
    }
    per_resample.to_csv(per_resample_path, index=False)
    write_json(comparison_path, comparison_summary)
    write_json(bootstrap_path, bootstrap)
    write_json(decision_path, decision)
    write_json(report_path, payload)

    artifacts = {
        "paired_resample_scores": per_resample_path,
        "comparison_summary": comparison_path,
        "bootstrap_summary": bootstrap_path,
        "promotion_decision": decision_path,
        "promotion_report": report_path,
    }
    record_comparison(
        config.registry_path,
        comparison_id=comparison_id,
        champion_id=champion_id,
        challenger_id=challenger_id,
        paired_summary=comparison_summary,
        bootstrap_summary=bootstrap,
        promotion_decision=decision["decision"],
        promotion_rationale=decision["rationale"],
        artifacts=artifacts,
    )
    return artifacts


def compare_against_current_champion(config: ProjectConfig, challenger_id: str) -> dict[str, Path]:
    """Compare challenger against official champion, falling back to point-estimate champion."""

    official = get_official_champion(config.registry_path)
    champion = official["champion_id"] if official else current_champion_id(config)
    if champion == challenger_id:
        raise ValueError("Challenger is already the current champion")
    return compare_experiments(config, champion, challenger_id)


def current_champion_id(config: ProjectConfig) -> str:
    """Return the lowest search-time mean-score experiment id."""

    rows = [row for row in list_experiments(config.registry_path) if row.get("mean_score") is not None]
    if not rows:
        raise ValueError("No scored experiments are available")
    return min(rows, key=lambda row: row["mean_score"])["experiment_id"]


def _artifact_path(config: ProjectConfig, experiment_id: str, artifact_type: str) -> Path:
    from autoresearch.experiment_registry.registry import list_artifacts

    artifacts = list_artifacts(config.registry_path, experiment_id)
    for artifact in artifacts:
        if artifact["artifact_type"] == artifact_type:
            return Path(artifact["path"])
    raise ValueError(f"Experiment {experiment_id} has no {artifact_type!r} artifact")


def _comparison_id(champion_id: str, challenger_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{_short(champion_id)}_vs_{_short(challenger_id)}"


def _short(value: str) -> str:
    return value.replace(" ", "_")[:48]
