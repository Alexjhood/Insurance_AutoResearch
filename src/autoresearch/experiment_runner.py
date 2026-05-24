"""Deterministic Phase 2 experiment runner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tomllib
from typing import Any

import pandas as pd

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.preprocessing import apply_claim_capping
from autoresearch.evaluation.metrics import evaluate_predictions
from autoresearch.experiment_registry.registry import init_registry, record_experiment
from autoresearch.models.baselines import RAW_CLAIM_COST, run_baseline_model
from autoresearch.utils.io import write_json


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load an experiment TOML file."""

    with path.open("rb") as f:
        return tomllib.load(f)


def run_experiment(config: ProjectConfig, experiment_config_path: Path) -> dict[str, Path]:
    """Run one deterministic baseline experiment end to end."""

    ensure_project_dirs(config)
    init_registry(config.registry_path)
    exp = load_experiment_config(experiment_config_path)

    experiment_id = _experiment_id(exp["experiment_name"])
    run_dir = config.artifacts_dir / "experiments" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(config.processed_dir / f"{config.agent_dataset_name}.parquet")
    split_frame = pd.read_csv(config.splits_dir / "split_pack.csv")

    preprocessing = exp.get("preprocessing", {})
    cap_enabled = bool(preprocessing.get("claim_capping_enabled", config.claim_capping_enabled))
    cap_threshold = float(preprocessing.get("claim_cap_threshold", config.claim_cap_threshold))
    frame, capping_diagnostics = apply_claim_capping(
        frame,
        claim_column=RAW_CLAIM_COST,
        threshold=cap_threshold,
        enabled=cap_enabled,
    )

    model_config = exp.get("model", {})
    result = run_baseline_model(
        frame=frame,
        split_frame=split_frame,
        target_strategy=exp["target_strategy"],
        train_split=config.ordinary_train_split,
        score_splits=config.ordinary_eval_splits,
        alpha=float(model_config.get("alpha", 1.0)),
        feature_inclusions=model_config.get("feature_inclusions"),
        feature_exclusions=model_config.get("feature_exclusions"),
    )
    metrics = evaluate_predictions(result.predictions, config.ordinary_eval_splits)

    config_snapshot = {
        "experiment_id": experiment_id,
        "experiment_config_path": str(experiment_config_path),
        "experiment": exp,
        "project_preprocessing_defaults": {
            "claim_capping_enabled": config.claim_capping_enabled,
            "claim_cap_threshold": config.claim_cap_threshold,
        },
        "effective_preprocessing": {
            "claim_capping_enabled": cap_enabled,
            "claim_cap_threshold": cap_threshold,
        },
        "ordinary_train_split": config.ordinary_train_split,
        "ordinary_eval_splits": list(config.ordinary_eval_splits),
        "milestone_holdout_accessed": False,
    }

    metrics_payload = {
        **metrics,
        "experiment_id": experiment_id,
        "experiment_name": exp["experiment_name"],
        "model_family": exp["model_family"],
        "target_strategy": exp["target_strategy"],
        "preprocessing": config_snapshot["effective_preprocessing"],
        "model_notes": result.model_notes,
    }

    config_path = run_dir / "config_snapshot.json"
    metrics_path = run_dir / "metrics.json"
    split_metrics_path = run_dir / "split_metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    capping_path = run_dir / "capping_diagnostics.json"

    write_json(config_path, config_snapshot)
    write_json(metrics_path, metrics_payload)
    write_json(capping_path, capping_diagnostics)
    pd.DataFrame(metrics["split_metrics"]).to_csv(split_metrics_path, index=False)
    result.predictions.to_csv(predictions_path, index=False)

    artifacts = {
        "config_snapshot": config_path,
        "metrics": metrics_path,
        "split_metrics": split_metrics_path,
        "predictions": predictions_path,
        "capping_diagnostics": capping_path,
    }
    record_experiment(
        config.registry_path,
        experiment_id=experiment_id,
        experiment_name=exp["experiment_name"],
        model_family=exp["model_family"],
        target_strategy=exp["target_strategy"],
        preprocessing_summary=config_snapshot["effective_preprocessing"],
        claim_cap_threshold=cap_threshold if cap_enabled else None,
        status="completed",
        parent_experiment_id=exp.get("parent_experiment_id") or None,
        config_snapshot_path=config_path,
        metrics_path=metrics_path,
        artifacts=artifacts,
        code_version=_git_hash(config.root),
        notes="Deterministic Phase 2 baseline; milestone holdout not accessed.",
    )
    return artifacts


def run_all_baselines(config: ProjectConfig) -> list[dict[str, Path]]:
    """Run all checked-in baseline experiment configs."""

    exp_dir = config.root / "configs" / "experiments"
    return [run_experiment(config, path) for path in sorted(exp_dir.glob("*.toml"))]


def _experiment_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
    return f"{stamp}_{safe_name}"


def _git_hash(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
