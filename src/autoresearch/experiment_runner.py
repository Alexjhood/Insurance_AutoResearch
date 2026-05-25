"""Deterministic experiment runner supporting all model families."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.holdout_vault import load_search_dataset
from autoresearch.data.preprocessing import apply_claim_capping
from autoresearch.evaluation.metrics import evaluate_predictions
from autoresearch.experiment_registry.registry import init_registry, record_experiment
from autoresearch.models.baselines import RAW_CLAIM_COST
from autoresearch.models.dispatcher import dispatch_model
from autoresearch.run_artifacts import next_iteration_dir
from autoresearch.utils.environment import capture_environment
from autoresearch.utils.integrity import (
    check_integrity,
    run_pytest,
    scan_file_for_holdout_access,
    scan_for_holdout_access,
)
from autoresearch.utils.io import write_json


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load an experiment TOML file."""

    with path.open("rb") as f:
        return tomllib.load(f)


def run_experiment(
    config: ProjectConfig,
    experiment_config_path: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Run one deterministic experiment end to end."""

    ensure_project_dirs(config)
    init_registry(config.registry_path)
    exp = load_experiment_config(experiment_config_path)

    experiment_id = _experiment_id(exp["experiment_name"])

    # ── Gate 1: holdout-access scan ──────────────────────────────────────────
    holdout_violations = scan_for_holdout_access(config.root)
    if holdout_violations:
        msg = "Holdout-access scan failed:\n" + "\n".join(holdout_violations)
        record_experiment(
            config.registry_path,
            experiment_id=experiment_id,
            experiment_name=exp.get("experiment_name", "unknown"),
            model_family=exp.get("model_family", "unknown"),
            target_strategy=exp.get("target_strategy", "unknown"),
            preprocessing_summary={},
            claim_cap_threshold=None,
            status="failed",
            parent_experiment_id=None,
            config_snapshot_path=None,
            metrics_path=None,
            artifacts={},
            code_version=None,
            notes=msg,
        )
        raise ValueError(msg)

    # ── Gate 2: mandatory pytest ─────────────────────────────────────────────
    pytest_passed, pytest_output = run_pytest(config.root)
    if not pytest_passed:
        msg = f"Pytest gate failed — fix tests before running experiments.\n{pytest_output}"
        record_experiment(
            config.registry_path,
            experiment_id=experiment_id,
            experiment_name=exp.get("experiment_name", "unknown"),
            model_family=exp.get("model_family", "unknown"),
            target_strategy=exp.get("target_strategy", "unknown"),
            preprocessing_summary={},
            claim_cap_threshold=None,
            status="failed",
            parent_experiment_id=None,
            config_snapshot_path=None,
            metrics_path=None,
            artifacts={},
            code_version=None,
            notes=msg,
        )
        raise ValueError(msg)
    run_dir = output_dir or (next_iteration_dir(config, exp["experiment_name"]) / "experiment")
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = load_search_dataset(config.processed_dir, config.agent_dataset_name)
    split_frame = pd.read_csv(config.splits_dir / "split_pack.csv")

    # Preprocessing
    preprocessing = exp.get("preprocessing", {})
    cap_enabled = bool(preprocessing.get("claim_capping_enabled", config.claim_capping_enabled))
    cap_threshold = float(preprocessing.get("claim_cap_threshold", config.claim_cap_threshold))
    frame, capping_diagnostics = apply_claim_capping(
        frame,
        claim_column=RAW_CLAIM_COST,
        threshold=cap_threshold,
        enabled=cap_enabled,
    )

    # Model dispatch
    model_cfg = exp.get("model", {})
    model_family = exp.get("model_family", "regularized_linear")
    target_strategy = exp.get("target_strategy", "direct_pure_premium")
    hyperparameters = {k: v for k, v in model_cfg.items()
                       if k not in {"feature_inclusions", "feature_exclusions"}}
    model_script_path = _resolve_model_script_path(experiment_config_path, model_cfg)
    if model_script_path is not None:
        script_violations = scan_file_for_holdout_access(model_script_path)
        if script_violations:
            msg = "Model-script holdout-access scan failed:\n" + "\n".join(script_violations)
            record_experiment(
                config.registry_path,
                experiment_id=experiment_id,
                experiment_name=exp.get("experiment_name", "unknown"),
                model_family=exp.get("model_family", "unknown"),
                target_strategy=exp.get("target_strategy", "unknown"),
                preprocessing_summary={},
                claim_cap_threshold=None,
                status="failed",
                parent_experiment_id=None,
                config_snapshot_path=None,
                metrics_path=None,
                artifacts={},
                code_version=None,
                notes=msg,
            )
            raise ValueError(msg)
        hyperparameters.pop("script_path", None)
        hyperparameters.pop("model_script_path", None)
        hyperparameters.pop("script_sha256", None)

    result = dispatch_model(
        frame=frame,
        split_frame=split_frame,
        model_family=model_family,
        target_strategy=target_strategy,
        train_split=config.ordinary_train_split,
        score_splits=config.ordinary_eval_splits,
        hyperparameters=hyperparameters,
        feature_inclusions=model_cfg.get("feature_inclusions"),
        feature_exclusions=model_cfg.get("feature_exclusions") or None,
        model_script_path=model_script_path,
    )

    metrics = evaluate_predictions(
        result.predictions,
        config.ordinary_eval_splits,
        tweedie_power=config.tweedie_power,
        primary_metric=config.primary_metric,
    )

    # Diagnostics
    from autoresearch.evaluation.diagnostics import compute_diagnostics
    diagnostics = compute_diagnostics(result.predictions, eval_split=config.ordinary_eval_splits[0])

    config_snapshot = {
        "experiment_id": experiment_id,
        "experiment_config_path": str(experiment_config_path),
        "experiment": exp,
        "model_script_path": str(model_script_path) if model_script_path else None,
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
        "model_family": model_family,
        "target_strategy": target_strategy,
        "preprocessing": config_snapshot["effective_preprocessing"],
        "model_notes": result.model_notes,
    }

    # Environment manifest
    env_manifest = capture_environment(
        config.root,
        data_files={
            "split_pack": config.splits_dir / "split_pack.csv",
            "agent_dataset_search": config.processed_dir / "agent_dataset_search.parquet",
        },
    )

    config_path = run_dir / "config_snapshot.json"
    metrics_path = run_dir / "metrics.json"
    split_metrics_path = run_dir / "split_metrics.csv"
    predictions_path = run_dir / "predictions.csv"
    capping_path = run_dir / "capping_diagnostics.json"
    diagnostics_path = run_dir / "diagnostics.json"
    env_path = run_dir / "environment_manifest.json"

    write_json(config_path, config_snapshot)
    write_json(metrics_path, metrics_payload)
    write_json(capping_path, capping_diagnostics)
    write_json(diagnostics_path, diagnostics)
    write_json(env_path, env_manifest)
    pd.DataFrame(metrics["split_metrics"]).to_csv(split_metrics_path, index=False)
    result.predictions.to_csv(predictions_path, index=False)

    artifacts = {
        "config_snapshot": config_path,
        "metrics": metrics_path,
        "split_metrics": split_metrics_path,
        "predictions": predictions_path,
        "capping_diagnostics": capping_path,
        "diagnostics": diagnostics_path,
        "environment_manifest": env_path,
    }
    if model_script_path is not None:
        artifacts["model_script"] = model_script_path
    record_experiment(
        config.registry_path,
        experiment_id=experiment_id,
        experiment_name=exp["experiment_name"],
        model_family=model_family,
        target_strategy=target_strategy,
        preprocessing_summary=config_snapshot["effective_preprocessing"],
        claim_cap_threshold=cap_threshold if cap_enabled else None,
        status="completed",
        parent_experiment_id=exp.get("parent_experiment_id") or None,
        config_snapshot_path=config_path,
        metrics_path=metrics_path,
        artifacts=artifacts,
        code_version=env_manifest.get("git_sha"),
        notes=f"Experiment with {model_family}/{target_strategy}; milestone holdout not accessed.",
    )
    return artifacts


def run_all_baselines(config: ProjectConfig) -> list[dict[str, Path]]:
    """Run all checked-in baseline experiment configs."""

    exp_dir = config.root / "configs" / "experiments"
    return [run_experiment(config, path) for path in sorted(exp_dir.glob("*.toml"))]


def _experiment_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in name)
    return f"{stamp}_{safe_name}"


def _resolve_model_script_path(experiment_config_path: Path, model_cfg: dict[str, Any]) -> Path | None:
    raw = model_cfg.get("script_path") or model_cfg.get("model_script_path")
    if not raw:
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = experiment_config_path.parent / path
    return path.resolve()
