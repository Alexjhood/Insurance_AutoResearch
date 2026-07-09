"""Deterministic experiment runner supporting all model families."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import math
from pathlib import Path
import signal
import threading
import time
import tomllib
import traceback as _tb_module
from typing import Any

import pandas as pd


class ComputeBudgetExceeded(Exception):
    """Raised when the per-experiment wall-clock budget is exceeded."""


class PreflightFailed(Exception):
    """Raised when the cheap preflight smoke-test catches a runtime error."""

    def __init__(self, message: str, *, exception_class: str = "", traceback_str: str = "") -> None:
        super().__init__(message)
        self.exception_class = exception_class
        self.traceback_str = traceback_str


@contextlib.contextmanager
def _compute_budget_alarm(budget_sec: float | None):
    """Context manager that raises ComputeBudgetExceeded if budget_sec is exceeded.

    Uses SIGALRM (POSIX only) when called on the main thread.  Silently skips
    when budget_sec is None/zero, or when not on the main thread.
    """
    if budget_sec is None or budget_sec <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise ComputeBudgetExceeded(f"SIGALRM fired after {budget_sec:.0f}s budget")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(math.ceil(budget_sec)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.data.holdout_vault import load_search_dataset
from autoresearch.data.preprocessing import DEFAULT_CAPPED_COLUMN, apply_claim_capping
from autoresearch.evaluation.metrics import evaluate_predictions
from autoresearch.experiment_registry.registry import init_registry, record_experiment
from autoresearch.models.dispatcher import (
    CLAIM_COST,
    CLAIM_COUNT,
    CLAIM_EVENTS,
    EXPOSURE,
    RAW_CLAIM_COST,
    RECORD_ID,
    dispatch_model,
)
from autoresearch.run_artifacts import next_iteration_dir
from autoresearch.utils.environment import capture_environment
from autoresearch.utils.integrity import (
    check_integrity,
    ensure_pytest_gate,
    scan_file_for_holdout_access,
    scan_file_for_non_predictive_feature_use,
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
    compute_budget_sec: float | None = None,
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
            target_mode=config.target_mode,
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
    pytest_gate = ensure_pytest_gate(config.root, config.artifacts_dir)
    if not pytest_gate["passed"]:
        msg = f"Pytest gate failed — fix tests before running experiments.\n{pytest_gate['output']}"
        record_experiment(
            config.registry_path,
            experiment_id=experiment_id,
            experiment_name=exp.get("experiment_name", "unknown"),
            model_family=exp.get("model_family", "unknown"),
            target_strategy=exp.get("target_strategy", "unknown"),
            target_mode=config.target_mode,
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

    # Preprocessing. A dataset without a declared cap can never enable capping,
    # regardless of what a (French-shaped) experiment config requests.
    preprocessing = exp.get("preprocessing", {})
    cap_spec = config.dataset.cap
    cap_enabled = bool(preprocessing.get("claim_capping_enabled", config.claim_capping_enabled)) and cap_spec is not None
    cap_threshold = float(preprocessing.get("claim_cap_threshold", config.claim_cap_threshold) or config.claim_cap_threshold)
    cap_column = cap_spec.column if cap_spec is not None else RAW_CLAIM_COST
    cap_output = cap_spec.output_column if cap_spec is not None else DEFAULT_CAPPED_COLUMN
    frame, capping_diagnostics = apply_claim_capping(
        frame,
        claim_column=cap_column,
        threshold=cap_threshold,
        enabled=cap_enabled,
        output_column=cap_output,
    )

    # Model dispatch
    model_cfg = exp.get("model", {})
    model_family = exp.get("model_family", "global_mean")
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
                target_mode=config.target_mode,
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
        feature_policy_violations = scan_file_for_non_predictive_feature_use(model_script_path)
        if feature_policy_violations:
            msg = "Model-script feature-policy scan failed:\n" + "\n".join(feature_policy_violations)
            record_experiment(
                config.registry_path,
                experiment_id=experiment_id,
                experiment_name=exp.get("experiment_name", "unknown"),
                model_family=exp.get("model_family", "unknown"),
                target_strategy=exp.get("target_strategy", "unknown"),
                target_mode=config.target_mode,
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

    dispatch_kwargs: dict[str, Any] = dict(
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
        target_mode=config.target_mode,
    )

    # ── Preflight smoke-test ──────────────────────────────────────────────────
    if getattr(config, "preflight_enabled", True):
        _run_preflight_smoke_test(frame, dispatch_kwargs, config)

    # ── Full fit with optional wall-clock budget ──────────────────────────────
    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()
    fit_wall_seconds: float | None = None
    fit_cpu_seconds: float | None = None
    timed_out = False

    effective_budget = compute_budget_sec if getattr(config, "compute_enforce", True) else None
    try:
        with _compute_budget_alarm(effective_budget):
            result = dispatch_model(**dispatch_kwargs)
    except ComputeBudgetExceeded:
        fit_wall_seconds = time.perf_counter() - t_wall_start
        timed_out = True
        msg = (
            f"Compute budget exceeded: ran {fit_wall_seconds:.1f}s, "
            f"budget {effective_budget:.0f}s. "
            "Reduce n_estimators / use early stopping / lower model complexity."
        )
        record_experiment(
            config.registry_path,
            experiment_id=experiment_id,
            experiment_name=exp.get("experiment_name", "unknown"),
            model_family=model_family,
            target_strategy=target_strategy,
            target_mode=config.target_mode,
            preprocessing_summary={},
            claim_cap_threshold=cap_threshold if cap_enabled else None,
            status="failed",
            parent_experiment_id=exp.get("parent_experiment_id") or None,
            config_snapshot_path=None,
            metrics_path=None,
            artifacts={},
            code_version=None,
            fit_wall_seconds=fit_wall_seconds,
            compute_budget_seconds=effective_budget,
            timed_out=True,
            notes=msg,
        )
        raise ComputeBudgetExceeded(msg)

    fit_wall_seconds = time.perf_counter() - t_wall_start
    fit_cpu_seconds = time.process_time() - t_cpu_start

    # ── Join feature columns into predictions ─────────────────────────────────
    # Columns that are targets, system identifiers, or already in predictions
    # are excluded from the feature join.  Everything else in the original frame
    # is treated as a potential predictor and joined so that diagnostics and
    # interpretation exhibits can reference factor values.
    from autoresearch.models import columns as _model_columns
    _LEAKAGE = {RECORD_ID, config.id_column, _model_columns.EXPOSURE} | set(_model_columns.leak_columns())
    _feature_cols = [c for c in frame.columns if c not in _LEAKAGE]
    if _feature_cols:
        _feat_df = frame[["record_id"] + _feature_cols].copy()
        predictions_enriched = result.predictions.merge(_feat_df, on="record_id", how="left")
    else:
        predictions_enriched = result.predictions

    metrics = evaluate_predictions(
        result.predictions,
        config.ordinary_eval_splits,
        tweedie_power=config.tweedie_power,
        primary_metric=config.primary_metric,
        target_mode=config.target_mode,
    )

    # Diagnostics — pass enriched frame so segment analysis sees feature columns
    from autoresearch.evaluation.diagnostics import compute_diagnostics
    diagnostics = compute_diagnostics(
        predictions_enriched,
        eval_split=config.ordinary_eval_splits[0],
        target_mode=config.target_mode,
    )

    # Automatic interpretation — computed from predictions + features, no model needed
    from autoresearch.models.interpretation import compute_automatic_interpretation
    _eval_split = config.ordinary_eval_splits[0]
    _train_split = config.ordinary_train_split
    _eval_enriched = predictions_enriched[predictions_enriched["split"] == _eval_split].copy()
    _train_enriched = predictions_enriched[predictions_enriched["split"] == _train_split].copy()
    interpretation = compute_automatic_interpretation(
        eval_df=_eval_enriched,
        feature_cols=_feature_cols,
        interpret_fn=result.interpret_fn,
        train_df=_train_enriched if not _train_enriched.empty else None,
        target_mode=config.target_mode,
    )

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
        "target_mode": config.target_mode,
        "milestone_holdout_accessed": False,
    }

    budget_util = (
        round(fit_wall_seconds / effective_budget, 4)
        if effective_budget and effective_budget > 0 and fit_wall_seconds is not None
        else None
    )
    metrics_payload = {
        **metrics,
        "experiment_id": experiment_id,
        "experiment_name": exp["experiment_name"],
        "model_family": model_family,
        "target_strategy": target_strategy,
        "target_mode": config.target_mode,
        "preprocessing": config_snapshot["effective_preprocessing"],
        "model_notes": result.model_notes,
        "timing": {
            "fit_wall_seconds": round(fit_wall_seconds, 3) if fit_wall_seconds is not None else None,
            "fit_cpu_seconds": round(fit_cpu_seconds, 3) if fit_cpu_seconds is not None else None,
            "compute_budget_seconds": effective_budget,
            "budget_utilisation": budget_util,
            "timed_out": timed_out,
        },
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
    predictions_path = run_dir / "predictions.parquet"
    capping_path = run_dir / "capping_diagnostics.json"
    diagnostics_path = run_dir / "diagnostics.json"
    interpretation_path = run_dir / "interpretation.json"
    env_path = run_dir / "environment_manifest.json"

    write_json(config_path, config_snapshot)
    write_json(metrics_path, metrics_payload)
    write_json(capping_path, capping_diagnostics)
    write_json(diagnostics_path, diagnostics)
    write_json(interpretation_path, interpretation)
    write_json(env_path, env_manifest)
    pd.DataFrame(metrics["split_metrics"]).to_csv(split_metrics_path, index=False)
    result.predictions.to_parquet(predictions_path, index=False)

    artifacts = {
        "config_snapshot": config_path,
        "metrics": metrics_path,
        "split_metrics": split_metrics_path,
        "predictions": predictions_path,
        "capping_diagnostics": capping_path,
        "diagnostics": diagnostics_path,
        "interpretation": interpretation_path,
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
        target_mode=config.target_mode,
        preprocessing_summary=config_snapshot["effective_preprocessing"],
        claim_cap_threshold=cap_threshold if cap_enabled else None,
        status="completed",
        parent_experiment_id=exp.get("parent_experiment_id") or None,
        config_snapshot_path=config_path,
        metrics_path=metrics_path,
        artifacts=artifacts,
        code_version=env_manifest.get("git_sha"),
        fit_wall_seconds=fit_wall_seconds,
        fit_cpu_seconds=fit_cpu_seconds,
        compute_budget_seconds=effective_budget,
        timed_out=timed_out,
        notes=f"Experiment with {model_family}/{target_strategy}; milestone holdout not accessed.",
    )

    # Record declarative recipes for the reuse library (#6). Best-effort.
    recipe_obj = model_cfg.get("recipe")
    if isinstance(recipe_obj, dict) and recipe_obj:
        from autoresearch.models.recipe_library import record_recipe

        eval_split = config.ordinary_eval_splits[0] if config.ordinary_eval_splits else None
        gini = next(
            (row.get("gini_weighted") for row in metrics.get("split_metrics", [])
             if row.get("split") == eval_split),
            None,
        )
        record_recipe(
            config,
            recipe_obj,
            experiment_id=experiment_id,
            outcome="completed",
            score=gini,
            target_strategy=target_strategy,
            feature_inclusions=model_cfg.get("feature_inclusions"),
            feature_exclusions=model_cfg.get("feature_exclusions"),
            model_spec={"target_strategy": target_strategy, **model_cfg},
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
    if path.is_absolute():
        return path.resolve()

    # Anchor relative paths to the experiment config's directory (the proposal dir).
    local_path = (experiment_config_path.parent / path).resolve()
    if local_path.exists():
        return local_path

    # Fallback: try from the project root to handle repo-root-relative paths like
    # "artifacts/tracks/..." stored in a proposal that was generated from a different cwd.
    # This prevents the doubled-path bug where joining such a path onto an already-nested
    # experiment_config_path.parent produces ".../artifacts/tracks/artifacts/tracks/...".
    from autoresearch.config import PROJECT_ROOT
    root_path = (PROJECT_ROOT / path).resolve()
    if root_path.exists():
        return root_path

    return local_path


def _preflight_dispatch_kwargs(dispatch_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy of the dispatch kwargs with TabPFN thinking params stripped.

    Thinking mode draws from a separate (much smaller) Prior Labs daily budget
    and is charged per fit request, so leaving it enabled here would burn a
    second thinking session per experiment on the 5k-row smoke sample. The
    smoke test only needs to prove the model runs end-to-end; the plain
    (non-thinking) fit exercises the same code path. Thinking-specific
    parameter validation still happens client-side at the real fit.
    """
    import copy

    from autoresearch.models.recipe.foundation import _THINKING_PARAMS

    kwargs = dict(dispatch_kwargs)
    hp = copy.deepcopy(kwargs.get("hyperparameters") or {})

    def _strip(params: Any) -> None:
        if isinstance(params, dict):
            for key in _THINKING_PARAMS:
                params.pop(key, None)

    _strip(hp)  # script models receive hyperparameters directly
    recipe = hp.get("recipe")
    if isinstance(recipe, dict):
        _strip(recipe.get("params"))
        stages = recipe.get("stages")
        if isinstance(stages, dict):
            for stage in stages.values():
                if isinstance(stage, dict):
                    _strip(stage.get("params"))
    kwargs["hyperparameters"] = hp
    return kwargs


def _run_preflight_smoke_test(
    frame: Any,
    dispatch_kwargs: dict[str, Any],
    config: ProjectConfig,
) -> None:
    """Run model on a small sample to catch API/type/loss-compatibility errors quickly.

    Raises PreflightFailed (with full traceback) if the model script errors on
    the sample.  Skips gracefully if the model family has no script or if the
    sample produces a degenerate split.
    """
    n = min(getattr(config, "preflight_sample_rows", 5000), len(frame))
    if n <= 0:
        return

    dispatch_kwargs = _preflight_dispatch_kwargs(dispatch_kwargs)

    sample = frame.sample(n=n, random_state=42)
    # Filter split_frame to sampled rows so the splits are consistent.
    sample_splits = dispatch_kwargs.get("split_frame")
    if sample_splits is not None and "record_id" in sample_splits.columns:
        sample_splits = sample_splits[sample_splits["record_id"].isin(sample["record_id"])]

    effective_splits = (
        sample_splits
        if (sample_splits is not None and len(sample_splits) > 0)
        else dispatch_kwargs.get("split_frame")
    )
    try:
        from autoresearch.models.dispatcher import dispatch_model as _dm
        _dm(**{**dispatch_kwargs, "frame": sample, "split_frame": effective_splits})
    except Exception:
        tb = _tb_module.format_exc()
        import sys
        exc = sys.exc_info()[1]
        raise PreflightFailed(
            str(exc),
            exception_class=type(exc).__name__,
            traceback_str=tb,
        ) from exc
