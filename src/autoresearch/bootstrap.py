"""Idempotent setup for isolated research tracks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.controller.champion import initialise_official_champion
from autoresearch.controller.handoff import export_context_bundle, write_proposal_template
from autoresearch.data.pipeline import prepare_data
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    init_registry,
    list_experiments,
)
from autoresearch.experiment_runner import run_experiment
from autoresearch.run_artifacts import bootstrap_iteration_dir
from autoresearch.utils.integrity import ensure_pytest_gate


def bootstrap_track(
    config: ProjectConfig,
    *,
    prepare_shared_data: bool = True,
    force_prepare_data: bool = False,
    run_baselines: bool = True,
    default_max_cycles: int | None = None,
    enable_foundation_models: bool = False,
) -> dict[str, Any]:
    """Prepare everything an agent needs to run an isolated research track.

    The operation is intentionally idempotent. Existing data, registry rows,
    and champion state are reused unless a force flag requires rebuilding.
    """

    if config.track_id == "default":
        raise ValueError("bootstrap-track requires --track <name>; do not bootstrap the shared default registry.")

    if not config.model_provider or not config.model_name:
        raise ValueError(
            "bootstrap-track requires model identity. "
            "Pass --model-provider and --model-name (e.g. --model-provider anthropic --model-name claude-sonnet-4-6). "
            "Use --model-version and --harness for additional context."
        )

    # Persist the selected run before any fallible gate so retries and harness
    # scope binding cannot fall back to a previous run.
    ensure_project_dirs(config)

    if default_max_cycles is not None:
        if default_max_cycles <= 0:
            raise ValueError("--cycles must be a positive integer")
        _pin_default_max_cycles(config, default_max_cycles)

    _set_manifest_flag(config, "foundation_models", bool(enable_foundation_models))
    if enable_foundation_models:
        apply_foundation_models_gate(config)

    steps: list[dict[str, Any]] = []

    if prepare_shared_data:
        required_data_paths = _required_prepared_data_paths(config)
        missing = [path for path in required_data_paths if not path.exists()]
        if force_prepare_data or missing:
            outputs = prepare_data(config)
            steps.append(
                {
                    "step": "prepare-data",
                    "status": "ran",
                    "outputs": {name: str(path) for name, path in outputs.items()},
                }
            )
        else:
            steps.append(
                {
                    "step": "prepare-data",
                    "status": "skipped",
                    "reason": "required shared data artifacts already exist",
                }
            )
    else:
        steps.append({"step": "prepare-data", "status": "skipped", "reason": "disabled by caller"})

    test_gate = ensure_pytest_gate(config.root, config.artifacts_dir)
    steps.append(
        {
            "step": "test-gate",
            "status": (
                "skipped" if test_gate.get("skipped")
                else "cached" if test_gate.get("cached")
                else "ran"
            ),
            "duration_seconds": test_gate.get("duration_seconds"),
            "output": test_gate.get("output"),
        }
    )
    if not test_gate["passed"]:
        raise ValueError("Pytest gate failed during bootstrap:\n" + str(test_gate["output"]))

    registry_existed = config.registry_path.exists()
    registry_path = init_registry(config.registry_path)
    steps.append(
        {
            "step": "init-registry",
            "status": "skipped" if registry_existed else "ran",
            "registry": str(registry_path),
        }
    )

    experiments = list_experiments(config.registry_path)
    if run_baselines and not experiments:
        baseline_runs, baseline_errors = _run_starting_baseline(config)
        steps.append(
            {
                "step": "run-starting-baseline",
                "status": "ran_with_errors" if baseline_errors else "ran",
                "runs": [
                    {name: str(path) for name, path in outputs.items()}
                    for outputs in baseline_runs
                ],
                "errors": baseline_errors,
            }
        )
        if baseline_errors and not list_experiments(config.registry_path):
            raise ValueError("The global-mean starting baseline did not complete during bootstrap: " + "; ".join(baseline_errors))
    elif run_baselines:
        steps.append(
            {
                "step": "run-starting-baseline",
                "status": "skipped",
                "reason": f"{len(experiments)} experiment(s) already registered",
            }
        )
    else:
        steps.append({"step": "run-starting-baseline", "status": "skipped", "reason": "disabled by caller"})

    champion = get_official_champion(config.registry_path)
    if champion is None:
        champion = initialise_official_champion(config)
        steps.append({"step": "init-official-champion", "status": "ran", "champion": champion["champion_id"]})
    else:
        steps.append(
            {
                "step": "init-official-champion",
                "status": "skipped",
                "champion": champion["champion_id"],
            }
        )

    template_outputs = write_proposal_template(config)
    steps.append(
        {
            "step": "write-proposal-template",
            "status": "ran",
            "outputs": {name: str(path) for name, path in template_outputs.items()},
        }
    )

    context_outputs = export_context_bundle(config)
    steps.append(
        {
            "step": "export-context",
            "status": "ran",
            "outputs": {name: str(path) for name, path in context_outputs.items()},
        }
    )

    return {
        "track": config.track_id,
        "run_id": config.run_id,
        "run_dir": str(config.artifacts_dir),
        "default_max_cycles": default_max_cycles,
        "registry": str(config.registry_path),
        "context": str(context_outputs["latest_context_json"]),
        "handoff": str(context_outputs["latest_handoff_markdown"]),
        "steps": steps,
    }


def _manifest_path(config: ProjectConfig) -> Path:
    return config.artifacts_dir / "run_manifest.json"


def read_run_manifest(config: ProjectConfig) -> dict[str, Any]:
    """Return the run manifest dict (empty if it does not exist yet)."""

    import json

    path = _manifest_path(config)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _set_manifest_flag(config: ProjectConfig, key: str, value: Any) -> None:
    import json

    manifest = read_run_manifest(config)
    manifest[key] = value
    _manifest_path(config).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _pin_default_max_cycles(config: ProjectConfig, cycles: int) -> None:
    """Record the run's cycle budget so sessions enforce it framework-side."""

    _set_manifest_flag(config, "default_max_cycles", int(cycles))


def apply_foundation_models_gate(config: ProjectConfig) -> list[str]:
    """Enable foundation estimators for a run that opted in.

    Reads the run manifest's ``foundation_models`` flag; when set, exports
    ``AUTORESEARCH_FOUNDATION_MODELS`` (so any in-process recipe import — now or
    later — self-registers) and registers the estimators into the live registry
    immediately (import-order independent). A no-op when the run did not opt in.
    Returns the estimator names that became available.
    """

    if not read_run_manifest(config).get("foundation_models"):
        return []
    import os

    os.environ["AUTORESEARCH_FOUNDATION_MODELS"] = "1"
    from autoresearch.models.recipe import enable_foundation_models

    return enable_foundation_models()


def _required_prepared_data_paths(config: ProjectConfig) -> tuple[Path, ...]:
    # The holdout parquet and capping diagnostics are part of a complete
    # prepare-data run.  Omitting them let a data plane that is missing the
    # holdout (so milestone evaluation would later fail) look "already prepared"
    # and skip re-preparation.
    return (
        config.processed_dir / f"{config.agent_dataset_name}.parquet",
        config.processed_dir / "agent_dataset_search.parquet",
        config.holdout_vault_dir / "agent_dataset_holdout.parquet",
        config.metadata_dir / "dataset_schema.json",
        config.metadata_dir / "dataset_profile.json",
        config.metadata_dir / "capping_diagnostics.json",
        config.splits_dir / "split_pack.csv",
        config.splits_dir / "split_pack_manifest.json",
        config.splits_dir / "split_pack_folds.parquet",
    )


def _run_starting_baseline(config: ProjectConfig) -> tuple[list[dict[str, Path]], list[str]]:
    """Run only the global-mean model used to initialise every research run."""

    runs: list[dict[str, Path]] = []
    errors: list[str] = []
    path = config.root / "configs" / "experiments" / "global_mean.toml"
    base_dir = bootstrap_iteration_dir(config) / "baseline_experiments"
    if not path.is_file():
        return runs, [f"{path.name}: starting baseline config not found at {path}"]

    try:
        runs.append(run_experiment(config, path, output_dir=base_dir / path.stem))
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")
    return runs, errors
