"""Idempotent setup for isolated research tracks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig
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


def bootstrap_track(
    config: ProjectConfig,
    *,
    prepare_shared_data: bool = True,
    force_prepare_data: bool = False,
    run_baselines: bool = True,
) -> dict[str, Any]:
    """Prepare everything an agent needs to run an isolated research track.

    The operation is intentionally idempotent. Existing data, registry rows,
    and champion state are reused unless a force flag requires rebuilding.
    """

    if config.track_id == "default":
        raise ValueError("bootstrap-track requires --track <name>; do not bootstrap the shared default registry.")

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
        baseline_runs, baseline_errors = _run_baselines_resilient(config)
        steps.append(
            {
                "step": "run-all-baselines",
                "status": "ran_with_errors" if baseline_errors else "ran",
                "runs": [
                    {name: str(path) for name, path in outputs.items()}
                    for outputs in baseline_runs
                ],
                "errors": baseline_errors,
            }
        )
        if baseline_errors and not list_experiments(config.registry_path):
            raise ValueError("No baseline experiments completed during bootstrap: " + "; ".join(baseline_errors))
    elif run_baselines:
        steps.append(
            {
                "step": "run-all-baselines",
                "status": "skipped",
                "reason": f"{len(experiments)} experiment(s) already registered",
            }
        )
    else:
        steps.append({"step": "run-all-baselines", "status": "skipped", "reason": "disabled by caller"})

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
        "registry": str(config.registry_path),
        "context": str(context_outputs["latest_context_json"]),
        "handoff": str(context_outputs["latest_handoff_markdown"]),
        "steps": steps,
    }


def _required_prepared_data_paths(config: ProjectConfig) -> tuple[Path, ...]:
    # The holdout parquet and capping diagnostics are part of a complete
    # prepare-data run.  Omitting them let a data plane that is missing the
    # holdout (so milestone evaluation would later fail) look "already prepared"
    # and skip re-preparation.
    return (
        config.processed_dir / f"{config.agent_dataset_name}.parquet",
        config.processed_dir / "agent_dataset_search.parquet",
        config.holdout_vault_dir / "agent_dataset_holdout.parquet",
        config.metadata_dir / "agent_schema.json",
        config.metadata_dir / "dataset_profile.json",
        config.metadata_dir / "capping_diagnostics.json",
        config.splits_dir / "split_pack.csv",
        config.splits_dir / "split_pack_manifest.json",
        config.splits_dir / "split_pack_folds.parquet",
    )


def _run_baselines_resilient(config: ProjectConfig) -> tuple[list[dict[str, Path]], list[str]]:
    """Run checked-in baselines, preserving successful runs if one fails."""

    runs: list[dict[str, Path]] = []
    errors: list[str] = []
    exp_dir = config.root / "configs" / "experiments"
    base_dir = bootstrap_iteration_dir(config) / "baseline_experiments"
    for path in sorted(exp_dir.glob("*.toml")):
        try:
            runs.append(run_experiment(config, path, output_dir=base_dir / path.stem))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    return runs, errors
