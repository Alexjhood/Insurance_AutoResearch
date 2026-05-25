"""Official champion state management."""

from __future__ import annotations

from autoresearch.config import ProjectConfig
from autoresearch.experiment_registry.registry import (
    get_experiment,
    get_official_champion,
    list_experiments,
    set_official_champion,
    upsert_branch,
)


OFFICIAL_BRANCH_ID = "main"


def initialise_official_champion(config: ProjectConfig, experiment_id: str | None = None) -> dict[str, object]:
    """Initialise official champion as the no-model global-mean baseline.

    Every research run starts from this flat exposure-weighted burning-cost
    baseline; every subsequent proposal develops relative to it.
    """

    selected_id = experiment_id or _starting_baseline_experiment(config)
    experiment = get_experiment(config.registry_path, selected_id)
    if experiment.get("target_strategy") != "direct_pure_premium":
        raise ValueError("Official starting champion must be a direct pure premium experiment")

    upsert_branch(
        config.registry_path,
        branch_id=OFFICIAL_BRANCH_ID,
        parent_branch_id=None,
        root_experiment_id=selected_id,
        current_experiment_id=selected_id,
        status="active",
        description="Official research branch rooted at the global-mean no-model baseline.",
    )
    reason = (
        "Official starting champion set by product decision: the no-model global-mean "
        "burning cost is the first iteration of every run. All later models must demonstrate "
        "real lift over this flat rate to enter the champion lineage."
    )
    set_official_champion(
        config.registry_path,
        champion_id=selected_id,
        branch_id=OFFICIAL_BRANCH_ID,
        reason=reason,
        action="initialised",
    )
    return get_official_champion(config.registry_path) or {}


def _starting_baseline_experiment(config: ProjectConfig) -> str:
    """Pick the global-mean baseline; fall back to any completed direct-pp run."""

    completed = [
        row
        for row in list_experiments(config.registry_path)
        if row.get("status") == "completed" and row.get("target_strategy") == "direct_pure_premium"
    ]
    if not completed:
        raise ValueError(
            "No completed baseline exists. Run `autoresearch run-all-baselines` "
            "(which executes the global_mean starting baseline) first."
        )
    for row in completed:
        if row.get("model_family") == "global_mean":
            return row["experiment_id"]
    return completed[0]["experiment_id"]
