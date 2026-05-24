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
    """Initialise official champion as the direct pure-premium baseline."""

    selected_id = experiment_id or _latest_direct_pure_premium(config)
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
        description="Official research branch rooted at the direct pure premium baseline.",
    )
    reason = (
        "Official starting champion set by product decision: direct pure premium baseline "
        "starts the controlled research process even if another baseline has a better point estimate."
    )
    set_official_champion(
        config.registry_path,
        champion_id=selected_id,
        branch_id=OFFICIAL_BRANCH_ID,
        reason=reason,
        action="initialised",
    )
    return get_official_champion(config.registry_path) or {}


def _latest_direct_pure_premium(config: ProjectConfig) -> str:
    rows = [
        row
        for row in list_experiments(config.registry_path)
        if row.get("status") == "completed" and row.get("target_strategy") == "direct_pure_premium"
    ]
    if not rows:
        raise ValueError("No completed direct pure premium baseline exists. Run that baseline first.")
    return rows[0]["experiment_id"]
