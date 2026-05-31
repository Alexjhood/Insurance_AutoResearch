"""Experiment registry scaffolding."""

from autoresearch.experiment_registry.schema import init_registry, registry_counts
from autoresearch.experiment_registry.experiments import (
    record_experiment,
    get_experiment,
    list_experiments,
    list_artifacts,
    record_experiment_artifacts,
)
from autoresearch.experiment_registry.comparisons import record_comparison, list_comparisons, update_comparison_decision
from autoresearch.experiment_registry.champions import (
    set_official_champion,
    get_official_champion,
    list_champion_history,
)
from autoresearch.experiment_registry.branches import upsert_branch, list_branches
from autoresearch.experiment_registry.proposals import (
    record_proposal,
    update_proposal_status,
    next_queued_proposal,
    list_proposals,
    get_proposal,
)
from autoresearch.experiment_registry.sessions import (
    upsert_session,
    record_session_event,
    list_sessions,
    get_session,
    list_session_events,
)
from autoresearch.experiment_registry.research_tree import (
    upsert_research_node,
    list_research_nodes,
    find_research_node_by_experiment,
)

__all__ = [
    "init_registry",
    "registry_counts",
    "record_experiment",
    "get_experiment",
    "list_experiments",
    "list_artifacts",
    "record_experiment_artifacts",
    "record_comparison",
    "list_comparisons",
    "update_comparison_decision",
    "set_official_champion",
    "get_official_champion",
    "list_champion_history",
    "upsert_branch",
    "list_branches",
    "record_proposal",
    "update_proposal_status",
    "next_queued_proposal",
    "list_proposals",
    "get_proposal",
    "upsert_session",
    "record_session_event",
    "list_sessions",
    "get_session",
    "list_session_events",
    "upsert_research_node",
    "list_research_nodes",
    "find_research_node_by_experiment",
]
