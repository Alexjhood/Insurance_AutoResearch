"""Shim: re-exports all public symbols for backward-compatible imports.

Existing callers using:
    from autoresearch.experiment_registry.registry import X
continue to work without modification.
"""

from autoresearch.experiment_registry import (  # noqa: F401
    init_registry,
    registry_counts,
    record_experiment,
    get_experiment,
    list_experiments,
    list_artifacts,
    record_experiment_artifacts,
    record_comparison,
    list_comparisons,
    update_comparison_decision,
    set_official_champion,
    get_official_champion,
    list_champion_history,
    upsert_branch,
    list_branches,
    record_proposal,
    update_proposal_status,
    next_queued_proposal,
    list_proposals,
    get_proposal,
    upsert_session,
    record_session_event,
    list_sessions,
    get_session,
    list_session_events,
    upsert_research_node,
    list_research_nodes,
    find_research_node_by_experiment,
)
