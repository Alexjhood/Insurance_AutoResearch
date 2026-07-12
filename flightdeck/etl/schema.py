"""Snapshot dataclasses — the single place snapshot field names live.

Mirrors ``flightdeck/DATA.md`` exactly. ``app/src/lib/types.ts`` must stay in
lockstep with this module (SPEC §8). Any schema change bumps
``SNAPSHOT_SCHEMA_VERSION`` and updates DATA.md + types.ts together.

Design notes:
- Every optional field defaults to ``None`` (DATA.md: "Missing/unknown values:
  null, never omitted keys") so a defensively-degraded snapshot still carries
  the full key set.
- ``Any`` is used where DATA.md declares ``unknown`` (raw passthrough blobs).
- Serialisation is via :func:`to_jsonable` (``dataclasses.asdict`` with a couple
  of guards) so the emitted JSON key names come straight from the field names.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

SNAPSHOT_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# Shared value objects
# --------------------------------------------------------------------------- #
@dataclass
class TokenTotals:
    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0  # reasoning_output_tokens


# --------------------------------------------------------------------------- #
# index.json
# --------------------------------------------------------------------------- #
@dataclass
class IndexEntry:
    orch_id: str
    alias: Optional[str]
    dataset: str
    target_mode: str
    status: str
    created_at: str
    ended_at: Optional[str]
    orchestrator_model: str
    backends: list[str]
    n_delegations: int
    cycles_committed: int
    cycles_used: int
    cycles_forfeited: int
    final_gini: Optional[float]
    baseline_gini: Optional[float]
    total_tokens: TokenTotals
    cache_hit_rate: Optional[float]
    wall_clock_minutes: Optional[float]
    distress_count: int
    takeover_count: int
    champion_spark: list[float]


@dataclass
class SnapshotIndex:
    orchestrations: list[IndexEntry] = field(default_factory=list)
    built_at: str = ""
    snapshot_schema_version: int = SNAPSHOT_SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# snapshot.json — campaign / delegation
# --------------------------------------------------------------------------- #
@dataclass
class BuildInfo:
    built_at: str
    source_mtime: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class Consolidation:
    run_id: str
    track: str


@dataclass
class Campaign:
    orch_id: str
    dataset: str
    target_mode: str
    status: str
    created_at: str
    ended_at: Optional[str]
    cycles_committed: int
    orchestrator_model: str
    consolidation: Optional[Consolidation]
    framework_computed: Any  # campaign_report.json .framework_computed passthrough
    campaign_report_md: Optional[str]
    orchestration_log_md: Optional[str]


@dataclass
class Brief:
    name: Optional[str]
    direction: str
    constraints: list[str]
    cycle_budget: int
    starting_knowledge: list[str]
    foundation_models: bool
    seed_champion: Optional[dict[str, Any]]  # {experiment_id, from_run}


@dataclass
class ModelIdentity:
    provider: str
    name: str
    harness: str


@dataclass
class DelegationBudget:
    committed: int
    attempted: int
    completed: int
    decided: int
    used: int
    forfeited: int
    refunded: bool


@dataclass
class DelegationChampion:
    experiment_id: str
    gini_weighted: float
    rank_gini_weighted: Optional[float]
    asym_pricing_loss: Optional[float]
    calibration_ratio: Optional[float]
    model_family: str
    target_strategy: str
    beat_seed_baseline: Optional[bool]


@dataclass
class Distress:
    active: list[str]
    all_flags: list[str]
    detail: Optional[str]


@dataclass
class RepairSummary:
    max_attempts_in_a_cycle: int
    total_attempts: int


@dataclass
class DelegationCost:
    tokens: TokenTotals
    model_calls: int
    tool_calls: int
    tool_failures: int
    cache_hit_rate: Optional[float]
    wall_clock_minutes: Optional[float]
    cost_usd: Optional[float]


@dataclass
class DelegationFiles:
    prompt: Optional[str] = None
    brief: Optional[str] = None
    report: Optional[str] = None
    stdout_log: Optional[str] = None
    exit_json: Optional[str] = None
    research_log: Optional[str] = None
    llm_usage: Optional[str] = None


@dataclass
class Delegation:
    delegation_id: str
    backend: str
    track: str
    run_id: str
    run_path: str
    command: list[str]
    resolved_executable: Optional[str]
    resolved_version: Optional[str]
    model_identity: Optional[ModelIdentity]
    status: str
    clean_exit: Optional[bool]
    exit_code: Optional[int]
    spawned_at: Optional[str]
    ended_at: Optional[str]
    timeout_minutes: Optional[float]
    continue_run: bool
    respawn_of: Optional[str]
    taken_over: bool
    budget: DelegationBudget
    brief: Brief
    agent_summary: Optional[str]
    champion: Optional[DelegationChampion]
    distress: Distress
    repairs: Optional[RepairSummary]
    cost: DelegationCost
    files: DelegationFiles


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    hypothesis: Optional[str]
    change_summary: Optional[str]
    expected_benefit: Optional[str]
    key_risk: Optional[str]
    exploration_axis: Optional[str]
    approach_family: Optional[str]


@dataclass
class ExperimentMetrics:
    gini_weighted: Optional[float]
    rank_gini_weighted: Optional[float]
    asym_pricing_loss: Optional[float]
    calibration_ratio: Optional[float]
    fit_wall_seconds: Optional[float]


@dataclass
class Screening:
    gini_weighted: Optional[float]
    lift: Optional[float]
    win_rate: Optional[float]


@dataclass
class Comparison:
    comparison_id: str
    champion_id: str
    cv_mean_lift: Optional[float]
    fold_win_rate: Optional[float]
    decision: Optional[str]  # promote | local_promote | reject | null
    decided_by: Optional[str]
    decision_reason_code: Optional[str]
    decision_rationale: Optional[str]
    guardrail_status: Optional[str]


@dataclass
class DecisionMeta:
    interpretation: Optional[str]
    next_step: Optional[str]
    outcome: Optional[str]


@dataclass
class Lift:
    vs_then_champion: Optional[float]
    vs_baseline: Optional[float]
    kind: Optional[str]  # 'baseline_relative' | 'incremental' | null


@dataclass
class RepairAttempt:
    attempt: int
    kind: Optional[str]  # 'recipe' | 'script' | null
    failed_checks: list[str]
    resolved: bool
    raw: Any


@dataclass
class ExperimentUsage:
    tokens: TokenTotals
    total_cumulative: int
    model_calls: int
    tool_calls: int
    tool_failures: int
    cache_hit_rate: Optional[float]


@dataclass
class TreeNode:
    node_id: str
    parent_node_id: Optional[str]


@dataclass
class Experiment:
    experiment_id: str
    delegation_id: Optional[str]  # null ⇒ consolidation/playoff run
    cycle: Optional[int]
    seq: int
    created_at: str
    name: str
    status: str
    is_baseline: bool
    is_seed: bool
    parent_experiment_id: Optional[str]
    model_family: str
    target_strategy: str
    recipe: Any  # model.recipe or {script: true, path}
    proposal: Proposal
    metrics: ExperimentMetrics
    screening: Optional[Screening]
    comparison: Optional[Comparison]
    decision_meta: DecisionMeta
    lift: Lift
    repairs: list[RepairAttempt]
    usage: Optional[ExperimentUsage]
    research_line_id: Optional[str]
    tree: Optional[TreeNode]


# --------------------------------------------------------------------------- #
# Champion timeline / notes / playoff
# --------------------------------------------------------------------------- #
@dataclass
class ChampionEvent:
    at: str
    delegation_id: Optional[str]
    action: str
    previous_champion_id: Optional[str]
    new_champion_id: str
    new_champion_gini: Optional[float]
    comparison_id: Optional[str]
    scope: str  # 'delegation' | 'consolidation'
    is_seed_transfer: bool


@dataclass
class OperatorNote:
    at: str
    kind: str  # 'reflection' | 'takeover' | 'decision' | str
    delegation_id: Optional[str]
    text: str


@dataclass
class PlayoffFinalist:
    delegation_id: str
    experiment_id: str
    gini_weighted: float
    model_family: str
    replay_experiment_id: Optional[str]


@dataclass
class PlayoffExclusion:
    delegation_id: Optional[str]
    reason: str


@dataclass
class PlayoffFinal:
    delegation_id: str
    source_experiment_id: str
    consolidation_experiment_id: str


@dataclass
class Playoff:
    decision_mode: str
    consolidation: Consolidation
    finalists: list[PlayoffFinalist]
    exclusions: list[PlayoffExclusion]
    final: Optional[PlayoffFinal]
    report_md: Optional[str]


# --------------------------------------------------------------------------- #
# Telemetry summary (in snapshot.json) + files manifest
# --------------------------------------------------------------------------- #
@dataclass
class TelemetryByDelegation:
    delegation_id: str
    tokens: TokenTotals
    model_calls: int
    tool_calls: int
    tool_failures: int
    cache_hit_rate: Optional[float]


@dataclass
class ToolMixEntry:
    delegation_id: str
    tool: str
    calls: int
    failures: int
    total_duration_ms: float


@dataclass
class TelemetryTotals:
    input: int = 0
    cached_input: int = 0
    output: int = 0
    reasoning: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    tool_failures: int = 0


@dataclass
class TelemetrySummary:
    totals: TelemetryTotals
    cache_hit_rate: Optional[float]
    by_delegation: list[TelemetryByDelegation]
    tool_mix: list[ToolMixEntry]


@dataclass
class FileEntry:
    path: str
    kind: str  # 'markdown' | 'json' | 'log' | 'text'
    bytes: int
    truncated: bool
    source: str


@dataclass
class Snapshot:
    snapshot_schema_version: int
    build: BuildInfo
    campaign: Campaign
    delegations: list[Delegation]
    experiments: list[Experiment]
    champion_timeline: list[ChampionEvent]
    notes: list[OperatorNote]
    playoff: Optional[Playoff]
    telemetry_summary: TelemetrySummary
    files: list[FileEntry]


# --------------------------------------------------------------------------- #
# telemetry_<dNN>.json (lazy per-delegation event lists)
# --------------------------------------------------------------------------- #
@dataclass
class ModelCallEvent:
    at: str
    model: str
    input: int
    cached_input: int
    output: int
    reasoning: int
    duration_hint_ms: None  # reserved; not stored
    workflow_event_id: Optional[int]


@dataclass
class ToolCallEvent:
    name: str
    detail: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_ms: Optional[float]
    status: Optional[str]
    success: Optional[bool]
    input_bytes: Optional[int]
    output_bytes: Optional[int]
    error_type: Optional[str]
    workflow_event_id: Optional[int]


@dataclass
class WorkflowEvent:
    id: int
    command: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_ms: Optional[float]
    status: Optional[str]
    error_type: Optional[str]


@dataclass
class TelemetryCheckpoint:
    experiment_name: str
    completed_at: Optional[str]
    total_cumulative: int


@dataclass
class DelegationTelemetry:
    delegation_id: str
    model_calls: list[ModelCallEvent]
    tool_calls: list[ToolCallEvent]
    workflow_events: list[WorkflowEvent]
    checkpoints: list[TelemetryCheckpoint]


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses to plain JSON-serialisable structures."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
