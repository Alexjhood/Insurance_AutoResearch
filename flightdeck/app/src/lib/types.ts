export interface TokenTotals { input: number; cached_input: number; output: number; reasoning: number }
export interface IndexEntry {
  orch_id: string; alias: string | null; dataset: string; target_mode: string; status: string; created_at: string;
  ended_at: string | null; orchestrator_model: string; backends: string[]; n_delegations: number;
  cycles_committed: number; cycles_used: number; cycles_forfeited: number; final_gini: number | null;
  baseline_gini: number | null; total_tokens: TokenTotals; cache_hit_rate: number | null;
  wall_clock_minutes: number | null; distress_count: number; takeover_count: number; champion_spark: number[];
}
export interface SnapshotIndex { orchestrations: IndexEntry[]; built_at: string; snapshot_schema_version: number }
export interface BuildInfo { built_at: string; source_mtime: string; warnings: string[] }
export interface Consolidation { run_id: string; track: string }
export interface Campaign { orch_id: string; dataset: string; target_mode: string; status: string; created_at: string; ended_at: string | null; cycles_committed: number; orchestrator_model: string; consolidation: Consolidation | null; framework_computed: unknown; campaign_report_md: string | null; orchestration_log_md: string | null }
export interface Brief { name: string | null; direction: string; constraints: string[]; cycle_budget: number; starting_knowledge: string[]; foundation_models: boolean; seed_champion: { experiment_id: string; from_run: string } | null }
export interface ModelIdentity { provider: string; name: string; harness: string }
export interface DelegationBudget { committed: number; attempted: number; completed: number; decided: number; used: number; forfeited: number; refunded: boolean }
export interface DelegationChampion { experiment_id: string; gini_weighted: number; rank_gini_weighted: number | null; asym_pricing_loss: number | null; calibration_ratio: number | null; model_family: string; target_strategy: string; beat_seed_baseline: boolean | null }
export interface Distress { active: string[]; all_flags: string[]; detail: string | null }
export interface RepairSummary { max_attempts_in_a_cycle: number; total_attempts: number }
export interface DelegationCost { tokens: TokenTotals; model_calls: number; tool_calls: number; tool_failures: number; cache_hit_rate: number | null; wall_clock_minutes: number | null; cost_usd: number | null }
export interface DelegationFiles { prompt: string | null; brief: string | null; report: string | null; stdout_log: string | null; exit_json: string | null; research_log: string | null; llm_usage: string | null }
export interface Delegation { delegation_id: string; backend: string; track: string; run_id: string; run_path: string; command: string[]; resolved_executable: string | null; resolved_version: string | null; model_identity: ModelIdentity | null; status: string; clean_exit: boolean | null; exit_code: number | null; spawned_at: string | null; ended_at: string | null; timeout_minutes: number | null; continue_run: boolean; respawn_of: string | null; taken_over: boolean; budget: DelegationBudget; brief: Brief; agent_summary: string | null; champion: DelegationChampion | null; distress: Distress; repairs: RepairSummary | null; cost: DelegationCost; files: DelegationFiles }
export interface Proposal { hypothesis: string | null; change_summary: string | null; expected_benefit: string | null; key_risk: string | null; exploration_axis: string | null; approach_family: string | null }
export interface ExperimentMetrics { gini_weighted: number | null; rank_gini_weighted: number | null; asym_pricing_loss: number | null; calibration_ratio: number | null; fit_wall_seconds: number | null }
export interface Screening { gini_weighted: number | null; lift: number | null; win_rate: number | null }
export type Decision = 'promote' | 'local_promote' | 'reject';
export interface Comparison { comparison_id: string; champion_id: string; cv_mean_lift: number | null; fold_win_rate: number | null; decision: Decision | null; decided_by: string | null; decision_reason_code: string | null; decision_rationale: string | null; guardrail_status: string | null }
export interface DecisionMeta { interpretation: string | null; next_step: string | null; outcome: string | null }
export interface Lift { vs_then_champion: number | null; vs_baseline: number | null; kind: 'baseline_relative' | 'incremental' | null }
export interface RepairAttempt { attempt: number; kind: 'recipe' | 'script' | null; failed_checks: string[]; resolved: boolean; raw: unknown }
export interface ExperimentUsage { tokens: TokenTotals; total_cumulative: number; model_calls: number; tool_calls: number; tool_failures: number; cache_hit_rate: number | null }
export interface TreeNode { node_id: string; parent_node_id: string | null }
export interface Experiment { experiment_id: string; delegation_id: string | null; cycle: number | null; seq: number; created_at: string; name: string; status: string; is_baseline: boolean; is_seed: boolean; parent_experiment_id: string | null; model_family: string; target_strategy: string; recipe: unknown | null; proposal: Proposal; metrics: ExperimentMetrics; screening: Screening | null; comparison: Comparison | null; decision_meta: DecisionMeta; lift: Lift; repairs: RepairAttempt[]; usage: ExperimentUsage | null; research_line_id: string | null; tree: TreeNode | null }
export interface ChampionEvent { at: string; delegation_id: string | null; action: string; previous_champion_id: string | null; new_champion_id: string; new_champion_gini: number | null; comparison_id: string | null; scope: 'delegation' | 'consolidation'; is_seed_transfer: boolean }
export interface OperatorNote { at: string; kind: string; delegation_id: string | null; text: string }
export interface PlayoffFinalist { delegation_id: string; experiment_id: string; gini_weighted: number; model_family: string; replay_experiment_id: string | null }
export interface PlayoffExclusion { delegation_id: string | null; reason: string }
export interface PlayoffFinal { delegation_id: string; source_experiment_id: string; consolidation_experiment_id: string }
export interface Playoff { decision_mode: string; consolidation: Consolidation; finalists: PlayoffFinalist[]; exclusions: PlayoffExclusion[]; final: PlayoffFinal | null; report_md: string | null }
export interface TelemetryByDelegation { delegation_id: string; tokens: TokenTotals; model_calls: number; tool_calls: number; tool_failures: number; cache_hit_rate: number | null }
export interface ToolMixEntry { delegation_id: string; tool: string; calls: number; failures: number; total_duration_ms: number }
export interface TelemetryTotals extends TokenTotals { model_calls: number; tool_calls: number; tool_failures: number }
export interface TelemetrySummary { totals: TelemetryTotals; cache_hit_rate: number | null; by_delegation: TelemetryByDelegation[]; tool_mix: ToolMixEntry[] }
export interface FileEntry { path: string; kind: 'markdown' | 'json' | 'log' | 'text'; bytes: number; truncated: boolean; source: string; included_in_export?: boolean }
export interface Snapshot { snapshot_schema_version: number; build: BuildInfo; campaign: Campaign; delegations: Delegation[]; experiments: Experiment[]; champion_timeline: ChampionEvent[]; notes: OperatorNote[]; playoff: Playoff | null; telemetry_summary: TelemetrySummary; files: FileEntry[] }
export interface ModelCallEvent { at: string; model: string; input: number; cached_input: number; output: number; reasoning: number; duration_hint_ms: null; workflow_event_id: number | null }
export interface ToolCallEvent { name: string; detail: string | null; started_at: string | null; completed_at: string | null; duration_ms: number | null; status: string | null; success: boolean | null; input_bytes: number | null; output_bytes: number | null; error_type: string | null; workflow_event_id: number | null }
export interface WorkflowEvent { id: number; command: string; started_at: string | null; completed_at: string | null; duration_ms: number | null; status: string | null; error_type: string | null }
export interface TelemetryCheckpoint { experiment_name: string; completed_at: string | null; total_cumulative: number }
export interface DelegationTelemetry { delegation_id: string; model_calls: ModelCallEvent[]; tool_calls: ToolCallEvent[]; workflow_events: WorkflowEvent[]; checkpoints: TelemetryCheckpoint[] }
