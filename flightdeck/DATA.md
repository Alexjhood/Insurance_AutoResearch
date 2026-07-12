# Flight Deck — Data Contracts & Source Mapping

**Schema version 2** (`snapshot_schema_version: 2`). Companion to SPEC.md §2–3.
This file defines the snapshot JSON the ETL emits and where every field comes
from. `app/src/lib/types.ts` and `etl/schema.py` must mirror it exactly.

Reference fixture: `artifacts/orchestrations/20260711T164959Z` (french_motor /
frequency, 5 codex delegations, 1 takeover, playoff, complete telemetry). All
source paths below are relative to `artifacts/orchestrations/<orch_id>/`.

General rules:
- Timestamps: ISO-8601 UTC strings exactly as stored; ETL never reformats.
- Missing/unknown values: `null`, never omitted keys (keeps TS types simple).
- The ETL is **defensive**: any missing file or table degrades to `null`s plus
  an entry in `snapshot.build.warnings[]` — it must succeed on partial/crashed
  campaigns (that's when reports matter most).
- IDs: `orch_id` = folder name; `delegation_id` = `d01`-style; experiment and
  comparison ids as stored in registries.

---

## 1. `snapshots/index.json`

```ts
interface SnapshotIndex {
  snapshot_schema_version: 2;
  built_at: string;
  orchestrations: IndexEntry[];
}
interface IndexEntry {
  orch_id: string;
  alias: string | null;              // from snapshots/aliases.json {orch_id: alias}, user-editable, ETL preserves
  dataset: string;                   // orchestration.json .dataset
  target_mode: string;               // campaign_report.json .target_mode
  status: string;                    // orchestration.json .status ("completed" | "running" | ...)
  created_at: string;
  ended_at: string | null;           // max delegation ended_at, else null
  orchestrator_model: string;        // campaign_report.json .orchestrator (e.g. "openai/gpt-5.6")
  orchestrator: { provider: string; model: string; effort: string | null };
  stale: boolean;                    // orchestration.json mtime newer than snapshot build
  backends: string[];                // distinct delegation .backend values
  n_delegations: number;
  cycles_committed: number;          // orchestration.json .cycles_committed
  cycles_used: number;               // Σ reports/<d>.json .cycles.used
  cycles_forfeited: number;
  final_gini: number | null;         // playoff final champion's gini_weighted (finalists entry), else best delegation champion gini
  baseline_gini: number | null;      // first champion_history entry's champion gini in the first run (global_mean baseline)
  total_tokens: TokenTotals;         // Σ delegation tool_usage (see §2.6)
  cache_hit_rate: number | null;     // cached_input / input
  wall_clock_minutes: number | null; // ended_at - created_at
  distress_count: number;            // Σ active distress flags across delegations
  takeover_count: number;            // count of delegations with taken_over=true + notes kind="takeover" (dedup by delegation)
  champion_spark: number[];          // ordered champion gini after each promotion (for card sparkline)
  cost_usd: number | null;           // always displayed as estimated when present
  cost_estimated: boolean;
}
```

## 2. `snapshots/<orch_id>/snapshot.json`

```ts
interface Snapshot {
  snapshot_schema_version: 2;
  build: { built_at: string; source_mtime: string; warnings: string[] };
  campaign: Campaign;
  delegations: Delegation[];         // ordered d01..dNN
  experiments: Experiment[];         // ALL experiments across delegations + consolidation, campaign-chronological
  champion_timeline: ChampionEvent[];
  notes: OperatorNote[];             // chronological
  playoff: Playoff | null;
  telemetry_summary: TelemetrySummary;
  files: FileEntry[];                // manifest of snapshots/<id>/files/**
}
```

### 2.1 Campaign

```ts
interface Campaign {
  orch_id: string;
  dataset: string;                   // orchestration.json .dataset
  target_mode: string;               // campaign_report.json .target_mode
  status: string;
  created_at: string;
  ended_at: string | null;
  cycles_committed: number;
  orchestrator_model: string;
  orchestrator: { provider: string; model: string; effort: string | null };
  consolidation: { run_id: string; track: string } | null;   // orchestration.json .consolidation
  framework_computed: unknown;       // campaign_report.json .framework_computed passthrough (headline stats block; render as-is where useful)
  campaign_report_md: string | null; // files/ path to CAMPAIGN_REPORT.md copy
  orchestration_log_md: string | null;
}
```

### 2.2 Delegation

Source: `orchestration.json .delegations[]` (authoritative ledger) joined with
`reports/<dNN>.json` (agent-facing report) and `briefs/<dNN>.json`.

```ts
interface Delegation {
  delegation_id: string;
  backend: string;                   // ledger .backend  e.g. "codex-gpt-5-6-luna-medium"
  track: string;                     // "codex" | "claude" | "opencode"
  run_id: string;                    // per-delegation research run id
  run_path: string;                  // repo-relative
  command: string[];                 // ledger .command (argv)
  resolved_executable: string | null;
  resolved_version: string | null;
  model_identity: { provider: string; name: string; harness: string } | null;  // runs/<d>/run_manifest.json .model_identity
  status: string;                    // ledger .status
  clean_exit: boolean | null;
  exit_code: number | null;
  spawned_at: string | null;
  ended_at: string | null;
  timeout_minutes: number | null;
  continue_run: boolean;
  respawn_of: string | null;         // delegation_id this respawned
  taken_over: boolean;
  budget: {                          // ledger + report .cycles
    committed: number;               // ledger .cycle_budget
    attempted: number; completed: number; decided: number; used: number;  // report .cycles
    forfeited: number;               // ledger .cycles_forfeited
    refunded: boolean;               // ledger .budget_refunded
  };
  brief: Brief;                      // §2.3
  agent_summary: string | null;      // report .agent_summary (also in ledger)
  champion: DelegationChampion | null; // report .champion
  distress: {                        // report .distress
    active: string[];                // flags raised
    all_flags: string[];             // full vocabulary at time of run (report .distress.flags)
    detail: string | null;
  };
  repairs: { max_attempts_in_a_cycle: number; total_attempts: number } | null; // report .repairs
  cost: DelegationCost;              // §2.6
  files: {                           // files/ paths (null if missing)
    prompt: string | null;           // prompts/<d>.md
    brief: string | null;            // briefs/<d>.json
    report: string | null;           // reports/<d>.json
    stdout_log: string | null;       // logs/<d>.stdout.log (possibly truncated copy)
    exit_json: string | null;
    research_log: string | null;     // runs/<d>/RESEARCH_LOG.md
    llm_usage: string | null;        // runs/<d>/LLM_USAGE.md
  };
}

interface Brief {                    // briefs/<dNN>.json (note: also has _source_path to the named brief)
  name: string | null;               // basename of _source_path e.g. "structural_family_comparison"
  direction: string;
  constraints: string[];
  cycle_budget: number;
  starting_knowledge: string[];
  foundation_models: boolean;
  seed_champion: { experiment_id: string; from_run: string } | null;
}

interface DelegationChampion {       // report .champion
  experiment_id: string;
  gini_weighted: number; rank_gini_weighted: number | null;
  asym_pricing_loss: number | null; calibration_ratio: number | null;
  model_family: string; target_strategy: string;
  beat_seed_baseline: boolean | null;
}
```

### 2.3 Experiment (the heart of the Journey)

Sources per delegation run `runs/<dNN>/registry.sqlite`:
`experiments` ⋈ `proposals` (on experiment_id) ⋈ `comparisons` (challenger_id)
⋈ `research_log_entries` (experiment_id) ⋈ `research_nodes` (experiment_id),
plus report `.experiments[]` for decision/interpretation cross-check.
Consolidation-run experiments come from the consolidation run's registry (path
from delegation ledger pattern `runs/<consolidation track>/...`; if the run dir
is not under the orchestration folder, resolve from
`artifacts/tracks/<track>/runs/<run_id>` and warn if unreadable).

```ts
interface Experiment {
  experiment_id: string;
  delegation_id: string | null;      // null ⇒ consolidation/playoff run
  cycle: number | null;              // research_log_entries.cycle
  seq: number;                       // campaign-wide chronological index (ETL-assigned)
  created_at: string;
  name: string;                      // experiments.experiment_name
  status: string;                    // experiments.status
  is_baseline: boolean;              // name/family heuristic: model_family "constant" or name contains "global_mean"
  is_seed: boolean;                  // name starts with "orchestration_delegation_seed" / "orchestration_seed"
  parent_experiment_id: string | null;
  model_family: string; target_strategy: string;
  recipe: unknown | null;            // proposals.config_json → experiment_config.model.recipe (or script marker {script: true, path})
  proposal: {
    hypothesis: string | null;       // research_nodes.hypothesis (fallback proposals.rationale)
    change_summary: string | null; expected_benefit: string | null; key_risk: string | null;
    exploration_axis: string | null; approach_family: string | null;   // from config_json top-level fields when present
  };
  metrics: {                         // parse experiments.metrics_path JSON if within run dir; else from research_nodes.metrics_json
    gini_weighted: number | null; rank_gini_weighted: number | null;
    asym_pricing_loss: number | null; calibration_ratio: number | null;
    fit_wall_seconds: number | null;  // experiments.fit_wall_seconds
  };
  screening: {                       // research_nodes.screening_json (nullable passthrough of its fields)
    gini_weighted: number | null; lift: number | null; win_rate: number | null;
  } | null;
  comparison: {                      // comparisons row where challenger_id = experiment_id (latest)
    comparison_id: string;
    champion_id: string;             // then-champion
    cv_mean_lift: number | null; fold_win_rate: number | null;   // parse comparisons.bootstrap_summary JSON: mean lift + challenger_win_rate
    decision: 'promote'|'local_promote'|'reject'|null;
    decided_by: string | null; decision_reason_code: string | null;
    decision_rationale: string | null;
    guardrail_status: string | null;
  } | null;
  decision_meta: {                   // research_log_entries
    interpretation: string | null; next_step: string | null; outcome: string | null;
  };
  lift: {                            // ETL-derived; see SPEC §3
    vs_then_champion: number | null;
    vs_baseline: number | null;      // gini - campaign baseline gini
    kind: 'baseline_relative' | 'incremental' | null;
  };
  repairs: RepairAttempt[];          // from session_events where event_type indicates repair (event_type LIKE '%repair%'); [] if none
  usage: ExperimentUsage | null;     // §2.6 per-experiment checkpoint
  research_line_id: string | null;   // research_nodes.line_id
  tree: { node_id: string; parent_node_id: string | null } | null;  // for ResearchTree
}
interface RepairAttempt {
  attempt: number; kind: 'recipe'|'script'|null;
  failed_checks: string[]; resolved: boolean;
  raw: unknown;                      // details_json passthrough
}
```

### 2.4 ChampionEvent & OperatorNote

```ts
interface ChampionEvent {            // stitched champion_history across delegation runs + consolidation, chronological
  at: string; delegation_id: string | null;
  action: string;                    // champion_history.action
  previous_champion_id: string | null; new_champion_id: string;
  new_champion_gini: number | null;  // join to Experiment.metrics
  comparison_id: string | null;
  scope: 'delegation' | 'consolidation';
  is_seed_transfer: boolean;         // new champion is_seed (champion arriving via orchestrator seeding, not a scientific win)
}

interface OperatorNote {             // notes.json .notes[]
  at: string;                        // .timestamp
  kind: 'reflection' | 'takeover' | 'decision' | string;
  delegation_id: string | null;
  text: string;
}
```

### 2.5 Playoff

Source `playoff/playoff_report.json` (+ `.md` copy in files/).

```ts
interface Playoff {
  decision_mode: string;             // "auto" | ...
  consolidation: { run_id: string; track: string };
  finalists: {
    delegation_id: string; experiment_id: string;
    gini_weighted: number; model_family: string;
    replay_experiment_id: string | null;   // consolidation-run experiment id when matchable
  }[];
  exclusions: { delegation_id: string | null; reason: string }[];  // shape passthrough; [] in fixture
  pairings: { order: number; delegation_id: string; source_experiment_id: string;
    incumbent_experiment_id: string; replayed_experiment_id: string | null;
    comparison_id: string | null; decision: string | null; decision_reason: string | null;
    mean_lift: number | null; challenger_win_rate: number | null;
    gates: Record<string, boolean>; guardrail_passed: boolean | null }[];
  final: {
    delegation_id: string; source_experiment_id: string;
    consolidation_experiment_id: string;
  } | null;
  status: string;                    // completed | incomplete | failed
  failure_reason: string | null;
  report_md: string | null;          // files/ path
}
```

### 2.6 Cost & telemetry summary

Sources: ledger `.tool_usage` (per delegation totals), report `.cost`
(`llm_usage` + `wall_clock_minutes`), and `runs/<d>/telemetry.sqlite`
(`experiment_usage_checkpoints` for per-experiment increments — this backs the
LLM_USAGE.md table; `llm_model_calls` / `llm_tool_calls` aggregates).

```ts
interface TokenTotals {
  input: number; cached_input: number; output: number;
  reasoning: number;                 // reasoning_output_tokens
}
interface DelegationCost {
  tokens: TokenTotals;               // ledger .tool_usage
  model_calls: number; tool_calls: number; tool_failures: number;  // telemetry counts
  cache_hit_rate: number | null;
  wall_clock_minutes: number | null; // report .cost.wall_clock_minutes
  cost_usd: number | null;
}
interface ExperimentUsage {          // experiment_usage_checkpoints row for this experiment step
  tokens: TokenTotals; total_cumulative: number;
  model_calls: number; tool_calls: number; tool_failures: number;
  cache_hit_rate: number | null;
}
interface TelemetrySummary {         // campaign rollup for Overview/Telemetry pages
  totals: TokenTotals & { model_calls: number; tool_calls: number; tool_failures: number };
  cache_hit_rate: number | null;
  by_delegation: { delegation_id: string; tokens: TokenTotals; model_calls: number;
                   tool_calls: number; tool_failures: number; cache_hit_rate: number | null }[];
  tool_mix: { delegation_id: string; tool: string; calls: number;
              failures: number; total_duration_ms: number }[];   // llm_tool_calls GROUP BY name
  usage_by_model: { key: string; provider: string; model: string; effort: string | null;
                    role: string; tokens: TokenTotals | null; unmeasured: boolean;
                    cost_usd: number | null; cost_estimated: boolean }[];
  cost_usd: number | null;
  cost_estimated: boolean;
}
```

> ETL note: `experiment_usage_checkpoints` / `llm_usage_checkpoints` column
> names must be discovered with `PRAGMA table_info` at build time and mapped
> defensively (they are framework-internal). If absent, fall back to parsing
> the LLM_USAGE.md table (documented stable format) and warn.

### 2.7 Files manifest

```ts
interface FileEntry {
  path: string;                      // relative under snapshots/<id>/files/
  kind: 'markdown'|'json'|'log'|'text';
  bytes: number; truncated: boolean; // stdout logs copied with tail-2MB cap
  source: string;                    // original repo-relative path
}
```

Copied set: CAMPAIGN_REPORT.md, ORCHESTRATION_LOG.md, playoff_report.{md,json},
prompts/*.md, briefs/*.json, reports/*.json, logs/*.stdout.log (tail-capped),
logs/*.exit.json, per-run RESEARCH_LOG.md + LLM_USAGE.md + run_manifest.json.
**Never copy** registry.sqlite/telemetry.sqlite (queried, not shipped), and
never anything from `baseline/`.

## 3. `snapshots/<orch_id>/telemetry_<dNN>.json` (lazy)

Full event lists for the Flight Recorder. Source: `runs/<dNN>/telemetry.sqlite`.
WAL files may be present; open read-only with
`sqlite3.connect(f"file:{path}?mode=ro", uri=True)`.

```ts
interface DelegationTelemetry {
  delegation_id: string;
  model_calls: {                     // llm_model_calls, ordered by occurred_at
    at: string; model: string;
    input: number; cached_input: number; output: number; reasoning: number;
    duration_hint_ms: null;          // not stored; reserved
    workflow_event_id: number | null;
  }[];
  tool_calls: {                      // llm_tool_calls, ordered by started_at
    name: string; detail: string | null;
    started_at: string | null; completed_at: string | null; duration_ms: number | null;
    status: string | null; success: boolean | null;
    input_bytes: number | null; output_bytes: number | null; error_type: string | null;
    workflow_event_id: number | null;
  }[];
  workflow_events: {                 // workflow_events (the autoresearch CLI commands the agent ran)
    id: number; command: string; started_at: string | null; completed_at: string | null;
    duration_ms: number | null; status: string | null; error_type: string | null;
  }[];
  checkpoints: {                     // experiment_usage_checkpoints in order — used as segment markers
    experiment_name: string; completed_at: string | null; total_cumulative: number;
  }[];
}
```

## 4. Known data quirks (encode in ETL tests)

1. **d02-style takeover runs** may have `cycles.attempted > decided` and
   experiments registered with no decision — render as "forfeited/undecided",
   never drop them.
2. Seed experiments (`orchestration_delegation_seed_*`) duplicate a prior
   delegation's champion — mark `is_seed`, exclude from lift aggregates, and
   draw as lineage, not as new science.
3. An experiment can appear in a delegation registry **and** the consolidation
   registry (replays). Dedup by experiment_id per registry; link via
   `Playoff.finalists.replay_experiment_id`.
4. Mixed lift bases: `screening.lift` for a first experiment is vs global-mean
   baseline; later ones vs current champion. The `lift.kind` field exists so
   the UI never averages across kinds (historical reporting bug).
5. `telemetry.sqlite-wal/-shm` present ⇒ always read-only URI mode; never
   checkpoint someone else's DB.
6. Registries may be **integrity-locked or partially written** on crashed
   campaigns; every sqlite read wrapped, degrade to nulls + warning.
7. `orchestration.json .delegations[].tool_usage` can be missing for crashed
   delegations — fall back to summing `llm_model_calls`.
