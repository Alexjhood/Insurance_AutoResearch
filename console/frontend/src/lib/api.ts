import { API_BASE } from "./config";
const BASE = `${API_BASE}/api`;

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return res.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${path}`);
  return res.json();
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Track {
  track_id: string;
  n_runs: number;
}

export interface RunSummary {
  run_id: string;
  track_id: string;
  has_registry: boolean;
  manifest?: Record<string, unknown>;
  counts?: {
    experiments: number;
    artifacts: number;
    comparisons: number;
    proposals: number;
  };
  champion?: Experiment | null;
  research_log_preview?: string;
  error?: string;
}

export interface Experiment {
  experiment_id: string;
  experiment_name?: string;
  model_family?: string;
  primary_metric?: string;
  mean_score?: number;
  gini_weighted?: number;
  target_strategy?: string;
  target_mode?: string;
  champion_id?: string;
  status?: string;
  created_at?: string;
  [key: string]: unknown;
}

export interface Comparison {
  comparison_id: string;
  champion_id?: string;
  challenger_id?: string;
  mean_lift?: number;
  decision?: string;
  created_at?: string;
  [key: string]: unknown;
}

export interface LeaderboardData {
  available: boolean;
  models: Model[];
  runs: LeaderboardRun[];
  top_experiments: LeaderboardExperiment[];
  insights: Insight[];
}

export interface Model {
  model_id: string;
  provider: string;
  name: string;
  last_seen: string;
}

export interface LeaderboardRun {
  run_uid: string;
  track_id: string;
  run_id: string;
  model_name: string;
  provider: string;
  peak_gini: number | null;
  n_experiments: number;
  n_promotions: number;
  started_at: string;
}

export interface LeaderboardExperiment {
  experiment_uid: string;
  run_uid: string;
  track_id: string;
  run_id: string;
  model_name: string;
  model_family?: string;
  gini_weighted: number | null;
  mean_score?: number;
  status: string;
}

export interface Insight {
  id: number;
  claim: string;
  evidence?: string;
  recorded_at: string;
  [key: string]: unknown;
}

export interface Job {
  id: string;
  track: string;
  run_id: string;
  surface: string;
  model_provider?: string;
  model_name?: string;
  agent_model?: string;
  agent_effort?: string;
  cycles: number;
  memory_access: string;
  scope: string;
  guidance?: string;
  status: string;
  created_at: string;
  updated_at: string;
  pid?: number;
  worktree_path?: string;
}

export interface JobEvent {
  id: number;
  job_id: string;
  ts: string;
  event_type: string;
  payload_json?: string;
}

export interface TelemetrySummary {
  session_count?: number;
  turn_count: number;
  model_call_count?: number;
  input_tokens: number;
  cached_input_tokens: number;
  cache_creation_input_tokens: number;
  uncached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  output_chars: number;
  cache_hit_ratio: number | null;
  provider_reported_cost_usd: number | null;
  cost_coverage_turns: number;
  tool_call_count: number;
  completed_tool_call_count: number;
  tool_failure_count: number;
  tool_duration_ms: number | null;
  signal_count: number;
  workflow_event_count?: number;
  imported_bytes?: number;
  source_bytes?: number;
  import_complete?: boolean;
}

export interface JobTelemetry {
  job_id: string;
  summary: TelemetrySummary;
  turns: Record<string, unknown>[];
  tool_calls: Record<string, unknown>[];
  signals: Record<string, unknown>[];
}

export interface RunTelemetry extends Omit<JobTelemetry, "job_id"> {
  available: boolean;
  sessions: Record<string, unknown>[];
  workflow_events: Record<string, unknown>[];
  coverage: Record<string, unknown>[];
}

export interface SurfaceInfo {
  surface: string;
  available: boolean;
  models: string[];
  default_model: string;
  efforts: string[];
  effort_label: string;
  allows_custom_model: boolean;
}

export interface LaunchRequest {
  track: string;
  surface: string;
  model: string;
  effort: string;
  cycles: number;
  memory_access: string;
  scope: string;
  guidance: string;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  health: () => get<{ status: string }>("/health"),

  surfaces: () => get<SurfaceInfo[]>("/surfaces"),

  tracks: () => get<Track[]>("/tracks"),
  runs: (track: string) => get<RunSummary[]>(`/tracks/${track}/runs`),
  runSummary: (track: string, runId: string) =>
    get<RunSummary>(`/tracks/${track}/runs/${runId}`),
  experiments: (track: string, runId: string) =>
    get<Experiment[]>(`/tracks/${track}/runs/${runId}/experiments`),
  comparisons: (track: string, runId: string) =>
    get<Comparison[]>(`/tracks/${track}/runs/${runId}/comparisons`),
  champion: (track: string, runId: string) =>
    get<Experiment | null>(`/tracks/${track}/runs/${runId}/champion`),
  championHistory: (track: string, runId: string) =>
    get<Experiment[]>(`/tracks/${track}/runs/${runId}/champion-history`),
  proposals: (track: string, runId: string) =>
    get<unknown[]>(`/tracks/${track}/runs/${runId}/proposals`),
  sessions: (track: string, runId: string) =>
    get<unknown[]>(`/tracks/${track}/runs/${runId}/sessions`),
  researchLines: (track: string, runId: string) =>
    get<unknown[]>(`/tracks/${track}/runs/${runId}/research-lines`),
  runTelemetry: (track: string, runId: string) =>
    get<RunTelemetry>(`/tracks/${track}/runs/${runId}/telemetry`),
  artifactPaths: (track: string, runId: string) =>
    get<string[]>(`/tracks/${track}/runs/${runId}/artifact-paths`),

  leaderboard: () => get<LeaderboardData>("/leaderboard"),
  playbook: () => get<{ available: boolean; content: string | null }>("/leaderboard/playbook"),

  jobs: () => get<Job[]>("/jobs"),
  job: (id: string) => get<Job>(`/jobs/${id}`),
  jobEvents: (id: string, after = 0) =>
    get<JobEvent[]>(`/jobs/${id}/events?after=${after}`),
  jobTelemetry: (id: string) => get<JobTelemetry>(`/jobs/${id}/telemetry`),
  launchJob: (req: LaunchRequest) => post<{ job_id: string }>("/jobs", req),
  steerJob: (id: string, message: string, interrupt = true) =>
    post<{ mode: string; ok: boolean; detail: string }>(`/jobs/${id}/steer`, {
      message,
      interrupt,
    }),
  pauseJob: (id: string) => post<{ status: string }>(`/jobs/${id}/pause`, {}),
  resumeJob: (id: string) => post<{ status: string }>(`/jobs/${id}/resume`, {}),
  stopJob: (id: string) => post<{ status: string }>(`/jobs/${id}/stop`, {}),

  evaluateHoldout: (track: string, run_id: string, token: string) =>
    post<{ ok: boolean; stdout: string; stderr: string }>(
      "/holdout/evaluate",
      { track, run_id, token }
    ),
};
