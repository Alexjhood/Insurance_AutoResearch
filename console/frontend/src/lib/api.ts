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

export interface LaunchRequest {
  track: string;
  surface: string;
  model_provider: string;
  model_name: string;
  cycles: number;
  memory_access: string;
  scope: string;
  guidance: string;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const api = {
  health: () => get<{ status: string }>("/health"),

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
  artifactPaths: (track: string, runId: string) =>
    get<string[]>(`/tracks/${track}/runs/${runId}/artifact-paths`),

  leaderboard: () => get<LeaderboardData>("/leaderboard"),
  playbook: () => get<{ available: boolean; content: string | null }>("/leaderboard/playbook"),

  jobs: () => get<Job[]>("/jobs"),
  job: (id: string) => get<Job>(`/jobs/${id}`),
  jobEvents: (id: string, after = 0) =>
    get<JobEvent[]>(`/jobs/${id}/events?after=${after}`),
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
