"use client";

import { useState } from "react";
import { api } from "@/lib/api";

interface Props {
  tracks: Array<{ track_id: string; n_runs: number }>;
  runs?: Record<string, Array<{ run_id: string }>>;
}

export function HoldoutPanel({ tracks }: Props) {
  const [open, setOpen] = useState(false);
  const [track, setTrack] = useState(tracks[0]?.track_id ?? "");
  const [runId, setRunId] = useState("");
  const [token, setToken] = useState("");
  const [state, setState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [result, setResult] = useState<string | null>(null);

  async function runEvaluation() {
    if (!track || !runId || !token) return;
    setState("running");
    setResult(null);
    try {
      const { API_BASE } = await import("@/lib/config");
      const res = await fetch(`${API_BASE}/api/holdout/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track, run_id: runId, token }),
      });
      const data = await res.json();
      setState(data.ok ? "done" : "error");
      setResult(data.ok
        ? `Evaluation complete.\n${data.stdout?.trim().split("\n").slice(-10).join("\n")}`
        : `Failed.\n${data.stderr || data.stdout}`
      );
    } catch (err) {
      setState("error");
      setResult(String(err));
    } finally {
      // Clear token from state immediately after request
      setToken("");
    }
  }

  return (
    <section>
      <div className="border border-dashed border-yellow-800 rounded-lg p-4 bg-yellow-950/20">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-yellow-400">
              Milestone Holdout Evaluation
            </h2>
            <p className="text-yellow-700 text-xs mt-0.5">
              Token is used once and never stored server-side. Human-only action.
            </p>
          </div>
          <button
            onClick={() => setOpen((v) => !v)}
            className="text-xs px-3 py-1.5 rounded border border-yellow-800 text-yellow-400 hover:bg-yellow-900/30 transition-colors"
          >
            {open ? "Close" : "Unlock"}
          </button>
        </div>

        {open && (
          <div className="mt-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-yellow-600 mb-1 block">Track</label>
                <select
                  value={track}
                  onChange={(e) => setTrack(e.target.value)}
                  className="input w-full text-xs"
                >
                  {tracks.map((t) => (
                    <option key={t.track_id} value={t.track_id}>{t.track_id}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-yellow-600 mb-1 block">Run ID</label>
                <input
                  value={runId}
                  onChange={(e) => setRunId(e.target.value)}
                  placeholder="20260601T074524Z"
                  className="input w-full text-xs font-mono"
                />
              </div>
            </div>

            <div>
              <label className="text-xs text-yellow-600 mb-1 block">
                AUTORESEARCH_MILESTONE_TOKEN
              </label>
              <input
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Token is cleared after submission"
                className="input w-full text-xs font-mono"
                autoComplete="off"
              />
            </div>

            <button
              onClick={runEvaluation}
              disabled={state === "running" || !track || !runId || !token}
              className="px-4 py-1.5 rounded border border-yellow-700 text-yellow-400
                         hover:bg-yellow-900/40 text-xs font-medium transition-colors
                         disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {state === "running" ? "Evaluating…" : "Run Holdout Evaluation"}
            </button>

            {result && (
              <pre className={`text-xs font-mono whitespace-pre-wrap rounded p-2 ${
                state === "error" ? "bg-red-950 text-red-400" : "bg-green-950 text-green-300"
              }`}>
                {result}
              </pre>
            )}

            <p className="text-xs text-yellow-800">
              Equivalent CLI:{" "}
              <code className="text-yellow-600">
                AUTORESEARCH_MILESTONE_TOKEN=&lt;token&gt; autoresearch --track &lt;track&gt; --run-id &lt;run-id&gt; evaluate-milestone
              </code>
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
