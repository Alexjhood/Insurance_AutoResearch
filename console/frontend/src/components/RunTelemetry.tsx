"use client";

import { useEffect, useState } from "react";
import { api, type RunTelemetry as RunTelemetryData } from "@/lib/api";

interface Props {
  track: string;
  runId: string;
  initial: RunTelemetryData;
}

export function RunTelemetry({ track, runId, initial }: Props) {
  const [telemetry, setTelemetry] = useState(initial);

  useEffect(() => {
    let closed = false;
    const refresh = async () => {
      const next = await api.runTelemetry(track, runId).catch(() => null);
      if (!closed && next) setTelemetry(next);
    };
    const timer = setInterval(refresh, 3000);
    return () => {
      closed = true;
      clearInterval(timer);
    };
  }, [track, runId]);

  if (!telemetry.available) {
    return (
      <section className="card">
        <h2 className="text-lg font-semibold text-gray-300">LLM Telemetry</h2>
        <p className="text-sm text-gray-500 mt-2">
          No desktop telemetry has been imported for this run yet.
        </p>
      </section>
    );
  }

  const s = telemetry.summary;
  const turns = telemetry.turns as Record<string, unknown>[];
  const tools = telemetry.tool_calls as Record<string, unknown>[];
  const workflows = telemetry.workflow_events as Record<string, unknown>[];

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-300">LLM Telemetry</h2>
        <span className={`badge ${s.import_complete ? "badge-green" : "badge-yellow"}`}>
          {s.import_complete ? "import complete" : "import pending"}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Metric label="Total tokens" value={formatCount(s.total_tokens)} />
        <Metric label="Input tokens" value={formatCount(s.input_tokens)} />
        <Metric label="Output tokens" value={formatCount(s.output_tokens)} />
        <Metric label="Cache hit ratio" value={formatPercent(s.cache_hit_ratio)} />
        <Metric label="Reasoning tokens" value={formatCount(s.reasoning_tokens)} />
        <Metric label="Model calls" value={String(s.model_call_count ?? 0)} />
        <Metric label="Tool calls" value={String(s.tool_call_count)} />
        <Metric
          label="Tool failures"
          value={`${s.tool_failure_count}/${s.completed_tool_call_count}`}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        <div className="card overflow-auto">
          <div className="text-sm font-semibold text-gray-300 mb-2">
            Turns ({turns.length})
          </div>
          <table className="table-base">
            <thead>
              <tr>
                <th>Surface</th>
                <th>Model</th>
                <th>Effort</th>
                <th>Tokens</th>
                <th>Cache</th>
                <th>Reasoning</th>
                <th>Duration</th>
              </tr>
            </thead>
            <tbody>
              {turns.slice(-12).map((turn, index) => (
                <tr key={String(turn.turn_key ?? index)}>
                  <td>{surfaceForTurn(turn, telemetry.sessions)}</td>
                  <td className="font-mono text-xs text-gray-400">{String(turn.model ?? "—")}</td>
                  <td>
                    {turn.effort ? (
                      <span className="badge badge-yellow">{String(turn.effort)}</span>
                    ) : (
                      <span className="text-gray-600">—</span>
                    )}
                  </td>
                  <td className="font-mono">{formatCount(Number(turn.total_tokens ?? 0))}</td>
                  <td>{formatPercent(turnCacheRatio(turn))}</td>
                  <td className="font-mono">{formatCount(Number(turn.reasoning_tokens ?? 0))}</td>
                  <td>{formatDuration(turn.duration_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card overflow-auto">
          <div className="text-sm font-semibold text-gray-300 mb-2">
            Framework Steps ({workflows.length})
          </div>
          {workflows.length === 0 ? (
            <p className="text-xs text-gray-500">
              Framework step attribution starts with commands run after this update.
            </p>
          ) : (
            <table className="table-base">
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Status</th>
                  <th>Entities</th>
                  <th>Attributed tokens</th>
                </tr>
              </thead>
              <tbody>
                {workflows.slice(-12).map((event, index) => (
                  <tr key={String(event.id ?? index)}>
                    <td className="font-mono">{String(event.command ?? "unknown")}</td>
                    <td>
                      <span className={`badge ${event.status === "completed" ? "badge-green" : "badge-red"}`}>
                        {String(event.status ?? "unknown")}
                      </span>
                    </td>
                    <td>{entitySummary(event.entities)}</td>
                    <td className="font-mono">{formatCount(Number(event.direct_tokens ?? 0))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card overflow-auto">
        <div className="text-sm font-semibold text-gray-300 mb-2">
          Recent Tool Calls ({tools.length})
        </div>
        <table className="table-base">
          <thead>
            <tr>
              <th>Tool</th>
              <th>Detail</th>
              <th>Status</th>
              <th>Duration</th>
              <th>Output</th>
            </tr>
          </thead>
          <tbody>
            {tools.slice(-20).map((tool, index) => (
              <tr key={String(tool.call_key ?? index)}>
                <td className="font-mono">{String(tool.name ?? "tool")}</td>
                <td>{String(tool.detail ?? "")}</td>
                <td>
                  <span className={`badge ${tool.success === 0 ? "badge-red" : "badge-gray"}`}>
                    {String(tool.status ?? "unknown")}
                  </span>
                </td>
                <td>{formatDuration(tool.duration_ms)}</td>
                <td>{formatBytes(Number(tool.output_bytes ?? 0))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <div className="text-gray-500 text-xs">{label}</div>
      <div className="text-lg font-semibold font-mono text-gray-200 mt-1">{value}</div>
    </div>
  );
}

function surfaceForTurn(
  turn: Record<string, unknown>,
  sessions: Record<string, unknown>[]
): string {
  return String(
    sessions.find((session) => session.session_key === turn.session_key)?.surface ?? "unknown"
  );
}

function turnCacheRatio(turn: Record<string, unknown>): number | null {
  const input = Number(turn.input_tokens ?? 0);
  return input > 0 ? Number(turn.cached_input_tokens ?? 0) / input : null;
}

function entitySummary(value: unknown): string {
  if (!Array.isArray(value) || value.length === 0) return "none";
  return value
    .map((entity) => {
      const item = entity as Record<string, unknown>;
      return `${String(item.entity_type)}:${String(item.entity_id).slice(0, 8)}`;
    })
    .join(", ");
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US", { notation: "compact" }).format(value);
}

function formatPercent(value: number | null): string {
  return value == null ? "not reported" : `${(value * 100).toFixed(1)}%`;
}

function formatDuration(value: unknown): string {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms <= 0) return "n/a";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
