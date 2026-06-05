"use client";

import { useEffect, useRef, useState } from "react";
import type { Job, JobEvent } from "@/lib/api";
import { api } from "@/lib/api";
import { WS_BASE } from "@/lib/config";

interface Props {
  job: Job;
  initialEvents: JobEvent[];
}

export function LiveMonitor({ job, initialEvents }: Props) {
  const [events, setEvents] = useState<JobEvent[]>(initialEvents);
  const [status, setStatus] = useState(job.status);
  const [steerMsg, setSteerMsg] = useState("");
  const [steerInterrupt, setSteerInterrupt] = useState(true);
  const [steerResult, setSteerResult] = useState<string | null>(null);
  const [actioning, setActioning] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Track the highest event id we've appended, in a ref so the polling
  // fallback never reads a stale closure of `events`.
  const lastIdRef = useRef<number>(
    initialEvents.length ? initialEvents[initialEvents.length - 1].id : 0
  );

  const isTerminal = ["done", "stopped", "failed", "interrupted"].includes(status);
  const isPaused = status === "paused";

  const appendEvent = (data: JobEvent) => {
    setEvents((prev) => {
      if (prev.some((ev) => ev.id === data.id)) return prev;
      return [...prev, data];
    });
    if (typeof data.id === "number" && data.id > lastIdRef.current) {
      lastIdRef.current = data.id;
    }
  };

  // Live stream via WebSocket, with a single polling fallback if the socket
  // never opens or drops. One effect owns both; cleanup tears down both.
  // Re-runs only when the job id or terminal state changes — not on every
  // status flip — so we don't churn connections.
  useEffect(() => {
    if (isTerminal) return;

    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let closed = false;

    const startPolling = () => {
      if (pollTimer || closed) return;
      pollTimer = setInterval(async () => {
        const latest = await api.job(job.id).catch(() => null);
        if (latest) setStatus(latest.status);
        const newEvts = await api
          .jobEvents(job.id, lastIdRef.current)
          .catch(() => [] as JobEvent[]);
        newEvts.forEach(appendEvent);
      }, 2000);
    };

    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket(`${WS_BASE}/api/jobs/${job.id}/stream`);
      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.event_type === "stream_end") {
          setStatus(data.status);
          return;
        }
        appendEvent(data);
      };
      ws.onerror = () => startPolling();
      ws.onclose = () => {
        if (!closed && !isTerminal) startPolling();
      };
    } catch {
      startPolling();
    }

    return () => {
      closed = true;
      if (pollTimer) clearInterval(pollTimer);
      if (ws) ws.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, isTerminal]);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  async function sendSteer() {
    if (!steerMsg.trim()) return;
    const result = await api.steerJob(job.id, steerMsg, steerInterrupt).catch((e) => ({
      mode: "error",
      ok: false,
      detail: String(e),
    }));
    setSteerResult(`[${result.mode}] ${result.ok ? "✓" : "✗"} — ${result.detail}`);
    setSteerMsg("");
    setTimeout(() => setSteerResult(null), 6000);
  }

  async function doAction(action: "pause" | "resume" | "stop") {
    setActioning(action);
    try {
      if (action === "pause") {
        await api.pauseJob(job.id);
        setStatus("paused");
      } else if (action === "resume") {
        await api.resumeJob(job.id);
        setStatus("running");
      } else {
        await api.stopJob(job.id);
        setStatus("stopped");
      }
    } catch (err) {
      setSteerResult(`${action} failed: ${err}`);
    }
    setActioning(null);
  }

  // Build token stream from events
  const tokenBuffer = events
    .filter((e) => e.event_type === "token")
    .map((e) => {
      try { return (JSON.parse(e.payload_json ?? "{}") as {text?: string}).text ?? ""; }
      catch { return ""; }
    })
    .join("");

  const toolEvents = events.filter((e) => e.event_type === "tool_use");
  const systemEvents = events.filter((e) =>
    ["system", "steer_sent", "steer_queued", "agent_exit"].includes(e.event_type)
  );

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Main stream */}
      <div className="lg:col-span-2 space-y-3">
        <div
          ref={scrollRef}
          className="bg-gray-900 border border-gray-800 rounded-lg p-3 h-[500px] overflow-y-auto font-mono text-xs text-gray-300 whitespace-pre-wrap"
        >
          {tokenBuffer || (
            <span className="text-gray-600">
              {status === "running"
                ? "Waiting for output…"
                : status === "paused"
                ? "Session paused."
                : `Session ${status}.`}
            </span>
          )}
          {status === "running" && (
            <span className="animate-pulse text-brand ml-0.5">▌</span>
          )}
        </div>

        {/* Tool calls */}
        {toolEvents.length > 0 && (
          <div>
            <div className="text-xs text-gray-500 mb-1">
              Tool calls ({toolEvents.length})
            </div>
            <div className="bg-gray-900 border border-gray-800 rounded p-2 text-xs font-mono overflow-auto max-h-32">
              {toolEvents.slice(-10).map((e, i) => {
                let name = "?";
                try { name = (JSON.parse(e.payload_json ?? "{}") as {name?: string}).name ?? "?"; }
                catch {}
                return (
                  <div key={i} className="text-blue-400">
                    {e.ts.slice(11, 19)} ⚡ {name}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Right panel */}
      <div className="space-y-4">
        {/* Controls */}
        <div className="card space-y-3">
          <div className="text-sm font-semibold text-gray-300">Controls</div>

          <div className="flex gap-2">
            {isPaused ? (
              <button
                onClick={() => doAction("resume")}
                disabled={!!actioning}
                className="flex-1 py-1.5 bg-green-900 hover:bg-green-800 rounded text-green-300 text-xs font-medium transition-colors disabled:opacity-40"
              >
                {actioning === "resume" ? "Resuming…" : "Resume"}
              </button>
            ) : (
              <button
                onClick={() => doAction("pause")}
                disabled={isTerminal || !!actioning}
                className="flex-1 py-1.5 bg-yellow-900 hover:bg-yellow-800 rounded text-yellow-300 text-xs font-medium transition-colors disabled:opacity-40"
              >
                {actioning === "pause" ? "Pausing…" : "Pause"}
              </button>
            )}
            <button
              onClick={() => doAction("stop")}
              disabled={isTerminal || !!actioning}
              className="px-3 py-1.5 bg-red-900 hover:bg-red-800 rounded text-red-300 text-xs font-medium transition-colors disabled:opacity-40"
            >
              {actioning === "stop" ? "Stopping…" : "Stop"}
            </button>
          </div>

          {isTerminal && (
            <p className="text-xs text-gray-600">
              Session {status}. Launch a new run from the{" "}
              <a href="/launch" className="text-brand hover:underline">Launch</a> page.
            </p>
          )}
        </div>

        {/* Steer */}
        <div className="card space-y-3">
          <div className="text-sm font-semibold text-gray-300">Steer Agent</div>

          <textarea
            value={steerMsg}
            onChange={(e) => setSteerMsg(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) sendSteer();
            }}
            rows={4}
            className="input w-full resize-none text-xs"
            placeholder="Type guidance… (⌘Enter to send)"
            disabled={isTerminal}
          />

          <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-400">
            <input
              type="checkbox"
              checked={steerInterrupt}
              onChange={(e) => setSteerInterrupt(e.target.checked)}
              className="accent-brand"
            />
            Attempt interrupt (fall back to queued)
          </label>

          <button
            onClick={sendSteer}
            disabled={isTerminal || !steerMsg.trim()}
            className="w-full py-1.5 bg-brand hover:bg-brand-dark rounded text-white text-xs font-medium transition-colors disabled:opacity-40"
          >
            Send
          </button>

          {steerResult && (
            <div
              className={`text-xs font-mono ${
                steerResult.includes("✗") ? "text-red-400" : "text-yellow-400"
              }`}
            >
              {steerResult}
            </div>
          )}
        </div>

        {/* System events */}
        <div>
          <div className="text-xs text-gray-500 mb-1">
            Events ({systemEvents.length})
          </div>
          <div className="bg-gray-900 border border-gray-800 rounded p-2 text-xs font-mono overflow-auto max-h-64">
            {systemEvents.length === 0 ? (
              <span className="text-gray-600">No system events yet.</span>
            ) : (
              systemEvents.map((e, i) => {
                let msg = e.event_type;
                try {
                  const p = JSON.parse(e.payload_json ?? "{}") as Record<string, unknown>;
                  msg = (p.msg ?? p.error ?? p.message ?? e.event_type) as string;
                } catch {}
                const color =
                  e.event_type === "steer_sent"
                    ? "text-brand"
                    : e.event_type === "steer_queued"
                    ? "text-yellow-400"
                    : e.event_type === "agent_exit"
                    ? "text-red-400"
                    : "text-gray-500";
                return (
                  <div key={i} className={color}>
                    {e.ts.slice(11, 19)} {msg}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
