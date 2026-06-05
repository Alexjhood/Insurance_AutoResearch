"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

const SURFACES = ["claude", "codex", "opencode"] as const;
const BIN_KEY: Record<string, string> = {
  claude: "claude_bin",
  codex: "codex_bin",
  opencode: "opencode_bin",
};

const MODEL_DEFAULTS: Record<string, { provider: string; name: string }> = {
  claude: { provider: "anthropic", name: "claude-sonnet-4-6" },
  codex: { provider: "openai", name: "codex-mini-latest" },
  opencode: { provider: "openai", name: "gpt-4o" },
};

const MEMORY_OPTIONS = [
  { value: "none", label: "None — fully isolated (default)" },
  { value: "own", label: "Own — this model's history only" },
  { value: "all", label: "All — all models, attributed" },
];

export default function LaunchPage() {
  const router = useRouter();
  const [surface, setSurface] = useState<string>("claude");
  const [track, setTrack] = useState("claude");
  const [modelProvider, setModelProvider] = useState("anthropic");
  const [modelName, setModelName] = useState("claude-sonnet-4-6");
  const [cycles, setCycles] = useState(3);
  const [memoryAccess, setMemoryAccess] = useState("none");
  const [scope, setScope] = useState("research");
  const [guidance, setGuidance] = useState("");
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [bins, setBins] = useState<Record<string, string | null>>({});

  // Check which agent binaries the orchestrator can find
  useEffect(() => {
    (async () => {
      try {
        const { API_BASE } = await import("@/lib/config");
        const h = await fetch(`${API_BASE}/api/health`).then((r) => r.json());
        setBins({
          claude_bin: h.claude_bin,
          codex_bin: h.codex_bin,
          opencode_bin: h.opencode_bin,
        });
      } catch {
        /* health unavailable — leave bins empty */
      }
    })();
  }, []);

  const surfaceAvailable = (s: string) => Boolean(bins[BIN_KEY[s]]);

  function onSurfaceChange(s: string) {
    setSurface(s);
    const defaults = MODEL_DEFAULTS[s];
    if (defaults) {
      setModelProvider(defaults.provider);
      setModelName(defaults.name);
    }
    setTrack(s);
  }

  async function launch() {
    setLaunching(true);
    setError(null);
    try {
      const { job_id } = await api.launchJob({
        track,
        surface,
        model_provider: modelProvider,
        model_name: modelName,
        cycles,
        memory_access: memoryAccess,
        scope,
        guidance,
      });
      router.push(`/monitor/${job_id}`);
    } catch (err) {
      setError(String(err));
      setLaunching(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Launch Run</h1>
        <p className="text-gray-500 mt-1">
          Start a new agent research session in an isolated git worktree.
        </p>
      </div>

      {/* Surface */}
      <Field label="Agent surface" hint="Greyed-out surfaces aren't installed on the orchestrator host.">
        <div className="flex gap-2">
          {SURFACES.map((s) => {
            const available = surfaceAvailable(s);
            return (
              <button
                key={s}
                onClick={() => onSurfaceChange(s)}
                disabled={!available}
                title={available ? "" : `${s} CLI not found on the server`}
                className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                  surface === s
                    ? "bg-brand text-white"
                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                } ${!available ? "opacity-40 cursor-not-allowed line-through" : ""}`}
              >
                {s}
              </button>
            );
          })}
        </div>
        {!surfaceAvailable(surface) && (
          <p className="text-yellow-600 text-xs mt-2">
            ⚠ The <strong>{surface}</strong> CLI was not found on the orchestrator host.
            Install it (and ensure it&apos;s authenticated) before launching, or set the{" "}
            <code>{surface.toUpperCase()}_BIN</code> env var.
          </p>
        )}
      </Field>

      {/* Track */}
      <Field label="Track ID" hint="e.g. claude, codex, experiment-1">
        <input
          value={track}
          onChange={(e) => setTrack(e.target.value)}
          className="input"
          placeholder="claude"
        />
      </Field>

      {/* Model */}
      <Field label="Model">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Provider</label>
            <input
              value={modelProvider}
              onChange={(e) => setModelProvider(e.target.value)}
              className="input"
              placeholder="anthropic"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Model name</label>
            <input
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="input"
              placeholder="claude-sonnet-4-6"
            />
          </div>
        </div>
      </Field>

      {/* Cycles */}
      <Field label="Cycles" hint="Number of research cycles to run">
        <input
          type="number"
          min={1}
          max={20}
          value={cycles}
          onChange={(e) => setCycles(Number(e.target.value))}
          className="input w-24"
        />
      </Field>

      {/* Memory access */}
      <Field label="Memory access" hint="Cross-run knowledge available to the agent">
        <div className="space-y-1.5">
          {MEMORY_OPTIONS.map((opt) => (
            <label key={opt.value} className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="memory"
                value={opt.value}
                checked={memoryAccess === opt.value}
                onChange={() => setMemoryAccess(opt.value)}
                className="accent-brand"
              />
              <span className="text-sm text-gray-300">{opt.label}</span>
            </label>
          ))}
        </div>
      </Field>

      {/* Scope */}
      <Field label="Scope">
        <div className="flex gap-2">
          {["research", "analyst"].map((s) => (
            <button
              key={s}
              onClick={() => setScope(s)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                scope === s
                  ? s === "analyst"
                    ? "bg-yellow-700 text-yellow-100"
                    : "bg-brand text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              }`}
            >
              {s}
              {s === "analyst" && " ⚠"}
            </button>
          ))}
        </div>
        {scope === "analyst" && (
          <p className="text-yellow-600 text-xs mt-1">
            Analyst scope lifts the run-scope guard. Only use for deliberate cross-run analysis.
          </p>
        )}
      </Field>

      {/* Guidance */}
      <Field label="Guidance" hint="Free-text appended to the seed prompt">
        <textarea
          value={guidance}
          onChange={(e) => setGuidance(e.target.value)}
          rows={4}
          className="input w-full resize-none"
          placeholder="e.g. Focus on GLM-based models next. Try log-linked Tweedie."
        />
      </Field>

      {error && (
        <div className="card border-red-900 text-red-400 text-sm">{error}</div>
      )}

      <button
        onClick={launch}
        disabled={launching || !surfaceAvailable(surface)}
        className="px-6 py-2.5 bg-brand hover:bg-brand-dark rounded text-white font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {launching ? "Launching…" : "Launch Run"}
      </button>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-1">{label}</label>
      {hint && <p className="text-xs text-gray-600 mb-2">{hint}</p>}
      {children}
    </div>
  );
}
