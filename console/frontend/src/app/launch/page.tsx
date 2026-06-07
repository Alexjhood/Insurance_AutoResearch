"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type SurfaceInfo } from "@/lib/api";

const MEMORY_OPTIONS = [
  { value: "none", label: "None — fully isolated (default)" },
  { value: "own", label: "Own — this model's history only" },
  { value: "all", label: "All — all models, attributed" },
];

export default function LaunchPage() {
  const router = useRouter();
  const [surfaces, setSurfaces] = useState<SurfaceInfo[]>([]);
  const [surface, setSurface] = useState<string>("claude");
  const [track, setTrack] = useState("claude");
  const [model, setModel] = useState("");
  const [effort, setEffort] = useState("");
  const [cycles, setCycles] = useState(3);
  const [memoryAccess, setMemoryAccess] = useState("none");
  const [scope, setScope] = useState("research");
  const [guidance, setGuidance] = useState("");
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load the per-surface model + effort catalog once
  useEffect(() => {
    api.surfaces().then((s) => {
      setSurfaces(s);
      const first = s.find((x) => x.available) ?? s[0];
      if (first) {
        setSurface(first.surface);
        setTrack(first.surface);
        setModel(first.default_model);
        setEffort("");
      }
    }).catch(() => {});
  }, []);

  const current = useMemo(
    () => surfaces.find((s) => s.surface === surface),
    [surfaces, surface]
  );
  const surfaceAvailable = (s: string) =>
    Boolean(surfaces.find((x) => x.surface === s)?.available);

  function onSurfaceChange(s: string) {
    setSurface(s);
    setTrack(s);
    const info = surfaces.find((x) => x.surface === s);
    setModel(info?.default_model ?? "");
    setEffort("");
  }

  async function launch() {
    setLaunching(true);
    setError(null);
    try {
      const { job_id } = await api.launchJob({
        track,
        surface,
        model,
        effort,
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
          {surfaces.map((s) => (
            <button
              key={s.surface}
              onClick={() => onSurfaceChange(s.surface)}
              disabled={!s.available}
              title={s.available ? "" : `${s.surface} CLI not found on the server`}
              className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                surface === s.surface
                  ? "bg-brand text-white"
                  : "bg-gray-800 text-gray-400 hover:bg-gray-700"
              } ${!s.available ? "opacity-40 cursor-not-allowed line-through" : ""}`}
            >
              {s.surface}
            </button>
          ))}
        </div>
        {current && !current.available && (
          <p className="text-yellow-600 text-xs mt-2">
            ⚠ The <strong>{surface}</strong> CLI was not found on the orchestrator host.
            Install it (and ensure it&apos;s authenticated), or set the{" "}
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

      {/* Agent model — datalist gives type-to-filter + custom entry */}
      <Field
        label="Agent model"
        hint={
          current?.allows_custom_model
            ? "Pick a suggestion or type any model the CLI accepts."
            : `Choose from ${current?.models.length ?? 0} models the CLI reports.`
        }
      >
        <input
          list="model-options"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="input w-full"
          placeholder={current?.default_model || "default model"}
        />
        <datalist id="model-options">
          {(current?.models ?? []).map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
        {model === "" && (
          <p className="text-gray-600 text-xs mt-1">
            Empty = let the {surface} CLI use its configured default model.
          </p>
        )}
      </Field>

      {/* Thinking / reasoning effort */}
      <Field
        label={current?.effort_label ?? "Thinking effort"}
        hint="Higher effort = more reasoning/thinking budget (and cost). Default leaves it to the model."
      >
        <select
          value={effort}
          onChange={(e) => setEffort(e.target.value)}
          className="input w-48"
        >
          {(current?.efforts ?? [""]).map((e) => (
            <option key={e} value={e}>
              {e === "" ? "Default" : e}
            </option>
          ))}
        </select>
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
