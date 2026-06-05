import { api } from "@/lib/api";
import { HoldoutPanel } from "@/components/HoldoutPanel";
import { HarvestButton } from "@/components/HarvestButton";

export const dynamic = "force-dynamic";

export default async function LeaderboardPage() {
  const [lb, playbook, tracks] = await Promise.all([
    api.leaderboard().catch(() => null),
    api.playbook().catch(() => null),
    api.tracks().catch(() => [] as import("@/lib/api").Track[]),
  ]);

  if (!lb?.available) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Leaderboard</h1>
        <div className="card space-y-3">
          <p className="text-gray-400">
            Memory aggregator not populated yet. Click below to harvest all existing runs.
          </p>
          <HarvestButton />
          <p className="text-gray-600 text-xs">
            Equivalent CLI: <code className="text-brand">autoresearch memory harvest --all</code>
          </p>
        </div>
        <HoldoutPanel tracks={tracks} />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Leaderboard</h1>
          <p className="text-gray-500 mt-1">
            Cross-run search-split metrics — safe for agent access. Holdout evaluation is gated below.
          </p>
        </div>
        <HarvestButton />
      </div>

      {/* Top experiments */}
      <section>
        <h2 className="text-lg font-semibold mb-3 text-gray-300">Top Experiments (search split)</h2>
        <div className="card overflow-auto">
          <table className="table-base">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Track</th>
                <th>Run</th>
                <th>Model</th>
                <th>Family</th>
                <th>Gini (weighted)</th>
                <th>Mean Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {lb.top_experiments.map((e, i) => (
                <tr key={e.experiment_uid}>
                  <td className="text-gray-500">{i + 1}</td>
                  <td>{e.track_id}</td>
                  <td className="font-mono text-xs text-gray-400">{e.run_id}</td>
                  <td className="text-gray-400">{e.model_name}</td>
                  <td>{e.model_family ?? "—"}</td>
                  <td
                    className={
                      i === 0 ? "text-yellow-400 font-mono font-bold" : "text-green-400 font-mono"
                    }
                  >
                    {e.gini_weighted != null ? e.gini_weighted.toFixed(4) : "—"}
                  </td>
                  <td className="font-mono text-gray-400">
                    {e.mean_score != null ? Number(e.mean_score).toFixed(4) : "—"}
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        e.status === "ok" ? "badge-green" : "badge-gray"
                      }`}
                    >
                      {e.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Runs summary */}
      <section>
        <h2 className="text-lg font-semibold mb-3 text-gray-300">Runs Summary</h2>
        <div className="card overflow-auto">
          <table className="table-base">
            <thead>
              <tr>
                <th>Track</th>
                <th>Run</th>
                <th>Model</th>
                <th>Provider</th>
                <th>Experiments</th>
                <th>Promotions</th>
                <th>Peak Gini</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {lb.runs.map((r) => (
                <tr key={r.run_uid}>
                  <td>{r.track_id}</td>
                  <td className="font-mono text-xs text-gray-400">{r.run_id}</td>
                  <td>{r.model_name}</td>
                  <td className="text-gray-500">{r.provider}</td>
                  <td>{r.n_experiments}</td>
                  <td>{r.n_promotions}</td>
                  <td className="text-green-400 font-mono">
                    {r.peak_gini != null ? r.peak_gini.toFixed(4) : "—"}
                  </td>
                  <td className="text-gray-500">{r.started_at?.slice(0, 16) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Insights / Playbook */}
      {(lb.insights?.length > 0 || playbook?.available) && (
        <section>
          <h2 className="text-lg font-semibold mb-3 text-gray-300">Insights Playbook</h2>
          {playbook?.content && (
            <div className="card">
              <pre className="text-xs text-gray-300 whitespace-pre-wrap overflow-auto max-h-64">
                {playbook.content}
              </pre>
            </div>
          )}
        </section>
      )}

      {/* Gated holdout panel */}
      <HoldoutPanel tracks={tracks} />
    </div>
  );
}
