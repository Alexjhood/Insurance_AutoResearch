import Link from "next/link";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  const [tracks, leaderboard] = await Promise.allSettled([
    api.tracks(),
    api.leaderboard(),
  ]);

  const trackList = tracks.status === "fulfilled" ? tracks.value : [];
  const lb = leaderboard.status === "fulfilled" ? leaderboard.value : null;

  const totalRuns = trackList.reduce((s, t) => s + t.n_runs, 0);
  const bestGini = lb?.runs?.length
    ? Math.max(...lb.runs.map((r) => r.peak_gini ?? 0))
    : null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">AutoResearch Console</h1>
        <p className="text-gray-500 mt-1">Insurance target-modelling research loop</p>
      </div>

      {/* Quick stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat label="Tracks" value={trackList.length} />
        <Stat label="Total Runs" value={totalRuns} />
        <Stat label="Models Seen" value={lb?.models?.length ?? "—"} />
        <Stat label="Best Gini" value={bestGini != null ? bestGini.toFixed(4) : "—"} />
      </div>

      {/* Track cards */}
      <div>
        <h2 className="text-lg font-semibold mb-3 text-gray-300">Tracks</h2>
        {trackList.length === 0 ? (
          <p className="text-gray-500 text-sm">
            No tracks yet. Run{" "}
            <code className="text-brand">autoresearch bootstrap-track</code> to create one.
          </p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trackList.map((t) => (
              <Link key={t.track_id} href={`/runs?track=${t.track_id}`}>
                <div className="card hover:border-brand/50 cursor-pointer transition-colors">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-100 font-semibold">{t.track_id}</span>
                    <span className="badge badge-gray">{t.n_runs} runs</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>

      {/* Recent top experiments */}
      {lb?.top_experiments?.length ? (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold text-gray-300">Top Experiments</h2>
            <Link href="/leaderboard" className="text-brand text-xs hover:underline">
              Full leaderboard →
            </Link>
          </div>
          <div className="card overflow-auto">
            <table className="table-base">
              <thead>
                <tr>
                  <th>Track</th>
                  <th>Run</th>
                  <th>Model</th>
                  <th>Family</th>
                  <th>Gini (weighted)</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {lb.top_experiments.slice(0, 10).map((e) => (
                  <tr key={e.experiment_uid}>
                    <td>{e.track_id}</td>
                    <td>
                      <Link
                        href={`/runs/${e.track_id}/${e.run_id}`}
                        className="text-brand hover:underline"
                      >
                        {e.run_id}
                      </Link>
                    </td>
                    <td className="text-gray-400">{e.model_name}</td>
                    <td>{e.model_family ?? "—"}</td>
                    <td className="text-green-400 font-mono">
                      {e.gini_weighted != null ? e.gini_weighted.toFixed(4) : "—"}
                    </td>
                    <td>
                      <StatusBadge status={e.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="card text-center">
      <div className="text-2xl font-bold text-gray-100">{value}</div>
      <div className="text-gray-500 text-xs mt-1">{label}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "ok" || status === "promoted"
      ? "badge-green"
      : status === "failed" || status === "rejected"
      ? "badge-red"
      : status === "pending"
      ? "badge-yellow"
      : "badge-gray";
  return <span className={`badge ${cls}`}>{status}</span>;
}
