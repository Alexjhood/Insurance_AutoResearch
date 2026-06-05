import Link from "next/link";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function RunsPage({
  searchParams,
}: {
  searchParams: { track?: string };
}) {
  const tracks = await api.tracks().catch(() => []);
  const activeTrack = searchParams.track ?? tracks[0]?.track_id ?? null;

  const runs = activeTrack ? await api.runs(activeTrack).catch(() => []) : [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-100">Runs</h1>
        <p className="text-gray-500 mt-1">All research runs across tracks</p>
      </div>

      {/* Track tabs */}
      <div className="flex gap-2 flex-wrap">
        {tracks.map((t) => (
          <Link
            key={t.track_id}
            href={`/runs?track=${t.track_id}`}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
              t.track_id === activeTrack
                ? "bg-brand text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"
            }`}
          >
            {t.track_id}
            <span className="ml-1.5 opacity-60">{t.n_runs}</span>
          </Link>
        ))}
      </div>

      {/* Runs table */}
      {!activeTrack ? (
        <p className="text-gray-500 text-sm">No tracks found.</p>
      ) : runs.length === 0 ? (
        <p className="text-gray-500 text-sm">No runs in track "{activeTrack}".</p>
      ) : (
        <div className="card overflow-auto">
          <table className="table-base">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Experiments</th>
                <th>Comparisons</th>
                <th>Proposals</th>
                <th>Champion Gini</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td>
                    <Link
                      href={`/runs/${activeTrack}/${r.run_id}`}
                      className="text-brand hover:underline font-mono"
                    >
                      {r.run_id}
                    </Link>
                  </td>
                  <td>{r.counts?.experiments ?? "—"}</td>
                  <td>{r.counts?.comparisons ?? "—"}</td>
                  <td>{r.counts?.proposals ?? "—"}</td>
                  <td className="text-green-400 font-mono">
                    {r.champion?.gini_weighted != null
                      ? Number(r.champion.gini_weighted).toFixed(4)
                      : "—"}
                  </td>
                  <td>
                    {r.has_registry ? (
                      <span className="badge badge-green">active</span>
                    ) : (
                      <span className="badge badge-gray">no registry</span>
                    )}
                  </td>
                  <td>
                    <Link
                      href={`/runs/${activeTrack}/${r.run_id}`}
                      className="text-gray-500 hover:text-gray-300 text-xs"
                    >
                      Inspect →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
