import Link from "next/link";
import { api } from "@/lib/api";
import { ArtifactViewer } from "@/components/ArtifactViewer";
import { ResearchLog } from "@/components/ResearchLog";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({
  params,
}: {
  params: { track: string; runId: string };
}) {
  const { track, runId } = params;

  const [summary, experiments, comparisons, researchLines, proposals, sessions] =
    await Promise.allSettled([
      api.runSummary(track, runId),
      api.experiments(track, runId),
      api.comparisons(track, runId),
      api.researchLines(track, runId),
      api.proposals(track, runId),
      api.sessions(track, runId),
    ]);

  const s = summary.status === "fulfilled" ? summary.value : null;
  const exps = experiments.status === "fulfilled" ? experiments.value : [];
  const comps = comparisons.status === "fulfilled" ? comparisons.value : [];
  const lines = researchLines.status === "fulfilled" ? researchLines.value : [];
  const props = proposals.status === "fulfilled" ? proposals.value : [];
  const sess = sessions.status === "fulfilled" ? sessions.value : [];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="text-gray-500 text-xs mb-1">
          <Link href="/runs" className="hover:text-gray-300">Runs</Link>
          {" / "}
          <Link href={`/runs?track=${track}`} className="hover:text-gray-300">{track}</Link>
          {" / "}
          <span className="text-gray-300">{runId}</span>
        </div>
        <h1 className="text-2xl font-bold font-mono">{runId}</h1>
        <p className="text-gray-500 mt-0.5">Track: {track}</p>
      </div>

      {/* Counts bar */}
      {s?.counts && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <CountCard label="Experiments" value={s.counts.experiments} />
          <CountCard label="Comparisons" value={s.counts.comparisons} />
          <CountCard label="Proposals" value={s.counts.proposals} />
          <CountCard label="Artifacts" value={s.counts.artifacts} />
        </div>
      )}

      {/* Champion */}
      {s?.champion && (
        <section className="card border-green-900">
          <h2 className="text-sm font-semibold text-green-400 mb-3">Current Champion</h2>
          <ChampionCard champ={s.champion as Record<string, unknown>} />
        </section>
      )}

      {/* Research lines */}
      {lines.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-3 text-gray-300">Research Lines</h2>
          <div className="grid gap-2">
            {(lines as Record<string, unknown>[]).map((l, i) => (
              <div key={i} className="card flex items-start gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-200">
                      {(l.label as string) ?? (l.line_id as string)}
                    </span>
                    <StatusBadge status={(l.status as string) ?? "active"} />
                  </div>
                  {l.hypothesis != null && (
                    <p className="text-gray-500 text-xs mt-1">{String(l.hypothesis)}</p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Experiments table */}
      <section>
        <h2 className="text-lg font-semibold mb-3 text-gray-300">
          Experiments ({exps.length})
        </h2>
        {exps.length === 0 ? (
          <p className="text-gray-500 text-sm">No experiments yet.</p>
        ) : (
          <div className="card overflow-auto">
            <table className="table-base">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Name</th>
                  <th>Model Family</th>
                  <th>Gini (weighted)</th>
                  <th>Target</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {(exps as Record<string, unknown>[]).map((e, i) => (
                  <tr key={i}>
                    <td className="font-mono text-xs text-gray-400">
                      {(e.experiment_id as string)?.slice(0, 8)}…
                    </td>
                    <td>{(e.experiment_name as string) ?? "—"}</td>
                    <td>{(e.model_family as string) ?? "—"}</td>
                    <td className="text-green-400 font-mono">
                      {e.gini_weighted != null
                        ? Number(e.gini_weighted).toFixed(4)
                        : "—"}
                    </td>
                    <td className="text-gray-400">
                      {(e.target_strategy as string) ?? (e.target_mode as string) ?? "—"}
                    </td>
                    <td>
                      <StatusBadge status={(e.status as string) ?? ""} />
                    </td>
                    <td className="text-gray-500">
                      {e.created_at ? String(e.created_at).slice(0, 16) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Comparisons table */}
      <section>
        <h2 className="text-lg font-semibold mb-3 text-gray-300">
          Comparisons ({comps.length})
        </h2>
        {comps.length === 0 ? (
          <p className="text-gray-500 text-sm">No comparisons yet.</p>
        ) : (
          <div className="card overflow-auto">
            <table className="table-base">
              <thead>
                <tr>
                  <th>Champion</th>
                  <th>Challenger</th>
                  <th>Mean Lift</th>
                  <th>Win Rate</th>
                  <th>Decision</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {(comps as Record<string, unknown>[]).map((c, i) => (
                  <tr key={i}>
                    <td className="font-mono text-xs text-gray-400">
                      {(c.champion_id as string)?.slice(0, 8)}…
                    </td>
                    <td className="font-mono text-xs text-gray-400">
                      {(c.challenger_id as string)?.slice(0, 8)}…
                    </td>
                    <td
                      className={
                        (c.mean_lift as number) > 0
                          ? "text-green-400 font-mono"
                          : "text-red-400 font-mono"
                      }
                    >
                      {c.mean_lift != null
                        ? `${Number(c.mean_lift) > 0 ? "+" : ""}${Number(c.mean_lift).toFixed(4)}`
                        : "—"}
                    </td>
                    <td className="font-mono">
                      {c.win_rate != null ? `${(Number(c.win_rate) * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td>
                      <DecisionBadge decision={(c.decision as string) ?? ""} />
                    </td>
                    <td className="text-gray-500">
                      {c.created_at ? String(c.created_at).slice(0, 16) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Research Log */}
      {s?.research_log_preview && (
        <section>
          <h2 className="text-lg font-semibold mb-3 text-gray-300">Research Log</h2>
          <ResearchLog text={s.research_log_preview} track={track} runId={runId} />
        </section>
      )}

      {/* Artifact browser */}
      <section>
        <h2 className="text-lg font-semibold mb-3 text-gray-300">Artifacts</h2>
        <ArtifactViewer track={track} runId={runId} />
      </section>
    </div>
  );
}

function CountCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="card text-center">
      <div className="text-xl font-bold text-gray-100">{value}</div>
      <div className="text-gray-500 text-xs mt-0.5">{label}</div>
    </div>
  );
}

function ChampionCard({ champ }: { champ: Record<string, unknown> }) {
  return (
    <div className="flex flex-wrap gap-6 text-sm">
      <KV k="Name" v={(champ.experiment_name as string) ?? "—"} />
      <KV k="Model" v={(champ.model_family as string) ?? "—"} />
      <KV
        k="Gini (weighted)"
        v={
          champ.gini_weighted != null
            ? Number(champ.gini_weighted).toFixed(4)
            : "—"
        }
        highlight
      />
      <KV
        k="Champion Experiment"
        v={(champ.champion_id as string)?.slice(0, 24) ?? "—"}
      />
      <KV k="Status" v={(champ.status as string) ?? "—"} />
    </div>
  );
}

function KV({
  k,
  v,
  highlight,
}: {
  k: string;
  v: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="text-gray-500 text-xs">{k}</div>
      <div className={highlight ? "text-green-400 font-mono" : "text-gray-200"}>
        {v}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === "ok" || status === "promoted" || status === "active"
      ? "badge-green"
      : status === "failed" || status === "rejected"
      ? "badge-red"
      : status === "pending" || status === "queued"
      ? "badge-yellow"
      : "badge-gray";
  return <span className={`badge ${cls}`}>{status || "unknown"}</span>;
}

function DecisionBadge({ decision }: { decision: string }) {
  const cls =
    decision === "promoted" ? "badge-green" : decision === "rejected" ? "badge-red" : "badge-gray";
  return <span className={`badge ${cls}`}>{decision || "—"}</span>;
}
