import Link from "next/link";
import { api } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function MonitorIndexPage() {
  const jobs = await api.jobs().catch(() => []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-100">Monitor</h1>
          <p className="text-gray-500 mt-1">Active and recent agent jobs</p>
        </div>
        <Link
          href="/launch"
          className="px-4 py-2 bg-brand hover:bg-brand-dark rounded text-white text-sm transition-colors"
        >
          + Launch Run
        </Link>
      </div>

      {jobs.length === 0 ? (
        <div className="card text-gray-500">
          No jobs yet.{" "}
          <Link href="/launch" className="text-brand hover:underline">
            Launch a run
          </Link>{" "}
          to get started.
        </div>
      ) : (
        <div className="card overflow-auto">
          <table className="table-base">
            <thead>
              <tr>
                <th>Job</th>
                <th>Track</th>
                <th>Surface</th>
                <th>Model</th>
                <th>Cycles</th>
                <th>Status</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id}>
                  <td className="font-mono text-xs">{j.id}</td>
                  <td>{j.track}</td>
                  <td>
                    <span className="badge badge-blue">{j.surface}</span>
                  </td>
                  <td className="text-gray-400">{j.model_name ?? "—"}</td>
                  <td>{j.cycles}</td>
                  <td>
                    <JobStatusBadge status={j.status} />
                  </td>
                  <td className="text-gray-500">{j.created_at?.slice(0, 16)}</td>
                  <td>
                    <Link
                      href={`/monitor/${j.id}`}
                      className="text-brand hover:underline text-xs"
                    >
                      View →
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

function JobStatusBadge({ status }: { status: string }) {
  const cls =
    status === "running"
      ? "badge-green"
      : status === "done"
      ? "badge-blue"
      : status === "failed"
      ? "badge-red"
      : status === "stopped"
      ? "badge-gray"
      : "badge-yellow";
  return <span className={`badge ${cls}`}>{status}</span>;
}
