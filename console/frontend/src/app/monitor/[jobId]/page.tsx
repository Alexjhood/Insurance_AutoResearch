import { api } from "@/lib/api";
import { LiveMonitor } from "@/components/LiveMonitor";

export const dynamic = "force-dynamic";

export default async function MonitorJobPage({
  params,
}: {
  params: { jobId: string };
}) {
  const { jobId } = params;
  const job = await api.job(jobId).catch(() => null);
  const initialEvents = await api.jobEvents(jobId).catch(() => []);

  if (!job) {
    return (
      <div className="card text-gray-500">
        Job <code className="text-brand">{jobId}</code> not found.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="text-gray-500 text-xs mb-1">Monitor / {jobId}</div>
          <h1 className="text-xl font-bold font-mono">
            {job.track} / {job.run_id}
          </h1>
          <div className="flex gap-3 mt-2">
            <span className="badge badge-blue">{job.surface}</span>
            <span className="text-gray-500 text-xs">{job.model_provider}/{job.model_name}</span>
            <span className="text-gray-500 text-xs">{job.cycles} cycles</span>
            <span className="text-gray-500 text-xs">memory: {job.memory_access}</span>
          </div>
        </div>
        <JobStatusPill status={job.status} />
      </div>

      {/* Live monitor + steer panel (client component) */}
      <LiveMonitor job={job} initialEvents={initialEvents} />

      {/* Link to run detail once running */}
      {job.run_id && (
        <div className="text-xs text-gray-500">
          Run artifacts:{" "}
          <a
            href={`/runs/${job.track}/${job.run_id}`}
            className="text-brand hover:underline"
          >
            /runs/{job.track}/{job.run_id} →
          </a>
        </div>
      )}
    </div>
  );
}

function JobStatusPill({ status }: { status: string }) {
  const cls =
    status === "running"
      ? "bg-green-900 text-green-300 animate-pulse"
      : status === "done"
      ? "bg-blue-900 text-blue-300"
      : status === "failed"
      ? "bg-red-900 text-red-300"
      : "bg-gray-800 text-gray-400";
  return (
    <span className={`${cls} px-3 py-1 rounded-full text-xs font-medium`}>
      {status}
    </span>
  );
}
