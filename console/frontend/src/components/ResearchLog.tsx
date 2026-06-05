"use client";

import { useState } from "react";

export function ResearchLog({
  text,
  track,
  runId,
}: {
  text: string;
  track: string;
  runId: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const preview = text.slice(0, 800);
  const truncated = text.length > 800;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500 font-mono">RESEARCH_LOG.md</span>
        <a
          href={`/api/tracks/${track}/runs/${runId}/artifacts/RESEARCH_LOG.md`}
          target="_blank"
          rel="noreferrer"
          className="text-xs text-brand hover:underline"
        >
          Open raw ↗
        </a>
      </div>
      <pre className="text-xs text-gray-300 whitespace-pre-wrap overflow-auto">
        {expanded ? text : preview}
        {truncated && !expanded && "…"}
      </pre>
      {truncated && (
        <button
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 text-xs text-brand hover:underline"
        >
          {expanded ? "Show less" : "Show full log"}
        </button>
      )}
    </div>
  );
}
