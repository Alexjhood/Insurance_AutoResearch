"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export function ArtifactViewer({
  track,
  runId,
}: {
  track: string;
  runId: string;
}) {
  const [paths, setPaths] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    api.artifactPaths(track, runId).then(setPaths).catch(() => setPaths([]));
  }, [track, runId]);

  async function open(path: string) {
    setSelected(path);
    setContent(null);
    setLoading(true);
    try {
      const res = await fetch(`/api/tracks/${track}/runs/${runId}/artifacts/${path}`);
      if (path.endsWith(".html")) {
        setContent(await res.text());
      } else {
        setContent(await res.text());
      }
    } catch {
      setContent("(failed to load)");
    }
    setLoading(false);
  }

  const filtered = paths.filter((p) =>
    filter ? p.toLowerCase().includes(filter.toLowerCase()) : true
  );

  const isHtml = selected?.endsWith(".html");

  return (
    <div className="flex gap-4 min-h-[400px]">
      {/* File tree */}
      <div className="w-64 shrink-0">
        <input
          type="text"
          placeholder="Filter…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs mb-2 outline-none focus:border-brand"
        />
        <div className="bg-gray-900 border border-gray-800 rounded overflow-y-auto max-h-[500px]">
          {filtered.length === 0 ? (
            <p className="text-gray-600 text-xs p-3">No artifacts.</p>
          ) : (
            filtered.map((p) => (
              <button
                key={p}
                onClick={() => open(p)}
                className={`w-full text-left px-2 py-1 text-xs font-mono truncate hover:bg-gray-800 transition-colors ${
                  selected === p ? "bg-gray-800 text-brand" : "text-gray-400"
                }`}
              >
                {p}
              </button>
            ))
          )}
        </div>
      </div>

      {/* Preview pane */}
      <div className="flex-1 card overflow-auto">
        {!selected && (
          <p className="text-gray-600 text-sm">Select a file to preview.</p>
        )}
        {loading && <p className="text-gray-500 text-sm">Loading…</p>}
        {content !== null && !loading && (
          <>
            <div className="text-xs text-gray-500 mb-3 font-mono">{selected}</div>
            {isHtml ? (
              <iframe
                srcDoc={content}
                className="w-full h-[600px] border border-gray-800 rounded bg-white"
                sandbox="allow-scripts"
              />
            ) : (
              <pre className="text-xs text-gray-300 whitespace-pre-wrap overflow-auto max-h-[600px]">
                {content}
              </pre>
            )}
          </>
        )}
      </div>
    </div>
  );
}
