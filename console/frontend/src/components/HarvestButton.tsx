"use client";

import { useState } from "react";

export function HarvestButton({ onDone }: { onDone?: () => void }) {
  const [state, setState] = useState<"idle" | "running" | "done" | "error">("idle");
  const [detail, setDetail] = useState("");

  async function harvest() {
    setState("running");
    setDetail("");
    try {
      const res = await fetch(`${(await import("@/lib/config")).API_BASE}/api/leaderboard/harvest`, { method: "POST" });
      const data = await res.json();
      if (data.ok) {
        setState("done");
        setDetail(data.stdout?.trim().split("\n").slice(-3).join(" | ") || "Harvest complete.");
        onDone?.();
      } else {
        setState("error");
        setDetail(data.stderr || data.stdout || "Harvest failed.");
      }
    } catch (err) {
      setState("error");
      setDetail(String(err));
    }
  }

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={harvest}
        disabled={state === "running"}
        className="px-3 py-1.5 rounded border border-gray-700 text-xs text-gray-400
                   hover:bg-gray-800 hover:text-gray-200 transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {state === "running" ? "Harvesting…" : "Harvest memory"}
      </button>
      {detail && (
        <span
          className={`text-xs font-mono truncate max-w-xs ${
            state === "error" ? "text-red-400" : "text-green-400"
          }`}
        >
          {detail}
        </span>
      )}
    </div>
  );
}
