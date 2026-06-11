// Telemetry import — OpenCode adapter.
//
// Brings OpenCode to parity with Codex and Claude Code, both of which import
// agent telemetry on SessionStart/Stop. OpenCode has no file transcript: its
// history lives in ~/.local/share/opencode/opencode.db, which the shared Python
// importer reads directly (`--surface opencode`). We trigger that import when a
// session goes idle (a turn has completed) — the importer is idempotent, so
// re-importing the same session on every idle event simply refreshes the run's
// telemetry store. Without this hook, OpenCode runs record no usage data and
// cross-track cost comparisons are not apples-to-apples.

import { spawnSync } from "node:child_process";
import { join } from "node:path";

export const TelemetryImport = async ({ directory, worktree }) => {
  const root = worktree || directory || process.cwd();
  const script = join(root, "scripts", "import_agent_telemetry.py");

  const importSession = (sessionId) => {
    if (!sessionId) return;
    const payload = JSON.stringify({
      hook_event_name: "Stop",
      session_id: sessionId,
      cwd: root,
    });
    // Fail open: a telemetry fault must never interrupt the research session.
    spawnSync(
      "python3",
      [script, "--surface", "opencode", "--finalize-turn"],
      { input: payload, encoding: "utf8" },
    );
  };

  return {
    event: async ({ event }) => {
      if (!event || event.type !== "session.idle") return;
      const props = event.properties || {};
      importSession(props.sessionID || props.session_id || props.sessionId);
    },
  };
};
