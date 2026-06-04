// Run-scope guard — OpenCode adapter.
//
// Thin bridge to the shared Python guard at scripts/run_scope_guard.py so all
// three harnesses (Claude Code, Codex, OpenCode) share one decision "brain".
// The `tool.execute.before` hook blocks a foreign-run access by throwing; the
// `tool.execute.after` hook lets the Python side bind the session to its run.
//
// OpenCode tool names are lower-case (bash, read, grep, glob, edit, write) and
// pass args like { command } or { filePath }; the Python guard normalises both.

import { spawnSync } from "node:child_process";
import { join } from "node:path";

export const RunScopeGuard = async ({ directory, worktree }) => {
  const root = worktree || directory || process.cwd();
  const script = join(root, "scripts", "run_scope_guard.py");

  const invoke = (event, input, output) => {
    const args = (output && output.args) || (input && input.args) || {};
    const payload = JSON.stringify({
      hook_event_name: event,
      session_id: (input && (input.sessionID || input.session_id)) || "",
      cwd: root,
      tool_name: (input && input.tool) || "",
      tool_input: args,
    });
    // spawnSync: a guard fault returns no clean exit-2, so we fail open below.
    return spawnSync("python3", [script], { input: payload, encoding: "utf8" });
  };

  return {
    "tool.execute.before": async (input, output) => {
      const res = invoke("PreToolUse", input, output);
      if (res && res.status === 2) {
        throw new Error((res.stderr || "blocked by run-scope guard").trim());
      }
      // any other outcome (0, error, missing python) -> allow (fail open)
    },
    "tool.execute.after": async (input, output) => {
      invoke("PostToolUse", input, output);
    },
  };
};
