"""Tests for the run-scope guard hook (.claude/hooks/run_scope_guard.py).

The guard keeps a research session inside its own run folder while letting an
analyst session see everything. These tests exercise the pure decision logic,
the reference parser, and the path extractor; the hook's I/O wiring is thin
around them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_GUARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_scope_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("run_scope_guard", _GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


BOUND = {"mode": "research", "track": "claude", "run_id": "RUN_A"}
ANALYST = {"mode": "analyst"}

OWN = "artifacts/tracks/claude/runs/RUN_A/RESEARCH_LOG.md"
FOREIGN_RUN = "artifacts/tracks/claude/runs/RUN_B/RESEARCH_LOG.md"
FOREIGN_TRACK = "artifacts/tracks/codex/runs/RUN_X/registry.sqlite"
RUNS_DIR = "artifacts/tracks/claude/runs"
ABS_OWN = "/Users/x/Insurance_AutoResearch/artifacts/tracks/claude/runs/RUN_A/context/latest_context.json"

ORCHESTRATOR = {
    "mode": "orchestrator",
    "orchestration_id": "20260710T120000Z",
    "child_runs": [
        {"track": "claude", "run_id": "20260710T120100Z"},
        {"track": "codex", "run_id": "20260710T120200Z"},
    ],
}


# ── reference parser ────────────────────────────────────────────────────────

def test_find_run_refs_specific_and_dir():
    assert guard.find_run_refs(OWN) == [("claude", "RUN_A")]
    assert guard.find_run_refs(RUNS_DIR) == [("claude", None)]


def test_find_run_refs_ignores_non_run_paths():
    # latest_run.json and the .scope dir are not under runs/ and must not match.
    assert guard.find_run_refs("artifacts/tracks/claude/latest_run.json") == []
    assert guard.find_run_refs("artifacts/tracks/.scope/sess.json") == []
    assert guard.find_run_refs("src/autoresearch/models/global_mean.py") == []


# ── decision logic ──────────────────────────────────────────────────────────

def test_own_run_allowed():
    assert guard.decide(BOUND, [OWN])[0]
    assert guard.decide(BOUND, [ABS_OWN])[0]


def test_foreign_run_same_track_denied():
    allow, reason = guard.decide(BOUND, [FOREIGN_RUN])
    assert not allow and "different run" in reason


def test_foreign_track_denied():
    assert not guard.decide(BOUND, [FOREIGN_TRACK])[0]


def test_runs_dir_enumeration_denied():
    allow, reason = guard.decide(BOUND, [RUNS_DIR])
    assert not allow and "sibling runs" in reason


def test_unbound_denies_run_artifacts_until_bootstrap():
    for text in (OWN, FOREIGN_RUN, FOREIGN_TRACK, RUNS_DIR):
        allow, reason = guard.decide(None, [text])
        assert not allow
        assert "run not bound yet" in reason


def test_analyst_sees_everything():
    for text in (OWN, FOREIGN_RUN, FOREIGN_TRACK, RUNS_DIR):
        assert guard.decide(ANALYST, [text])[0]


def test_non_run_paths_always_allowed():
    for scope in (None, BOUND, ANALYST):
        assert guard.decide(scope, ["src/autoresearch/cli.py", "configs/default.toml"])[0]


def test_research_denies_orchestrate_actions_except_finish_delegation():
    for action in (
        "status --orchestration-id 20260710T120000Z",
        "collect",
        "report",
        "spawn --brief b.json --backend stub",
        "respawn --delegation d01 --brief b.json",
        "kill --delegation d01",
        "playoff --auto-decide",
        "new --dataset french_motor --total-cycles 2",
        "note --text x --kind other",
        "backend-stats",
        "list-backends",
    ):
        allow, reason = guard.decide(BOUND, [f"autoresearch orchestrate {action}"])
        assert not allow, action
        assert "finish-delegation" in reason


def test_research_allows_own_finish_delegation():
    cmd = (
        "autoresearch --track claude --run-id RUN_A "
        'orchestrate finish-delegation --summary "done"'
    )

    assert guard.decide(BOUND, [cmd])[0]


# ── orchestrator scope ──────────────────────────────────────────────────────

def test_orchestrator_reads_own_child_run():
    path = "artifacts/tracks/claude/runs/20260710T120100Z/context/latest_context.json"

    assert guard.decide(ORCHESTRATOR, [path])[0]


def test_orchestrator_denies_foreign_run():
    path = "artifacts/tracks/claude/runs/20260710T999999Z/RESEARCH_LOG.md"
    allow, reason = guard.decide(ORCHESTRATOR, [path])

    assert not allow
    assert "unlisted child run" in reason


def test_orchestrator_denies_another_orchestration_folder():
    path = "artifacts/orchestrations/20260710T130000Z/orchestration.json"
    allow, reason = guard.decide(ORCHESTRATOR, [path])

    assert not allow
    assert "another orchestration" in reason


def test_orchestrator_allows_child_takeover_command():
    cmd = (
        "autoresearch --track codex --run-id 20260710T120200Z "
        "run-session-cycles 1"
    )

    assert guard.decide(ORCHESTRATOR, [cmd])[0]


def test_orchestrator_denies_runs_dir_enumeration():
    allow, reason = guard.decide(ORCHESTRATOR, ["artifacts/tracks/claude/runs"])

    assert not allow
    assert "enumerating sibling runs" in reason


def test_orchestrator_hydrates_consolidation_run_from_manifest(tmp_path, monkeypatch):
    oid = "20260710T120000Z"
    orchestration_dir = tmp_path / "artifacts" / "orchestrations" / oid
    orchestration_dir.mkdir(parents=True)
    (orchestration_dir / "orchestration.json").write_text(
        json.dumps(
            {
                "delegations": [],
                "consolidation": {
                    "track": "claude",
                    "run_id": "20260710T130000Z",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(guard, "ROOT", tmp_path)

    scope = guard._hydrate_orchestrator_scope(
        {"mode": "orchestrator", "orchestration_id": oid}
    )

    assert {"track": "claude", "run_id": "20260710T130000Z"} in scope["child_runs"]
    command = (
        "autoresearch --track claude --run-id 20260710T130000Z "
        "record-decision cmp --decision reject"
    )
    assert guard.decide(scope, [command])[0]


def test_orchestrator_denies_implicit_latest_when_another_campaign_is_newer():
    scope = {
        **ORCHESTRATOR,
        "latest_orchestration_id": "20260710T130000Z",
    }
    allow, reason = guard.decide(scope, ["autoresearch orchestrate status"])

    assert not allow
    assert "would resolve to 20260710T130000Z" in reason


def test_command_with_embedded_foreign_path_denied():
    cmd = "cat artifacts/tracks/codex/runs/RUN_X/RESEARCH_LOG.md | head"
    assert not guard.decide(BOUND, [cmd])[0]


def test_bound_session_denies_autoresearch_foreign_run_id():
    cmd = "autoresearch --track claude --run-id 20260604T081500Z start-session main"
    allow, reason = guard.decide(BOUND, [cmd])

    assert not allow
    assert "targets run claude/20260604T081500Z" in reason


def test_bound_session_denies_autoresearch_new_run():
    cmd = "autoresearch --track claude --new-run bootstrap-track --model-provider openai --model-name codex"
    allow, reason = guard.decide(BOUND, [cmd])

    assert not allow
    assert "--new-run" in reason


def test_bound_session_denies_autoresearch_other_track():
    cmd = "autoresearch --track codex --run-id RUN_A list-experiments"
    allow, reason = guard.decide(BOUND, [cmd])

    assert not allow
    assert "targets track" in reason


def test_unbound_research_refuses_default_track_binding():
    cmd = "autoresearch --track default start-session main"
    allow, reason = guard.decide(None, [cmd])

    assert not allow
    assert "only bind to one of" in reason
    assert "default" in reason


def test_unbound_research_refuses_custom_track_binding():
    cmd = "autoresearch --track experiment_x --new-run bootstrap-track --model-provider openai --model-name codex"
    allow, reason = guard.decide(None, [cmd])

    assert not allow
    assert "only bind to one of" in reason
    assert "experiment_x" in reason


def test_analyst_can_use_custom_track_binding_commands():
    cmd = "autoresearch --track experiment_x --new-run bootstrap-track --model-provider openai --model-name codex"

    assert guard.decide(ANALYST, [cmd])[0]


def test_unbound_research_refuses_non_timestamp_agent_run_id():
    cmd = "autoresearch --track codex --run-id CodexTimeX start-session main"
    allow, reason = guard.decide(None, [cmd])

    assert not allow
    assert "UTC timestamps" in reason
    assert "CodexTimeX" in reason


def test_bound_session_denies_autoresearch_implicit_foreign_latest(monkeypatch):
    monkeypatch.setattr(guard, "_latest_run_id", lambda track: "RUN_B")
    cmd = "autoresearch --track claude list-experiments"
    allow, reason = guard.decide(BOUND, [cmd])

    assert not allow
    assert "without --run-id would resolve" in reason


# ── path extraction (target vs content) ─────────────────────────────────────

def test_extract_paths_scans_targets_not_content():
    assert guard.extract_paths("Bash", {"command": f"cat {FOREIGN_RUN}"}) == [f"cat {FOREIGN_RUN}"]
    assert guard.extract_paths("Read", {"file_path": OWN}) == [OWN]
    # A file that merely *mentions* a run path in its content is not "accessing" it.
    assert guard.extract_paths("Write", {"file_path": "tests/t.py", "content": FOREIGN_RUN}) == ["tests/t.py"]
    assert guard.extract_paths("Edit", {"file_path": "src/x.py", "new_string": FOREIGN_RUN}) == ["src/x.py"]


def test_write_with_run_path_in_content_allowed():
    # Regression: writing a file whose *content* references runs must not be blocked.
    paths = guard.extract_paths("Write", {"file_path": "tests/fixtures.py", "content": FOREIGN_RUN})
    assert guard.decide(BOUND, paths)[0]


# ── binding parsers ─────────────────────────────────────────────────────────

def test_autoresearch_flag_parsing():
    cmd = "autoresearch --track claude --run-id RUN_A run-session-cycles 1"
    assert guard._TRACK_FLAG.search(cmd).group(1) == "claude"
    assert guard._RUNID_FLAG.search(cmd).group(1) == "RUN_A"


def test_binder_fires_only_on_real_invocation():
    # Real invocations (bare, venv path, after env-assign / cd) bind.
    assert guard.command_invokes_autoresearch("autoresearch --track codex list-experiments")
    assert guard.command_invokes_autoresearch(".venv/bin/autoresearch --track codex bootstrap-track")
    assert guard.command_invokes_autoresearch("python3 -m src.autoresearch.cli --track opencode --new-run bootstrap-track")
    assert guard.command_invokes_autoresearch("python -m autoresearch.cli --track opencode start-session main")
    assert guard.command_invokes_autoresearch("cd /repo && autoresearch --track claude start-session main")
    assert guard.command_invokes_autoresearch("AUTORESEARCH_X=1 autoresearch --track claude list-experiments")
    # Mere *mentions* must NOT bind (the footgun that bound the dev session).
    assert not guard.command_invokes_autoresearch('echo "run: autoresearch --track claude --run-id RUN_A"')
    assert not guard.command_invokes_autoresearch("grep autoresearch docs/*.md")
    assert not guard.command_invokes_autoresearch("cat notes_about_autoresearch.txt")


# ── cross-harness: tool-name normalisation & arg shapes ─────────────────────

def test_opencode_lowercase_and_camelcase_args():
    # OpenCode tool names are lower-case and reads carry `filePath`.
    assert guard.extract_paths("read", {"filePath": OWN}) == [OWN]
    assert guard.extract_paths("bash", {"command": f"cat {FOREIGN_RUN}"}) == [f"cat {FOREIGN_RUN}"]


def test_opencode_broad_glob_maps_to_run_enumeration():
    paths = guard.extract_paths(
        "glob",
        {"path": str(Path(__file__).resolve().parents[1]), "pattern": "**/*"},
    )

    assert "artifacts/tracks/*/runs" in paths
    allow, reason = guard.decide(None, paths)
    assert not allow
    assert "run not bound yet" in reason


def test_python_module_autoresearch_allows_only_named_agent_tracks():
    cmd = "python3 -m src.autoresearch.cli --track set3 --new-run bootstrap-track --model-provider manual --model-name user"
    allow, reason = guard.decide(None, [cmd])

    assert not allow
    assert "set3" in reason


def test_codex_apply_patch_scans_command():
    # Codex edits arrive as apply_patch with the path in the patch body.
    patch = "*** Begin Patch\n*** Update File: artifacts/tracks/codex/runs/RUN_X/model.py\n*** End Patch"
    paths = guard.extract_paths("apply_patch", {"command": patch})
    assert paths == ["artifacts/tracks/codex/runs/RUN_X/model.py"]
    assert not guard.decide(BOUND, paths)[0]  # foreign run -> denied


def test_codex_apply_patch_ignores_body_mentions():
    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Update File: tests/test_run_scope_guard.py",
            "@@",
            "+mentioned = 'artifacts/tracks/codex/runs/RUN_X/model.py'",
            "*** End Patch",
        ]
    )
    paths = guard.extract_paths("apply_patch", {"command": patch})

    assert paths == ["tests/test_run_scope_guard.py"]
    assert guard.decide(None, paths)[0]


def test_env_analyst_fallback(monkeypatch):
    # With AUTORESEARCH_SCOPE=analyst and no scope file, the session is analyst.
    monkeypatch.setenv("AUTORESEARCH_SCOPE", "analyst")
    scope = guard.resolve_scope("session-with-no-file")
    assert scope and scope.get("mode") == "analyst"
    assert guard.decide(scope, [FOREIGN_RUN])[0]


def test_env_unset_is_unbound(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)
    assert guard.resolve_scope("another-session-no-file") is None


def test_session_start_logs_hook_active(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)

    assert guard.handle_session_start({"session_id": "codex-session", "source": "startup"}) == 0

    assert "hook active (SessionStart source=startup)" in (tmp_path / "guard.log").read_text(encoding="utf-8")


def test_session_start_binds_console_assigned_research_run(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.setenv("AUTORESEARCH_SCOPE", "research")
    monkeypatch.setenv("AUTORESEARCH_TRACK", "codex")
    monkeypatch.setenv("AUTORESEARCH_RUN_ID", "20260610T072525Z")

    assert guard.handle_session_start({"session_id": "console-run", "source": "startup"}) == 0

    scope = guard.resolve_scope("console-run")
    assert scope["track"] == "codex"
    assert scope["run_id"] == "20260610T072525Z"


def test_failed_autoresearch_command_does_not_bind(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.setattr(guard, "_latest_run_id", lambda track: "RUN_A")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)

    payload = {
        "session_id": "failed-bind",
        "status": "completed",
        "tool_input": {"command": "autoresearch --track claude start-session --new-run"},
        "tool_response": {"exit_code": 2},
    }

    assert guard.handle_post_tool_use(payload) == 0
    assert guard.resolve_scope("failed-bind") is None


def test_env_analyst_without_scope_file_is_never_demoted_by_auto_bind(monkeypatch, tmp_path):
    # Harnesses without a SessionStart event honour analyst mode via env only;
    # a bindable command must not silently rebind such a session.
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.setattr(guard, "_latest_run_id", lambda track: "20260610T063646Z")
    monkeypatch.setenv("AUTORESEARCH_SCOPE", "analyst")

    for command in (
        "autoresearch --track claude --run-id 20260610T063646Z start-session review",
        "autoresearch orchestrate new --dataset french_motor --total-cycles 2",
    ):
        payload = {"session_id": "env-analyst", "tool_input": {"command": command}}
        assert guard.handle_post_tool_use(payload) == 0

    assert guard.resolve_scope("env-analyst") == {"mode": "analyst", "source": "env"}


def test_ambiguous_new_run_does_not_bind_stale_latest(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.setattr(guard, "_latest_run_id", lambda track: "20260610T063646Z")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)

    payload = {
        "session_id": "ambiguous-new-run",
        "tool_input": {
            "command": (
                "autoresearch --track codex --new-run bootstrap-track "
                "--model-provider openai --model-name codex-mini-latest"
            )
        },
    }

    assert guard.handle_post_tool_use(payload) == 0
    assert guard.resolve_scope("ambiguous-new-run") is None


def test_new_run_binds_run_id_reported_by_bootstrap_output(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)

    payload = {
        "session_id": "reported-new-run",
        "tool_input": {
            "command": (
                "autoresearch --track codex --new-run bootstrap-track "
                "--model-provider openai --model-name codex-mini-latest"
            )
        },
        "tool_response": {
            "status": "completed",
            "output": '{"track": "codex", "run_id": "20260610T072525Z"}',
        },
    }

    assert guard.handle_post_tool_use(payload) == 0
    assert guard.resolve_scope("reported-new-run")["run_id"] == "20260610T072525Z"


def test_invalid_start_session_shape_does_not_bind(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.setattr(guard, "_latest_run_id", lambda track: "RUN_A")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)

    payload = {
        "session_id": "invalid-bind",
        "tool_input": {"command": "autoresearch --track claude start-session --new-run"},
    }

    assert guard.handle_post_tool_use(payload) == 0
    assert guard.resolve_scope("invalid-bind") is None


def test_orchestrate_new_auto_binds_from_success_output(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)
    payload = {
        "session_id": "orchestrator-new",
        "tool_input": {
            "command": "autoresearch orchestrate new --dataset french_motor --total-cycles 6"
        },
        "tool_response": {
            "exit_code": 0,
            "output": '{"orchestration_id": "20260710T120000Z"}',
        },
    }

    assert guard.handle_post_tool_use(payload) == 0
    scope = guard.load_scope_file("orchestrator-new")
    assert scope["mode"] == "orchestrator"
    assert scope["orchestration_id"] == "20260710T120000Z"


def test_orchestrate_spawn_auto_binds_explicit_orchestration(monkeypatch, tmp_path):
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path)
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "guard.log")
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)
    payload = {
        "session_id": "orchestrator-spawn",
        "tool_input": {
            "command": (
                "autoresearch orchestrate spawn --orchestration-id 20260710T120000Z "
                "--brief b.json --backend stub --no-wait"
            )
        },
        "tool_response": {"exit_code": 0},
    }

    assert guard.handle_post_tool_use(payload) == 0
    assert guard.load_scope_file("orchestrator-spawn")["mode"] == "orchestrator"


def test_codex_hooks_json_uses_codex_schema_and_wires_guards():
    hooks_path = Path(__file__).resolve().parents[1] / ".codex" / "hooks.json"
    payload = json.loads(hooks_path.read_text(encoding="utf-8"))

    assert sorted(payload) == ["hooks"]
    hooks = payload["hooks"]
    assert {"SessionStart", "PreToolUse", "PostToolUse", "Stop"} <= set(hooks)

    def commands(event: str) -> list[str]:
        return [
            hook["command"]
            for group in hooks[event]
            for hook in group["hooks"]
            if hook.get("type") == "command"
        ]

    assert any("scripts/run_scope_guard.py" in command for command in commands("SessionStart"))
    assert any("scripts/run_scope_guard.py" in command for command in commands("PreToolUse"))
    assert any("scripts/run_scope_guard.py" in command for command in commands("PostToolUse"))
    assert not any("research_log_guard" in command for command in commands("Stop"))
    assert any("scripts/import_agent_telemetry.py" in command for command in commands("SessionStart"))
    assert any("scripts/import_agent_telemetry.py" in command for command in commands("Stop"))


def test_retired_codex_research_log_hook_has_compatibility_shim():
    shim = Path(__file__).resolve().parents[1] / "scripts" / "codex_research_log_guard.py"

    assert shim.exists()
    assert "Compatibility no-op" in shim.read_text(encoding="utf-8")


def test_claude_hooks_wire_desktop_telemetry():
    hooks_path = Path(__file__).resolve().parents[1] / ".claude" / "settings.json"
    hooks = json.loads(hooks_path.read_text(encoding="utf-8"))["hooks"]

    assert {"SessionStart", "PreToolUse", "PostToolUse", "Stop"} <= set(hooks)
    commands = [
        hook["command"]
        for event in ("SessionStart", "Stop")
        for group in hooks[event]
        for hook in group["hooks"]
        if hook.get("type") == "command"
    ]
    assert any("--surface claude" in command for command in commands)
