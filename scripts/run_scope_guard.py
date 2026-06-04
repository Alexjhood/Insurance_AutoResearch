#!/usr/bin/env python3
"""Run-scope guard — keeps a bound research session inside its own run folder.

One shared "brain" for three agent harnesses (Claude Code, Codex, OpenCode).
The autoresearch framework is already run-isolated at the data layer: every
command scopes its artifacts under ``artifacts/tracks/<track>/runs/<run-id>/``.
The only contamination channel left is the *agent's own* free-form file access
(shell `cat`/`grep`, file reads/edits) reaching into a sibling run's folder.
This guard closes that channel at the harness layer — a denied tool call cannot
be overridden by the model.

Policy (default-analyst):
  * Only a *bound research* session is confined to its own run.
  * An *unbound* session (before its first ``autoresearch`` command) and an
    *analyst* session both see everything. A research agent must therefore
    bootstrap before inspecting artifacts (enforced by AGENT.md convention).

Events (dispatched on ``hook_event_name``; the JS adapter maps OpenCode's
before/after hooks onto PreToolUse/PostToolUse):
  * SessionStart  — record analyst/research scope from env, if requested.
  * PostToolUse   — when the agent runs ``autoresearch --track <T> ...``, bind
    this session to the one run that command targets.
  * PreToolUse    — for a bound research session, deny access to a foreign run
    folder or to the ``runs/`` listing; allow everything else.

Scope is keyed on the harness session id, so parallel runs in separate threads
each get their own scope with no global ambiguity. Analyst mode is also honoured
via ``AUTORESEARCH_SCOPE=analyst`` in the environment (the hook subprocess
inherits the harness's launch environment), so it works on all three harnesses
even without a SessionStart event.

Fail-open by design: any unexpected error allows the call (a guard bug must
never block legitimate research) and is noted in the guard log.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


def _find_root(start: Path) -> Path:
    """Walk up until we find the repo root (has pyproject.toml or .git)."""
    p = start
    for _ in range(8):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start


ROOT = _find_root(Path(__file__).resolve().parent)
TRACKS_DIR = ROOT / "artifacts" / "tracks"
SCOPE_DIR = TRACKS_DIR / ".scope"
LOG_PATH = SCOPE_DIR / "guard.log"

# Matches a reference to a run folder anywhere in a string:
#   artifacts/tracks/<track>/runs            -> (track, None)   "runs dir" listing
#   artifacts/tracks/<track>/runs/<run-id>   -> (track, run)    a specific run
_RUN_REF = re.compile(
    r"artifacts/tracks/([^/\s'\";:|&]+)/runs(?:/([^/\s'\";:|&]+))?"
)
_TRACK_FLAG = re.compile(r"--track[\s=]+([A-Za-z0-9_\-]+)")
_RUNID_FLAG = re.compile(r"--run-id[\s=]+([A-Za-z0-9_\-]+)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_GLOBAL_VALUE_FLAGS = {"--config", "--track", "--run-id", "--target-mode"}
_GLOBAL_BOOL_FLAGS = {"--new-run"}
ALLOWED_RESEARCH_TRACKS = {"codex", "claude", "opencode"}
RUN_ID_TIMESTAMP = re.compile(r"^\d{8}T\d{6}Z$")


def command_invokes_autoresearch(command: str) -> bool:
    """True only if ``autoresearch`` is actually *run* as the executable.

    Guards against false binding when a command merely *mentions* the string
    (e.g. inside an ``echo``/heredoc/grep). We split on shell separators and, for
    each segment, skip leading ``VAR=val`` env assignments and check whether the
    first real token's basename is ``autoresearch``.
    """
    if not isinstance(command, str):
        return False
    for segment in re.split(r"[;&|]+|&&|\|\|", command):
        tokens = segment.strip().split()
        idx = 0
        while idx < len(tokens) and _ENV_ASSIGN.match(tokens[idx]):
            idx += 1
        if idx < len(tokens) and tokens[idx].split("/")[-1] == "autoresearch":
            return True
    return False


def _shell_segments(command: str) -> list[str]:
    if not isinstance(command, str):
        return []
    return [segment.strip() for segment in re.split(r"[;&|]+|&&|\|\|", command) if segment.strip()]


def _autoresearch_tokens(command: str) -> list[list[str]]:
    """Return tokenized shell segments whose executable is ``autoresearch``."""
    invocations: list[list[str]] = []
    for segment in _shell_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        idx = 0
        while idx < len(tokens) and _ENV_ASSIGN.match(tokens[idx]):
            idx += 1
        if idx < len(tokens) and tokens[idx].split("/")[-1] == "autoresearch":
            invocations.append(tokens[idx:])
    return invocations


def _flag_value(tokens: list[str], flag: str) -> str:
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        prefix = flag + "="
        if token.startswith(prefix):
            return token[len(prefix):]
    return ""


def _has_flag(tokens: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(flag + "=") for token in tokens)


def _autoresearch_subcommand(tokens: list[str]) -> str:
    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in _GLOBAL_BOOL_FLAGS:
            idx += 1
            continue
        if token in _GLOBAL_VALUE_FLAGS:
            idx += 2
            continue
        if any(token.startswith(flag + "=") for flag in _GLOBAL_VALUE_FLAGS):
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return token
    return ""


def _has_start_session_name(tokens: list[str]) -> bool:
    try:
        start_idx = tokens.index("start-session")
    except ValueError:
        return False
    idx = start_idx + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"--model-provider", "--model-name", "--model-version", "--harness", "--max-cycles"}:
            idx += 2
            continue
        if any(token.startswith(flag + "=") for flag in {"--model-provider", "--model-name", "--model-version", "--harness", "--max-cycles"}):
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return True
    return False


def _post_tool_succeeded(payload: dict) -> bool:
    """Best-effort success check for harness payload variants."""
    candidates = [
        payload,
        payload.get("tool_response") or {},
        payload.get("tool_output") or {},
        payload.get("result") or {},
    ]
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        for key in ("exit_code", "exitCode", "status_code", "statusCode", "returncode"):
            if key in obj:
                try:
                    return int(obj[key]) == 0
                except (TypeError, ValueError):
                    return False
        status = str(obj.get("status", "")).lower()
        if status in {"success", "succeeded", "ok", "completed"}:
            return True
        if status in {"error", "failed", "failure"}:
            return False
    return True


def _bindable_autoresearch_tokens(command: str) -> list[list[str]]:
    out: list[list[str]] = []
    for tokens in _autoresearch_tokens(command):
        subcommand = _autoresearch_subcommand(tokens)
        if subcommand == "bootstrap-track":
            out.append(tokens)
        elif subcommand == "start-session" and not _has_flag(tokens, "--new-run") and _has_start_session_name(tokens):
            out.append(tokens)
    return out


def find_autoresearch_scope_violations(scope: dict | None, texts: list[str]) -> list[str]:
    """Return semantic run-scope violations for ``autoresearch`` commands.

    Path scanning catches shell access such as ``cat artifacts/.../runs/R``.
    This catches the equivalent framework access, e.g. ``autoresearch --run-id R``.
    """
    scope = scope or {}
    if scope.get("mode") != "research":
        return []

    bound_track = str(scope.get("track") or "")
    bound_run = str(scope.get("run_id") or "")
    if not bound_track or not bound_run:
        return []

    violations: list[str] = []
    for text in texts:
        for tokens in _autoresearch_tokens(text):
            track = _flag_value(tokens, "--track")
            run_id = _flag_value(tokens, "--run-id")
            new_run = _has_flag(tokens, "--new-run")
            subcommand = _autoresearch_subcommand(tokens)

            if track and track != bound_track:
                violations.append(
                    f"this research run is scoped to {bound_track}/{bound_run}; "
                    f"autoresearch command targets track {track!r}."
                )
                continue
            if track and new_run:
                violations.append(
                    f"this research run is already scoped to {bound_track}/{bound_run}; "
                    "`autoresearch --new-run` would create or switch to a sibling run."
                )
                continue
            if track and run_id and run_id != bound_run:
                violations.append(
                    f"this research run is scoped to {bound_track}/{bound_run}; "
                    f"autoresearch command targets run {track}/{run_id}."
                )
                continue
            if track and not run_id:
                latest = _latest_run_id(track)
                if latest and latest != bound_run:
                    violations.append(
                        f"this research run is scoped to {bound_track}/{bound_run}; "
                        f"autoresearch command without --run-id would resolve to {track}/{latest}."
                    )
    return violations

# Which tool_input fields denote an access *target* (a path), keyed by the
# lower-cased tool name. We deliberately do NOT scan file *content*
# (Write.content, Edit.new_string, the apply_patch body beyond its header, a
# Grep regex): a file that merely *mentions* a run path is not *accessing* it.
# camelCase variants (filePath) cover OpenCode; snake_case covers Claude/Codex.
_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "apply_patch": ("command",),          # Codex edits; path lives in patch header
    "read": ("file_path", "filePath"),
    "edit": ("file_path", "filePath"),
    "write": ("file_path", "filePath"),
    "multiedit": ("file_path", "filePath"),
    "notebookedit": ("notebook_path",),
    "glob": ("path", "pattern"),          # pattern may carry the target directory
    "grep": ("path", "glob"),             # the regex itself is content, not a path
}
_DEFAULT_PATH_FIELDS = ("file_path", "filePath", "path", "notebook_path", "command")


# ── helpers ────────────────────────────────────────────────────────────────

def _log(message: str) -> None:
    try:
        SCOPE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {message}\n")
    except Exception:
        pass


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)[:120] or "session"


def _scope_path(session_id: str) -> Path:
    return SCOPE_DIR / f"{_safe(session_id)}.json"


def load_scope_file(session_id: str) -> dict | None:
    path = _scope_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_scope(session_id: str) -> dict | None:
    """Scope for a session: bound file wins; else env analyst; else unbound."""
    scope = load_scope_file(session_id)
    if scope is not None:
        return scope
    if os.environ.get("AUTORESEARCH_SCOPE", "").strip().lower() == "analyst":
        return {"mode": "analyst", "source": "env"}
    return None


def write_scope(session_id: str, scope: dict) -> None:
    SCOPE_DIR.mkdir(parents=True, exist_ok=True)
    scope = dict(scope)
    scope.setdefault("bound_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _scope_path(session_id).write_text(
        json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def extract_paths(tool_name: str, tool_input: dict) -> list[str]:
    """Return the path-bearing strings a tool call would actually touch."""
    fields = _PATH_FIELDS.get((tool_name or "").lower(), _DEFAULT_PATH_FIELDS)
    out: list[str] = []
    for field in fields:
        value = tool_input.get(field)
        if isinstance(value, str) and value:
            out.append(value)
    return out


def find_run_refs(text: str) -> list[tuple[str, str | None]]:
    """Return (track, run_id-or-None) for every run reference in *text*."""
    return [(m.group(1), m.group(2)) for m in _RUN_REF.finditer(text)]


# ── decision (pure, unit-tested) ────────────────────────────────────────────

def decide(scope: dict | None, texts: list[str]) -> tuple[bool, str]:
    """Decide whether a tool call touching *texts* is allowed.

    Only a *bound research* session is confined to its own run. An unbound
    session (before its first ``autoresearch`` command) and an analyst session
    both see everything — so a research agent must bootstrap before it inspects
    any artifacts (that ordering is enforced by AGENT.md convention, not here).

    Returns (allow, reason). ``reason`` is only meaningful when denied.
    """
    scope = scope or {}
    if scope.get("mode") != "analyst":
        for text in texts:
            for tokens in _bindable_autoresearch_tokens(text):
                track = _flag_value(tokens, "--track")
                if track and track not in ALLOWED_RESEARCH_TRACKS:
                    return (
                        False,
                        "research sessions may only bind to one of "
                        f"{sorted(ALLOWED_RESEARCH_TRACKS)}; got track {track!r}. "
                        "Use `codex`, `claude`, or `opencode`, or relaunch in analyst mode for admin work.",
                    )
                run_id = _flag_value(tokens, "--run-id")
                if track in ALLOWED_RESEARCH_TRACKS and run_id and not RUN_ID_TIMESTAMP.fullmatch(run_id):
                    return (
                        False,
                        f"research run ids must be UTC timestamps in YYYYMMDDTHHMMSSZ form; got {run_id!r}. "
                        "Use `--new-run` to create one, or omit `--run-id` to continue the latest run.",
                    )

    if scope.get("mode") != "research":
        return True, ""  # unbound or analyst -> unrestricted

    bound_track = scope.get("track")
    bound_run = scope.get("run_id")
    if not bound_run:
        return True, ""  # defensive: a research scope must name a run

    semantic_violations = find_autoresearch_scope_violations(scope, texts)
    if semantic_violations:
        return False, semantic_violations[0]

    refs: list[tuple[str, str | None]] = []
    for text in texts:
        refs.extend(find_run_refs(text))

    for track, run in refs:
        if run is None:
            return (
                False,
                f"enumerating sibling runs under artifacts/tracks/{track}/runs is "
                "not allowed — this run is scoped to its own folder only.",
            )
        if track == bound_track and run == bound_run:
            continue
        return (
            False,
            f"this research run is scoped to {bound_track}/{bound_run}; the path "
            f"references a different run ({track}/{run}). Knowledge of other runs "
            "reaches you only through memory. "
            "(For deliberate cross-run analysis, relaunch with AUTORESEARCH_SCOPE=analyst.)",
        )
    return True, ""


# ── event handlers ──────────────────────────────────────────────────────────

def handle_session_start(payload: dict) -> int:
    session_id = payload.get("session_id") or ""
    if not session_id:
        return 0
    source = payload.get("source") or "unknown"
    _log(f"session {session_id}: hook active (SessionStart source={source})")
    requested = os.environ.get("AUTORESEARCH_SCOPE", "").strip().lower()
    if requested == "analyst":
        write_scope(session_id, {"mode": "analyst", "source": "env"})
        _log(f"session {session_id}: analyst scope (env)")
    elif requested == "research":
        track = os.environ.get("AUTORESEARCH_TRACK", "").strip()
        run_id = os.environ.get("AUTORESEARCH_RUN_ID", "").strip() or _latest_run_id(track)
        if track and run_id:
            write_scope(session_id, _research_scope(track, run_id, source="env"))
            _log(f"session {session_id}: research scope {track}/{run_id} (env)")
    return 0


def handle_post_tool_use(payload: dict) -> int:
    """Auto-bind a research session to the run its autoresearch command targets."""
    session_id = payload.get("session_id") or ""
    if not session_id:
        return 0
    if resolve_scope(session_id) is not None:  # already analyst or bound
        return 0
    if not _post_tool_succeeded(payload):
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    invocations = _bindable_autoresearch_tokens(command)
    if not invocations:
        return 0
    track = _flag_value(invocations[0], "--track")
    if not track:
        return 0
    if track not in ALLOWED_RESEARCH_TRACKS:
        _log(
            f"session {session_id}: refusing research scope for non-agent track "
            f"{track!r}; allowed={sorted(ALLOWED_RESEARCH_TRACKS)}"
        )
        return 0
    run_id = _flag_value(invocations[0], "--run-id") or _latest_run_id(track)
    if not run_id:
        return 0
    if not RUN_ID_TIMESTAMP.fullmatch(run_id):
        _log(f"session {session_id}: refusing research scope for non-timestamp run_id {track}/{run_id}")
        return 0
    write_scope(session_id, _research_scope(track, run_id, source="auto"))
    _log(f"session {session_id}: research scope {track}/{run_id} (auto)")
    return 0


def handle_pre_tool_use(payload: dict) -> int:
    session_id = payload.get("session_id") or ""
    scope = resolve_scope(session_id)
    texts = extract_paths(payload.get("tool_name", ""), payload.get("tool_input") or {})
    allow, reason = decide(scope, texts)
    if allow:
        return 0
    sys.stderr.write(f"[run-scope-guard] Denied: {reason}\n")
    _log(f"session {session_id}: DENY {payload.get('tool_name')} :: {reason}")
    return 2  # exit 2 == block the tool call, stderr shown to the model


# ── small utilities ─────────────────────────────────────────────────────────

def _research_scope(track: str, run_id: str, *, source: str) -> dict:
    return {
        "mode": "research",
        "track": track,
        "run_id": run_id,
        "run_dir": str(TRACKS_DIR / track / "runs" / run_id),
        "source": source,
    }


def _latest_run_id(track: str) -> str:
    if not track:
        return ""
    latest = TRACKS_DIR / track / "latest_run.json"
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        return str(payload.get("run_id", "")).strip()
    except Exception:
        return ""


_HANDLERS = {
    "SessionStart": handle_session_start,
    "PreToolUse": handle_pre_tool_use,
    "PostToolUse": handle_post_tool_use,
}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0  # fail open
    handler = _HANDLERS.get(payload.get("hook_event_name", ""))
    if handler is None:
        return 0
    try:
        return handler(payload)
    except Exception as exc:  # fail open — never block research on a guard bug
        _log(f"ERROR in {payload.get('hook_event_name')}: {exc!r}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
