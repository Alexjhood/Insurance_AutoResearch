#!/usr/bin/env python3
"""Run-scope guard — keeps a bound research session inside its own run folder.

One shared "brain" for three agent harnesses (Claude Code, Codex, OpenCode).
The autoresearch framework is already run-isolated at the data layer: every
command scopes its artifacts under ``artifacts/tracks/<track>/runs/<run-id>/``.
The only contamination channel left is the *agent's own* free-form file access
(shell `cat`/`grep`, file reads/edits) reaching into a sibling run's folder.
This guard closes that channel at the harness layer — a denied tool call cannot
be overridden by the model.

Policy (research by default for run artifacts):
  * Only a *bound research* session may inspect a run folder, and only its own.
  * An *unbound* session (before its first ``autoresearch`` command) may inspect
    source/docs/configs but not run folders. A research agent must bootstrap
    before inspecting artifacts.
  * An *analyst* session sees everything.
  * An *orchestrator* session sees its own orchestration folder and only the
    child runs listed in that orchestration's manifest.

Events (dispatched on ``hook_event_name``; the JS adapter maps OpenCode's
before/after hooks onto PreToolUse/PostToolUse):
  * SessionStart  — record analyst/research scope from env, if requested.
  * PostToolUse   — when the agent runs ``autoresearch --track <T> ...``, bind
    this session to the one run that command targets.
  * PreToolUse    — deny run-folder access before binding; after binding, deny
    access to a foreign run folder or to the ``runs/`` listing; allow everything
    else.

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
_ORCHESTRATION_REF = re.compile(
    r"artifacts/orchestrations(?:/([^/\s'\";:|&]+))?"
)
_TRACK_FLAG = re.compile(r"--track[\s=]+([A-Za-z0-9_\-]+)")
_RUNID_FLAG = re.compile(r"--run-id[\s=]+([A-Za-z0-9_\-]+)")
_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_GLOBAL_VALUE_FLAGS = {"--config", "--track", "--run-id", "--target-mode"}
_GLOBAL_BOOL_FLAGS = {"--new-run"}
ALLOWED_RESEARCH_TRACKS = {"codex", "claude", "opencode"}
RUN_ID_TIMESTAMP = re.compile(r"^\d{8}T\d{6}Z$")
_PYTHON_BIN = re.compile(r"^(?:python|python\d+(?:\.\d+)?)$")
_AUTORESEARCH_MODULES = {"autoresearch.cli", "src.autoresearch.cli"}


def command_invokes_autoresearch(command: str) -> bool:
    """True only if ``autoresearch`` is actually *run* as the executable.

    Guards against false binding when a command merely *mentions* the string
    (e.g. inside an ``echo``/heredoc/grep). We split on shell separators and, for
    each segment, skip leading ``VAR=val`` env assignments and check whether the
    first real token invokes the CLI, either as an ``autoresearch`` executable or
    as ``python -m autoresearch.cli`` / ``python -m src.autoresearch.cli``.
    """
    if not isinstance(command, str):
        return False
    for segment in _shell_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        if _normalise_autoresearch_tokens(tokens):
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
        normalised = _normalise_autoresearch_tokens(tokens)
        if normalised:
            invocations.append(normalised)
    return invocations


def _normalise_autoresearch_tokens(tokens: list[str]) -> list[str]:
    """Return tokens in ``autoresearch ...`` shape for supported CLI invocations."""
    idx = 0
    while idx < len(tokens) and _ENV_ASSIGN.match(tokens[idx]):
        idx += 1
    if idx < len(tokens) and tokens[idx].split("/")[-1] == "autoresearch":
        return tokens[idx:]
    if idx + 2 < len(tokens):
        executable = tokens[idx].split("/")[-1]
        if (
            _PYTHON_BIN.fullmatch(executable)
            and tokens[idx + 1] == "-m"
            and tokens[idx + 2] in _AUTORESEARCH_MODULES
        ):
            return ["autoresearch", *tokens[idx + 3:]]
    return []


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


def _post_tool_succeeded(payload: dict) -> bool | None:
    """Return explicit tool success/failure, or None when the payload is silent."""
    candidates = [
        payload.get("tool_response") or {},
        payload.get("tool_output") or {},
        payload.get("result") or {},
        payload,
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
    return None


def _run_id_from_post_payload(payload: dict) -> str:
    """Extract a timestamp run id from bootstrap output, if present."""

    pattern = re.compile(r"""["']?run_id["']?\s*[:=]\s*["']?(\d{8}T\d{6}Z)""")

    def _search(value: object) -> str:
        if isinstance(value, dict):
            direct = str(value.get("run_id") or "")
            if RUN_ID_TIMESTAMP.fullmatch(direct):
                return direct
            for child in value.values():
                found = _search(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = _search(child)
                if found:
                    return found
        elif isinstance(value, str):
            match = pattern.search(value)
            if match:
                return match.group(1)
        return ""

    return _search(payload)


def _bindable_autoresearch_tokens(command: str) -> list[list[str]]:
    out: list[list[str]] = []
    for tokens in _autoresearch_tokens(command):
        subcommand = _autoresearch_subcommand(tokens)
        if subcommand == "bootstrap-track":
            out.append(tokens)
        elif subcommand == "start-session" and not _has_flag(tokens, "--new-run") and _has_start_session_name(tokens):
            out.append(tokens)
    return out


def _orchestrate_action(tokens: list[str]) -> str:
    try:
        index = tokens.index("orchestrate")
    except ValueError:
        return ""
    return tokens[index + 1] if index + 1 < len(tokens) else ""


def _orchestrator_bindable_tokens(command: str) -> list[list[str]]:
    return [
        tokens
        for tokens in _autoresearch_tokens(command)
        if _autoresearch_subcommand(tokens) == "orchestrate"
        and _orchestrate_action(tokens) in {"new", "spawn"}
        and not _has_flag(tokens, "--dry-run")
    ]


def _orchestration_id_from_post_payload(payload: dict) -> str:
    pattern = re.compile(
        r'''["']?(?:orchestration_id|orchestration)["']?\s*[:=]?\s*["']?'''
        r"(\d{8}T\d{6}Z)"
    )

    def _search(value: object) -> str:
        if isinstance(value, dict):
            direct = str(value.get("orchestration_id") or "")
            if RUN_ID_TIMESTAMP.fullmatch(direct):
                return direct
            for child in value.values():
                found = _search(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = _search(child)
                if found:
                    return found
        elif isinstance(value, str):
            match = pattern.search(value)
            if match:
                return match.group(1)
        return ""

    return _search(payload)


def find_autoresearch_scope_violations(scope: dict | None, texts: list[str]) -> list[str]:
    """Return semantic run-scope violations for ``autoresearch`` commands.

    Path scanning catches shell access such as ``cat artifacts/.../runs/R``.
    This catches the equivalent framework access, e.g. ``autoresearch --run-id R``.
    """
    scope = scope or {}
    if scope.get("mode") not in {"research", "orchestrator"}:
        return []

    if scope.get("mode") == "orchestrator":
        return _find_orchestrator_command_violations(scope, texts)

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


def _find_orchestrator_command_violations(scope: dict, texts: list[str]) -> list[str]:
    orchestration_id = str(scope.get("orchestration_id") or "")
    children = {
        (str(child.get("track") or ""), str(child.get("run_id") or ""))
        for child in scope.get("child_runs") or ()
        if isinstance(child, dict)
    }
    violations: list[str] = []
    for text in texts:
        for tokens in _autoresearch_tokens(text):
            subcommand = _autoresearch_subcommand(tokens)
            if subcommand == "orchestrate":
                action = _orchestrate_action(tokens)
                target_oid = _flag_value(tokens, "--orchestration-id")
                if action == "new":
                    violations.append(
                        f"this session is already scoped to orchestration {orchestration_id}; "
                        "`orchestrate new` would create another orchestration."
                    )
                elif target_oid and target_oid != orchestration_id:
                    violations.append(
                        f"this session is scoped to orchestration {orchestration_id}; "
                        f"the command targets orchestration {target_oid}."
                    )
                elif not target_oid and action not in {"list-backends", "finish-delegation"}:
                    latest = str(scope.get("latest_orchestration_id") or "")
                    if latest and latest != orchestration_id:
                        violations.append(
                            f"this session is scoped to orchestration {orchestration_id}; "
                            f"an omitted --orchestration-id would resolve to {latest}."
                        )
                continue

            track = _flag_value(tokens, "--track")
            run_id = _flag_value(tokens, "--run-id")
            if _has_flag(tokens, "--new-run"):
                violations.append(
                    f"this session is scoped to orchestration {orchestration_id}; "
                    "`autoresearch --new-run` would create an unlisted run."
                )
            elif run_id:
                matches = [child for child in children if child[1] == run_id]
                if not matches or (track and (track, run_id) not in children):
                    target = f"{track}/{run_id}" if track else run_id
                    violations.append(
                        f"this session is scoped to orchestration {orchestration_id}; "
                        f"autoresearch command targets unlisted child run {target}."
                    )
            elif track:
                violations.append(
                    f"this session is scoped to orchestration {orchestration_id}; "
                    "takeover commands must pass an explicit --run-id for a listed child."
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
    """Scope for a session, hydrating orchestrator children from its manifest."""
    scope = load_scope_file(session_id)
    if scope is not None:
        return _hydrate_orchestrator_scope(scope)
    requested = os.environ.get("AUTORESEARCH_SCOPE", "").strip().lower()
    if requested == "analyst":
        return {"mode": "analyst", "source": "env"}
    if requested == "orchestrator":
        orchestration_id = os.environ.get("AUTORESEARCH_ORCHESTRATION_ID", "").strip()
        if orchestration_id:
            return _hydrate_orchestrator_scope(
                {"mode": "orchestrator", "orchestration_id": orchestration_id, "source": "env"}
            )
    return None


def _hydrate_orchestrator_scope(scope: dict) -> dict:
    """Return a fresh pure-decision scope from the orchestration manifest.

    The scope file stores only the orchestration id. Reading the manifest here,
    outside :func:`decide`, keeps decision logic pure while making newly spawned
    children visible on every hook invocation.
    """

    if scope.get("mode") != "orchestrator":
        return scope
    orchestration_id = str(scope.get("orchestration_id") or "")
    if not orchestration_id:
        return scope
    path = ROOT / "artifacts" / "orchestrations" / orchestration_id / "orchestration.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    hydrated = dict(scope)
    hydrated["orchestration_dir"] = f"artifacts/orchestrations/{orchestration_id}"
    hydrated["child_runs"] = [
        {"track": str(item["track"]), "run_id": str(item["run_id"])}
        for item in payload.get("delegations") or ()
    ]
    consolidation = payload.get("consolidation") or {}
    if consolidation.get("track") and consolidation.get("run_id"):
        hydrated["child_runs"].append(
            {
                "track": str(consolidation["track"]),
                "run_id": str(consolidation["run_id"]),
            }
        )
    root = ROOT / "artifacts" / "orchestrations"
    known = sorted(
        candidate.name
        for candidate in root.iterdir()
        if candidate.is_dir() and (candidate / "orchestration.json").exists()
    )
    hydrated["latest_orchestration_id"] = known[-1] if known else orchestration_id
    return hydrated


def write_scope(session_id: str, scope: dict) -> None:
    SCOPE_DIR.mkdir(parents=True, exist_ok=True)
    scope = dict(scope)
    scope.setdefault("bound_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    _scope_path(session_id).write_text(
        json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def extract_paths(tool_name: str, tool_input: dict) -> list[str]:
    """Return the path-bearing strings a tool call would actually touch."""
    tool_key = (tool_name or "").lower()
    if tool_key == "apply_patch":
        command = tool_input.get("command")
        return _apply_patch_targets(command) if isinstance(command, str) else []
    fields = _PATH_FIELDS.get(tool_key, _DEFAULT_PATH_FIELDS)
    out: list[str] = []
    for field in fields:
        value = tool_input.get(field)
        if isinstance(value, str) and value:
            out.append(value)
    if tool_key == "glob":
        out.extend(_glob_run_scope_targets(tool_input))
    return out


def _apply_patch_targets(command: str) -> list[str]:
    """Return only file paths from apply_patch headers, not mentioned content."""
    targets: list[str] = []
    prefixes = (
        "*** Add File: ",
        "*** Delete File: ",
        "*** Update File: ",
        "*** Move to: ",
    )
    for line in command.splitlines():
        for prefix in prefixes:
            if line.startswith(prefix):
                targets.append(line[len(prefix):].strip())
                break
    return targets


def _glob_run_scope_targets(tool_input: dict) -> list[str]:
    """Map broad OpenCode glob calls to the run folders they would enumerate."""
    path = tool_input.get("path")
    pattern = tool_input.get("pattern")
    if not isinstance(path, str) or not isinstance(pattern, str):
        return []
    if not _is_broad_glob(pattern):
        return []
    try:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = ROOT / target
        resolved = target.resolve()
    except Exception:
        return []

    candidates = (ROOT, ROOT / "artifacts", TRACKS_DIR)
    if any(_same_or_parent(candidate, resolved) for candidate in candidates):
        return ["artifacts/tracks/*/runs"]
    return []


def _is_broad_glob(pattern: str) -> bool:
    stripped = pattern.strip()
    return stripped in {"*", "**", "**/*", "./**/*"} or stripped.startswith("**/")


def _same_or_parent(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_run_refs(text: str) -> list[tuple[str, str | None]]:
    """Return (track, run_id-or-None) for every run reference in *text*."""
    return [(m.group(1), m.group(2)) for m in _RUN_REF.finditer(text)]


def find_orchestration_refs(text: str) -> list[str | None]:
    """Return orchestration ids (or ``None`` for root enumeration) in *text*."""

    return [match.group(1) for match in _ORCHESTRATION_REF.finditer(text)]


# ── decision (pure, unit-tested) ────────────────────────────────────────────

def decide(scope: dict | None, texts: list[str]) -> tuple[bool, str]:
    """Decide whether a tool call touching *texts* is allowed.

    Only a *bound research* session is confined to its own run. An unbound
    session (before its first ``autoresearch`` command) may inspect source,
    docs, and configs, but not run artifacts. Analyst sessions see everything.

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

    if scope.get("mode") == "analyst":
        return True, ""

    semantic_violations = find_autoresearch_scope_violations(scope, texts)
    if semantic_violations:
        return False, semantic_violations[0]

    orchestration_refs: list[str | None] = []
    for text in texts:
        orchestration_refs.extend(find_orchestration_refs(text))

    if scope.get("mode") == "orchestrator":
        orchestration_id = str(scope.get("orchestration_id") or "")
        children = {
            (str(child.get("track") or ""), str(child.get("run_id") or ""))
            for child in scope.get("child_runs") or ()
            if isinstance(child, dict)
        }
        for referenced_id in orchestration_refs:
            if referenced_id == orchestration_id:
                continue
            if referenced_id is None:
                return False, "enumerating artifacts/orchestrations is not allowed"
            return (
                False,
                f"this session is scoped to orchestration {orchestration_id}; the path "
                f"references another orchestration ({referenced_id}).",
            )
        refs = [ref for text in texts for ref in find_run_refs(text)]
        for track, run in refs:
            if run is None:
                return (
                    False,
                    f"enumerating sibling runs under artifacts/tracks/{track}/runs is "
                    "not allowed in orchestrator scope.",
                )
            if (track, run) not in children:
                return (
                    False,
                    f"this session is scoped to orchestration {orchestration_id}; the path "
                    f"references an unlisted child run ({track}/{run}).",
                )
        return True, ""

    if scope.get("mode") != "research":
        refs: list[tuple[str, str | None]] = []
        for text in texts:
            refs.extend(find_run_refs(text))
        if refs or orchestration_refs:
            if orchestration_refs:
                return (
                    False,
                    "orchestration not bound yet — orchestration artifacts are not allowed. "
                    "Run `autoresearch orchestrate new` first, or relaunch in analyst mode.",
                )
            track, run = refs[0]
            if run is None:
                return (
                    False,
                    f"run not bound yet — enumerating artifacts/tracks/{track}/runs is not allowed. "
                    "Run your `autoresearch --track <you> ... bootstrap-track` (or start-session) first; "
                    "for deliberate cross-run analysis, relaunch with AUTORESEARCH_SCOPE=analyst.",
                )
            return (
                False,
                f"run not bound yet — path references {track}/{run}. "
                "Run your `autoresearch --track <you> ... bootstrap-track` (or start-session) first; "
                "for deliberate cross-run analysis, relaunch with AUTORESEARCH_SCOPE=analyst.",
            )
        return True, ""  # unbound, but not touching run artifacts

    bound_track = scope.get("track")
    bound_run = scope.get("run_id")
    if not bound_run:
        return True, ""  # defensive: a research scope must name a run

    if orchestration_refs:
        return (
            False,
            "research sessions may not inspect orchestration artifacts; obey the brief "
            "in your own handoff and report through `finish-delegation`.",
        )

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
    elif requested == "orchestrator":
        orchestration_id = os.environ.get("AUTORESEARCH_ORCHESTRATION_ID", "").strip()
        if orchestration_id:
            write_scope(
                session_id,
                {"mode": "orchestrator", "orchestration_id": orchestration_id, "source": "env"},
            )
            _log(f"session {session_id}: orchestrator scope {orchestration_id} (env)")
    return 0


def handle_post_tool_use(payload: dict) -> int:
    """Auto-bind a research session to the run its autoresearch command targets."""
    session_id = payload.get("session_id") or ""
    if not session_id:
        return 0
    if load_scope_file(session_id) is not None:  # already analyst or bound
        return 0
    succeeded = _post_tool_succeeded(payload)
    if succeeded is False:
        return 0
    command = (payload.get("tool_input") or {}).get("command", "")
    orchestrator_invocations = _orchestrator_bindable_tokens(command)
    if orchestrator_invocations:
        orchestration_id = _flag_value(orchestrator_invocations[0], "--orchestration-id")
        orchestration_id = orchestration_id or _orchestration_id_from_post_payload(payload)
        if orchestration_id and RUN_ID_TIMESTAMP.fullmatch(orchestration_id):
            write_scope(
                session_id,
                {"mode": "orchestrator", "orchestration_id": orchestration_id, "source": "auto"},
            )
            _log(f"session {session_id}: orchestrator scope {orchestration_id} (auto)")
        return 0
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
    run_id = _flag_value(invocations[0], "--run-id")
    if not run_id and _has_flag(invocations[0], "--new-run"):
        run_id = _run_id_from_post_payload(payload)
        if not run_id and succeeded is True:
            run_id = _latest_run_id(track)
        if not run_id:
            _log(
                f"session {session_id}: refusing ambiguous --new-run auto-bind; "
                "tool result did not identify the created run"
            )
            return 0
    if not run_id:
        run_id = _latest_run_id(track)
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
