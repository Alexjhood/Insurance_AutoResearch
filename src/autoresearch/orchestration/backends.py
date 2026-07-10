"""Backend registry — the only list of spawnable sub-agent tools.

A backend is a **model × thinking-effort combination**: effort belongs in the
command template, so "Sonnet 5 low" and "Sonnet 5 medium" are two named backends,
selectable and measurable independently.

``configs/orchestration/backends.toml`` is human-owned ground truth (design §4.7
Layer 1). Adding a model is a config edit with ``status = "trial"``; retiring one
is ``status = "deprecated"`` — existing delegations finish, new spawns refuse it.
The orchestrator never names a model from its own training data; it reads
``orchestrate list-backends`` and chooses from what is offered.

Adding a whole new *tool* (OpenCode, …) is a config edit plus, at most, an
adapter shim in :mod:`autoresearch.orchestration.spawner`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.config import PROJECT_ROOT


BACKENDS_CONFIG_PATH = PROJECT_ROOT / "configs" / "orchestration" / "backends.toml"

#: Tracks a research sub-agent may bind to — mirrors the guard's allow-list.
ALLOWED_TRACKS = frozenset({"claude", "codex", "opencode"})

TIERS = frozenset({"cheap", "mid", "frontier"})
STATUSES = frozenset({"default", "trial", "deprecated"})
PROMPT_CHANNELS = frozenset({"stdin", "argv"})

#: Placeholders the spawner substitutes into a command template.
_TEMPLATE_KEYS = frozenset({"prompt", "max_turns", "max_budget_usd"})

#: Enumerated flag values we validate ourselves, per tool, because the tool does
#: not. Claude Code 2.1.206 accepts ``--effort bogus`` with a *warning* and then
#: silently uses the default effort — which would quietly collapse a "low" and a
#: "medium" backend into the same configuration while the scorecard compared them
#: as rivals. A registry that lies about what it spawned is worse than no
#: registry, so a bad value fails at load time instead.
_TOOL_FLAG_ENUMS: dict[str, dict[str, frozenset[str]]] = {
    "claude": {
        "--effort": frozenset({"low", "medium", "high", "xhigh", "max"}),
        "--output-format": frozenset({"text", "json", "stream-json"}),
        "--permission-mode": frozenset(
            {"acceptEdits", "auto", "bypassPermissions", "manual", "dontAsk", "plan"}
        ),
    },
    "codex": {
        "-a": frozenset({"untrusted", "on-failure", "on-request", "never"}),
        "--ask-for-approval": frozenset(
            {"untrusted", "on-failure", "on-request", "never"}
        ),
        "-s": frozenset({"read-only", "workspace-write", "danger-full-access"}),
        "--sandbox": frozenset(
            {"read-only", "workspace-write", "danger-full-access"}
        ),
        "--color": frozenset({"always", "never", "auto"}),
    },
}

_CODEX_REASONING_LEVELS = frozenset({"minimal", "low", "medium", "high", "xhigh"})


@dataclass(frozen=True)
class Backend:
    """One spawnable model × effort combination."""

    name: str
    tool: str
    command: tuple[str, ...]
    prompt_via: str
    track: str
    model_provider: str
    model_name: str
    tier: str = "mid"
    status: str = "trial"
    cost_hint: str = "$$"
    good_for: tuple[str, ...] = ()
    avoid_for: tuple[str, ...] = ()
    max_turns: int | None = None
    max_budget_usd: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.track not in ALLOWED_TRACKS:
            raise ValueError(
                f"backend {self.name!r}: track {self.track!r} is not one of "
                f"{sorted(ALLOWED_TRACKS)}; the run-scope guard would refuse to bind it."
            )
        if self.tier not in TIERS:
            raise ValueError(
                f"backend {self.name!r}: tier {self.tier!r} must be one of {sorted(TIERS)}"
            )
        if self.status not in STATUSES:
            raise ValueError(
                f"backend {self.name!r}: status {self.status!r} must be one of {sorted(STATUSES)}"
            )
        if self.prompt_via not in PROMPT_CHANNELS:
            raise ValueError(
                f"backend {self.name!r}: prompt_via {self.prompt_via!r} must be one of "
                f"{sorted(PROMPT_CHANNELS)}"
            )
        if not self.command:
            raise ValueError(f"backend {self.name!r}: command must not be empty")
        if self.prompt_via == "argv" and "{prompt}" not in self.command:
            raise ValueError(
                f"backend {self.name!r}: prompt_via='argv' requires a '{{prompt}}' "
                "placeholder in the command template"
            )
        if self.prompt_via == "stdin" and "{prompt}" in self.command:
            raise ValueError(
                f"backend {self.name!r}: prompt_via='stdin' must not carry a "
                "'{prompt}' placeholder in the command template"
            )
        unknown = _unknown_placeholders(self.command)
        if unknown:
            raise ValueError(
                f"backend {self.name!r}: unknown command placeholder(s) "
                f"{sorted(unknown)}; known: {sorted(_TEMPLATE_KEYS)}"
            )
        for placeholder, value in (
            ("max_turns", self.max_turns),
            ("max_budget_usd", self.max_budget_usd),
        ):
            if any(f"{{{placeholder}}}" in token for token in self.command) and value is None:
                raise ValueError(
                    f"backend {self.name!r}: command uses '{{{placeholder}}}' but no "
                    f"{placeholder} is configured"
                )
        self._validate_enumerated_flags()
        self._validate_tool_contract()

    def _validate_enumerated_flags(self) -> None:
        enums = _TOOL_FLAG_ENUMS.get(self.tool)
        if not enums:
            return
        for index, token in enumerate(self.command):
            allowed = enums.get(token)
            if allowed is None:
                continue
            if index + 1 >= len(self.command):
                raise ValueError(
                    f"backend {self.name!r}: {token} is the last token and carries no value"
                )
            value = self.command[index + 1]
            if value not in allowed:
                fallback = (
                    f" ({self.tool} would warn and silently fall back to its default.)"
                    if self.tool == "claude"
                    else ""
                )
                raise ValueError(
                    f"backend {self.name!r}: {token}={value!r} is not accepted by "
                    f"{self.tool!r}; valid values: {sorted(allowed)}.{fallback}"
                )

    def _validate_tool_contract(self) -> None:
        """Fail at registry load when a real tool command is not headless-safe."""

        if self.tool != "codex" or Path(self.command[0]).name != "codex":
            return
        if "exec" not in self.command:
            raise ValueError(f"backend {self.name!r}: Codex backends must use `codex exec`")
        if "--json" not in self.command:
            raise ValueError(
                f"backend {self.name!r}: Codex backends require --json for exit and usage parsing"
            )
        if "--max-turns" in self.command:
            raise ValueError(
                f"backend {self.name!r}: codex exec 0.137.0 has no --max-turns flag"
            )
        if any(token.startswith("--dangerously-bypass-") for token in self.command):
            raise ValueError(
                f"backend {self.name!r}: Codex backends may not bypass approvals or sandboxing"
            )
        if _flag_value(self.command, "-s", "--sandbox") != "workspace-write":
            raise ValueError(
                f"backend {self.name!r}: Codex backends must pin --sandbox workspace-write"
            )
        if self.prompt_via == "stdin" and self.command[-1] != "-":
            raise ValueError(
                f"backend {self.name!r}: stdin Codex backends must end with `-` to read the prompt"
            )

        effort_values = []
        for index, token in enumerate(self.command[:-1]):
            if token not in {"-c", "--config"}:
                continue
            value = self.command[index + 1]
            if value.startswith("model_reasoning_effort="):
                effort_values.append(value.partition("=")[2].strip().strip("\"'"))
        if len(effort_values) != 1 or effort_values[0] not in _CODEX_REASONING_LEVELS:
            raise ValueError(
                f"backend {self.name!r}: Codex backends must pin one "
                "model_reasoning_effort in "
                f"{sorted(_CODEX_REASONING_LEVELS)}"
            )

    @property
    def is_spawnable(self) -> bool:
        return self.status != "deprecated"

    def render_command(self, *, prompt: str) -> tuple[str, ...]:
        """Substitute template placeholders, yielding the exact argv to launch."""

        substitutions = {
            "{prompt}": prompt,
            "{max_turns}": str(self.max_turns) if self.max_turns is not None else "",
            "{max_budget_usd}": (
                f"{self.max_budget_usd:g}" if self.max_budget_usd is not None else ""
            ),
        }
        rendered: list[str] = []
        for token in self.command:
            for key, value in substitutions.items():
                token = token.replace(key, value)
            rendered.append(token)
        return tuple(rendered)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tool": self.tool,
            "command": list(self.command),
            "prompt_via": self.prompt_via,
            "track": self.track,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "tier": self.tier,
            "status": self.status,
            "cost_hint": self.cost_hint,
            "good_for": list(self.good_for),
            "avoid_for": list(self.avoid_for),
            "max_turns": self.max_turns,
            "max_budget_usd": self.max_budget_usd,
            "notes": self.notes,
        }


def _unknown_placeholders(command: tuple[str, ...]) -> set[str]:
    import re

    found = set()
    for token in command:
        found.update(re.findall(r"\{([a-z_]+)\}", token))
    return found - _TEMPLATE_KEYS


def _flag_value(command: tuple[str, ...], *flags: str) -> str | None:
    for index, token in enumerate(command[:-1]):
        if token in flags:
            return command[index + 1]
    return None


def load_backends(path: Path | None = None) -> dict[str, Backend]:
    """Load and validate every backend from ``backends.toml``."""

    config_path = path or BACKENDS_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(
            f"No backend registry at {config_path}. Orchestration needs at least one "
            "backend; see configs/orchestration/backends.toml."
        )
    with config_path.open("rb") as fh:
        raw = tomllib.load(fh)

    table = raw.get("backends") or {}
    if not table:
        raise ValueError(f"Backend registry {config_path} defines no [backends.*] entries")

    backends: dict[str, Backend] = {}
    for name, entry in table.items():
        backends[name] = _parse_backend(name, entry, config_path)
    return backends


def _parse_backend(name: str, entry: dict[str, Any], config_path: Path) -> Backend:
    missing = [
        key
        for key in ("tool", "command", "prompt_via", "track", "model_provider", "model_name")
        if key not in entry
    ]
    if missing:
        raise ValueError(
            f"Backend {name!r} in {config_path} is missing required key(s): {', '.join(missing)}"
        )
    return Backend(
        name=name,
        tool=str(entry["tool"]),
        command=tuple(str(c) for c in entry["command"]),
        prompt_via=str(entry["prompt_via"]),
        track=str(entry["track"]),
        model_provider=str(entry["model_provider"]),
        model_name=str(entry["model_name"]),
        tier=str(entry.get("tier", "mid")),
        status=str(entry.get("status", "trial")),
        cost_hint=str(entry.get("cost_hint", "$$")),
        good_for=tuple(str(g) for g in entry.get("good_for", [])),
        avoid_for=tuple(str(a) for a in entry.get("avoid_for", [])),
        max_turns=(int(entry["max_turns"]) if entry.get("max_turns") is not None else None),
        max_budget_usd=(
            float(entry["max_budget_usd"]) if entry.get("max_budget_usd") is not None else None
        ),
        notes=str(entry.get("notes", "")),
    )


def get_backend(name: str, *, path: Path | None = None) -> Backend:
    """Resolve one backend by name, refusing deprecated entries."""

    backends = load_backends(path)
    backend = backends.get(name)
    if backend is None:
        available = ", ".join(
            sorted(n for n, b in backends.items() if b.is_spawnable)
        ) or "(none)"
        raise KeyError(
            f"Unknown backend {name!r}. Spawnable backends: {available}. "
            "Run `autoresearch orchestrate list-backends` to see the registry."
        )
    if not backend.is_spawnable:
        raise ValueError(
            f"Backend {name!r} is deprecated and may not be spawned. "
            "Choose a `default` or `trial` backend from `orchestrate list-backends`."
        )
    return backend


#: Home-state directory each tool must be able to write. The 2026-07-10 Sol
#: campaign lost 8/10 budget cycles to exactly these two failures: an
#: unauthenticated `claude` and a `codex` child that could not write ~/.codex
#: under the orchestrator harness's sandbox.
_TOOL_STATE_DIRS = {"claude": ".claude", "codex": ".codex"}


def preflight_backend(backend: Backend) -> None:
    """Fail a spawn fast, with a clear message, on known environment problems.

    Checks the backend executable is on PATH and (for tools with local state)
    that the tool's home-state directory is actually writable — a real write
    probe, because a harness sandbox denies at the syscall, not the permission
    bits. Cannot verify authentication without a paid call; an auth failure
    still surfaces through the delegation report's ``crashed`` flag.
    """

    import os
    import shutil
    from pathlib import Path as _Path

    executable = backend.command[0]
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"Backend {backend.name!r}: executable {executable!r} is not on PATH. "
            "Install it (or fix PATH) before spawning."
        )

    state_name = _TOOL_STATE_DIRS.get(backend.tool)
    if state_name is None:
        return
    state_dir = _Path.home() / state_name
    probe = state_dir / f".autoresearch_preflight_{os.getpid()}"
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("preflight\n", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Backend {backend.name!r}: cannot write {state_dir} ({exc}). The "
            f"spawned `{backend.tool}` CLI needs writable home state; if you are "
            "orchestrating from a sandboxed harness, relaunch it with enough "
            "process privileges (e.g. Codex `--sandbox danger-full-access`)."
        ) from exc


def preflight_foundation_models() -> None:
    """Fail a foundation-opted spawn fast when the environment cannot run TabPFN.

    Called by :func:`spawner.spawn`/``respawn`` only when the brief sets
    ``foundation_models: true`` (skipped on ``--dry-run``). Checks, each naming
    its fix:

    * **at least one** foundation estimator's extra must be importable (TabPFN's
      ``[foundation]`` or TabFM's ``[foundation-modal]``) — the child inherits
      the parent env and interpreter, so an extra-less orchestrator means an
      extra-less child; and
    * if ``AUTORESEARCH_TABPFN_BACKEND`` resolves to ``api``, ``TABPFN_TOKEN``
      must be set — the child inherits an exported token, so this catches the
      case where neither the orchestrator nor its shell has one.

    TabFM's own readiness (a deployed Modal app + ``~/.modal.toml``) surfaces at
    fit time via the estimator's ``_authenticate_modal`` check; a Modal deploy is
    not something a spawn can cheaply verify. Neither can we verify a token is
    *valid* without a paid call; an invalid one still surfaces through the
    delegation report's ``crashed`` flag.
    """

    import os

    from autoresearch.models.recipe.foundation import tabfm_available, tabpfn_available

    if not (tabpfn_available() or tabfm_available()):
        raise RuntimeError(
            "Brief opts into foundation models but no foundation extra is importable "
            "in this environment. Install one (`pip install -e '.[foundation]'` for "
            "TabPFN, or `'.[foundation-modal]'` for TabFM) before spawning a "
            "foundation delegation; the child inherits this interpreter."
        )
    backend = os.environ.get("AUTORESEARCH_TABPFN_BACKEND", "local").strip().lower()
    if backend == "api" and not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError(
            "Brief opts into foundation models with AUTORESEARCH_TABPFN_BACKEND=api "
            "but TABPFN_TOKEN is unset. Export TABPFN_TOKEN (Prior Labs API key from "
            "https://ux.priorlabs.ai/account) before spawning; the child inherits it."
        )


def format_backend_table(
    backends: dict[str, Backend],
    stats: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render the registry as the one view the orchestrator consults when choosing.

    *stats* is the empirical scorecard (design §4.7 Layer 3), appended per entry
    when available. Absent stats render as ``—`` rather than being hidden, so a
    never-used backend is visibly untested rather than silently missing.
    """

    stats = stats or {}
    lines = [
        "Spawnable backends (from configs/orchestration/backends.toml):",
        "",
    ]
    for name in sorted(backends):
        backend = backends[name]
        marker = {"default": "", "trial": "  [TRIAL]", "deprecated": "  [DEPRECATED]"}[
            backend.status
        ]
        lines.append(f"- {name}{marker}")
        lines.append(
            f"    tool={backend.tool}  track={backend.track}  tier={backend.tier}  "
            f"cost={backend.cost_hint}"
        )
        lines.append(f"    model={backend.model_provider}/{backend.model_name}")
        if backend.good_for:
            lines.append(f"    good_for: {', '.join(backend.good_for)}")
        if backend.avoid_for:
            lines.append(f"    avoid_for: {', '.join(backend.avoid_for)}")
        if backend.notes:
            lines.append(f"    notes: {backend.notes}")
        entry = stats.get(name)
        if entry:
            lines.append(
                f"    scorecard: {entry.get('delegations', 0)} delegations / "
                f"{entry.get('cycles', 0)} cycles; "
                f"promotion_rate={_fmt(entry.get('promotion_rate'))}; "
                f"distress_rate={_fmt(entry.get('distress_rate'))}"
            )
        else:
            lines.append("    scorecard: — (no delegations recorded yet)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
