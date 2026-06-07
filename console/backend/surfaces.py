"""
Per-surface model catalogs and thinking/reasoning-effort options.

Model enumeration differs by surface:
  - OpenCode exposes `opencode models` → fully dynamic list (cached per process).
  - Claude and Codex CLIs have no list command → curated static lists. The UI
    still allows a custom free-text entry, so these lists are suggestions.

Effort maps to each CLI's native flag:
  - Claude : --effort <low|medium|high|xhigh|max>
  - Codex  : -c model_reasoning_effort="<minimal|low|medium|high>"
  - OpenCode: --variant <provider-specific, e.g. minimal|low|medium|high|max>
"""

from __future__ import annotations

import subprocess
import threading

from console.backend import config as cfg

# Curated suggestions for CLIs that can't enumerate. Free text is still allowed.
_CLAUDE_MODELS = ["sonnet", "opus", "haiku", "claude-opus-4-8", "claude-sonnet-4-6"]
_CODEX_MODELS = ["gpt-5.5", "gpt-5-codex", "gpt-5", "o3", "o4-mini"]

_EFFORTS = {
    # "" = let the CLI/model use its default
    "claude": ["", "low", "medium", "high", "xhigh", "max"],
    "codex": ["", "minimal", "low", "medium", "high"],
    "opencode": ["", "minimal", "low", "medium", "high", "max"],
}

_EFFORT_LABEL = {
    "claude": "Thinking effort (--effort)",
    "codex": "Reasoning effort (model_reasoning_effort)",
    "opencode": "Variant / reasoning effort (--variant)",
}

# Default agent model per surface (used if the user doesn't pick one)
DEFAULT_MODEL = {
    "claude": "sonnet",
    "codex": "gpt-5.5",
    "opencode": "",   # let opencode use its configured default
}

_opencode_cache: list[str] | None = None
_lock = threading.Lock()


def _opencode_models() -> list[str]:
    global _opencode_cache
    if _opencode_cache is not None:
        return _opencode_cache
    with _lock:
        if _opencode_cache is not None:
            return _opencode_cache
        models: list[str] = []
        if cfg.OPENCODE_BIN:
            try:
                result = subprocess.run(
                    [cfg.OPENCODE_BIN, "models"],
                    capture_output=True, text=True, timeout=20,
                )
                models = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            except Exception:
                models = []
        _opencode_cache = models
        return models


def surface_info(surface: str) -> dict:
    bin_map = {
        "claude": cfg.CLAUDE_BIN,
        "codex": cfg.CODEX_BIN,
        "opencode": cfg.OPENCODE_BIN,
    }
    available = bool(bin_map.get(surface))
    if surface == "claude":
        models = _CLAUDE_MODELS
    elif surface == "codex":
        models = _CODEX_MODELS
    elif surface == "opencode":
        models = _opencode_models()
    else:
        models = []
    return {
        "surface": surface,
        "available": available,
        "models": models,
        "default_model": DEFAULT_MODEL.get(surface, ""),
        "efforts": _EFFORTS.get(surface, [""]),
        "effort_label": _EFFORT_LABEL.get(surface, "Effort"),
        "allows_custom_model": surface in ("claude", "codex"),
    }


def all_surfaces() -> list[dict]:
    return [surface_info(s) for s in ("claude", "codex", "opencode")]


def attribution_for(surface: str, model: str) -> tuple[str, str]:
    """Derive (model_provider, model_name) for memory-aggregator attribution."""
    model = (model or DEFAULT_MODEL.get(surface, "")).strip()
    if surface == "claude":
        return "anthropic", model or "sonnet"
    if surface == "codex":
        return "openai", model or "gpt-5.5"
    if surface == "opencode":
        # opencode models are "provider/model[/variant]" — split first segment
        if "/" in model:
            provider, _, name = model.partition("/")
            return provider, name
        return "opencode", model or "default"
    return surface, model or "unknown"
