"""Cross-run memory aggregator — checkpoint wrapper and access gate.

Public API:
  maybe_memory_checkpoint(config, state) — harvest + reflection prompt (session loop).
  resolve_memory_access(config) -> str   — 'none' | 'own' | 'all' from env or manifest.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoresearch.config import ProjectConfig

logger = logging.getLogger(__name__)

_VALID_ACCESS_LEVELS = frozenset({"none", "own", "all"})


def resolve_memory_access(config: "ProjectConfig | None" = None) -> str:
    """Resolve AUTORESEARCH_MEMORY_ACCESS to 'none' | 'own' | 'all'.

    Priority: env var > manifest value > default 'none'.
    The agent cannot set its own env var, so it cannot self-escalate.
    """
    env_val = os.environ.get("AUTORESEARCH_MEMORY_ACCESS", "").strip().lower()
    if env_val in _VALID_ACCESS_LEVELS:
        return env_val

    # Fall back to the manifest value (recorded at bootstrap time).
    if config is not None:
        try:
            manifest_path = config.artifacts_dir / "run_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_val = str(manifest.get("memory_access") or "").strip().lower()
                if manifest_val in _VALID_ACCESS_LEVELS:
                    return manifest_val
        except (OSError, json.JSONDecodeError):
            pass

    return "none"


def maybe_memory_checkpoint(config: "ProjectConfig", state: dict[str, Any]) -> None:
    """Harvest the current run into the aggregator. Never raises into the loop."""
    try:
        _run_checkpoint(config)
    except Exception as exc:
        logger.warning("memory checkpoint failed (non-fatal): %s", exc)


def _run_checkpoint(config: "ProjectConfig") -> None:
    from autoresearch.config import PROJECT_ROOT
    from autoresearch.memory import harvester

    memory_path = PROJECT_ROOT / "artifacts" / "memory" / "memory.sqlite"
    manifest_path = config.artifacts_dir / "run_manifest.json"
    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    model_identity = manifest.get("model_identity")
    if not model_identity or not model_identity.get("provider") or not model_identity.get("name"):
        return

    harvester.harvest_run(
        memory_path,
        config.registry_path,
        model_identity,
        track_id=config.track_id,
        run_id=config.run_id,
    )

    _write_reflection_prompt(config)


def _write_reflection_prompt(config: "ProjectConfig") -> None:
    """Write pending_reflection.md into the run's handoff dir.

    This prompts the agent to record 0-3 evidence-bound insights via
    `autoresearch memory record-insight --file <path.json>`.
    Capture is best-effort; the validator gates trust.
    """
    try:
        handoff_dir = config.handoff_handoffs_dir
        handoff_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = handoff_dir / "pending_reflection.md"
        content = (
            "# Reflection Prompt\n\n"
            "A memory checkpoint just ran. You may optionally record 0-3 evidence-bound "
            "insights from this session. Each insight must cite real experiment_ids and/or "
            "comparison_ids from your run's registry so the validator can verify it.\n\n"
            "## How to record an insight\n\n"
            "Write a JSON file matching the schema below, then run:\n\n"
            "```bash\n"
            "autoresearch memory record-insight --file /path/to/insight.json\n"
            "```\n\n"
            "## Insight schema\n\n"
            "```json\n"
            "{\n"
            '  "claim": "your concise claim about what works or plateaus",\n'
            '  "scope": "general",\n'
            '  "confidence": 0.8,\n'
            '  "evidence": {\n'
            '    "experiment_ids": ["exp_id_1", "exp_id_2"],\n'
            '    "comparison_ids": ["cmp_id_1"],\n'
            '    "metric": "gini_weighted",\n'
            '    "delta": 0.05\n'
            "  },\n"
            '  "supersedes": null,\n'
            '  "contradicts": null\n'
            "}\n"
            "```\n\n"
            "Insights with fabricated evidence will be stored with `verified=0` and "
            "excluded from the playbook and default query results.\n"
        )
        prompt_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write pending_reflection.md: %s", exc)
