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
    from autoresearch.memory import harvester
    from autoresearch.memory.store import default_memory_store_path

    memory_path = default_memory_store_path()
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
    _maybe_rebuild_playbook(config, memory_path)


def _maybe_rebuild_playbook(config: "ProjectConfig", memory_path: "Path") -> None:
    """Regenerate the playbook if new verified insights have landed since last build."""
    try:
        from autoresearch.memory.playbook import build_playbook, playbook_needs_rebuild

        playbook_dir = memory_path.parent / "playbook"
        latest_path = playbook_dir / "latest.md"
        threshold = getattr(config, "structural_gini_threshold", 0.37)

        if playbook_needs_rebuild(memory_path, latest_path):
            build_playbook(memory_path, structural_gini_threshold=threshold)
    except Exception as exc:
        logger.debug("Playbook rebuild skipped (non-fatal): %s", exc)


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
            "A checkpoint just ran for THIS run. You may optionally record 0-3 short, "
            "evidence-bound insights about what you have learned so far in this run. "
            "This is entirely optional and only concerns your own run -- skip it if there "
            "is nothing genuinely new to say.\n\n"
            "Every insight must cite real `experiment_id` / `comparison_id` values from "
            "this run's registry so it can be checked against the recorded metrics. "
            "Insights whose cited evidence does not match the registry are stored but "
            "flagged `verified=0` and ignored downstream, so only record claims you can "
            "back with IDs.\n\n"
            "## Where to find the IDs\n\n"
            "- The handoff / context bundle for this run lists recent experiments and "
            "comparisons.\n"
            "- Or list them directly:\n\n"
            "```bash\n"
            "autoresearch list-experiments       # experiment_id values for this run\n"
            "autoresearch list-promotions        # comparison_id values + decisions\n"
            "```\n\n"
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
