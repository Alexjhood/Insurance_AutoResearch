"""Cross-run memory aggregator — checkpoint wrapper.

The public entry point for the session loop is maybe_memory_checkpoint().
It is a no-op-safe wrapper: exceptions are caught and logged so the loop
never crashes from memory errors.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autoresearch.config import ProjectConfig

logger = logging.getLogger(__name__)


def maybe_memory_checkpoint(config: "ProjectConfig", state: dict[str, Any]) -> None:
    """Harvest the current run into the aggregator. Never raises into the loop."""
    try:
        _run_checkpoint(config)
    except Exception as exc:
        logger.warning("memory checkpoint failed (non-fatal): %s", exc)


def _run_checkpoint(config: "ProjectConfig") -> None:
    from autoresearch.config import PROJECT_ROOT
    from autoresearch.memory.harvester import harvest_run

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

    harvest_run(
        memory_path,
        config.registry_path,
        model_identity,
        track_id=config.track_id,
        run_id=config.run_id,
    )
