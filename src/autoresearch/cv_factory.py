"""Build callable model factories from stored experiment config snapshots.

Used by the repeated-CV comparison path to refit champion and challenger on
each (repeat, fold) partition without re-running the full experiment runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.models.dispatcher import dispatch_model_on_explicit_frames
from autoresearch.utils.io import read_json


def build_model_factory_from_experiment(
    config: Any,
    experiment_id: str,
) -> Any:
    """Return a model factory reconstructed from a stored config_snapshot.

    The factory has the signature::

        factory(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame

    where the returned DataFrame is a predictions frame containing
    ``actual_target``, ``predicted_target``, ``exposure``, ``record_id``,
    and ``target_mode`` columns (same schema as ``dispatch_model`` output).

    Parameters
    ----------
    config:
        ``ProjectConfig`` for the active run — used to locate the registry.
    experiment_id:
        ID of the registered experiment whose config snapshot to load.
    """

    from autoresearch.experiment_registry.registry import list_artifacts

    # ── Locate and load config_snapshot ─────────────────────────────────────
    artifacts = list_artifacts(config.registry_path, experiment_id)
    artifact_map = {a["artifact_type"]: a["path"] for a in artifacts}

    snapshot_path = artifact_map.get("config_snapshot")
    if snapshot_path is None:
        raise ValueError(
            f"Experiment {experiment_id!r} has no config_snapshot artifact — "
            "cannot build a CV model factory from it"
        )
    snapshot = read_json(Path(snapshot_path))

    exp = snapshot.get("experiment", {})
    model_cfg = exp.get("model", {})
    model_family = exp.get("model_family", "global_mean")
    target_strategy = exp.get("target_strategy", "direct_pure_premium")
    target_mode = snapshot.get("target_mode", config.target_mode)

    feature_inclusions: list[str] | None = model_cfg.get("feature_inclusions")
    feature_exclusions: list[str] | None = model_cfg.get("feature_exclusions") or None

    hyperparameters: dict[str, Any] = {
        k: v for k, v in model_cfg.items()
        if k not in {"feature_inclusions", "feature_exclusions",
                     "script_path", "model_script_path", "script_sha256"}
    }

    # ── Resolve model script path ────────────────────────────────────────────
    model_script_path: Path | None = None
    raw_script = model_cfg.get("script_path") or model_cfg.get("model_script_path")
    if raw_script:
        # Prefer the registered artifact path (absolute, survives moves)
        registered = artifact_map.get("model_script")
        if registered and Path(registered).exists():
            model_script_path = Path(registered)
        else:
            # Fall back to the path recorded in the config snapshot
            candidate = Path(str(raw_script))
            if not candidate.is_absolute():
                candidate = config.root / candidate
            if candidate.exists():
                model_script_path = candidate

    if raw_script and model_script_path is None:
        raise ValueError(
            f"Experiment {experiment_id!r} requires a model script but it "
            f"cannot be located (tried {raw_script!r}).  "
            "Ensure the script file still exists at the recorded path."
        )

    # ── Build and return factory closure ────────────────────────────────────
    def _factory(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        result = dispatch_model_on_explicit_frames(
            train_df,
            val_df,
            model_family=model_family,
            target_strategy=target_strategy,
            hyperparameters=dict(hyperparameters),   # copy — hp may be mutated
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
            model_script_path=model_script_path,
            target_mode=target_mode,
        )
        return result.predictions

    _factory.__name__ = f"model_factory_{experiment_id[:32]}"
    return _factory
