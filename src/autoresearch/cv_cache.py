"""Per-fold prediction cache for the CV bootstrap comparison path.

Champion predictions are stable within a run until a promotion occurs.
Caching them avoids O(n_folds) model refits per comparison cycle.

Cache key: sha256(experiment_id, dataset_hash, partition_seed, cap_threshold).
Cache location: <artifacts_dir>/cv_cache/<experiment_id>/partition_<index>.parquet
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.data.splits import generate_fold_assignments, fold_seed_from_run_id


def _dataset_hash(frame: pd.DataFrame) -> str:
    """Return a stable hash of the search dataset frame shape and column checksums."""
    payload = f"{len(frame)}|{sorted(frame.columns.tolist())}|{int(frame.select_dtypes('number').sum().sum())}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_key(
    experiment_id: str,
    dataset_hash: str,
    partition_seed: int,
    cap_threshold: float,
) -> str:
    raw = f"{experiment_id}|{dataset_hash}|{partition_seed}|{cap_threshold}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _cache_dir(artifacts_dir: Path, experiment_id: str) -> Path:
    return artifacts_dir / "cv_cache" / experiment_id


def get_or_build_fold_predictions(
    config: Any,
    experiment_id: str,
    partition_index: int,
    frame: pd.DataFrame,
    *,
    force_rebuild: bool = False,
) -> list[pd.DataFrame]:
    """Return cached per-fold validation predictions, building if necessary.

    Parameters
    ----------
    config:
        ``ProjectConfig`` for the active run.
    experiment_id:
        ID of the registered experiment whose factory to rebuild if cache misses.
    partition_index:
        Which fold partition to use (0 = base; 1..N = escalation partitions).
    frame:
        Full search dataset (train + search_validation, cap applied) used for CV.
    force_rebuild:
        Skip cache lookup and always refit (--no-cache flag).

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per fold containing ``record_id``, ``actual_target``,
        ``predicted_target``, ``exposure``, and ``target_mode`` columns.
    """

    from autoresearch.cv_factory import build_model_factory_from_experiment

    partition_seed = fold_seed_from_run_id(config.run_id, partition_index)
    ds_hash = _dataset_hash(frame)
    cap = float(config.claim_cap_threshold)
    key = _cache_key(experiment_id, ds_hash, partition_seed, cap)

    cache_dir = _cache_dir(config.artifacts_dir, experiment_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"partition_{partition_index}.parquet"
    key_file = cache_dir / f"partition_{partition_index}.key"

    # Cache hit?
    if not force_rebuild and cache_file.exists() and key_file.exists():
        stored_key = key_file.read_text(encoding="utf-8").strip()
        if stored_key == key:
            cached = pd.read_parquet(cache_file)
            n_folds = int(cached["fold"].max())
            return [cached[cached["fold"] == f + 1].drop(columns=["fold"]).reset_index(drop=True)
                    for f in range(n_folds)]

    # Cache miss — build fold assignments and refit.
    # The search dataset is keyed by "record_id" (post-anonymisation), matching
    # the data pipeline — NOT config.id_column (the raw "IDpol").
    n_folds = getattr(config, "cv_folds", 4)
    fold_assignments = generate_fold_assignments(
        frame,
        "record_id",
        n_folds,
        partition_seed,
        partition_index=partition_index,
    )

    factory = build_model_factory_from_experiment(config, experiment_id)
    merged = frame.merge(fold_assignments[["record_id", "fold"]], on="record_id", how="inner")

    fold_frames: list[pd.DataFrame] = []
    for fold in range(1, n_folds + 1):
        val_mask = merged["fold"] == fold
        train_data = merged[~val_mask].copy()
        val_data = merged[val_mask].copy()
        preds = factory(train_data, val_data)
        preds = preds.copy()
        preds["fold"] = fold
        fold_frames.append(preds)

    all_preds = pd.concat(fold_frames, ignore_index=True)
    all_preds.to_parquet(cache_file, index=False)
    key_file.write_text(key, encoding="utf-8")

    return [fold_frames[f].drop(columns=["fold"], errors="ignore").reset_index(drop=True)
            for f in range(n_folds)]


def invalidate_cache(artifacts_dir: Path, experiment_id: str) -> None:
    """Remove all cached fold predictions for an experiment."""
    cache_dir = _cache_dir(artifacts_dir, experiment_id)
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
