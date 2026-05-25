"""Stable split-pack and k-fold assignment generation."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


SPLIT_ORDER = ("train", "search_validation", "milestone_holdout")


def stable_unit(value: object, seed: int) -> float:
    """Map a record id to a deterministic float in [0, 1)."""

    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    integer = int(digest[:16], 16)
    return integer / float(16**16)


def validate_split_ratios(ratios: dict[str, float]) -> None:
    """Validate split names and ensure ratios sum to one."""

    missing = set(SPLIT_ORDER).difference(ratios)
    if missing:
        raise ValueError(f"Missing split ratios: {sorted(missing)}")
    total = sum(ratios[name] for name in SPLIT_ORDER)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if any(ratios[name] <= 0 for name in SPLIT_ORDER):
        raise ValueError("All split ratios must be positive")


def assign_split(unit_value: float, ratios: dict[str, float]) -> str:
    """Assign a split from a stable unit interval value."""

    cumulative = 0.0
    for name in SPLIT_ORDER:
        cumulative += ratios[name]
        if unit_value < cumulative:
            return name
    return SPLIT_ORDER[-1]


def generate_fold_assignments(
    frame: pd.DataFrame,
    id_column: str,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Assign deterministic k-fold labels (1..n_folds) to each row."""

    if id_column not in frame.columns:
        raise ValueError(f"Fold id column {id_column!r} is not present")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    fold_frame = pd.DataFrame({"record_id": frame[id_column]})
    fold_frame["fold_unit"] = fold_frame["record_id"].map(lambda v: stable_unit(v, seed + 1))
    fold_frame = fold_frame.sort_values("fold_unit").reset_index(drop=True)
    n = len(fold_frame)
    fold_frame["fold"] = [i % n_folds + 1 for i in range(n)]
    return fold_frame[["record_id", "fold"]].sort_values("record_id").reset_index(drop=True)


def generate_split_pack(
    frame: pd.DataFrame,
    id_column: str,
    ratios: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate and describe persistent split definitions."""

    validate_split_ratios(ratios)
    if id_column not in frame.columns:
        raise ValueError(f"Split id column {id_column!r} is not present")
    if frame[id_column].duplicated().any():
        raise ValueError(f"Split id column {id_column!r} must be unique")

    split_frame = pd.DataFrame({"record_id": frame[id_column]})
    split_frame["split_unit"] = split_frame["record_id"].map(lambda value: stable_unit(value, seed))
    split_frame["split"] = split_frame["split_unit"].map(lambda value: assign_split(value, ratios))
    split_frame = split_frame.sort_values("record_id").reset_index(drop=True)

    counts = split_frame["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    manifest = {
        "split_pack_version": 1,
        "seed": seed,
        "id_column": id_column,
        "ratios": {name: ratios[name] for name in SPLIT_ORDER},
        "counts": {name: int(counts[name]) for name in SPLIT_ORDER},
        "ordinary_search_splits": ["train", "search_validation"],
        "milestone_holdout_ratio": ratios["milestone_holdout"],
        "holdout_policy": (
            "milestone_holdout is 20% of the full dataset and is reserved for milestone "
            "evaluation only. Ordinary baseline evaluation must use train for fitting and "
            "search_validation for search-time scoring."
        ),
    }
    return split_frame, manifest
