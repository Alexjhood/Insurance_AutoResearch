"""Stable split-pack and k-fold assignment generation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
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
    strata = _split_strata(frame)
    if strata is None:
        split_frame["split"] = split_frame["split_unit"].map(lambda value: assign_split(value, ratios))
        split_method = "stable_hash"
    else:
        split_frame["split_stratum"] = strata.to_numpy()
        split_frame["split"] = _assign_stratified_splits(split_frame, ratios)
        split_method = "target_exposure_stratified_hash"
    split_frame = split_frame.sort_values("record_id").reset_index(drop=True)

    counts = split_frame["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    manifest = {
        "split_pack_version": 2,
        "seed": seed,
        "id_column": id_column,
        "split_method": split_method,
        "stratification": _stratification_manifest(strata),
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


def _assign_stratified_splits(split_frame: pd.DataFrame, ratios: dict[str, float]) -> pd.Series:
    """Assign each stratum to splits in the configured proportions."""

    assigned = pd.Series(index=split_frame.index, dtype="object")
    for _, group in split_frame.groupby("split_stratum", sort=True):
        ordered = group.sort_values(["split_unit", "record_id"])
        counts = _proportional_counts(len(ordered), ratios)
        start = 0
        for split in SPLIT_ORDER:
            end = start + counts[split]
            assigned.loc[ordered.index[start:end]] = split
            start = end
    if assigned.isna().any():
        raise ValueError("Internal split assignment error: some rows were not assigned")
    return assigned


def _proportional_counts(n_rows: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: ratios[split] * n_rows for split in SPLIT_ORDER}
    counts = {split: int(np.floor(raw[split])) for split in SPLIT_ORDER}
    remaining = n_rows - sum(counts.values())
    order = sorted(SPLIT_ORDER, key=lambda split: (raw[split] - counts[split], ratios[split]), reverse=True)
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def _split_strata(frame: pd.DataFrame) -> pd.Series | None:
    """Return deterministic target/exposure strata when the needed columns exist."""

    claim_column = _first_existing(frame, ("ClaimAmount", "claim_cost_observed_k", "claim_cost_capped_active"))
    exposure_column = _first_existing(frame, ("Exposure", "exposure_term_a"))
    if claim_column is None or exposure_column is None:
        return None

    claim = frame[claim_column].astype(float).clip(lower=0, upper=100000)
    exposure = frame[exposure_column].astype(float).clip(lower=0)
    claim_band = pd.cut(
        claim,
        bins=[-0.01, 0.0, 1000.0, 5000.0, 10000.0, 20000.0, 50000.0, 75000.0, 99999.999, float("inf")],
        labels=[
            "zero",
            "lt_1k",
            "1k_5k",
            "5k_10k",
            "10k_20k",
            "20k_50k",
            "50k_75k",
            "75k_100k",
            "capped_100k",
        ],
        include_lowest=True,
    ).astype(str)
    exposure_band = _quantile_band(exposure, 5, "exp")
    return claim_band + "|" + exposure_band


def _quantile_band(series: pd.Series, n_bins: int, prefix: str) -> pd.Series:
    try:
        bands = pd.qcut(series, n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series([f"{prefix}_all"] * len(series), index=series.index)
    return bands.fillna(-1).astype(int).map(lambda value: f"{prefix}_{value}")


def _first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _stratification_manifest(strata: pd.Series | None) -> dict[str, Any] | None:
    if strata is None:
        return None
    counts = strata.value_counts()
    return {
        "columns": ["claim_cost_capped_at_100000", "exposure"],
        "stratum_count": int(counts.size),
        "min_stratum_size": int(counts.min()),
        "max_stratum_size": int(counts.max()),
    }
