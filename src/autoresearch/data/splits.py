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


def fold_seed_from_run_id(run_id: str, partition_index: int = 0) -> int:
    """Return a deterministic integer seed for a given run_id and partition_index.

    Partition 0 is the base partition used for every run; escalation partitions
    use indices 1, 2, ... Each run gets a unique seed derived from its run_id so
    that different runs always use different fold assignments, while remaining
    fully reproducible within a run.
    """

    digest = hashlib.sha256(f"fold_seed|{run_id}|{partition_index}".encode("utf-8")).hexdigest()
    # Use the first 15 hex digits to stay within numpy int64 range
    return int(digest[:15], 16)


def generate_fold_assignments(
    frame: pd.DataFrame,
    id_column: str,
    n_folds: int,
    seed: int,
    *,
    partition_index: int = 0,
    unit_column: str | None = None,
    stratify_target: str | None = None,
    stratify_weight: str | None = None,
    stratify_bands: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Assign deterministic stratified k-fold labels (1..n_folds) to each row.

    Stratification mirrors the split-pack logic. When ``unit_column`` names a
    grouping column with repeated values, folds are assigned at the unit level
    and broadcast so no unit straddles a fold boundary (group-aware CV).
    """

    if id_column not in frame.columns:
        raise ValueError(f"Fold id column {id_column!r} is not present")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    effective_seed = seed + partition_index
    grouping = (
        bool(unit_column)
        and unit_column in frame.columns
        and unit_column != id_column
        and bool(frame[unit_column].duplicated().any())
    )

    if grouping:
        agg: dict[str, Any] = {}
        if stratify_target and stratify_target in frame.columns:
            agg[stratify_target] = (stratify_target, "sum")
        if stratify_weight and stratify_weight in frame.columns:
            agg[stratify_weight] = (stratify_weight, "sum")
        grouped = frame.groupby(unit_column, sort=True)
        units = grouped.agg(**agg).reset_index() if agg else pd.DataFrame({unit_column: list(grouped.groups)})
        unit_fold = pd.DataFrame({"record_id": units[unit_column]})
        unit_fold["fold_unit"] = unit_fold["record_id"].map(lambda v: stable_unit(v, effective_seed + 1))
        strata = _compute_strata(units, stratify_target, stratify_weight, stratify_bands)
        unit_fold["fold"] = _folds_within_strata(unit_fold, strata, n_folds)
        unit_to_fold = dict(zip(units[unit_column], unit_fold["fold"]))
        out = pd.DataFrame({"record_id": frame[id_column].to_numpy()})
        out["fold"] = frame[unit_column].map(unit_to_fold).to_numpy()
        return out.sort_values("record_id").reset_index(drop=True)

    fold_frame = pd.DataFrame({"record_id": frame[id_column]})
    fold_frame["fold_unit"] = fold_frame["record_id"].map(lambda v: stable_unit(v, effective_seed + 1))
    strata = _compute_strata(frame, stratify_target, stratify_weight, stratify_bands)
    fold_frame["fold"] = _folds_within_strata(fold_frame, strata, n_folds)
    return fold_frame[["record_id", "fold"]].sort_values("record_id").reset_index(drop=True)


def _folds_within_strata(fold_frame: pd.DataFrame, strata: pd.Series | None, n_folds: int) -> np.ndarray:
    """Assign round-robin folds within each stratum (or globally when unstratified).

    Returns an array aligned positionally to ``fold_frame`` rows (works for any
    index, including the non-contiguous index of a search-partition frame).
    """

    work = fold_frame.reset_index(drop=True)
    folds = np.empty(len(work), dtype=int)
    if strata is not None:
        work = work.copy()
        work["stratum"] = np.asarray(strata)
        for _, group in work.groupby("stratum", sort=True):
            ordered = group.sort_values(["fold_unit", "record_id"])
            folds[ordered.index.to_numpy()] = [i % n_folds + 1 for i in range(len(ordered))]
    else:
        ordered = work.sort_values("fold_unit")
        folds[ordered.index.to_numpy()] = [i % n_folds + 1 for i in range(len(ordered))]
    return folds


def generate_split_pack(
    frame: pd.DataFrame,
    id_column: str,
    ratios: dict[str, float],
    seed: int,
    *,
    unit_column: str | None = None,
    stratify_target: str | None = None,
    stratify_weight: str | None = None,
    stratify_bands: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate and describe persistent split definitions.

    When ``unit_column`` names a grouping column with repeated values (e.g. an
    AllState ``Household_ID``), splitting is *group-aware*: every row of a unit
    follows that unit into the same split, so no unit straddles the boundary.
    Stratification reads ``stratify_target``/``stratify_weight`` when given,
    otherwise auto-detects the French claim/exposure columns for byte-identical
    backward compatibility.
    """

    validate_split_ratios(ratios)
    if id_column not in frame.columns:
        raise ValueError(f"Split id column {id_column!r} is not present")
    if frame[id_column].duplicated().any():
        raise ValueError(f"Split id column {id_column!r} must be unique")

    grouping = (
        bool(unit_column)
        and unit_column in frame.columns
        and unit_column != id_column
        and bool(frame[unit_column].duplicated().any())
    )

    if grouping:
        return _generate_grouped_split_pack(
            frame, id_column, unit_column, ratios, seed,
            stratify_target=stratify_target, stratify_weight=stratify_weight,
            stratify_bands=stratify_bands,
        )

    split_frame = pd.DataFrame({"record_id": frame[id_column]})
    split_frame["split_unit"] = split_frame["record_id"].map(lambda value: stable_unit(value, seed))
    strata = _compute_strata(frame, stratify_target, stratify_weight, stratify_bands)
    if strata is None:
        split_frame["split"] = split_frame["split_unit"].map(lambda value: assign_split(value, ratios))
        split_method = "stable_hash"
    else:
        split_frame["split_stratum"] = strata.to_numpy()
        split_frame["split"] = _assign_stratified_splits(split_frame, ratios)
        split_method = "target_exposure_stratified_hash"
    split_frame = split_frame.sort_values("record_id").reset_index(drop=True)

    counts = split_frame["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    manifest = _split_manifest(seed, id_column, split_method, strata, ratios, counts, unit_column=None)
    return split_frame, manifest


def _generate_grouped_split_pack(
    frame: pd.DataFrame,
    id_column: str,
    unit_column: str,
    ratios: dict[str, float],
    seed: int,
    *,
    stratify_target: str | None,
    stratify_weight: str | None,
    stratify_bands: tuple[float, ...] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign splits at the unit level and broadcast to every row of the unit."""

    # One row per unit, with the unit's aggregated target/weight for strata.
    agg: dict[str, Any] = {}
    if stratify_target and stratify_target in frame.columns:
        agg[stratify_target] = (stratify_target, "sum")
    if stratify_weight and stratify_weight in frame.columns:
        agg[stratify_weight] = (stratify_weight, "sum")
    grouped = frame.groupby(unit_column, sort=True)
    units = grouped.agg(**agg).reset_index() if agg else pd.DataFrame({unit_column: list(grouped.groups)})

    unit_assign = pd.DataFrame({"unit": units[unit_column]})
    unit_assign["split_unit"] = unit_assign["unit"].map(lambda value: stable_unit(value, seed))
    strata = _compute_strata(units, stratify_target, stratify_weight, stratify_bands)
    if strata is None:
        unit_assign["split"] = unit_assign["split_unit"].map(lambda value: assign_split(value, ratios))
        split_method = "grouped_stable_hash"
    else:
        unit_assign["split_stratum"] = strata.to_numpy()
        unit_assign = unit_assign.rename(columns={"unit": "record_id"})
        unit_assign["split"] = _assign_stratified_splits(unit_assign, ratios)
        unit_assign = unit_assign.rename(columns={"record_id": "unit"})
        split_method = "grouped_target_stratified_hash"

    unit_to_split = dict(zip(unit_assign["unit"], unit_assign["split"]))
    split_frame = pd.DataFrame({"record_id": frame[id_column].to_numpy()})
    split_frame["split"] = frame[unit_column].map(unit_to_split).to_numpy()
    split_frame = split_frame.sort_values("record_id").reset_index(drop=True)

    counts = split_frame["split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    manifest = _split_manifest(seed, id_column, split_method, strata, ratios, counts, unit_column=unit_column)
    manifest["unit_count"] = int(len(unit_assign))
    return split_frame, manifest


def _split_manifest(seed, id_column, split_method, strata, ratios, counts, *, unit_column):
    manifest = {
        "split_pack_version": 2,
        "seed": seed,
        "id_column": id_column,
        "split_unit_column": unit_column or "record_id",
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
    return manifest


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


def _compute_strata(
    frame: pd.DataFrame,
    stratify_target: str | None,
    stratify_weight: str | None,
    stratify_bands: tuple[float, ...] | None,
) -> pd.Series | None:
    """Choose the stratification for a frame.

    The French claim/exposure auto-detection is preferred whenever those columns
    are present (reproducing the historical split byte-for-byte); otherwise a
    generic zero-band + target-quantile stratification keyed on the configured
    ``stratify_target`` (crossed with a weight quantile band when available).
    """

    legacy = _split_strata(frame)
    if legacy is not None and stratify_target in (None, "ClaimAmount", "ClaimAmountCapped"):
        return legacy
    if stratify_target and stratify_target in frame.columns:
        return _generic_strata(frame, stratify_target, stratify_weight, stratify_bands)
    return legacy


def _generic_strata(
    frame: pd.DataFrame,
    target_col: str,
    weight_col: str | None,
    bands: tuple[float, ...] | None,
) -> pd.Series:
    """Zero-band + quantile target strata, optionally crossed with a weight band."""

    target = frame[target_col].astype(float).clip(lower=0)
    if bands:
        edges = [-float("inf"), *[float(b) for b in bands], float("inf")]
        target_band = pd.cut(target, bins=edges, labels=False, include_lowest=True)
        target_band = target_band.fillna(-1).astype(int).map(lambda v: f"t_{v}")
    else:
        target_band = pd.Series("zero", index=frame.index, dtype="object")
        positive = target > 0
        if positive.any():
            pos_bands = _quantile_band(target[positive], 8, "tpos")
            target_band.loc[positive] = pos_bands
    if weight_col and weight_col in frame.columns and frame[weight_col].nunique(dropna=True) > 1:
        weight_band = _quantile_band(frame[weight_col].astype(float).clip(lower=0), 5, "w")
    else:
        weight_band = pd.Series("w_all", index=frame.index, dtype="object")
    return target_band.astype(str) + "|" + weight_band.astype(str)


def _split_strata(frame: pd.DataFrame) -> pd.Series | None:
    """Return deterministic target/exposure strata when the needed columns exist."""

    claim_column = _first_existing(frame, ("ClaimAmount", "ClaimAmountCapped"))
    exposure_column = _first_existing(frame, ("Exposure",))
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
