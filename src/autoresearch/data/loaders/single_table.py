"""Generic single-table loader.

Reads one labelled training table (CSV or parquet) from a dataset's ``raw_dir``,
applies missing-value handling, evaluates declarative derived columns, and
returns a uniform :class:`~autoresearch.data.sources.RawDataset`.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.data.sources import RawDataset

_TABLE_SUFFIXES = {".csv", ".parquet", ".pq"}


def _find_train_file(raw_dir: Path, pattern: str | None) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    glob_pattern = pattern or "*"
    candidates: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TABLE_SUFFIXES:
            continue
        if fnmatch.fnmatch(path.name, glob_pattern) or fnmatch.fnmatch(path.stem, glob_pattern):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No table matching {glob_pattern!r} found under {raw_dir}"
        )
    # Prefer real data over synthetic, then the largest file (the training table).
    def _key(p: Path) -> tuple[int, int]:
        synthetic = 1 if "synthetic" in p.name.lower() else 0
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        return (synthetic, -size)

    return sorted(candidates, key=_key)[0]


def _read_table(path: Path, na_values: tuple[str, ...], nrows: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, na_values=list(na_values) or None, nrows=nrows)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
        return frame.head(nrows) if nrows is not None else frame
    raise ValueError(f"Unsupported table format: {path}")


def _apply_na_marker(frame: pd.DataFrame, na_marker: Any) -> pd.DataFrame:
    if na_marker is None:
        return frame
    import re

    pattern = na_marker.columns_matching
    for col in frame.columns:
        if pattern is not None and not re.search(pattern, col):
            continue
        frame[col] = frame[col].where(frame[col] != na_marker.value)
    return frame


def _apply_derived(frame: pd.DataFrame, derived: tuple[Any, ...]) -> pd.DataFrame:
    for col in derived:
        # Restricted evaluation over existing columns only (no arbitrary Python).
        frame[col.name] = frame.eval(col.expr, engine="python")
    return frame


def load(spec: Any, *, nrows: int | None = None) -> RawDataset:
    path = _find_train_file(spec.raw_dir, spec.train_file_pattern)
    frame = _read_table(path, spec.na_values, nrows)
    if spec.id_column not in frame.columns:
        raise ValueError(
            f"Training table {path} does not contain id column {spec.id_column!r}"
        )
    frame = _apply_na_marker(frame, spec.na_marker)
    frame = _apply_derived(frame, spec.derived_columns)
    return RawDataset(
        frame=frame,
        provenance={
            "loader": "single_table",
            "train_file": str(path),
            "nrows": nrows,
            "na_values": list(spec.na_values),
        },
    )
