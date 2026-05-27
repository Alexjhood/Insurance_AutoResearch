"""Raw freMTPL2 dataset loading.

The loader discovers local frequency and severity files recursively under
``data/raw``. It supports CSV and parquet inputs and returns one modelling
table keyed by policy id with aggregated claim amount columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RawDataset:
    """Loaded raw dataset plus provenance needed for metadata."""

    frame: pd.DataFrame
    frequency_path: Path
    severity_path: Path | None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _find_one(raw_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            continue
        lower_name = path.name.lower()
        if any(pattern in lower_name for pattern in patterns):
            candidates.append(path)
    if not candidates:
        return None
    # Prefer real data over synthetic smoke-test data even if the latter sits at
    # a shorter path. ``generate_synthetic_data.py`` writes ``*_synthetic.parquet``
    # files which would otherwise tie or beat the real CSVs on path length.
    def _is_synthetic(p: Path) -> bool:
        return "synthetic" in p.name.lower() or "_synthetic" in {part.lower() for part in p.parts}

    return sorted(candidates, key=lambda p: (_is_synthetic(p), len(p.parts), str(p)))[0]


def discover_fremtpl2_files(raw_dir: Path) -> tuple[Path, Path | None]:
    """Find likely frequency and severity files under a raw data directory."""

    freq_path = _find_one(raw_dir, ("fremtpl2freq", "freq"))
    sev_path = _find_one(raw_dir, ("fremtpl2sev", "sev"))
    if freq_path is None:
        raise FileNotFoundError(f"Could not find a freMTPL2 frequency file under {raw_dir}")
    return freq_path, sev_path


def load_fremtpl2(raw_dir: Path, id_column: str = "IDpol") -> RawDataset:
    """Load and join local freMTPL2 frequency/severity files."""

    freq_path, sev_path = discover_fremtpl2_files(raw_dir)
    freq = _read_table(freq_path)
    if id_column not in freq.columns:
        raise ValueError(f"Frequency file {freq_path} does not contain id column {id_column!r}")

    frame = freq.copy()
    if sev_path is not None:
        sev = _read_table(sev_path)
        required = {id_column, "ClaimAmount"}
        missing = required.difference(sev.columns)
        if missing:
            raise ValueError(f"Severity file {sev_path} is missing columns: {sorted(missing)}")
        sev_agg = (
            sev.groupby(id_column, as_index=False)
            .agg(ClaimAmount=("ClaimAmount", "sum"), ClaimAmountCount=("ClaimAmount", "size"))
        )
        frame = frame.merge(sev_agg, on=id_column, how="left")
    else:
        frame["ClaimAmount"] = 0.0
        frame["ClaimAmountCount"] = 0

    frame["ClaimAmount"] = frame["ClaimAmount"].fillna(0.0)
    frame["ClaimAmountCount"] = frame["ClaimAmountCount"].fillna(0).astype("int64")
    return RawDataset(frame=frame, frequency_path=freq_path, severity_path=sev_path)
