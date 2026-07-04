#!/usr/bin/env python3
"""Benchmark TabPFN context sizes on the real dataset to pick run defaults.

Run once per machine (M2 Pro first) to choose ``max_context_rows`` and the
predict batch size that fit the per-experiment compute budget, then set them as
the defaults in ``foundation.py`` / document the per-machine override.

    python scripts/benchmark_foundation.py                       # sweep on best device
    python scripts/benchmark_foundation.py --context 10000 40000 # custom grid
    python scripts/benchmark_foundation.py --device cpu          # force device

For each context size it times a single fit on an exposure-weighted subsample of
the search split and a full batched predict over the whole search frame, and
reports peak process memory. The comparison refits ~5× (1 fit + 4 CV folds), up
to ~13× when the gate escalates, so a single fit should land near ~1/5 of the
budget (``10 + 5 × (N // 5)`` minutes).
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SEARCH_PARQUET = REPO / "data" / "processed" / "agent_dataset_search.parquet"

FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
            "VehBrand", "VehGas", "Area", "Region"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]


def _peak_mem_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _encode(df: pd.DataFrame) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    ct = ColumnTransformer(
        [("num", "passthrough", numeric),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="drop",
    )
    return ct.fit_transform(df[FEATURES]).astype(float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, nargs="+", default=[10_000, 20_000, 40_000, 60_000])
    parser.add_argument("--device", default=None, help="Force device (mps/cpu/cuda).")
    parser.add_argument("--batch", type=int, default=50_000, help="Predict batch size.")
    args = parser.parse_args()

    if not SEARCH_PARQUET.exists():
        print(f"Missing {SEARCH_PARQUET}; run prepare-data first.")
        return 1

    from autoresearch.models.recipe.foundation import _select_device, subsample_context
    from tabpfn import TabPFNRegressor

    device = args.device or _select_device()
    df = pd.read_parquet(SEARCH_PARQUET)
    exposure = df["Exposure"].astype(float).to_numpy()
    y = df["ClaimAmount"].astype(float).to_numpy() / np.clip(exposure, 1e-12, None)
    X = _encode(df)
    print(f"dataset: {X.shape[0]:,} rows × {X.shape[1]} cols | device={device} | batch={args.batch:,}")
    print(f"{'context':>10} {'fit_s':>8} {'predict_s':>10} {'total_s':>9} {'peak_MB':>9}")

    for ctx_rows in args.context:
        Xc, yc, _, n = subsample_context(X, y, exposure, ctx_rows, "exposure", seed=42)
        reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True, random_state=42)
        t0 = time.perf_counter()
        reg.fit(Xc, yc)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = np.concatenate([
            reg.predict(X[s : s + args.batch]) for s in range(0, X.shape[0], args.batch)
        ])
        predict_s = time.perf_counter() - t1
        assert preds.shape[0] == X.shape[0]
        print(f"{n:>10,} {fit_s:>8.1f} {predict_s:>10.1f} {fit_s + predict_s:>9.1f} {_peak_mem_mb():>9.0f}")

    print("\nPick the largest context whose single fit+predict is ≲ budget/5 with memory headroom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
