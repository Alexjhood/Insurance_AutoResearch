#!/usr/bin/env python3
"""Reproduce TabPFN API batched prediction with faulthandler enabled.

This is a diagnostic harness only. It avoids the experiment registry and holdout,
uses the ordinary search train/search_validation split, and exercises the same
TabPFN API estimator + _BatchedRegressor path used by recipe experiments.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

faulthandler.enable(file=sys.stderr, all_threads=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from autoresearch.data.holdout_vault import load_search_dataset
from autoresearch.data.preprocessing import apply_claim_capping
from autoresearch.models.recipe.foundation import _BatchedRegressor, _authenticate_api


FEATURES = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Area",
    "Density",
    "Region",
]


def _ordinalize(train: pd.DataFrame, score: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_train = train[FEATURES].copy()
    x_score = score[FEATURES].copy()
    for col in FEATURES:
        if x_train[col].dtype == "object" or str(x_train[col].dtype).startswith("category"):
            cats = pd.Index(pd.concat([x_train[col], x_score[col]], ignore_index=True).astype("string").unique())
            mapping = {value: idx for idx, value in enumerate(cats)}
            x_train[col] = x_train[col].astype("string").map(mapping).fillna(-1).astype(float)
            x_score[col] = x_score[col].astype("string").map(mapping).fillna(-1).astype(float)
        else:
            x_train[col] = pd.to_numeric(x_train[col], errors="coerce").fillna(0.0).astype(float)
            x_score[col] = pd.to_numeric(x_score[col], errors="coerce").fillna(0.0).astype(float)
    return x_train.to_numpy(dtype=float), x_score.to_numpy(dtype=float)


def _exposure_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    rng = np.random.default_rng(seed)
    weights = np.clip(frame["Exposure"].astype(float).to_numpy(), 0.0, None)
    total = float(weights.sum())
    p = weights / total if total > 0 else None
    idx = rng.choice(len(frame), size=n, replace=False, p=p)
    idx.sort()
    return frame.iloc[idx].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-rows", type=int, default=1000)
    parser.add_argument("--score-rows", type=int, default=120000)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        {
            "python": sys.version,
            "executable": sys.executable,
            "context_rows": args.context_rows,
            "score_rows": args.score_rows,
            "batch_size": args.batch_size,
            "repeat": args.repeat,
        },
        flush=True,
    )

    if not os.environ.get("TABPFN_TOKEN"):
        raise RuntimeError("TABPFN_TOKEN is not set")

    _authenticate_api()
    from tabpfn_client import TabPFNRegressor

    frame = load_search_dataset(Path(ROOT) / "data" / "processed", "freMTPL2freq")
    frame, _ = apply_claim_capping(
        frame,
        claim_column="ClaimAmount",
        threshold=100000.0,
        enabled=True,
    )
    splits = pd.read_csv(os.path.join(ROOT, "data", "splits", "split_pack.csv"))
    frame = frame.merge(splits[["record_id", "split"]], on="record_id", how="inner")
    train_full = frame[frame["split"] == "train"].copy()
    score_full = frame[frame["split"].isin(["train", "search_validation"])].copy()

    train = _exposure_sample(train_full, args.context_rows, args.seed)
    score = score_full.head(args.score_rows).copy()
    y = train["ClaimAmountCapped"].astype(float).to_numpy() / np.clip(
        train["Exposure"].astype(float).to_numpy(), 1e-12, None
    )
    x_train, x_score = _ordinalize(train, score)

    print(
        {
            "x_train_shape": x_train.shape,
            "x_score_shape": x_score.shape,
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
            "y_mean": float(np.mean(y)),
        },
        flush=True,
    )

    t0 = time.perf_counter()
    model = TabPFNRegressor(n_estimators=1)
    print("fit:start", flush=True)
    model.fit(x_train, y)
    print({"fit_seconds": round(time.perf_counter() - t0, 3)}, flush=True)

    wrapped = _BatchedRegressor(model, batch_size=args.batch_size)
    for i in range(args.repeat):
        t1 = time.perf_counter()
        print({"predict_start": i + 1}, flush=True)
        pred = wrapped.predict(x_score)
        print(
            {
                "predict_done": i + 1,
                "seconds": round(time.perf_counter() - t1, 3),
                "n": int(len(pred)),
                "finite": bool(np.isfinite(pred).all()),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "mean": float(np.mean(pred)),
            },
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
