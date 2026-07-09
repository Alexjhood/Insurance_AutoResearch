"""Phase 3 — generic pipeline on three synthetic dataset archetypes.

Archetypes:
  (a) two-file exposure + frequency/severity (French-like, adapter loader)
  (b) single-table amount + household grouping + sampling (AllState-like)
  (c) single-table binary target with -1 missing markers (Porto-like)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoresearch.config import load_config
from autoresearch.data.pipeline import prepare_data
from autoresearch.datasets import SampleSpec, load_dataset_spec


def _config_for(spec, tmp_path: Path):
    """Build a ProjectConfig whose data paths live under tmp_path for `spec`."""

    base = load_config()
    data_dir = tmp_path / "data"
    return replace(
        base,
        dataset=spec,
        raw_data_dir=spec.raw_dir,
        processed_dir=data_dir / "processed",
        metadata_dir=data_dir / "metadata",
        splits_dir=data_dir / "splits",
        holdout_vault_dir=data_dir / "holdout_vault",
        artifacts_dir=tmp_path / "artifacts",
        handoff_base_dir=tmp_path / "auto_research",
        handoff_context_dir=tmp_path / "auto_research" / "context",
        handoff_proposal_inbox_dir=tmp_path / "auto_research" / "inbox",
        handoff_proposal_processed_dir=tmp_path / "auto_research" / "processed",
        handoff_results_dir=tmp_path / "auto_research" / "results",
        handoff_handoffs_dir=tmp_path / "auto_research" / "handoffs",
        research_log_path=tmp_path / "RESEARCH_LOG.md",
        registry_path=tmp_path / "artifacts" / "registry.sqlite",
        id_column=spec.id_column,
        claim_capping_enabled=spec.cap is not None,
        claim_cap_threshold=spec.cap.threshold if spec.cap else 100000,
        cv_folds=3,
        track_id="test",
    )


# --- (a) French-like: two-file freq/sev adapter ----------------------------


def test_archetype_french_adapter(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(0)
    n = 400
    pd.DataFrame(
        {
            "IDpol": range(1, n + 1),
            "ClaimNb": rng.integers(0, 2, n),
            "Exposure": rng.uniform(0.2, 1.0, n).round(3),
            "VehPower": rng.integers(4, 12, n),
            "Area": rng.choice(list("ABCDE"), n),
        }
    ).to_csv(raw / "freMTPL2freq.csv", index=False)
    claim_ids = rng.choice(range(1, n + 1), 60, replace=True)
    pd.DataFrame({"IDpol": claim_ids, "ClaimAmount": rng.uniform(100, 5000, 60)}).to_csv(
        raw / "freMTPL2sev.csv", index=False
    )

    spec = replace(load_dataset_spec("french_motor"), raw_dir=raw)
    config = _config_for(spec, tmp_path)
    outputs = prepare_data(config)

    frame = pd.read_parquet(outputs["processed_dataset"])
    assert "record_id" in frame.columns
    assert "Exposure" in frame.columns  # real weight column, not synthesised
    assert "unit_weight" not in frame.columns
    split = pd.read_csv(outputs["split_pack"])
    assert set(split["split"]) <= {"train", "search_validation", "milestone_holdout"}


# --- (b) AllState-like: grouping + sampling --------------------------------


def _write_grouped(raw: Path, n_households: int = 300, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for hh in range(1, n_households + 1):
        for _ in range(int(rng.integers(1, 4))):
            rid += 1
            amount = 0.0 if rng.random() > 0.1 else float(rng.uniform(50, 3000))
            rows.append(
                {
                    "Row_ID": rid,
                    "Household_ID": hh,
                    "Cat1": rng.choice(["A", "B", "?"]),
                    "Var1": float(rng.normal()),
                    "Claim_Amount": amount,
                }
            )
    pd.DataFrame(rows).to_csv(raw / "train_set.csv", index=False)


def test_archetype_grouped_sampling_and_leakage(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_grouped(raw)

    spec = replace(
        load_dataset_spec("allstate"),
        raw_dir=raw,
        sample=SampleSpec(rows=400, unit_column="Household_ID", stratify_column="Claim_Amount", seed=7),
    )
    config = _config_for(spec, tmp_path)
    outputs = prepare_data(config)

    frame = pd.read_parquet(outputs["processed_dataset"])
    assert "unit_weight" in frame.columns
    assert (frame["unit_weight"] == 1.0).all()
    assert "claim_occurred" in frame.columns

    # Group-aware leakage: no household straddles splits.
    split = pd.read_csv(outputs["split_pack"])
    merged = frame[["record_id", "Household_ID"]].merge(split, on="record_id")
    per_hh_splits = merged.groupby("Household_ID")["split"].nunique()
    assert (per_hh_splits == 1).all(), "a household straddled a split boundary"

    # Folds are group-aware too.
    folds = pd.read_parquet(outputs["fold_assignments"])
    fmerged = frame[["record_id", "Household_ID"]].merge(folds, on="record_id")
    assert (fmerged.groupby("Household_ID")["fold"].nunique() == 1).all()

    # Sampling determinism.
    config2 = _config_for(spec, tmp_path / "again")
    outputs2 = prepare_data(config2)
    frame2 = pd.read_parquet(outputs2["processed_dataset"])
    assert sorted(frame["record_id"]) == sorted(frame2["record_id"])
    assert len(frame) <= 500  # near the 400-row budget, whole households


# --- (c) Porto-like: binary target, -1 markers -----------------------------


def test_archetype_binary_target(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(3)
    n = 500
    pd.DataFrame(
        {
            "id": range(n),
            "target": rng.integers(0, 2, n),
            "ps_ind_01": rng.integers(0, 5, n),
            "ps_car_03_cat": rng.choice([-1, 0, 1, 2], n),  # -1 = missing marker
            "ps_reg_01": rng.uniform(0, 1, n).round(2),
        }
    ).to_csv(raw / "train.csv", index=False)

    spec = replace(load_dataset_spec("porto_seguro"), raw_dir=raw)
    config = _config_for(spec, tmp_path)
    outputs = prepare_data(config)

    frame = pd.read_parquet(outputs["processed_dataset"])
    assert "unit_weight" in frame.columns
    assert "target" in frame.columns
    # -1 recoded to NaN only in *_cat columns.
    assert frame["ps_car_03_cat"].isna().any()
    assert not (frame["ps_car_03_cat"] == -1).any()

    split = pd.read_csv(outputs["split_pack"])
    # Stratified on the binary target — both classes present in each split.
    merged = frame[["record_id", "target"]].merge(split, on="record_id")
    for _, grp in merged.groupby("split"):
        assert grp["target"].nunique() == 2
