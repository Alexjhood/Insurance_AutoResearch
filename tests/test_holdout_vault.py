"""Tests for the milestone holdout vault: separation, access control, and no leakage."""

import numpy as np
import pandas as pd
import pytest

from autoresearch.data.holdout_vault import (
    load_holdout_dataset,
    load_search_dataset,
    write_vault,
)


def _make_full_dataset(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "record_id": np.arange(n),
        "split": (["train"] * 60 + ["search_validation"] * 20 + ["milestone_holdout"] * 20),
        "claim_cost_capped_active": rng.exponential(100, n),
        "exposure_term_a": np.ones(n),
    })


def test_write_vault_creates_both_files(tmp_path) -> None:
    full = _make_full_dataset()
    search = full[full["split"] != "milestone_holdout"].reset_index(drop=True)
    holdout = full[full["split"] == "milestone_holdout"].reset_index(drop=True)

    paths = write_vault(search, holdout, tmp_path / "processed", tmp_path / "vault")

    assert paths["search"].exists()
    assert paths["holdout"].exists()
    assert (tmp_path / "vault" / ".locked").exists()


def test_search_dataset_contains_no_holdout_record_ids(tmp_path) -> None:
    full = _make_full_dataset(n=100)
    holdout_ids = set(full.loc[full["split"] == "milestone_holdout", "record_id"])
    search = full[full["split"] != "milestone_holdout"].reset_index(drop=True)
    holdout = full[full["split"] == "milestone_holdout"].reset_index(drop=True)

    write_vault(search, holdout, tmp_path / "processed", tmp_path / "vault")
    loaded = load_search_dataset(tmp_path / "processed")

    assert set(loaded["record_id"]).isdisjoint(holdout_ids), (
        "Search dataset must not contain any holdout record IDs"
    )


def test_search_and_holdout_are_disjoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "milestone")
    full = _make_full_dataset(n=100)
    search = full[full["split"] != "milestone_holdout"].reset_index(drop=True)
    holdout = full[full["split"] == "milestone_holdout"].reset_index(drop=True)

    write_vault(search, holdout, tmp_path / "processed", tmp_path / "vault")
    loaded_search = load_search_dataset(tmp_path / "processed")
    loaded_holdout = load_holdout_dataset(tmp_path / "vault", milestone_token="milestone")

    search_ids = set(loaded_search["record_id"])
    holdout_ids = set(loaded_holdout["record_id"])
    assert search_ids.isdisjoint(holdout_ids)
    assert len(search_ids) + len(holdout_ids) == 100


def test_holdout_row_counts_are_correct(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "milestone")
    full = _make_full_dataset(n=100)
    search = full[full["split"] != "milestone_holdout"].reset_index(drop=True)
    holdout = full[full["split"] == "milestone_holdout"].reset_index(drop=True)

    write_vault(search, holdout, tmp_path / "processed", tmp_path / "vault")
    loaded_search = load_search_dataset(tmp_path / "processed")
    loaded_holdout = load_holdout_dataset(tmp_path / "vault", milestone_token="milestone")

    assert len(loaded_search) == 80
    assert len(loaded_holdout) == 20


def test_wrong_token_raises_permission_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "milestone")
    full = _make_full_dataset()
    search = full[full["split"] != "milestone_holdout"].reset_index(drop=True)
    holdout = full[full["split"] == "milestone_holdout"].reset_index(drop=True)

    write_vault(search, holdout, tmp_path / "processed", tmp_path / "vault")

    with pytest.raises(PermissionError):
        load_holdout_dataset(tmp_path / "vault", milestone_token="wrong_token")


def test_missing_token_env_fails_closed(tmp_path, monkeypatch) -> None:
    """With no token configured, holdout access is refused (no default fallback)."""
    monkeypatch.delenv("AUTORESEARCH_MILESTONE_TOKEN", raising=False)
    full = _make_full_dataset()
    search = full[full["split"] != "milestone_holdout"].reset_index(drop=True)
    holdout = full[full["split"] == "milestone_holdout"].reset_index(drop=True)

    write_vault(search, holdout, tmp_path / "processed", tmp_path / "vault")

    # Even though the holdout file exists, access fails without a configured token.
    with pytest.raises(PermissionError, match="AUTORESEARCH_MILESTONE_TOKEN"):
        load_holdout_dataset(tmp_path / "vault")


def test_load_search_dataset_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_search_dataset(tmp_path / "nonexistent")


def test_load_search_dataset_falls_back_to_legacy(tmp_path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    legacy = pd.DataFrame({"record_id": np.arange(50), "value": np.ones(50)})
    legacy.to_parquet(processed / "agent_dataset.parquet", index=False)

    loaded = load_search_dataset(processed, agent_dataset_name="agent_dataset")
    assert len(loaded) == 50
