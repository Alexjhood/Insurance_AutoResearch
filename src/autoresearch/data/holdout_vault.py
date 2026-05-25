"""Architecturally-separated milestone holdout vault.

The vault stores holdout rows in a dedicated directory that the ordinary
experiment runner never touches.  Access requires an explicit token so that
accidental reads fail loudly rather than silently.

Ordinary workflows read only ``agent_dataset_search.parquet`` via
``load_search_dataset()``.  Milestone evaluation reads the vault via
``load_holdout_dataset()`` and must supply the correct token.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_SENTINEL_FILENAME = ".locked"
_HOLDOUT_FILENAME = "agent_dataset_holdout.parquet"
_SEARCH_FILENAME = "agent_dataset_search.parquet"
_TOKEN_ENV_VAR = "AUTORESEARCH_MILESTONE_TOKEN"
_DEFAULT_TOKEN = "milestone"  # used only in tests; production should override via env


def write_vault(
    search_frame: pd.DataFrame,
    holdout_frame: pd.DataFrame,
    processed_dir: Path,
    holdout_vault_dir: Path,
) -> dict[str, Path]:
    """Write split datasets: search partition to processed/, holdout to vault."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    holdout_vault_dir.mkdir(parents=True, exist_ok=True)

    search_path = processed_dir / _SEARCH_FILENAME
    holdout_path = holdout_vault_dir / _HOLDOUT_FILENAME
    sentinel_path = holdout_vault_dir / _SENTINEL_FILENAME

    search_frame.to_parquet(search_path, index=False)
    holdout_frame.to_parquet(holdout_path, index=False)
    sentinel_path.write_text(
        "This directory contains the milestone holdout dataset.\n"
        "It must not be read by ordinary experiment workflows.\n"
        "Access requires AUTORESEARCH_MILESTONE_TOKEN environment variable.\n",
        encoding="utf-8",
    )
    return {"search": search_path, "holdout": holdout_path}


def load_search_dataset(processed_dir: Path, agent_dataset_name: str = "agent_dataset") -> pd.DataFrame:
    """Load the search-partition dataset (no holdout rows)."""

    search_path = processed_dir / _SEARCH_FILENAME
    if search_path.exists():
        return pd.read_parquet(search_path)
    # Fallback: legacy full dataset with holdout rows filtered out by caller.
    legacy_path = processed_dir / f"{agent_dataset_name}.parquet"
    if legacy_path.exists():
        return pd.read_parquet(legacy_path)
    raise FileNotFoundError(
        f"No search dataset found at {search_path}. Run `autoresearch prepare-data` first."
    )


def load_holdout_dataset(holdout_vault_dir: Path, *, milestone_token: str | None = None) -> pd.DataFrame:
    """Load the holdout dataset.  Requires the correct milestone token."""

    token = milestone_token or os.environ.get(_TOKEN_ENV_VAR) or _DEFAULT_TOKEN
    expected = os.environ.get(_TOKEN_ENV_VAR) or _DEFAULT_TOKEN
    if token != expected:
        raise PermissionError(
            "Incorrect milestone token. Set AUTORESEARCH_MILESTONE_TOKEN to access the holdout vault."
        )
    holdout_path = holdout_vault_dir / _HOLDOUT_FILENAME
    if not holdout_path.exists():
        raise FileNotFoundError(
            f"Holdout dataset not found at {holdout_path}. Run `autoresearch prepare-data` first."
        )
    return pd.read_parquet(holdout_path)
