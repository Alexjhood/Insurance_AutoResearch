from pathlib import Path

import pandas as pd

from autoresearch.data.loader import load_fremtpl2


def test_load_fremtpl2_discovers_and_aggregates_csvs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame(
        {
            "IDpol": [1, 2],
            "ClaimNb": [1, 0],
            "Exposure": [0.5, 1.0],
            "VehPower": [5, 6],
        }
    ).to_csv(raw_dir / "freMTPL2freq.csv", index=False)
    pd.DataFrame(
        {
            "IDpol": [1, 1],
            "ClaimAmount": [100.0, 25.0],
        }
    ).to_csv(raw_dir / "freMTPL2sev.csv", index=False)

    loaded = load_fremtpl2(raw_dir)

    assert loaded.frequency_path == raw_dir / "freMTPL2freq.csv"
    assert loaded.severity_path == raw_dir / "freMTPL2sev.csv"
    assert loaded.frame.loc[loaded.frame["IDpol"] == 1, "ClaimAmount"].item() == 125.0
    assert loaded.frame.loc[loaded.frame["IDpol"] == 1, "ClaimAmountCount"].item() == 2
    assert loaded.frame.loc[loaded.frame["IDpol"] == 2, "ClaimAmount"].item() == 0.0
