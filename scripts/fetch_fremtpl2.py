"""Download real freMTPL2 frequency and severity data from OpenML."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUTPUT_DIR = Path("data/raw")
FREQ_FILE = OUTPUT_DIR / "freMTPL2freq.parquet"
SEV_FILE = OUTPUT_DIR / "freMTPL2sev.parquet"

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215
FREQ_URL = "https://www.openml.org/d/41214"
SEV_URL = "https://www.openml.org/d/41215"


def fetch(force: bool = False) -> None:
    from sklearn.datasets import fetch_openml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = [p for p in (FREQ_FILE, SEV_FILE) if p.exists()]
    if existing and not force:
        names = ", ".join(str(p) for p in existing)
        print(
            f"Files already exist: {names}\n"
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching freMTPL2freq from OpenML (id {FREQ_OPENML_ID})...")
    try:
        freq_bunch = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True)
    except Exception as exc:
        print(
            f"Failed to fetch freMTPL2freq: {exc}\n"
            f"If openml.org is unreachable, download manually from {FREQ_URL} "
            "and place the file in data/raw/.",
            file=sys.stderr,
        )
        sys.exit(1)

    freq_df = freq_bunch.frame
    print(f"  freMTPL2freq: {len(freq_df)} rows")
    print(f"Writing {FREQ_FILE}...")
    freq_df.to_parquet(FREQ_FILE, index=False)

    print(f"Fetching freMTPL2sev from OpenML (id {SEV_OPENML_ID})...")
    try:
        sev_bunch = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True)
    except Exception as exc:
        print(
            f"Failed to fetch freMTPL2sev: {exc}\n"
            f"If openml.org is unreachable, download manually from {SEV_URL} "
            "and place the file in data/raw/.",
            file=sys.stderr,
        )
        sys.exit(1)

    sev_df = sev_bunch.frame
    print(f"  freMTPL2sev: {len(sev_df)} rows")
    print(f"Writing {SEV_FILE}...")
    sev_df.to_parquet(SEV_FILE, index=False)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real freMTPL2 data from OpenML."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    fetch(force=args.force)


if __name__ == "__main__":
    main()
