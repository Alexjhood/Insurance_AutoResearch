"""Generate synthetic freMTPL2-style parquet files for smoke-testing and CI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260526
N_FREQ = 5000
OUTPUT_DIR = Path("data/raw")
FREQ_FILE = OUTPUT_DIR / "freMTPL2freq_synthetic.parquet"
SEV_FILE = OUTPUT_DIR / "freMTPL2sev_synthetic.parquet"

REGIONS = [
    "R11", "R21", "R22", "R23", "R24", "R25", "R26",
    "R31", "R41", "R42", "R43", "R52", "R53", "R54",
    "R72", "R73", "R74", "R82", "R83", "R91", "R93",
]
BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
AREAS = ["A", "B", "C", "D", "E", "F"]
GAS_TYPES = ["Regular", "Diesel"]


def generate(force: bool = False) -> None:
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

    rng = np.random.default_rng(seed=SEED)

    idpol = np.arange(1, N_FREQ + 1, dtype=np.int64)
    claim_nb = np.clip(rng.poisson(0.05, N_FREQ), 0, 4).astype(np.int64)
    exposure = rng.uniform(0.05, 1.0, N_FREQ)
    veh_power = rng.integers(4, 16, N_FREQ).astype(np.int64)   # 4..15 inclusive
    veh_age = rng.integers(0, 21, N_FREQ).astype(np.int64)     # 0..20 inclusive
    driv_age = rng.integers(18, 91, N_FREQ).astype(np.int64)   # 18..90 inclusive
    bonus_malus = rng.integers(50, 151, N_FREQ).astype(np.int64)  # 50..150 inclusive
    veh_brand = rng.choice(BRANDS, N_FREQ)
    veh_gas = rng.choice(GAS_TYPES, N_FREQ)
    area = rng.choice(AREAS, N_FREQ)
    log_density = rng.uniform(np.log(1), np.log(30000), N_FREQ)
    density = np.clip(np.exp(log_density).astype(np.int64), 1, 30000)
    region = rng.choice(REGIONS, N_FREQ)

    freq_df = pd.DataFrame({
        "IDpol": idpol,
        "ClaimNb": claim_nb,
        "Exposure": exposure,
        "VehPower": veh_power,
        "VehAge": veh_age,
        "DrivAge": driv_age,
        "BonusMalus": bonus_malus,
        "VehBrand": veh_brand,
        "VehGas": veh_gas,
        "Area": area,
        "Density": density,
        "Region": region,
    })

    claim_idpols = np.repeat(idpol, claim_nb)
    n_sev = int(len(claim_idpols))
    # lognormal: E[X] = exp(mu + sigma^2/2) = 2000 with sigma=1.5
    sigma = 1.5
    mu = float(np.log(2000) - 0.5 * sigma ** 2)
    claim_amounts = rng.lognormal(mean=mu, sigma=sigma, size=n_sev)

    sev_df = pd.DataFrame({
        "IDpol": claim_idpols.astype(np.int64),
        "ClaimAmount": claim_amounts,
    })

    print(f"Writing {FREQ_FILE} ({N_FREQ} rows)...")
    freq_df.to_parquet(FREQ_FILE, index=False)
    print(f"Writing {SEV_FILE} ({n_sev} rows)...")
    sev_df.to_parquet(SEV_FILE, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic freMTPL2-style parquet files for smoke testing."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()
    generate(force=args.force)


if __name__ == "__main__":
    main()
