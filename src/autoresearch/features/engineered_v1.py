from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # Log-transform skewed continuous features
    frame["density_log"] = np.log1p(frame["Density"].astype(float))
    frame["risk_score_log"] = np.log1p(frame["BonusMalus"].astype(float) - frame["BonusMalus"].astype(float).min() + 1.0)
    frame["density_risk_interaction"] = frame["density_log"] * frame["risk_score_log"]

    # Age-band interactions
    frame["driver_vehicle_age_interaction"] = (
        frame["DrivAge"].astype(float) * frame["VehAge"].astype(float)
    )
    frame["driver_power_interaction"] = (
        frame["DrivAge"].astype(float) * frame["VehPower"].astype(float)
    )

    # Risk score squared (capture non-linear risk effects)
    frame["risk_score_sq"] = frame["BonusMalus"].astype(float) ** 2

    # Density bins
    dens = frame["Density"].astype(float)
    frame["density_bin_low"] = (dens <= 100).astype(float)
    frame["density_bin_medium"] = ((dens > 100) & (dens <= 1000)).astype(float)
    frame["density_bin_high"] = (dens > 1000).astype(float)

    return frame
