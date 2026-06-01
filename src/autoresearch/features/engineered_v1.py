from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # Log-transform skewed continuous features
    frame["density_log"] = np.log1p(frame["density_index_i"].astype(float))
    frame["risk_score_log"] = np.log1p(frame["risk_score_index_e"].astype(float) - frame["risk_score_index_e"].astype(float).min() + 1.0)
    frame["density_risk_interaction"] = frame["density_log"] * frame["risk_score_log"]

    # Age-band interactions
    frame["driver_vehicle_age_interaction"] = (
        frame["driver_age_band_d"].astype(float) * frame["vehicle_age_band_c"].astype(float)
    )
    frame["driver_power_interaction"] = (
        frame["driver_age_band_d"].astype(float) * frame["vehicle_power_band_b"].astype(float)
    )

    # Risk score squared (capture non-linear risk effects)
    frame["risk_score_sq"] = frame["risk_score_index_e"].astype(float) ** 2

    # Density bins
    dens = frame["density_index_i"].astype(float)
    frame["density_bin_low"] = (dens <= 100).astype(float)
    frame["density_bin_medium"] = ((dens > 100) & (dens <= 1000)).astype(float)
    frame["density_bin_high"] = (dens > 1000).astype(float)

    return frame
