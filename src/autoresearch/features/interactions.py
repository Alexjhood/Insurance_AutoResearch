from __future__ import annotations

import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if all(c in frame.columns for c in ["risk_score_index_e", "density_index_i"]):
        frame["risk_x_density"] = frame["risk_score_index_e"].astype(float) * frame["density_index_i"].astype(float)

    if all(c in frame.columns for c in ["driver_age_band_d", "vehicle_power_band_b"]):
        frame["age_x_power"] = frame["driver_age_band_d"].astype(float) * frame["vehicle_power_band_b"].astype(float)

    if all(c in frame.columns for c in ["driver_age_band_d", "density_index_i"]):
        frame["age_x_density"] = frame["driver_age_band_d"].astype(float) * frame["density_index_i"].astype(float)

    if all(c in frame.columns for c in ["vehicle_make_group_f", "vehicle_age_band_c"]):
        frame["make_x_veh_age"] = frame["vehicle_make_group_f"].astype(str) + "_x_" + frame["vehicle_age_band_c"].astype(str)

    return frame
