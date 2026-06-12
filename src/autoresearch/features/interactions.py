from __future__ import annotations

import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if all(c in frame.columns for c in ["BonusMalus", "Density"]):
        frame["risk_x_density"] = frame["BonusMalus"].astype(float) * frame["Density"].astype(float)

    if all(c in frame.columns for c in ["DrivAge", "VehPower"]):
        frame["age_x_power"] = frame["DrivAge"].astype(float) * frame["VehPower"].astype(float)

    if all(c in frame.columns for c in ["DrivAge", "Density"]):
        frame["age_x_density"] = frame["DrivAge"].astype(float) * frame["Density"].astype(float)

    if all(c in frame.columns for c in ["VehBrand", "VehAge"]):
        frame["make_x_veh_age"] = frame["VehBrand"].astype(str) + "_x_" + frame["VehAge"].astype(str)

    return frame
