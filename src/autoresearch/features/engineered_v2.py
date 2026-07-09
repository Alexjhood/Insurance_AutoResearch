from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # Log transforms for skewed continuous variables
    frame["density_log"] = np.log1p(frame["Density"].astype(float))
    bm = frame["BonusMalus"].astype(float)
    bm_offset = bm - bm.min() + 1.0
    frame["bonus_malus_log"] = np.log1p(bm_offset)

    # BonusMalus binned categories (known nonlinear risk relationship)
    frame["bm_under_100"] = (bm <= 100).astype(float)
    frame["bm_100_to_120"] = ((bm > 100) & (bm <= 120)).astype(float)
    frame["bm_over_120"] = (bm > 120).astype(float)

    # Vehicle age bins
    va = frame["VehAge"].astype(float)
    frame["veh_age_new"] = (va <= 1).astype(float)
    frame["veh_age_mid"] = ((va > 1) & (va <= 10)).astype(float)
    frame["veh_age_old"] = (va > 10).astype(float)

    # Driver age bins
    da = frame["DrivAge"].astype(float)
    frame["driv_age_young"] = (da <= 25).astype(float)
    frame["driv_age_mid"] = ((da > 25) & (da <= 55)).astype(float)
    frame["driv_age_senior"] = (da > 55).astype(float)

    # Vehicle power bins
    vp = frame["VehPower"].astype(float)
    frame["veh_power_low"] = (vp < 6).astype(float)
    frame["veh_power_mid"] = ((vp >= 6) & (vp < 9)).astype(float)
    frame["veh_power_high"] = (vp >= 9).astype(float)

    # Density bins
    dens = frame["Density"].astype(float)
    frame["density_rural"] = (dens <= 100).astype(float)
    frame["density_suburban"] = ((dens > 100) & (dens <= 1000)).astype(float)
    frame["density_urban"] = ((dens > 1000) & (dens <= 10000)).astype(float)
    frame["density_dense_urban"] = (dens > 10000).astype(float)

    # Key interactions
    frame["driv_age_x_bonus_malus"] = da * bm
    frame["driv_age_x_density_log"] = da * frame["density_log"]
    frame["bonus_malus_x_density_log"] = bm * frame["density_log"]
    frame["driv_age_x_veh_power"] = da * vp
    frame["veh_age_x_veh_power"] = va * vp

    # Squared terms for nonlinear effects
    frame["driv_age_sq"] = da ** 2
    frame["bonus_malus_sq"] = bm ** 2
    frame["veh_age_sq"] = va ** 2

    return frame
