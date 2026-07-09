from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    # Log transforms
    frame["density_log"] = np.log1p(frame["Density"].astype(float))
    bm = frame["BonusMalus"].astype(float)
    bm_offset = bm - bm.min() + 1.0
    frame["bonus_malus_log"] = np.log1p(bm_offset)
    da = frame["DrivAge"].astype(float)
    vp = frame["VehPower"].astype(float)
    va = frame["VehAge"].astype(float)

    # Granular bins
    frame["bm_lt_80"] = (bm < 80).astype(float)
    frame["bm_80_to_100"] = ((bm >= 80) & (bm < 100)).astype(float)
    frame["bm_100_to_120"] = ((bm >= 100) & (bm < 120)).astype(float)
    frame["bm_120_to_150"] = ((bm >= 120) & (bm < 150)).astype(float)
    frame["bm_ge_150"] = (bm >= 150).astype(float)

    frame["driv_age_le_21"] = (da <= 21).astype(float)
    frame["driv_age_22_to_25"] = ((da > 21) & (da <= 25)).astype(float)
    frame["driv_age_26_to_35"] = ((da > 25) & (da <= 35)).astype(float)
    frame["driv_age_36_to_50"] = ((da > 35) & (da <= 50)).astype(float)
    frame["driv_age_51_to_65"] = ((da > 50) & (da <= 65)).astype(float)
    frame["driv_age_gt_65"] = (da > 65).astype(float)

    frame["veh_age_le_1"] = (va <= 1).astype(float)
    frame["veh_age_2_to_5"] = ((va > 1) & (va <= 5)).astype(float)
    frame["veh_age_6_to_10"] = ((va > 5) & (va <= 10)).astype(float)
    frame["veh_age_11_to_20"] = ((va > 10) & (va <= 20)).astype(float)
    frame["veh_age_gt_20"] = (va > 20).astype(float)

    frame["veh_power_le_4"] = (vp <= 4).astype(float)
    frame["veh_power_5_to_7"] = ((vp >= 5) & (vp <= 7)).astype(float)
    frame["veh_power_8_to_9"] = ((vp >= 8) & (vp <= 9)).astype(float)
    frame["veh_power_ge_10"] = (vp >= 10).astype(float)

    # Ratio features
    frame["bm_per_driv_age"] = bm / np.maximum(da, 1.0)
    frame["density_per_driv_age"] = frame["Density"].astype(float) / np.maximum(da, 1.0)
    frame["veh_power_per_age"] = vp / np.maximum(va + da, 1.0)

    # Polynomial features
    frame["driv_age_cubed"] = da ** 3
    frame["bonus_malus_cubed"] = bm ** 3
    frame["dens_log_sq"] = frame["density_log"] ** 2

    # Binned versions for interaction
    frame["da_bin"] = pd.cut(da, bins=[0, 25, 35, 50, 65, 120], labels=["young", "mid_young", "mid", "mid_old", "old"])
    frame["bm_bin"] = pd.cut(bm, bins=[0, 80, 100, 120, 150, 300], labels=["low", "mid_low", "mid", "mid_high", "high"])
    frame["dens_bin"] = pd.cut(frame["Density"].astype(float), bins=[0, 100, 500, 2000, 10000, 1e8], labels=["rural", "suburban", "town", "urban", "dense"])

    # Categorical x binned-continuous interactions
    for cat_col in ["VehBrand", "VehGas", "Area", "Region"]:
        if cat_col in frame.columns:
            for bin_col, suffix in [("da_bin", "age"), ("bm_bin", "bm"), ("dens_bin", "dens")]:
                if bin_col in frame.columns:
                    col_name = f"{cat_col}_x_{suffix}"
                    if col_name not in frame.columns:
                        frame[col_name] = frame[cat_col].astype(str) + "_" + frame[bin_col].astype(str)

    # Cross-category interactions
    if "VehBrand" in frame.columns and "VehGas" in frame.columns:
        frame["brand_x_gas"] = frame["VehBrand"].astype(str) + "_" + frame["VehGas"].astype(str)
    if "Area" in frame.columns and "Region" in frame.columns:
        frame["area_x_region"] = frame["Area"].astype(str) + "_" + frame["Region"].astype(str)

    # Drop helper bin columns (they are not useful as standalone features)
    for bc in ["da_bin", "bm_bin", "dens_bin"]:
        frame.drop(columns=[bc], inplace=True)

    return frame
