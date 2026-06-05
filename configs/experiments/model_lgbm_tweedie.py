from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from autoresearch.models.calibration import apply_training_calibration

EXPOSURE = "exposure_term_a"
CLAIM_COST = "claim_cost_capped_active"

NUMERIC = [
    "vehicle_power_band_b", "vehicle_age_band_c", "driver_age_band_d",
    "risk_score_index_e", "density_index_i",
]
CATEGORICAL = [
    "vehicle_make_group_f", "vehicle_energy_type_g",
    "territory_band_h", "region_cluster_j",
]


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters):
    exp_tr = train[EXPOSURE].astype(float).to_numpy()
    exp_sc = score[EXPOSURE].astype(float).to_numpy()
    y_rate = train[CLAIM_COST].astype(float).to_numpy() / np.clip(exp_tr, 1e-9, None)

    le_dict = {}
    train_enc = train[NUMERIC + CATEGORICAL].copy()
    score_enc = score[NUMERIC + CATEGORICAL].copy()
    for col in CATEGORICAL:
        le = LabelEncoder()
        combined = pd.concat([train_enc[col], score_enc[col]]).astype(str)
        le.fit(combined)
        train_enc[col] = le.transform(train_enc[col].astype(str))
        score_enc[col] = le.transform(score_enc[col].astype(str))
        le_dict[col] = le

    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=500, learning_rate=0.05, max_depth=5,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(train_enc, y_rate, sample_weight=exp_tr)

    rate_tr = np.clip(model.predict(train_enc), 0.0, None)
    rate_sc = np.clip(model.predict(score_enc), 0.0, None)
    pred_tr = rate_tr * exp_tr
    pred_sc = rate_sc * exp_sc

    pred_sc, calib = apply_training_calibration(pred_sc, pred_tr, train[CLAIM_COST].to_numpy())
    notes = {
        "model_family": "lgbm_tweedie",
        "n_estimators": 500,
        "max_depth": 5,
        "calib_factor": round(float(calib), 4),
    }
    return pred_sc, notes
