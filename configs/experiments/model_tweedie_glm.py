from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    power = float(hyperparameters.get("tweedie_power", 1.5))
    alpha = float(hyperparameters.get("alpha", 1.0))

    exp_tr = train[EXPOSURE].astype(float).to_numpy()
    exp_sc = score[EXPOSURE].astype(float).to_numpy()
    y_rate = train[CLAIM_COST].astype(float).to_numpy() / np.clip(exp_tr, 1e-9, None)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    model = Pipeline([
        ("pre", pre),
        ("glm", TweedieRegressor(power=power, alpha=alpha, max_iter=1000)),
    ])
    model.fit(train[NUMERIC + CATEGORICAL], y_rate, glm__sample_weight=exp_tr)

    rate_tr = np.clip(model.predict(train[NUMERIC + CATEGORICAL]), 0.0, None)
    rate_sc = np.clip(model.predict(score[NUMERIC + CATEGORICAL]), 0.0, None)
    pred_tr = rate_tr * exp_tr
    pred_sc = rate_sc * exp_sc

    pred_sc, calib = apply_training_calibration(pred_sc, pred_tr, train[CLAIM_COST].to_numpy())
    notes = {
        "model_family": "tweedie_glm",
        "tweedie_power": power,
        "alpha": alpha,
        "calib_factor": round(float(calib), 4),
    }
    return pred_sc, notes
