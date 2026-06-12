from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "Density" in frame.columns:
        frame["log1p_density"] = np.log1p(frame["Density"].astype(float))
    return frame
