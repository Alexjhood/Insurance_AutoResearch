from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "density_index_i" in frame.columns:
        frame["log1p_density"] = np.log1p(frame["density_index_i"].astype(float))
    return frame
