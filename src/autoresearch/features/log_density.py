"""Log-density feature builder.

Adds log1p(density_index_i) as an additional numeric feature. Urban density
has a right-skewed distribution; the log transform linearises this for GBMs
and also benefits GLMs operating on linear predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["log_density"] = np.log1p(frame["density_index_i"])
    return frame
