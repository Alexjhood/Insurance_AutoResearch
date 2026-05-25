"""Identity feature builder — pass-through, no transforms.

Use as a template when creating new feature engineering modules.

Entry point contract (must be satisfied by all feature builders):

    def build_features(frame: pd.DataFrame) -> pd.DataFrame

Requirements:
- Accept the full modelling frame (all columns intact).
- Return a DataFrame with at least all original columns present.
- May add new columns; must not drop or rename existing ones relied upon
  by the model (target columns, exposure, record_id).
- Must be deterministic and free of I/O or global state.
- Must not access or reference the holdout vault in any way.
"""

from __future__ import annotations

import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame unchanged (no-op baseline)."""
    return frame
