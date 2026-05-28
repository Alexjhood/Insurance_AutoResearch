"""Feature eligibility policy shared by handoffs, validators, and runners."""
from __future__ import annotations

from collections.abc import Iterable


EXPOSURE_COLUMN = "exposure_term_a"
NON_PREDICTIVE_COLUMNS = frozenset({EXPOSURE_COLUMN})


def is_predictive_feature(column: str, role: str | None = None) -> bool:
    """Return whether a column may be used as a model predictor."""

    if column in NON_PREDICTIVE_COLUMNS:
        return False
    return role not in {"target_or_outcome", "record_id", "exposure_offset"}


def predictive_columns(columns: Iterable[dict]) -> list[str]:
    """Extract agent-facing columns that are eligible as predictors."""

    return [
        item["name"]
        for item in columns
        if is_predictive_feature(item["name"], item.get("role"))
    ]
