"""Feature eligibility policy shared by handoffs, validators, and runners.

The set of non-predictive columns (weight/offset, ids, record id, and any the
dataset explicitly marks) is dataset-specific and bound at config load via
:func:`bind`. The French defaults are retained so unbound imports keep working.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


# French defaults — overwritten by bind() at config load.
EXPOSURE_COLUMN = "Exposure"
_DEFAULT_NON_PREDICTIVE = frozenset({"Exposure", "record_id", "IDpol"})
NON_PREDICTIVE_COLUMNS = _DEFAULT_NON_PREDICTIVE


def non_predictive_columns(dataset_spec: Any) -> frozenset[str]:
    """Return the non-predictive columns for a dataset spec.

    The dataset's declared ``non_predictive`` set, plus ``record_id``, the id
    column, and the weight/offset column (never a feature).
    """

    from autoresearch.targets import UNIT_WEIGHT_COLUMN

    cols: set[str] = set(getattr(dataset_spec, "non_predictive", ()) or ())
    cols.add("record_id")
    cols.add(dataset_spec.id_column)
    # The weight/offset column (a real one, or the synthesised unit weight).
    cols.add(dataset_spec.weight_column or UNIT_WEIGHT_COLUMN)
    return frozenset(cols)


def bind(dataset_spec: Any) -> None:
    """Bind the active dataset's non-predictive column set."""

    global NON_PREDICTIVE_COLUMNS, EXPOSURE_COLUMN
    NON_PREDICTIVE_COLUMNS = non_predictive_columns(dataset_spec)
    from autoresearch.targets import UNIT_WEIGHT_COLUMN

    EXPOSURE_COLUMN = dataset_spec.weight_column or UNIT_WEIGHT_COLUMN


def is_predictive_feature(column: str, role: str | None = None) -> bool:
    """Return whether a column may be used as a model predictor."""

    if column in NON_PREDICTIVE_COLUMNS:
        return False
    return role not in {"target_or_outcome", "record_id", "exposure_offset"}


def predictive_columns(columns: Iterable[dict]) -> list[str]:
    """Extract dataset columns that are eligible as predictors."""

    return [
        item["name"]
        for item in columns
        if is_predictive_feature(item["name"], item.get("role"))
    ]
