"""Process-bound column identity for the active dataset.

The model layer historically hard-coded French freMTPL2 column names
(``Exposure``, ``ClaimAmountCapped``, …). Those names now come from the active
:class:`~autoresearch.datasets.DatasetSpec`, bound once at config load via
:func:`bind`. The module-level constants keep their French defaults so that
unbound imports (and agent scripts written for French runs) keep working; after
``bind`` they reflect the active dataset.

The *leak set* — every target-bearing column plus the raw uncapped source — is
what a run-local script must never see for its scored rows. It is derived from
the dataset's targets, cap, and count columns so script-frame sanitisation is
automatically correct per dataset.
"""

from __future__ import annotations

from typing import Any

from autoresearch.targets import UNIT_WEIGHT_COLUMN

RECORD_ID = "record_id"

# French defaults — overwritten by bind() at config load.
EXPOSURE = "Exposure"
CLAIM_COUNT = "ClaimNb"
CLAIM_EVENTS = "ClaimAmountCount"
CLAIM_COST = "ClaimAmountCapped"
RAW_CLAIM_COST = "ClaimAmount"

_FRENCH_LEAK = ("ClaimAmount", "ClaimAmountCapped", "ClaimAmountCount", "ClaimNb")

_STATE: dict[str, Any] = {
    "id_column": "IDpol",
    "weight_column": "Exposure",
    "count_column": "ClaimNb",
    "event_count_column": "ClaimAmountCount",
    "leak_columns": _FRENCH_LEAK,
    "raw_train_strip": ("ClaimAmount",),
}


def bind(dataset_spec: Any) -> None:
    """Bind the active dataset's column identity into this module."""

    global EXPOSURE, CLAIM_COUNT, CLAIM_EVENTS, CLAIM_COST, RAW_CLAIM_COST

    leak: set[str] = set()
    for cfg in getattr(dataset_spec, "target_configs", ()) or ():
        leak.add(str(cfg["source_column"]))
    cap = getattr(dataset_spec, "cap", None)
    raw_strip: set[str] = set()
    if cap is not None:
        leak.add(cap.column)
        leak.add(cap.output_column)
        raw_strip.add(cap.column)  # raw uncapped source is never a train input
    if dataset_spec.count_column:
        leak.add(dataset_spec.count_column)
    if dataset_spec.event_count_column:
        leak.add(dataset_spec.event_count_column)

    _STATE.update(
        id_column=dataset_spec.id_column,
        weight_column=dataset_spec.weight_column or UNIT_WEIGHT_COLUMN,
        count_column=dataset_spec.count_column,
        event_count_column=dataset_spec.event_count_column,
        leak_columns=tuple(sorted(leak)),
        raw_train_strip=tuple(sorted(raw_strip)),
    )

    # Reassign the exported French-style constants for backwards compatibility.
    EXPOSURE = dataset_spec.weight_column or UNIT_WEIGHT_COLUMN
    CLAIM_COUNT = dataset_spec.count_column or CLAIM_COUNT
    CLAIM_EVENTS = dataset_spec.event_count_column or CLAIM_EVENTS
    CLAIM_COST = cap.output_column if cap is not None else CLAIM_COST
    RAW_CLAIM_COST = cap.column if cap is not None else RAW_CLAIM_COST
    _sync_exports()


def _sync_exports() -> None:
    """Reflect the bound constants onto the model modules that re-export them, so
    agent scripts doing ``from autoresearch.models.dispatcher import EXPOSURE``
    see the active dataset's column names."""

    import sys

    values = {
        "EXPOSURE": EXPOSURE,
        "CLAIM_COUNT": CLAIM_COUNT,
        "CLAIM_EVENTS": CLAIM_EVENTS,
        "CLAIM_COST": CLAIM_COST,
        "RAW_CLAIM_COST": RAW_CLAIM_COST,
    }
    for name in (
        "autoresearch.models.dispatcher",
        "autoresearch.models.global_mean",
        "autoresearch.models.recipe.interpreter",
    ):
        module = sys.modules.get(name)
        if module is not None:
            for key, value in values.items():
                if hasattr(module, key):
                    setattr(module, key, value)


def leak_columns() -> tuple[str, ...]:
    """Target-bearing columns that must be stripped from a script's score frame."""

    return tuple(_STATE["leak_columns"])


def raw_train_strip_columns() -> tuple[str, ...]:
    """Columns stripped from a script's train frame (raw uncapped source)."""

    return tuple(_STATE["raw_train_strip"])
