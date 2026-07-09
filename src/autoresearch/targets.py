"""Target-mode definitions for experiment prediction and evaluation.

A :class:`TargetSpec` is the column/label contract for one evaluation target.
Historically three French specs were hand-written; they are now *generated* from
the active dataset's ``[[targets]]`` tables by :func:`build_target_spec`, driven
by two config fields — ``entity_label`` (the total/count entity being predicted)
and ``rate_label``/``rate_slug`` (the per-weight rate). Applied to the French
config this reproduces the historical literal strings exactly, so old registries,
prediction parquets, and reports stay readable.

Dataset relativity: ``target_spec(mode)`` and ``normalise_target_mode(value)``
resolve against the process-bound dataset (set at config load via
:func:`bind_dataset`), falling back to the built-in French specs. This keeps the
protected evaluation stack — which calls ``target_spec(mode)`` with no dataset
argument — working unchanged across datasets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


BURNING_COST = "burning_cost"
FREQUENCY = "frequency"
SEVERITY = "severity"
VALID_TARGET_MODES = {BURNING_COST, FREQUENCY, SEVERITY}

# Population selectors — which rows a mode trains and evaluates on. Retained as
# thin string shims; filtering is now driven by ``TargetSpec.population_column``.
POPULATION_ALL = "all"
POPULATION_CLAIM_ROWS = "claim_rows"  # rows with ClaimNb > 0
POPULATION_POSITIVE_CLAIM_AMOUNT_ROWS = "positive_claim_amount_rows"  # rows with ClaimAmountCount > 0

# Column name synthesised by the pipeline when a dataset has no weight column.
UNIT_WEIGHT_COLUMN = "unit_weight"


@dataclass(frozen=True)
class TargetSpec:
    """Column and label contract for an active evaluation target."""

    mode: str
    source_column: str
    predicted_column: str
    rate_actual_column: str
    rate_predicted_column: str
    actual_alias: str
    predicted_alias: str
    rate_label: str
    total_actual_key: str
    total_predicted_key: str
    mae_key: str
    rmse_key: str
    mean_actual_rate_key: str
    mean_predicted_rate_key: str
    default_primary_metric: str
    # Weight/offset column — the denominator that turns a target total into a
    # rate and the sample weight for every metric. Exposure for population-wide
    # French modes; claim count for severity; the synthesised unit weight for
    # datasets without an exposure column.
    weight_column: str = "Exposure"
    # Row population the mode trains and scores on (descriptive string).
    population: str = POPULATION_ALL
    # Column whose ``> 0`` rows define the population; None → all rows.
    population_column: str | None = None
    # Generation inputs, preserved for handoff/reporting text.
    entity_label: str = ""
    rate_slug: str = ""


def build_target_spec(cfg: Mapping[str, Any], *, unit_weight_column: str = UNIT_WEIGHT_COLUMN) -> TargetSpec:
    """Generate a :class:`TargetSpec` from a dataset ``[[targets]]`` config table.

    Config fields: ``mode``, ``source_column``, ``weight`` ("unit" | column),
    ``population`` ("all" | {positive_column: col}), ``rate_label``, optional
    ``rate_slug`` (defaults to the rate label slugified), ``entity_label``.
    """

    mode = str(cfg["mode"])
    source_column = str(cfg["source_column"])
    rate_label = str(cfg["rate_label"])
    entity_label = str(cfg["entity_label"])
    rate_slug = str(cfg.get("rate_slug") or rate_label.strip().lower().replace(" ", "_"))

    weight_raw = str(cfg.get("weight", "unit"))
    weight_column = unit_weight_column if weight_raw == "unit" else weight_raw

    population = cfg.get("population", POPULATION_ALL)
    population_column: str | None = None
    population_label = POPULATION_ALL
    if isinstance(population, Mapping):
        population_column = str(population["positive_column"])
        # Preserve the historical French severity population string.
        population_label = (
            POPULATION_POSITIVE_CLAIM_AMOUNT_ROWS
            if population_column == "ClaimAmountCount"
            else f"positive:{population_column}"
        )
    elif str(population) not in {POPULATION_ALL, ""}:
        population_label = str(population)

    return TargetSpec(
        mode=mode,
        source_column=source_column,
        predicted_column=f"predicted_{entity_label}",
        rate_actual_column=f"actual_{rate_slug}",
        rate_predicted_column=f"predicted_{rate_slug}",
        actual_alias=f"actual_{entity_label}",
        predicted_alias=f"predicted_{entity_label}",
        rate_label=rate_label,
        total_actual_key=f"total_actual_{entity_label}",
        total_predicted_key=f"total_predicted_{entity_label}",
        mae_key=f"weighted_mae_{entity_label}",
        rmse_key=f"weighted_rmse_{entity_label}",
        mean_actual_rate_key=f"mean_actual_{rate_slug}",
        mean_predicted_rate_key=f"mean_predicted_{rate_slug}",
        default_primary_metric=str(cfg.get("default_primary_metric", "gini_weighted")),
        weight_column=weight_column,
        population=population_label,
        population_column=population_column,
        entity_label=entity_label,
        rate_slug=rate_slug,
    )


# Built-in French specs (the historical default), generated once from the
# canonical config so the literal strings live in exactly one place.
_FRENCH_TARGET_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "mode": BURNING_COST,
        "source_column": "ClaimAmountCapped",
        "weight": "Exposure",
        "population": POPULATION_ALL,
        "rate_label": "pure premium",
        "entity_label": "claim_cost",
    },
    {
        "mode": FREQUENCY,
        "source_column": "ClaimNb",
        "weight": "Exposure",
        "population": POPULATION_ALL,
        "rate_label": "claim frequency",
        "rate_slug": "frequency",
        "entity_label": "claim_count",
    },
    {
        "mode": SEVERITY,
        "source_column": "ClaimAmountCapped",
        "weight": "ClaimAmountCount",
        "population": {"positive_column": "ClaimAmountCount"},
        "rate_label": "severity",
        "entity_label": "claim_cost",
    },
)

SPECS: dict[str, TargetSpec] = {cfg["mode"]: build_target_spec(cfg) for cfg in _FRENCH_TARGET_CONFIGS}


# ---------------------------------------------------------------------------
# Process-bound active dataset (set at config load)
# ---------------------------------------------------------------------------

_ACTIVE_SPECS: dict[str, TargetSpec] = {}
_ACTIVE_MODES: frozenset[str] = frozenset()


def bind_dataset(dataset_spec: Any) -> None:
    """Bind the active dataset so ``target_spec``/``normalise_target_mode`` resolve
    its modes. ``dataset_spec`` needs ``.target_configs`` (iterable of mappings).

    Idempotent per process; the last-loaded dataset wins. Built-in French specs
    remain available as a fallback so protected metric code that names a French
    mode keeps working even when another dataset is bound.
    """

    global _ACTIVE_SPECS, _ACTIVE_MODES
    configs = getattr(dataset_spec, "target_configs", None) or ()
    _ACTIVE_SPECS = {str(cfg["mode"]): build_target_spec(cfg) for cfg in configs}
    _ACTIVE_MODES = frozenset(_ACTIVE_SPECS)


def _resolve_specs(dataset: Any | None) -> tuple[dict[str, TargetSpec], frozenset[str]]:
    if dataset is not None:
        configs = getattr(dataset, "target_configs", None) or ()
        specs = {str(cfg["mode"]): build_target_spec(cfg) for cfg in configs}
        return specs, frozenset(specs)
    # Active binding first, French built-ins as fallback.
    merged = dict(SPECS)
    merged.update(_ACTIVE_SPECS)
    modes = frozenset(merged)
    return merged, modes


def available_target_modes(dataset: Any | None = None) -> frozenset[str]:
    _, modes = _resolve_specs(dataset)
    return modes


def normalise_target_mode(value: str | None, dataset: Any | None = None) -> str:
    """Return a validated target mode.

    With no dataset and no active binding, validates against the built-in French
    modes (historical behaviour). Otherwise validates against the active or
    supplied dataset's modes (French modes remain accepted as a fallback).
    """

    specs, modes = _resolve_specs(dataset)
    if value is None:
        # Preserve historical default when unspecified.
        return BURNING_COST if BURNING_COST in modes else next(iter(sorted(modes)))
    mode = value.strip().lower().replace("-", "_")
    if mode not in modes:
        raise ValueError(f"target_mode must be one of {sorted(modes)}, got {value!r}")
    return mode


def target_spec(target_mode: str | None, dataset: Any | None = None) -> TargetSpec:
    """Return the target specification for a validated target mode."""

    specs, modes = _resolve_specs(dataset)
    mode = normalise_target_mode(target_mode, dataset)
    return specs[mode]
