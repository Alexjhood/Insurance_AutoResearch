"""Backwards-compatibility lock: generated French TargetSpecs == today's literals.

These literal values are the pre-multi-dataset ``targets.SPECS`` field values,
copied here verbatim. The refactor that makes TargetSpec construction generic
(``build_target_spec`` driven by entity_label/rate_label) MUST reproduce them
byte-for-byte so old registries, prediction parquets, and reports stay readable
with zero special-casing.
"""

from __future__ import annotations

from autoresearch.datasets import load_dataset_spec
from autoresearch.targets import (
    POPULATION_ALL,
    POPULATION_POSITIVE_CLAIM_AMOUNT_ROWS,
    build_target_spec,
    target_spec,
)


# The frozen truth, lifted from the original hand-written SPECS dict.
FRENCH_LITERALS = {
    "burning_cost": {
        "mode": "burning_cost",
        "source_column": "ClaimAmountCapped",
        "predicted_column": "predicted_claim_cost",
        "rate_actual_column": "actual_pure_premium",
        "rate_predicted_column": "predicted_pure_premium",
        "actual_alias": "actual_claim_cost",
        "predicted_alias": "predicted_claim_cost",
        "rate_label": "pure premium",
        "total_actual_key": "total_actual_claim_cost",
        "total_predicted_key": "total_predicted_claim_cost",
        "mae_key": "weighted_mae_claim_cost",
        "rmse_key": "weighted_rmse_claim_cost",
        "mean_actual_rate_key": "mean_actual_pure_premium",
        "mean_predicted_rate_key": "mean_predicted_pure_premium",
        "default_primary_metric": "gini_weighted",
        "weight_column": "Exposure",
        "population": POPULATION_ALL,
    },
    "frequency": {
        "mode": "frequency",
        "source_column": "ClaimNb",
        "predicted_column": "predicted_claim_count",
        "rate_actual_column": "actual_frequency",
        "rate_predicted_column": "predicted_frequency",
        "actual_alias": "actual_claim_count",
        "predicted_alias": "predicted_claim_count",
        "rate_label": "claim frequency",
        "total_actual_key": "total_actual_claim_count",
        "total_predicted_key": "total_predicted_claim_count",
        "mae_key": "weighted_mae_claim_count",
        "rmse_key": "weighted_rmse_claim_count",
        "mean_actual_rate_key": "mean_actual_frequency",
        "mean_predicted_rate_key": "mean_predicted_frequency",
        "default_primary_metric": "gini_weighted",
        "weight_column": "Exposure",
        "population": POPULATION_ALL,
    },
    "severity": {
        "mode": "severity",
        "source_column": "ClaimAmountCapped",
        "predicted_column": "predicted_claim_cost",
        "rate_actual_column": "actual_severity",
        "rate_predicted_column": "predicted_severity",
        "actual_alias": "actual_claim_cost",
        "predicted_alias": "predicted_claim_cost",
        "rate_label": "severity",
        "total_actual_key": "total_actual_claim_cost",
        "total_predicted_key": "total_predicted_claim_cost",
        "mae_key": "weighted_mae_claim_cost",
        "rmse_key": "weighted_rmse_claim_cost",
        "mean_actual_rate_key": "mean_actual_severity",
        "mean_predicted_rate_key": "mean_predicted_severity",
        "default_primary_metric": "gini_weighted",
        "weight_column": "ClaimAmountCount",
        "population": POPULATION_POSITIVE_CLAIM_AMOUNT_ROWS,
    },
}

_LOCKED_FIELDS = tuple(FRENCH_LITERALS["burning_cost"].keys())


def test_build_target_spec_reproduces_french_literals():
    spec = load_dataset_spec("french_motor")
    for mode, expected in FRENCH_LITERALS.items():
        built = build_target_spec(spec.target_config(mode))
        for field in _LOCKED_FIELDS:
            assert getattr(built, field) == expected[field], (
                f"{mode}.{field}: got {getattr(built, field)!r}, expected {expected[field]!r}"
            )


def test_target_spec_resolves_french_modes_by_name():
    # target_spec(mode) with no dataset argument must still yield French specs.
    for mode, expected in FRENCH_LITERALS.items():
        spec = target_spec(mode)
        for field in _LOCKED_FIELDS:
            assert getattr(spec, field) == expected[field]


def test_severity_population_column():
    spec = target_spec("severity")
    assert spec.population_column == "ClaimAmountCount"
    assert target_spec("burning_cost").population_column is None


def test_generic_specs_for_unit_weight_datasets():
    from autoresearch.targets import UNIT_WEIGHT_COLUMN

    allstate = load_dataset_spec("allstate")
    pp = build_target_spec(allstate.target_config("pure_premium"))
    assert pp.weight_column == UNIT_WEIGHT_COLUMN
    assert pp.source_column == "Claim_Amount"
    assert pp.predicted_alias == "predicted_claim_cost"
    assert pp.rate_actual_column == "actual_pure_premium"
    assert pp.population_column is None

    sev = build_target_spec(allstate.target_config("severity"))
    assert sev.population_column == "claim_occurred"

    porto = load_dataset_spec("porto_seguro")
    inc = build_target_spec(porto.target_config("claim_incidence"))
    assert inc.weight_column == UNIT_WEIGHT_COLUMN
    assert inc.source_column == "target"
    assert inc.predicted_alias == "predicted_claim_indicator"
    assert inc.rate_actual_column == "actual_claim_probability"
    assert inc.rate_label == "claim probability"


def test_bind_dataset_resolves_new_modes_with_french_fallback():
    import autoresearch.targets as targets

    porto = load_dataset_spec("porto_seguro")
    targets.bind_dataset(porto)
    try:
        # Porto's own mode resolves.
        assert target_spec("claim_incidence").source_column == "target"
        # French modes remain resolvable via fallback (protected metric code
        # may still name them even when another dataset is bound).
        assert target_spec("burning_cost").source_column == "ClaimAmountCapped"
    finally:
        targets.bind_dataset(load_dataset_spec("french_motor"))
