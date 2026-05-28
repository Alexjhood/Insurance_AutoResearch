"""Target-mode definitions for experiment prediction and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


BURNING_COST = "burning_cost"
FREQUENCY = "frequency"
VALID_TARGET_MODES = {BURNING_COST, FREQUENCY}


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


SPECS = {
    BURNING_COST: TargetSpec(
        mode=BURNING_COST,
        source_column="claim_cost_capped_active",
        predicted_column="predicted_claim_cost",
        rate_actual_column="actual_pure_premium",
        rate_predicted_column="predicted_pure_premium",
        actual_alias="actual_claim_cost",
        predicted_alias="predicted_claim_cost",
        rate_label="pure premium",
        total_actual_key="total_actual_claim_cost",
        total_predicted_key="total_predicted_claim_cost",
        mae_key="weighted_mae_claim_cost",
        rmse_key="weighted_rmse_claim_cost",
        mean_actual_rate_key="mean_actual_pure_premium",
        mean_predicted_rate_key="mean_predicted_pure_premium",
        default_primary_metric="gini_weighted",
    ),
    FREQUENCY: TargetSpec(
        mode=FREQUENCY,
        source_column="claim_count_signal_q",
        predicted_column="predicted_claim_count",
        rate_actual_column="actual_frequency",
        rate_predicted_column="predicted_frequency",
        actual_alias="actual_claim_count",
        predicted_alias="predicted_claim_count",
        rate_label="claim frequency",
        total_actual_key="total_actual_claim_count",
        total_predicted_key="total_predicted_claim_count",
        mae_key="weighted_mae_claim_count",
        rmse_key="weighted_rmse_claim_count",
        mean_actual_rate_key="mean_actual_frequency",
        mean_predicted_rate_key="mean_predicted_frequency",
        default_primary_metric="gini_weighted",
    ),
}


def normalise_target_mode(value: str | None) -> str:
    """Return a validated target mode, defaulting to burning cost."""

    mode = (value or BURNING_COST).strip().lower().replace("-", "_")
    if mode not in VALID_TARGET_MODES:
        raise ValueError(f"target_mode must be one of {sorted(VALID_TARGET_MODES)}, got {value!r}")
    return mode


def target_spec(target_mode: str | None) -> TargetSpec:
    """Return the target specification for a validated target mode."""

    return SPECS[normalise_target_mode(target_mode)]
