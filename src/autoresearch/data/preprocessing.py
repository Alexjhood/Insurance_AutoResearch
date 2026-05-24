"""Explicit preprocessing transformations used by data prep and experiments."""

from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_CAPPED_COLUMN = "claim_cost_capped_active"


def apply_claim_capping(
    frame: pd.DataFrame,
    claim_column: str,
    threshold: float,
    enabled: bool = True,
    output_column: str = DEFAULT_CAPPED_COLUMN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply transparent claim capping and return diagnostics.

    The original claim column is preserved. The effective modelling target is
    written to ``output_column`` so downstream artifacts can show capped and
    uncapped values side by side.
    """

    if claim_column not in frame.columns:
        raise ValueError(f"Claim column {claim_column!r} is not present")
    if threshold <= 0:
        raise ValueError("Claim cap threshold must be positive")

    result = frame.copy()
    uncapped = result[claim_column].astype(float)
    capped = uncapped.clip(upper=threshold) if enabled else uncapped
    result[output_column] = capped
    capped_mask = uncapped > threshold if enabled else pd.Series(False, index=result.index)

    diagnostics: dict[str, Any] = {
        "claim_capping_enabled": bool(enabled),
        "claim_cap_threshold": float(threshold),
        "source_claim_column": claim_column,
        "output_claim_column": output_column,
        "row_count": int(len(result)),
        "capped_row_count": int(capped_mask.sum()),
        "capped_row_rate": float(capped_mask.mean()) if len(result) else 0.0,
        "uncapped_total_claim_cost": float(uncapped.sum()),
        "capped_total_claim_cost": float(capped.sum()),
        "total_claim_cost_reduction": float(uncapped.sum() - capped.sum()),
        "uncapped_max_claim_cost": float(uncapped.max()) if len(result) else 0.0,
        "capped_max_claim_cost": float(capped.max()) if len(result) else 0.0,
        "uncapped_quantiles": _quantiles(uncapped),
        "capped_quantiles": _quantiles(capped),
    }
    return result, diagnostics


def _quantiles(series: pd.Series) -> dict[str, float]:
    quantiles = series.quantile([0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {str(index): float(value) for index, value in quantiles.items()}
