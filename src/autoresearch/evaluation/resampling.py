"""Repeated resampling, k-fold CV, paired comparison, and promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autoresearch.evaluation.metrics import (
    full_metric_panel,
    infer_target_mode,
    lower_is_better,
    prediction_target_columns,
    regression_metrics,  # noqa: F401 — re-exported for external callers
)


@dataclass(frozen=True)
class PromotionRules:
    """Configurable thresholds for champion/challenger promotion."""

    minimum_mean_lift: float          # absolute lift (legacy, kept for compat)
    min_relative_lift: float          # lift as fraction of champion score (primary)
    min_absolute_lift: float          # hard floor in metric units
    minimum_win_rate: float
    bootstrap_lower_bound: float
    bootstrap_lower_bound_relative: float
    confidence_level: float
    max_predicted_to_actual_drift: float  # reject if calibration worsens > this
    require_diagnostics: bool
    bonferroni_lookback: int          # number of prior comparisons to adjust for
    require_sign_agreement: bool = True   # rank_gini and gini_weighted must agree


def repeated_scores(
    predictions: pd.DataFrame,
    *,
    eval_split: str,
    n_resamples: int,
    seed: int,
    resample_fraction: float = 1.0,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
    target_mode: str | None = None,
) -> pd.DataFrame:
    """Score one experiment across deterministic search-time resamples."""

    evaluation_frame = _ordinary_eval_frame(predictions, eval_split)
    target_mode = infer_target_mode(predictions, target_mode)
    actual_col, predicted_col = prediction_target_columns(evaluation_frame, target_mode)
    sample_size = _sample_size(len(evaluation_frame), resample_fraction)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for resample_id in range(n_resamples):
        positions = rng.integers(0, len(evaluation_frame), size=sample_size)
        sample = evaluation_frame.iloc[positions]
        metrics = full_metric_panel(
            sample[actual_col], sample[predicted_col], sample["exposure"],
            tweedie_power=tweedie_power,
            target_mode=target_mode,
        )
        rows.append({
            "resample_id": resample_id,
            "split": eval_split,
            "sample_size": sample_size,
            "score": metrics[primary_metric],
            **{k: v for k, v in metrics.items() if isinstance(v, float)},
        })
    return pd.DataFrame(rows)


def paired_comparison(
    champion_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    *,
    champion_id: str,
    challenger_id: str,
    eval_split: str,
    n_resamples: int,
    seed: int,
    resample_fraction: float = 1.0,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
    target_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare two experiments on identical search-time resamples (paired)."""

    champion = _ordinary_eval_frame(champion_predictions, eval_split)
    challenger = _ordinary_eval_frame(challenger_predictions, eval_split)
    target_mode = infer_target_mode(challenger_predictions, target_mode)
    actual_col, champion_predicted_col = prediction_target_columns(champion, target_mode)
    _, challenger_predicted_col = prediction_target_columns(challenger, target_mode)
    paired = champion[["record_id", actual_col, champion_predicted_col, "exposure"]].merge(
        challenger[["record_id", challenger_predicted_col]],
        on="record_id",
        suffixes=("_champion", "_challenger"),
        how="inner",
    )
    if paired.empty:
        raise ValueError("Champion and challenger predictions have no overlapping evaluation rows")

    sample_size = _sample_size(len(paired), resample_fraction)
    metric_lower_is_better = lower_is_better(primary_metric)
    actual_merged_col = actual_col
    champion_merged_col = f"{champion_predicted_col}_champion"
    challenger_merged_col = f"{challenger_predicted_col}_challenger"
    if champion_predicted_col != challenger_predicted_col:
        champion_merged_col = champion_predicted_col
        challenger_merged_col = challenger_predicted_col
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for resample_id in range(n_resamples):
        positions = rng.integers(0, len(paired), size=sample_size)
        sample = paired.iloc[positions]
        champ_metrics = full_metric_panel(
            sample[actual_merged_col], sample[champion_merged_col], sample["exposure"],
            tweedie_power=tweedie_power,
            target_mode=target_mode,
        )
        chal_metrics = full_metric_panel(
            sample[actual_merged_col], sample[challenger_merged_col], sample["exposure"],
            tweedie_power=tweedie_power,
            target_mode=target_mode,
        )
        champ_score = champ_metrics[primary_metric]
        chal_score = chal_metrics[primary_metric]
        lift = champ_score - chal_score if metric_lower_is_better else chal_score - champ_score
        rows.append({
            "resample_id": resample_id,
            "split": eval_split,
            "sample_size": sample_size,
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "champion_score": champ_score,
            "challenger_score": chal_score,
            "lift": lift,
            "challenger_won": bool(lift > 0),
        })

    per_resample = pd.DataFrame(rows)
    summary = {
        "champion_id": champion_id,
        "challenger_id": challenger_id,
        "eval_split": eval_split,
        "primary_metric": primary_metric,
        "target_mode": target_mode,
        "lower_is_better": metric_lower_is_better,
        "n_resamples": n_resamples,
        "resample_fraction": resample_fraction,
        "seed": seed,
        "mean_lift": float(per_resample["lift"].mean()),
        "median_lift": float(per_resample["lift"].median()),
        "std_lift": float(per_resample["lift"].std(ddof=0)),
        "challenger_win_rate": float(per_resample["challenger_won"].mean()),
        "champion_mean_score": float(per_resample["champion_score"].mean()),
        "challenger_mean_score": float(per_resample["challenger_score"].mean()),
    }
    return per_resample, summary


def paired_cv_comparison(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    champion_factory,
    challenger_factory,
    champion_id: str,
    challenger_id: str,
    n_folds: int,
    n_repeats: int,
    tweedie_power: float = 1.5,
    gate_primary_metric: str = "rank_gini_weighted",
    seed: int = 0,
    target_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Paired comparison via repeated k-fold CV — the honest partition-noise gate.

    Both models are refit on *identical* (repeat, fold) partitions (common
    random numbers).  The per-partition lift distribution therefore reflects
    true partition variance, not within-split bootstrap noise.

    ``champion_factory`` and ``challenger_factory`` are callables with signature
    ``factory(train_df, val_df) -> predictions_df`` where predictions_df contains
    at least ``actual_target``, ``predicted_target``, ``exposure``, and
    ``target_mode`` columns.

    Returns (per_partition_df, summary_dict).  The summary dict is API-compatible
    with ``paired_comparison`` so ``promotion_decision`` can consume it directly.
    """

    data = frame.merge(fold_assignments[["record_id", "fold"]], on="record_id", how="inner")
    if data.empty:
        raise ValueError("No rows after merging frame with fold_assignments")

    metric_lower = lower_is_better(gate_primary_metric)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for repeat in range(n_repeats):
        for fold in range(1, n_folds + 1):
            val_mask = data["fold"] == fold
            train_data = data[~val_mask].copy()
            val_data = data[val_mask].copy()

            if n_repeats > 1:
                # Reshuffle training rows across repeats (different row order = different
                # mini-batch ordering for gradient methods; same folds).
                train_data = train_data.sample(
                    frac=1.0, random_state=int(rng.integers(0, 2**31))
                ).copy()

            champ_preds = champion_factory(train_data, val_data)
            chal_preds = challenger_factory(train_data, val_data)

            fold_target_mode = infer_target_mode(chal_preds, target_mode)
            actual_col, champ_pred_col = prediction_target_columns(champ_preds, fold_target_mode)
            _, chal_pred_col = prediction_target_columns(chal_preds, fold_target_mode)

            # Merge on record_id with explicit suffixes to handle identical column names
            paired = champ_preds[["record_id", actual_col, champ_pred_col, "exposure"]].merge(
                chal_preds[["record_id", chal_pred_col]],
                on="record_id",
                suffixes=("_champ", "_chal"),
                how="inner",
            )
            if paired.empty:
                continue

            # Resolve column names after merge (suffix applied when names collide)
            if champ_pred_col == chal_pred_col:
                merged_champ_col = f"{champ_pred_col}_champ"
                merged_chal_col = f"{chal_pred_col}_chal"
            else:
                merged_champ_col = champ_pred_col
                merged_chal_col = chal_pred_col
            actual_merged_col = actual_col  # not suffixed (only in champ frame)

            actual = paired[actual_merged_col]
            champ_pred = paired[merged_champ_col]
            chal_pred = paired[merged_chal_col]
            exp = paired["exposure"]

            champ_metrics = full_metric_panel(
                actual, champ_pred, exp,
                tweedie_power=tweedie_power,
                target_mode=fold_target_mode,
            )
            chal_metrics = full_metric_panel(
                actual, chal_pred, exp,
                tweedie_power=tweedie_power,
                target_mode=fold_target_mode,
            )

            champ_gate = champ_metrics[gate_primary_metric]
            chal_gate = chal_metrics[gate_primary_metric]
            gate_lift = champ_gate - chal_gate if metric_lower else chal_gate - champ_gate

            # Also track KPI (gini_weighted) lift for sign-agreement check
            champ_kpi = champ_metrics.get("gini_weighted", float("nan"))
            chal_kpi = chal_metrics.get("gini_weighted", float("nan"))
            kpi_lift = chal_kpi - champ_kpi  # gini_weighted is always higher-is-better

            row: dict[str, Any] = {
                "repeat": repeat,
                "fold": fold,
                "n_val": len(paired),
                "champion_id": champion_id,
                "challenger_id": challenger_id,
                "champion_gate_score": champ_gate,
                "challenger_gate_score": chal_gate,
                "lift": gate_lift,
                "challenger_won": bool(gate_lift > 0),
                "champion_kpi_score": champ_kpi,
                "challenger_kpi_score": chal_kpi,
                "kpi_lift": kpi_lift,
            }
            # Include all metric lifts for the report exhibit
            for metric_name in champ_metrics:
                if isinstance(champ_metrics[metric_name], float):
                    c_val = champ_metrics[metric_name]
                    h_val = chal_metrics.get(metric_name, float("nan"))
                    row[f"champ_{metric_name}"] = c_val
                    row[f"chal_{metric_name}"] = h_val
            rows.append(row)

    if not rows:
        raise ValueError("No fold results produced — check that frame and fold_assignments overlap")

    per_partition = pd.DataFrame(rows)
    n_partitions = len(per_partition)

    mean_lift = float(per_partition["lift"].mean())
    between_partition_std = float(per_partition["lift"].std(ddof=0))
    win_rate = float(per_partition["challenger_won"].mean())
    champion_mean_score = float(per_partition["champion_gate_score"].mean())
    challenger_mean_score = float(per_partition["challenger_gate_score"].mean())
    mean_kpi_lift = float(per_partition["kpi_lift"].mean())

    summary: dict[str, Any] = {
        "gate_mode": "repeated_cv",
        "gate_primary_metric": gate_primary_metric,
        "champion_id": champion_id,
        "challenger_id": challenger_id,
        "target_mode": target_mode,
        "lower_is_better": metric_lower,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_partitions": n_partitions,
        "seed": seed,
        # API-compatible aliases used by promotion_decision and reporting
        "primary_metric": gate_primary_metric,
        "eval_split": "cv_folds",
        "n_resamples": n_partitions,
        "mean_lift": mean_lift,
        "median_lift": float(per_partition["lift"].median()),
        "std_lift": between_partition_std,
        "between_partition_std": between_partition_std,
        "challenger_win_rate": win_rate,
        "champion_mean_score": champion_mean_score,
        "challenger_mean_score": challenger_mean_score,
        # KPI tracking
        "mean_kpi_lift": mean_kpi_lift,
        "kpi_lift_positive": bool(mean_kpi_lift > 0),
        "kpi_metric": "gini_weighted",
    }
    return per_partition, summary


def cv_bootstrap_comparison(
    *,
    champion_fold_predictions: dict[int, list[pd.DataFrame]],
    challenger_fold_predictions: dict[int, list[pd.DataFrame]],
    gate_primary_metric: str = "gini_weighted",
    bootstrap_per_fold: int = 20,
    tweedie_power: float = 1.5,
    seed: int = 0,
    target_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bootstrap-within-fold paired comparison using pre-computed fold predictions.

    ``champion_fold_predictions[partition_idx]`` is a list of DataFrames, one per
    fold, each containing validation-set predictions with columns ``record_id``,
    ``actual_target`` (or alias), ``predicted_target`` (or alias), ``exposure``.

    For each (partition, fold): inner-joins champion and challenger on record_id,
    draws ``bootstrap_per_fold`` bootstrap index sets with replacement (same indices
    applied to both models — common random numbers seeded by (seed, partition, fold)),
    then evaluates the full metric panel on each bootstrap sample.

    Returns ``(per_sample_df, summary_dict)`` where per_sample_df has one row per
    (partition × fold × bootstrap) sample and summary_dict is API-compatible with
    ``promotion_decision`` and ``bootstrap_lift_summary``.
    """

    metric_lower = lower_is_better(gate_primary_metric)
    partition_indices = sorted(champion_fold_predictions.keys())

    rows: list[dict[str, Any]] = []
    inferred_target_mode: str | None = None

    for p_idx in partition_indices:
        champ_folds = champion_fold_predictions[p_idx]
        chal_folds = challenger_fold_predictions[p_idx]
        for f_idx, (champ_preds, chal_preds) in enumerate(zip(champ_folds, chal_folds)):
            fold_target_mode = infer_target_mode(chal_preds, target_mode)
            if inferred_target_mode is None:
                inferred_target_mode = fold_target_mode

            actual_col, champ_pred_col = prediction_target_columns(champ_preds, fold_target_mode)
            _, chal_pred_col = prediction_target_columns(chal_preds, fold_target_mode)

            paired = champ_preds[["record_id", actual_col, champ_pred_col, "exposure"]].merge(
                chal_preds[["record_id", chal_pred_col]],
                on="record_id",
                suffixes=("_champ", "_chal"),
                how="inner",
            )
            if paired.empty:
                continue

            if champ_pred_col == chal_pred_col:
                merged_champ_col = f"{champ_pred_col}_champ"
                merged_chal_col = f"{chal_pred_col}_chal"
            else:
                merged_champ_col = champ_pred_col
                merged_chal_col = chal_pred_col

            n_rows = len(paired)
            rng = np.random.default_rng([seed, p_idx, f_idx])

            for b_idx in range(bootstrap_per_fold):
                idx = rng.integers(0, n_rows, size=n_rows)
                sample = paired.iloc[idx]

                champ_metrics = full_metric_panel(
                    sample[actual_col], sample[merged_champ_col], sample["exposure"],
                    tweedie_power=tweedie_power,
                    target_mode=fold_target_mode,
                )
                chal_metrics = full_metric_panel(
                    sample[actual_col], sample[merged_chal_col], sample["exposure"],
                    tweedie_power=tweedie_power,
                    target_mode=fold_target_mode,
                )

                champ_gate = champ_metrics[gate_primary_metric]
                chal_gate = chal_metrics[gate_primary_metric]
                gate_lift = champ_gate - chal_gate if metric_lower else chal_gate - champ_gate

                row: dict[str, Any] = {
                    "partition_idx": p_idx,
                    "fold_idx": f_idx,
                    "bootstrap_idx": b_idx,
                    "lift": gate_lift,
                    "challenger_won": bool(gate_lift > 0),
                }
                for metric_name, c_val in champ_metrics.items():
                    if isinstance(c_val, float):
                        row[f"champ_{metric_name}"] = c_val
                        row[f"chal_{metric_name}"] = chal_metrics.get(metric_name, float("nan"))
                rows.append(row)

    if not rows:
        raise ValueError("cv_bootstrap_comparison produced no samples — check that fold predictions overlap")

    per_sample = pd.DataFrame(rows)
    n_samples = len(per_sample)
    n_partitions = len(partition_indices)
    max_folds = max(len(v) for v in champion_fold_predictions.values())

    mean_lift = float(per_sample["lift"].mean())
    std_lift = float(per_sample["lift"].std(ddof=0))
    win_rate = float(per_sample["challenger_won"].mean())
    champ_gate_col = f"champ_{gate_primary_metric}"
    chal_gate_col = f"chal_{gate_primary_metric}"
    champion_mean_score = float(per_sample[champ_gate_col].mean()) if champ_gate_col in per_sample else float("nan")
    challenger_mean_score = float(per_sample[chal_gate_col].mean()) if chal_gate_col in per_sample else float("nan")

    summary: dict[str, Any] = {
        "gate_mode": "cv_bootstrap",
        "gate_primary_metric": gate_primary_metric,
        "target_mode": inferred_target_mode,
        "lower_is_better": metric_lower,
        "n_partitions": n_partitions,
        "n_folds": max_folds,
        "bootstrap_per_fold": bootstrap_per_fold,
        "n_samples": n_samples,
        # promotion_decision / bootstrap_lift_summary compatibility aliases
        "primary_metric": gate_primary_metric,
        "eval_split": "cv_bootstrap_folds",
        "n_resamples": n_samples,
        "mean_lift": mean_lift,
        "median_lift": float(per_sample["lift"].median()),
        "std_lift": std_lift,
        "between_partition_std": std_lift,
        "challenger_win_rate": win_rate,
        "champion_mean_score": champion_mean_score,
        "challenger_mean_score": challenger_mean_score,
    }
    return per_sample, summary


def evaluate_guardrails(
    challenger_metrics: dict[str, Any],
    comparison_summary: dict[str, Any],
    *,
    min_gini: float = 0.0,
    pred_to_actual_lo: float = 0.5,
    pred_to_actual_hi: float = 2.0,
) -> dict[str, Any]:
    """Evaluate hard guardrails that block a promotion regardless of LLM decision.

    Returns ``{"passed": bool, "failures": [str], "checks": {name: bool}}``.

    Hard fails (each blocks promotion):
    - No discrimination: challenger gini_weighted ≤ min_gini or NaN.
    - Gross miscalibration: predicted_to_actual_ratio outside [pred_to_actual_lo, pred_to_actual_hi].
    - Invalid predictions: any NaN or negative predicted target (detected via total_predicted_target).
    - Clearly worse: gate-metric bootstrap CI lies entirely below zero (requires
      ``comparison_summary`` to contain ``bootstrap_ci_lower`` or the per_sample lift vector).
    """

    checks: dict[str, bool] = {}

    # 1. Discrimination floor
    gini = challenger_metrics.get("gini_weighted")
    checks["gini_above_zero"] = (
        gini is not None and not (isinstance(gini, float) and np.isnan(gini)) and float(gini) > min_gini
    )

    # 2. Calibration sanity
    ratio = challenger_metrics.get("predicted_to_actual_ratio")
    if ratio is not None and not (isinstance(ratio, float) and np.isnan(ratio)):
        checks["calibration_sane"] = pred_to_actual_lo <= float(ratio) <= pred_to_actual_hi
    else:
        checks["calibration_sane"] = False

    # 3. Valid predictions (no NaN / negative totals)
    total_pred = challenger_metrics.get("total_predicted_target")
    if total_pred is None:
        total_pred = challenger_metrics.get("total_predicted_claim_cost") or challenger_metrics.get("total_predicted_claim_count")
    if total_pred is not None and not (isinstance(total_pred, float) and np.isnan(total_pred)):
        checks["predictions_valid"] = float(total_pred) >= 0.0
    else:
        checks["predictions_valid"] = False

    # 4. Clearly worse: the gate-metric lift CI lies *entirely below zero*, i.e.
    #    even the optimistic (upper) bound is negative ⇒ challenger is significantly
    #    worse, not merely inconclusive. Blocks only this unambiguous case.
    ci_upper = comparison_summary.get("lift_ci_upper")
    if ci_upper is None:
        ci_upper = comparison_summary.get("bootstrap_ci_upper")
    if ci_upper is not None and not (isinstance(ci_upper, float) and np.isnan(ci_upper)):
        checks["not_clearly_worse"] = float(ci_upper) >= 0.0
    else:
        checks["not_clearly_worse"] = True  # cannot determine — pass by default

    passed = all(checks.values())
    failures = [name for name, ok in checks.items() if not ok]
    return {"passed": passed, "failures": failures, "checks": checks}


def bootstrap_lift_summary(
    lifts: pd.Series,
    *,
    iterations: int,
    seed: int,
    confidence_level: float,
    n_comparisons: int = 1,
) -> dict[str, Any]:
    """Bootstrap the mean paired lift, with optional Bonferroni correction."""

    values = lifts.astype(float).to_numpy()
    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty lift series")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    # Bonferroni: widen alpha by number of comparisons
    adjusted_level = 1.0 - (1.0 - confidence_level) / max(1, n_comparisons)
    adjusted_level = min(adjusted_level, 0.9999)

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()

    alpha = 1.0 - adjusted_level
    lower = float(np.quantile(means, alpha / 2.0))
    upper = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {
        "bootstrap_iterations": iterations,
        "seed": seed,
        "confidence_level": confidence_level,
        "adjusted_confidence_level": float(adjusted_level),
        "n_comparisons_bonferroni": n_comparisons,
        "mean_lift": float(values.mean()),
        "interval_lower": lower,
        "interval_upper": upper,
        "probability_challenger_outperforms": float((means > 0).mean()),
    }


def cv_repeated_scores(
    frame: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    *,
    model_factory,
    n_folds: int,
    n_repeats: int = 1,
    tweedie_power: float = 1.5,
    primary_metric: str = "tweedie_deviance_p15",
    seed: int = 0,
    target_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """K-fold CV scores with variance decomposition.

    ``model_factory(train_df, val_df)`` must return ``predictions_df`` containing
    active actual/predicted target columns and exposure.
    """

    data = frame.merge(fold_assignments[["record_id", "fold"]], on="record_id", how="inner")
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    fold_scores: list[float] = []
    summary_target_mode: str | None = None

    for repeat in range(n_repeats):
        fold_fold_scores = []
        for fold in range(1, n_folds + 1):
            val_ids = set(data.loc[data["fold"] == fold, "record_id"].tolist())
            train_data = data[~data["record_id"].isin(val_ids)].copy()
            val_data = data[data["record_id"].isin(val_ids)].copy()
            if n_repeats > 1:
                # Reshuffle training rows within each repeat
                train_data = train_data.sample(frac=1.0, random_state=rng.integers(0, 2**31)).copy()
            preds = model_factory(train_data, val_data)
            fold_target_mode = infer_target_mode(preds, target_mode)
            summary_target_mode = fold_target_mode
            actual_col, predicted_col = prediction_target_columns(preds, fold_target_mode)
            metrics = full_metric_panel(
                preds[actual_col], preds[predicted_col], preds["exposure"],
                tweedie_power=tweedie_power,
                target_mode=fold_target_mode,
            )
            score = metrics[primary_metric]
            fold_fold_scores.append(score)
            rows.append({
                "repeat": repeat,
                "fold": fold,
                "n_val": len(val_data),
                "score": score,
                **{k: v for k, v in metrics.items() if isinstance(v, float)},
            })
        fold_scores.extend(fold_fold_scores)

    cv_frame = pd.DataFrame(rows)

    # Variance decomposition: between-fold vs total
    fold_means = cv_frame.groupby("fold")["score"].mean().to_numpy()
    between_fold_var = float(np.var(fold_means, ddof=0))
    total_var = float(np.var(cv_frame["score"].to_numpy(), ddof=0))
    within_fold_var = max(0.0, total_var - between_fold_var)
    ratio_between = between_fold_var / total_var if total_var > 0 else float("nan")

    summary = {
        "primary_metric": primary_metric,
        "target_mode": summary_target_mode,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "mean_score": float(cv_frame["score"].mean()),
        "std_score": float(cv_frame["score"].std(ddof=0)),
        "between_fold_variance": between_fold_var,
        "within_fold_variance": within_fold_var,
        "total_variance": total_var,
        "ratio_between_to_total": ratio_between,
        "warning_between_dominates": ratio_between > 0.5 if not np.isnan(ratio_between) else False,
    }
    return cv_frame, summary


def promotion_decision(
    comparison_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    rules: PromotionRules,
    *,
    challenger_diagnostics: dict[str, Any] | None = None,
    n_prior_comparisons: int = 0,
) -> dict[str, Any]:
    """Make an explicit promote/inconclusive decision from configured rules."""

    champion_score = comparison_summary.get("champion_mean_score", 0.0) or 1e-9
    champion_scale = abs(champion_score) or 1e-9
    mean_lift = comparison_summary["mean_lift"]
    win_rate = comparison_summary["challenger_win_rate"]
    boot_lower = bootstrap_summary["interval_lower"]
    n_resamples = comparison_summary.get("n_resamples", 30)

    # Relative lift fractions
    relative_lift = mean_lift / champion_scale
    boot_lower_relative = boot_lower / champion_scale

    # Minimum detectable effect estimate (rough 95% one-sided)
    std_lift = comparison_summary.get("std_lift", 0.0)
    mde_relative = (2 * std_lift / max(n_resamples ** 0.5, 1)) / champion_scale

    if relative_lift > (mde_relative or 1.0):
        power_note = "effect_above_mde"
    elif relative_lift > rules.min_relative_lift:
        power_note = "effect_below_mde"
    else:
        power_note = "effect_below_min_relative_lift"

    checks: dict[str, bool] = {
        "mean_lift_positive": mean_lift >= rules.minimum_mean_lift,
        "relative_lift": relative_lift >= rules.min_relative_lift,
        "absolute_lift": mean_lift >= rules.min_absolute_lift,
        "challenger_win_rate": win_rate >= rules.minimum_win_rate,
        "bootstrap_lower_bound": boot_lower >= rules.bootstrap_lower_bound,
        "bootstrap_lower_bound_relative": boot_lower_relative >= rules.bootstrap_lower_bound_relative,
    }

    # Sign-agreement: gate metric (rank_gini) and KPI (gini_weighted) must both
    # point in the same direction.  Prevents promoting a model that improves on
    # the robust gate but regresses on the business KPI.
    if rules.require_sign_agreement:
        kpi_lift_positive = comparison_summary.get("kpi_lift_positive")
        if kpi_lift_positive is None:
            # Single-partition mode: derive from champion/challenger mean scores
            # stored in comparison_summary (gini_weighted is always in the panel)
            champ_kpi = comparison_summary.get("champion_kpi_score")
            chal_kpi = comparison_summary.get("challenger_kpi_score")
            if champ_kpi is not None and chal_kpi is not None:
                kpi_lift_positive = bool(float(chal_kpi) > float(champ_kpi))
            else:
                kpi_lift_positive = None  # cannot determine — skip check
        if kpi_lift_positive is not None:
            checks["sign_agreement_kpi"] = bool(kpi_lift_positive)

    # Calibration check
    calibration_ok = True
    if rules.max_predicted_to_actual_drift < 1.0 and challenger_diagnostics:
        chal_ratio = _get_pred_to_actual(challenger_diagnostics)
        if chal_ratio is not None:
            drift = abs(1.0 - chal_ratio)
            calibration_ok = drift <= rules.max_predicted_to_actual_drift
    checks["calibration_ok"] = calibration_ok

    # Diagnostics presence check
    if rules.require_diagnostics and challenger_diagnostics:
        checks["diagnostics_present"] = not bool(challenger_diagnostics.get("error"))
    elif rules.require_diagnostics:
        checks["diagnostics_present"] = False

    promoted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]

    if promoted:
        decision = "promote"
        rationale = "Challenger passed all configured promotion thresholds."
    else:
        decision = "inconclusive"
        rationale = "No promotion: failed thresholds " + ", ".join(failed) + "."

    return {
        "decision": decision,
        "promoted": promoted,
        "rationale": rationale,
        "checks": checks,
        "effect_size": {
            "mean_lift": mean_lift,
            "relative_lift": relative_lift,
            "mde_relative": mde_relative,
            "power_note": power_note,
            "n_resamples": n_resamples,
        },
        "thresholds": {
            "minimum_mean_lift": rules.minimum_mean_lift,
            "min_relative_lift": rules.min_relative_lift,
            "min_absolute_lift": rules.min_absolute_lift,
            "minimum_win_rate": rules.minimum_win_rate,
            "bootstrap_lower_bound": rules.bootstrap_lower_bound,
            "bootstrap_lower_bound_relative": rules.bootstrap_lower_bound_relative,
            "confidence_level": rules.confidence_level,
            "max_predicted_to_actual_drift": rules.max_predicted_to_actual_drift,
        },
        "n_prior_comparisons_bonferroni": n_prior_comparisons,
    }


def _get_pred_to_actual(diagnostics: dict[str, Any]) -> float | None:
    decile_table = diagnostics.get("calibration_by_pred_decile", [])
    if not decile_table:
        return None
    total_actual = sum(d.get("actual_pp", 0) * d.get("n", 0) for d in decile_table)
    total_pred = sum(d.get("pred_pp", 0) * d.get("n", 0) for d in decile_table)
    return total_pred / total_actual if total_actual > 0 else None


# ── private helpers ───────────────────────────────────────────────────────────

def _ordinary_eval_frame(predictions: pd.DataFrame, eval_split: str) -> pd.DataFrame:
    if (predictions["split"] == "milestone_holdout").any():
        raise ValueError("Repeated ordinary evaluation cannot include milestone_holdout rows")
    frame = predictions[predictions["split"] == eval_split].copy()
    if frame.empty:
        raise ValueError(f"No rows found for evaluation split {eval_split!r}")
    return frame.reset_index(drop=True)


def _sample_size(row_count: int, resample_fraction: float) -> int:
    if row_count <= 0:
        raise ValueError("Cannot resample an empty frame")
    if resample_fraction <= 0:
        raise ValueError("resample_fraction must be positive")
    return max(1, int(round(row_count * resample_fraction)))
