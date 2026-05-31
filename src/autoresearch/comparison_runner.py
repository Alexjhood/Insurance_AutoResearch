"""Volatility-aware repeated evaluation and comparison workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from autoresearch.config import ProjectConfig, ensure_project_dirs
from autoresearch.evaluation.metrics import lower_is_better
from autoresearch.evaluation.metrics import infer_target_mode
from autoresearch.utils.integrity import check_integrity
from autoresearch.evaluation.resampling import (
    PromotionRules,
    bootstrap_lift_summary,
    cv_bootstrap_comparison,
    evaluate_guardrails,
    paired_comparison,
    paired_cv_comparison,
    promotion_decision,
    repeated_scores,
)
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    get_experiment,
    init_registry,
    list_comparisons,
    record_comparison,
    record_experiment_artifacts,
    update_comparison_decision,
)
from autoresearch.run_artifacts import next_iteration_dir
from autoresearch.reporting import write_comparison_html_report
from autoresearch.utils.io import read_json, write_json


def run_repeated_evaluation(config: ProjectConfig, experiment_id: str) -> dict[str, Path]:
    """Create repeated search-time scores for one registered experiment."""

    ensure_project_dirs(config)
    init_registry(config.registry_path)
    experiment = get_experiment(config.registry_path, experiment_id)
    predictions_path = _artifact_path(config, experiment_id, "predictions")
    predictions = pd.read_parquet(predictions_path)
    eval_split = config.ordinary_eval_splits[0]

    scores = repeated_scores(
        predictions,
        eval_split=eval_split,
        n_resamples=config.repeated_resamples,
        seed=config.resampling_seed,
        resample_fraction=config.resample_fraction,
        tweedie_power=config.tweedie_power,
        primary_metric=config.primary_metric,
        target_mode=config.target_mode,
    )
    summary = {
        "experiment_id": experiment_id,
        "experiment_name": experiment.get("experiment_name"),
        "eval_split": eval_split,
        "primary_metric": config.primary_metric,
        "target_mode": config.target_mode,
        "lower_is_better": lower_is_better(config.primary_metric),
        "n_resamples": config.repeated_resamples,
        "resample_fraction": config.resample_fraction,
        "seed": config.resampling_seed,
        "mean_score": float(scores["score"].mean()),
        "median_score": float(scores["score"].median()),
        "std_score": float(scores["score"].std(ddof=0)),
    }

    out_dir = Path(experiment.get("metrics_path") or _artifact_path(config, experiment_id, "predictions")).parent
    score_path = out_dir / "repeated_scores.csv"
    summary_path = out_dir / "repeated_summary.json"
    scores.to_csv(score_path, index=False)
    write_json(summary_path, summary)
    record_experiment_artifacts(
        config.registry_path,
        experiment_id,
        {"repeated_scores": score_path, "repeated_summary": summary_path},
    )
    return {"repeated_scores": score_path, "repeated_summary": summary_path}


def compare_experiments(
    config: ProjectConfig,
    champion_id: str,
    challenger_id: str,
    *,
    output_dir: Path | None = None,
    record: bool = True,
) -> dict[str, Path]:
    """Run a paired volatility-aware comparison and persist promotion evidence."""

    ensure_project_dirs(config)
    init_registry(config.registry_path)

    # ── Protected-file integrity check ────────────────────────────────────────
    integrity_violations = check_integrity(config.root, config.artifacts_dir)
    if integrity_violations:
        msg = (
            "Protected-file integrity violation — comparison blocked:\n"
            + "\n".join(integrity_violations)
            + "\nRun `autoresearch update-integrity-manifest` to accept the change."
        )
        raise ValueError(msg)

    champion_predictions = pd.read_parquet(_artifact_path(config, champion_id, "predictions"))
    challenger_predictions = pd.read_parquet(_artifact_path(config, challenger_id, "predictions"))
    champion_target_mode = infer_target_mode(champion_predictions)
    challenger_target_mode = infer_target_mode(challenger_predictions)
    if champion_target_mode != challenger_target_mode:
        raise ValueError(
            "Cannot compare experiments with different target modes: "
            f"{champion_id}={champion_target_mode}, {challenger_id}={challenger_target_mode}"
        )
    if champion_target_mode != config.target_mode:
        raise ValueError(
            f"Configured target_mode {config.target_mode!r} does not match comparison artifacts "
            f"({champion_target_mode!r})"
        )

    # Load challenger diagnostics for calibration gate check
    challenger_diagnostics = _load_diagnostics(config, challenger_id)

    # Bonferroni family size = prior comparisons against this champion plus the
    # current one (capped by the configured lookback).  Including the current
    # comparison is what makes the correction account for every attempt to beat
    # this champion; counting only priors left it one comparison too lenient.
    all_comparisons = list_comparisons(config.registry_path)
    n_prior = sum(1 for c in all_comparisons if c.get("champion_id") == champion_id)
    bonferroni_count = min(n_prior + 1, config.bonferroni_lookback)

    promotion_rules = PromotionRules(
        minimum_mean_lift=config.minimum_mean_lift,
        min_relative_lift=config.min_relative_lift,
        min_absolute_lift=config.min_absolute_lift,
        minimum_win_rate=config.minimum_win_rate,
        bootstrap_lower_bound=config.bootstrap_lower_bound,
        bootstrap_lower_bound_relative=config.bootstrap_lower_bound_relative,
        confidence_level=config.confidence_level,
        max_predicted_to_actual_drift=config.max_predicted_to_actual_drift,
        require_diagnostics=config.require_diagnostics,
        bonferroni_lookback=config.bonferroni_lookback,
    )

    gate_mode = getattr(config, "gate_mode", "cv_bootstrap")

    if gate_mode == "cv_bootstrap":
        per_resample, comparison_summary, guardrail_result, escalated = _run_cv_bootstrap_comparison(
            config, champion_id, challenger_id, champion_target_mode,
        )
    elif gate_mode == "repeated_cv":
        per_resample, comparison_summary = _run_cv_comparison(
            config, champion_id, challenger_id, champion_target_mode, bonferroni_count,
        )
        eval_chall = challenger_predictions[challenger_predictions["split"] == config.ordinary_eval_splits[0]]
        from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns
        actual_col_g, chal_pred_col_g = prediction_target_columns(eval_chall, config.target_mode)
        chal_panel_g = full_metric_panel(
            eval_chall[actual_col_g], eval_chall[chal_pred_col_g], eval_chall["exposure"],
            tweedie_power=config.tweedie_power, target_mode=config.target_mode,
        )
        guardrail_result = evaluate_guardrails(chal_panel_g, comparison_summary)
        escalated = False
    else:
        eval_split = config.ordinary_eval_splits[0]
        gate_metric = getattr(config, "gate_primary_metric", config.primary_metric)
        per_resample, comparison_summary = paired_comparison(
            champion_predictions,
            challenger_predictions,
            champion_id=champion_id,
            challenger_id=challenger_id,
            eval_split=eval_split,
            n_resamples=config.repeated_resamples,
            seed=config.resampling_seed,
            resample_fraction=config.resample_fraction,
            tweedie_power=config.tweedie_power,
            primary_metric=gate_metric,
            target_mode=config.target_mode,
        )
        comparison_summary.setdefault("gate_mode", "single_partition")
        comparison_summary.setdefault("gate_primary_metric", gate_metric)
        comparison_summary = _enrich_summary_with_kpi(
            comparison_summary, champion_predictions, challenger_predictions, config,
        )
        # Guardrails on the single-partition eval split metrics
        eval_chall = challenger_predictions[challenger_predictions["split"] == eval_split]
        from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns
        _, chal_pred_col = prediction_target_columns(eval_chall, config.target_mode)
        actual_col, _ = prediction_target_columns(eval_chall, config.target_mode)
        chal_panel = full_metric_panel(
            eval_chall[actual_col], eval_chall[chal_pred_col], eval_chall["exposure"],
            tweedie_power=config.tweedie_power, target_mode=config.target_mode,
        )
        guardrail_result = evaluate_guardrails(chal_panel, comparison_summary)
        escalated = False

    bootstrap = bootstrap_lift_summary(
        per_resample["lift"],
        iterations=config.bootstrap_iterations,
        seed=config.resampling_seed + 1,
        confidence_level=config.confidence_level,
        n_comparisons=max(1, bonferroni_count),
    )
    advisory_decision = promotion_decision(
        comparison_summary,
        bootstrap,
        promotion_rules,
        challenger_diagnostics=challenger_diagnostics,
        n_prior_comparisons=bonferroni_count,
    )

    # Comparison always starts as pending — LLM decides via record-decision.
    pending_decision = {
        "decision": "pending_llm",
        "rationale": "Awaiting LLM decision.",
        "promoted": False,
        "checks": advisory_decision.get("checks", {}),
        "effect_size": advisory_decision.get("effect_size", {}),
        "thresholds": advisory_decision.get("thresholds", {}),
        "advisory_decision": advisory_decision.get("decision"),
        "guardrail_passed": guardrail_result["passed"],
        "guardrail_failures": guardrail_result["failures"],
        "escalated": escalated,
    }

    metric_lift_table = _build_metric_lift_table(
        config, champion_id, challenger_id, per_resample, comparison_summary,
    )

    comparison_id = _comparison_id(champion_id, challenger_id)
    out_dir = output_dir or (next_iteration_dir(config, comparison_id) / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "comparison_id": comparison_id,
        "gate_mode": gate_mode,
        "comparison_summary": comparison_summary,
        "bootstrap_summary": bootstrap,
        "promotion_decision": pending_decision,
        "advisory_promotion_decision": advisory_decision,
        "guardrail_result": guardrail_result,
        "metric_lift_table": metric_lift_table,
        "escalated": escalated,
    }
    per_resample_path = out_dir / "paired_resample_scores.csv"
    comparison_path = out_dir / "comparison_summary.json"
    bootstrap_path = out_dir / "bootstrap_summary.json"
    decision_path = out_dir / "promotion_decision.json"
    report_path = out_dir / "promotion_report.json"
    html_report_path = out_dir / "comparison_report.html"

    per_resample.to_csv(per_resample_path, index=False)
    write_json(comparison_path, comparison_summary)
    write_json(bootstrap_path, bootstrap)
    write_json(decision_path, pending_decision)
    write_json(report_path, payload)
    write_comparison_html_report(
        config=config,
        comparison_id=comparison_id,
        champion_id=champion_id,
        challenger_id=challenger_id,
        comparison_summary=comparison_summary,
        bootstrap_summary=bootstrap,
        decision=pending_decision,
        metric_lift_table=metric_lift_table,
        per_partition=per_resample,
        output_path=html_report_path,
    )

    artifacts = {
        "paired_resample_scores": per_resample_path,
        "comparison_summary": comparison_path,
        "bootstrap_summary": bootstrap_path,
        "promotion_decision": decision_path,
        "promotion_report": report_path,
        "html_report": html_report_path,
    }
    if record:
        record_comparison(
            config.registry_path,
            comparison_id=comparison_id,
            champion_id=champion_id,
            challenger_id=challenger_id,
            paired_summary=comparison_summary,
            bootstrap_summary=bootstrap,
            promotion_decision="pending_llm",
            promotion_rationale="Awaiting LLM decision.",
            artifacts=artifacts,
            guardrail_status=guardrail_result,
        )
    return artifacts


def screen_challenger_single_split(
    config: ProjectConfig,
    champion_id: str,
    challenger_id: str,
) -> dict[str, Any]:
    """Cheap full search-validation screen before CV bootstrap.

    This is a low hurdle, not a promotion gate. It rejects challengers only
    when they are clearly worse on the complete ordinary eval split. Similar
    or better challengers continue to the normal CV/bootstrap comparison.
    """

    from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns

    eval_split = config.ordinary_eval_splits[0]
    gate_metric = getattr(config, "gate_primary_metric", config.primary_metric)
    min_abs = float(getattr(config, "screening_min_absolute_lift", -0.001))
    min_rel = float(getattr(config, "screening_min_relative_lift", -0.002))

    champion_predictions = pd.read_parquet(_artifact_path(config, champion_id, "predictions"))
    challenger_predictions = pd.read_parquet(_artifact_path(config, challenger_id, "predictions"))
    target_mode = infer_target_mode(challenger_predictions, config.target_mode)
    champion = champion_predictions[champion_predictions["split"] == eval_split].copy()
    challenger = challenger_predictions[challenger_predictions["split"] == eval_split].copy()

    base = {
        "gate_mode": "single_split_screen",
        "eval_split": eval_split,
        "gate_metric": gate_metric,
        "target_mode": target_mode,
        "champion_id": champion_id,
        "challenger_id": challenger_id,
        "min_absolute_lift": min_abs,
        "min_relative_lift": min_rel,
    }
    if champion.empty or challenger.empty:
        return {**base, "passed": False, "reason": f"Missing rows for eval split {eval_split!r}"}

    actual_col, champion_pred_col = prediction_target_columns(champion, target_mode)
    _, challenger_pred_col = prediction_target_columns(challenger, target_mode)
    paired = champion[["record_id", actual_col, champion_pred_col, "exposure"]].merge(
        challenger[["record_id", challenger_pred_col]],
        on="record_id",
        how="inner",
        suffixes=("_champion", "_challenger"),
    )
    if paired.empty:
        return {**base, "passed": False, "reason": "Champion and challenger predictions have no overlapping eval rows"}

    champion_merged_col = f"{champion_pred_col}_champion"
    challenger_merged_col = f"{challenger_pred_col}_challenger"
    if champion_pred_col != challenger_pred_col:
        champion_merged_col = champion_pred_col
        challenger_merged_col = challenger_pred_col

    champion_panel = full_metric_panel(
        paired[actual_col],
        paired[champion_merged_col],
        paired["exposure"],
        tweedie_power=config.tweedie_power,
        target_mode=target_mode,
    )
    challenger_panel = full_metric_panel(
        paired[actual_col],
        paired[challenger_merged_col],
        paired["exposure"],
        tweedie_power=config.tweedie_power,
        target_mode=target_mode,
    )
    champion_score = float(champion_panel[gate_metric])
    challenger_score = float(challenger_panel[gate_metric])
    lower_better = lower_is_better(gate_metric)
    lift = champion_score - challenger_score if lower_better else challenger_score - champion_score
    relative_lift = lift / max(abs(champion_score), 1e-12)
    finite = all(pd.notna(v) for v in (champion_score, challenger_score, lift, relative_lift))
    passed = bool(finite and lift >= min_abs and relative_lift >= min_rel)
    reason = (
        "passed low-hurdle single-split screen"
        if passed
        else (
            f"single-split lift {lift:.6g} / relative {relative_lift:.6g} "
            f"below low hurdle ({min_abs:.6g}, {min_rel:.6g})"
        )
    )
    return {
        **base,
        "passed": passed,
        "reason": reason,
        "champion_score": champion_score,
        "challenger_score": challenger_score,
        "lift": float(lift),
        "relative_lift": float(relative_lift),
        "overlap_rows": int(len(paired)),
        "champion_metric_panel": champion_panel,
        "challenger_metric_panel": challenger_panel,
    }


def compare_against_current_champion(
    config: ProjectConfig,
    challenger_id: str,
    *,
    auto_promote: bool = False,
) -> dict[str, Path]:
    """Compare challenger against official champion.

    When ``auto_promote=False`` (default), the comparison writes decision=pending_llm
    and returns — the LLM must call ``record_decision`` to finalise.

    When ``auto_promote=True`` (legacy behaviour), the mechanical advisory decision
    is used to auto-promote if all advisory gates pass.
    """
    from autoresearch.experiment_registry.registry import set_official_champion
    from autoresearch.milestone import evaluate_on_holdout

    official = get_official_champion(config.registry_path)
    if official is None:
        raise ValueError(
            "Official champion is not initialised. Run init-official-champion first."
        )
    champion_id = official["champion_id"]
    if champion_id == challenger_id:
        raise ValueError("Challenger is already the current champion")

    artifacts = compare_experiments(config, champion_id, challenger_id)

    if auto_promote:
        report = read_json(artifacts["promotion_report"])
        advisory = report.get("advisory_promotion_decision", {})
        comparison_id = report.get("comparison_id", "")
        if advisory.get("decision") == "promote":
            branch_id = official.get("branch_id", "main") if official else "main"
            set_official_champion(
                config.registry_path,
                champion_id=challenger_id,
                branch_id=branch_id,
                reason=advisory.get("rationale", "Promoted via compare-to-champion"),
                action="promoted",
                comparison_id=comparison_id,
            )
            evaluate_on_holdout(config, challenger_id, comparison_id)

    return artifacts


def _load_guardrail_status(comp: dict[str, Any]) -> dict[str, Any]:
    """Decode the guardrail status stored on a comparison registry row.

    The canonical source is the ``guardrail_status`` column (JSON string) written
    at comparison time — NOT the HTML report (which is not JSON).
    """
    raw = comp.get("guardrail_status")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def record_decision(
    config: ProjectConfig,
    comparison_id: str,
    *,
    decision: str,
    rationale: str,
) -> dict[str, Any]:
    """Record the LLM's promote/reject verdict for a pending comparison.

    ``decision`` must be ``"promote"`` or ``"reject"``.

    On ``promote``: re-evaluates guardrails and blocks if any hard fail is detected.
    On pass: updates the champion, fires holdout, persists the decision.
    On ``reject``: persists the rationale without promoting.

    Returns the final decision dict.
    """
    from autoresearch.experiment_registry.registry import (
        set_official_champion,
        list_comparisons,
        list_proposals,
        upsert_research_node,
        update_proposal_status,
    )
    from autoresearch.milestone import evaluate_on_holdout
    from datetime import datetime, timezone

    decision = decision.lower().strip()
    if decision not in ("promote", "reject"):
        raise ValueError(f"decision must be 'promote' or 'reject', got {decision!r}")

    # Find the comparison record to locate artifacts
    all_comps = list_comparisons(config.registry_path)
    comp = next((c for c in all_comps if c["comparison_id"] == comparison_id), None)
    if comp is None:
        raise ValueError(f"Comparison {comparison_id!r} not found in registry")

    champion_id = comp["champion_id"]
    challenger_id = comp["challenger_id"]
    guardrail_result: dict[str, Any] = _load_guardrail_status(comp)

    # Link back to the originating proposal (if this comparison came from a cycle)
    proposal = next(
        (p for p in list_proposals(config.registry_path) if p.get("comparison_id") == comparison_id),
        None,
    )
    proposal_id = proposal.get("proposal_id") if proposal else None

    official = get_official_champion(config.registry_path)
    branch_id = official.get("branch_id", "main") if official else "main"

    if decision == "promote":
        if not guardrail_result.get("passed", True):
            failures = guardrail_result.get("failures", [])
            raise ValueError(
                f"Promotion blocked by hard guardrails: {', '.join(failures)}. "
                "Fix the underlying issues before promoting."
            )
        set_official_champion(
            config.registry_path,
            champion_id=challenger_id,
            branch_id=branch_id,
            reason=rationale,
            action="promoted",
            comparison_id=comparison_id,
            **({"proposal_id": proposal_id} if proposal_id else {}),
        )
        evaluate_on_holdout(config, challenger_id, comparison_id)
        if proposal_id:
            update_proposal_status(
                config.registry_path, proposal_id, "promoted",
                comparison_id=comparison_id, notes=rationale,
            )
            upsert_research_node(
                config.registry_path,
                node_id=proposal_id,
                proposal_id=proposal_id,
                experiment_id=challenger_id,
                comparison_id=comparison_id,
                status="promoted",
                outcome_type="promoted",
                guidance=rationale,
            )
    else:  # reject — retain the incumbent champion
        set_official_champion(
            config.registry_path,
            champion_id=champion_id,
            branch_id=branch_id,
            reason=rationale,
            action="retained",
            comparison_id=comparison_id,
            **({"proposal_id": proposal_id} if proposal_id else {}),
        )
        if proposal_id:
            update_proposal_status(
                config.registry_path, proposal_id, "inconclusive",
                comparison_id=comparison_id, notes=rationale,
            )
            upsert_research_node(
                config.registry_path,
                node_id=proposal_id,
                proposal_id=proposal_id,
                experiment_id=challenger_id,
                comparison_id=comparison_id,
                status="rejected",
                outcome_type="llm_rejected",
                guidance=rationale,
            )

    decided_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    update_comparison_decision(
        config.registry_path,
        comparison_id,
        decision=decision,
        rationale=rationale,
        decided_by="llm",
        decided_at=decided_at,
        guardrail_status=guardrail_result or None,
    )

    final_decision = {
        "decision": decision,
        "rationale": rationale,
        "decided_by": "llm",
        "decided_at": decided_at,
        "promoted": decision == "promote",
        "guardrail_passed": guardrail_result.get("passed", True),
        "guardrail_failures": guardrail_result.get("failures", []),
    }

    # Update the decision.json artifact
    decision_artifact = Path(comp.get("promotion_decision_path", ""))
    if decision_artifact.exists():
        existing = read_json(decision_artifact)
        existing.update(final_decision)
        write_json(decision_artifact, existing)

    # Re-render the HTML comparison report so the verdict + rationale are captured.
    _finalise_comparison_report(config, comp, final_decision)

    return {
        "comparison_id": comparison_id,
        "decision": decision,
        "rationale": rationale,
        "decided_by": "llm",
        "decided_at": decided_at,
        "proposal_id": proposal_id,
        "guardrail_result": guardrail_result,
    }


def _finalise_comparison_report(
    config: ProjectConfig,
    comp: dict[str, Any],
    final_decision: dict[str, Any],
) -> None:
    """Re-render the comparison HTML report with the recorded LLM verdict.

    Best-effort: a failure to re-render must not undo the recorded decision.
    """
    try:
        decision_artifact = Path(comp.get("promotion_decision_path", ""))
        payload_path = decision_artifact.parent / "promotion_report.json"
        if not payload_path.exists():
            return
        payload = read_json(payload_path)
        comparison_summary = payload.get("comparison_summary", {})
        bootstrap_summary = payload.get("bootstrap_summary", {})
        metric_lift_table = payload.get("metric_lift_table", [])
        advisory = payload.get("advisory_promotion_decision", {})

        # Carry advisory checks/thresholds into the final decision so the
        # (advisory) gate panel still renders.
        render_decision = dict(final_decision)
        render_decision.setdefault("checks", advisory.get("checks", {}))
        render_decision.setdefault("thresholds", advisory.get("thresholds", {}))
        render_decision["advisory_decision"] = advisory.get("decision")

        per_partition = None
        scores_path = Path(comp.get("paired_scores_path", ""))
        if scores_path.exists():
            per_partition = pd.read_csv(scores_path)

        html_path = Path(comp.get("report_path", ""))
        if not html_path or not str(html_path):
            return
        write_comparison_html_report(
            config=config,
            comparison_id=comp["comparison_id"],
            champion_id=comp["champion_id"],
            challenger_id=comp["challenger_id"],
            comparison_summary=comparison_summary,
            bootstrap_summary=bootstrap_summary,
            decision=render_decision,
            metric_lift_table=metric_lift_table,
            per_partition=per_partition,
            output_path=html_path,
        )
    except Exception:
        pass  # report re-render is best-effort; the decision is already recorded



def _build_metric_lift_table(
    config: ProjectConfig,
    champion_id: str,
    challenger_id: str,
    per_partition: Any,
    comparison_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a tidy per-metric lift table for the report exhibit.

    For CV mode, lifts are averaged across partitions from per_partition.
    For single-partition mode, lifts are computed directly from the
    prediction parquets over the eval split.
    """

    from autoresearch.evaluation.metrics import (
        full_metric_panel, prediction_target_columns, HIGHER_IS_BETTER_METRICS,
    )

    gate_metric = getattr(config, "gate_primary_metric", "rank_gini_weighted")
    gate_mode = comparison_summary.get("gate_mode", "single_partition")

    rows: list[dict[str, Any]] = []

    if gate_mode in ("cv_bootstrap", "repeated_cv") and per_partition is not None and not per_partition.empty:
        # Average champion/challenger scores per metric across partitions
        champ_cols = [c for c in per_partition.columns if c.startswith("champ_")]
        for champ_col in champ_cols:
            metric = champ_col[len("champ_"):]
            chal_col = f"chal_{metric}"
            if chal_col not in per_partition.columns:
                continue
            c_mean = float(per_partition[champ_col].mean())
            h_mean = float(per_partition[chal_col].mean())
            higher_better = metric in HIGHER_IS_BETTER_METRICS
            lift = h_mean - c_mean if higher_better else c_mean - h_mean
            lifts = (per_partition[chal_col] - per_partition[champ_col])
            if not higher_better:
                lifts = -lifts
            rows.append({
                "metric": metric,
                "champion_score": round(c_mean, 6),
                "challenger_score": round(h_mean, 6),
                "mean_lift": round(lift, 6),
                "lift_std": round(float(lifts.std(ddof=0)), 6),
                "win_rate": round(float((lifts > 0).mean()), 4),
                "higher_is_better": higher_better,
                "is_gate_metric": metric == gate_metric,
                "is_kpi_metric": metric == "gini_weighted",
                "per_partition_lifts": [round(float(x), 6) for x in lifts.tolist()],
            })
    else:
        # Single-partition: compute full metric panel on the eval split
        try:
            eval_split = config.ordinary_eval_splits[0]
            champ_preds = pd.read_parquet(_artifact_path(config, champion_id, "predictions"))
            chal_preds = pd.read_parquet(_artifact_path(config, challenger_id, "predictions"))
            champ_eval = champ_preds[champ_preds["split"] == eval_split]
            chal_eval = chal_preds[chal_preds["split"] == eval_split]
            target_mode = config.target_mode
            actual_col, champ_pred_col = prediction_target_columns(champ_eval, target_mode)
            _, chal_pred_col = prediction_target_columns(chal_eval, target_mode)
            champ_m = full_metric_panel(
                champ_eval[actual_col], champ_eval[champ_pred_col], champ_eval["exposure"],
                tweedie_power=config.tweedie_power, target_mode=target_mode,
            )
            chal_m = full_metric_panel(
                chal_eval[actual_col], chal_eval[chal_pred_col], chal_eval["exposure"],
                tweedie_power=config.tweedie_power, target_mode=target_mode,
            )
            per_resample_lifts: dict[str, list[float]] = {}
            if per_partition is not None and not per_partition.empty and "lift" in per_partition.columns:
                per_resample_lifts[gate_metric] = [float(x) for x in per_partition["lift"].tolist()]
            for metric, c_val in champ_m.items():
                if not isinstance(c_val, float):
                    continue
                h_val = chal_m.get(metric, float("nan"))
                higher_better = metric in HIGHER_IS_BETTER_METRICS
                lift = h_val - c_val if higher_better else c_val - h_val
                rows.append({
                    "metric": metric,
                    "champion_score": round(c_val, 6),
                    "challenger_score": round(float(h_val), 6),
                    "mean_lift": round(float(lift), 6),
                    "lift_std": None,
                    "win_rate": None,
                    "higher_is_better": higher_better,
                    "is_gate_metric": metric == gate_metric,
                    "is_kpi_metric": metric == "gini_weighted",
                    "per_partition_lifts": per_resample_lifts.get(metric, []),
                })
        except Exception:
            pass  # Report will show empty metric table rather than crashing

    # Sort: gate metric first, KPI second, rest alphabetically
    def _sort_key(r: dict) -> tuple:
        return (0 if r["is_gate_metric"] else 1 if r["is_kpi_metric"] else 2, r["metric"])

    rows.sort(key=_sort_key)
    return rows


def _run_cv_bootstrap_comparison(
    config: ProjectConfig,
    champion_id: str,
    challenger_id: str,
    target_mode: str,
) -> tuple[Any, dict[str, Any], dict[str, Any], bool]:
    """Run the cv_bootstrap comparison path with caching and close-call escalation.

    Returns (per_sample_df, comparison_summary, guardrail_result, escalated).
    """
    from autoresearch.cv_cache import get_or_build_fold_predictions
    from autoresearch.data.holdout_vault import load_search_dataset
    from autoresearch.data.preprocessing import apply_claim_capping
    from autoresearch.models.dispatcher import RAW_CLAIM_COST
    from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns

    gate_metric = getattr(config, "gate_primary_metric", "gini_weighted")
    bootstrap_per_fold = getattr(config, "bootstrap_per_fold", 20)
    escalation_lo = getattr(config, "escalation_win_rate_low", 0.40)
    escalation_hi = getattr(config, "escalation_win_rate_high", 0.60)
    escalation_n = getattr(config, "escalation_partitions", 2)

    frame = load_search_dataset(config.processed_dir, config.agent_dataset_name)
    frame, _ = apply_claim_capping(
        frame,
        claim_column=RAW_CLAIM_COST,
        threshold=config.claim_cap_threshold,
        enabled=config.claim_capping_enabled,
    )

    # Base partition (index 0)
    champ_folds_0 = get_or_build_fold_predictions(config, champion_id, 0, frame)
    chal_folds_0 = get_or_build_fold_predictions(config, challenger_id, 0, frame)

    champion_fp: dict[int, list] = {0: champ_folds_0}
    challenger_fp: dict[int, list] = {0: chal_folds_0}

    per_sample, summary = cv_bootstrap_comparison(
        champion_fold_predictions=champion_fp,
        challenger_fold_predictions=challenger_fp,
        gate_primary_metric=gate_metric,
        bootstrap_per_fold=bootstrap_per_fold,
        tweedie_power=config.tweedie_power,
        seed=getattr(config, "cv_seed", config.resampling_seed),
        target_mode=target_mode,
    )
    summary["champion_id"] = champion_id
    summary["challenger_id"] = challenger_id

    escalated = False
    win_rate = summary["challenger_win_rate"]

    if escalation_lo <= win_rate <= escalation_hi and escalation_n > 0:
        # Close call — add more partitions
        pre_escalation_win_rate = win_rate
        for p_idx in range(1, escalation_n + 1):
            champion_fp[p_idx] = get_or_build_fold_predictions(config, champion_id, p_idx, frame)
            challenger_fp[p_idx] = get_or_build_fold_predictions(config, challenger_id, p_idx, frame)

        per_sample, summary = cv_bootstrap_comparison(
            champion_fold_predictions=champion_fp,
            challenger_fold_predictions=challenger_fp,
            gate_primary_metric=gate_metric,
            bootstrap_per_fold=bootstrap_per_fold,
            tweedie_power=config.tweedie_power,
            seed=getattr(config, "cv_seed", config.resampling_seed),
            target_mode=target_mode,
        )
        summary["champion_id"] = champion_id
        summary["challenger_id"] = challenger_id
        summary["pre_escalation_win_rate"] = pre_escalation_win_rate
        escalated = True

    summary["escalated"] = escalated

    # Empirical lift CI across all (fold × bootstrap) samples — feeds the
    # "clearly worse" guardrail (CI entirely below zero ⇒ block).
    if "lift" in per_sample.columns and len(per_sample) > 1:
        lifts = per_sample["lift"]
        summary["lift_ci_lower"] = float(lifts.quantile(0.05))
        summary["lift_ci_upper"] = float(lifts.quantile(0.95))

    # Guardrails: evaluate on the average challenger metrics across all bootstrap samples
    chal_gate_col = f"chal_{gate_metric}"
    chal_gini_col = "chal_gini_weighted"
    chal_ratio_col = "chal_predicted_to_actual_ratio"
    chal_total_col = "chal_total_predicted_target"

    challenger_agg: dict[str, Any] = {}
    if chal_gini_col in per_sample.columns:
        challenger_agg["gini_weighted"] = float(per_sample[chal_gini_col].mean())
    if chal_ratio_col in per_sample.columns:
        challenger_agg["predicted_to_actual_ratio"] = float(per_sample[chal_ratio_col].mean())
    if chal_total_col in per_sample.columns:
        challenger_agg["total_predicted_target"] = float(per_sample[chal_total_col].mean())
    elif "chal_total_predicted_claim_cost" in per_sample.columns:
        challenger_agg["total_predicted_target"] = float(per_sample["chal_total_predicted_claim_cost"].mean())

    guardrail_result = evaluate_guardrails(challenger_agg, summary)
    return per_sample, summary, guardrail_result, escalated


def _run_cv_comparison(
    config: ProjectConfig,
    champion_id: str,
    challenger_id: str,
    target_mode: str,
    bonferroni_count: int,
) -> tuple[Any, dict[str, Any]]:
    """Run the repeated-CV paired comparison path."""

    from autoresearch.cv_factory import build_model_factory_from_experiment
    from autoresearch.data.holdout_vault import load_search_dataset
    from autoresearch.data.preprocessing import apply_claim_capping
    from autoresearch.models.dispatcher import RAW_CLAIM_COST

    gate_metric = getattr(config, "gate_primary_metric", "rank_gini_weighted")
    n_folds = getattr(config, "cv_folds", 4)
    n_repeats = getattr(config, "cv_n_repeats", 4)
    cv_seed = getattr(config, "cv_seed", config.resampling_seed)

    # Load the full search partition (train + search_validation; no holdout)
    frame = load_search_dataset(config.processed_dir, config.agent_dataset_name)
    # Apply the fixed claim cap so the active target column exists, mirroring
    # the experiment runner (models read claim_cost_capped_active, not the raw
    # observed column).  The cap is a fixed product decision applied uniformly.
    frame, _ = apply_claim_capping(
        frame,
        claim_column=RAW_CLAIM_COST,
        threshold=config.claim_cap_threshold,
        enabled=config.claim_capping_enabled,
    )
    fold_path = config.splits_dir / "split_pack_folds.parquet"
    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fold assignments not found at {fold_path}. "
            "Run `autoresearch prepare-data` to generate them."
        )
    fold_assignments = pd.read_parquet(fold_path)

    champion_factory = build_model_factory_from_experiment(config, champion_id)
    challenger_factory = build_model_factory_from_experiment(config, challenger_id)

    return paired_cv_comparison(
        frame,
        fold_assignments,
        champion_factory=champion_factory,
        challenger_factory=challenger_factory,
        champion_id=champion_id,
        challenger_id=challenger_id,
        n_folds=n_folds,
        n_repeats=n_repeats,
        tweedie_power=config.tweedie_power,
        gate_primary_metric=gate_metric,
        seed=cv_seed,
        target_mode=target_mode,
    )


def _enrich_summary_with_kpi(
    summary: dict[str, Any],
    champion_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, Any]:
    """Add KPI (gini_weighted) scores to a single-partition summary dict.

    Enables the sign-agreement check in promotion_decision even when the
    gate runs on a non-Gini metric (e.g. rank_gini_weighted).
    """

    from autoresearch.evaluation.metrics import full_metric_panel, prediction_target_columns

    eval_split = config.ordinary_eval_splits[0]
    champ_eval = champion_predictions[champion_predictions["split"] == eval_split]
    chal_eval = challenger_predictions[challenger_predictions["split"] == eval_split]

    if champ_eval.empty or chal_eval.empty:
        return summary

    try:
        target_mode = config.target_mode
        actual_col, champ_pred_col = prediction_target_columns(champ_eval, target_mode)
        _, chal_pred_col = prediction_target_columns(chal_eval, target_mode)
        champ_kpi = full_metric_panel(
            champ_eval[actual_col], champ_eval[champ_pred_col], champ_eval["exposure"],
            tweedie_power=config.tweedie_power, target_mode=target_mode,
        ).get("gini_weighted")
        chal_kpi = full_metric_panel(
            chal_eval[actual_col], chal_eval[chal_pred_col], chal_eval["exposure"],
            tweedie_power=config.tweedie_power, target_mode=target_mode,
        ).get("gini_weighted")
        enriched = dict(summary)
        enriched["champion_kpi_score"] = champ_kpi
        enriched["challenger_kpi_score"] = chal_kpi
        if champ_kpi is not None and chal_kpi is not None:
            enriched["kpi_lift_positive"] = bool(float(chal_kpi) > float(champ_kpi))
        return enriched
    except Exception:
        return summary


def _artifact_path(config: ProjectConfig, experiment_id: str, artifact_type: str) -> Path:
    from autoresearch.experiment_registry.registry import list_artifacts

    artifacts = list_artifacts(config.registry_path, experiment_id)
    for artifact in artifacts:
        if artifact["artifact_type"] == artifact_type:
            return Path(artifact["path"])
    raise ValueError(f"Experiment {experiment_id} has no {artifact_type!r} artifact")


def _load_diagnostics(config: ProjectConfig, experiment_id: str) -> dict[str, Any] | None:
    """Load diagnostics.json for an experiment if it exists."""

    try:
        diag_path = _artifact_path(config, experiment_id, "diagnostics")
        return read_json(diag_path)
    except Exception:
        for diag_path in config.artifacts_dir.glob(f"iterations/*/experiment/diagnostics.json"):
            if experiment_id in str(diag_path.parent):
                return read_json(diag_path)
        return None


def _comparison_id(champion_id: str, challenger_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{_short(champion_id)}_vs_{_short(challenger_id)}"


def _short(value: str) -> str:
    return value.replace(" ", "_")[:48]
