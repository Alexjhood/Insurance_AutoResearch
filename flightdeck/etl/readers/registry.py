"""Read one run's ``registry.sqlite`` into a list of :class:`Experiment`.

Sources joined per DATA.md §2.3: ``experiments`` ⋈ ``proposals`` ⋈
``comparisons`` (challenger_id) ⋈ ``research_log_entries`` ⋈ ``research_nodes``.
Every read is defensive — a locked/partial DB degrades to ``[]`` + a warning
(quirk 6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..schema import (
    Comparison,
    DecisionMeta,
    Experiment,
    ExperimentMetrics,
    Lift,
    Proposal,
    RepairAttempt,
    Screening,
    TreeNode,
)
from ..util import (
    Warnings,
    has_table,
    load_json,
    loads_or_none,
    ro_connect,
    safe_query,
    table_columns,
    to_float,
    to_int,
)

_SEED_PREFIXES = ("orchestration_delegation_seed", "orchestration_seed")


def is_seed_name(name: str) -> bool:
    return any(name.startswith(p) for p in _SEED_PREFIXES)


def is_baseline_row(name: str, model_family: str) -> bool:
    """DATA.md: model_family 'constant'/'global_mean' or name contains global_mean."""
    return model_family in ("constant", "global_mean") or "global_mean" in name


def _row_get(row: Any, key: str) -> Any:
    if row is None:
        return None
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _split_panel(metrics_json_doc: Any) -> Optional[dict]:
    """Pull the search-validation split panel from a metrics.json document."""
    if not isinstance(metrics_json_doc, dict):
        return None
    splits = metrics_json_doc.get("split_metrics")
    if not isinstance(splits, list):
        return None
    prefer = metrics_json_doc.get("ordinary_eval_splits") or ["search_validation"]
    prefer_name = prefer[0] if prefer else "search_validation"
    chosen = None
    for s in splits:
        if isinstance(s, dict) and s.get("split") == prefer_name:
            chosen = s
            break
    if chosen is None and splits and isinstance(splits[0], dict):
        chosen = splits[0]
    return chosen


def _resolve_metrics_file(metrics_path: Optional[str], run_dir: Path,
                          warnings: Warnings) -> Optional[dict]:
    """Read the recorded metrics.json, rebased into this run dir when needed.

    The stored ``metrics_path`` is an absolute path into the canonical
    ``artifacts/tracks/…`` run. DATA.md says to read it only "if within run
    dir"; we rebase its ``iterations/…`` suffix onto ``run_dir`` (which keeps
    the read inside the orchestration folder) and fall back to callers'
    in-DB panels if that file is absent.
    """
    if not metrics_path:
        return None
    candidates: list[Path] = []
    marker = "/iterations/"
    if marker in metrics_path:
        suffix = "iterations/" + metrics_path.split(marker, 1)[1]
        candidates.append(run_dir / suffix)
    raw = Path(metrics_path)
    # Only trust the raw absolute path if it lands inside this run dir.
    try:
        raw.relative_to(run_dir)
        candidates.append(raw)
    except ValueError:
        pass
    for cand in candidates:
        if cand.exists():
            doc = load_json(cand, warnings, label=str(cand))
            panel = _split_panel(doc)
            if panel is not None:
                return panel
    return None


def _metrics(row: Any, run_dir: Path, screening_panel: Optional[dict],
             champion_panel: Optional[dict], warnings: Warnings) -> ExperimentMetrics:
    fit = to_float(_row_get(row, "fit_wall_seconds"))
    panel = (
        _resolve_metrics_file(_row_get(row, "metrics_path"), run_dir, warnings)
        or screening_panel
        or champion_panel
    )
    if not panel:
        return ExperimentMetrics(None, None, None, None, fit)
    return ExperimentMetrics(
        gini_weighted=to_float(panel.get("gini_weighted")),
        rank_gini_weighted=to_float(panel.get("rank_gini_weighted")),
        asym_pricing_loss=to_float(panel.get("asym_pricing_loss")),
        calibration_ratio=to_float(panel.get("predicted_to_actual_ratio")),
        fit_wall_seconds=fit,
    )


def _proposal(cfg: Optional[dict], prow: Any) -> tuple[Proposal, Any, str]:
    """Return (Proposal, recipe, target_strategy) from a proposals row + config."""
    cfg = cfg or {}
    exp_cfg = cfg.get("experiment_config") or {}
    model = cfg.get("model") or exp_cfg.get("model") or {}
    recipe: Any = None
    if isinstance(model, dict):
        if model.get("recipe") is not None:
            recipe = model["recipe"]
        elif model.get("script_path"):
            recipe = {"script": True, "path": model["script_path"]}
    target_strategy = cfg.get("target_strategy") or exp_cfg.get("target_strategy") or ""

    def pick(*keys: str) -> Optional[str]:
        for k in keys:
            v = cfg.get(k)
            if v not in (None, ""):
                return v
        return None

    proposal = Proposal(
        hypothesis=None,  # filled from research_nodes later (fallback rationale here)
        change_summary=_row_get(prow, "change_summary"),
        expected_benefit=_row_get(prow, "expected_benefit"),
        key_risk=_row_get(prow, "key_risk"),
        exploration_axis=pick("exploration_axis"),
        approach_family=pick("approach_family"),
    )
    return proposal, recipe, target_strategy


def _comparison(crow: Any, node_metrics: Optional[dict]) -> Comparison:
    boot = loads_or_none(_row_get(crow, "bootstrap_summary")) or {}
    node_metrics = node_metrics or {}
    # cv mean lift: bootstrap_summary.mean_lift, else research node cv_mean_lift.
    cv_mean_lift = to_float(boot.get("mean_lift"))
    if cv_mean_lift is None:
        cv_mean_lift = to_float(node_metrics.get("cv_mean_lift"))
    # fold win rate is not in bootstrap_summary in observed data; the true
    # per-fold rate lives in research_nodes.metrics_json.cv_win_rate.
    fold_win_rate = to_float(node_metrics.get("cv_win_rate"))
    if fold_win_rate is None:
        for k in ("challenger_win_rate", "fold_win_rate", "win_rate"):
            if boot.get(k) is not None:
                fold_win_rate = to_float(boot.get(k))
                break
    return Comparison(
        comparison_id=_row_get(crow, "comparison_id"),
        champion_id=_row_get(crow, "champion_id"),
        cv_mean_lift=cv_mean_lift,
        fold_win_rate=fold_win_rate,
        decision=_row_get(crow, "decision"),
        decided_by=_row_get(crow, "decided_by"),
        decision_reason_code=_row_get(crow, "decision_reason_code"),
        decision_rationale=_row_get(crow, "decision_rationale"),
        guardrail_status=_row_get(crow, "guardrail_status"),
    )


def _repairs(conn, experiment_id: str, warnings: Warnings) -> list[RepairAttempt]:
    if not has_table(conn, "session_events"):
        return []
    rows = safe_query(
        conn,
        "SELECT event_type, message, details_json FROM session_events "
        "WHERE experiment_id=? AND lower(event_type) LIKE '%repair%' ORDER BY event_id",
        (experiment_id,),
        warnings=warnings,
        label="session_events repairs",
    )
    out: list[RepairAttempt] = []
    for i, r in enumerate(rows, start=1):
        details = loads_or_none(_row_get(r, "details_json")) or {}
        kind = details.get("repair_kind") if isinstance(details, dict) else None
        failed = details.get("failed_checks") if isinstance(details, dict) else None
        out.append(
            RepairAttempt(
                attempt=to_int(details.get("attempt")) or i if isinstance(details, dict) else i,
                kind=kind,
                failed_checks=list(failed) if isinstance(failed, list) else [],
                resolved=bool(details.get("resolved")) if isinstance(details, dict) else False,
                raw=details,
            )
        )
    return out


def read_registry_experiments(
    registry_path: Path,
    run_dir: Path,
    delegation_id: Optional[str],
    warnings: Warnings,
) -> list[Experiment]:
    """Extract experiments from one registry. ``delegation_id`` None = consolidation."""
    experiments: list[Experiment] = []
    with ro_connect(registry_path, warnings) as conn:
        if conn is None:
            return experiments
        if not has_table(conn, "experiments"):
            warnings.add(f"registry has no experiments table: {registry_path}")
            return experiments

        proposals = {
            _row_get(r, "experiment_id"): r
            for r in safe_query(conn, "SELECT * FROM proposals", warnings=warnings)
            if _row_get(r, "experiment_id")
        }
        nodes: dict[str, Any] = {}
        if has_table(conn, "research_nodes"):
            for r in safe_query(conn, "SELECT * FROM research_nodes", warnings=warnings):
                eid = _row_get(r, "experiment_id")
                if eid and eid not in nodes:
                    nodes[eid] = r
        logs: dict[str, Any] = {}
        if has_table(conn, "research_log_entries"):
            for r in safe_query(conn, "SELECT * FROM research_log_entries", warnings=warnings):
                eid = _row_get(r, "experiment_id")
                if eid:
                    logs[eid] = r
        # comparisons keyed by challenger, keeping the latest (by created_at).
        comparisons: dict[str, Any] = {}
        champion_panels: dict[str, dict] = {}
        for r in safe_query(conn, "SELECT * FROM comparisons ORDER BY created_at",
                            warnings=warnings):
            ch = _row_get(r, "challenger_id")
            if ch:
                comparisons[ch] = r  # later rows overwrite → latest wins
            scr = loads_or_none(_row_get(r, "bootstrap_summary"))
            _ = scr  # noqa: F841 (documented shape; panels come from nodes below)
        # champion metric panels from any node screening_json (for baselines
        # that were never a challenger themselves).
        for r in nodes.values():
            sj = loads_or_none(_row_get(r, "screening_json"))
            if isinstance(sj, dict):
                cid = sj.get("champion_id")
                panel = sj.get("champion_metric_panel")
                if cid and isinstance(panel, dict) and cid not in champion_panels:
                    champion_panels[cid] = panel

        for row in safe_query(conn, "SELECT * FROM experiments ORDER BY created_at",
                              warnings=warnings):
            eid = _row_get(row, "experiment_id")
            if not eid:
                continue
            name = _row_get(row, "experiment_name") or ""
            model_family = _row_get(row, "model_family") or ""
            prow = proposals.get(eid)
            cfg = loads_or_none(_row_get(prow, "config_json")) if prow is not None else None
            proposal, recipe, cfg_target = _proposal(cfg, prow)

            node = nodes.get(eid)
            screening_json = loads_or_none(_row_get(node, "screening_json")) if node else None
            node_metrics = loads_or_none(_row_get(node, "metrics_json")) if node else None
            screening_panel = (
                screening_json.get("challenger_metric_panel")
                if isinstance(screening_json, dict) else None
            )
            # hypothesis: research node, fallback to proposal rationale.
            hypothesis = _row_get(node, "hypothesis") if node else None
            if not hypothesis and prow is not None:
                hypothesis = _row_get(prow, "rationale")
            proposal.hypothesis = hypothesis
            if node is not None and not proposal.change_summary:
                proposal.change_summary = _row_get(node, "change_summary")
            if node is not None and not proposal.expected_benefit:
                proposal.expected_benefit = _row_get(node, "expected_benefit")
            if node is not None and not proposal.key_risk:
                proposal.key_risk = _row_get(node, "key_risk")

            metrics = _metrics(row, run_dir, screening_panel,
                               champion_panels.get(eid), warnings)

            screening = None
            if isinstance(screening_json, dict):
                screening = Screening(
                    gini_weighted=to_float(screening_json.get("challenger_score")),
                    lift=to_float(screening_json.get("lift")),
                    win_rate=to_float(screening_json.get("bootstrap_win_rate")),
                )

            crow = comparisons.get(eid)
            comparison = _comparison(crow, node_metrics) if crow is not None else None

            lrow = logs.get(eid)
            decision_meta = DecisionMeta(
                interpretation=_row_get(lrow, "interpretation") if lrow else None,
                next_step=_row_get(lrow, "next_step") if lrow else None,
                outcome=_row_get(lrow, "outcome") if lrow else None,
            )
            cycle = to_int(_row_get(lrow, "cycle")) if lrow else None

            tree = None
            if node is not None:
                tree = TreeNode(
                    node_id=_row_get(node, "node_id"),
                    parent_node_id=_row_get(node, "parent_node_id"),
                )
            research_line_id = _row_get(node, "line_id") if node else None

            experiments.append(
                Experiment(
                    experiment_id=eid,
                    delegation_id=delegation_id,
                    cycle=cycle,
                    seq=-1,  # assigned campaign-wide by the builder
                    created_at=_row_get(row, "created_at") or "",
                    name=name,
                    status=_row_get(row, "status") or "",
                    is_baseline=is_baseline_row(name, model_family),
                    is_seed=is_seed_name(name),
                    parent_experiment_id=_row_get(row, "parent_experiment_id"),
                    model_family=model_family,
                    target_strategy=_row_get(row, "target_strategy") or cfg_target or "",
                    recipe=recipe,
                    proposal=proposal,
                    metrics=metrics,
                    screening=screening,
                    comparison=comparison,
                    decision_meta=decision_meta,
                    lift=Lift(None, None, None),  # filled campaign-wide
                    repairs=_repairs(conn, eid, warnings),
                    usage=None,  # filled from telemetry
                    research_line_id=research_line_id,
                    tree=tree,
                )
            )
    return experiments


def read_champion_history(registry_path: Path, warnings: Warnings) -> list[dict]:
    """Raw champion_history rows (dicts) for the timeline stitcher."""
    out: list[dict] = []
    with ro_connect(registry_path, warnings) as conn:
        if conn is None or not has_table(conn, "champion_history"):
            return out
        cols = table_columns(conn, "champion_history")
        for r in safe_query(conn, "SELECT * FROM champion_history ORDER BY history_id",
                            warnings=warnings):
            out.append({c: _row_get(r, c) for c in cols})
    return out
