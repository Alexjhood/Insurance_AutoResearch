"""The final campaign report — what the orchestration actually cost and produced.

Two top-level sections, and the split is the point:

* ``framework_computed`` — cycles, decisions, promotions, distress, playoff
  result, champion lineage, wall-clock, and whatever usage/cost telemetry exists.
  Every number here is a read of a registry, a manifest, or a delegation report.
* ``agent_testimony`` — the sub-agents' ``finish-delegation`` summaries, verbatim.
  Testimony explains; it never scores.

A delegation whose report is missing is listed as such rather than guessed at, so
an aggregate is never quietly computed over a subset. Cost is ``null`` when no
delegation reported one — an unmeasured campaign is not a free campaign.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch.config import PROJECT_ROOT
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    campaign_report_json_path,
    campaign_report_markdown_path,
    load_orchestration,
    manifest_lock,
    playoff_dir,
    relative_to_project,
    report_path as delegation_report_path,
    save_orchestration,
    utc_stamp,
)
from autoresearch.orchestration.report import DISTRESS_FLAGS, INFORMATIONAL_FLAGS
from autoresearch.utils.io import read_json, write_json


PLAYOFF_JSON = "playoff_report.json"

_PROMOTE = "promote"
_LOCAL_PROMOTE = "local_promote"


# ── delegation report lookup ─────────────────────────────────────────────────


def _resolve_report(orch: Orchestration, delegation: Delegation) -> dict[str, Any] | None:
    """Load a delegation's collected report, or ``None`` when it was never built."""

    candidates: list[Path] = [
        delegation_report_path(orch.orchestration_id, delegation.delegation_id)
    ]
    if delegation.report_path:
        stored = Path(delegation.report_path)
        candidates.append(stored if stored.is_absolute() else PROJECT_ROOT / stored)
    for path in candidates:
        if path.exists():
            try:
                return read_json(path)
            except ValueError:
                return None
    return None


# ── aggregation ──────────────────────────────────────────────────────────────


def build_campaign_report(
    orch: Orchestration, *, generated_at: str | None = None
) -> dict[str, Any]:
    """Assemble the campaign report from manifest + delegation reports (no writes)."""

    generated_at = generated_at or utc_stamp()

    delegations: list[dict[str, Any]] = []
    testimony: dict[str, str | None] = {}
    missing_reports: list[str] = []

    cycles_attempted = 0
    cycles_completed = 0
    cycles_decided = 0
    experiments_total = 0
    by_decision: dict[str, int] = {}
    promoted_lifts: list[float] = []
    baseline_promotion_lifts: list[float] = []
    incremental_promotion_lifts: list[float] = []
    distress_by_flag: dict[str, int] = {flag: 0 for flag in DISTRESS_FLAGS}
    informational_by_flag: dict[str, int] = {flag: 0 for flag in INFORMATIONAL_FLAGS}
    delegations_in_distress = 0
    repair_requests = 0

    wall_clock_minutes = 0.0
    wall_clock_seen = False
    cost_usd = 0.0
    cost_reported_by: list[str] = []
    cost_missing_for: list[str] = []
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    usage_by_model: dict[str, dict[str, Any]] = {}
    any_estimated_cost = False

    for delegation in orch.delegations:
        report = _resolve_report(orch, delegation)
        if report is None:
            missing_reports.append(delegation.delegation_id)
            delegations.append(
                {
                    "delegation_id": delegation.delegation_id,
                    "backend": delegation.backend,
                    "track": delegation.track,
                    "run_id": delegation.run_id,
                    "status": delegation.status,
                    "cycle_budget": delegation.cycle_budget,
                    "report": "missing",
                }
            )
            testimony[delegation.delegation_id] = delegation.agent_summary
            continue

        cycle_facts = report.get("cycles") or {}
        completed = int(cycle_facts.get("completed", cycle_facts.get("used", 0)) or 0)
        attempted = int(cycle_facts.get("attempted", completed) or 0)
        decided = int(cycle_facts.get("decided", completed) or 0)
        cycles_attempted += attempted
        cycles_completed += completed
        cycles_decided += decided
        rows = report.get("experiments") or []
        experiments_total += len(rows)
        for row in rows:
            decision = row.get("decision")
            if not decision:
                continue
            by_decision[decision] = by_decision.get(decision, 0) + 1
            if decision == _PROMOTE and row.get("lift_vs_champion") is not None:
                lift = float(row["lift_vs_champion"])
                promoted_lifts.append(lift)
                # vs_baseline is absent from reports generated before the split
                # existed; leave those out of both buckets rather than mislabel.
                if row.get("vs_baseline") is True:
                    baseline_promotion_lifts.append(lift)
                elif row.get("vs_baseline") is False:
                    incremental_promotion_lifts.append(lift)

        active = list((report.get("distress") or {}).get("active") or ())
        if active:
            delegations_in_distress += 1
        for flag in active:
            distress_by_flag[flag] = distress_by_flag.get(flag, 0) + 1
        informational = list((report.get("distress") or {}).get("informational") or ())
        for flag in informational:
            informational_by_flag[flag] = informational_by_flag.get(flag, 0) + 1

        repair_requests += int((report.get("repairs") or {}).get("requests") or 0)

        cost = report.get("cost") or {}
        minutes = cost.get("wall_clock_minutes")
        if minutes is not None:
            wall_clock_minutes += float(minutes)
            wall_clock_seen = True
        usage = cost.get("llm_usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        backend_usage = usage.get("backend") or {}
        details = backend_usage.get("details") if isinstance(backend_usage.get("details"), dict) else {}
        cached = int(backend_usage.get("cached_input_tokens") or backend_usage.get("cache_read_input_tokens")
                     or details.get("cached_input_tokens") or details.get("cache_read_input_tokens") or 0)
        reasoning = int(backend_usage.get("reasoning_tokens") or details.get("reasoning_tokens") or 0)
        cached_tokens += cached
        reasoning_tokens += reasoning
        model_key, model_usage = _delegation_model_usage(delegation, backend_usage)
        aggregate = usage_by_model.get(model_key)
        if aggregate is None:
            aggregate = {**model_usage, "input_tokens": 0, "cached_tokens": 0,
                         "output_tokens": 0, "reasoning_tokens": 0}
            usage_by_model[model_key] = aggregate
        elif delegation.backend not in aggregate["backends"]:
            aggregate["backends"].append(delegation.backend)
        for token_key in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            aggregate[token_key] += int(model_usage[token_key])
        if usage.get("cost_usd") is not None:
            delegation_cost = float(usage["cost_usd"])
            cost_usd += delegation_cost
            cost_reported_by.append(delegation.delegation_id)
            aggregate["cost_usd"] = round(float(aggregate.get("cost_usd") or 0) + delegation_cost, 6)
            estimated = bool(usage.get("cost_estimated"))
            aggregate["cost_estimated"] = bool(aggregate.get("cost_estimated")) or estimated
            any_estimated_cost = any_estimated_cost or estimated
        else:
            cost_missing_for.append(delegation.delegation_id)

        champion = report.get("champion") or {}
        delegations.append(
            {
                "delegation_id": delegation.delegation_id,
                "backend": delegation.backend,
                "track": delegation.track,
                "run_id": delegation.run_id,
                "status": delegation.status,
                "cycle_budget": delegation.cycle_budget,
                "cycles_attempted": attempted,
                "cycles_completed": completed,
                "cycles_decided": decided,
                "cycles_used": completed,
                "respawn_of": delegation.respawn_of,
                "champion": {
                    "experiment_id": champion.get("experiment_id"),
                    "model_family": champion.get("model_family"),
                    "gini_weighted": champion.get("gini_weighted"),
                    "beat_seed_baseline": champion.get("beat_seed_baseline"),
                },
                "distress": active,
                "informational": informational,
                "wall_clock_minutes": minutes,
                "report": relative_to_project(
                    delegation_report_path(orch.orchestration_id, delegation.delegation_id)
                ),
            }
        )
        testimony[delegation.delegation_id] = report.get("agent_summary")

    playoff = _playoff_summary(orch)
    promotions = by_decision.get(_PROMOTE, 0)

    framework = {
        "cycles": {
            "total_budget": orch.total_cycle_budget,
            "committed": orch.cycles_committed,
            "attempted": cycles_attempted,
            "completed": cycles_completed,
            "decided": cycles_decided,
            "used": cycles_completed,
            "remaining": orch.cycles_remaining,
            "forfeited": sum(d.cycles_forfeited for d in orch.delegations),
        },
        "delegations": delegations,
        "missing_reports": missing_reports,
        "experiments": {
            "total": experiments_total,
            "by_decision": dict(sorted(by_decision.items())),
        },
        "promotions": {
            "promote": promotions,
            "local_promote": by_decision.get(_LOCAL_PROMOTE, 0),
            # Retained for older consumers; do not read it for analysis — it
            # averages baseline-relative and incremental lifts, which differ by
            # an order of magnitude. Use the two split stats below.
            "mean_promoted_gini_lift": (
                round(sum(promoted_lifts) / len(promoted_lifts), 6)
                if promoted_lifts
                else None
            ),
            "mean_baseline_promotion_gini_lift": (
                round(sum(baseline_promotion_lifts) / len(baseline_promotion_lifts), 6)
                if baseline_promotion_lifts
                else None
            ),
            "mean_incremental_promotion_gini_lift": (
                round(sum(incremental_promotion_lifts) / len(incremental_promotion_lifts), 6)
                if incremental_promotion_lifts
                else None
            ),
            "baseline_promotions": len(baseline_promotion_lifts),
            "incremental_promotions": len(incremental_promotion_lifts),
        },
        "distress": {
            "by_flag": {flag: distress_by_flag[flag] for flag in DISTRESS_FLAGS},
            "delegations_in_distress": delegations_in_distress,
        },
        "informational": {
            "by_flag": {flag: informational_by_flag[flag] for flag in INFORMATIONAL_FLAGS},
        },
        "repairs": {
            "requests": repair_requests,
            "per_cycle": (
                round(repair_requests / cycles_completed, 4) if cycles_completed else None
            ),
        },
        "playoff": playoff,
        "final_champion": _final_champion(orch, playoff),
        "wall_clock": {
            "campaign_elapsed_minutes": _elapsed_minutes(orch.created_at, generated_at),
            "delegation_minutes_total": (
                round(wall_clock_minutes, 2) if wall_clock_seen else None
            ),
        },
        "cost": {
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cost_usd": (round(cost_usd, 6) if cost_reported_by else None),
            "cost_estimated": any_estimated_cost,
            "cost_usd_reported_by": cost_reported_by,
            "cost_usd_missing_for": cost_missing_for,
            "cost_per_cycle_usd": (
                round(cost_usd / cycles_completed, 6)
                if cost_reported_by and cycles_completed
                else None
            ),
            "cost_per_promotion_usd": (
                round(cost_usd / promotions, 6) if cost_reported_by and promotions else None
            ),
            "usage_by_model": _usage_with_orchestrator(orch, usage_by_model),
        },
    }

    return {
        "orchestration_id": orch.orchestration_id,
        "dataset": orch.dataset,
        "target_mode": orch.target_mode,
        "created_at": orch.created_at,
        "generated_at": generated_at,
        "status": orch.status,
        "orchestrator": {
            "model_provider": orch.model_provider,
            "model_name": orch.model_name,
            "provider": orch.model_provider,
            "model": orch.model_name,
            "effort": orch.model_effort,
        },
        "framework_computed": framework,
        "agent_testimony": testimony,
    }


def _delegation_model_usage(delegation: Delegation, usage: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from autoresearch.orchestration.backends import get_backend

    try:
        backend = get_backend(delegation.backend)
        provider, model = backend.model_provider, backend.model_name
    except (KeyError, ValueError, FileNotFoundError):
        provider, model = "unknown", delegation.backend
    details = usage.get("details") if isinstance(usage.get("details"), dict) else {}
    cached = int(usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens")
                 or details.get("cached_input_tokens") or details.get("cache_read_input_tokens") or 0)
    reasoning = int(usage.get("reasoning_tokens") or details.get("reasoning_tokens") or 0)
    return f"{provider}/{model}", {
        "provider": provider, "model": model, "backends": [delegation.backend],
        "input_tokens": int(usage.get("input_tokens") or 0), "cached_tokens": cached,
        "output_tokens": int(usage.get("output_tokens") or 0), "reasoning_tokens": reasoning,
        "cost_usd": None, "cost_estimated": False, "unmeasured": False,
    }


def _usage_with_orchestrator(orch: Orchestration, usage_by_model: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = dict(sorted(usage_by_model.items()))
    provider, model = orch.model_provider or "unknown", orch.model_name or "unknown"
    result[f"orchestrator:{provider}/{model}"] = {
        "role": "orchestrator", "provider": orch.model_provider, "model": orch.model_name,
        "effort": orch.model_effort, "input_tokens": None, "cached_tokens": None,
        "output_tokens": None, "reasoning_tokens": None, "cost_usd": None,
        "cost_estimated": False, "unmeasured": True,
    }
    return result


def _playoff_summary(orch: Orchestration) -> dict[str, Any] | None:
    path = playoff_dir(orch.orchestration_id) / PLAYOFF_JSON
    if not path.exists():
        return None
    state = read_json(path)
    return {
        "status": state.get("status"),
        "decision_mode": state.get("decision_mode"),
        "consolidation": state.get("consolidation"),
        "exclusions": state.get("exclusions") or [],
        "finalists": [
            {
                "delegation_id": item.get("delegation_id"),
                "gini_weighted": item.get("gini_weighted"),
            }
            for item in state.get("finalists") or ()
        ],
        "pairings": [
            {
                "order": pair.get("order"),
                "delegation_id": (pair.get("finalist") or {}).get("delegation_id"),
                "comparison_id": pair.get("comparison_id"),
                "decision": pair.get("decision"),
            }
            for pair in state.get("pairings") or ()
        ],
        "final_champion_lineage": state.get("final_champion_lineage"),
    }


def _final_champion(
    orch: Orchestration, playoff: dict[str, Any] | None
) -> dict[str, Any] | None:
    """The orchestration champion and where it came from, once the playoff ran."""

    if not playoff:
        return None
    lineage = playoff.get("final_champion_lineage")
    if not lineage:
        return None
    return {
        "consolidation_track": orch.consolidation.track,
        "consolidation_run_id": orch.consolidation.run_id,
        "consolidation_experiment_id": lineage.get("consolidation_experiment_id"),
        "source": lineage.get("source"),
        "holdout": _holdout_status(
            orch.consolidation.track,
            orch.consolidation.run_id,
            lineage.get("consolidation_experiment_id"),
        ),
    }


def _holdout_status(
    track: str | None, run_id: str | None, champion_id: str | None
) -> dict[str, Any]:
    """The protected-holdout evaluation status for the final champion.

    A campaign is not validated until this ran; a skipped evaluation (no
    milestone token in an agent context) must be visible in the report rather
    than buried in the consolidation run's milestone_reports directory.
    """

    pending = {
        "status": "pending",
        "detail": "No milestone report found for the final champion — the protected "
        "holdout evaluation has not run. Operator action required.",
    }
    if not (track and run_id and champion_id):
        return pending
    reports_dir = PROJECT_ROOT / "artifacts" / "tracks" / track / "runs" / run_id / "milestone_reports"
    if not reports_dir.is_dir():
        return pending
    latest: dict[str, Any] | None = None
    latest_mtime = -1.0
    for path in reports_dir.glob("*.json"):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if report.get("champion_id") != champion_id:
            continue
        mtime = path.stat().st_mtime
        if mtime > latest_mtime:
            latest, latest_mtime = report, mtime
    if latest is None:
        return pending
    status = str(latest.get("status") or "unknown")
    detail = latest.get("reason") or latest.get("summary")
    result: dict[str, Any] = {"status": status}
    if status == "skipped":
        result["detail"] = (
            "Holdout evaluation was SKIPPED (no milestone token in the agent "
            "context). The champion is unvalidated until an operator runs it."
        )
    elif detail:
        result["detail"] = str(detail)
    return result


def _elapsed_minutes(start: str, end: str) -> float | None:
    parsed = [_parse(start), _parse(end)]
    if any(value is None for value in parsed):
        return None
    return round((parsed[1] - parsed[0]).total_seconds() / 60.0, 2)  # type: ignore[operator]


def _parse(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ── status transition ────────────────────────────────────────────────────────


def finalized_status(orch: Orchestration) -> str:
    """The status the campaign should carry once its final report is written.

    A campaign completes when every delegation has stopped and no playoff is in
    flight. ``consolidating`` is left alone: the playoff owns that transition and
    a half-finished gauntlet is not a finished campaign. ``abandoned`` and
    ``completed`` are terminal.
    """

    if orch.status in {"completed", "abandoned", "consolidating"}:
        return orch.status
    if any(not d.is_terminal for d in orch.delegations):
        return orch.status
    return "completed"


# ── persistence ──────────────────────────────────────────────────────────────


def write_campaign_report(
    orchestration_id: str, *, generated_at: str | None = None
) -> dict[str, Any]:
    """Build, persist, and register the final campaign report.

    Returns the payload plus the two written paths. The manifest's
    ``campaign_report`` pointer and its status are updated in the same locked
    read-modify-write, so a report can never exist without the manifest naming it.
    """

    orch = load_orchestration(orchestration_id)
    payload = build_campaign_report(orch, generated_at=generated_at)

    json_path = campaign_report_json_path(orchestration_id)
    markdown_path = campaign_report_markdown_path(orchestration_id)

    with manifest_lock(orchestration_id):
        current = load_orchestration(orchestration_id)
        status = finalized_status(current)
        payload["status"] = status
        write_json(json_path, payload)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_campaign_markdown(payload), encoding="utf-8")
        save_orchestration(
            replace(
                current,
                status=status,
                campaign_report=relative_to_project(json_path),
            )
        )

    return {
        "payload": payload,
        "json_path": json_path,
        "markdown_path": markdown_path,
    }


# ── markdown ─────────────────────────────────────────────────────────────────


def render_campaign_markdown(payload: dict[str, Any]) -> str:
    framework = payload["framework_computed"]
    cycles = framework["cycles"]
    lines = [
        f"# Campaign Report — {payload['orchestration_id']}",
        "",
        "<!-- Generated by `autoresearch orchestrate report`. Do not edit by hand. -->",
        "",
        f"- Dataset: `{payload['dataset']}` / target mode `{payload['target_mode']}`",
        f"- Status: {payload['status']}",
        f"- Created: {payload['created_at']}  ·  Report generated: {payload['generated_at']}",
    ]
    orchestrator = payload.get("orchestrator") or {}
    if orchestrator.get("model_name"):
        effort = orchestrator.get("effort")
        lines.append(
            f"- Orchestrator model: {orchestrator.get('model_provider') or 'unknown'}"
            f"/{orchestrator['model_name']}"
            + (f" · {effort}" if effort else "")
        )
    lines.extend(
        [
            "",
            "Everything under **Framework-computed** is read from run registries, the "
            "orchestration manifest, and delegation reports. **Agent testimony** is the "
            "sub-agents' own words: it explains results, it does not establish them.",
            "",
            "## Framework-computed",
            "",
            "### Cycle budget",
            "",
            f"- Budget {cycles['total_budget']} · reserved {cycles['committed']} · "
            f"attempted {cycles.get('attempted', cycles['used'])} · "
            f"completed {cycles.get('completed', cycles['used'])} · "
            f"decided {cycles.get('decided', cycles['used'])} · "
            f"unreserved {cycles['remaining']}"
            + (
                f" · forfeited-and-reclaimed {cycles['forfeited']}"
                if cycles.get("forfeited")
                else ""
            ),
            "",
            "### Delegations",
            "",
        ]
    )

    if not framework["delegations"]:
        lines.extend(["No delegations were spawned.", ""])
    else:
        lines.append("| id | backend | run | cycles (A/C/D/B) | champion | gini | signals |")
        lines.append("|---|---|---|---|---|---|---|")
        for item in framework["delegations"]:
            if item.get("report") == "missing":
                lines.append(
                    f"| {item['delegation_id']} | {item['backend']} | `{item['run_id']}` "
                    f"| — | (report missing) | — | — |"
                )
                continue
            champion = item["champion"]
            signals = list(item["distress"])
            signals.extend(f"info:{flag}" for flag in item.get("informational") or ())
            lines.append(
                f"| {item['delegation_id']} | {item['backend']} | `{item['run_id']}` "
                f"| {item.get('cycles_attempted', item['cycles_used'])}/"
                f"{item.get('cycles_completed', item['cycles_used'])}/"
                f"{item.get('cycles_decided', item['cycles_used'])}/"
                f"{item['cycle_budget']} "
                f"| {champion.get('model_family') or '—'} "
                f"| {_fmt(champion.get('gini_weighted'))} "
                f"| {', '.join(signals) or '—'} |"
            )
        lines.append("")

    if framework["missing_reports"]:
        lines.extend(
            [
                f"Delegations without a collected report (excluded from every aggregate "
                f"below): {', '.join(framework['missing_reports'])}. Run "
                "`autoresearch orchestrate collect` and regenerate.",
                "",
            ]
        )

    experiments = framework["experiments"]
    promotions = framework["promotions"]
    lines.extend(
        [
            "### Experiments and promotions",
            "",
            f"- Experiments recorded: {experiments['total']}",
            f"- Decisions: {_fmt_counts(experiments['by_decision'])}",
            f"- Promotions: {promotions['promote']} "
            f"(local: {promotions['local_promote']})",
            f"- Baseline-relative promotion lift (vs the flat start, "
            f"n={promotions['baseline_promotions']}): "
            f"{_fmt(promotions['mean_baseline_promotion_gini_lift'])}",
            f"- Incremental promotion lift (champion vs champion, "
            f"n={promotions['incremental_promotions']}): "
            f"{_fmt(promotions['mean_incremental_promotion_gini_lift'])}",
            "",
            "### Distress",
            "",
        ]
    )
    active_flags = {k: v for k, v in framework["distress"]["by_flag"].items() if v}
    if active_flags:
        for flag, count in active_flags.items():
            lines.append(f"- `{flag}`: {count} delegation(s)")
    else:
        lines.append("- No distress flags fired.")
    lines.append("")

    lines.extend(["### Playoff", ""])
    playoff = framework["playoff"]
    if not playoff:
        lines.extend(["No playoff was run; the campaign has no consolidated champion.", ""])
    else:
        lines.append(f"- Status: {playoff['status']} ({playoff['decision_mode']} decisions)")
        finalists = ", ".join(
            f"{item['delegation_id']} ({_fmt(item['gini_weighted'])})"
            for item in playoff["finalists"]
        )
        lines.append(f"- Finalists in ascending order: {finalists or '—'}")
        if playoff["exclusions"]:
            excluded = ", ".join(
                f"{item.get('delegation_id')} ({item.get('reason')})"
                for item in playoff["exclusions"]
            )
            lines.append(f"- Excluded: {excluded}")
        for pair in playoff["pairings"]:
            lines.append(
                f"- Pairing {pair['order']} — {pair['delegation_id']}: "
                f"{pair['decision'] or 'pending'}"
            )
        lines.append("")

    lines.extend(["### Final champion", ""])
    champion = framework["final_champion"]
    if not champion:
        lines.extend(["No orchestration champion: the playoff has not completed.", ""])
    else:
        source = champion.get("source") or {}
        lines.append(
            f"- Consolidation experiment `{champion['consolidation_experiment_id']}` "
            f"in `{champion['consolidation_track']}/{champion['consolidation_run_id']}`"
        )
        lines.append(
            f"- Replayed from delegation {source.get('delegation_id') or '—'}, run "
            f"`{source.get('run_id') or '—'}`, experiment "
            f"`{source.get('experiment_id') or '—'}`"
        )
        holdout = champion.get("holdout") or {}
        holdout_line = f"- Protected holdout: **{holdout.get('status', 'unknown')}**"
        if holdout.get("detail"):
            holdout_line += f" — {holdout['detail']}"
        lines.append(holdout_line)
        lines.append("")

    wall_clock = framework["wall_clock"]
    cost = framework["cost"]
    lines.extend(
        [
            "### Wall clock and cost",
            "",
            f"- Campaign elapsed: {_fmt(wall_clock['campaign_elapsed_minutes'])} min",
            f"- Delegation wall clock (sum): {_fmt(wall_clock['delegation_minutes_total'])} min",
            f"- Tokens: {cost['input_tokens']} in / {cost.get('cached_tokens', 0)} cached / "
            f"{cost['output_tokens']} out / {cost.get('reasoning_tokens', 0)} reasoning",
            f"- Cost: {_fmt_cost(cost['cost_usd'])}"
            + (" (estimated)" if cost.get("cost_estimated") else "")
            + (
                f" · per cycle {_fmt_cost(cost['cost_per_cycle_usd'])}"
                f" · per promotion {_fmt_cost(cost['cost_per_promotion_usd'])}"
                if cost["cost_usd"] is not None
                else ""
            ),
        ]
    )
    if cost["cost_usd_missing_for"]:
        missing = ", ".join(cost["cost_usd_missing_for"])
        if cost["cost_usd_reported_by"]:
            lines.append(
                f"- No provider cost was reported for: {missing}. The total above "
                f"covers only {', '.join(cost['cost_usd_reported_by'])}."
            )
        else:
            lines.append(
                f"- No delegation reported a provider cost ({missing}). This campaign "
                "is unmeasured, not free."
            )
    lines.append("")

    usage_by_model = cost.get("usage_by_model") or {}
    if usage_by_model:
        lines.extend(["### Usage by model", "", "| model | input | cached | output | reasoning | cost |",
                      "|---|---:|---:|---:|---:|---:|"])
        for key, usage in usage_by_model.items():
            if usage.get("unmeasured"):
                lines.append(f"| {key} | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |")
                continue
            model_cost = _fmt_cost(usage.get("cost_usd"))
            if usage.get("cost_estimated") and usage.get("cost_usd") is not None:
                model_cost += " (estimated)"
            lines.append(f"| {key} | {usage.get('input_tokens', 0)} | {usage.get('cached_tokens', 0)} "
                         f"| {usage.get('output_tokens', 0)} | {usage.get('reasoning_tokens', 0)} | {model_cost} |")
        lines.append("")

    lines.extend(["## Agent testimony", ""])
    testimony = payload["agent_testimony"]
    if not testimony:
        lines.append("No delegations.")
    for delegation_id in sorted(testimony):
        summary = testimony[delegation_id]
        lines.append(f"### {delegation_id}")
        lines.append("")
        if summary and summary.strip():
            lines.extend(summary.strip().splitlines())
        else:
            lines.append(
                "_No summary: this sub-agent never called `finish-delegation`._"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _fmt_cost(value: Any) -> str:
    if value is None:
        return "not reported"
    return f"${float(value):.4f}"


def _fmt_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{name} {count}" for name, count in counts.items())
