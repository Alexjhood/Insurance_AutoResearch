"""Read ``playoff/playoff_report.json`` into a :class:`Playoff` (DATA.md §2.5)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schema import (
    Consolidation,
    Playoff,
    PlayoffExclusion,
    PlayoffFinal,
    PlayoffFinalist,
    PlayoffPairing,
)
from ..util import Warnings, load_json, to_float


def _match_replay(delegation_id: str, consolidation_experiment_ids: list[str]) -> Optional[str]:
    """Match a finalist to its consolidation replay by delegation id in the name."""
    needle = f"_{delegation_id}_"
    for eid in consolidation_experiment_ids:
        if needle in eid:
            return eid
    return None


def read_playoff(
    orch_dir: Path,
    consolidation_experiment_ids: list[str],
    report_md_rel: Optional[str],
    warnings: Warnings,
) -> Optional[Playoff]:
    path = orch_dir / "playoff" / "playoff_report.json"
    if not path.exists():
        return None
    doc = load_json(path, warnings, label="playoff_report.json")
    if not isinstance(doc, dict):
        warnings.add("playoff_report.json unreadable or malformed")
        return None

    cons_doc = doc.get("consolidation") or {}
    consolidation = Consolidation(
        run_id=cons_doc.get("run_id", ""),
        track=cons_doc.get("track", ""),
    )

    finalists: list[PlayoffFinalist] = []
    for f in doc.get("finalists") or []:
        if not isinstance(f, dict):
            continue
        source = f.get("source") or {}
        did = f.get("delegation_id") or source.get("delegation_id") or ""
        finalists.append(
            PlayoffFinalist(
                delegation_id=did,
                experiment_id=source.get("experiment_id", ""),
                gini_weighted=to_float(f.get("gini_weighted")) or 0.0,
                model_family=f.get("model_family", ""),
                replay_experiment_id=_match_replay(did, consolidation_experiment_ids),
            )
        )

    exclusions: list[PlayoffExclusion] = []
    for e in doc.get("exclusions") or []:
        if isinstance(e, dict):
            exclusions.append(
                PlayoffExclusion(
                    delegation_id=e.get("delegation_id"),
                    reason=e.get("reason", ""),
                )
            )

    final = None
    lineage = doc.get("final_champion_lineage")
    if isinstance(lineage, dict):
        source = lineage.get("source") or {}
        final = PlayoffFinal(
            delegation_id=source.get("delegation_id", ""),
            source_experiment_id=source.get("experiment_id", ""),
            consolidation_experiment_id=lineage.get("consolidation_experiment_id", ""),
        )

    pairings: list[PlayoffPairing] = []
    for pairing in doc.get("pairings") or []:
        if not isinstance(pairing, dict):
            continue
        finalist = pairing.get("finalist") or {}
        source = finalist.get("source") or {}
        summary = pairing.get("comparison_summary") or {}
        evidence = pairing.get("gate_evidence") or {}
        decision = pairing.get("decision") or {}
        pairings.append(PlayoffPairing(
            order=int(pairing.get("order") or len(pairings) + 1),
            delegation_id=finalist.get("delegation_id") or source.get("delegation_id") or "",
            source_experiment_id=source.get("experiment_id") or "",
            incumbent_experiment_id=pairing.get("incumbent_experiment_id") or "",
            replayed_experiment_id=pairing.get("replayed_experiment_id"),
            comparison_id=pairing.get("comparison_id"),
            decision=decision.get("decision"),
            decision_reason=decision.get("rationale") or evidence.get("advisory_rationale"),
            mean_lift=to_float(summary.get("mean_lift")),
            challenger_win_rate=to_float(summary.get("challenger_win_rate")),
            gates=dict(evidence.get("standard_checks") or {}),
            guardrail_passed=(evidence.get("guardrail") or {}).get("passed"),
        ))

    status = doc.get("status") or ("completed" if final else "incomplete")
    failure_reason = doc.get("failure_reason") or doc.get("error")
    if isinstance(failure_reason, dict):
        failure_reason = failure_reason.get("message") or str(failure_reason)

    return Playoff(
        decision_mode=doc.get("decision_mode", ""),
        consolidation=consolidation,
        finalists=finalists,
        exclusions=exclusions,
        pairings=pairings,
        final=final,
        status=status,
        failure_reason=failure_reason,
        report_md=report_md_rel,
    )
