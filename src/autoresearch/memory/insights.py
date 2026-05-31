"""Evidence-bound self-reflection insights for the cross-run memory aggregator.

Insights are recorded by the research agent and validated against the run's
own registry (opened read-only). Only verified insights are included in the
playbook and in default query results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOLERANCE_FLOOR = 1e-6
_TOLERANCE_RELATIVE = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stable_insight_id(run_uid: str, claim: str, evidence: dict[str, Any]) -> str:
    exp_ids = sorted(evidence.get("experiment_ids") or [])
    cmp_ids = sorted(evidence.get("comparison_ids") or [])
    key = json.dumps({"run_uid": run_uid, "claim": claim, "exp_ids": exp_ids, "cmp_ids": cmp_ids}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:24]


def _guard_path(path: Path | str) -> None:
    p = str(path)
    for fragment in ("holdout_vault", "milestone_reports"):
        if fragment in p:
            raise ValueError(
                f"Insight validator refused to open forbidden path (holdout guard): {p!r}"
            )


def validate_insight(
    run_registry_path: Path,
    insight_dict: dict[str, Any],
) -> tuple[bool, str]:
    """Validate an insight against the run's registry (opened read-only).

    For every experiment_id / comparison_id in evidence_json, confirm it exists
    in the registry. For the cited metric/delta, confirm the delta matches the
    actual difference within tolerance.

    Returns (verified, note). Failures do not raise; they return (False, note).
    """
    _guard_path(run_registry_path)

    evidence = insight_dict.get("evidence") or {}
    exp_ids = evidence.get("experiment_ids") or []
    cmp_ids = evidence.get("comparison_ids") or []
    claimed_delta = evidence.get("delta")
    claimed_metric = evidence.get("metric") or "gini_weighted"

    if not Path(run_registry_path).exists():
        return False, f"Registry not found: {run_registry_path}"

    ro_uri = f"file:{run_registry_path}?mode=ro"
    try:
        con = sqlite3.connect(ro_uri, uri=True)
    except sqlite3.OperationalError as exc:
        return False, f"Cannot open registry: {exc}"

    try:
        # Verify experiment_ids exist
        for eid in exp_ids:
            row = con.execute(
                "SELECT experiment_id FROM experiments WHERE experiment_id = ?", (eid,)
            ).fetchone()
            if row is None:
                return False, f"experiment_id not found in registry: {eid!r}"

        # Verify comparison_ids exist
        try:
            for cid in cmp_ids:
                row = con.execute(
                    "SELECT comparison_id FROM comparisons WHERE comparison_id = ?", (cid,)
                ).fetchone()
                if row is None:
                    return False, f"comparison_id not found in registry: {cid!r}"
        except sqlite3.OperationalError:
            if cmp_ids:
                return False, "comparisons table not found; cannot verify comparison_ids"

        # Verify delta if claimed and there are experiments to compare
        if claimed_delta is not None and len(exp_ids) >= 2:
            metrics_values: list[float] = []
            for eid in exp_ids[:2]:
                row = con.execute(
                    "SELECT metrics_path FROM experiments WHERE experiment_id = ?", (eid,)
                ).fetchone()
                if row and row[0]:
                    mp = Path(row[0])
                    if mp.exists():
                        try:
                            data = json.loads(mp.read_text(encoding="utf-8"))
                            splits = {
                                sm.get("split"): sm.get(claimed_metric)
                                for sm in data.get("split_metrics", [])
                            }
                            eval_splits = data.get("ordinary_eval_splits") or []
                            val = None
                            for s in eval_splits:
                                if s in splits and splits[s] is not None:
                                    val = splits[s]
                                    break
                            if val is None:
                                agg = data.get("aggregate", {})
                                val = agg.get(claimed_metric)
                            if val is not None:
                                metrics_values.append(float(val))
                        except (OSError, json.JSONDecodeError, ValueError):
                            pass

            if len(metrics_values) == 2:
                actual_delta = abs(metrics_values[1] - metrics_values[0])
                tol = max(_TOLERANCE_FLOOR, _TOLERANCE_RELATIVE * abs(actual_delta) if actual_delta else _TOLERANCE_FLOOR)
                if abs(abs(float(claimed_delta)) - actual_delta) > tol:
                    return False, (
                        f"Claimed delta {claimed_delta:.4f} differs from actual "
                        f"{actual_delta:.4f} (tolerance {tol:.6f}) for metric {claimed_metric!r}"
                    )
    finally:
        con.close()

    return True, "ok"


def record_insight(
    memory_path: Path,
    run_uid: str,
    model_identity: dict[str, str],
    insight_dict: dict[str, Any],
    *,
    run_registry_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and upsert an insight into the aggregator.

    The insight is stored regardless of verification outcome.
    verified=1 when evidence checks pass; verified=0 with a note otherwise.

    Returns the stored insight row as a dict.
    """
    from autoresearch.memory.store import init_memory_store

    provider = (model_identity.get("provider") or "").lower().strip()
    name = (model_identity.get("name") or "").lower().strip()
    model_id = f"{provider}/{name}"

    claim = str(insight_dict.get("claim") or "")
    scope = str(insight_dict.get("scope") or "general")
    confidence = insight_dict.get("confidence")
    evidence = insight_dict.get("evidence") or {}
    supersedes = insight_dict.get("supersedes")
    contradicts = insight_dict.get("contradicts")

    insight_id = _stable_insight_id(run_uid, claim, evidence)

    if run_registry_path is not None:
        verified, note = validate_insight(run_registry_path, insight_dict)
    else:
        verified, note = False, "no registry path provided; unverified"

    init_memory_store(memory_path)
    now = _now()
    with sqlite3.connect(memory_path) as con:
        con.execute(
            """
            INSERT INTO insights (
                insight_id, run_uid, model_id, created_at,
                claim, scope, confidence, evidence_json,
                verified, verification_note, supersedes, contradicts
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(insight_id) DO UPDATE SET
                verified=excluded.verified,
                verification_note=excluded.verification_note
            """,
            (
                insight_id, run_uid, model_id, now,
                claim, scope,
                float(confidence) if confidence is not None else None,
                json.dumps(evidence),
                int(verified), note,
                supersedes, contradicts,
            ),
        )

    return {
        "insight_id": insight_id,
        "run_uid": run_uid,
        "model_id": model_id,
        "verified": verified,
        "verification_note": note,
        "claim": claim,
    }


def list_insights(
    memory_path: Path,
    *,
    verified_only: bool = True,
    run_uid: str | None = None,
) -> list[dict[str, Any]]:
    """Return insight rows from the aggregator.

    Parameters
    ----------
    verified_only:
        When True (default), only return verified=1 rows.
    run_uid:
        Filter to a specific run if provided.
    """
    if not memory_path.exists():
        return []
    clauses = []
    params: list[Any] = []
    if verified_only:
        clauses.append("verified = 1")
    if run_uid:
        clauses.append("run_uid = ?")
        params.append(run_uid)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with sqlite3.connect(memory_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(
            f"SELECT * FROM insights {where} ORDER BY created_at DESC", params
        )
        return [dict(row) for row in cur.fetchall()]
