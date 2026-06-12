"""Query/analysis tool for the cross-run memory aggregator.

All queries respect the access gate (none / own / all).
With 'none', every query function raises AccessDeniedError.
With 'own', results are filtered to the run's own model_id.
With 'all', all models are returned, fully attributed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class AccessDeniedError(Exception):
    """Raised when memory access is none."""


def _check_access(access: str) -> None:
    if access == "none":
        raise AccessDeniedError(
            "Memory access is disabled (AUTORESEARCH_MEMORY_ACCESS=none). "
            "Set it to 'own' or 'all' to enable memory queries."
        )


def _require_own_identity(access: str, own_model_id: str | None) -> None:
    """Fail closed when 'own' access cannot resolve the caller's own model id.

    Without this, an 'own'-scoped query with an underivable own_model_id (missing
    or corrupt manifest identity) added no WHERE clause and silently returned
    *every* model's data — a lock that opens when confused
    (process_review_20260610_followup.md E3). Deny instead.
    """
    if access == "own" and not own_model_id:
        raise AccessDeniedError(
            "Memory access is 'own' but this run's own model identity could not be "
            "resolved (missing/corrupt run_manifest.json model_identity). Refusing "
            "to return all models' data. Run `autoresearch memory backfill-identity` "
            "or set AUTORESEARCH_MEMORY_ACCESS explicitly."
        )


def query_insights(
    memory_path: Path,
    access: str,
    *,
    model_id: str | None = None,
    family: str | None = None,
    target_strategy: str | None = None,
    verified_only: bool = True,
    run_uid: str | None = None,
    own_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve insights from the aggregator.

    With access='own', restricts to own_model_id.
    With access='all', returns all models (fully attributed).
    """
    _check_access(access)
    _require_own_identity(access, own_model_id)
    if not memory_path.exists():
        return []

    clauses: list[str] = []
    params: list[Any] = []

    if access == "own":
        # Under 'own', the filter is always the caller's own model id. An explicit
        # model_id argument can only narrow within own (a divergent one must not
        # become a peek into another model's insights).
        clauses.append("i.model_id = ?")
        params.append(own_model_id)
    elif model_id:
        clauses.append("i.model_id = ?")
        params.append(model_id)

    if verified_only:
        clauses.append("i.verified = 1")
    if run_uid:
        clauses.append("i.run_uid = ?")
        params.append(run_uid)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with sqlite3.connect(memory_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(
            f"SELECT * FROM insights i {where} ORDER BY i.created_at DESC", params
        )
        return [dict(row) for row in cur.fetchall()]


def query_experiments(
    memory_path: Path,
    access: str,
    *,
    own_model_id: str | None = None,
    filter_str: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve experiments from the aggregator."""
    _check_access(access)
    _require_own_identity(access, own_model_id)
    if not memory_path.exists():
        return []

    clauses: list[str] = []
    params: list[Any] = []

    if access == "own":
        clauses.append("r.model_id = ?")
        params.append(own_model_id)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with sqlite3.connect(memory_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(
            f"""
            SELECT e.*, r.model_id, m.provider
            FROM experiments e
            JOIN runs r ON r.run_uid = e.run_uid
            JOIN models m ON m.model_id = r.model_id
            {where}
            ORDER BY e.gini_weighted DESC
            LIMIT 50
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Canned analytical queries
# ---------------------------------------------------------------------------

# Experiments whose comparison decision was an explicit reject must not surface
# as peaks/jumps — they include known artifacts caught at the decision layer
# (process_review_20260610_followup.md E2). Mirrors harvester._rejected_challenger_ids.
_NOT_REJECTED = (
    "e.experiment_id NOT IN ("
    "SELECT challenger_id FROM comparisons "
    "WHERE challenger_id IS NOT NULL AND LOWER(decision) = 'reject')"
)

_ANALYSES = {
    "peak-gini-by-framing": """
        SELECT r.model_id, m.provider, e.target_strategy, e.model_family,
               MAX(e.gini_weighted) AS peak_gini,
               COUNT(*) AS n_experiments
        FROM experiments e
        JOIN runs r ON r.run_uid = e.run_uid
        JOIN models m ON m.model_id = r.model_id
        WHERE e.status = 'completed' AND e.gini_weighted IS NOT NULL
        AND {not_rejected}
        {model_filter}
        GROUP BY r.model_id, e.target_strategy, e.model_family
        ORDER BY peak_gini DESC
    """,
    "plateau-families": """
        SELECT r.model_id, m.provider, e.model_family, e.target_strategy,
               MAX(e.gini_weighted) AS peak_gini,
               COUNT(*) AS n_experiments
        FROM experiments e
        JOIN runs r ON r.run_uid = e.run_uid
        JOIN models m ON m.model_id = r.model_id
        WHERE e.status = 'completed' AND e.gini_weighted IS NOT NULL
        AND {not_rejected}
        {model_filter}
        GROUP BY r.model_id, e.model_family, e.target_strategy
        HAVING peak_gini < {threshold}
        ORDER BY peak_gini DESC
    """,
    "biggest-single-jumps": """
        SELECT e.run_uid, r.model_id, m.provider, e.cycle_index,
               e.gini_weighted,
               e.gini_weighted - LAG(e.gini_weighted, 1) OVER
                   (PARTITION BY e.run_uid ORDER BY e.cycle_index) AS gini_jump
        FROM experiments e
        JOIN runs r ON r.run_uid = e.run_uid
        JOIN models m ON m.model_id = r.model_id
        WHERE e.status = 'completed' AND e.gini_weighted IS NOT NULL
        AND {not_rejected}
        {model_filter}
        ORDER BY gini_jump DESC
        LIMIT 20
    """,
    "efficiency-by-model": """
        SELECT r.model_id, m.provider,
               COUNT(DISTINCT r.run_uid) AS n_runs,
               SUM(r.n_experiments) AS total_experiments,
               MAX(r.peak_gini) AS peak_gini,
               CAST(MAX(r.peak_gini) AS REAL) /
                   NULLIF(SUM(r.n_experiments), 0) AS efficiency_per_exp
        FROM runs r
        JOIN models m ON m.model_id = r.model_id
        {model_filter_runs}
        GROUP BY r.model_id
        ORDER BY efficiency_per_exp DESC
    """,
}


def run_analysis(
    memory_path: Path,
    access: str,
    analysis_name: str,
    *,
    own_model_id: str | None = None,
    threshold: float = 0.37,
) -> list[dict[str, Any]]:
    """Run a named canned analysis. Returns a list of result rows."""
    _check_access(access)
    _require_own_identity(access, own_model_id)
    if analysis_name not in _ANALYSES:
        raise ValueError(
            f"Unknown analysis {analysis_name!r}. "
            f"Available: {', '.join(sorted(_ANALYSES))}"
        )
    if not memory_path.exists():
        return []

    model_filter = ""
    model_filter_runs = ""
    params: list[Any] = []

    if access == "own" and own_model_id:
        model_filter = "AND r.model_id = ?"
        model_filter_runs = "WHERE r.model_id = ?"
        params.append(own_model_id)

    sql_template = _ANALYSES[analysis_name]
    sql = sql_template.format(
        model_filter=model_filter,
        model_filter_runs=model_filter_runs,
        threshold=float(threshold),
        not_rejected=_NOT_REJECTED,
    )

    with sqlite3.connect(memory_path) as con:
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Analysis query failed: {exc}") from exc
