"""Read-only harvester from per-run registries into the aggregator.

Opens every per-run registry with mode=ro and never writes to them.
Only search-split metrics are harvested; holdout/milestone paths are
never opened.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch.memory.store import init_memory_store

logger = logging.getLogger(__name__)

_FORBIDDEN_PATH_FRAGMENTS = ("holdout_vault", "milestone_reports")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _guard_path(path: Path | str) -> None:
    """Raise ValueError if the path touches forbidden locations."""
    p = str(path)
    for fragment in _FORBIDDEN_PATH_FRAGMENTS:
        if fragment in p:
            raise ValueError(
                f"Harvester refused to open forbidden path (holdout guard): {p!r}"
            )


def _read_metrics(metrics_path: str | None) -> dict[str, Any]:
    """Load a metrics.json and return search-split summary values."""
    if not metrics_path:
        return {}
    try:
        _guard_path(metrics_path)
        data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}

    agg = data.get("aggregate", {})
    mean_score = agg.get("mean_score")
    std_score = agg.get("std_score")

    # Derive gini_weighted: restrict to ordinary_eval_splits (search-split metrics only).
    # If ordinary_eval_splits is absent, fall back to excluding known non-search splits
    # (holdout, milestone_holdout, train) by name.
    split_metrics = data.get("split_metrics", [])
    eval_splits = set(data.get("ordinary_eval_splits") or [])
    search_ginis = [
        sm["gini_weighted"]
        for sm in split_metrics
        if "gini_weighted" in sm
        and (
            sm.get("split") in eval_splits
            if eval_splits
            else sm.get("split") not in ("milestone_holdout", "holdout", "train")
        )
    ]
    # Return None when no eligible search-split gini exists.
    # Do NOT substitute mean_score (which is a Tweedie deviance, wrong scale/polarity).
    gini_weighted = sum(search_ginis) / len(search_ginis) if search_ginis else None

    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "gini_weighted": gini_weighted,
    }


def harvest_run(
    memory_path: Path,
    run_registry_path: Path,
    model_identity: dict[str, str],
    *,
    track_id: str = "",
    run_id: str = "",
) -> None:
    """Upsert one run's data from its registry into the aggregator.

    Parameters
    ----------
    memory_path:
        Path to artifacts/memory/memory.sqlite (aggregator).
    run_registry_path:
        Path to the per-run registry.sqlite opened read-only.
    model_identity:
        Dict with keys: provider, name, version (optional), harness (optional).
    track_id / run_id:
        Identifiers used to build run_uid and locate the manifest.
    """
    _guard_path(run_registry_path)

    provider = model_identity.get("provider", "").lower().strip()
    name = model_identity.get("name", "").lower().strip()
    version = model_identity.get("version") or ""
    harness = model_identity.get("harness") or ""
    model_id = f"{provider}/{name}"

    if not provider or not name:
        logger.warning("harvest_run skipped: model_identity missing provider/name")
        return

    run_uid = f"{track_id}/{run_id}" if track_id and run_id else str(run_registry_path.parent)

    init_memory_store(memory_path)

    ro_uri = f"file:{run_registry_path}?mode=ro"
    try:
        src = sqlite3.connect(ro_uri, uri=True)
    except sqlite3.OperationalError as exc:
        logger.warning("harvest_run: cannot open registry %s: %s", run_registry_path, exc)
        return

    try:
        experiments_raw = _fetch_experiments(src)
        comparisons_raw = _fetch_comparisons(src)
        started_at = _fetch_started_at(src, run_registry_path)
        final_champion_id = _fetch_final_champion(src)
    finally:
        src.close()

    now = _now()
    with sqlite3.connect(memory_path) as dst:
        # Upsert model
        dst.execute(
            """
            INSERT INTO models (model_id, provider, name, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET last_seen=excluded.last_seen
            """,
            (model_id, provider, name, now, now),
        )

        # Build experiment rows with cycle_index (order by created_at)
        exp_rows = _build_experiment_rows(experiments_raw, run_uid)
        n_experiments = len(exp_rows)
        # n_promotions counts comparison rows with decision='promote'.
        # Note: reverted promotions may inflate this count; a more precise proxy
        # would count distinct new_champion_id entries in champion_history, but
        # that requires a separate query and this proxy is acceptable for the leaderboard.
        n_promotions = sum(
            1 for c in comparisons_raw
            if (c.get("decision") or c.get("promotion_decision") or "").lower() == "promote"
        )
        # Only completed experiments can represent a true search-split peak.
        # Failed/timed-out experiments with a stray partial metric must not top the board.
        peak_gini = max(
            (
                r["gini_weighted"]
                for r in exp_rows
                if r["gini_weighted"] is not None and r.get("status") == "completed"
            ),
            default=None,
        )

        # Upsert run
        dst.execute(
            """
            INSERT INTO runs (
                run_uid, model_id, track_id, run_id, version, harness,
                started_at, last_harvested_at,
                n_experiments, n_promotions, peak_gini, final_champion_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_uid) DO UPDATE SET
                last_harvested_at=excluded.last_harvested_at,
                n_experiments=excluded.n_experiments,
                n_promotions=excluded.n_promotions,
                peak_gini=excluded.peak_gini,
                final_champion_id=excluded.final_champion_id
            """,
            (
                run_uid, model_id, track_id, run_id, version, harness,
                started_at, now,
                n_experiments, n_promotions, peak_gini, final_champion_id,
            ),
        )

        # Upsert experiments
        for row in exp_rows:
            dst.execute(
                """
                INSERT INTO experiments (
                    experiment_uid, run_uid, experiment_id, cycle_index,
                    model_family, target_strategy, target_mode,
                    features_json, hyperparameters_json,
                    mean_score, std_score, gini_weighted,
                    fit_wall_seconds, compute_budget_seconds, timed_out, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(experiment_uid) DO UPDATE SET
                    gini_weighted=excluded.gini_weighted,
                    mean_score=excluded.mean_score,
                    std_score=excluded.std_score,
                    status=excluded.status
                """,
                (
                    row["experiment_uid"], run_uid, row["experiment_id"],
                    row["cycle_index"],
                    row["model_family"], row["target_strategy"], row["target_mode"],
                    None, None,  # features_json / hyperparameters_json (future)
                    row["mean_score"], row["std_score"], row["gini_weighted"],
                    row["fit_wall_seconds"], row["compute_budget_seconds"],
                    row["timed_out"], row["status"],
                ),
            )

        # Upsert comparisons
        for cmp in comparisons_raw:
            cmp_uid = f"{run_uid}/{cmp['comparison_id']}"
            paired = {}
            try:
                paired = json.loads(cmp.get("paired_summary") or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            dst.execute(
                """
                INSERT INTO comparisons (
                    comparison_uid, run_uid,
                    champion_id, challenger_id,
                    mean_lift, challenger_win_rate, std_lift,
                    decision, guardrail_status, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(comparison_uid) DO UPDATE SET
                    decision=excluded.decision,
                    guardrail_status=excluded.guardrail_status
                """,
                (
                    cmp_uid, run_uid,
                    cmp.get("champion_id"), cmp.get("challenger_id"),
                    paired.get("mean_lift"), paired.get("challenger_win_rate"),
                    paired.get("std_lift") or paired.get("between_partition_std"),
                    cmp.get("decision") or cmp.get("promotion_decision"),
                    cmp.get("guardrail_status"),
                    cmp.get("created_at"),
                ),
            )


def _fetch_experiments(src: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = src.execute(
        """
        SELECT experiment_id, created_at, status, model_family,
               target_strategy, target_mode, metrics_path,
               fit_wall_seconds, compute_budget_seconds, timed_out
        FROM experiments
        ORDER BY created_at
        """
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_comparisons(src: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        cur = src.execute(
            """
            SELECT comparison_id, created_at, champion_id, challenger_id,
                   paired_summary, promotion_decision, decision, guardrail_status
            FROM comparisons
            """
        )
    except sqlite3.OperationalError:
        return []
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_started_at(src: sqlite3.Connection, registry_path: Path) -> str:
    """Best-effort: first session created_at, or the registry file mtime."""
    try:
        row = src.execute(
            "SELECT created_at FROM auto_sessions ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row and row[0]:
            return row[0]
    except sqlite3.OperationalError:
        pass
    try:
        mtime = registry_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return _now()


def _fetch_final_champion(src: sqlite3.Connection) -> str | None:
    try:
        row = src.execute(
            "SELECT new_champion_id FROM champion_history ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    try:
        row = src.execute(
            "SELECT champion_id FROM champion_state LIMIT 1"
        ).fetchone()
        if row:
            return row[0]
    except sqlite3.OperationalError:
        pass
    return None


def _build_experiment_rows(
    raw: list[dict[str, Any]], run_uid: str
) -> list[dict[str, Any]]:
    rows = []
    for idx, exp in enumerate(raw):
        metrics = _read_metrics(exp.get("metrics_path"))
        rows.append(
            {
                "experiment_uid": f"{run_uid}/{exp['experiment_id']}",
                "experiment_id": exp["experiment_id"],
                "cycle_index": idx,
                "model_family": exp.get("model_family"),
                "target_strategy": exp.get("target_strategy"),
                "target_mode": exp.get("target_mode"),
                "mean_score": metrics.get("mean_score"),
                "std_score": metrics.get("std_score"),
                "gini_weighted": metrics.get("gini_weighted"),
                "fit_wall_seconds": exp.get("fit_wall_seconds"),
                "compute_budget_seconds": exp.get("compute_budget_seconds"),
                "timed_out": exp.get("timed_out"),
                "status": exp.get("status"),
            }
        )
    return rows


def harvest_all(memory_path: Path, tracks_base: Path | None = None) -> dict[str, Any]:
    """Discover and harvest every registry under artifacts/tracks/.

    Skips runs whose run_manifest.json lacks model_identity, with a warning.
    Returns a summary dict.
    """
    from autoresearch.config import PROJECT_ROOT

    base = tracks_base or (PROJECT_ROOT / "artifacts" / "tracks")
    if not base.exists():
        logger.warning("harvest_all: tracks directory does not exist: %s", base)
        return {"harvested": 0, "skipped": 0, "errors": []}

    harvested = 0
    skipped = 0
    errors: list[str] = []

    for registry_path in sorted(base.rglob("registry.sqlite")):
        _guard_path(registry_path)
        run_dir = registry_path.parent
        manifest_path = run_dir / "run_manifest.json"

        if not manifest_path.exists():
            logger.warning("harvest_all: no run_manifest.json in %s — skipping", run_dir)
            skipped += 1
            continue

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("harvest_all: cannot read manifest %s: %s", manifest_path, exc)
            errors.append(f"{run_dir}: {exc}")
            skipped += 1
            continue

        model_identity = manifest.get("model_identity")
        if not model_identity or not model_identity.get("provider") or not model_identity.get("name"):
            logger.warning(
                "harvest_all: run_manifest.json in %s has no model_identity — "
                "run `autoresearch memory backfill-identity` first",
                run_dir,
            )
            skipped += 1
            continue

        track_id = manifest.get("track_id", "")
        run_id = manifest.get("run_id", "")

        try:
            harvest_run(
                memory_path,
                registry_path,
                model_identity,
                track_id=track_id,
                run_id=run_id,
            )
            harvested += 1
        except Exception as exc:
            logger.warning("harvest_all: error harvesting %s: %s", run_dir, exc)
            errors.append(f"{run_dir}: {exc}")

    return {"harvested": harvested, "skipped": skipped, "errors": errors}
