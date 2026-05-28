"""Experiment and artifact registry operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.experiment_registry._common import dumps
from autoresearch.experiment_registry.schema import init_registry
from autoresearch.utils.io import read_json


def record_experiment(
    path: Path,
    *,
    experiment_id: str,
    experiment_name: str,
    model_family: str,
    target_strategy: str,
    target_mode: str = "burning_cost",
    preprocessing_summary: dict[str, Any],
    claim_cap_threshold: float | None,
    status: str,
    parent_experiment_id: str | None,
    config_snapshot_path: Path,
    metrics_path: Path,
    artifacts: dict[str, Path],
    code_version: str | None = None,
    notes: str | None = None,
) -> None:
    """Insert or replace a completed experiment and its artifact records."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id,
                updated_at,
                experiment_name,
                status,
                parent_experiment_id,
                model_family,
                target_strategy,
                target_mode,
                preprocessing_summary,
                claim_cap_threshold,
                config_snapshot_path,
                code_version,
                metrics_path,
                notes
            )
            VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                experiment_name,
                status,
                parent_experiment_id,
                model_family,
                target_strategy,
                target_mode,
                dumps(preprocessing_summary),
                claim_cap_threshold,
                str(config_snapshot_path),
                code_version,
                str(metrics_path),
                notes,
            ),
        )
        con.execute("DELETE FROM artifacts WHERE experiment_id = ?", (experiment_id,))
        con.executemany(
            """
            INSERT INTO artifacts (experiment_id, artifact_type, path)
            VALUES (?, ?, ?)
            """,
            [(experiment_id, artifact_type, str(artifact_path)) for artifact_type, artifact_path in artifacts.items()],
        )


def record_experiment_artifacts(path: Path, experiment_id: str, artifacts: dict[str, Path]) -> None:
    """Add or replace selected artifact records for an existing experiment."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        for artifact_type, artifact_path in artifacts.items():
            con.execute(
                "DELETE FROM artifacts WHERE experiment_id = ? AND artifact_type = ?",
                (experiment_id, artifact_type),
            )
            con.execute(
                """
                INSERT INTO artifacts (experiment_id, artifact_type, path)
                VALUES (?, ?, ?)
                """,
                (experiment_id, artifact_type, str(artifact_path)),
            )


def get_experiment(path: Path, experiment_id: str) -> dict[str, Any]:
    """Fetch one experiment row by id."""

    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT *
            FROM experiments
            WHERE experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"Unknown experiment id: {experiment_id}")
    return dict(row)


def list_experiments(path: Path) -> list[dict[str, Any]]:
    """Return registry rows with selected metrics hydrated from disk."""

    if not path.exists():
        return []
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                experiment_id,
                created_at,
                updated_at,
                experiment_name,
                status,
                parent_experiment_id,
                model_family,
                target_strategy,
                target_mode,
                preprocessing_summary,
                claim_cap_threshold,
                config_snapshot_path,
                metrics_path,
                notes
            FROM experiments
            ORDER BY created_at DESC, experiment_id DESC
            """
        ).fetchall()

    experiments: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metrics_path = Path(item["metrics_path"]) if item.get("metrics_path") else None
        if metrics_path and metrics_path.exists():
            metrics = read_json(metrics_path)
            item["mean_score"] = metrics.get("aggregate", {}).get("mean_score")
            item["std_score"] = metrics.get("aggregate", {}).get("std_score")
            item["primary_metric"] = metrics.get("primary_metric")
            item["target_mode"] = item.get("target_mode") or metrics.get("target_mode")
            item["ordinary_eval_splits"] = metrics.get("ordinary_eval_splits")
        experiments.append(item)
    return experiments


def list_artifacts(path: Path, experiment_id: str) -> list[dict[str, Any]]:
    """Return artifacts for one experiment."""

    if not path.exists():
        return []
    init_registry(path)
    with sqlite3.connect(path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT artifact_type, path, created_at
            FROM artifacts
            WHERE experiment_id = ?
            ORDER BY artifact_type
            """,
            (experiment_id,),
        ).fetchall()
    return [dict(row) for row in rows]
