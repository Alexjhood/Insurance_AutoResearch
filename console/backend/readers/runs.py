"""Read-only access to per-run registries and artifact files."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from autoresearch.config import load_config
from autoresearch.experiment_registry import (
    list_experiments,
    list_comparisons,
    get_official_champion,
    list_champion_history,
    list_proposals,
    list_sessions,
    list_session_events,
    list_research_lines,
    registry_counts,
)

# Avoid circular import — resolve ARTIFACTS_DIR directly here
_REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = _REPO_ROOT / "artifacts" / "tracks"


def list_tracks() -> list[dict]:
    if not ARTIFACTS_DIR.exists():
        return []
    tracks = []
    for p in sorted(ARTIFACTS_DIR.iterdir()):
        if p.name.startswith(".") or not p.is_dir():
            continue
        runs_dir = p / "runs"
        n_runs = len(list(runs_dir.iterdir())) if runs_dir.exists() else 0
        tracks.append({"track_id": p.name, "n_runs": n_runs})
    return tracks


def list_runs(track_id: str) -> list[dict]:
    track_dir = ARTIFACTS_DIR / track_id / "runs"
    if not track_dir.exists():
        return []
    runs = []
    for p in sorted(track_dir.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        reg = p / "registry.sqlite"
        manifest = p / "run_manifest.json"
        info: dict[str, Any] = {
            "run_id": p.name,
            "track_id": track_id,
            "has_registry": reg.exists(),
            "manifest": None,
        }
        if manifest.exists():
            try:
                info["manifest"] = json.loads(manifest.read_text())
            except Exception:
                pass
        if reg.exists():
            try:
                cfg = load_config(track_id=track_id, run_id=p.name)
                counts = registry_counts(cfg.registry_path)
                info["counts"] = counts
                champ_row = get_official_champion(cfg.registry_path)
                champ = _row_to_dict(champ_row)
                if champ:
                    champ = _enrich_champion(champ, cfg.registry_path)
                info["champion"] = champ
            except Exception as exc:
                info["error"] = str(exc)
        runs.append(info)
    return runs


def run_summary(track_id: str, run_id: str) -> dict:
    cfg = load_config(track_id=track_id, run_id=run_id)
    out: dict[str, Any] = {
        "track_id": track_id,
        "run_id": run_id,
        "registry_exists": cfg.registry_path.exists(),
    }
    if not cfg.registry_path.exists():
        return out
    try:
        out["counts"] = registry_counts(cfg.registry_path)
        champ = _row_to_dict(get_official_champion(cfg.registry_path))
        if champ:
            champ = _enrich_champion(champ, cfg.registry_path)
        out["champion"] = champ
    except Exception as exc:
        out["error"] = str(exc)

    run_dir = ARTIFACTS_DIR / track_id / "runs" / run_id
    log_path = run_dir / "RESEARCH_LOG.md"
    if log_path.exists():
        out["research_log_preview"] = log_path.read_text(encoding="utf-8")[:2000]

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            out["manifest"] = json.loads(manifest_path.read_text())
        except Exception:
            pass
    return out


def get_experiments(track_id: str, run_id: str) -> list[dict]:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return []
    rows = [_row_to_dict(r) for r in list_experiments(cfg.registry_path)]
    # Normalise: add gini_weighted alias for mean_score when primary_metric matches
    for row in rows:
        if row and row.get("primary_metric") == "gini_weighted" and "mean_score" in row:
            row.setdefault("gini_weighted", row["mean_score"])
    return rows


def get_comparisons(track_id: str, run_id: str) -> list[dict]:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return []
    rows = [_row_to_dict(r) for r in list_comparisons(cfg.registry_path)]
    # Normalise: use final_decision as the canonical decision field
    for row in rows:
        if row and "final_decision" in row:
            row.setdefault("decision", row["final_decision"])
        if row and "challenger_win_rate" in row:
            row.setdefault("win_rate", row["challenger_win_rate"])
    return rows


def get_champion(track_id: str, run_id: str) -> dict | None:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return None
    champ = _row_to_dict(get_official_champion(cfg.registry_path))
    if champ:
        champ = _enrich_champion(champ, cfg.registry_path)
    return champ


def get_champion_history(track_id: str, run_id: str) -> list[dict]:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return []
    rows = [_row_to_dict(r) for r in list_champion_history(cfg.registry_path)]
    for row in rows:
        if row and row.get("champion_id"):
            row.update(_fetch_experiment_metrics(row["champion_id"], cfg.registry_path))
    return rows


def get_proposals(track_id: str, run_id: str) -> list[dict]:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return []
    return [_row_to_dict(r) for r in list_proposals(cfg.registry_path)]


def get_sessions(track_id: str, run_id: str) -> list[dict]:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return []
    rows = [_row_to_dict(r) for r in list_sessions(cfg.registry_path)]
    for row in rows:
        try:
            row["events"] = [_row_to_dict(e) for e in list_session_events(cfg.registry_path, row["session_id"])]
        except Exception:
            row["events"] = []
    return rows


def get_research_lines(track_id: str, run_id: str) -> list[dict]:
    cfg = load_config(track_id=track_id, run_id=run_id)
    if not cfg.registry_path.exists():
        return []
    return [_row_to_dict(r) for r in list_research_lines(cfg.registry_path)]


def get_run_telemetry(track_id: str, run_id: str) -> dict:
    from autoresearch.telemetry.store import get_run_telemetry as read_telemetry

    run_dir = ARTIFACTS_DIR / track_id / "runs" / run_id
    return read_telemetry(run_dir)


def list_artifact_paths(track_id: str, run_id: str) -> list[str]:
    run_dir = ARTIFACTS_DIR / track_id / "runs" / run_id
    if not run_dir.exists():
        return []
    paths = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            paths.append(str(p.relative_to(run_dir)))
    return paths


def read_artifact(track_id: str, run_id: str, rel_path: str) -> Path | None:
    run_dir = ARTIFACTS_DIR / track_id / "runs" / run_id
    target = (run_dir / rel_path).resolve()
    # Prevent path traversal out of run_dir
    if run_dir.resolve() not in target.parents and target != run_dir.resolve():
        return None
    return target if target.exists() and target.is_file() else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich_champion(champ: dict, registry_path: Path) -> dict:
    """Join champion row with its experiment to surface gini_weighted."""
    champion_id = champ.get("champion_id")
    if champion_id:
        metrics = _fetch_experiment_metrics(champion_id, registry_path)
        champ.update(metrics)
    return champ


def _fetch_experiment_metrics(experiment_id: str, registry_path: Path) -> dict:
    """
    Read metrics for a specific experiment_id.
    Uses the same JSON-hydration pattern as list_experiments() so mean_score
    is populated from the metrics.json file rather than the DB row.
    """
    try:
        from autoresearch.utils.io import read_json as _read_json
        conn = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT experiment_id, experiment_name, model_family, status, metrics_path "
            "FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        conn.close()
        if not row:
            return {}
        item = dict(row)
        metrics_path = Path(item["metrics_path"]) if item.get("metrics_path") else None
        if metrics_path and metrics_path.exists():
            metrics = _read_json(metrics_path)
            item["mean_score"] = metrics.get("aggregate", {}).get("mean_score")
            item["std_score"] = metrics.get("aggregate", {}).get("std_score")
            item["primary_metric"] = metrics.get("primary_metric")
        if item.get("primary_metric") == "gini_weighted" and item.get("mean_score") is not None:
            item["gini_weighted"] = item["mean_score"]
        return item
    except Exception:
        return {}


def _row_to_dict(row: Any) -> dict | None:
    if row is None:
        return None
    if hasattr(row, "_asdict"):
        return dict(row._asdict())
    if hasattr(row, "keys"):
        return dict(row)
    return dict(row)
