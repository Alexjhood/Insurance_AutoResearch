"""Retrospective migration to the descriptive ``<ts>__<dataset>__<target>`` id scheme.

Spec: ``docs/internal/run_identity_cost_flightdeck_spec_20260713.md`` §8 (Phase 7).

For every existing orchestration (``artifacts/orchestrations/*``) and solo track run
(``artifacts/tracks/*/runs/*`` without an ``orchestration_id``) this script:

1. **Resolves true identity + cost** *before* renaming, so the descriptor label is
   correct. Orchestrations use the §5.1 transcript import
   (:func:`autoresearch.orchestration.usage.import_orchestrator_usage`); solo runs use
   :func:`autoresearch.telemetry.importer.reconcile_model_identity` +
   :func:`autoresearch.telemetry.costing.compute_run_cost`. A missing transcript
   degrades gracefully (declared identity kept, ``identity_verified=false``).
2. **Computes the new id** ``<timestamp>__<datasetAbbr>__<targetAbbr>`` (the timestamp
   is the existing id's prefix — for orchestrations that is the whole current id).
3. **Renames the directory** and **reroutes every reference** (spec §8): the orchestration
   manifest, campaign report, per-delegation reports, playoff report, the consolidation
   run's back-pointer, delegation compatibility symlinks, run-scope files; for solo runs
   the run manifest, ``latest_run.json``, and cross-run memory rows.
4. **Leaves an old→new symlink** so any stale absolute path still resolves.
5. **Writes the ``descriptor`` block + a ``RUN_INDEX.json`` entry** for the new id.
6. Emits **``migration_report.json``** mapping old→new for every entity.

Then it rebuilds Flight Deck (``--force``) and verifies every migrated entity has a
snapshot.

Safety: dry-run by default (prints every rename + reference rewrite); ``--apply`` writes;
a dirty git tree is refused unless ``--force``. Each entity is processed fully (resolve →
rename → reroute) and fsynced before the next, so a crash leaves a consistent partial
state. Idempotent: an id already carrying a ``__`` suffix is skipped.

Usage::

    python scripts/migrate_run_naming.py [--dry-run] [--apply] [--force]
                                         [--repo-root PATH] [--skip-flightdeck]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Make ``autoresearch`` importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from autoresearch.config import build_descriptor, parse_run_timestamp, run_slug  # noqa: E402

#: Text files whose contents may embed an id/path reference and are rewritten in place.
_TEXT_SUFFIXES = (".json", ".md", ".log", ".txt")


# --------------------------------------------------------------------------- #
# Small IO helpers (repo-root parameterised so the fixture test never touches live)
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fsync_dir(path: Path) -> None:
    """Best-effort directory fsync so a rename is durable before the next entity."""

    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# Id computation + token rewriting
# --------------------------------------------------------------------------- #
def _already_descriptive(run_id: str) -> bool:
    return "__" in run_id


def compute_new_id(old_id: str, dataset: str, target: str) -> str | None:
    """Return ``<ts>__<datasetAbbr>__<targetAbbr>`` or ``None`` if unresolvable/already done."""

    if _already_descriptive(old_id):
        return None
    try:
        ts = parse_run_timestamp(old_id)
        return f"{ts}__{run_slug(dataset, target)}"
    except (ValueError, FileNotFoundError, KeyError):
        return None


def _rewrite_token(text: str, old_id: str, new_id: str) -> tuple[str, int]:
    """Replace whole-token occurrences of *old_id* with *new_id*.

    Guards: the match must not be part of a larger alnum token (lookbehind) and must
    not already be followed by the ``__`` descriptive suffix (lookahead), so a
    second run is a no-op and a longer, coincidentally-prefixed id is never touched.
    """

    pattern = re.compile(r"(?<![0-9A-Za-z_])" + re.escape(old_id) + r"(?!__)")
    new_text, n = pattern.subn(new_id, text)
    return new_text, n


def _reroute_text_files(root: Path, old_id: str, new_id: str, *, apply: bool) -> list[str]:
    """Token-rewrite every text file under *root*; return human-readable actions."""

    actions: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        new_text, n = _rewrite_token(text, old_id, new_id)
        if n == 0:
            continue
        actions.append(f"rewrite {n}× {old_id}→{new_id} in {path.name}")
        if apply:
            path.write_text(new_text, encoding="utf-8")
    return actions


# --------------------------------------------------------------------------- #
# Symlinks
# --------------------------------------------------------------------------- #
def _repoint_orchestration_compat_links(
    repo_root: Path, old_id: str, new_id: str, *, apply: bool
) -> list[str]:
    """Re-point delegation compatibility symlinks from the old campaign dir to the new one."""

    actions: list[str] = []
    tracks = repo_root / "artifacts" / "tracks"
    if not tracks.is_dir():
        return actions
    needle = f"/orchestrations/{old_id}/"
    tail = f"/orchestrations/{old_id}"
    for track_dir in sorted(tracks.iterdir()):
        runs = track_dir / "runs"
        if not runs.is_dir():
            continue
        for link in sorted(runs.iterdir()):
            if not link.is_symlink():
                continue
            target = os.readlink(link)
            if needle in target or target.endswith(tail):
                new_target = target.replace(
                    f"/orchestrations/{old_id}", f"/orchestrations/{new_id}", 1
                )
                actions.append(f"repoint compat link {link.name} → {new_target}")
                if apply:
                    link.unlink()
                    link.symlink_to(new_target)
    return actions


def _leave_alias(old_dir: Path, new_dir: Path, *, apply: bool) -> str:
    """Leave an ``old_id → new_dir`` symlink so stale absolute paths still resolve."""

    action = f"symlink alias {old_dir.name} → {new_dir.name}"
    if apply:
        # ``old_dir`` no longer exists (it was renamed); create the alias in its place.
        old_dir.symlink_to(new_dir.resolve())
    return action


# --------------------------------------------------------------------------- #
# Scope + latest-run + memory reroute
# --------------------------------------------------------------------------- #
def _reroute_scope_files(
    repo_root: Path, old_id: str, new_id: str, *, apply: bool
) -> list[str]:
    actions: list[str] = []
    scope_dir = repo_root / "artifacts" / "tracks" / ".scope"
    if not scope_dir.is_dir():
        return actions
    for path in sorted(scope_dir.glob("*.json")):
        data = _load_json(path)
        if not isinstance(data, dict) or str(data.get("orchestration_id") or "") != old_id:
            continue
        actions.append(f"reroute scope {path.name}.orchestration_id → {new_id}")
        if apply:
            data["orchestration_id"] = new_id
            _write_json(path, data)
    return actions


def _reroute_latest_run(
    repo_root: Path, track: str, old_id: str, new_id: str, new_dir: Path, *, apply: bool
) -> list[str]:
    actions: list[str] = []
    path = repo_root / "artifacts" / "tracks" / track / "latest_run.json"
    data = _load_json(path)
    if not isinstance(data, dict) or str(data.get("run_id") or "") != old_id:
        return actions
    actions.append(f"reroute latest_run.json ({track}) run_id → {new_id}")
    if apply:
        data["run_id"] = new_id
        data["run_dir"] = str(new_dir.resolve())
        _write_json(path, data)
    return actions


def _reroute_memory(
    dataset: str, track: str, old_id: str, new_id: str, *, apply: bool
) -> list[str]:
    """Rewrite cross-run memory rows keyed by the old ``<track>/<run_id>`` uid."""

    actions: list[str] = []
    try:
        from autoresearch.memory.store import default_memory_store_path
    except Exception:
        return actions
    db = default_memory_store_path(dataset)
    if not db.exists():
        return actions
    old_uid = f"{track}/{old_id}"
    new_uid = f"{track}/{new_id}"
    import sqlite3

    con = sqlite3.connect(db)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM runs WHERE run_uid = ?", (old_uid,)
        ).fetchone()[0]
        if not n:
            return actions
        actions.append(f"reroute memory rows {old_uid} → {new_uid} ({dataset})")
        if apply:
            con.execute("PRAGMA foreign_keys = OFF")
            con.execute(
                "UPDATE runs SET run_uid = ?, run_id = ? WHERE run_uid = ?",
                (new_uid, new_id, old_uid),
            )
            for table in ("experiments", "comparisons", "insights"):
                con.execute(
                    f"UPDATE {table} SET run_uid = ? WHERE run_uid = ?",
                    (new_uid, old_uid),
                )
            con.commit()
    finally:
        con.close()
    return actions


# --------------------------------------------------------------------------- #
# RUN_INDEX (repo-root parameterised)
# --------------------------------------------------------------------------- #
def _run_index_path(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "RUN_INDEX.json"


def _update_run_index(
    repo_root: Path,
    *,
    new_id: str,
    old_id: str,
    kind: str,
    path: Path,
    descriptor: dict[str, Any] | None,
    cost_usd: float | None,
    created_at: str | None,
    apply: bool,
) -> None:
    if not apply:
        return
    index_path = _run_index_path(repo_root)
    payload = _load_json(index_path) if index_path.exists() else None
    runs = (payload or {}).get("runs") if isinstance(payload, dict) else payload
    entries = [dict(e) for e in runs] if isinstance(runs, list) else []
    by_id = {str(e.get("id")): e for e in entries if e.get("id")}
    by_id.pop(old_id, None)  # drop the stale seed entry
    entry = by_id.get(new_id, {"id": new_id})
    entry["kind"] = kind
    entry["path"] = _rel(path, repo_root)
    if descriptor is not None:
        entry["descriptor"] = descriptor
    if cost_usd is not None:
        entry["cost_usd"] = cost_usd
    if created_at is not None:
        entry["created_at"] = created_at
    by_id[new_id] = entry
    ordered = [by_id[k] for k in sorted(by_id)]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(index_path, {"runs": ordered})


# --------------------------------------------------------------------------- #
# Cost helpers
# --------------------------------------------------------------------------- #
def _cost_block(run_cost) -> dict[str, Any]:
    """Serialise a :class:`RunCost` into a manifest ``cost`` block (mirror cost_writer)."""

    return {
        "source": "telemetry",
        "currency": "usd",
        "cost_usd": round(run_cost.cost_usd, 6),
        "cost_estimated": bool(run_cost.cost_estimated),
        "unpriced_models": list(run_cost.unpriced_models),
        "by_model": [
            {
                "role": u.role,
                "model": u.model,
                "effort": u.effort,
                "cost_usd": round(u.cost_usd, 6),
                "priced": bool(u.priced),
                "tokens": asdict(u.tokens),
            }
            for u in run_cost.by_model
        ],
        "subagents": {
            "count": run_cost.subagent.count,
            "cost_usd": round(run_cost.subagent.cost_usd, 6),
            "tokens": asdict(run_cost.subagent.tokens),
        },
    }


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_orchestrations(repo_root: Path) -> list[str]:
    root = repo_root / "artifacts" / "orchestrations"
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.is_symlink() and (p / "orchestration.json").exists()
    )


def discover_solo_runs(repo_root: Path) -> list[tuple[str, str, Path]]:
    """Return ``(track, run_id, run_dir)`` for every real solo track run.

    Skips symlinks (delegation compat links / rename aliases) and campaign-owned runs
    (those carrying ``orchestration_id``, ``delegation_id``, or ``consolidation``).
    """

    tracks = repo_root / "artifacts" / "tracks"
    if not tracks.is_dir():
        return []
    out: list[tuple[str, str, Path]] = []
    for track_dir in sorted(tracks.iterdir()):
        runs = track_dir / "runs"
        if not track_dir.is_dir() or not runs.is_dir():
            continue
        for run_dir in sorted(runs.iterdir()):
            if run_dir.is_symlink() or not run_dir.is_dir():
                continue
            manifest = _load_json(run_dir / "run_manifest.json")
            if not isinstance(manifest, dict):
                continue
            if (
                manifest.get("orchestration_id")
                or manifest.get("delegation_id")
                or manifest.get("consolidation")
            ):
                continue
            out.append((track_dir.name, run_dir.name, run_dir))
    return out


# --------------------------------------------------------------------------- #
# Per-entity migration
# --------------------------------------------------------------------------- #
def migrate_orchestration(
    old_id: str, repo_root: Path, *, apply: bool, live: bool, log
) -> dict[str, Any] | None:
    orch_root = repo_root / "artifacts" / "orchestrations"
    old_dir = orch_root / old_id
    manifest_path = old_dir / "orchestration.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        log(f"  skip {old_id}: no readable orchestration.json")
        return None
    dataset = str(manifest.get("dataset") or "")
    target = str(manifest.get("target_mode") or "")
    new_id = compute_new_id(old_id, dataset, target)
    if new_id is None:
        log(f"  skip {old_id}: already descriptive or unresolvable ({dataset}/{target})")
        return None

    actions: list[str] = []

    # 1. Resolve identity + cost BEFORE renaming (spec ordering). Live+apply only —
    #    transcript import mutates disk and needs the real repo layout.
    identity_verified = bool(manifest.get("orchestrator", {}).get("identity_verified"))
    if apply and live:
        try:
            from autoresearch.orchestration.campaign_report import write_campaign_report
            from autoresearch.orchestration.manifest import campaign_report_json_path
            from autoresearch.orchestration.usage import import_orchestrator_usage

            usage = import_orchestrator_usage(old_id)
            if campaign_report_json_path(old_id).exists():
                write_campaign_report(old_id)
            identity_verified = bool(usage.get("identity_verified"))
            actions.append(
                f"resolve identity: verified={identity_verified} "
                f"headline={usage.get('headline')}"
            )
            manifest = _load_json(manifest_path) or manifest
        except Exception as exc:  # never wedge the migration on a resolution failure
            actions.append(f"identity resolution failed ({type(exc).__name__}); kept declared")

    orch = manifest.get("orchestrator", {}) if isinstance(manifest.get("orchestrator"), dict) else {}
    principal_model = orch.get("model") or manifest.get("model_name")
    principal_effort = orch.get("effort") or manifest.get("model_effort")
    cost_usd = _read_campaign_cost(old_dir)

    descriptor = build_descriptor(
        run_id=new_id, dataset=dataset, target=target, kind="orchestrated",
        principal_model=principal_model, principal_effort=principal_effort,
        identity_verified=identity_verified,
    )

    cons = manifest.get("consolidation") or {}
    cons_track = cons.get("track")
    cons_run_id = cons.get("run_id")

    new_dir = orch_root / new_id
    if new_dir.exists():
        log(f"  skip {old_id}: destination {new_id} already exists")
        return None

    log(f"  {old_id} → {new_id}  ({dataset}/{target})")

    # 2. Rename the directory.
    actions.append(f"rename dir {old_id} → {new_id}")
    if apply:
        old_dir.rename(new_dir)

    work_dir = new_dir if apply else old_dir

    # 3. Reroute references inside the campaign dir (manifest, reports, playoff, md, logs).
    actions += _reroute_text_files(work_dir, old_id, new_id, apply=apply)

    # 3b. Refresh the descriptor on the (rewritten) manifest so the resolved
    #     identity/verified flag is reflected in the label.
    if apply:
        m = _load_json(work_dir / "orchestration.json")
        if isinstance(m, dict):
            m["descriptor"] = descriptor
            _write_json(work_dir / "orchestration.json", m)

    # 4. Consolidation run back-pointer.
    if cons_track and cons_run_id:
        cons_manifest = (
            repo_root / "artifacts" / "tracks" / cons_track / "runs" / cons_run_id
            / "run_manifest.json"
        )
        cm = _load_json(cons_manifest)
        if isinstance(cm, dict) and str(cm.get("orchestration_id") or "") == old_id:
            actions.append(f"reroute consolidation {cons_track}/{cons_run_id} → {new_id}")
            if apply:
                cm["orchestration_id"] = new_id
                _write_json(cons_manifest, cm)

    # 5. Delegation compatibility symlinks + scope files.
    actions += _repoint_orchestration_compat_links(repo_root, old_id, new_id, apply=apply)
    actions += _reroute_scope_files(repo_root, old_id, new_id, apply=apply)

    # 6. Old→new alias symlink.
    actions.append(_leave_alias(old_dir, new_dir, apply=apply))

    # 7. Descriptor + RUN_INDEX entry.
    _update_run_index(
        repo_root, new_id=new_id, old_id=old_id, kind="orchestrated", path=new_dir,
        descriptor=descriptor, cost_usd=cost_usd,
        created_at=str(manifest.get("created_at") or "") or None, apply=apply,
    )

    if apply:
        _fsync_dir(orch_root)

    for a in actions:
        log(f"      {a}")
    return {
        "kind": "orchestrated", "old_id": old_id, "new_id": new_id,
        "dataset": dataset, "target": target, "identity_verified": identity_verified,
        "cost_usd": cost_usd, "actions": actions,
    }


def _read_campaign_cost(orch_dir: Path) -> float | None:
    report = _load_json(orch_dir / "campaign_report.json")
    if not isinstance(report, dict):
        return None
    cost = ((report.get("framework_computed") or {}).get("cost") or {})
    val = cost.get("cost_usd")
    return float(val) if isinstance(val, (int, float)) else None


def migrate_solo(
    track: str, old_id: str, run_dir: Path, repo_root: Path, *, apply: bool, live: bool, log
) -> dict[str, Any] | None:
    manifest = _load_json(run_dir / "run_manifest.json")
    if not isinstance(manifest, dict):
        log(f"  skip {track}/{old_id}: no readable run_manifest.json")
        return None
    dataset = str(manifest.get("dataset") or "")
    target = str(manifest.get("target_mode") or "")
    new_id = compute_new_id(old_id, dataset, target)
    if new_id is None:
        log(f"  skip {track}/{old_id}: already descriptive or unresolvable")
        return None

    actions: list[str] = []
    cost_usd: float | None = None
    identity_verified = True  # solo identity is telemetry-reconciled

    # 1. Resolve identity + cost BEFORE renaming.
    if apply:
        try:
            from autoresearch.telemetry.costing import compute_run_cost
            from autoresearch.telemetry.importer import reconcile_model_identity
            from autoresearch.telemetry.store import telemetry_path

            recon = reconcile_model_identity(run_dir)
            actions.append(f"reconcile identity: {recon.get('status')}")
            run_cost = compute_run_cost(telemetry_path(run_dir))
            cost_usd = round(run_cost.cost_usd, 6)
            manifest = _load_json(run_dir / "run_manifest.json") or manifest
            manifest["cost"] = _cost_block(run_cost)
            _write_json(run_dir / "run_manifest.json", manifest)
        except Exception as exc:
            actions.append(f"identity/cost resolution failed ({type(exc).__name__})")

    model_identity = manifest.get("model_identity") or {}
    descriptor = build_descriptor(
        run_id=new_id, dataset=dataset, target=target, kind="solo",
        principal_model=model_identity.get("name"),
        principal_effort=None, identity_verified=identity_verified,
    )

    new_dir = run_dir.parent / new_id
    if new_dir.exists():
        log(f"  skip {track}/{old_id}: destination {new_id} already exists")
        return None

    log(f"  {track}/{old_id} → {new_id}  ({dataset}/{target})")

    # 2. Rename dir.
    actions.append(f"rename dir {old_id} → {new_id}")
    if apply:
        run_dir.rename(new_dir)
    work_dir = new_dir if apply else run_dir

    # 3. Update manifest run_id/run_dir/descriptor.
    actions.append(f"update run_manifest run_id → {new_id}")
    if apply:
        m = _load_json(work_dir / "run_manifest.json") or {}
        m["run_id"] = new_id
        m["run_dir"] = str(work_dir.resolve())
        m["descriptor"] = descriptor
        _write_json(work_dir / "run_manifest.json", m)

    # 4. latest_run.json + memory rows.
    actions += _reroute_latest_run(repo_root, track, old_id, new_id, work_dir, apply=apply)
    actions += _reroute_memory(dataset, track, old_id, new_id, apply=apply)

    # 5. Old→new alias symlink.
    actions.append(_leave_alias(run_dir, new_dir, apply=apply))

    # 6. RUN_INDEX entry.
    _update_run_index(
        repo_root, new_id=new_id, old_id=old_id, kind="solo", path=new_dir,
        descriptor=descriptor, cost_usd=cost_usd,
        created_at=str(manifest.get("created_at") or "") or None, apply=apply,
    )

    if apply:
        _fsync_dir(run_dir.parent)

    for a in actions:
        log(f"      {a}")
    return {
        "kind": "solo", "track": track, "old_id": old_id, "new_id": new_id,
        "dataset": dataset, "target": target, "cost_usd": cost_usd, "actions": actions,
    }


# --------------------------------------------------------------------------- #
# Flight Deck rebuild + verification
# --------------------------------------------------------------------------- #
def rebuild_flightdeck(repo_root: Path, entities: list[dict], *, apply: bool, log) -> dict:
    """Rebuild Flight Deck snapshots and verify every migrated entity renders."""

    if not apply:
        return {"status": "skipped_dry_run"}

    snapshots = repo_root / "flightdeck" / "snapshots"
    # Drop stale snapshots (old ids + the merged index) so the rebuild is clean.
    for ent in entities:
        old_snap = snapshots / ent["old_id"]
        if old_snap.exists():
            import shutil

            shutil.rmtree(old_snap, ignore_errors=True)
    (snapshots / "index.json").unlink(missing_ok=True)

    cmd = [sys.executable, "-m", "flightdeck.etl", "--all", "--force"]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  flightdeck build FAILED: {proc.stderr.strip()[:500]}")
        return {"status": "build_failed", "stderr": proc.stderr[-2000:]}

    index = _load_json(snapshots / "index.json") or {}
    present = {e.get("orch_id") for e in index.get("orchestrations", [])}
    missing = [ent["new_id"] for ent in entities if ent["new_id"] not in present]
    for ent in entities:
        ok = ent["new_id"] in present and (snapshots / ent["new_id"] / "snapshot.json").exists()
        log(f"  render {ent['new_id']}: {'OK' if ok else 'MISSING'}")
    return {
        "status": "ok" if not missing else "incomplete",
        "rendered": sorted(present & {ent["new_id"] for ent in entities}),
        "missing": missing,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _git_is_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out.strip())


def migrate(
    repo_root: Path, *, apply: bool, force: bool, skip_flightdeck: bool, log=print
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    live = repo_root == _REPO_ROOT.resolve()

    if apply and not force and _git_is_dirty(repo_root):
        raise SystemExit(
            "ABORT: git tree is dirty. Commit or stash first, or pass --force to override."
        )

    entities: list[dict] = []

    log("Orchestrations:")
    for oid in discover_orchestrations(repo_root):
        result = migrate_orchestration(oid, repo_root, apply=apply, live=live, log=log)
        if result is not None:
            entities.append(result)

    log("Solo runs:")
    for track, run_id, run_dir in discover_solo_runs(repo_root):
        result = migrate_solo(track, run_id, run_dir, repo_root, apply=apply, live=live, log=log)
        if result is not None:
            entities.append(result)

    flightdeck = {"status": "skipped"}
    if not skip_flightdeck:
        log("Flight Deck rebuild:")
        flightdeck = rebuild_flightdeck(repo_root, entities, apply=apply, log=log)

    report = {
        "dry_run": not apply,
        "repo_root": str(repo_root),
        "n_migrated": len(entities),
        "mapping": [
            {"kind": e["kind"], "old_id": e["old_id"], "new_id": e["new_id"]}
            for e in entities
        ],
        "entities": entities,
        "flightdeck": flightdeck,
    }
    if apply:
        report_path = repo_root / "artifacts" / "migration_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_path, report)
        log(f"\nWrote {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="Write changes. Without it, print planned actions (dry-run).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Explicit dry-run (the default when --apply is absent).")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if the git tree is dirty.")
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT,
                        help="Repository root (defaults to this checkout; used by tests).")
    parser.add_argument("--skip-flightdeck", action="store_true",
                        help="Do not rebuild Flight Deck snapshots afterwards.")
    args = parser.parse_args(argv)

    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    apply = args.apply

    report = migrate(
        args.repo_root, apply=apply, force=args.force,
        skip_flightdeck=args.skip_flightdeck,
    )
    if not apply:
        print("\n" + json.dumps(
            {"dry_run": True, "n_would_migrate": report["n_migrated"],
             "mapping": report["mapping"]},
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
