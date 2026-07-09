"""One-time migration of the French dataset into the uniform multi-dataset layout.

Moves ``data/{raw,processed,metadata,splits,holdout_vault}`` into
``data/datasets/french_motor/…`` without regenerating anything — the split pack
is verified byte-identical (SHA-256) before and after. Existing run manifests are
backfilled with ``dataset = "french_motor"``.

Idempotent: re-running after a completed migration is a no-op. Safe to run before
the raw freMTPL2 files exist (it simply skips what is absent).

Usage:
    python scripts/migrate_dataset_layout.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
DEST = DATA / "datasets" / "french_motor"

# Content directories moved wholesale (all git-ignored generated artifacts).
MOVE_DIRS = ("processed", "metadata", "splits", "holdout_vault")
# Raw entries relocated into french_motor/raw, leaving tracked placeholders behind.
RAW_KEEP = {".gitkeep", "README.md", ".DS_Store"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _move_tree(src: Path, dst: Path, *, dry_run: bool) -> list[str]:
    actions: list[str] = []
    if not src.exists():
        return actions
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Merge: move any entries not already present at the destination.
        for entry in src.iterdir():
            target = dst / entry.name
            if target.exists():
                continue
            actions.append(f"move {entry} -> {target}")
            if not dry_run:
                shutil.move(str(entry), str(target))
    else:
        actions.append(f"move {src} -> {dst}")
        if not dry_run:
            shutil.move(str(src), str(dst))
    return actions


def _migrate_memory(*, dry_run: bool) -> list[str]:
    """Move an unscoped cross-run memory store into the french_motor/ subfolder.

    The store lives outside the working tree; existing insights predate
    multi-dataset support and belong to French. The recipe ledger
    (``recipes.jsonl``) stays at the root (it is not dataset-scoped)."""

    actions: list[str] = []
    try:
        from autoresearch.memory.store import memory_root
    except Exception:
        return actions
    root = memory_root()  # unscoped
    dest = root / "french_motor"
    for name in ("memory.sqlite", "playbook"):
        src = root / name
        if not src.exists():
            continue
        target = dest / name
        if target.exists():
            continue
        actions.append(f"move {src} -> {target}")
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(target))
    return actions


def migrate(dry_run: bool = False) -> dict:
    actions: list[str] = []
    DEST.mkdir(parents=True, exist_ok=True)

    # 0. Capture the split-pack hash before moving, for the byte-identical check.
    old_split = DATA / "splits" / "split_pack.csv"
    new_split = DEST / "splits" / "split_pack.csv"
    pre_hash = _sha256(old_split) if old_split.exists() else (
        _sha256(new_split) if new_split.exists() else None
    )

    # 1. Move generated content directories.
    for name in MOVE_DIRS:
        actions += _move_tree(DATA / name, DEST / name, dry_run=dry_run)

    # 2. Relocate raw data files (leave tracked placeholders in data/raw).
    raw_src = DATA / "raw"
    raw_dst = DEST / "raw"
    if raw_src.exists():
        raw_dst.mkdir(parents=True, exist_ok=True)
        for entry in raw_src.iterdir():
            if entry.name in RAW_KEEP:
                continue
            target = raw_dst / entry.name
            if target.exists():
                continue
            actions.append(f"move {entry} -> {target}")
            if not dry_run:
                shutil.move(str(entry), str(target))

    # 3. Verify the split pack is byte-identical.
    post_hash = _sha256(new_split) if new_split.exists() else None
    if pre_hash is not None and post_hash is not None and pre_hash != post_hash:
        raise SystemExit(
            f"ABORT: split pack hash changed during migration "
            f"({pre_hash} -> {post_hash}). No regeneration should have occurred."
        )

    # 3b. Move existing (unscoped) cross-run memory into the french_motor scope.
    actions += _migrate_memory(dry_run=dry_run)

    # 4. Backfill dataset pin into existing run manifests.
    tracks = PROJECT_ROOT / "artifacts" / "tracks"
    backfilled = 0
    if tracks.exists():
        for manifest in tracks.rglob("run_manifest.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("dataset"):
                continue
            payload["dataset"] = "french_motor"
            actions.append(f"backfill dataset -> {manifest}")
            backfilled += 1
            if not dry_run:
                manifest.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

    return {
        "dry_run": dry_run,
        "split_pack_hash": post_hash or pre_hash,
        "split_pack_byte_identical": (pre_hash == post_hash) if (pre_hash and post_hash) else None,
        "manifests_backfilled": backfilled,
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without moving anything.")
    args = parser.parse_args()
    result = migrate(dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
