"""Copy the verbatim artifact set into ``snapshots/<id>/files/`` + manifest.

Copy set and rules per DATA.md §2.7: markdown/json/logs copied verbatim, stdout
logs tail-capped to the last 2 MB with a truncation marker. Never copies
sqlite DBs (queried, not shipped) or anything under ``baseline/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..schema import FileEntry
from ..util import Warnings

STDOUT_TAIL_CAP = 2 * 1024 * 1024  # 2 MB
_TRUNCATION_MARKER = "[... truncated to last 2 MB by flightdeck ETL ...]\n"


def _kind(rel: str) -> str:
    low = rel.lower()
    if low.endswith(".md"):
        return "markdown"
    if low.endswith(".json"):
        return "json"
    if low.endswith(".log"):
        return "log"
    return "text"


def _iter_copy_set(orch_dir: Path) -> Iterable[Path]:
    """Yield existing source paths to copy (absolute)."""
    singles = ["CAMPAIGN_REPORT.md", "ORCHESTRATION_LOG.md",
               "playoff/playoff_report.md", "playoff/playoff_report.json"]
    for rel in singles:
        p = orch_dir / rel
        if p.is_file():
            yield p
    for sub, pattern in [("prompts", "*.md"), ("briefs", "*.json"),
                         ("reports", "*.json"), ("logs", "*.stdout.log"),
                         ("logs", "*.exit.json")]:
        d = orch_dir / sub
        if d.is_dir():
            for p in sorted(d.glob(pattern)):
                if p.is_file():
                    yield p
    runs = orch_dir / "runs"
    if runs.is_dir():
        for run_dir in sorted(runs.iterdir()):
            if not run_dir.is_dir():
                continue
            for name in ("RESEARCH_LOG.md", "LLM_USAGE.md", "run_manifest.json"):
                p = run_dir / name
                if p.is_file():
                    yield p


def copy_files(
    orch_dir: Path, out_files_dir: Path, repo_root: Path, warnings: Warnings
) -> list[FileEntry]:
    manifest: list[FileEntry] = []
    for src in _iter_copy_set(orch_dir):
        rel = src.relative_to(orch_dir).as_posix()
        if rel.startswith("baseline/"):
            continue
        dst = out_files_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            truncated = False
            if rel.endswith(".stdout.log"):
                truncated = _copy_tail_capped(src, dst)
            else:
                dst.write_bytes(src.read_bytes())
            try:
                source_rel = src.relative_to(repo_root).as_posix()
            except ValueError:
                source_rel = str(src)
            manifest.append(
                FileEntry(
                    path=rel,
                    kind=_kind(rel),
                    bytes=dst.stat().st_size,
                    truncated=truncated,
                    source=source_rel,
                )
            )
        except OSError as exc:
            warnings.add(f"failed to copy {rel}: {exc}")
    manifest.sort(key=lambda e: e.path)
    return manifest


def _copy_tail_capped(src: Path, dst: Path) -> bool:
    """Copy a log, keeping only the last STDOUT_TAIL_CAP bytes. Returns truncated."""
    size = src.stat().st_size
    if size <= STDOUT_TAIL_CAP:
        dst.write_bytes(src.read_bytes())
        return False
    with src.open("rb") as fh:
        fh.seek(size - STDOUT_TAIL_CAP)
        tail = fh.read()
    with dst.open("wb") as out:
        out.write(_TRUNCATION_MARKER.encode("utf-8"))
        out.write(tail)
    return True
