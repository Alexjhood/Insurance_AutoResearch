"""Static Flight Deck export builder.

The generated folder is deliberately a dumb file bundle: the application,
snapshot, telemetry, and small evidence files all load without an HTTP server.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

MAX_INLINE_FILE_BYTES = 512 * 1024


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_for_script(value: object) -> str:
    # Escaping '<' prevents file contents containing </script> from terminating
    # an application/json element early.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _script(script_id: str, value: object, **attributes: str) -> str:
    attrs = " ".join(f'{name}="{value}"' for name, value in attributes.items())
    suffix = f" {attrs}" if attrs else ""
    return f'<script type="application/json" id="{script_id}"{suffix}>{_json_for_script(value)}</script>'


def _inline_bundle(destination: Path, html: str) -> str:
    """Inline the single-chunk JS/CSS bundle into index.html.

    Browsers block module scripts and dynamic import() from file:// (the page
    origin is null, so every asset fetch fails CORS). Inlining makes the export
    a genuinely self-contained page.
    """
    pattern = re.compile(
        r'<script type="module"[^>]*src="\./(assets/[^"]+\.js)"></script>'
        r'|<link rel="stylesheet"[^>]*href="\./(assets/[^"]+\.css)">'
    )

    def replace(match: re.Match[str]) -> str:
        js_rel, css_rel = match.group(1), match.group(2)
        if js_rel:
            body = (destination / js_rel).read_text(encoding="utf-8").replace("</script", "<\\/script")
            return f'<script type="module">{body}</script>'
        body = (destination / css_rel).read_text(encoding="utf-8").replace("</style", "<\\/style")
        return f"<style>{body}</style>"

    inlined = pattern.sub(replace, html)
    assets = destination / "assets"
    remaining = [p.name for p in assets.rglob("*") if p.is_file() and p.suffix not in {".js", ".css", ".map"}]
    if remaining:
        raise ValueError(f"Export bundle references non-inlinable assets: {remaining}")
    shutil.rmtree(assets)
    return inlined


def build_export(
    orchestration_id: str,
    *,
    out: Path | None = None,
    build_app: bool = True,
    repo_root: Path | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Path, Path]:
    root = (repo_root or _repo_root()).resolve()
    flightdeck = root / "flightdeck"
    snapshot_root = flightdeck / "snapshots"
    source = snapshot_root / orchestration_id
    snapshot_path = source / "snapshot.json"
    index_path = snapshot_root / "index.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}. Run flightdeck.etl first.")
    if not index_path.is_file():
        raise FileNotFoundError(f"Snapshot index not found: {index_path}")

    app = flightdeck / "app"
    dist = app / "dist-export"
    if build_app:
        log("Building Flight Deck export bundle...")
        subprocess.run(["npm", "run", "build:export"], cwd=app, check=True)
    if not (dist / "index.html").is_file():
        raise FileNotFoundError(
            "flightdeck/app/dist-export is missing; build the app (npm run build:export) or omit --skip-app-build"
        )

    destination = (out or flightdeck / "export" / "out" / orchestration_id).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(dist, destination)

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["orchestrations"] = [
        entry for entry in index.get("orchestrations", []) if entry.get("orch_id") == orchestration_id
    ]
    if not index["orchestrations"]:
        raise ValueError(f"Orchestration {orchestration_id} is absent from snapshots/index.json")

    embedded_files: dict[str, str] = {}
    omitted: set[str] = set()
    for entry in snapshot.get("files", []):
        relative = entry.get("path")
        path = source / "files" / relative if relative else None
        if path and path.is_file() and path.stat().st_size < MAX_INLINE_FILE_BYTES:
            embedded_files[relative] = path.read_text(encoding="utf-8", errors="replace")
        elif relative:
            omitted.add(relative)
    snapshot_copy = copy.deepcopy(snapshot)
    for entry in snapshot_copy.get("files", []):
        entry["included_in_export"] = entry.get("path") not in omitted

    built_at = datetime.now(UTC).isoformat()
    tags = [
        _script("fd-embedded-meta", {"orchestration_id": orchestration_id, "built_at": built_at}, **{"data-built-at": built_at}),
        _script("fd-embedded-index", index),
        _script(f"fd-embedded-snapshot-{orchestration_id}", snapshot_copy),
        _script(f"fd-embedded-files-{orchestration_id}", embedded_files),
    ]
    for telemetry_path in sorted(source.glob("telemetry_*.json")):
        delegation_id = telemetry_path.stem.removeprefix("telemetry_")
        tags.append(_script(
            f"fd-embedded-telemetry-{orchestration_id}-{delegation_id}",
            json.loads(telemetry_path.read_text(encoding="utf-8")),
        ))

    html_path = destination / "index.html"
    html = html_path.read_text(encoding="utf-8")
    html = _inline_bundle(destination, html)
    # Inject before the LAST </head>: the inlined JS bundle contains the literal
    # string "</head>" (e.g. in DOMPurify), and the document's real close tag is
    # the final occurrence.
    head_close = html.rindex("</head>")
    html = html[:head_close] + "\n".join(tags) + "\n" + html[head_close:]
    html_path.write_text(html, encoding="utf-8")

    archive_base = destination.parent / f"{destination.name}-flightdeck"
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=destination))
    telemetry_count = sum(1 for tag in tags if 'id="fd-embedded-telemetry-' in tag)
    log(f"Embedded {telemetry_count} telemetry payloads and {len(embedded_files)} evidence files; omitted {len(omitted)} large files.")
    return destination, archive
