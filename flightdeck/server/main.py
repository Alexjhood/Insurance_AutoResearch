"""Thin HTTP facade over immutable Flight Deck snapshots."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

FLIGHTDECK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FLIGHTDECK_DIR.parent
SNAPSHOTS_DIR = Path(os.environ.get("FLIGHTDECK_SNAPSHOTS_DIR", FLIGHTDECK_DIR / "snapshots"))
APP_DIST = FLIGHTDECK_DIR / "app" / "dist"

app = FastAPI(title="Flight Deck API", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5199",
        "http://127.0.0.1:5199",
        "http://localhost:8799",
        "http://127.0.0.1:8799",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _json_file(path: Path) -> JSONResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot resource not found")
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Snapshot resource is invalid") from exc


def _orch_dir(orch_id: str) -> Path:
    if not orch_id or orch_id in {".", ".."} or "/" in orch_id or "\\" in orch_id:
        raise HTTPException(status_code=404, detail="Orchestration not found")
    directory = SNAPSHOTS_DIR / orch_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Orchestration not found")
    return directory


@app.get("/api/index")
def get_index() -> JSONResponse:
    path = SNAPSHOTS_DIR / "index.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot resource not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.get("orchestrations", []):
        oid = entry.get("orch_id")
        source = REPO_ROOT / "artifacts" / "orchestrations" / str(oid) / "orchestration.json"
        snapshot = SNAPSHOTS_DIR / str(oid) / "snapshot.json"
        entry["stale"] = bool(source.exists() and snapshot.exists() and source.stat().st_mtime > snapshot.stat().st_mtime)
    return JSONResponse(data)


@app.get("/api/orchestrations/{orch_id}")
def get_orchestration(orch_id: str) -> JSONResponse:
    return _json_file(_orch_dir(orch_id) / "snapshot.json")


@app.get("/api/orchestrations/{orch_id}/telemetry/{delegation_id}")
def get_telemetry(orch_id: str, delegation_id: str) -> JSONResponse:
    if not delegation_id.isalnum():
        raise HTTPException(status_code=404, detail="Telemetry not found")
    return _json_file(_orch_dir(orch_id) / f"telemetry_{delegation_id}.json")


@app.get("/api/orchestrations/{orch_id}/files/{file_path:path}")
def get_file(orch_id: str, file_path: str) -> FileResponse:
    root = (_orch_dir(orch_id) / "files").resolve()
    candidate = (root / file_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path traversal rejected") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    content_type, _ = mimetypes.guess_type(candidate.name)
    if candidate.suffix == ".md":
        content_type = "text/markdown"
    elif candidate.suffix == ".log":
        content_type = "text/plain"
    return FileResponse(candidate, media_type=content_type or "text/plain")


def _sse(event: str, data: object) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def _rebuild_stream(orch_id: str | None = None) -> AsyncIterator[bytes]:
    yield _sse("progress", {"message": "Starting snapshot rebuild"})
    args = [sys.executable, "-m", "flightdeck.etl"]
    args.extend(["--orchestration", orch_id] if orch_id else ["--all"])
    args.append("--force")
    process = await asyncio.create_subprocess_exec(
        *args,
        cwd=FLIGHTDECK_DIR.parent,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    while line := await process.stdout.readline():
        yield _sse("progress", {"message": line.decode(errors="replace").rstrip()})
    return_code = await process.wait()
    if return_code:
        yield _sse("error", {"message": "Snapshot rebuild failed", "return_code": return_code})
    else:
        yield _sse("complete", {"message": "Snapshots rebuilt", "refresh": True})


@app.post("/api/etl/rebuild")
def rebuild(orchestration_id: str | None = None) -> StreamingResponse:
    if orchestration_id and ("/" in orchestration_id or "\\" in orchestration_id or orchestration_id in {".", ".."}):
        raise HTTPException(status_code=400, detail="Invalid orchestration id")
    stream = _rebuild_stream(orchestration_id) if orchestration_id else _rebuild_stream()
    return StreamingResponse(stream, media_type="text/event-stream")


@app.get("/healthz")
def health() -> dict[str, object]:
    index_path = SNAPSHOTS_DIR / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index = {"orchestrations": [], "built_at": None}
    return {
        "status": "ok",
        "snapshot_count": len(index.get("orchestrations", [])),
        "built_at": index.get("built_at"),
    }


if APP_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=APP_DIST / "assets"), name="assets")

    @app.get("/{spa_path:path}", include_in_schema=False)
    def spa_fallback(spa_path: str) -> FileResponse:
        return FileResponse(APP_DIST / "index.html")
