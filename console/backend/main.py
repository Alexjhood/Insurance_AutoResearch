"""AutoResearch Console — FastAPI orchestrator."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Ensure repo root is on sys.path so `console.*` and `autoresearch.*` resolve
# regardless of how uvicorn is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from console.backend import config as cfg
from console.backend.auth import AuthMiddleware
from console.backend.readers import runs as run_reader
from console.backend.readers import memory as mem_reader
from console.backend.orchestrator import db
from console.backend.orchestrator import jobs as job_mgr


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: reconcile any jobs that were running before restart
    job_mgr.reattach_running_jobs()
    yield


app = FastAPI(title="AutoResearch Console", version="0.1.0", lifespan=lifespan)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "repo": str(cfg.REPO_ROOT),
        "claude_bin": cfg.CLAUDE_BIN,
        "codex_bin": cfg.CODEX_BIN,
        "opencode_bin": cfg.OPENCODE_BIN,
        "memory_db_exists": cfg.MEMORY_DB.exists(),
    }


# ── Tracks & Runs (read-only) ─────────────────────────────────────────────────

@app.get("/api/tracks")
def tracks() -> list[dict]:
    return run_reader.list_tracks()


@app.get("/api/tracks/{track}/runs")
def runs(track: str) -> list[dict]:
    return run_reader.list_runs(track)


@app.get("/api/tracks/{track}/runs/{run_id}")
def run_summary(track: str, run_id: str) -> dict:
    return run_reader.run_summary(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/experiments")
def experiments(track: str, run_id: str) -> list[dict]:
    return run_reader.get_experiments(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/comparisons")
def comparisons(track: str, run_id: str) -> list[dict]:
    return run_reader.get_comparisons(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/champion")
def champion(track: str, run_id: str) -> dict | None:
    return run_reader.get_champion(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/champion-history")
def champion_history(track: str, run_id: str) -> list[dict]:
    return run_reader.get_champion_history(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/proposals")
def proposals(track: str, run_id: str) -> list[dict]:
    return run_reader.get_proposals(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/sessions")
def sessions(track: str, run_id: str) -> list[dict]:
    return run_reader.get_sessions(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/research-lines")
def research_lines(track: str, run_id: str) -> list[dict]:
    return run_reader.get_research_lines(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/artifact-paths")
def artifact_paths(track: str, run_id: str) -> list[str]:
    return run_reader.list_artifact_paths(track, run_id)


@app.get("/api/tracks/{track}/runs/{run_id}/artifacts/{path:path}")
async def artifact(track: str, run_id: str, path: str):
    file_path = run_reader.read_artifact(track, run_id, path)
    if not file_path:
        raise HTTPException(404, "Artifact not found or access denied")
    mime, _ = mimetypes.guess_type(str(file_path))
    mime = mime or "application/octet-stream"
    if mime.startswith("text/") or mime in ("application/json",):
        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = await f.read()
        return Response(content=content, media_type=mime)
    return FileResponse(str(file_path), media_type=mime)


# ── Leaderboard ───────────────────────────────────────────────────────────────

@app.get("/api/leaderboard")
def leaderboard() -> dict:
    return mem_reader.leaderboard()


@app.get("/api/leaderboard/playbook")
def playbook() -> dict:
    text = mem_reader.playbook_text()
    return {"available": text is not None, "content": text}


class HoldoutRequest(BaseModel):
    track: str
    run_id: str
    token: str  # AUTORESEARCH_MILESTONE_TOKEN — never stored, used once and discarded


@app.post("/api/holdout/evaluate")
def evaluate_holdout(req: HoldoutRequest) -> dict:
    """
    Run `autoresearch evaluate-milestone` with the user-supplied token.
    The token is passed as an env var to the subprocess and is never written
    to orchestrator.db, logs, or any persistent store.
    """
    autoresearch = cfg.AUTORESEARCH_BIN
    env = {**os.environ, "AUTORESEARCH_MILESTONE_TOKEN": req.token}
    token_preview = req.token[:4] + "…" if len(req.token) > 4 else "…"
    try:
        result = subprocess.run(
            [autoresearch, "--track", req.track, "--run-id", req.run_id,
             "evaluate-milestone"],
            cwd=str(cfg.REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "ok": result.returncode == 0,
            "track": req.track,
            "run_id": req.run_id,
            "token_preview": token_preview,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "evaluate-milestone timed out after 300s")
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/leaderboard/harvest")
def harvest_memory() -> dict:
    """
    Trigger `autoresearch memory harvest --all` to populate the cross-run
    memory aggregator. Runs synchronously — may take a few seconds.
    """
    autoresearch = cfg.AUTORESEARCH_BIN
    try:
        result = subprocess.run(
            [autoresearch, "memory", "harvest", "--all"],
            cwd=str(cfg.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Harvest timed out after 120s")
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Jobs ──────────────────────────────────────────────────────────────────────

class LaunchRequest(BaseModel):
    track: str
    surface: str                   # claude | codex | opencode
    model_provider: str
    model_name: str
    cycles: int = 3
    memory_access: str = "none"    # none | own | all
    scope: str = "research"        # research | analyst
    guidance: str = ""


@app.post("/api/jobs", status_code=201)
def launch_job(req: LaunchRequest) -> dict:
    try:
        job_id = job_mgr.launch_job(
            track=req.track,
            surface=req.surface,
            model_provider=req.model_provider,
            model_name=req.model_name,
            cycles=req.cycles,
            memory_access=req.memory_access,
            scope=req.scope,
            guidance=req.guidance,
        )
        return {"job_id": job_id}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return db.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/events")
def get_events(job_id: str, after: int = 0) -> list[dict]:
    return db.get_events(job_id, after_id=after)


class SteerRequest(BaseModel):
    message: str
    interrupt: bool = True


@app.post("/api/jobs/{job_id}/steer")
def steer(job_id: str, req: SteerRequest) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    result = job_mgr.steer_job(job_id, req.message, interrupt=req.interrupt)
    return {"mode": result.mode, "ok": result.ok, "detail": result.detail}


@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    try:
        job_mgr.pause_job(job_id)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"status": "paused"}


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    try:
        job_mgr.resume_job(job_id)
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {"status": "running"}


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_mgr.stop_job(job_id)
    return {"status": "stopped"}


@app.websocket("/api/jobs/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str):
    """Stream new events to the client as they arrive in orchestrator.db."""
    await websocket.accept()
    last_id = 0
    try:
        while True:
            events = db.get_events(job_id, after_id=last_id)
            for event in events:
                last_id = event["id"]
                await websocket.send_text(json.dumps(event))
            job = db.get_job(job_id)
            if job and job["status"] in ("done", "stopped", "failed", "interrupted"):
                await websocket.send_text(json.dumps({
                    "event_type": "stream_end",
                    "status": job["status"],
                }))
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=cfg.HOST, port=cfg.PORT, reload=True)
