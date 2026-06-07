"""
HTTP + WebSocket integration tests against the real FastAPI app via TestClient.

Covers: read endpoints over real registry data, leaderboard, the auth
middleware no-op path, and the WebSocket stream replaying recorded events.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from console.backend.main import app
from console.backend.orchestrator import db

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_tracks_and_runs():
    r = client.get("/api/tracks")
    assert r.status_code == 200
    tracks = r.json()
    assert isinstance(tracks, list)
    if tracks:
        track = tracks[0]["track_id"]
        rr = client.get(f"/api/tracks/{track}/runs")
        assert rr.status_code == 200


def test_run_telemetry_endpoint_has_stable_shape():
    tracks = client.get("/api/tracks").json()
    if not tracks:
        return
    track = tracks[0]["track_id"]
    runs = client.get(f"/api/tracks/{track}/runs").json()
    if not runs:
        return
    run_id = runs[0]["run_id"]

    response = client.get(f"/api/tracks/{track}/runs/{run_id}/telemetry")

    assert response.status_code == 200
    body = response.json()
    assert "available" in body
    assert "summary" in body
    assert "turns" in body
    assert "workflow_events" in body


def test_artifact_path_traversal_blocked():
    """Ensure ../ escapes are rejected by read_artifact."""
    tracks = client.get("/api/tracks").json()
    if not tracks:
        return
    track = tracks[0]["track_id"]
    runs = client.get(f"/api/tracks/{track}/runs").json()
    if not runs:
        return
    run_id = runs[0]["run_id"]
    # Attempt to escape the run dir
    r = client.get(f"/api/tracks/{track}/runs/{run_id}/artifacts/../../../../etc/passwd")
    assert r.status_code == 404


def test_leaderboard_shape():
    r = client.get("/api/leaderboard")
    assert r.status_code == 200
    data = r.json()
    assert "available" in data
    assert "top_experiments" in data


def test_jobs_list():
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_surfaces_endpoint():
    r = client.get("/api/surfaces")
    assert r.status_code == 200
    surfaces = r.json()
    names = {s["surface"] for s in surfaces}
    assert names == {"claude", "codex", "opencode"}
    for s in surfaces:
        assert "models" in s and "efforts" in s and "effort_label" in s
        assert "" in s["efforts"]


def test_job_telemetry_endpoint():
    job_id = db.create_job(
        track="telemetryT", run_id="telemetryR", surface="codex",
        model_provider="openai", model_name="codex", cycles=1,
        memory_access="none", scope="research", guidance="",
        seed_prompt="", env={},
    )
    turn_id = db.start_telemetry_turn(
        job_id, surface="codex", session_id="thread-1",
        model="gpt-5", effort="medium",
    )
    db.finish_telemetry_turn(
        turn_id, session_id="thread-1", duration_ms=100,
        is_error=False, output_chars=12,
        usage={
            "input_tokens": 100, "cached_input_tokens": 75,
            "cache_creation_input_tokens": None, "uncached_input_tokens": 25,
            "output_tokens": 10, "reasoning_tokens": 4, "total_tokens": 110,
            "provider_cost_usd": None,
        },
        raw_usage={"input_tokens": 100, "cached_input_tokens": 75},
        raw_result={"usage": {"input_tokens": 100}},
    )
    response = client.get(f"/api/jobs/{job_id}/telemetry")
    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["cache_hit_ratio"] == 0.75
    assert summary["reasoning_tokens"] == 4

    conn = db._connect()
    conn.execute("DELETE FROM telemetry_turns WHERE job_id=?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()


def test_websocket_replays_recorded_events():
    """
    Create a job + events directly in the DB, mark it done, then connect the
    WS — it should replay all events and send stream_end.
    """
    job_id = db.create_job(
        track="wsT", run_id="wsR", surface="claude", model_provider="a",
        model_name="m", cycles=1, memory_access="none", scope="research",
        guidance="", seed_prompt="", env={},
    )
    db.record_event(job_id, "token", {"text": "hello"})
    db.record_event(job_id, "turn_end", {"is_error": False})
    db.update_job(job_id, status="done")

    received = []
    with client.websocket_connect(f"/api/jobs/{job_id}/stream") as ws:
        # Read until stream_end
        for _ in range(10):
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("event_type") == "stream_end":
                break

    event_types = [m.get("event_type") for m in received]
    assert "token" in event_types
    assert "turn_end" in event_types
    assert "stream_end" in event_types

    # Clean up the test job + events from the shared DB
    conn = db._connect()
    conn.execute("DELETE FROM events WHERE job_id=?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()


def test_holdout_rejects_bad_token_gracefully():
    tracks = client.get("/api/tracks").json()
    if not tracks:
        return
    track = tracks[0]["track_id"]
    runs = client.get(f"/api/tracks/{track}/runs").json()
    if not runs:
        return
    r = client.post("/api/holdout/evaluate", json={
        "track": track, "run_id": runs[0]["run_id"], "token": "definitely-wrong",
    })
    # Should return 200 with ok=False (graceful), not crash
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    # Token must never be echoed back in full
    assert "definitely-wrong" not in str(body)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
