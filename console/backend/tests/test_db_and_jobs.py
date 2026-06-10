"""
Tests for orchestrator DB lifecycle, steer queue, and reattach logic.

Uses a temporary orchestrator.db so the real one is untouched.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _fresh_db_module(tmpdir: str):
    """Reload db module pointed at a temp database."""
    os.environ["AUTORESEARCH_CONSOLE_DB"] = str(Path(tmpdir) / "test_orch.db")
    import console.backend.config as cfg
    importlib.reload(cfg)
    import console.backend.orchestrator.db as db
    importlib.reload(db)
    return db


def test_job_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fresh_db_module(tmp)
        job_id = db.create_job(
            track="t", run_id="r1", surface="claude",
            model_provider="anthropic", model_name="claude-opus-4-8",
            cycles=3, memory_access="none", scope="research",
            guidance="", seed_prompt="go", env={},
        )
        assert job_id
        job = db.get_job(job_id)
        assert job["status"] == "pending"
        assert job["track"] == "t"

        db.update_job(job_id, status="running", pid=999)
        assert db.get_job(job_id)["status"] == "running"
        assert db.get_job(job_id)["pid"] == 999

        jobs = db.list_jobs()
        assert any(j["id"] == job_id for j in jobs)


def test_events_ordering_and_after_id():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fresh_db_module(tmp)
        job_id = db.create_job(
            track="t", run_id="r1", surface="claude",
            model_provider="a", model_name="m", cycles=1,
            memory_access="none", scope="research", guidance="",
            seed_prompt="", env={},
        )
        for i in range(5):
            db.record_event(job_id, "token", {"text": f"tok{i}"})

        all_events = db.get_events(job_id)
        assert len(all_events) == 5
        # Monotonic ids
        ids = [e["id"] for e in all_events]
        assert ids == sorted(ids)

        # after_id filtering
        after = db.get_events(job_id, after_id=ids[2])
        assert len(after) == 2
        assert all(e["id"] > ids[2] for e in after)


def test_agent_session_id_persists():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fresh_db_module(tmp)
        job_id = db.create_job(
            track="t", run_id="r1", surface="claude",
            model_provider="a", model_name="m", cycles=1,
            memory_access="none", scope="research", guidance="",
            seed_prompt="", env={},
        )
        db.update_job(job_id, agent_session_id="sess-xyz")
        assert db.get_job(job_id)["agent_session_id"] == "sess-xyz"


def test_steer_queue():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fresh_db_module(tmp)
        job_id = db.create_job(
            track="t", run_id="r1", surface="claude",
            model_provider="a", model_name="m", cycles=1,
            memory_access="none", scope="research", guidance="",
            seed_prompt="", env={},
        )
        db.enqueue_steer(job_id, "try a GLM", interrupt=True)
        db.enqueue_steer(job_id, "now try GBM", interrupt=False)

        conn = db._connect()
        rows = conn.execute(
            "SELECT message, status FROM steer_queue WHERE job_id=? ORDER BY id", (job_id,)
        ).fetchall()
        conn.close()
        assert [r["message"] for r in rows] == ["try a GLM", "now try GBM"]
        assert all(r["status"] == "pending" for r in rows)


def test_reattach_marks_running_interrupted():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fresh_db_module(tmp)
        # Need jobs module to see the same DB
        import console.backend.orchestrator.jobs as jobs
        importlib.reload(jobs)

        job_id = db.create_job(
            track="t", run_id="r1", surface="claude",
            model_provider="a", model_name="m", cycles=1,
            memory_access="none", scope="research", guidance="",
            seed_prompt="", env={},
        )
        # Simulate a job that was running with a dead PID
        db.update_job(job_id, status="running", pid=999999)

        jobs.reattach_running_jobs()

        assert db.get_job(job_id)["status"] == "interrupted"
        events = db.get_events(job_id)
        assert any("interrupted" in (e["payload_json"] or "") for e in events)


def test_job_environment_binds_agent_to_assigned_run():
    import console.backend.orchestrator.jobs as jobs

    env = jobs._build_env(
        {
            "scope": "research",
            "memory_access": "none",
            "track": "codex",
            "run_id": "20260610T072525Z",
        }
    )

    assert env["AUTORESEARCH_TRACK"] == "codex"
    assert env["AUTORESEARCH_RUN_ID"] == "20260610T072525Z"

    prompt = jobs.SEED_TEMPLATES["codex"].format(
        track="codex",
        run_id="20260610T072525Z",
        cycles=3,
        model_provider="openai",
        model_name="codex-mini-latest",
        guidance="",
    )
    assert "--run-id 20260610T072525Z bootstrap-track" in prompt
    assert "Do not use `--new-run`" in prompt


def test_schema_init_runs_once():
    """Reconnecting many times should not re-run schema each time (perf)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fresh_db_module(tmp)
        # First connect initializes
        db._connect().close()
        assert db._initialized is True
        # Subsequent connects are cheap (no exception, _initialized stays True)
        for _ in range(10):
            db._connect().close()
        assert db._initialized is True


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
