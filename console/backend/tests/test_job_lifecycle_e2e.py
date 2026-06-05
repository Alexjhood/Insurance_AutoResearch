"""
End-to-end job lifecycle test using a stub adapter and stub runner.

Exercises launch_job → drain → record events → steer → stop, WITHOUT needing
an authenticated agent CLI or real git worktrees. This is the test that would
have caught the positional-INSERT bug in create_job.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for p in (str(_REPO), str(_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from console.backend.orchestrator.adapters.base import AgentEvent, EventType, SessionHandle, SteerResult


class StubAdapter:
    """Emits a scripted sequence of events, supports steer + stop."""

    def __init__(self) -> None:
        import queue
        self._q: "queue.Queue" = queue.Queue()
        self.steers: list[str] = []
        self.stopped = False

    def launch(self, *, cwd, env, seed_prompt):
        self._q.put(AgentEvent(EventType.SYSTEM, {"msg": "init", "session_id": "stub-sess-1"}))
        self._q.put(AgentEvent(EventType.TOKEN, {"text": "Bootstrapping run…"}))
        self._q.put(AgentEvent(EventType.TOOL_USE, {"name": "Bash", "input": {"command": "autoresearch bootstrap-track"}}))
        self._q.put(AgentEvent(EventType.TURN_END, {"session_id": "stub-sess-1", "is_error": False}))
        self._q.put(AgentEvent(EventType.AGENT_EXIT, {"returncode": 0, "session_id": "stub-sess-1"}))
        self._q.put(None)
        return SessionHandle(session_id="stub-sess-1", pid=12345)

    def stream(self):
        while True:
            ev = self._q.get()
            if ev is None:
                return
            yield ev

    def steer(self, message, *, interrupt=True):
        self.steers.append(message)
        return SteerResult(mode="interrupt" if interrupt else "queued", ok=True)

    def stop(self):
        self.stopped = True
        self._q.put(None)


def _setup(tmp: str):
    os.environ["AUTORESEARCH_CONSOLE_DB"] = str(Path(tmp) / "orch.db")
    import console.backend.config as cfg
    importlib.reload(cfg)
    import console.backend.orchestrator.db as db
    importlib.reload(db)
    import console.backend.orchestrator.runner as runner
    importlib.reload(runner)
    import console.backend.orchestrator.jobs as jobs
    importlib.reload(jobs)

    # Stub the runner so no real git worktree is created
    class StubRunner:
        def create_worktree(self, track, run_id):
            d = Path(tmp) / f"wt-{track}-{run_id}"
            d.mkdir(parents=True, exist_ok=True)
            return d
        def remove_worktree(self, path):
            pass

    jobs._runner = StubRunner()
    return db, jobs


def test_launch_drain_steer_stop():
    with tempfile.TemporaryDirectory() as tmp:
        db, jobs = _setup(tmp)
        stub = StubAdapter()
        jobs._get_adapter = lambda surface: stub  # type: ignore

        job_id = jobs.launch_job(
            track="claude", surface="claude",
            model_provider="anthropic", model_name="claude-opus-4-8",
            cycles=2, memory_access="none", scope="research",
            guidance="focus on GLMs",
        )
        assert job_id

        # Wait for the drain thread to consume the scripted events
        for _ in range(50):
            job = db.get_job(job_id)
            if job["status"] == "done":
                break
            time.sleep(0.05)

        job = db.get_job(job_id)
        assert job["status"] == "done", f"status={job['status']}"
        # session_id captured from the stream
        assert job["agent_session_id"] == "stub-sess-1"

        events = db.get_events(job_id)
        types = {e["event_type"] for e in events}
        assert "token" in types
        assert "tool_use" in types
        assert "turn_end" in types
        assert "agent_exit" in types

        # Guidance made it into the seed prompt
        assert "focus on GLMs" in job["seed_prompt"]


def test_steer_while_live():
    with tempfile.TemporaryDirectory() as tmp:
        db, jobs = _setup(tmp)

        # An adapter that stays "live" (doesn't auto-exit) so we can steer it
        import queue
        class LiveStub(StubAdapter):
            def launch(self, *, cwd, env, seed_prompt):
                self._q.put(AgentEvent(EventType.SYSTEM, {"msg": "init", "session_id": "live-1"}))
                return SessionHandle(session_id="live-1", pid=222)

        stub = LiveStub()
        jobs._get_adapter = lambda surface: stub  # type: ignore
        job_id = jobs.launch_job(
            track="claude", surface="claude", model_provider="a", model_name="m",
            cycles=1, memory_access="none", scope="research", guidance="",
        )
        time.sleep(0.2)

        result = jobs.steer_job(job_id, "try a GBM next", interrupt=True)
        assert result.ok
        assert result.mode == "interrupt"
        assert "try a GBM next" in stub.steers

        # Stop cleans up
        jobs.stop_job(job_id)
        assert stub.stopped
        assert db.get_job(job_id)["status"] == "stopped"


def test_steer_when_not_live_queues():
    with tempfile.TemporaryDirectory() as tmp:
        db, jobs = _setup(tmp)
        # Create a job but don't launch an adapter (not in _LIVE)
        job_id = db.create_job(
            track="t", run_id="r", surface="claude", model_provider="a",
            model_name="m", cycles=1, memory_access="none", scope="research",
            guidance="", seed_prompt="", env={},
        )
        result = jobs.steer_job(job_id, "do X", interrupt=True)
        assert result.mode == "queued"
        # It should be in the steer_queue
        conn = db._connect()
        n = conn.execute("SELECT COUNT(*) FROM steer_queue WHERE job_id=?", (job_id,)).fetchone()[0]
        conn.close()
        assert n == 1


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
