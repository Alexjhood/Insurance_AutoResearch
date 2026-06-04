from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_GUARD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "codex_research_log_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("codex_research_log_guard", _GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


guard = _load_guard()


def _write_bound_run(tmp_path: Path, *, current_cycle: int = 2) -> tuple[str, Path]:
    session_id = "codex-session"
    scope_dir = tmp_path / "artifacts" / "tracks" / ".scope"
    run_dir = tmp_path / "artifacts" / "tracks" / "codex" / "runs" / "RUN_A"
    session_dir = run_dir / "sessions" / "session_main"
    scope_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (scope_dir / f"{session_id}.json").write_text(
        json.dumps(
            {
                "mode": "research",
                "track": "codex",
                "run_id": "RUN_A",
                "run_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "sessions" / "latest_session_id.txt").write_text("session_main", encoding="utf-8")
    (session_dir / "state.json").write_text(json.dumps({"current_cycle": current_cycle}), encoding="utf-8")
    return session_id, run_dir


def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "TRACKS_DIR", tmp_path / "artifacts" / "tracks")
    monkeypatch.setattr(guard, "SCOPE_DIR", tmp_path / "artifacts" / "tracks" / ".scope")
    monkeypatch.setattr(guard, "LOG_PATH", tmp_path / "artifacts" / "tracks" / ".scope" / "research-log-guard.log")


def _payload(session_id: str) -> dict:
    return {"hook_event_name": "Stop", "session_id": session_id}


def test_missing_research_log_blocks_bound_codex_stop(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)
    session_id, run_dir = _write_bound_run(tmp_path, current_cycle=1)

    code, message = guard.evaluate(_payload(session_id))

    assert code == 2
    assert str(run_dir / "RESEARCH_LOG.md") in message
    assert "Cycle 1 section" in message


def test_complete_research_log_allows_bound_codex_stop(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("AUTORESEARCH_SCOPE", raising=False)
    session_id, run_dir = _write_bound_run(tmp_path, current_cycle=2)
    (run_dir / "RESEARCH_LOG.md").write_text(
        """
# Research Log

## Cycle 1 - 2026-06-04
**Hypothesis**: One.
**Changes**: A.
**Outcome**: inconclusive
**Metrics**: gini=0.1
**Interpretation**: Needs work.
**Next**: Continue.

## Cycle 2 - 2026-06-04
**Hypothesis**: Two.
**Changes**: B.
**Outcome**: promoted
**Metrics**: gini=0.2
**Interpretation**: Better.
**Next**: Compare.
""".lstrip(),
        encoding="utf-8",
    )

    code, message = guard.evaluate(_payload(session_id))

    assert code == 0
    assert message == ""


def test_analyst_scope_exempts_research_log_guard(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTORESEARCH_SCOPE", "analyst")
    session_id, _ = _write_bound_run(tmp_path, current_cycle=3)

    assert guard.evaluate(_payload(session_id)) == (0, "")
