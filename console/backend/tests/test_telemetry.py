"""Normalization and persistence tests for experimental-run telemetry."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for path in (str(_REPO), str(_REPO / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from console.backend.orchestrator import telemetry


def _setup_db(tmp: str):
    os.environ["AUTORESEARCH_CONSOLE_DB"] = str(Path(tmp) / "telemetry.db")
    import console.backend.config as cfg
    importlib.reload(cfg)
    import console.backend.orchestrator.db as db
    importlib.reload(db)
    return db


def test_anthropic_cache_tokens_are_separate_from_input_tokens():
    usage = telemetry.normalize_usage({
        "input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 300,
        "output_tokens": 40,
    })
    assert usage["input_tokens"] == 420
    assert usage["uncached_input_tokens"] == 100
    assert usage["cached_input_tokens"] == 300
    assert usage["cache_creation_input_tokens"] == 20
    assert usage["total_tokens"] == 460


def test_codex_cache_tokens_are_included_in_input_tokens():
    usage = telemetry.normalize_usage({
        "input_tokens": 1000,
        "cached_input_tokens": 700,
        "output_tokens": 50,
        "output_tokens_details": {"reasoning_tokens": 20},
    })
    assert usage["input_tokens"] == 1000
    assert usage["uncached_input_tokens"] == 300
    assert usage["reasoning_tokens"] == 20
    assert usage["total_tokens"] == 1050


def test_opencode_nested_tokens_and_cost_are_normalized():
    usage = telemetry.normalize_usage(
        {
            "input": 80,
            "output": 15,
            "reasoning": 5,
            "cache": {"read": 30, "write": 10},
        },
        raw_result={"cost": 0.012},
    )
    assert usage["input_tokens"] == 80
    assert usage["cached_input_tokens"] == 30
    assert usage["cache_creation_input_tokens"] == 10
    assert usage["provider_cost_usd"] == 0.012


def test_tool_lifecycle_and_summary_are_persisted():
    with tempfile.TemporaryDirectory() as tmp:
        db = _setup_db(tmp)
        job_id = db.create_job(
            track="t", run_id="r", surface="codex", model_provider="openai",
            model_name="codex", cycles=1, memory_access="none", scope="research",
            guidance="", seed_prompt="", env={},
        )
        turn_id = db.start_telemetry_turn(
            job_id, surface="codex", session_id="thread-1",
            model="gpt-5", effort="medium",
        )
        started = telemetry.normalize_tool_payload({
            "provider_call_id": "call-1", "name": "shell",
            "input": {"command": "false"}, "status": "started",
        })
        db.record_telemetry_tool_call(
            job_id, turn_id, normalized=started,
            raw_payload={"provider_call_id": "call-1", "name": "shell"},
        )
        finished = telemetry.normalize_tool_payload({
            "provider_call_id": "call-1", "name": "shell",
            "status": "failed", "exit_code": 1, "output": "failed",
        })
        db.finish_telemetry_tool_call(
            job_id, turn_id, normalized=finished,
            raw_payload={"provider_call_id": "call-1", "exit_code": 1},
        )
        usage = telemetry.normalize_usage({
            "input_tokens": 100, "cached_input_tokens": 60,
            "output_tokens": 20,
        })
        db.finish_telemetry_turn(
            turn_id, session_id="thread-1", duration_ms=200,
            is_error=False, output_chars=50, usage=usage,
            raw_usage={"input_tokens": 100}, raw_result={},
        )
        report = db.get_job_telemetry(job_id)
        assert report["summary"]["turn_count"] == 1
        assert report["summary"]["cache_hit_ratio"] == 0.6
        assert report["summary"]["tool_call_count"] == 1
        assert report["summary"]["tool_failure_count"] == 1
        assert report["summary"]["tool_duration_ms"] is not None


def test_repair_signals_are_extracted():
    signals = telemetry.extract_signals({
        "output": "ExperimentNeedsRepair: preflight failed on attempt 2 because schema mismatch"
    })
    types = {signal["signal_type"] for signal in signals}
    assert "repair_requested" in types
    assert "preflight_failed" in types
    assert all(signal["attempt"] == 2 for signal in signals)


if __name__ == "__main__":
    import traceback
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception:
            print(f"✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
