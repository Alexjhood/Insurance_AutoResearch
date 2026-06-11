"""Desktop transcript import and run-scoped telemetry tests."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import threading
import time
from pathlib import Path

from autoresearch.experiment_registry.experiments import record_experiment
from autoresearch.telemetry.importer import (
    _should_extract_signals,
    reconcile_model_identity,
    sync_session,
)
from autoresearch.telemetry.store import (
    connect,
    ensure_turn,
    get_run_telemetry,
    record_experiment_checkpoint,
    record_workflow_event,
    update_turn_context,
    upsert_model_call,
    upsert_session,
)
from autoresearch.telemetry.usage_report import write_usage_report


def _write_jsonl(path: Path, records: list[dict], *, partial: str = "") -> None:
    text = "".join(json.dumps(record) + "\n" for record in records) + partial
    path.write_text(text, encoding="utf-8")


def test_telemetry_reconciles_manifest_model_identity(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "track_id": "codex",
                "run_id": "run-1",
                "model_identity": {
                    "provider": "openai",
                    "name": "operator-guess",
                    "version": "",
                    "harness": "codex",
                },
            }
        ),
        encoding="utf-8",
    )
    transcript = tmp_path / "rollout-identity.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-06-07T10:00:00.000Z",
                "type": "turn_context",
                "payload": {"model": "gpt-observed", "effort": "low"},
            }
        ],
    )

    result = sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="identity-session",
        transcript_path=transcript,
    )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert result["model_identity"]["status"] == "verified"
    assert manifest["model_identity"]["name"] == "gpt-observed"
    assert manifest["model_identity"]["provider"] == "openai"
    assert manifest["model_identity_declared"]["name"] == "operator-guess"
    assert manifest["model_identity_source"] == "telemetry"


def test_telemetry_records_conflict_for_multiple_observed_models(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps(
            {
                "track_id": "codex",
                "run_id": "run-1",
                "model_identity": {"provider": "openai", "name": "gpt-a"},
            }
        ),
        encoding="utf-8",
    )
    for session_id, model in (("session-a", "gpt-a"), ("session-b", "gpt-b")):
        transcript = tmp_path / f"{session_id}.jsonl"
        _write_jsonl(
            transcript,
            [
                {
                    "timestamp": "2026-06-07T10:00:00.000Z",
                    "type": "turn_context",
                    "payload": {"model": model},
                }
            ],
        )
        sync_session(
            run_dir=tmp_path,
            surface="codex",
            native_session_id=session_id,
            transcript_path=transcript,
        )

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    result = reconcile_model_identity(tmp_path)
    assert result["status"] == "conflict"
    assert [item["name"] for item in manifest["model_identity_conflict"]["observed"]] == [
        "gpt-a",
        "gpt-b",
    ]


def test_codex_import_is_idempotent_and_links_workflow(tmp_path):
    transcript = tmp_path / "rollout-codex-session.jsonl"
    records = [
        {
            "timestamp": "2026-06-07T10:00:00.000Z",
            "type": "session_meta",
            "payload": {
                "id": "codex-session",
                "cwd": "/repo",
                "cli_version": "1.0",
                "timestamp": "2026-06-07T10:00:00.000Z",
            },
        },
        {
            "timestamp": "2026-06-07T10:00:01.000Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-1"},
        },
        {
            "timestamp": "2026-06-07T10:00:01.010Z",
            "type": "turn_context",
            "payload": {"model": "gpt-test", "effort": "high"},
        },
        {
            "timestamp": "2026-06-07T10:00:02.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call-1",
                "arguments": json.dumps(
                    {"cmd": "autoresearch --track codex run-baseline model.toml"}
                ),
            },
        },
        {
            "timestamp": "2026-06-07T10:00:03.000Z",
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "Process exited with code 0\nOutput:\nexperiment complete",
            },
        },
        {
            "timestamp": "2026-06-07T10:00:03.050Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Experiment complete"}],
            },
        },
        {
            "timestamp": "2026-06-07T10:00:03.100Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 60,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 60,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                },
            },
        },
        {
            "timestamp": "2026-06-07T10:00:04.000Z",
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": "turn-1",
                "duration_ms": 3000,
                "time_to_first_token_ms": 500,
            },
        },
    ]
    _write_jsonl(transcript, records)
    record_workflow_event(
        tmp_path,
        event_key="workflow-1",
        command="run-baseline",
        started_at="2026-06-07T10:00:02.100+00:00",
        completed_at="2026-06-07T10:00:02.900+00:00",
        duration_ms=800,
        status="completed",
        error_type=None,
        entities=[("experiment", "experiment-1", "created")],
    )

    first = sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="codex-session",
        transcript_path=transcript,
    )
    second = sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="codex-session",
        transcript_path=transcript,
    )
    rebuilt = sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="codex-session",
        transcript_path=transcript,
        rebuild=True,
    )
    report = get_run_telemetry(tmp_path)

    assert first["records_read"] == len(records)
    assert second["records_read"] == 0
    assert rebuilt["records_read"] == len(records)
    assert report["summary"]["total_tokens"] == 120
    assert report["summary"]["cached_input_tokens"] == 60
    assert report["summary"]["reasoning_tokens"] == 5
    assert report["summary"]["output_chars"] == len("Experiment complete")
    assert report["summary"]["tool_call_count"] == 1
    assert report["summary"]["import_complete"] is True
    assert report["turns"][0]["model"] == "gpt-test"
    assert report["turns"][0]["effort"] == "high"
    assert report["workflow_events"][0]["direct_tokens"] == 120
    assert report["workflow_events"][0]["entities"][0]["entity_id"] == "experiment-1"
    assert report["sessions"][0]["import_status"] == "completed"

    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="codex-session",
        transcript_path=transcript,
        rebuild=True,
    )
    repeated = get_run_telemetry(tmp_path)
    assert repeated["workflow_events"][0]["direct_tokens"] == 120


def test_codex_requires_authoritative_task_complete_for_complete_import(tmp_path):
    transcript = tmp_path / "rollout-late-completion.jsonl"
    initial = [
        {
            "timestamp": "2026-06-07T10:00:00.000Z",
            "type": "session_meta",
            "payload": {"id": "late-session", "timestamp": "2026-06-07T10:00:00.000Z"},
        },
        {
            "timestamp": "2026-06-07T10:00:01.000Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-late"},
        },
        {
            "timestamp": "2026-06-07T10:00:02.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": 12},
                    "last_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 5,
                        "output_tokens": 2,
                        "total_tokens": 12,
                    },
                },
            },
        },
    ]
    _write_jsonl(transcript, initial)

    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="late-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    pending = get_run_telemetry(tmp_path)
    assert pending["summary"]["import_complete"] is False
    assert pending["turns"][0]["status"] == "active"
    assert pending["sessions"][0]["import_status"] == "active"

    completion = {
        "timestamp": "2026-06-07T10:00:03.000Z",
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "turn_id": "turn-late",
            "duration_ms": 2000,
            "time_to_first_token_ms": 250,
        },
    }
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(completion) + "\n")
    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="late-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    complete = get_run_telemetry(tmp_path)
    assert complete["summary"]["import_complete"] is True
    assert complete["turns"][0]["duration_ms"] == 2000
    assert complete["sessions"][0]["import_status"] == "completed"


def test_experiment_usage_markdown_tracks_incremental_and_cumulative_usage(tmp_path):
    transcript = tmp_path / "rollout-usage.jsonl"
    records = [
        {
            "timestamp": "2026-06-07T10:00:00.000Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-usage"},
        },
        {
            "timestamp": "2026-06-07T10:00:00.500Z",
            "type": "turn_context",
            "payload": {"model": "gpt-usage-test", "effort": "medium"},
        },
        {
            "timestamp": "2026-06-07T10:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": 120},
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 60,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                },
            },
        },
        {
            "timestamp": "2026-06-07T10:00:02.000Z",
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": "turn-usage"},
        },
    ]
    _write_jsonl(transcript, records)
    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="usage-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    record_experiment_checkpoint(
        tmp_path,
        experiment_id="experiment-1",
        experiment_name="baseline",
        status="completed",
        completed_at="2026-06-07T10:00:01.500+00:00",
    )
    path = write_usage_report(tmp_path)

    assert path == tmp_path / "LLM_USAGE.md"
    text = path.read_text(encoding="utf-8")
    assert "| baseline | experiment | completed |" in text
    assert "| gpt-usage-test | medium | 100 | 60 | 40 | 20 | 5 | 120/120 | 60.0% | 1/1 |" in text
    assert "| User breakpoint (codex turn 1) | user_breakpoint | settled |" in text
    assert "Current imported ledger: 100 input" in text


def test_recording_experiment_automatically_updates_usage_markdown(tmp_path):
    registry = tmp_path / "registry.sqlite"
    record_experiment(
        registry,
        experiment_id="automatic-1",
        experiment_name="automatic",
        model_family="test",
        target_strategy="direct_pure_premium",
        preprocessing_summary={},
        claim_cap_threshold=None,
        status="completed",
        parent_experiment_id=None,
        config_snapshot_path=tmp_path / "config.json",
        metrics_path=tmp_path / "metrics.json",
        artifacts={},
    )

    text = (tmp_path / "LLM_USAGE.md").read_text(encoding="utf-8")
    assert "| automatic | experiment | completed |" in text


def test_usage_report_migrates_legacy_experiment_checkpoints(tmp_path):
    con = connect(tmp_path)
    try:
        con.execute(
            """
            INSERT INTO experiment_usage_checkpoints (
                experiment_id, experiment_name, status, completed_at
            ) VALUES (?,?,?,?)
            """,
            (
                "legacy-experiment",
                "legacy",
                "completed",
                "2026-06-07T10:00:00.000Z",
            ),
        )
        con.commit()
    finally:
        con.close()

    write_usage_report(tmp_path)

    text = (tmp_path / "LLM_USAGE.md").read_text(encoding="utf-8")
    assert "| legacy | experiment | completed |" in text


def test_usage_report_attributes_post_experiment_usage_to_user_breakpoint(tmp_path):
    transcript = tmp_path / "rollout-post-experiment.jsonl"
    records = [
        {
            "timestamp": "2026-06-07T10:00:00.000Z",
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": "turn-post"},
        },
        {
            "timestamp": "2026-06-07T10:00:00.100Z",
            "type": "turn_context",
            "payload": {"model": "gpt-post", "effort": "low"},
        },
        {
            "timestamp": "2026-06-07T10:00:01.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": 120},
                    "last_token_usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 60,
                        "output_tokens": 20,
                        "reasoning_output_tokens": 5,
                        "total_tokens": 120,
                    },
                },
            },
        },
        {
            "timestamp": "2026-06-07T10:00:03.000Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": 237},
                    "last_token_usage": {
                        "input_tokens": 110,
                        "cached_input_tokens": 90,
                        "output_tokens": 7,
                        "reasoning_output_tokens": 3,
                        "total_tokens": 117,
                    },
                },
            },
        },
        {
            "timestamp": "2026-06-07T10:00:04.000Z",
            "type": "event_msg",
            "payload": {"type": "task_complete", "turn_id": "turn-post"},
        },
    ]
    _write_jsonl(transcript, records)
    record_experiment_checkpoint(
        tmp_path,
        experiment_id="experiment-post",
        experiment_name="challenger",
        status="completed",
        completed_at="2026-06-07T10:00:01.500+00:00",
    )
    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="post-session",
        transcript_path=transcript,
        finalize_turn=True,
    )

    text = (tmp_path / "LLM_USAGE.md").read_text(encoding="utf-8")
    assert "| challenger | experiment | completed |" in text
    assert "| gpt-post | low | 100 | 60 | 40 | 20 | 5 | 120/120 |" in text
    assert "| User breakpoint (codex turn 1) | user_breakpoint | settled |" in text
    assert "| gpt-post | low | 110 | 90 | 20 | 7 | 3 | 117/237 |" in text
    assert "Current imported ledger: 210 input" in text
    assert "27 output, 8 reasoning, 237 total tokens" in text


def test_finalized_continuations_create_one_breakpoint_per_turn(tmp_path):
    transcript = tmp_path / "rollout-continuations.jsonl"
    records: list[dict] = []
    for index in range(1, 4):
        base = index * 10
        records.extend(
            [
                {
                    "timestamp": f"2026-06-07T10:00:{base:02d}.000Z",
                    "type": "event_msg",
                    "payload": {"type": "task_started", "turn_id": f"turn-{index}"},
                },
                {
                    "timestamp": f"2026-06-07T10:00:{base:02d}.100Z",
                    "type": "turn_context",
                    "payload": {"model": "gpt-continuation", "effort": "low"},
                },
                {
                    "timestamp": f"2026-06-07T10:00:{base + 1:02d}.000Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {"total_tokens": index * 12},
                            "last_token_usage": {
                                "input_tokens": 10,
                                "cached_input_tokens": 5,
                                "output_tokens": 2,
                                "reasoning_output_tokens": 1,
                                "total_tokens": 12,
                            },
                        },
                    },
                },
                {
                    "timestamp": f"2026-06-07T10:00:{base + 2:02d}.000Z",
                    "type": "event_msg",
                    "payload": {"type": "task_complete", "turn_id": f"turn-{index}"},
                },
            ]
        )
        _write_jsonl(transcript, records)
        sync_session(
            run_dir=tmp_path,
            surface="codex",
            native_session_id="continuation-session",
            transcript_path=transcript,
            finalize_turn=True,
        )

    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="continuation-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    with sqlite3.connect(tmp_path / "telemetry.sqlite") as con:
        breakpoint_count = con.execute(
            """
            SELECT COUNT(*) FROM llm_usage_checkpoints
            WHERE checkpoint_type='user_breakpoint'
            """
        ).fetchone()[0]

    text = (tmp_path / "LLM_USAGE.md").read_text(encoding="utf-8")
    assert breakpoint_count == 3
    assert text.count("| user_breakpoint | settled |") == 3
    assert "36 total tokens" in text


def test_usage_report_falls_back_to_turn_model_for_historical_calls(tmp_path):
    con = connect(tmp_path)
    try:
        session_key = upsert_session(
            con,
            surface="codex",
            native_session_id="historical-session",
            transcript_path=tmp_path / "historical.jsonl",
            metadata={"model": "session-model"},
        )
        turn_key = ensure_turn(
            con,
            session_key_value=session_key,
            native_turn_id="historical-turn",
            started_at="2026-06-07T10:00:00.000Z",
        )
        update_turn_context(con, turn_key, model="turn-model", effort="high")
        upsert_model_call(
            con,
            {
                "call_key": "historical-call",
                "session_key": session_key,
                "turn_key": turn_key,
                "occurred_at": "2026-06-07T10:00:01.000Z",
                "input_tokens": 8,
                "output_tokens": 2,
                "reasoning_tokens": 1,
                "total_tokens": 10,
            },
        )
        con.commit()
    finally:
        con.close()
    record_experiment_checkpoint(
        tmp_path,
        experiment_id="historical-experiment",
        experiment_name="historical",
        status="completed",
        completed_at="2026-06-07T10:00:02.000Z",
    )
    write_usage_report(tmp_path)

    text = (tmp_path / "LLM_USAGE.md").read_text(encoding="utf-8")
    assert "| turn-model | high |" in text


def test_usage_report_shows_unassigned_live_overhead(tmp_path):
    transcript = tmp_path / "rollout-live.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-06-07T10:00:00.000Z",
                "type": "event_msg",
                "payload": {"type": "task_started", "turn_id": "live-turn"},
            },
            {
                "timestamp": "2026-06-07T10:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {"total_tokens": 15},
                        "last_token_usage": {
                            "input_tokens": 12,
                            "cached_input_tokens": 4,
                            "output_tokens": 3,
                            "reasoning_output_tokens": 2,
                            "total_tokens": 15,
                        },
                    },
                },
            },
        ],
    )
    sync_session(
        run_dir=tmp_path,
        surface="codex",
        native_session_id="live-session",
        transcript_path=transcript,
        finalize_turn=False,
    )

    text = (tmp_path / "LLM_USAGE.md").read_text(encoding="utf-8")
    assert "| Between-cycle / wrap-up overhead | overhead | current import |" in text
    assert "Current imported ledger: 12 input" in text
    assert "3 output, 2 reasoning, 15 total tokens" in text


def test_deferred_codex_settle_waits_for_late_task_complete(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "import_agent_telemetry.py"
    spec = importlib.util.spec_from_file_location("import_agent_telemetry_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    transcript = tmp_path / "late.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-06-07T10:00:00.000Z",
                "type": "event_msg",
                "payload": {"type": "task_started", "turn_id": "late-turn"},
            }
        ],
    )

    def append_completion() -> None:
        time.sleep(0.08)
        with transcript.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "timestamp": "2026-06-07T10:00:01.000Z",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "late-turn"},
                    }
                )
                + "\n"
            )

    worker = threading.Thread(target=append_completion)
    worker.start()
    module._wait_for_transcript_settle(
        transcript,
        surface="codex",
        timeout=1.0,
        poll_interval=0.02,
    )
    worker.join()

    assert module._codex_latest_turn_complete(transcript)


def test_claude_deduplicates_request_usage_and_handles_partial_line(tmp_path):
    transcript = tmp_path / "claude-session.jsonl"
    prompt = {
        "type": "user",
        "sessionId": "claude-session",
        "uuid": "prompt-1",
        "timestamp": "2026-06-07T11:00:00.000Z",
        "message": {"content": "Run one experiment"},
    }
    usage = {
        "input_tokens": 1,
        "cache_read_input_tokens": 10,
        "cache_creation_input_tokens": 2,
        "output_tokens": 3,
    }
    assistant_thinking = {
        "type": "assistant",
        "sessionId": "claude-session",
        "requestId": "req-1",
        "timestamp": "2026-06-07T11:00:01.000Z",
        "message": {
            "model": "claude-test",
            "usage": usage,
            "content": [{"type": "thinking", "thinking": "private"}],
        },
    }
    assistant_tool = {
        "type": "assistant",
        "sessionId": "claude-session",
        "requestId": "req-1",
        "timestamp": "2026-06-07T11:00:01.000Z",
        "message": {
            "model": "claude-test",
            "usage": usage,
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "autoresearch --track claude run-session-cycle"},
                }
            ],
        },
    }
    tool_result = {
        "type": "user",
        "sessionId": "claude-session",
        "timestamp": "2026-06-07T11:00:02.000Z",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "is_error": False,
                    "content": "ok",
                }
            ]
        },
        "toolUseResult": {"stdout": "ok", "stderr": ""},
    }
    later = {
        "type": "assistant",
        "sessionId": "claude-session",
        "requestId": "req-2",
        "timestamp": "2026-06-07T11:00:03.000Z",
        "message": {
            "model": "claude-test",
            "usage": {
                "input_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "output_tokens": 7,
            },
            "content": [{"type": "text", "text": "Finished"}],
        },
    }
    partial = json.dumps(later)
    _write_jsonl(
        transcript,
        [prompt, assistant_thinking, assistant_tool, tool_result],
        partial=partial,
    )

    first = sync_session(
        run_dir=tmp_path,
        surface="claude",
        native_session_id="claude-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    second = sync_session(
        run_dir=tmp_path,
        surface="claude",
        native_session_id="claude-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    report = get_run_telemetry(tmp_path)

    assert first["records_read"] == 4
    assert second["records_read"] == 1
    assert report["summary"]["model_call_count"] == 2
    assert report["summary"]["input_tokens"] == 18
    assert report["summary"]["cached_input_tokens"] == 10
    assert report["summary"]["cache_creation_input_tokens"] == 2
    assert report["summary"]["output_tokens"] == 10
    assert report["summary"]["total_tokens"] == 28
    assert report["summary"]["tool_call_count"] == 1
    assert report["summary"]["import_complete"] is True


def test_claude_parent_import_includes_subagent_transcripts(tmp_path):
    transcript = tmp_path / "parent-session.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "sessionId": "parent-session",
                "uuid": "parent-prompt",
                "timestamp": "2026-06-07T12:00:00.000Z",
                "message": {"content": "delegate"},
            }
        ],
    )
    subagents = tmp_path / "parent-session" / "subagents"
    subagents.mkdir(parents=True)
    _write_jsonl(
        subagents / "agent-child.jsonl",
        [
            {
                "type": "user",
                "sessionId": "parent-session",
                "agentId": "child",
                "uuid": "child-prompt",
                "timestamp": "2026-06-07T12:00:01.000Z",
                "message": {"content": "subtask"},
            },
            {
                "type": "assistant",
                "sessionId": "parent-session",
                "agentId": "child",
                "requestId": "child-request",
                "timestamp": "2026-06-07T12:00:02.000Z",
                "message": {
                    "model": "claude-haiku-test",
                    "usage": {
                        "input_tokens": 5,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 10,
                        "output_tokens": 4,
                    },
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        ],
    )

    result = sync_session(
        run_dir=tmp_path / "run",
        surface="claude",
        native_session_id="parent-session",
        transcript_path=transcript,
        finalize_turn=True,
    )
    report = get_run_telemetry(tmp_path / "run")

    assert result["subagent_transcripts"] == 1
    assert report["summary"]["session_count"] == 2
    assert report["summary"]["total_tokens"] == 39
    assert any("subagent:child" in session["native_session_id"] for session in report["sessions"])


def test_empty_run_report_has_stable_shape(tmp_path):
    report = get_run_telemetry(tmp_path)

    assert report["available"] is False
    assert report["summary"]["total_tokens"] == 0
    assert report["workflow_events"] == []


def test_signal_extraction_ignores_documentation_reads():
    assert not _should_extract_signals(
        {"detail": "sed", "name": "exec_command"},
        True,
    )
    assert _should_extract_signals(
        {"detail": "autoresearch run-session-cycles", "name": "exec_command"},
        True,
    )
    assert _should_extract_signals(
        {"detail": "pytest", "name": "exec_command"},
        False,
    )
