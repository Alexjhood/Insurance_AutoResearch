"""End-to-end build assertions against the synthetic mini-fixture."""

from __future__ import annotations

import json

from flightdeck.etl.build import snapshots_dir
from flightdeck.etl.tests.fixtures.make_mini_fixture import EXP_A, ORCH_ID


def test_emits_expected_files(mini_repo):
    out = snapshots_dir(mini_repo) / ORCH_ID
    assert (out / "snapshot.json").exists()
    assert (snapshots_dir(mini_repo) / "index.json").exists()
    # A telemetry file per delegation that had a telemetry DB (d01, d02, d03).
    for did in ("d01", "d02", "d03"):
        assert (out / f"telemetry_{did}.json").exists()
    assert (out / "files" / "CAMPAIGN_REPORT.md").exists()


def test_index_headline_stats(mini_index):
    entry = mini_index["orchestrations"][0]
    assert entry["orch_id"] == ORCH_ID
    assert entry["dataset"] == "mini_motor"
    assert entry["target_mode"] == "frequency"
    assert entry["orchestrator_model"] == "openai/gpt-mini"
    assert entry["n_delegations"] == 3
    assert entry["cycles_committed"] == 6
    # d01 used 3 + d02 used 1 + d03 used 0.
    assert entry["cycles_used"] == 4
    assert entry["cycles_forfeited"] == 1
    assert entry["takeover_count"] == 1
    # final_gini follows the playoff final lineage (d01 finalist gini = 0.30).
    assert entry["final_gini"] == 0.30
    assert entry["baseline_gini"] == 0.001


def test_delegation_join(mini_snapshot):
    d01 = next(d for d in mini_snapshot["delegations"] if d["delegation_id"] == "d01")
    assert d01["backend"] == "codex-gpt-mini"
    assert d01["model_identity"] == {"provider": "openai", "name": "gpt-mini",
                                     "harness": "codex"}
    assert d01["champion"]["experiment_id"] == EXP_A
    assert d01["brief"]["name"] == "frequency_baseline"
    assert d01["brief"]["constraints"]
    assert d01["files"]["prompt"] == "prompts/d01.md"
    assert d01["files"]["research_log"] == "runs/d01/RESEARCH_LOG.md"


def test_seed_champion_brief(mini_snapshot):
    d02 = next(d for d in mini_snapshot["delegations"] if d["delegation_id"] == "d02")
    assert d02["brief"]["seed_champion"] == {"experiment_id": EXP_A, "from_run": "codex/R01"}
    assert d02["taken_over"] is True


def test_experiments_have_decisions_where_compared(mini_snapshot):
    for exp in mini_snapshot["experiments"]:
        if exp["comparison"] is not None:
            assert exp["comparison"]["decision"] is not None


def test_seq_is_chronological_and_dense(mini_snapshot):
    seqs = sorted(e["seq"] for e in mini_snapshot["experiments"])
    assert seqs == list(range(len(seqs)))


def test_champion_timeline_chronological_nonempty(mini_snapshot):
    tl = mini_snapshot["champion_timeline"]
    assert tl
    ats = [e["at"] for e in tl]
    assert ats == sorted(ats)
    # 'retained' (no-op) events are filtered out.
    assert all(e["action"] != "retained" for e in tl)


def test_telemetry_summary_totals(mini_snapshot):
    ts = mini_snapshot["telemetry_summary"]
    tot = ts["totals"]
    # d01 tool_usage input 3000 + d02 5000 + d03 (fallback) 1000.
    assert tot["input"] == 3000 + 5000 + 1000
    assert ts["cache_hit_rate"] is not None
    # tool-mix present with at least the failing d02 exec call flagged.
    d02_fail = [m for m in ts["tool_mix"]
                if m["delegation_id"] == "d02" and m["failures"] > 0]
    assert d02_fail


def test_files_manifest_kinds(mini_snapshot):
    manifest = {f["path"]: f for f in mini_snapshot["files"]}
    assert manifest["CAMPAIGN_REPORT.md"]["kind"] == "markdown"
    assert manifest["reports/d01.json"]["kind"] == "json"
    assert manifest["logs/d01.stdout.log"]["kind"] == "log"
    # sqlite is never copied.
    assert not any(p.endswith(".sqlite") for p in manifest)


def test_telemetry_lazy_file_shape(mini_repo):
    out = snapshots_dir(mini_repo) / ORCH_ID
    dt = json.loads((out / "telemetry_d01.json").read_text())
    assert dt["delegation_id"] == "d01"
    assert dt["model_calls"] and dt["tool_calls"]
    assert dt["checkpoints"]
