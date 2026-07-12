"""Parity checks against the real reference orchestration (acceptance #2).

Skipped automatically when ``artifacts/orchestrations/20260711T164959Z`` is
absent, so the suite still passes in a clean checkout.
"""

from __future__ import annotations

import json

import pytest

from flightdeck.etl.build import build_orchestration, snapshots_dir
from flightdeck.etl.tests.conftest import REAL_FIXTURE_ID


@pytest.fixture(scope="module")
def real_snapshot(real_repo_root):
    build_orchestration(REAL_FIXTURE_ID, real_repo_root, force=True, log=lambda *_: None)
    path = snapshots_dir(real_repo_root) / REAL_FIXTURE_ID / "snapshot.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def real_index(real_repo_root, real_snapshot):
    return json.loads((snapshots_dir(real_repo_root) / "index.json").read_text())


def _entry(index):
    return next(e for e in index["orchestrations"] if e["orch_id"] == REAL_FIXTURE_ID)


def test_final_gini_is_playoff_champion(real_index):
    assert _entry(real_index)["final_gini"] == 0.3373153637090053


def test_d01_input_tokens(real_snapshot):
    d01 = next(d for d in real_snapshot["delegations"] if d["delegation_id"] == "d01")
    assert d01["cost"]["tokens"]["input"] == 7_801_150


def test_d02_distress_active(real_snapshot):
    d02 = next(d for d in real_snapshot["delegations"] if d["delegation_id"] == "d02")
    assert "all_rejected" in d02["distress"]["active"]
    assert "cycles_forfeited" not in d02["distress"]["active"]


def test_five_delegations(real_snapshot):
    assert len(real_snapshot["delegations"]) == 5


def test_at_least_one_takeover_note(real_snapshot):
    assert any(n["kind"] == "takeover" for n in real_snapshot["notes"])


def test_champion_timeline_nonempty_and_chronological(real_snapshot):
    tl = real_snapshot["champion_timeline"]
    assert tl
    ats = [e["at"] for e in tl]
    assert ats == sorted(ats)


def test_every_compared_experiment_has_decision(real_snapshot):
    for exp in real_snapshot["experiments"]:
        if exp["comparison"] is not None:
            assert exp["comparison"]["decision"] is not None


def test_seed_experiments_flagged(real_snapshot):
    seeds = [e for e in real_snapshot["experiments"] if e["is_seed"]]
    assert seeds
    for e in seeds:
        assert e["name"].startswith(("orchestration_delegation_seed", "orchestration_seed"))


def test_emits_telemetry_files(real_repo_root, real_snapshot):
    out = snapshots_dir(real_repo_root) / REAL_FIXTURE_ID
    for did in ("d01", "d02", "d03", "d04", "d05"):
        assert (out / f"telemetry_{did}.json").exists()


def test_playoff_includes_pairing_gate_evidence(real_snapshot):
    playoff = real_snapshot["playoff"]
    assert playoff["status"] == "completed"
    assert playoff["pairings"]
    first = playoff["pairings"][0]
    assert first["comparison_id"]
    assert "challenger_win_rate" in first["gates"]
    assert first["guardrail_passed"] is True
    assert playoff["final"]["source_experiment_id"]


def test_normalized_orchestrator_and_model_usage(real_snapshot):
    assert real_snapshot["campaign"]["orchestrator"]["model"]
    rows = real_snapshot["telemetry_summary"]["usage_by_model"]
    assert any(row["role"] == "orchestrator" and row["unmeasured"] for row in rows)
