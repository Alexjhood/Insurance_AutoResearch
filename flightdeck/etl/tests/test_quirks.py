"""Each DATA.md §4 data quirk gets a dedicated test against the mini-fixture."""

from __future__ import annotations

from flightdeck.etl.tests.fixtures.make_mini_fixture import EXP_A


def _by_name(snapshot, name):
    return next(e for e in snapshot["experiments"] if e["name"] == name)


def test_quirk1_undecided_experiment_kept(mini_snapshot):
    """Takeover run: attempted > decided; undecided experiment is kept, not dropped."""
    d02 = next(d for d in mini_snapshot["delegations"] if d["delegation_id"] == "d02")
    assert d02["budget"]["attempted"] > d02["budget"]["decided"]
    exp = _by_name(mini_snapshot, "forfeited_undecided")
    assert exp["comparison"] is None
    assert exp["decision_meta"]["outcome"] is None  # never decided → renders forfeited


def test_quirk2_seed_flagged_and_excluded_from_lift(mini_snapshot):
    seed = _by_name(mini_snapshot,
                    "orchestration_delegation_seed_d02_lgbm_poisson_native_baseline")
    assert seed["is_seed"] is True
    assert seed["lift"] == {"vs_then_champion": None, "vs_baseline": None, "kind": None}


def test_quirk3_dedup_and_replay_link(mini_snapshot):
    """EXP_A appears in d01 and consolidation registries → deduped once (d01)."""
    hits = [e for e in mini_snapshot["experiments"] if e["experiment_id"] == EXP_A]
    assert len(hits) == 1
    assert hits[0]["delegation_id"] == "d01"
    # And the finalist links to its consolidation replay by delegation tag.
    finalists = {f["delegation_id"]: f for f in mini_snapshot["playoff"]["finalists"]}
    assert finalists["d01"]["replay_experiment_id"] is not None
    assert "_d01_" in finalists["d01"]["replay_experiment_id"]
    assert "_d02_" in finalists["d02"]["replay_experiment_id"]


def test_quirk4_mixed_lift_bases_tagged(mini_snapshot):
    first = _by_name(mini_snapshot, "lgbm_poisson_native_baseline")
    later = _by_name(mini_snapshot, "xgb_poisson_ordinal_root")
    assert first["lift"]["kind"] == "baseline_relative"
    assert later["lift"]["kind"] == "incremental"


def test_quirk5_wal_present_reads_ok(mini_repo, mini_snapshot):
    """A telemetry.sqlite-wal sidecar exists for d01 yet the read succeeded."""
    from flightdeck.etl.build import orchestrations_dir
    from flightdeck.etl.tests.fixtures.make_mini_fixture import ORCH_ID

    wal = (orchestrations_dir(mini_repo) / ORCH_ID / "runs" / "d01"
           / "telemetry.sqlite-wal")
    assert wal.exists()
    # Usage for d01 experiments was still populated from telemetry/LLM_USAGE.
    exp = _by_name(mini_snapshot, "lgbm_poisson_native_baseline")
    assert exp["usage"] is not None


def test_quirk6_unreadable_registry_degrades(mini_snapshot):
    """d03 registry is a non-sqlite file → degrade to nulls + warning, no crash."""
    warnings = mini_snapshot["build"]["warnings"]
    assert any("d03" in w and "registry" in w.lower() for w in warnings)
    d03_experiments = [e for e in mini_snapshot["experiments"]
                       if e["delegation_id"] == "d03"]
    assert d03_experiments == []


def test_quirk7_missing_tool_usage_falls_back_to_telemetry(mini_snapshot):
    """d03 ledger has no tool_usage → tokens summed from llm_model_calls."""
    d03 = next(d for d in mini_snapshot["delegations"] if d["delegation_id"] == "d03")
    # 700 + 300 input from the two telemetry model calls.
    assert d03["cost"]["tokens"]["input"] == 1000
    assert d03["cost"]["tokens"]["cached_input"] == 850


def test_usage_dual_source(mini_snapshot):
    """d01 usage comes from LLM_USAGE.md (token-less checkpoints); d02 from the DB."""
    d01_exp = _by_name(mini_snapshot, "lgbm_poisson_native_baseline")
    assert d01_exp["usage"]["tokens"]["input"] == 1000  # from LLM_USAGE.md row
    d02_exp = _by_name(mini_snapshot, "xgb_poisson_ordinal_root")
    assert d02_exp["usage"]["tokens"]["input"] == 1200  # from checkpoint token cols
