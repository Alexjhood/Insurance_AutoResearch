"""Schema consistency: emitted JSON carries exactly the declared fields.

DATA.md: "Missing/unknown values: null, never omitted keys". So every dataclass
field must appear in the serialised output (as null when unknown), and no extra
keys may leak in. This keeps ``schema.py`` and the emitted contract in lockstep.
"""

from __future__ import annotations

import dataclasses
import json

from flightdeck.etl import schema
from flightdeck.etl.schema import (
    Campaign,
    Delegation,
    Experiment,
    IndexEntry,
    Snapshot,
    to_jsonable,
)


def _field_names(dc) -> set[str]:
    return {f.name for f in dataclasses.fields(dc)}


def _assert_keys(obj: dict, dc) -> None:
    assert set(obj.keys()) == _field_names(dc), (
        f"{dc.__name__}: {set(obj.keys()) ^ _field_names(dc)}"
    )


def test_snapshot_top_level_keys(mini_snapshot):
    _assert_keys(mini_snapshot, Snapshot)
    assert mini_snapshot["snapshot_schema_version"] == schema.SNAPSHOT_SCHEMA_VERSION


def test_campaign_keys(mini_snapshot):
    _assert_keys(mini_snapshot["campaign"], Campaign)


def test_every_delegation_has_full_key_set(mini_snapshot):
    assert mini_snapshot["delegations"]
    for deleg in mini_snapshot["delegations"]:
        _assert_keys(deleg, Delegation)


def test_every_experiment_has_full_key_set(mini_snapshot):
    assert mini_snapshot["experiments"]
    for exp in mini_snapshot["experiments"]:
        _assert_keys(exp, Experiment)


def test_index_entry_keys(mini_index):
    assert mini_index["orchestrations"]
    for entry in mini_index["orchestrations"]:
        _assert_keys(entry, IndexEntry)


def test_to_jsonable_round_trips():
    entry = IndexEntry(
        orch_id="x", alias=None, dataset="d", target_mode="frequency", status="completed",
        created_at="2026-01-01T00:00:00Z", ended_at=None, orchestrator_model="openai/gpt",
        backends=["codex"], n_delegations=1, cycles_committed=1, cycles_used=1,
        cycles_forfeited=0, final_gini=0.3, baseline_gini=0.0,
        total_tokens=schema.TokenTotals(1, 1, 1, 1), cache_hit_rate=1.0,
        wall_clock_minutes=1.0, distress_count=0, takeover_count=0, champion_spark=[0.3],
    )
    blob = json.dumps(to_jsonable(entry))
    restored = json.loads(blob)
    _assert_keys(restored, IndexEntry)
    assert restored["total_tokens"] == {"input": 1, "cached_input": 1, "output": 1,
                                        "reasoning": 1}


def test_no_omitted_keys_are_none_not_missing(mini_snapshot):
    # A crashed delegation (d03) still carries champion=None, not a missing key.
    d03 = next(d for d in mini_snapshot["delegations"] if d["delegation_id"] == "d03")
    assert "champion" in d03 and d03["champion"] is None
    assert "agent_summary" in d03
