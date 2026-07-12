"""Defensive-degradation (Phase-1 acceptance #4) + aliases + --all coverage."""

from __future__ import annotations

import json

from flightdeck.etl.build import (
    build,
    build_orchestration,
    discover_orchestrations,
    orchestrations_dir,
    snapshots_dir,
)
from flightdeck.etl.tests.conftest import (
    REAL_FIXTURE_ID,
    skeleton_copy_orchestration,
)
from flightdeck.etl.tests.fixtures.make_mini_fixture import ORCH_ID


def test_mini_missing_report_still_builds_with_warnings(mini_repo_fresh):
    """Delete a delegation report on the mini-fixture → build succeeds + warns."""
    root = mini_repo_fresh
    (orchestrations_dir(root) / ORCH_ID / "reports" / "d02.json").unlink()
    entry, skipped = build_orchestration(ORCH_ID, root, log=lambda *_: None)
    assert entry is not None and not skipped
    snap = json.loads((snapshots_dir(root) / ORCH_ID / "snapshot.json").read_text())
    assert any("d02" in w for w in snap["build"]["warnings"])
    # Delegation still present, champion degraded to null.
    d02 = next(d for d in snap["delegations"] if d["delegation_id"] == "d02")
    assert d02["champion"] is None


def test_real_fixture_missing_d03_report(tmp_path, real_repo_root):
    """Acceptance #4: a copy of the real fixture with reports/d03.json deleted
    still builds, with warnings."""
    # Discover the consolidation (track, run_id) from the real ledger.
    ledger = json.loads(
        (orchestrations_dir(real_repo_root) / REAL_FIXTURE_ID / "orchestration.json").read_text()
    )
    cons = ledger.get("consolidation") or {}
    skeleton_copy_orchestration(
        real_repo_root, tmp_path, REAL_FIXTURE_ID,
        (cons.get("track"), cons.get("run_id")) if cons.get("run_id") else None,
    )
    (orchestrations_dir(tmp_path) / REAL_FIXTURE_ID / "reports" / "d03.json").unlink()

    entry, skipped = build_orchestration(REAL_FIXTURE_ID, tmp_path, log=lambda *_: None)
    assert entry is not None and not skipped
    snap = json.loads(
        (snapshots_dir(tmp_path) / REAL_FIXTURE_ID / "snapshot.json").read_text()
    )
    assert any("d03" in w for w in snap["build"]["warnings"])
    assert len(snap["delegations"]) == 5  # still 5, d03 degraded


def test_aliases_preserved(mini_repo_fresh):
    root = mini_repo_fresh
    sd = snapshots_dir(root)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "aliases.json").write_text(json.dumps({ORCH_ID: "My Mini Campaign"}))
    build(root, [ORCH_ID], log=lambda *_: None)
    index = json.loads((sd / "index.json").read_text())
    assert index["orchestrations"][0]["alias"] == "My Mini Campaign"
    # ETL never rewrites aliases.json.
    assert json.loads((sd / "aliases.json").read_text()) == {ORCH_ID: "My Mini Campaign"}


def test_build_all_on_mini(mini_repo_fresh):
    root = mini_repo_fresh
    found = discover_orchestrations(root)
    assert found == [ORCH_ID]
    build(root, found, log=lambda *_: None)
    assert (snapshots_dir(root) / "index.json").exists()
