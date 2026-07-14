"""Fixture-tree tests for scripts/migrate_run_naming.py (spec §8, Phase 7).

The rename/reroute logic is exercised on a *copied* fixture tree passed via
``--repo-root``, never on live ``artifacts/`` — the migration script is
repo-root-parameterised precisely so this test can run it end-to-end.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "migrate_run_naming", PROJECT_ROOT / "scripts" / "migrate_run_naming.py"
)
migrate_run_naming = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(migrate_run_naming)


# --------------------------------------------------------------------------- #
# Fixture tree builders
# --------------------------------------------------------------------------- #
DATASET = "porto_seguro"
TARGET = "claim_incidence"
NEW_SLUG = "porto__incid"  # run_slug(porto_seguro, claim_incidence)


def _wj(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_orchestration(repo: Path, oid: str, *, cons_run_id: str) -> None:
    base = repo / "artifacts" / "orchestrations" / oid
    for sub in ("briefs", "prompts", "logs", "reports", "runs", "playoff"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    # child delegation run dir + manifest
    child = base / "runs" / "d01"
    child.mkdir(parents=True, exist_ok=True)
    _wj(child / "run_manifest.json",
        {"delegation_id": "d01", "run_id": "20200101T020000Z", "dataset": DATASET})

    _wj(base / "orchestration.json", {
        "orchestration_id": oid,
        "dataset": DATASET,
        "target_mode": TARGET,
        "created_at": "2020-01-01T00:00:00Z",
        "status": "completed",
        "total_cycle_budget": 5,
        "orchestrator": {"provider": "openai", "model": "gpt-x", "effort": "medium",
                         "identity_verified": False},
        "consolidation": {"track": "codex", "run_id": cons_run_id,
                          "playoff_report": f"artifacts/orchestrations/{oid}/playoff/playoff_report.json"},
        "campaign_report": f"artifacts/orchestrations/{oid}/campaign_report.json",
        "delegations": [{
            "delegation_id": "d01",
            "run_path": f"artifacts/orchestrations/{oid}/runs/d01",
            "report_path": f"artifacts/orchestrations/{oid}/reports/d01.json",
            "log_path": f"artifacts/orchestrations/{oid}/logs/d01.stdout.log",
            "prompt_path": f"artifacts/orchestrations/{oid}/prompts/d01.md",
        }],
        "descriptor": {"label": "old", "kind": "orchestrated"},
    })
    _wj(base / "campaign_report.json", {
        "orchestration_id": oid,
        "framework_computed": {
            "cost": {"cost_usd": 1.23, "cost_estimated": False},
            "delegations": [{"report": f"artifacts/orchestrations/{oid}/reports/d01.json"}],
            "final_champion": {"source": {"orchestration_id": oid}},
        },
    })
    _wj(base / "reports" / "d01.json",
        {"report_path": f"artifacts/orchestrations/{oid}/reports/d01.json"})
    _wj(base / "playoff" / "playoff_report.json",
        {"orchestration_id": oid,
         "path": f"artifacts/orchestrations/{oid}/playoff/playoff_report.json"})
    (base / "prompts" / "d01.md").write_text("prompt\n", encoding="utf-8")
    (base / "logs" / "d01.stdout.log").write_text(
        f"running --orchestration-id {oid} ...\n", encoding="utf-8")
    (base / "CAMPAIGN_REPORT.md").write_text(
        f"# Campaign {oid}\nSee artifacts/orchestrations/{oid}/reports.\n", encoding="utf-8")


def _build_consolidation_run(repo: Path, oid: str, cons_run_id: str) -> None:
    d = repo / "artifacts" / "tracks" / "codex" / "runs" / cons_run_id
    d.mkdir(parents=True, exist_ok=True)
    _wj(d / "run_manifest.json",
        {"run_id": cons_run_id, "orchestration_id": oid, "consolidation": True,
         "dataset": DATASET, "target_mode": TARGET})


def _build_compat_link(repo: Path, oid: str, compat_id: str) -> Path:
    link = repo / "artifacts" / "tracks" / "codex" / "runs" / compat_id
    link.parent.mkdir(parents=True, exist_ok=True)
    target = repo / "artifacts" / "orchestrations" / oid / "runs" / "d01"
    link.symlink_to(target.resolve())
    return link


def _build_scope(repo: Path, oid: str, session: str) -> None:
    d = repo / "artifacts" / "tracks" / ".scope"
    d.mkdir(parents=True, exist_ok=True)
    _wj(d / f"{session}.json",
        {"mode": "orchestrator", "orchestration_id": oid, "source": "auto"})


def _build_solo_run(repo: Path, track: str, run_id: str) -> Path:
    d = repo / "artifacts" / "tracks" / track / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    _wj(d / "run_manifest.json", {
        "run_id": run_id, "dataset": DATASET, "target_mode": TARGET,
        "created_at": "2020-01-02T00:00:00Z",
        "model_identity": {"provider": "openai", "name": "gpt-solo"},
    })
    _wj(repo / "artifacts" / "tracks" / track / "latest_run.json",
        {"track_id": track, "run_id": run_id, "run_dir": str(d)})
    return d


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal but reference-complete fixture tree."""

    oid = "20200101T000000Z"
    cons_run_id = "20200101T010000Z"
    compat_id = "20200101T003000Z"
    _build_orchestration(tmp_path, oid, cons_run_id=cons_run_id)
    _build_consolidation_run(tmp_path, oid, cons_run_id)
    _build_compat_link(tmp_path, oid, compat_id)
    _build_scope(tmp_path, oid, "sess-abc")
    _build_solo_run(tmp_path, "codex", "20200102T000000Z")
    return tmp_path


def _run(repo: Path, **kw):
    return migrate_run_naming.migrate(
        repo, apply=kw.get("apply", False), force=kw.get("force", True),
        skip_flightdeck=kw.get("skip_flightdeck", True), log=lambda *a: None,
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_dry_run_changes_nothing(repo: Path):
    report = _run(repo, apply=False)
    assert report["dry_run"] is True
    assert report["n_migrated"] == 2  # one orchestration + one solo
    # Nothing renamed.
    assert (repo / "artifacts" / "orchestrations" / "20200101T000000Z").is_dir()
    assert not (repo / "artifacts" / "orchestrations" / f"20200101T000000Z__{NEW_SLUG}").exists()
    assert not (repo / "artifacts" / "migration_report.json").exists()


def test_apply_renames_and_reroutes_orchestration(repo: Path):
    _run(repo, apply=True)
    old = repo / "artifacts" / "orchestrations" / "20200101T000000Z"
    new_id = f"20200101T000000Z__{NEW_SLUG}"
    new = repo / "artifacts" / "orchestrations" / new_id

    assert new.is_dir()
    # Old path survives as a symlink alias to the new dir.
    assert old.is_symlink()
    assert old.resolve() == new.resolve()

    # orchestration.json: id + path references rerouted; descriptor refreshed.
    manifest = json.loads((new / "orchestration.json").read_text())
    assert manifest["orchestration_id"] == new_id
    assert manifest["delegations"][0]["run_path"].endswith(f"{new_id}/runs/d01")
    assert manifest["campaign_report"].endswith(f"{new_id}/campaign_report.json")
    assert manifest["consolidation"]["playoff_report"].endswith(f"{new_id}/playoff/playoff_report.json")
    assert manifest["descriptor"]["kind"] == "orchestrated"
    assert manifest["descriptor"]["dataset"] == DATASET

    # campaign_report.json: bare-id fields and path fields both rerouted.
    creport = json.loads((new / "campaign_report.json").read_text())
    assert creport["orchestration_id"] == new_id
    assert creport["framework_computed"]["final_champion"]["source"]["orchestration_id"] == new_id
    assert creport["framework_computed"]["delegations"][0]["report"].endswith(
        f"{new_id}/reports/d01.json")

    # reports + playoff + md + log rerouted.
    assert new_id in (new / "reports" / "d01.json").read_text()
    assert new_id in (new / "playoff" / "playoff_report.json").read_text()
    assert new_id in (new / "CAMPAIGN_REPORT.md").read_text()
    assert "20200101T000000Z ..." not in (new / "logs" / "d01.stdout.log").read_text()


def test_apply_repoints_compat_symlink(repo: Path):
    _run(repo, apply=True)
    new_id = f"20200101T000000Z__{NEW_SLUG}"
    link = repo / "artifacts" / "tracks" / "codex" / "runs" / "20200101T003000Z"
    assert link.is_symlink()
    target = os.readlink(link)
    assert f"/orchestrations/{new_id}/runs/d01" in target
    # And it still resolves to a real directory.
    assert link.resolve().is_dir()


def test_apply_reroutes_consolidation_and_scope(repo: Path):
    _run(repo, apply=True)
    new_id = f"20200101T000000Z__{NEW_SLUG}"
    cons = json.loads(
        (repo / "artifacts" / "tracks" / "codex" / "runs" / "20200101T010000Z"
         / "run_manifest.json").read_text())
    assert cons["orchestration_id"] == new_id
    scope = json.loads(
        (repo / "artifacts" / "tracks" / ".scope" / "sess-abc.json").read_text())
    assert scope["orchestration_id"] == new_id


def test_apply_migrates_solo_run(repo: Path):
    _run(repo, apply=True)
    new_id = f"20200102T000000Z__{NEW_SLUG}"
    old = repo / "artifacts" / "tracks" / "codex" / "runs" / "20200102T000000Z"
    new = repo / "artifacts" / "tracks" / "codex" / "runs" / new_id
    assert new.is_dir()
    assert old.is_symlink() and old.resolve() == new.resolve()

    manifest = json.loads((new / "run_manifest.json").read_text())
    assert manifest["run_id"] == new_id
    assert manifest["descriptor"]["kind"] == "solo"

    latest = json.loads(
        (repo / "artifacts" / "tracks" / "codex" / "latest_run.json").read_text())
    assert latest["run_id"] == new_id


def test_apply_writes_run_index_and_report(repo: Path):
    _run(repo, apply=True)
    index = json.loads((repo / "artifacts" / "RUN_INDEX.json").read_text())
    ids = {e["id"]: e for e in index["runs"]}
    orch_new = f"20200101T000000Z__{NEW_SLUG}"
    solo_new = f"20200102T000000Z__{NEW_SLUG}"
    assert orch_new in ids and ids[orch_new]["kind"] == "orchestrated"
    assert ids[orch_new]["cost_usd"] == 1.23
    assert solo_new in ids and ids[solo_new]["kind"] == "solo"
    # No stale old-id entries.
    assert "20200101T000000Z" not in ids
    assert "20200102T000000Z" not in ids

    report = json.loads((repo / "artifacts" / "migration_report.json").read_text())
    assert report["dry_run"] is False
    assert {m["new_id"] for m in report["mapping"]} == {orch_new, solo_new}


def test_idempotent_second_run_is_noop(repo: Path):
    _run(repo, apply=True)
    # Discovery now finds only the descriptive dirs (symlink aliases are skipped),
    # which are already migrated → nothing to do.
    second = _run(repo, apply=True)
    assert second["n_migrated"] == 0
