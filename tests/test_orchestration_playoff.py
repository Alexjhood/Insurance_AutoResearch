"""Phase 4 orchestration playoff and shared replay fixtures.

The playoff milestone uses deterministic registry fixtures and comparison
stubs. No sub-agent process or external model is launched.
"""

from __future__ import annotations

import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.orchestration import manifest as manifest_mod
from autoresearch.orchestration import playoff as playoff_mod
from autoresearch.orchestration import spawner as spawner_mod
from autoresearch.orchestration.backends import Backend
from autoresearch.orchestration.brief import Brief, SeedChampion
from autoresearch.orchestration.manifest import (
    Delegation,
    add_delegation,
    create_orchestration,
    load_orchestration,
    save_orchestration,
)
from autoresearch.orchestration.playoff import ReplaySource
from autoresearch.utils.io import read_json, write_json


@pytest.fixture
def orchestrations_root(tmp_path, monkeypatch):
    root = tmp_path / "orchestrations"
    root.mkdir()
    monkeypatch.setattr(manifest_mod, "ORCHESTRATIONS_DIR", root)
    return root


def _record_experiment(
    registry: Path,
    run_dir: Path,
    experiment_id: str,
    *,
    family: str,
    snapshot: dict | None = None,
    artifacts: dict[str, Path] | None = None,
) -> None:
    from autoresearch.experiment_registry.experiments import record_experiment

    snapshot_path = run_dir / f"{experiment_id}_snapshot.json"
    metrics_path = run_dir / f"{experiment_id}_metrics.json"
    if snapshot is not None:
        write_json(snapshot_path, snapshot)
    write_json(metrics_path, {})
    record_experiment(
        registry,
        experiment_id=experiment_id,
        experiment_name=experiment_id,
        model_family=family,
        target_strategy="direct_pure_premium",
        target_mode="burning_cost",
        preprocessing_summary={},
        claim_cap_threshold=None,
        status="completed",
        parent_experiment_id=None,
        config_snapshot_path=snapshot_path,
        metrics_path=metrics_path,
        artifacts=artifacts or {},
    )


def _set_champion(registry: Path, experiment_id: str) -> None:
    from autoresearch.experiment_registry.champions import set_official_champion

    set_official_champion(
        registry,
        champion_id=experiment_id,
        branch_id="main",
        reason="fixture",
        action="fixture",
    )


def test_shared_replay_handles_recipe_and_copies_script_plus_proposal(
    tmp_path, monkeypatch
):
    from autoresearch.experiment_registry.proposals import (
        record_proposal,
        update_proposal_status,
    )
    from autoresearch.experiment_registry.schema import init_registry
    import autoresearch.experiment_runner as runner_mod

    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    source_dir.mkdir()
    destination_dir.mkdir()
    source_registry = source_dir / "registry.sqlite"
    destination_registry = destination_dir / "registry.sqlite"
    init_registry(source_registry)
    init_registry(destination_registry)

    recipe_snapshot = {
        "target_mode": "burning_cost",
        "experiment": {
            "experiment_name": "recipe_finalist",
            "model_family": "recipe",
            "target_strategy": "direct_pure_premium",
            "preprocessing": {},
            "model": {
                "recipe": {
                    "structure": "direct",
                    "estimator": "lightgbm",
                    "objective": "tweedie",
                    "encoding": "native_categorical",
                }
            },
        },
    }
    _record_experiment(
        source_registry,
        source_dir,
        "recipe_source",
        family="recipe",
        snapshot=recipe_snapshot,
    )

    script = source_dir / "model_final.py"
    script.write_text(
        "def fit_predict(train, score, **kwargs):\n    return None\n",
        encoding="utf-8",
    )
    proposal = source_dir / "script_proposal.json"
    write_json(
        proposal, {"experiment_name": "script_finalist", "scientific": "audit me"}
    )
    script_snapshot = {
        "target_mode": "burning_cost",
        "model_script_path": str(script),
        "experiment": {
            "experiment_name": "script_finalist",
            "model_family": "scripted_challenger",
            "target_strategy": "direct_pure_premium",
            "preprocessing": {},
            "model": {"script_path": "model_final.py"},
        },
    }
    _record_experiment(
        source_registry,
        source_dir,
        "script_source",
        family="scripted_challenger",
        snapshot=script_snapshot,
        artifacts={"model_script": script},
    )
    record_proposal(
        source_registry,
        proposal_id="script_proposal",
        status="completed",
        parent_experiment_id=None,
        parent_branch_id="main",
        branch_id="main",
        experiment_name="script_finalist",
        rationale="fixture",
        change_summary="fixture",
        expected_benefit="fixture",
        key_risk="fixture",
        config={},
        validation_errors=[],
        llm_provider="fixture",
        llm_model=None,
        prompt_path=None,
        response_path=None,
        proposal_path=proposal,
    )
    update_proposal_status(
        source_registry,
        "script_proposal",
        "completed",
        experiment_id="script_source",
    )
    _record_experiment(
        source_registry,
        source_dir,
        "script_seed_replay",
        family="scripted_challenger",
        snapshot=script_snapshot,
        artifacts={"model_script": script},
    )
    source_lineage = source_dir / "orchestration_replay" / "lineage.json"
    source_lineage.parent.mkdir(parents=True)
    write_json(
        source_lineage,
        {
            "replays": [
                {
                    "destination_experiment_id": "script_seed_replay",
                    "source": ReplaySource(
                        "claude", "20260710T120001Z", "script_source"
                    ).to_dict(),
                }
            ]
        },
    )

    _record_experiment(
        destination_registry,
        destination_dir,
        "baseline",
        family="global_mean",
        snapshot={
            "target_mode": "burning_cost",
            "experiment": {
                "experiment_name": "baseline",
                "model_family": "global_mean",
                "target_strategy": "direct_pure_premium",
                "model": {},
            },
        },
    )
    _set_champion(destination_registry, "baseline")

    source_config = SimpleNamespace(
        dataset_name="french_motor",
        target_mode="burning_cost",
        registry_path=source_registry,
        track_id="claude",
        run_id="20260710T120001Z",
    )
    destination_config = SimpleNamespace(
        dataset_name="french_motor",
        target_mode="burning_cost",
        registry_path=destination_registry,
        artifacts_dir=destination_dir,
        track_id="claude",
        run_id="20260710T130001Z",
    )
    monkeypatch.setattr(playoff_mod, "load_config", lambda **kwargs: source_config)

    executed: list[dict] = []

    def fake_run_experiment(config, config_path, *, output_dir):
        with Path(config_path).open("rb") as handle:
            parsed = tomllib.load(handle)
        executed.append(parsed)
        experiment_id = f"replayed_{len(executed)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = output_dir / "config_snapshot.json"
        metrics_path = output_dir / "metrics.json"
        write_json(
            snapshot_path,
            {
                "experiment_id": experiment_id,
                "target_mode": "burning_cost",
                "experiment": parsed,
            },
        )
        write_json(metrics_path, {})
        _record_experiment(
            destination_registry,
            output_dir,
            experiment_id,
            family=str(parsed["model_family"]),
            snapshot=read_json(snapshot_path),
        )
        return {"config_snapshot": snapshot_path, "metrics": metrics_path}

    monkeypatch.setattr(runner_mod, "run_experiment", fake_run_experiment)

    recipe_replay = playoff_mod.replay_experiment(
        destination_config,
        ReplaySource("claude", "20260710T120001Z", "recipe_source", "d01", "orch"),
        label="recipe_d01",
    )
    recovered_recipe = playoff_mod.replay_experiment(
        destination_config,
        ReplaySource("claude", "20260710T120001Z", "recipe_source", "d01", "orch"),
        label="recipe_d01",
    )
    script_replay = playoff_mod.replay_experiment(
        destination_config,
        ReplaySource("claude", "20260710T120001Z", "script_source", "d02", "orch"),
        label="script_d02",
    )
    seed_script_replay = playoff_mod.replay_experiment(
        destination_config,
        ReplaySource("claude", "20260710T120001Z", "script_seed_replay", "d03", "orch"),
        label="script_seed_d03",
    )

    assert recipe_replay["representation"] == "recipe"
    assert (
        recovered_recipe["destination_experiment_id"]
        == recipe_replay["destination_experiment_id"]
    )
    assert executed[0]["model"]["recipe"]["estimator"] == "lightgbm"
    assert script_replay["representation"] == "script"
    assert executed[1]["model"]["script_path"] == "model_replay.py"
    copied_script = (
        destination_dir / "orchestration_replay" / "script_d02" / "model_replay.py"
    )
    copied_proposal = (
        destination_dir / "orchestration_replay" / "script_d02" / "source_proposal.json"
    )
    assert copied_script.read_text(encoding="utf-8") == script.read_text(
        encoding="utf-8"
    )
    assert read_json(copied_proposal)["scientific"] == "audit me"
    assert seed_script_replay["representation"] == "script"
    assert (
        read_json(
            destination_dir
            / "orchestration_replay"
            / "script_seed_d03"
            / "source_proposal.json"
        )["scientific"]
        == "audit me"
    )
    lineage = read_json(destination_dir / "orchestration_replay" / "lineage.json")
    assert [item["source"]["delegation_id"] for item in lineage["replays"]] == [
        "d01",
        "d02",
        "d03",
    ]


def test_failed_replay_writes_audit_and_snapshot_only_directory_is_retriable(
    tmp_path, monkeypatch
):
    from autoresearch.experiment_registry.schema import init_registry
    import autoresearch.experiment_runner as runner_mod

    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    source_dir.mkdir()
    destination_dir.mkdir()
    source_registry = source_dir / "registry.sqlite"
    destination_registry = destination_dir / "registry.sqlite"
    init_registry(source_registry)
    init_registry(destination_registry)
    snapshot = {
        "target_mode": "burning_cost",
        "experiment": {
            "experiment_name": "recipe",
            "model_family": "recipe",
            "target_strategy": "direct_pure_premium",
            "model": {"recipe": {"structure": "direct"}},
        },
    }
    _record_experiment(
        source_registry, source_dir, "source", family="recipe", snapshot=snapshot
    )
    _record_experiment(
        destination_registry,
        destination_dir,
        "baseline",
        family="global_mean",
        snapshot=snapshot,
    )
    _set_champion(destination_registry, "baseline")
    source_config = SimpleNamespace(
        dataset_name="french_motor",
        target_mode="burning_cost",
        registry_path=source_registry,
        artifacts_dir=source_dir,
        track_id="claude",
        run_id="source_run",
    )
    destination_config = SimpleNamespace(
        dataset_name="french_motor",
        target_mode="burning_cost",
        registry_path=destination_registry,
        artifacts_dir=destination_dir,
        track_id="claude",
        run_id="destination_run",
    )
    monkeypatch.setattr(playoff_mod, "load_config", lambda **kwargs: source_config)
    replay_dir = destination_dir / "orchestration_replay" / "retry"
    replay_dir.mkdir(parents=True)
    write_json(replay_dir / "source_config_snapshot.json", snapshot)

    monkeypatch.setattr(
        runner_mod,
        "run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        playoff_mod.replay_experiment(
            destination_config,
            ReplaySource("claude", "source_run", "source"),
            label="retry",
        )
    audit = read_json(replay_dir / "replay_manifest.json")
    assert audit["status"] == "failed"
    assert audit["failure"]["message"] == "boom"

    def successful_run(config, config_path, *, output_dir):
        output_dir.mkdir(parents=True)
        output = output_dir / "config_snapshot.json"
        write_json(
            output, {"experiment_id": "replayed", "experiment": snapshot["experiment"]}
        )
        _record_experiment(
            destination_registry,
            output_dir,
            "replayed",
            family="recipe",
            snapshot=read_json(output),
        )
        return {"config_snapshot": output}

    monkeypatch.setattr(runner_mod, "run_experiment", successful_run)
    replay = playoff_mod.replay_experiment(
        destination_config,
        ReplaySource("claude", "source_run", "source"),
        label="retry",
    )
    assert replay["status"] == "completed"
    assert replay["destination_experiment_id"] == "replayed"


def test_brief_seed_uses_shared_replay_path(tmp_path, monkeypatch):
    from autoresearch.config import load_config

    config = replace(
        load_config(),
        artifacts_dir=tmp_path / "child",
        track_id="claude",
        run_id="20260710T130001Z",
    )
    config.artifacts_dir.mkdir()
    backend = Backend(
        name="fixture",
        tool="stub",
        command=("stub",),
        prompt_via="stdin",
        track="claude",
        model_provider="fixture",
        model_name="fixture",
    )
    brief = Brief(
        direction="Continue the winner.",
        cycle_budget=1,
        seed_champion=SeedChampion(
            from_run="codex/20260710T120001Z", experiment_id="source_exp"
        ),
    )
    orch = SimpleNamespace(
        dataset="french_motor",
        target_mode="burning_cost",
        orchestration_id="20260710T120000Z",
    )
    monkeypatch.setattr(spawner_mod, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(
        "autoresearch.bootstrap.bootstrap_track", lambda *args, **kwargs: None
    )
    calls = []
    monkeypatch.setattr(
        playoff_mod,
        "seed_champion_from_source",
        lambda destination, source, label: calls.append((destination, source, label)),
    )

    spawner_mod._bootstrap_child_run(
        orch, backend=backend, brief=brief, delegation_id="d02"
    )

    assert calls[0][1].run_ref == "codex/20260710T120001Z"
    assert calls[0][1].experiment_id == "source_exp"
    assert calls[0][2] == "delegation_seed_d02"


def _playoff_fixture(tmp_path: Path, monkeypatch):
    from autoresearch.experiment_registry.comparisons import record_comparison
    from autoresearch.experiment_registry.schema import init_registry
    import autoresearch.comparison_runner as comparison_runner
    import autoresearch.milestone as milestone

    orch = create_orchestration(
        dataset="french_motor",
        target_mode="burning_cost",
        total_cycle_budget=4,
        model_provider="fixture",
        model_name="fixture",
        orchestration_id="20260710T140000Z",
    )
    finalist_specs = (
        ("d01", "20260710T140101Z", "recipe", "weak", 0.10, False),
        ("d02", "20260710T140102Z", "scripted_challenger", "mid", 0.20, False),
        ("d03", "20260710T140103Z", "recipe", "strong", 0.30, False),
        ("d04", "20260710T140104Z", "global_mean", "baseline", 0.00, True),
    )
    for (
        delegation_id,
        run_id,
        family,
        experiment_id,
        gini,
        is_baseline,
    ) in finalist_specs:
        report_path = manifest_mod.report_path(orch.orchestration_id, delegation_id)
        write_json(
            report_path,
            {
                "delegation_id": delegation_id,
                "champion": {
                    "experiment_id": experiment_id,
                    "model_family": family,
                    "gini_weighted": gini,
                },
                "distress": {"active": ["champion_is_baseline"] if is_baseline else []},
            },
        )
        orch = add_delegation(
            orch,
            Delegation(
                delegation_id=delegation_id,
                brief_path=f"briefs/{delegation_id}.json",
                backend="stub",
                track="claude",
                run_id=run_id,
                cycle_budget=1,
                status="completed",
                report_path=str(report_path),
            ),
        )
    save_orchestration(orch)

    consolidation_dir = tmp_path / "consolidation"
    consolidation_dir.mkdir()
    registry = consolidation_dir / "registry.sqlite"
    init_registry(registry)
    _record_experiment(
        registry,
        consolidation_dir,
        "flat",
        family="global_mean",
        snapshot={
            "target_mode": "burning_cost",
            "experiment": {
                "experiment_name": "flat",
                "model_family": "global_mean",
                "target_strategy": "direct_pure_premium",
                "model": {},
            },
        },
    )
    _set_champion(registry, "flat")
    consolidation = SimpleNamespace(
        registry_path=registry,
        artifacts_dir=consolidation_dir,
        track_id="claude",
        run_id="20260710T150000Z",
    )
    monkeypatch.setattr(
        playoff_mod, "_create_consolidation_run", lambda *args, **kwargs: consolidation
    )
    monkeypatch.setattr(
        playoff_mod, "_load_consolidation_config", lambda orch: consolidation
    )
    monkeypatch.setattr(
        comparison_runner, "_refresh_decision_outputs", lambda *args, **kwargs: None
    )

    order: list[str] = []

    def record_replay(experiment_id: str, source: ReplaySource) -> dict:
        _record_experiment(
            registry,
            consolidation_dir,
            experiment_id,
            family="recipe",
            snapshot={
                "target_mode": "burning_cost",
                "experiment": {
                    "experiment_name": experiment_id,
                    "model_family": "recipe",
                    "target_strategy": "direct_pure_premium",
                    "model": {"recipe": {"structure": "direct"}},
                },
            },
        )
        payload = {
            "source": source.to_dict(),
            "destination_experiment_id": experiment_id,
        }
        lineage_path = consolidation_dir / "orchestration_replay" / "lineage.json"
        lineage_path.parent.mkdir(parents=True, exist_ok=True)
        lineage = read_json(lineage_path) if lineage_path.exists() else {"replays": []}
        lineage["replays"].append(payload)
        write_json(lineage_path, lineage)
        return payload

    def fake_seed(config, source, *, label):
        order.append(f"seed:{source.delegation_id}")
        payload = record_replay(f"replayed_{source.delegation_id}", source)
        _set_champion(registry, str(payload["destination_experiment_id"]))
        return payload

    def fake_replay(config, source, *, label):
        order.append(str(source.delegation_id))
        return record_replay(f"replayed_{source.delegation_id}", source)

    def fake_compare(config, challenger_id):
        from autoresearch.experiment_registry.champions import get_official_champion

        champion_id = str(get_official_champion(registry)["champion_id"])
        comparison_id = f"cmp_{challenger_id}"
        passed = challenger_id.endswith("d03")
        comparison_dir = consolidation_dir / comparison_id
        comparison_dir.mkdir()
        decision_path = comparison_dir / "promotion_decision.json"
        report_path = comparison_dir / "promotion_report.json"
        write_json(decision_path, {})
        report = {
            "comparison_id": comparison_id,
            "champion_id": champion_id,
            "challenger_id": challenger_id,
            "comparison_summary": {
                "champion_id": champion_id,
                "challenger_id": challenger_id,
                "mean_lift": 0.01 if passed else -0.01,
            },
            "advisory_promotion_decision": {
                "decision": "promote" if passed else "reject",
                "rationale": "fixture gates",
                "checks": {"mean_lift": passed, "win_rate": passed},
            },
            "guardrail_result": {"passed": True, "failures": []},
        }
        write_json(report_path, report)
        record_comparison(
            registry,
            comparison_id=comparison_id,
            champion_id=champion_id,
            challenger_id=challenger_id,
            paired_summary={"mean_lift": 0.01 if passed else -0.01},
            bootstrap_summary={},
            promotion_decision="pending_llm",
            promotion_rationale="fixture",
            guardrail_status={"passed": True, "failures": []},
            artifacts={
                "promotion_decision": decision_path,
                "promotion_report": report_path,
            },
        )
        return {"promotion_report": report_path}

    monkeypatch.setattr(playoff_mod, "seed_champion_from_source", fake_seed)
    monkeypatch.setattr(playoff_mod, "replay_experiment", fake_replay)
    monkeypatch.setattr(playoff_mod, "_compare_replayed_experiment", fake_compare)

    milestone_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        milestone,
        "evaluate_on_holdout",
        lambda config, experiment_id, comparison_id: milestone_calls.append(
            (experiment_id, comparison_id)
        ),
    )
    return orch, consolidation, order, milestone_calls


def test_interactive_playoff_stops_at_each_pending_decision(
    orchestrations_root, tmp_path, monkeypatch
):
    orch, consolidation, order, milestone_calls = _playoff_fixture(
        tmp_path, monkeypatch
    )

    first = playoff_mod.run_playoff(orch.orchestration_id)
    assert first["status"] == "consolidating"
    assert order == ["seed:d01", "d02"]
    assert first["pairings"][0]["decision"] is None
    assert first["pairings"][0]["comparison_id"] == "cmp_replayed_d02"

    playoff_mod._record_playoff_decision(
        consolidation,
        "cmp_replayed_d02",
        decision="reject",
        rationale="Interactive fixture rejection.",
        reason_code="inferior",
    )
    second = playoff_mod.run_playoff(orch.orchestration_id)
    assert order == ["seed:d01", "d02", "d03"]
    assert second["pairings"][0]["decision"]["decision"] == "reject"
    assert second["pairings"][1]["decision"] is None
    assert milestone_calls == []
    assert load_orchestration(orch.orchestration_id).status == "consolidating"


def test_auto_playoff_orders_finalists_reports_lineage_and_calls_promotion_hook(
    orchestrations_root, tmp_path, monkeypatch
):
    orch, _consolidation, order, milestone_calls = _playoff_fixture(
        tmp_path, monkeypatch
    )

    result = playoff_mod.run_playoff(orch.orchestration_id, auto_decide=True)

    assert result["status"] == "completed"
    assert [item["delegation_id"] for item in result["finalists"]] == [
        "d01",
        "d02",
        "d03",
    ]
    assert result["exclusions"] == [
        {"delegation_id": "d04", "reason": "champion_is_baseline"}
    ]
    assert order == ["seed:d01", "d02", "d03"]
    assert [pair["decision"]["decision"] for pair in result["pairings"]] == [
        "reject",
        "promote",
    ]
    assert result["pairings"][0]["gate_evidence"]["all_standard_gates_pass"] is False
    assert result["pairings"][1]["gate_evidence"]["all_standard_gates_pass"] is True
    assert milestone_calls == [("replayed_d03", "cmp_replayed_d03")]
    assert result["final_champion_lineage"]["source"]["delegation_id"] == "d03"

    updated = load_orchestration(orch.orchestration_id)
    assert updated.status == "completed"
    assert updated.consolidation.track == "claude"
    assert updated.consolidation.run_id == "20260710T150000Z"
    assert updated.consolidation.playoff_report
    report_dir = manifest_mod.playoff_dir(orch.orchestration_id)
    assert read_json(report_dir / "playoff_report.json")["status"] == "completed"
    markdown = (report_dir / "playoff_report.md").read_text(encoding="utf-8")
    assert "Finalist order" in markdown
    assert "cmp_replayed_d03" in markdown
    assert "Final champion lineage" in markdown


def test_include_limits_the_ranked_finalist_set(
    orchestrations_root, tmp_path, monkeypatch
):
    orch, _consolidation, _order, _milestone_calls = _playoff_fixture(
        tmp_path, monkeypatch
    )

    finalists, exclusions = playoff_mod.collect_finalists(orch, ["d03", "d01"])

    assert [item.delegation_id for item in finalists] == ["d01", "d03"]
    assert exclusions == []


def test_collect_finalists_excludes_seed_replay_duplicate_and_renders_reason(
    orchestrations_root, tmp_path, monkeypatch
):
    orch, _consolidation, _order, _milestone_calls = _playoff_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        playoff_mod,
        "_seed_origin",
        lambda source: (
            ("claude", "20260710T140101Z", "weak")
            if source.delegation_id == "d02"
            else None
        ),
    )

    finalists, exclusions = playoff_mod.collect_finalists(orch)

    assert [item.delegation_id for item in finalists] == ["d01", "d03"]
    assert {"delegation_id": "d02", "reason": "duplicate_of d01"} in exclusions
    markdown = playoff_mod._render_markdown(
        {
            "orchestration_id": orch.orchestration_id,
            "status": "consolidating",
            "decision_mode": "interactive",
            "finalists": [item.to_dict() for item in finalists],
            "exclusions": exclusions,
            "pairings": [],
            "final_champion_lineage": None,
        }
    )
    assert "`d02`: duplicate_of d01" in markdown


def test_exclude_after_start_completes_with_current_promoted_champion(
    orchestrations_root, tmp_path, monkeypatch
):
    orch, consolidation, _order, milestone_calls = _playoff_fixture(
        tmp_path, monkeypatch
    )
    first = playoff_mod.run_playoff(orch.orchestration_id)
    playoff_mod._record_playoff_decision(
        consolidation,
        first["pairings"][0]["comparison_id"],
        decision="promote",
        rationale="Fixture promotion.",
        reason_code="clear_win",
    )

    result = playoff_mod.run_playoff(orch.orchestration_id, exclude=["d03"])

    assert result["status"] == "completed"
    assert {"delegation_id": "d03", "reason": "operator_excluded"} in result[
        "exclusions"
    ]
    assert result["final_champion_lineage"]["source"]["delegation_id"] == "d02"
    assert load_orchestration(orch.orchestration_id).status == "completed"
    assert milestone_calls == [("replayed_d02", "cmp_replayed_d02")]


def test_resume_excludes_a_hard_failed_replay_and_completes(
    orchestrations_root, tmp_path, monkeypatch
):
    orch, consolidation, _order, milestone_calls = _playoff_fixture(
        tmp_path, monkeypatch
    )
    first = playoff_mod.run_playoff(orch.orchestration_id)
    playoff_mod._record_playoff_decision(
        consolidation,
        first["pairings"][0]["comparison_id"],
        decision="promote",
        rationale="Fixture promotion.",
        reason_code="clear_win",
    )
    failed_dir = (
        consolidation.artifacts_dir / "orchestration_replay" / "challenger_02_d03"
    )
    failed_dir.mkdir(parents=True)
    write_json(
        failed_dir / "replay_manifest.json",
        {"status": "failed", "failure": {"type": "RuntimeError", "message": "boom"}},
    )

    result = playoff_mod.run_playoff(orch.orchestration_id, resume=True)

    assert result["status"] == "completed"
    assert any(
        item["delegation_id"] == "d03" and item["reason"] == "replay_failed"
        for item in result["exclusions"]
    )
    assert result["final_champion_lineage"]["source"]["delegation_id"] == "d02"
    assert milestone_calls == [("replayed_d02", "cmp_replayed_d02")]


def test_auto_decide_requires_checks_advisory_and_guardrail_all_to_pass():
    base = {
        "advisory_promotion_decision": {
            "decision": "promote",
            "checks": {"lift": True, "win_rate": True},
        },
        "guardrail_result": {"passed": True},
    }
    assert playoff_mod._gate_evidence(base)["all_standard_gates_pass"] is True
    failed_check = {
        **base,
        "advisory_promotion_decision": {
            "decision": "promote",
            "checks": {"lift": True, "win_rate": False},
        },
    }
    assert playoff_mod._gate_evidence(failed_check)["all_standard_gates_pass"] is False
    failed_guardrail = {**base, "guardrail_result": {"passed": False}}
    assert (
        playoff_mod._gate_evidence(failed_guardrail)["all_standard_gates_pass"] is False
    )


def test_replay_rejects_different_fixed_preprocessing():
    destination = SimpleNamespace(
        claim_capping_enabled=True,
        claim_cap_threshold=100000.0,
    )
    with pytest.raises(ValueError, match="claim-cap threshold"):
        playoff_mod._validate_fixed_preprocessing(
            {
                "effective_preprocessing": {
                    "claim_capping_enabled": True,
                    "claim_cap_threshold": 50000.0,
                }
            },
            {},
            destination,
        )


def test_playoff_cli_modes_and_include_are_wired(monkeypatch, capsys):
    from autoresearch import cli

    parser = cli.build_parser()
    assert parser.parse_args(["orchestrate", "playoff"]).auto_decide is False
    assert (
        parser.parse_args(["orchestrate", "playoff", "--interactive"]).auto_decide
        is False
    )
    assert (
        parser.parse_args(["orchestrate", "playoff", "--auto-decide"]).auto_decide
        is True
    )
    recovery = parser.parse_args(
        ["orchestrate", "playoff", "--exclude", "d09", "--resume"]
    )
    assert recovery.exclude == "d09"
    assert recovery.resume is True

    captured = {}
    monkeypatch.setattr(
        manifest_mod, "resolve_orchestration_id", lambda value: "20260710T140000Z"
    )
    monkeypatch.setattr(
        playoff_mod,
        "run_playoff",
        lambda orchestration_id, **kwargs: (
            captured.update({"orchestration_id": orchestration_id, **kwargs})
            or {
                "orchestration_id": orchestration_id,
                "status": "completed",
                "consolidation": {"track": "claude", "run_id": "20260710T150000Z"},
                "pairings": [],
            }
        ),
    )
    args = parser.parse_args(
        ["orchestrate", "playoff", "--include", "d03,d01", "--auto-decide"]
    )

    assert cli._orchestrate_playoff(args) == 0
    assert captured == {
        "orchestration_id": "20260710T140000Z",
        "include": ["d03", "d01"],
        "exclude": None,
        "resume": False,
        "auto_decide": True,
    }
    assert '"status": "completed"' in capsys.readouterr().out
