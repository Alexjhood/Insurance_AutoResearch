"""Phase 1 — orchestration core: manifest, briefs, backends, spawner, reports.

The distress predicates and the prompt/command composition are pure functions and
are unit-tested directly. Report generation is exercised against a fixture
registry, so the "reports are built from the registry, never from the sub-agent's
claims" property is checked rather than assumed.

Nothing here launches a sub-agent or reads holdout data.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.orchestration import backends as backends_mod
from autoresearch.orchestration import manifest as manifest_mod
from autoresearch.orchestration import report as report_mod
from autoresearch.orchestration import spawner as spawner_mod
from autoresearch.orchestration.backends import Backend, get_backend, load_backends
from autoresearch.orchestration.brief import Brief, render_brief_block, validate_brief
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    add_delegation,
    create_orchestration,
    load_orchestration,
    manifest_lock,
    read_run_backpointer,
    save_orchestration,
    update_delegation,
    write_run_backpointer,
)
from autoresearch.orchestration.report import assess_distress, build_report
from autoresearch.utils.io import write_json


@pytest.fixture
def orchestrations_root(tmp_path, monkeypatch):
    """Redirect artifacts/orchestrations/ into a tmp dir for the whole module."""

    root = tmp_path / "orchestrations"
    root.mkdir()
    monkeypatch.setattr(manifest_mod, "ORCHESTRATIONS_DIR", root)
    return root


# ── manifest ────────────────────────────────────────────────────────────────


def test_manifest_round_trip(orchestrations_root):
    orch = create_orchestration(
        dataset="porto_seguro",
        target_mode="claim_incidence",
        total_cycle_budget=12,
        model_provider="anthropic",
        model_name="claude-opus-4-8",
    )
    delegation = Delegation(
        delegation_id="d01",
        brief_path="briefs/d01.json",
        backend="stub",
        track="claude",
        run_id="20260712T091500Z",
        cycle_budget=4,
    )
    save_orchestration(add_delegation(orch, delegation))

    reloaded = load_orchestration(orch.orchestration_id)
    assert reloaded.to_dict() == add_delegation(orch, delegation).to_dict()
    assert reloaded.delegation("d01").backend == "stub"
    assert reloaded.cycles_committed == 4
    assert reloaded.cycles_remaining == 8


def test_manifest_rejects_bad_records(orchestrations_root):
    with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
        Orchestration(
            orchestration_id="not-a-timestamp",
            dataset="porto_seguro",
            target_mode="claim_incidence",
            created_at="now",
            total_cycle_budget=4,
        )
    with pytest.raises(ValueError, match="delegation_id"):
        Delegation(
            delegation_id="first",
            brief_path="b",
            backend="stub",
            track="claude",
            run_id="r",
            cycle_budget=1,
        )
    with pytest.raises(ValueError, match="cycle_budget"):
        Delegation(
            delegation_id="d01",
            brief_path="b",
            backend="stub",
            track="claude",
            run_id="r",
            cycle_budget=0,
        )


def test_next_delegation_id_increments(orchestrations_root):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=9
    )
    assert orch.next_delegation_id() == "d01"
    orch = add_delegation(
        orch,
        Delegation(
            delegation_id="d01",
            brief_path="b",
            backend="stub",
            track="claude",
            run_id="r",
            cycle_budget=1,
        ),
    )
    assert orch.next_delegation_id() == "d02"


def test_update_delegation_rejects_unknown_id(orchestrations_root):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=4
    )
    stray = Delegation(
        delegation_id="d09",
        brief_path="b",
        backend="stub",
        track="claude",
        run_id="r",
        cycle_budget=1,
    )
    with pytest.raises(KeyError):
        update_delegation(orch, stray)


def test_manifest_lock_is_exclusive_and_released(orchestrations_root):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=4
    )
    oid = orch.orchestration_id
    lock_file = manifest_mod.orchestration_dir(oid) / "orchestration.lock"

    with manifest_lock(oid):
        assert lock_file.exists()
        # A second acquirer must not slip in while the first holds the lock.
        with pytest.raises(TimeoutError):
            with manifest_lock(oid, timeout=0.2):
                pass
    assert not lock_file.exists()


def test_run_backpointer_round_trip(tmp_path):
    assert read_run_backpointer(tmp_path) is None
    write_run_backpointer(tmp_path, orchestration_id="20260712T090000Z", delegation_id="d01")
    assert read_run_backpointer(tmp_path) == ("20260712T090000Z", "d01")


def test_run_backpointer_preserves_existing_manifest_keys(tmp_path):
    write_json(tmp_path / "run_manifest.json", {"dataset": "porto_seguro", "default_max_cycles": 4})
    write_run_backpointer(tmp_path, orchestration_id="20260712T090000Z", delegation_id="d02")
    import json

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["dataset"] == "porto_seguro"
    assert manifest["default_max_cycles"] == 4
    assert manifest["delegation_id"] == "d02"


# ── briefs ──────────────────────────────────────────────────────────────────


def test_validate_brief_minimal():
    brief = validate_brief({"direction": "Try freq-sev", "cycle_budget": 3})
    assert brief.direction == "Try freq-sev"
    assert brief.cycle_budget == 3
    assert brief.constraints == ()


def test_validate_brief_requires_direction_and_budget():
    with pytest.raises(ValueError, match="missing required field"):
        validate_brief({"cycle_budget": 3})
    with pytest.raises(ValueError, match="missing required field"):
        validate_brief({"direction": "x"})


def test_validate_brief_rejects_unknown_field():
    """A typo'd key must fail loudly, not silently drop the orchestrator's intent."""
    with pytest.raises(ValueError, match="unknown field"):
        validate_brief({"direction": "x", "cycle_budget": 1, "constraint": ["typo"]})


def test_validate_brief_rejects_nonpositive_budget():
    with pytest.raises(ValueError, match="positive integer"):
        validate_brief({"direction": "x", "cycle_budget": 0})


def test_validate_brief_rejects_bool_budget():
    with pytest.raises(ValueError, match="must be an integer"):
        validate_brief({"direction": "x", "cycle_budget": True})


def test_validate_brief_allows_underscore_metadata():
    """The spawner stamps `_source_path` onto its archived copy; it must re-validate."""
    brief = validate_brief(
        {"direction": "x", "cycle_budget": 1, "_source_path": "/tmp/brief.json"}
    )
    assert brief.cycle_budget == 1


def test_stored_brief_round_trips_through_validation(orchestrations_root, tmp_path):
    """Regression: an archived brief must reload, or the child's handoff loses it."""
    from autoresearch.orchestration.brief import load_brief

    orch = create_orchestration(
        dataset="french_motor", target_mode="burning_cost", total_cycle_budget=4
    )
    source = tmp_path / "brief.json"
    write_json(source, {"direction": "Probe constants.", "cycle_budget": 2})
    stored = spawner_mod._store_brief(orch, "d01", validate_brief({"direction": "Probe constants.", "cycle_budget": 2}), source)

    reloaded = load_brief(stored)
    assert reloaded.direction == "Probe constants."
    assert reloaded.cycle_budget == 2


def test_seed_champion_parsed():
    brief = validate_brief(
        {
            "direction": "x",
            "cycle_budget": 1,
            "seed_champion": {"from_run": "claude/20260712T091500Z", "experiment_id": "exp1"},
        }
    )
    assert brief.seed_champion.track == "claude"
    assert brief.seed_champion.run_id == "20260712T091500Z"


def test_seed_champion_requires_qualified_run():
    with pytest.raises(ValueError, match="from_run"):
        validate_brief(
            {
                "direction": "x",
                "cycle_budget": 1,
                "seed_champion": {"from_run": "20260712T091500Z", "experiment_id": "e"},
            }
        )


def test_render_brief_block_carries_binding_language():
    brief = Brief(
        direction="Explore freq-sev.",
        cycle_budget=4,
        constraints=("Stay within gbm.",),
        starting_knowledge=("num_leaves > 63 overfits.",),
        success_criteria="Beat 0.32.",
    )
    block = "\n".join(render_brief_block(brief, cycle_budget=4, delegation_id="d02"))
    assert "## Orchestration brief" in block
    assert "binding" in block
    assert "d02" in block
    assert "Stay within gbm." in block
    assert "num_leaves > 63 overfits." in block
    assert "Beat 0.32." in block
    assert "finish-delegation" in block


# ── backends ────────────────────────────────────────────────────────────────


def test_repo_backend_registry_loads_and_validates():
    registry = load_backends()
    assert "stub" in registry
    for backend in registry.values():
        assert backend.track in backends_mod.ALLOWED_TRACKS


def test_distinct_backends_render_distinct_commands():
    """Two named backends that produce identical argv are the same backend twice.

    Regression: `claude-sonnet-low` and `claude-sonnet-medium` once differed only
    in `model_name` attribution, because no `--effort` flag was pinned — so the
    scorecard would have compared a backend against itself.
    """
    registry = load_backends()
    rendered = {
        name: backend.render_command(prompt="P")
        for name, backend in registry.items()
        if backend.is_spawnable
    }
    collisions = [
        (a, b)
        for a in rendered
        for b in rendered
        if a < b and rendered[a] == rendered[b]
    ]
    assert not collisions, f"backends render identical commands: {collisions}"


def test_claude_backends_pin_a_real_effort_level():
    registry = load_backends()
    low = registry["claude-sonnet-low"].command
    medium = registry["claude-sonnet-medium"].command
    assert "--effort" in low and low[low.index("--effort") + 1] == "low"
    assert "--effort" in medium and medium[medium.index("--effort") + 1] == "medium"
    # `--max-turns` does not exist in the Claude CLI; it must not reappear.
    assert "--max-turns" not in low and "--max-turns" not in medium


def test_bad_effort_value_fails_at_load_time():
    """Claude only *warns* on a bad --effort and uses the default; we must not."""
    with pytest.raises(ValueError, match="--effort='bogus' is not accepted"):
        Backend(
            name="bad",
            tool="claude",
            command=("claude", "-p", "--effort", "bogus"),
            prompt_via="stdin",
            track="claude",
            model_provider="anthropic",
            model_name="m",
        )


def test_bad_permission_mode_fails_at_load_time():
    with pytest.raises(ValueError, match="--permission-mode"):
        Backend(
            name="bad",
            tool="claude",
            command=("claude", "-p", "--permission-mode", "yolo"),
            prompt_via="stdin",
            track="claude",
            model_provider="anthropic",
            model_name="m",
        )


def test_enumerated_flag_validation_is_scoped_to_the_tool():
    """A non-claude tool may legitimately use `--effort` with its own vocabulary."""
    backend = Backend(
        name="ok",
        tool="opencode",
        command=("opencode", "--effort", "ludicrous", "{prompt}"),
        prompt_via="argv",
        track="opencode",
        model_provider="x",
        model_name="y",
    )
    assert backend.render_command(prompt="P")[2] == "ludicrous"


def test_max_budget_usd_placeholder_requires_a_value():
    with pytest.raises(ValueError, match="max_budget_usd is configured"):
        Backend(
            name="bad",
            tool="claude",
            command=("claude", "-p", "--max-budget-usd", "{max_budget_usd}"),
            prompt_via="stdin",
            track="claude",
            model_provider="anthropic",
            model_name="m",
        )


def test_max_budget_usd_renders_without_trailing_zeros():
    backend = Backend(
        name="b",
        tool="claude",
        command=("claude", "-p", "--max-budget-usd", "{max_budget_usd}"),
        prompt_via="stdin",
        track="claude",
        model_provider="anthropic",
        model_name="m",
        max_budget_usd=5.0,
    )
    assert backend.render_command(prompt="P") == ("claude", "-p", "--max-budget-usd", "5")


def test_backend_rejects_track_the_guard_would_refuse():
    with pytest.raises(ValueError, match="run-scope guard"):
        Backend(
            name="bad",
            tool="claude",
            command=("claude",),
            prompt_via="stdin",
            track="analyst",
            model_provider="anthropic",
            model_name="m",
        )


def test_backend_argv_prompt_requires_placeholder():
    with pytest.raises(ValueError, match="requires a '.prompt.' placeholder"):
        Backend(
            name="bad",
            tool="codex",
            command=("codex", "exec"),
            prompt_via="argv",
            track="codex",
            model_provider="openai",
            model_name="m",
        )


def test_backend_stdin_prompt_forbids_placeholder():
    with pytest.raises(ValueError, match="must not carry"):
        Backend(
            name="bad",
            tool="claude",
            command=("claude", "{prompt}"),
            prompt_via="stdin",
            track="claude",
            model_provider="anthropic",
            model_name="m",
        )


def test_backend_rejects_unknown_placeholder():
    with pytest.raises(ValueError, match="unknown command placeholder"):
        Backend(
            name="bad",
            tool="claude",
            command=("claude", "--model", "{model}"),
            prompt_via="stdin",
            track="claude",
            model_provider="anthropic",
            model_name="m",
        )


def test_backend_render_command_substitutes_prompt_and_max_turns():
    backend = Backend(
        name="b",
        tool="codex",
        command=("codex", "exec", "--max-turns", "{max_turns}", "{prompt}"),
        prompt_via="argv",
        track="codex",
        model_provider="openai",
        model_name="m",
        max_turns=42,
    )
    argv = backend.render_command(prompt="do science")
    assert argv == ("codex", "exec", "--max-turns", "42", "do science")


def test_get_backend_refuses_deprecated(tmp_path):
    config = tmp_path / "backends.toml"
    config.write_text(
        """
[backends.old]
tool = "claude"
command = ["claude", "-p"]
prompt_via = "stdin"
track = "claude"
model_provider = "anthropic"
model_name = "old"
status = "deprecated"
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deprecated"):
        get_backend("old", path=config)


def test_get_backend_unknown_name_lists_alternatives():
    with pytest.raises(KeyError, match="stub"):
        get_backend("no-such-backend")


# ── spawner: prompt, env, plan ──────────────────────────────────────────────


def test_compose_prompt_pins_run_and_forbids_bootstrap():
    prompt = spawner_mod.compose_prompt(track="claude", run_id="20260712T091500Z", cycle_budget=4)
    assert "already bootstrapped" in prompt
    assert "do **not** run `bootstrap-track`" in prompt
    assert "20260712T091500Z" in prompt
    assert "show-latest-handoff" in prompt
    assert "finish-delegation" in prompt
    # The stub parses its budget straight out of the prompt.
    assert spawner_mod.compose_prompt(
        track="claude", run_id="r", cycle_budget=7
    ).count("`7`")


def test_child_environment_binds_research_scope():
    env = spawner_mod.child_environment(track="codex", run_id="20260712T091500Z")
    assert env["AUTORESEARCH_SCOPE"] == "research"
    assert env["AUTORESEARCH_TRACK"] == "codex"
    assert env["AUTORESEARCH_RUN_ID"] == "20260712T091500Z"
    assert "AUTORESEARCH_MEMORY_ACCESS" not in env


def test_child_environment_strips_inherited_orchestrator_binding(monkeypatch):
    """A child must never inherit the orchestrator's scope, or it would bind wrong."""
    monkeypatch.setenv("AUTORESEARCH_ORCHESTRATION_ID", "20260712T090000Z")
    monkeypatch.setenv("AUTORESEARCH_MEMORY_ACCESS", "all")
    env = spawner_mod.child_environment(track="claude", run_id="r")
    assert "AUTORESEARCH_ORCHESTRATION_ID" not in env
    assert "AUTORESEARCH_MEMORY_ACCESS" not in env


def test_child_environment_grants_requested_memory_access():
    env = spawner_mod.child_environment(track="claude", run_id="r", memory_access="own")
    assert env["AUTORESEARCH_MEMORY_ACCESS"] == "own"


def test_compute_timeout_scales_with_budget():
    assert spawner_mod.compute_timeout_minutes(1) == 20.0
    # Cycles 0-4 cost 10 min each; the 6th cycle's budget steps up to 15.
    assert spawner_mod.compute_timeout_minutes(5) == 100.0
    assert spawner_mod.compute_timeout_minutes(6) == 130.0


def test_spawn_dry_run_shows_argv_and_env_without_launching(orchestrations_root, tmp_path):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=8
    )
    brief_file = tmp_path / "brief.json"
    write_json(brief_file, {"direction": "Probe constants.", "cycle_budget": 2})

    result = spawner_mod.spawn(
        orch.orchestration_id, brief_path=brief_file, backend_name="stub", dry_run=True
    )
    assert result["status"] == "dry_run"
    rendered = result["plan"].render()
    assert "scripts/stub_subagent.py" in rendered
    assert "AUTORESEARCH_SCOPE=research" in rendered
    assert "AUTORESEARCH_TRACK=claude" in rendered

    # Dry run must not create a delegation or a child run.
    assert load_orchestration(orch.orchestration_id).delegations == ()


def test_spawn_refuses_to_exceed_total_cycle_budget(orchestrations_root, tmp_path):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=3
    )
    brief_file = tmp_path / "brief.json"
    write_json(brief_file, {"direction": "Too big.", "cycle_budget": 4})
    with pytest.raises(ValueError, match="cycles remaining"):
        spawner_mod.spawn(
            orch.orchestration_id, brief_path=brief_file, backend_name="stub", dry_run=True
        )


def test_spawn_rejects_seed_champion_before_creating_anything(orchestrations_root, tmp_path):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=8
    )
    brief_file = tmp_path / "brief.json"
    write_json(
        brief_file,
        {
            "direction": "Seeded.",
            "cycle_budget": 2,
            "seed_champion": {"from_run": "claude/20260712T091500Z", "experiment_id": "e"},
        },
    )
    with pytest.raises(NotImplementedError, match="seed_champion"):
        spawner_mod.spawn(
            orch.orchestration_id, brief_path=brief_file, backend_name="stub", dry_run=True
        )
    assert load_orchestration(orch.orchestration_id).delegations == ()


# ── distress predicates (pure) ──────────────────────────────────────────────


def _distress(**overrides):
    kwargs = dict(
        status="completed",
        exit_code=0,
        agent_summary="A real summary.",
        cycles_used=3,
        cycle_budget=3,
        decisions=["promote", "reject", "reject"],
        champion_model_family="lightgbm",
        calibration_ratio=1.01,
        max_repair_attempts_seen=0,
    )
    kwargs.update(overrides)
    return assess_distress(**kwargs)


def test_healthy_delegation_raises_no_flags():
    assert _distress().active == ()


def test_crashed_flag_on_nonzero_exit():
    assert "crashed" in _distress(status="failed", exit_code=1).active


def test_no_finish_delegation_flag():
    assert "no_finish_delegation" in _distress(agent_summary=None).active
    assert "no_finish_delegation" in _distress(agent_summary="   ").active


def test_all_rejected_flag():
    active = _distress(decisions=["reject", "reject", "reject"]).active
    assert "all_rejected" in active


def test_local_promote_counts_as_progress():
    assert "all_rejected" not in _distress(decisions=["reject", "local_promote"]).active


def test_champion_is_baseline_flag():
    assert "champion_is_baseline" in _distress(champion_model_family="global_mean").active


def test_budget_overrun_on_timeout_and_on_extra_cycles():
    assert "budget_overrun" in _distress(status="timed_out").active
    assert "budget_overrun" in _distress(cycles_used=5, cycle_budget=3).active


def test_repair_exhausted_flag():
    assert "repair_exhausted" in _distress(max_repair_attempts_seen=3).active
    assert "repair_exhausted" not in _distress(max_repair_attempts_seen=2).active


def test_calibration_anomaly_flag():
    assert "calibration_anomaly" in _distress(calibration_ratio=1.5).active
    assert "calibration_anomaly" not in _distress(calibration_ratio=1.05).active
    assert "calibration_anomaly" not in _distress(calibration_ratio=None).active


def test_zero_cycles_is_not_all_rejected():
    """A delegation that never ran a cycle failed some other way; don't double-flag."""
    assert "all_rejected" not in _distress(cycles_used=0, decisions=[]).active


def test_every_active_flag_is_a_declared_flag():
    assessment = _distress(
        status="failed",
        exit_code=2,
        agent_summary=None,
        decisions=["reject"],
        champion_model_family="global_mean",
        calibration_ratio=2.0,
        max_repair_attempts_seen=3,
    )
    assert set(assessment.active) <= set(report_mod.DISTRESS_FLAGS)
    assert assessment.detail


# ── report from a fixture registry ──────────────────────────────────────────


@pytest.fixture
def fixture_child_run(tmp_path, monkeypatch):
    """A minimal but real child registry: one baseline, one rejected challenger."""

    from autoresearch.config import load_config
    from autoresearch.experiment_registry.comparisons import record_comparison
    from autoresearch.experiment_registry.champions import set_official_champion
    from autoresearch.experiment_registry.experiments import record_experiment
    from autoresearch.experiment_registry.research_log import upsert_research_log_entry
    from autoresearch.experiment_registry.schema import init_registry
    from autoresearch.experiment_registry.sessions import upsert_session

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    registry = run_dir / "registry.sqlite"
    init_registry(registry)

    for experiment_id, name, family in (
        ("exp_baseline", "global_mean_baseline", "global_mean"),
        ("exp_challenger", "stub_constant_tweedie_c1", "constant"),
    ):
        record_experiment(
            registry,
            experiment_id=experiment_id,
            experiment_name=name,
            model_family=family,
            target_strategy="direct_pure_premium",
            preprocessing_summary={},
            claim_cap_threshold=None,
            status="completed",
            parent_experiment_id=None,
            config_snapshot_path=run_dir / f"{experiment_id}.toml",
            metrics_path=run_dir / f"{experiment_id}.json",
            artifacts={},
        )

    set_official_champion(
        registry, champion_id="exp_baseline", branch_id="main", reason="init", action="init"
    )
    record_comparison(
        registry,
        comparison_id="cmp1",
        champion_id="exp_baseline",
        challenger_id="exp_challenger",
        paired_summary={"mean_lift": -0.002},
        bootstrap_summary={},
        promotion_decision="pending_llm",
        promotion_rationale="gates not met",
        artifacts={},
    )
    from autoresearch.experiment_registry.comparisons import update_comparison_decision

    update_comparison_decision(
        registry,
        "cmp1",
        decision="reject",
        rationale="No lift.",
        reason_code="inferior",
        decided_at="2026-07-12T10:00:00Z",
    )
    upsert_session(
        registry,
        session_id="s1",
        name="stub_delegation",
        state="completed",
        current_cycle=1,
        max_cycles=2,
        stop_requested=False,
        state_path=run_dir / "state.json",
        summary_path=run_dir / "summary.json",
    )
    upsert_research_log_entry(
        registry,
        session_id="s1",
        cycle=1,
        proposal_id="p1",
        experiment_id="exp_challenger",
        comparison_id="cmp1",
        hypothesis="A constant cannot beat a flat rate.",
        changes="constant recipe",
        outcome="reject",
        metrics={},
        interpretation="Confirmed: no signal.",
        next_step="Stop.",
    )

    config = replace(load_config(), registry_path=registry, artifacts_dir=run_dir)
    monkeypatch.setattr(report_mod, "_child_config", lambda delegation: config)
    return run_dir


def _fixture_delegation(**overrides) -> Delegation:
    kwargs = dict(
        delegation_id="d01",
        brief_path="briefs/d01.json",
        backend="stub",
        track="claude",
        run_id="20260712T091500Z",
        cycle_budget=2,
        status="completed",
        exit_code=0,
        spawned_at="2026-07-12T10:00:00Z",
        ended_at="2026-07-12T10:30:00Z",
        agent_summary="Constant recipes cannot beat a flat baseline.",
    )
    kwargs.update(overrides)
    return Delegation(**kwargs)


def _fixture_orchestration() -> Orchestration:
    return Orchestration(
        orchestration_id="20260712T090000Z",
        dataset="french_motor",
        target_mode="burning_cost",
        created_at="2026-07-12T09:00:00Z",
        total_cycle_budget=8,
    )


def test_report_is_built_from_registry_not_agent_claims(fixture_child_run):
    report = build_report(_fixture_orchestration(), _fixture_delegation())

    assert report["cycles"] == {"budget": 2, "used": 1}
    assert report["champion"]["experiment_id"] == "exp_baseline"
    assert report["champion"]["model_family"] == "global_mean"
    assert report["champion"]["beat_seed_baseline"] is False

    (row,) = report["experiments"]
    assert row["cycle"] == 1
    assert row["decision"] == "reject"
    assert row["reason_code"] == "inferior"
    assert row["lift_vs_champion"] == pytest.approx(-0.002)
    assert row["interpretation"] == "Confirmed: no signal."
    assert row["name"] == "stub_constant_tweedie_c1"

    # The agent's testimony is stored verbatim, and never used as a metric.
    assert report["agent_summary"] == "Constant recipes cannot beat a flat baseline."
    assert report["cost"]["wall_clock_minutes"] == pytest.approx(30.0)


def test_report_flags_baseline_champion_and_all_rejected(fixture_child_run):
    report = build_report(_fixture_orchestration(), _fixture_delegation())
    active = set(report["distress"]["active"])
    assert "champion_is_baseline" in active
    assert "all_rejected" in active
    assert "no_finish_delegation" not in active
    assert report["distress"]["flags"] == list(report_mod.DISTRESS_FLAGS)


def test_report_flags_missing_finish_delegation(fixture_child_run):
    report = build_report(_fixture_orchestration(), _fixture_delegation(agent_summary=None))
    assert "no_finish_delegation" in report["distress"]["active"]


def test_report_flags_crash_on_nonzero_exit(fixture_child_run):
    report = build_report(
        _fixture_orchestration(), _fixture_delegation(status="failed", exit_code=1)
    )
    assert "crashed" in report["distress"]["active"]


def test_collect_report_writes_json(orchestrations_root, fixture_child_run):
    orch = create_orchestration(
        dataset="french_motor", target_mode="burning_cost", total_cycle_budget=8
    )
    delegation = _fixture_delegation()
    path = report_mod.collect_report(orch, delegation)
    assert path.exists()

    import json

    payload = json.loads(path.read_text())
    assert payload["delegation_id"] == "d01"
    assert payload["orchestration_id"] == orch.orchestration_id


def test_max_repair_attempts_seen_scans_run_dir(tmp_path):
    proposal_dir = tmp_path / "iterations" / "001_x"
    proposal_dir.mkdir(parents=True)
    (proposal_dir / "repair_request_1.json").write_text("{}")
    (proposal_dir / "repair_request_3.json").write_text("{}")
    assert report_mod._max_repair_attempts_seen(tmp_path) == 3
    assert report_mod._max_repair_attempts_seen(tmp_path / "iterations") == 3


def test_max_repair_attempts_seen_zero_when_clean(tmp_path):
    assert report_mod._max_repair_attempts_seen(tmp_path) == 0


# ── handoff: brief block appears only for orchestrated runs ─────────────────


def _handoff_for(config) -> str:
    from autoresearch.controller.context import build_llm_context
    from autoresearch.controller.handoff import render_handoff_markdown

    return render_handoff_markdown(config, build_llm_context(config))


def _prepared_run(tmp_path):
    from autoresearch.controller.champion import initialise_official_champion
    from tests.test_handoff import _record_direct
    from tests.test_runner import _make_config

    config = _make_config(tmp_path)
    _record_direct(config)
    initialise_official_champion(config)
    return config


def test_single_agent_handoff_has_no_orchestration_brief(tmp_path):
    """Acceptance criterion 7: a non-orchestrated run is untouched by this feature."""
    config = _prepared_run(tmp_path)
    handoff = _handoff_for(config)
    assert "Orchestration brief" not in handoff
    assert "## Active dataset" in handoff


def test_orchestrated_handoff_carries_the_brief_block(tmp_path, orchestrations_root):
    config = _prepared_run(tmp_path)
    orch = create_orchestration(
        dataset="french_motor", target_mode="burning_cost", total_cycle_budget=8
    )
    brief_file = manifest_mod.brief_path(orch.orchestration_id, "d01")
    write_json(
        brief_file,
        {
            "direction": "Explore frequency×severity structures.",
            "cycle_budget": 4,
            "constraints": ["Stay within approach_family=gbm."],
        },
    )
    save_orchestration(
        add_delegation(
            orch,
            Delegation(
                delegation_id="d01",
                # Stored relative to PROJECT_ROOT, as the spawner writes it.
                brief_path=str(brief_file),
                backend="stub",
                track="claude",
                run_id="20260712T091500Z",
                cycle_budget=4,
            ),
        )
    )
    write_run_backpointer(
        config.artifacts_dir, orchestration_id=orch.orchestration_id, delegation_id="d01"
    )

    handoff = _handoff_for(config)
    assert "## Orchestration brief" in handoff
    assert "Explore frequency×severity structures." in handoff
    assert "Stay within approach_family=gbm." in handoff
    # The brief sits beside the dataset block, not instead of it.
    assert handoff.index("## Orchestration brief") < handoff.index("## Active dataset")


def test_brief_block_survives_the_delta_handoff(tmp_path, orchestrations_root):
    """The brief is binding on every mid-run refresh, so `--delta` must keep it."""
    from autoresearch.controller.context import build_llm_context
    from autoresearch.controller.handoff import render_handoff_markdown

    config = _prepared_run(tmp_path)
    orch = create_orchestration(
        dataset="french_motor", target_mode="burning_cost", total_cycle_budget=4
    )
    brief_file = manifest_mod.brief_path(orch.orchestration_id, "d01")
    write_json(brief_file, {"direction": "Explore freq-sev.", "cycle_budget": 2})
    save_orchestration(
        add_delegation(
            orch,
            Delegation(
                delegation_id="d01",
                brief_path=str(brief_file),
                backend="stub",
                track="claude",
                run_id="20260712T091500Z",
                cycle_budget=2,
            ),
        )
    )
    write_run_backpointer(
        config.artifacts_dir, orchestration_id=orch.orchestration_id, delegation_id="d01"
    )

    delta = render_handoff_markdown(config, build_llm_context(config), delta=True)
    assert "## Orchestration brief" in delta
    assert "Explore freq-sev." in delta
    # The delta handoff still drops the static blocks it always dropped.
    assert "## Proposal quick-start" not in delta


def test_export_handoff_with_brief_refuses_a_child_that_cannot_see_its_brief(
    tmp_path, orchestrations_root
):
    """Regression: the spawner once exported the handoff before writing the brief.

    The renderer degrades silently by design, so the spawner must assert instead —
    launching a paid sub-agent with no direction is worse than failing the spawn.
    """
    config = _prepared_run(tmp_path)
    with pytest.raises(RuntimeError, match="no Orchestration brief block"):
        spawner_mod._export_handoff_with_brief(config, "d01")


def test_export_handoff_with_brief_passes_once_brief_and_delegation_exist(
    tmp_path, orchestrations_root
):
    config = _prepared_run(tmp_path)
    orch = create_orchestration(
        dataset="french_motor", target_mode="burning_cost", total_cycle_budget=4
    )
    brief_file = manifest_mod.brief_path(orch.orchestration_id, "d01")
    write_json(brief_file, {"direction": "Probe constants.", "cycle_budget": 2})
    save_orchestration(
        add_delegation(
            orch,
            Delegation(
                delegation_id="d01",
                brief_path=str(brief_file),
                backend="stub",
                track="claude",
                run_id="20260712T091500Z",
                cycle_budget=2,
            ),
        )
    )
    write_run_backpointer(
        config.artifacts_dir, orchestration_id=orch.orchestration_id, delegation_id="d01"
    )
    spawner_mod._export_handoff_with_brief(config, "d01")  # must not raise


def test_orchestrated_handoff_degrades_gracefully_when_brief_is_unreadable(
    tmp_path, orchestrations_root
):
    """A thinner handoff beats no handoff: an unreadable brief must not be fatal."""
    config = _prepared_run(tmp_path)
    write_run_backpointer(
        config.artifacts_dir, orchestration_id="20260712T090000Z", delegation_id="d01"
    )
    handoff = _handoff_for(config)
    assert "Orchestration brief" not in handoff
    assert "## Active dataset" in handoff
