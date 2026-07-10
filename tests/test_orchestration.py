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
from types import SimpleNamespace
import sys
import threading
import time

import pytest

from autoresearch.orchestration import adapters as adapters_mod
from autoresearch.orchestration import backends as backends_mod
from autoresearch.orchestration import manifest as manifest_mod
from autoresearch.orchestration import monitor as monitor_mod
from autoresearch.orchestration import report as report_mod
from autoresearch.orchestration import spawner as spawner_mod
from autoresearch.orchestration.backends import Backend, get_backend, load_backends
from autoresearch.orchestration.brief import Brief, render_brief_block, validate_brief
from autoresearch.orchestration.manifest import (
    Delegation,
    Orchestration,
    add_delegation,
    create_orchestration,
    exit_status_path,
    load_orchestration,
    manifest_lock,
    read_run_backpointer,
    save_orchestration,
    update_delegation,
    write_run_backpointer,
)
from autoresearch.orchestration.report import assess_distress, build_report
from autoresearch.utils.io import read_json, write_json


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
        clean_exit=True,
        tool_usage={"input_tokens": 12},
    )
    save_orchestration(add_delegation(orch, delegation))

    reloaded = load_orchestration(orch.orchestration_id)
    assert reloaded.to_dict() == add_delegation(orch, delegation).to_dict()
    assert reloaded.delegation("d01").backend == "stub"
    assert reloaded.delegation("d01").clean_exit is True
    assert reloaded.delegation("d01").tool_usage == {"input_tokens": 12}
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


def test_manifest_lock_serialises_parallel_spawners(orchestrations_root):
    from concurrent.futures import ThreadPoolExecutor

    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=4
    )
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def enter() -> None:
        nonlocal active, max_active
        with manifest_lock(orch.orchestration_id):
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: enter(), range(2)))

    assert max_active == 1


def test_run_backpointer_round_trip(tmp_path):
    assert read_run_backpointer(tmp_path) is None
    write_run_backpointer(tmp_path, orchestration_id="20260712T090000Z", delegation_id="d01")
    assert read_run_backpointer(tmp_path) == ("20260712T090000Z", "d01")


def test_run_backpointer_preserves_existing_manifest_keys(tmp_path):
    write_json(tmp_path / "run_manifest.json", {"dataset": "porto_seguro", "default_max_cycles": 4})
    write_run_backpointer(
        tmp_path,
        orchestration_id="20260712T090000Z",
        delegation_id="d02",
        target_mode="burning_cost",
    )
    import json

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["dataset"] == "porto_seguro"
    assert manifest["default_max_cycles"] == 4
    assert manifest["delegation_id"] == "d02"
    assert manifest["target_mode"] == "burning_cost"


def test_target_mode_manifest_read_is_gated_on_orchestration(tmp_path):
    from autoresearch.config import _read_orchestration_target_mode

    path = tmp_path / "run_manifest.json"
    write_json(path, {"target_mode": "claim_frequency"})
    assert _read_orchestration_target_mode(path) is None
    write_json(
        path,
        {
            "orchestration_id": "20260712T090000Z",
            "target_mode": "claim_frequency",
        },
    )
    assert _read_orchestration_target_mode(path) == "claim_frequency"


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


def test_codex_backend_pins_verified_headless_flags_and_metadata():
    backend = load_backends()["codex-gpt-5-5-medium"]
    command = backend.render_command(prompt="the prompt travels over stdin")

    assert command == (
        "codex",
        "-a",
        "never",
        "exec",
        "--json",
        "--sandbox",
        "workspace-write",
        "--skip-git-repo-check",
        "-C",
        ".",
        "-m",
        "gpt-5.5",
        "-c",
        'model_reasoning_effort="medium"',
        "--color",
        "never",
        "-",
    )
    assert "the prompt travels over stdin" not in command
    assert backend.prompt_via == "stdin"
    assert backend.track == "codex"
    assert backend.model_provider == "openai"
    assert backend.model_name == "gpt-5.5-medium"
    assert backend.tier == "mid"
    # Unvalidated against a real endpoint yet: enters as trial like every new
    # backend and earns default from its scorecard (design §4.7 lifecycle).
    assert backend.status == "trial"
    assert "diagnostic_probes" in backend.good_for


def test_codex_stub_uses_codex_adapter_and_track_without_a_real_cli():
    backend = load_backends()["stub-codex"]
    assert backend.tool == "codex"
    assert backend.track == "codex"
    assert backend.command == ("python3", "scripts/stub_codex_subagent.py")


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (("codex", "exec", "--sandbox", "workspace-write", "-"), "require --json"),
        (
            (
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "workspace-write",
                "--max-turns",
                "5",
                "-c",
                'model_reasoning_effort="medium"',
                "-",
            ),
            "no --max-turns",
        ),
        (
            (
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "danger-full-access",
                "-c",
                'model_reasoning_effort="medium"',
                "-",
            ),
            "workspace-write",
        ),
        (
            (
                "codex",
                "exec",
                "--json",
                "--sandbox",
                "workspace-write",
                "-c",
                'model_reasoning_effort="extreme"',
                "-",
            ),
            "model_reasoning_effort",
        ),
    ],
)
def test_invalid_codex_headless_contract_fails_at_load_time(command, message):
    with pytest.raises(ValueError, match=message):
        Backend(
            name="bad-codex",
            tool="codex",
            command=command,
            prompt_via="stdin",
            track="codex",
            model_provider="openai",
            model_name="m",
        )


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
        tool="opencode",
        command=("opencode", "--max-turns", "{max_turns}", "{prompt}"),
        prompt_via="argv",
        track="opencode",
        model_provider="opencode",
        model_name="m",
        max_turns=42,
    )
    argv = backend.render_command(prompt="do science")
    assert argv == ("opencode", "--max-turns", "42", "do science")


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


def test_child_environment_strips_holdout_token_and_pytest_gate_skip(monkeypatch):
    """A sub-agent must not inherit holdout access or a disabled pytest gate."""
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "secret")
    monkeypatch.setenv("AUTORESEARCH_SKIP_PYTEST_GATE", "1")
    env = spawner_mod.child_environment(track="claude", run_id="r")
    assert "AUTORESEARCH_MILESTONE_TOKEN" not in env
    assert "AUTORESEARCH_SKIP_PYTEST_GATE" not in env


def test_child_environment_grants_requested_memory_access():
    env = spawner_mod.child_environment(track="claude", run_id="r", memory_access="own")
    assert env["AUTORESEARCH_MEMORY_ACCESS"] == "own"


def test_should_refund_budget_only_for_zero_work_environment_failures():
    from autoresearch.orchestration.report import should_refund_budget

    crashed_no_work = {
        "status": "failed",
        "cycles": {"budget": 4, "used": 0},
        "cost": {"llm_usage": {"calls": 0}},
    }
    assert should_refund_budget(crashed_no_work)
    # Any evidence of work — cycles, LLM calls, backend tokens — means no refund.
    assert not should_refund_budget({**crashed_no_work, "cycles": {"budget": 4, "used": 1}})
    assert not should_refund_budget({**crashed_no_work, "cost": {"llm_usage": {"calls": 3}}})
    assert not should_refund_budget(
        {**crashed_no_work, "cost": {"llm_usage": {"backend": {"input_tokens": 10}}}}
    )
    # Completed-but-idle and taken-over delegations keep their budget committed.
    assert not should_refund_budget({**crashed_no_work, "status": "completed"})
    assert not should_refund_budget({**crashed_no_work, "taken_over": True})


def test_refunded_delegation_does_not_count_against_the_cycle_budget():
    orch = Orchestration(
        orchestration_id="20260710T160000Z",
        dataset="porto_seguro",
        target_mode="claim_incidence",
        created_at="2026-07-10T16:00:00Z",
        total_cycle_budget=10,
        delegations=(
            _running_delegation(status="failed", pid=None),
            replace(
                _running_delegation(status="failed", pid=None),
                delegation_id="d02",
                budget_refunded=True,
            ),
        ),
    )
    assert orch.cycles_committed == orch.delegations[0].cycle_budget


def test_preflight_refuses_missing_executable_and_unwritable_state_dir(monkeypatch, tmp_path):
    from autoresearch.orchestration.backends import load_backends, preflight_backend

    stub = load_backends()["stub"]
    claude = load_backends()["claude-sonnet-low"]

    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="not on PATH"):
        preflight_backend(stub)

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/fake")
    unwritable = tmp_path / "home"
    unwritable.mkdir()
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: unwritable))
    unwritable.chmod(0o500)
    try:
        with pytest.raises(RuntimeError, match="writable home state"):
            preflight_backend(claude)
    finally:
        unwritable.chmod(0o700)
    # The stub tool has no home-state requirement, so PATH is its only check.
    preflight_backend(stub)


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


def test_spawn_dry_run_accepts_seed_champion_without_creating_anything(
    orchestrations_root, tmp_path
):
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
    result = spawner_mod.spawn(
        orch.orchestration_id, brief_path=brief_file, backend_name="stub", dry_run=True
    )
    assert result["status"] == "dry_run"
    assert load_orchestration(orch.orchestration_id).delegations == ()


def test_detached_wrapper_records_exit_code_and_output(
    orchestrations_root, tmp_path, monkeypatch
):
    oid = "20260710T120000Z"
    manifest_mod.orchestration_dir(oid).mkdir(parents=True)
    prompt_file = manifest_mod.prompt_path(oid, "d01")
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("wrapper prompt", encoding="utf-8")
    log_file = tmp_path / "child.log"
    backend = Backend(
        name="test-wrapper",
        tool="stub",
        command=(sys.executable, "-c", "import sys; print(sys.stdin.read())"),
        prompt_via="stdin",
        track="claude",
        model_provider="test",
        model_name="test",
    )
    plan = spawner_mod.SpawnPlan(
        orchestration_id=oid,
        delegation_id="d01",
        backend=backend,
        track="claude",
        run_id="20260710T120100Z",
        cycle_budget=1,
        command=backend.command,
        env=dict(spawner_mod.os.environ),
        prompt="wrapper prompt",
        timeout_minutes=20,
    )
    delegation = Delegation(
        delegation_id="d01",
        brief_path="briefs/d01.json",
        backend=backend.name,
        track="claude",
        run_id=plan.run_id,
        cycle_budget=1,
        log_path=str(log_file),
    )

    process = spawner_mod._launch(plan, delegation)

    assert process.wait(timeout=10) == 0
    assert "wrapper prompt" in log_file.read_text(encoding="utf-8")
    status = manifest_mod.read_json(exit_status_path(oid, "d01"))
    assert status["exit_code"] == 0


def test_codex_wrapper_transports_stdin_and_records_usage(
    orchestrations_root, tmp_path
):
    import json

    oid = "20260710T120000Z"
    manifest_mod.orchestration_dir(oid).mkdir(parents=True)
    prompt_file = manifest_mod.prompt_path(oid, "d01")
    prompt_file.parent.mkdir(parents=True)
    prompt_file.write_text("wrapper prompt", encoding="utf-8")
    log_file = tmp_path / "codex-child.log"
    script = (
        "import json,sys; p=sys.stdin.read(); "
        "print(json.dumps({'type':'item.completed','item':{'text':p}})); "
        "print(json.dumps({'type':'turn.completed','usage':"
        "{'input_tokens':11,'output_tokens':3}}))"
    )
    backend = Backend(
        name="stub-codex-wrapper",
        tool="codex",
        command=(sys.executable, "-c", script),
        prompt_via="stdin",
        track="codex",
        model_provider="stub",
        model_name="stub",
    )
    plan = spawner_mod.SpawnPlan(
        orchestration_id=oid,
        delegation_id="d01",
        backend=backend,
        track="codex",
        run_id="20260710T120100Z",
        cycle_budget=1,
        command=backend.command,
        env=dict(spawner_mod.os.environ),
        prompt="wrapper prompt",
        timeout_minutes=20,
    )
    delegation = Delegation(
        delegation_id="d01",
        brief_path="briefs/d01.json",
        backend=backend.name,
        track="codex",
        run_id=plan.run_id,
        cycle_budget=1,
        log_path=str(log_file),
    )

    process = spawner_mod._launch(plan, delegation)

    assert process.wait(timeout=10) == 0
    events = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert events[0]["item"]["text"] == "wrapper prompt"
    status = manifest_mod.read_json(exit_status_path(oid, "d01"))
    assert status["exit_code"] == 0
    assert status["clean_exit"] is True
    assert status["usage"] == {
        "input_tokens": 11,
        "output_tokens": 3,
        "completed_turns": 1,
    }


def test_codex_usage_parser_aggregates_turns_and_tolerates_stderr(tmp_path):
    import json

    log = tmp_path / "codex.jsonl"
    log.write_text(
        "warning from stderr\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "details": {"reasoning_tokens": 1},
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "details": {"reasoning_tokens": 2},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    observed = adapters_mod.inspect_backend_exit("codex", log, exit_code=0)

    assert observed.clean_exit is True
    assert observed.terminal_event == "turn.completed"
    assert observed.malformed_lines == 1
    assert observed.usage == {
        "input_tokens": 17,
        "output_tokens": 5,
        "details": {"reasoning_tokens": 3},
        "completed_turns": 2,
    }


@pytest.mark.parametrize(
    ("exit_code", "event", "clean"),
    [
        (1, '{"type":"turn.completed","usage":{}}', False),
        (0, '{"type":"turn.failed"}', False),
        (0, '{"type":"item.completed"}', False),
    ],
)
def test_codex_clean_exit_requires_zero_code_and_completed_turn(
    tmp_path, exit_code, event, clean
):
    log = tmp_path / "codex.jsonl"
    log.write_text(event + "\n", encoding="utf-8")
    assert (
        adapters_mod.inspect_backend_exit("codex", log, exit_code=exit_code).clean_exit
        is clean
    )


def test_respawn_continue_run_reuses_child_and_records_lineage(
    orchestrations_root, tmp_path, monkeypatch
):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=5
    )
    source = Delegation(
        delegation_id="d01",
        brief_path="briefs/d01.json",
        backend="stub",
        track="claude",
        run_id="20260710T120100Z",
        cycle_budget=2,
        status="completed",
        pid=None,
    )
    save_orchestration(add_delegation(orch, source))
    brief_file = tmp_path / "revised.json"
    write_json(brief_file, {"direction": "Continue the useful line.", "cycle_budget": 3})
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child_config = SimpleNamespace(artifacts_dir=child_dir, registry_path=tmp_path / "registry.sqlite")

    monkeypatch.setattr(spawner_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(report_mod, "_child_config", lambda delegation: child_config)
    monkeypatch.setattr(report_mod, "_cycles_used", lambda registry: 2)
    monkeypatch.setattr(spawner_mod, "_export_handoff_with_brief", lambda config, did: None)
    monkeypatch.setattr(spawner_mod, "_launch", lambda plan, delegation: SimpleNamespace(pid=4321))

    result = spawner_mod.respawn(
        orch.orchestration_id,
        delegation_id="d01",
        brief_path=brief_file,
        continue_run=True,
        wait=False,
    )

    assert result["delegation_id"] == "d02"
    continued = load_orchestration(orch.orchestration_id).delegation("d02")
    assert continued.run_id == source.run_id
    assert continued.respawn_of == "d01"
    assert continued.continue_run is True
    assert continued.cycles_at_start == 2
    assert read_run_backpointer(child_dir) == (orch.orchestration_id, "d02")
    # The continuation re-pins the run's cycle cap to the new brief's budget;
    # otherwise the new session would inherit d01's smaller pin and stall early.
    assert read_json(child_dir / "run_manifest.json")["default_max_cycles"] == 3


def test_respawn_new_run_passes_lineage_to_spawn(
    orchestrations_root, tmp_path, monkeypatch
):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=4
    )
    source = _running_delegation(status="completed", pid=None)
    save_orchestration(add_delegation(orch, source))
    captured = {}

    def fake_spawn(orchestration_id, **kwargs):
        captured.update(kwargs)
        return {"status": "running", "delegation_id": "d02"}

    monkeypatch.setattr(spawner_mod, "spawn", fake_spawn)

    spawner_mod.respawn(
        orch.orchestration_id,
        delegation_id="d01",
        brief_path=tmp_path / "brief.json",
        wait=False,
    )

    assert captured["backend_name"] == "stub"
    assert captured["respawn_of"] == "d01"


def test_respawn_seed_champion_resolves_campaign_delegation(
    orchestrations_root, tmp_path, monkeypatch
):
    from autoresearch.orchestration import playoff as playoff_mod

    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=4
    )
    source = _running_delegation(status="completed", pid=None)
    save_orchestration(add_delegation(orch, source))
    brief_path = tmp_path / "brief.json"
    write_json(brief_path, {"direction": "Continue the winner.", "cycle_budget": 1})
    monkeypatch.setattr(
        playoff_mod,
        "replay_source_for_delegation",
        lambda current, delegation_id: playoff_mod.ReplaySource(
            track="claude",
            run_id="20260710T120100Z",
            experiment_id="winner",
            delegation_id=delegation_id,
            orchestration_id=current.orchestration_id,
        ),
    )
    captured = {}

    def fake_spawn(orchestration_id, **kwargs):
        captured.update(kwargs)
        return {"status": "running", "delegation_id": "d02"}

    monkeypatch.setattr(spawner_mod, "spawn", fake_spawn)

    spawner_mod.respawn(
        orch.orchestration_id,
        delegation_id="d01",
        brief_path=brief_path,
        seed_champion="from:d01",
        wait=False,
    )

    seed = captured["seed_champion_override"]
    assert seed.from_run == "claude/20260710T120100Z"
    assert seed.experiment_id == "winner"


def test_respawn_requires_timed_out_process_to_be_killed(
    orchestrations_root, tmp_path, monkeypatch
):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=4
    )
    source = _running_delegation(status="timed_out")
    orch = add_delegation(orch, source)
    save_orchestration(orch)
    monkeypatch.setattr(monitor_mod, "refresh_orchestration", lambda oid: orch)
    monkeypatch.setattr(monitor_mod, "process_is_alive", lambda pid: True)

    with pytest.raises(ValueError, match="kill it first"):
        spawner_mod.respawn(
            orch.orchestration_id,
            delegation_id="d01",
            brief_path=tmp_path / "brief.json",
        )


# ── detached monitor ────────────────────────────────────────────────────────


def _running_delegation(**overrides) -> Delegation:
    values = dict(
        delegation_id="d01",
        brief_path="briefs/d01.json",
        backend="stub",
        track="claude",
        run_id="20260710T120100Z",
        cycle_budget=1,
        status="running",
        pid=1234,
        spawned_at="2026-07-10T12:00:00Z",
        timeout_minutes=20,
    )
    values.update(overrides)
    return Delegation(**values)


def test_monitor_observes_clean_detached_exit(orchestrations_root, monkeypatch):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=2
    )
    save_orchestration(add_delegation(orch, _running_delegation()))
    write_json(
        exit_status_path(orch.orchestration_id, "d01"),
        {"exit_code": 0, "ended_at": "2026-07-10T12:05:00Z"},
    )
    monkeypatch.setattr(monitor_mod, "process_is_alive", lambda pid: False)

    refreshed = monitor_mod.refresh_orchestration(orch.orchestration_id)

    completed = refreshed.delegation("d01")
    assert completed.status == "completed"
    assert completed.exit_code == 0
    assert completed.ended_at == "2026-07-10T12:05:00Z"


def test_monitor_rejects_zero_exit_without_a_clean_codex_terminal_event(
    orchestrations_root, monkeypatch
):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=2
    )
    save_orchestration(add_delegation(orch, _running_delegation(track="codex")))
    write_json(
        exit_status_path(orch.orchestration_id, "d01"),
        {
            "exit_code": 0,
            "clean_exit": False,
            "usage": {"input_tokens": 9},
            "ended_at": "2026-07-10T12:05:00Z",
        },
    )
    monkeypatch.setattr(monitor_mod, "process_is_alive", lambda pid: False)

    refreshed = monitor_mod.refresh_orchestration(orch.orchestration_id)

    failed = refreshed.delegation("d01")
    assert failed.status == "failed"
    assert failed.exit_code == 0
    assert failed.clean_exit is False
    assert failed.tool_usage == {"input_tokens": 9}


def test_monitor_marks_timeout_without_killing(orchestrations_root, monkeypatch):
    from datetime import datetime, timezone

    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=2
    )
    save_orchestration(add_delegation(orch, _running_delegation()))
    monkeypatch.setattr(monitor_mod, "process_is_alive", lambda pid: True)
    kill_calls = []
    monkeypatch.setattr(monitor_mod.os, "killpg", lambda *args: kill_calls.append(args))

    refreshed = monitor_mod.refresh_orchestration(
        orch.orchestration_id,
        now=datetime(2026, 7, 10, 12, 21, tzinfo=timezone.utc),
    )

    assert refreshed.delegation("d01").status == "timed_out"
    assert kill_calls == []


def test_kill_terminates_process_group_and_records_status(orchestrations_root, monkeypatch):
    orch = create_orchestration(
        dataset="porto_seguro", target_mode="claim_incidence", total_cycle_budget=2
    )
    save_orchestration(add_delegation(orch, _running_delegation()))
    monkeypatch.setattr(monitor_mod, "process_is_alive", lambda pid: True)
    calls = []
    monkeypatch.setattr(monitor_mod.os, "killpg", lambda pid, sig: calls.append((pid, sig)))

    killed = monitor_mod.kill_delegation(orch.orchestration_id, "d01")

    assert calls and calls[0][0] == 1234
    assert killed.status == "killed"
    assert load_orchestration(orch.orchestration_id).delegation("d01").status == "killed"


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


def test_report_includes_best_effort_backend_usage(fixture_child_run):
    delegation = _fixture_delegation(
        tool_usage={"input_tokens": 21, "output_tokens": 8, "completed_turns": 1}
    )
    report = build_report(_fixture_orchestration(), delegation)
    assert report["cost"]["llm_usage"]["backend"] == delegation.tool_usage


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


def test_continued_report_excludes_cycles_before_its_offset(fixture_child_run):
    delegation = _fixture_delegation(
        delegation_id="d02",
        cycle_budget=1,
        respawn_of="d01",
        continue_run=True,
        cycles_at_start=1,
    )

    report = build_report(_fixture_orchestration(), delegation)

    assert report["cycles"] == {"budget": 1, "used": 0}
    assert report["experiments"] == []
    assert report["respawn_of"] == "d01"


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
