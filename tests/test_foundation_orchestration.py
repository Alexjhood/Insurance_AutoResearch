"""Phase 2 — orchestrated foundation models (TabPFN, API backend).

Covers the brief field, the spawner forwarding it to bootstrap, the child-env
hygiene, the spawn preflight (missing extra / missing api token), and the
playoff replay gating. Nothing here launches a sub-agent, makes a paid TabPFN
call, or reads holdout data — the transport boundary is always faked.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from autoresearch.orchestration import backends as backends_mod
from autoresearch.orchestration import playoff as playoff_mod
from autoresearch.orchestration import spawner as spawner_mod
from autoresearch.orchestration.backends import Backend, preflight_foundation_models
from autoresearch.orchestration.brief import Brief, render_brief_block, validate_brief


# ── brief field ──────────────────────────────────────────────────────────────

def test_brief_foundation_models_defaults_false():
    brief = validate_brief({"direction": "Try TabPFN on severity.", "cycle_budget": 3})
    assert brief.foundation_models is False
    assert brief.to_dict()["foundation_models"] is False


def test_brief_foundation_models_accepted_and_round_trips():
    brief = validate_brief(
        {"direction": "x", "cycle_budget": 2, "foundation_models": True}
    )
    assert brief.foundation_models is True
    assert validate_brief(brief.to_dict()).foundation_models is True


def test_brief_foundation_models_rejects_non_bool():
    with pytest.raises(ValueError, match="foundation_models must be a boolean"):
        validate_brief({"direction": "x", "cycle_budget": 1, "foundation_models": "yes"})


def test_brief_block_states_foundation_enabled():
    brief = Brief(direction="Probe TabPFN.", cycle_budget=2, foundation_models=True)
    block = "\n".join(render_brief_block(brief, cycle_budget=2, delegation_id="d01"))
    assert "Foundation estimators enabled" in block
    assert "tabpfn" in block
    # And the credit/budget caution is carried, per the build prompt.
    assert "credit" in block.lower()


def test_brief_block_omits_foundation_note_when_off():
    brief = Brief(direction="Plain GBM sweep.", cycle_budget=2)
    block = "\n".join(render_brief_block(brief, cycle_budget=2, delegation_id="d01"))
    assert "Foundation estimators enabled" not in block


# ── spawner forwarding ───────────────────────────────────────────────────────

def _fake_child_config(tmp_path, run_id="20260710T160001Z"):
    from autoresearch.config import load_config

    config = replace(
        load_config(),
        artifacts_dir=tmp_path / "child",
        track_id="claude",
        run_id=run_id,
    )
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return config


@pytest.mark.parametrize("foundation", [True, False])
def test_bootstrap_child_run_forwards_foundation_flag(tmp_path, monkeypatch, foundation):
    """The brief's foundation_models reaches bootstrap_track, which sets the run
    manifest flag the child's session then honours."""
    config = _fake_child_config(tmp_path)
    backend = Backend(
        name="fixture", tool="stub", command=("stub",), prompt_via="stdin",
        track="claude", model_provider="fixture", model_name="fixture",
    )
    from types import SimpleNamespace

    orch = SimpleNamespace(
        dataset="french_motor", target_mode="burning_cost",
        orchestration_id="20260710T160000Z",
    )
    captured: dict = {}
    monkeypatch.setattr(spawner_mod, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(
        "autoresearch.bootstrap.bootstrap_track",
        lambda cfg, **kwargs: captured.update(kwargs),
    )

    brief = Brief(direction="x", cycle_budget=2, foundation_models=foundation)
    spawner_mod._bootstrap_child_run(
        orch, backend=backend, brief=brief, delegation_id="d01"
    )

    assert captured["enable_foundation_models"] is foundation
    assert captured["default_max_cycles"] == 2


# ── child environment hygiene ────────────────────────────────────────────────

def test_child_env_strips_foundation_flag_but_keeps_token(monkeypatch):
    """A non-opted child must not inherit the orchestrator's in-process
    AUTORESEARCH_FOUNDATION_MODELS; but TABPFN_TOKEN must reach an opted child."""
    monkeypatch.setenv("AUTORESEARCH_FOUNDATION_MODELS", "1")
    monkeypatch.setenv("TABPFN_TOKEN", "priorlabs-key")
    env = spawner_mod.child_environment(track="claude", run_id="r")
    assert "AUTORESEARCH_FOUNDATION_MODELS" not in env
    assert env["TABPFN_TOKEN"] == "priorlabs-key"


# ── preflight ────────────────────────────────────────────────────────────────

def test_preflight_foundation_fails_when_extra_missing(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: False
    )
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabfm_available", lambda: False
    )
    with pytest.raises(RuntimeError, match="no foundation extra is importable"):
        preflight_foundation_models()


def test_preflight_foundation_accepts_tabfm_only(monkeypatch):
    """A TabFM-only environment (no TabPFN extra) passes the extra check."""
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: False
    )
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabfm_available", lambda: True
    )
    monkeypatch.setenv("AUTORESEARCH_TABPFN_BACKEND", "local")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    preflight_foundation_models()  # no raise: TabFM covers the foundation extra


def test_preflight_foundation_fails_on_api_without_token(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: True
    )
    monkeypatch.setenv("AUTORESEARCH_TABPFN_BACKEND", "api")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TABPFN_TOKEN is unset"):
        preflight_foundation_models()


def test_preflight_foundation_passes_api_with_token(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: True
    )
    monkeypatch.setenv("AUTORESEARCH_TABPFN_BACKEND", "api")
    monkeypatch.setenv("TABPFN_TOKEN", "priorlabs-key")
    preflight_foundation_models()  # no raise


def test_preflight_foundation_local_backend_needs_no_token(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: True
    )
    monkeypatch.setenv("AUTORESEARCH_TABPFN_BACKEND", "local")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    preflight_foundation_models()  # local backend never needs the api token


def test_spawn_runs_foundation_preflight_only_when_opted(monkeypatch, tmp_path):
    """spawn() invokes the foundation preflight iff the brief opted in."""
    from autoresearch.orchestration.manifest import create_orchestration
    from autoresearch.orchestration import manifest as manifest_mod
    from autoresearch.utils.io import write_json

    monkeypatch.setattr(manifest_mod, "ORCHESTRATIONS_DIR", tmp_path / "orch")
    orch = create_orchestration(
        dataset="french_motor", target_mode="burning_cost", total_cycle_budget=4
    )
    calls: list[bool] = []
    monkeypatch.setattr(backends_mod, "preflight_backend", lambda b: None)
    monkeypatch.setattr(
        backends_mod, "preflight_foundation_models", lambda: calls.append(True)
    )
    # Fail fast right after preflight so we never bootstrap/launch anything.
    monkeypatch.setattr(
        spawner_mod, "manifest_lock",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop after preflight")),
    )

    brief_off = tmp_path / "off.json"
    write_json(brief_off, {"direction": "plain", "cycle_budget": 1})
    with pytest.raises(RuntimeError, match="stop after preflight"):
        spawner_mod.spawn(orch.orchestration_id, brief_path=brief_off, backend_name="stub")
    assert calls == []  # not opted in → no foundation preflight

    brief_on = tmp_path / "on.json"
    write_json(brief_on, {"direction": "tabpfn", "cycle_budget": 1, "foundation_models": True})
    with pytest.raises(RuntimeError, match="stop after preflight"):
        spawner_mod.spawn(orch.orchestration_id, brief_path=brief_on, backend_name="stub")
    assert calls == [True]  # opted in → foundation preflight ran


# ── playoff replay gating ────────────────────────────────────────────────────

def test_recipe_foundation_estimators_extraction():
    direct = {"recipe": {"structure": "direct", "estimator": "tabpfn"}}
    assert playoff_mod._recipe_foundation_estimators(direct) == {"tabpfn"}
    gbm = {"recipe": {"structure": "direct", "estimator": "lightgbm"}}
    assert playoff_mod._recipe_foundation_estimators(gbm) == set()
    freqsev = {"recipe": {"structure": "frequency_severity", "stages": {
        "frequency": {"estimator": "lightgbm"}, "severity": {"estimator": "tabpfn"}}}}
    assert playoff_mod._recipe_foundation_estimators(freqsev) == {"tabpfn"}
    script = {"script_path": "model.py"}
    assert playoff_mod._recipe_foundation_estimators(script) == set()


def test_ensure_foundation_support_raises_clear_message_when_extra_missing(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: False
    )
    source = playoff_mod.ReplaySource(track="claude", run_id="r", experiment_id="e")
    dest = SimpleNamespace(track_id="claude", run_id="c")
    with pytest.raises(RuntimeError, match=r"\[foundation\] extra"):
        playoff_mod._ensure_foundation_support(
            {"recipe": {"estimator": "tabpfn"}}, source, dest
        )
    # And the misleading generic error is avoided.
    try:
        playoff_mod._ensure_foundation_support(
            {"recipe": {"estimator": "tabpfn"}}, source, dest
        )
    except RuntimeError as exc:
        assert "unknown estimator" not in str(exc).lower()


def test_ensure_foundation_support_registers_when_available(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", lambda: True
    )
    registered: list[bool] = []
    monkeypatch.setattr(
        "autoresearch.models.recipe.enable_foundation_models",
        lambda: registered.append(True) or [],
    )
    source = playoff_mod.ReplaySource(track="claude", run_id="r", experiment_id="e")
    dest = SimpleNamespace(track_id="claude", run_id="c")
    playoff_mod._ensure_foundation_support(
        {"recipe": {"estimator": "tabpfn"}}, source, dest
    )
    assert registered == [True]


def test_ensure_foundation_support_noop_for_non_foundation(monkeypatch):
    from types import SimpleNamespace

    def _boom():
        raise AssertionError("tabpfn_available should not be consulted for a GBM recipe")

    monkeypatch.setattr(
        "autoresearch.models.recipe.foundation.tabpfn_available", _boom
    )
    source = playoff_mod.ReplaySource(track="claude", run_id="r", experiment_id="e")
    dest = SimpleNamespace(track_id="claude", run_id="c")
    playoff_mod._ensure_foundation_support(
        {"recipe": {"estimator": "lightgbm"}}, source, dest
    )  # no raise, no availability probe


def test_any_finalist_needs_foundation_reads_source_manifest(monkeypatch):
    from types import SimpleNamespace

    finalists = [
        SimpleNamespace(source=SimpleNamespace(track="claude", run_id="r1")),
        SimpleNamespace(source=SimpleNamespace(track="claude", run_id="r2")),
    ]
    monkeypatch.setattr(
        playoff_mod, "load_config",
        lambda **kwargs: SimpleNamespace(run_id=kwargs["run_id"]),
    )
    manifests = {"r1": {"foundation_models": False}, "r2": {"foundation_models": True}}
    monkeypatch.setattr(
        "autoresearch.bootstrap.read_run_manifest",
        lambda cfg: manifests[cfg.run_id],
    )
    assert playoff_mod._any_finalist_needs_foundation(finalists) is True

    manifests["r2"] = {"foundation_models": False}
    assert playoff_mod._any_finalist_needs_foundation(finalists) is False
