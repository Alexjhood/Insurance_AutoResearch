import pytest

from autoresearch import cli


def test_fresh_bootstrap_requires_explicit_dataset(monkeypatch, capsys) -> None:
    def unexpected_load_config(*args, **kwargs):
        raise AssertionError("load_config must not run before dataset validation")

    monkeypatch.setattr(cli, "load_config", unexpected_load_config)

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--track",
                "codex",
                "--new-run",
                "bootstrap-track",
                "--model-provider",
                "openai",
                "--model-name",
                "test-model",
            ]
        )

    assert exc.value.code == 2
    assert "requires --dataset <name>" in capsys.readouterr().err


def test_existing_run_bootstrap_may_use_pinned_dataset(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(cli, "load_config", lambda *args, **kwargs: sentinel)
    monkeypatch.setitem(cli.COMMANDS, "bootstrap-track", lambda config, args: 0)
    monkeypatch.setattr(
        "autoresearch.bootstrap.apply_foundation_models_gate", lambda config: []
    )
    assert (
        cli.main(
            [
                "--track",
                "codex",
                "--run-id",
                "20260710T221244Z",
                "bootstrap-track",
                "--model-provider",
                "openai",
                "--model-name",
                "test-model",
            ]
        )
        == 0
    )


def test_record_decision_strips_inherited_milestone_token(monkeypatch) -> None:
    monkeypatch.setenv("AUTORESEARCH_MILESTONE_TOKEN", "operator-secret")

    def fake_record_decision(config, comparison_id, **kwargs):
        import os

        assert "AUTORESEARCH_MILESTONE_TOKEN" not in os.environ
        return {
            "decision": "reject",
            "rationale": "test",
            "reason_code": "inferior",
            "decided_at": "2026-07-11T00:00:00Z",
            "guardrail_result": {"passed": True},
        }

    monkeypatch.setattr(cli, "record_decision", fake_record_decision)
    monkeypatch.setattr(
        "autoresearch.experiment_registry.champions.get_official_champion",
        lambda path: None,
    )
    monkeypatch.setattr(
        "autoresearch.controller.context.build_llm_context", lambda config: {}
    )
    monkeypatch.setattr(
        "autoresearch.controller.handoff._next_supervised_command",
        lambda config, context: "done",
    )

    args = type(
        "Args",
        (),
        {
            "comparison_id": "cmp-1",
            "decision": "reject",
            "rationale": "test",
            "reason_code": "inferior",
            "interpretation": "test",
            "next_step": "done",
        },
    )()
    config = type("Config", (), {"registry_path": None})()

    assert cli._cmd_record_decision(config, args) == 0
