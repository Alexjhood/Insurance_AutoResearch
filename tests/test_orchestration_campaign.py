"""Phase 5 — campaign log, final report, backend scorecard, and the contract docs.

Everything here runs against fixture orchestrations under ``tmp_path``: no child
process, no model, no registry read, and — crucially for ``backend-stats`` — no
dependence on whatever campaigns happen to exist in the developer's
``artifacts/orchestrations/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.config import PROJECT_ROOT
from autoresearch.orchestration import campaign_log as log_mod
from autoresearch.orchestration import campaign_report as campaign_mod
from autoresearch.orchestration import manifest as manifest_mod
from autoresearch.orchestration import stats as stats_mod
from autoresearch.orchestration.campaign_log import (
    Note,
    append_note,
    load_notes,
    render_campaign_log,
)
from autoresearch.orchestration.campaign_report import (
    build_campaign_report,
    finalized_status,
    write_campaign_report,
)
from autoresearch.orchestration.manifest import (
    Delegation,
    add_delegation,
    create_orchestration,
    load_orchestration,
    save_orchestration,
)
from autoresearch.orchestration.stats import collect_backend_stats
from autoresearch.utils.io import read_json, write_json


@pytest.fixture
def orchestrations_root(tmp_path, monkeypatch):
    root = tmp_path / "orchestrations"
    root.mkdir()
    monkeypatch.setattr(manifest_mod, "ORCHESTRATIONS_DIR", root)
    return root


# ── fixture builders ────────────────────────────────────────────────────────


def _delegation(delegation_id: str, backend: str = "stub", **overrides) -> Delegation:
    defaults = dict(
        brief_path=f"briefs/{delegation_id}.json",
        backend=backend,
        track="claude",
        run_id=f"2026071{delegation_id[-1]}T090000Z",
        cycle_budget=2,
        status="completed",
        spawned_at="2026-07-12T09:00:00Z",
        ended_at="2026-07-12T09:30:00Z",
        agent_summary=f"{delegation_id} testimony.",
    )
    defaults.update(overrides)
    return Delegation(delegation_id=delegation_id, **defaults)


def _report(
    delegation: Delegation,
    *,
    cycles_used: int = 2,
    decisions: tuple[str, ...] = ("promote", "reject"),
    lift: float | None = 0.010,
    model_family: str = "recipe",
    gini: float = 0.31,
    distress: tuple[str, ...] = (),
    repairs: int = 0,
    cost_usd: float | None = 1.50,
    wall_clock_minutes: float | None = 30.0,
) -> dict:
    llm_usage: dict = {"calls": 4, "input_tokens": 100, "output_tokens": 20}
    if cost_usd is not None:
        llm_usage["cost_usd"] = cost_usd
    return {
        "delegation_id": delegation.delegation_id,
        "run_id": delegation.run_id,
        "track": delegation.track,
        "backend": delegation.backend,
        "status": delegation.status,
        "cycles": {
            "budget": delegation.cycle_budget,
            "attempted": cycles_used,
            "completed": cycles_used,
            "decided": len(decisions),
            "used": cycles_used,
        },
        "champion": {
            "experiment_id": f"exp_{delegation.delegation_id}",
            "model_family": model_family,
            "gini_weighted": gini,
            "beat_seed_baseline": model_family != "global_mean",
        },
        "experiments": [
            {
                "cycle": index + 1,
                "decision": decision,
                "lift_vs_champion": (lift if decision == "promote" else -0.001),
            }
            for index, decision in enumerate(decisions)
        ],
        "repairs": {"requests": repairs, "max_attempts_in_a_cycle": min(repairs, 3)},
        "agent_summary": delegation.agent_summary,
        "distress": {"flags": [], "active": list(distress), "detail": ""},
        "cost": {"wall_clock_minutes": wall_clock_minutes, "llm_usage": llm_usage},
    }


def _campaign(
    orchestration_id: str = "20260712T090000Z",
    *,
    delegations: tuple[Delegation, ...] = (),
    reports: dict[str, dict] | None = None,
    total_cycles: int = 10,
):
    orch = create_orchestration(
        dataset="porto_seguro",
        target_mode="claim_incidence",
        total_cycle_budget=total_cycles,
        model_provider="anthropic",
        model_name="claude-opus-4-8",
        orchestration_id=orchestration_id,
    )
    for delegation in delegations:
        orch = add_delegation(orch, delegation)
    save_orchestration(orch)
    for delegation_id, payload in (reports or {}).items():
        write_json(
            manifest_mod.report_path(orchestration_id, delegation_id), payload
        )
    return orch


# ── orchestrate note ────────────────────────────────────────────────────────


def test_notes_persist_structurally_and_render_deterministic_markdown(orchestrations_root):
    orch = _campaign(delegations=(_delegation("d01"),))

    append_note(
        orch.orchestration_id,
        text="Direct tweedie plateaued near 0.31.",
        kind="reflection",
        delegation_id="d01",
        timestamp="2026-07-12T10:00:00Z",
    )
    append_note(
        orch.orchestration_id,
        text="Taking over d01 to diagnose calibration.",
        kind="takeover",
        timestamp="2026-07-12T11:00:00Z",
    )

    stored = read_json(manifest_mod.notes_path(orch.orchestration_id))
    assert [note["kind"] for note in stored["notes"]] == ["reflection", "takeover"]
    assert stored["notes"][0]["delegation_id"] == "d01"
    assert stored["notes"][1]["delegation_id"] is None

    markdown = manifest_mod.log_markdown_path(orch.orchestration_id).read_text()
    assert "## Campaign (framework-computed)" in markdown
    assert "## Delegations (framework-computed)" in markdown
    assert "## Operator log" in markdown
    # Framework facts and operator commentary never share a section.
    facts, _, commentary = markdown.partition("## Operator log")
    assert "Direct tweedie plateaued" not in facts
    assert "Direct tweedie plateaued near 0.31." in commentary
    assert "2026-07-12T10:00:00Z — reflection · d01" in commentary
    assert "2026-07-12T11:00:00Z — takeover" in commentary

    # Structured storage is the source of truth: a hand-edited log is overwritten
    # byte-for-byte by a regeneration from notes.json + the manifest.
    manifest_mod.log_markdown_path(orch.orchestration_id).write_text("hand edited\n")
    regenerated = render_campaign_log(load_orchestration(orch.orchestration_id))
    assert regenerated.read_text() == markdown
    assert render_campaign_log(load_orchestration(orch.orchestration_id)).read_text() == markdown


def test_note_rejects_unknown_kind_empty_text_and_unknown_delegation(orchestrations_root):
    orch = _campaign(delegations=(_delegation("d01"),))

    with pytest.raises(ValueError, match="Unknown note kind"):
        Note(timestamp="2026-07-12T10:00:00Z", kind="rambling", text="hi")
    with pytest.raises(ValueError, match="needs text"):
        Note(timestamp="2026-07-12T10:00:00Z", kind="plan", text="   ")
    with pytest.raises(KeyError, match="d99"):
        append_note(orch.orchestration_id, text="x", delegation_id="d99")

    assert load_notes(orch.orchestration_id) == []


def test_takeover_note_marks_delegation_and_reattributes_run(
    orchestrations_root, tmp_path, monkeypatch
):
    orch = _campaign(delegations=(_delegation("d01", status="failed"),))
    delegation = orch.delegation("d01")
    run_dir = tmp_path / "artifacts" / "tracks" / delegation.track / "runs" / delegation.run_id
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "run_manifest.json",
        {"model_identity": {"provider": "stub", "name": "stub-scripted-agent", "harness": "stub"}},
    )
    monkeypatch.setattr(manifest_mod, "PROJECT_ROOT", tmp_path)

    append_note(
        orch.orchestration_id,
        text="Backend dead; driving d01 directly.",
        kind="takeover",
        delegation_id="d01",
        timestamp="2026-07-12T12:00:00Z",
    )

    updated = load_orchestration(orch.orchestration_id).delegation("d01")
    assert updated.taken_over is True
    identity = read_json(run_dir / "run_manifest.json")["model_identity"]
    # The orchestrator's own model now owns this run's results.
    assert identity == {
        "provider": "anthropic",
        "name": "claude-opus-4-8",
        "harness": "orchestrator-takeover",
    }
    # A non-takeover note (or one without a delegation) changes nothing.
    append_note(orch.orchestration_id, text="just a thought", kind="takeover")
    append_note(orch.orchestration_id, text="reflecting", kind="reflection", delegation_id="d01")
    assert load_orchestration(orch.orchestration_id).delegation("d01").taken_over is True


def test_campaign_log_renders_before_any_note_exists(orchestrations_root):
    orch = _campaign()
    markdown = render_campaign_log(orch).read_text()
    assert "No delegations spawned yet." in markdown
    assert "No operator entries yet." in markdown


# ── orchestrate report ──────────────────────────────────────────────────────


def test_campaign_report_aggregates_framework_facts_and_keeps_testimony_apart(
    orchestrations_root,
):
    d01 = _delegation("d01")
    d02 = _delegation("d02", backend="stub-codex", status="timed_out")
    orch = _campaign(
        delegations=(d01, d02),
        reports={
            "d01": _report(d01, decisions=("promote", "reject"), lift=0.012, repairs=1),
            "d02": _report(
                d02,
                cycles_used=1,
                decisions=("reject",),
                model_family="global_mean",
                gini=0.0,
                distress=("all_rejected", "champion_is_baseline", "budget_overrun"),
                cost_usd=0.75,
            ),
        },
    )

    payload = build_campaign_report(orch, generated_at="2026-07-12T10:00:00Z")
    framework = payload["framework_computed"]

    assert framework["cycles"] == {
        "total_budget": 10,
        "committed": 4,
        "attempted": 3,
        "completed": 3,
        "decided": 3,
        "used": 3,
        "remaining": 6,
        "forfeited": 0,
    }
    assert framework["experiments"]["total"] == 3
    assert framework["experiments"]["by_decision"] == {"promote": 1, "reject": 2}
    assert framework["promotions"]["promote"] == 1
    assert framework["promotions"]["mean_promoted_gini_lift"] == pytest.approx(0.012)
    assert framework["distress"]["delegations_in_distress"] == 1
    assert framework["distress"]["by_flag"]["all_rejected"] == 1
    assert framework["distress"]["by_flag"]["champion_is_baseline"] == 1
    assert framework["distress"]["by_flag"]["repair_exhausted"] == 0
    assert framework["repairs"] == {"requests": 1, "per_cycle": pytest.approx(0.3333)}
    assert framework["cost"]["cost_usd"] == pytest.approx(2.25)
    assert framework["cost"]["cost_per_cycle_usd"] == pytest.approx(0.75)
    assert framework["cost"]["cost_per_promotion_usd"] == pytest.approx(2.25)
    assert framework["wall_clock"]["delegation_minutes_total"] == pytest.approx(60.0)
    assert framework["missing_reports"] == []
    assert framework["playoff"] is None
    assert framework["final_champion"] is None

    # Testimony is quarantined: it appears only under agent_testimony.
    assert payload["agent_testimony"] == {
        "d01": "d01 testimony.",
        "d02": "d02 testimony.",
    }
    assert "testimony" not in str(framework)

    markdown = campaign_mod.render_campaign_markdown(payload)
    facts, _, testimony = markdown.partition("## Agent testimony")
    assert "d01 testimony." not in facts
    assert "d01 testimony." in testimony
    assert "Framework-computed" in facts


def test_campaign_report_excludes_delegations_without_a_collected_report(
    orchestrations_root,
):
    d01 = _delegation("d01")
    d02 = _delegation("d02")
    orch = _campaign(delegations=(d01, d02), reports={"d01": _report(d01)})

    framework = build_campaign_report(orch, generated_at="2026-07-12T10:00:00Z")[
        "framework_computed"
    ]
    assert framework["missing_reports"] == ["d02"]
    assert framework["cycles"]["used"] == 2  # d02 contributes nothing, not zero-with-flags
    assert framework["distress"]["delegations_in_distress"] == 0
    (missing,) = [d for d in framework["delegations"] if d["delegation_id"] == "d02"]
    assert missing["report"] == "missing"


def test_campaign_report_never_invents_a_zero_cost(orchestrations_root):
    d01 = _delegation("d01")
    d02 = _delegation("d02")
    orch = _campaign(
        delegations=(d01, d02),
        reports={
            "d01": _report(d01, cost_usd=None, wall_clock_minutes=None),
            "d02": _report(d02, cost_usd=None, wall_clock_minutes=None),
        },
    )

    cost = build_campaign_report(orch, generated_at="2026-07-12T10:00:00Z")[
        "framework_computed"
    ]["cost"]
    assert cost["cost_usd"] is None
    assert cost["cost_per_cycle_usd"] is None
    assert cost["cost_per_promotion_usd"] is None
    assert cost["cost_usd_missing_for"] == ["d01", "d02"]
    assert cost["input_tokens"] == 200

    wall_clock = build_campaign_report(orch, generated_at="2026-07-12T10:00:00Z")[
        "framework_computed"
    ]["wall_clock"]
    assert wall_clock["delegation_minutes_total"] is None

    markdown = campaign_mod.render_campaign_markdown(
        build_campaign_report(orch, generated_at="2026-07-12T10:00:00Z")
    )
    assert "Cost: not reported" in markdown
    assert "unmeasured, not free" in markdown
    assert "$0.0000" not in markdown


def test_campaign_report_partial_cost_names_which_delegations_it_covers(
    orchestrations_root,
):
    d01 = _delegation("d01")
    d02 = _delegation("d02")
    orch = _campaign(
        delegations=(d01, d02),
        reports={"d01": _report(d01, cost_usd=2.0), "d02": _report(d02, cost_usd=None)},
    )

    payload = build_campaign_report(orch, generated_at="2026-07-12T10:00:00Z")
    cost = payload["framework_computed"]["cost"]
    assert cost["cost_usd"] == pytest.approx(2.0)
    assert cost["cost_usd_reported_by"] == ["d01"]
    assert cost["cost_usd_missing_for"] == ["d02"]
    assert "covers only d01" in campaign_mod.render_campaign_markdown(payload)


def test_campaign_report_carries_completed_playoff_lineage(orchestrations_root):
    d01 = _delegation("d01")
    d03 = _delegation("d03")
    orch = _campaign(
        delegations=(d01, d03),
        reports={"d01": _report(d01, gini=0.10), "d03": _report(d03, gini=0.30)},
    )
    orch = manifest_mod.Orchestration.from_dict(
        {
            **orch.to_dict(),
            "status": "completed",
            "consolidation": {
                "track": "claude",
                "run_id": "20260712T150000Z",
                "playoff_report": "playoff/playoff_report.json",
            },
        }
    )
    save_orchestration(orch)
    write_json(
        manifest_mod.playoff_dir(orch.orchestration_id) / "playoff_report.json",
        {
            "status": "completed",
            "decision_mode": "auto",
            "consolidation": {"track": "claude", "run_id": "20260712T150000Z"},
            "exclusions": [{"delegation_id": "d02", "reason": "champion_is_baseline"}],
            "finalists": [
                {"delegation_id": "d01", "gini_weighted": 0.10},
                {"delegation_id": "d03", "gini_weighted": 0.30},
            ],
            "pairings": [
                {
                    "order": 1,
                    "finalist": {"delegation_id": "d03"},
                    "comparison_id": "cmp_replayed_d03",
                    "decision": "promote",
                }
            ],
            "final_champion_lineage": {
                "consolidation_experiment_id": "replayed_d03",
                "source": {
                    "delegation_id": "d03",
                    "run_id": "20260713T090000Z",
                    "experiment_id": "exp_d03",
                },
            },
        },
    )

    payload = build_campaign_report(orch, generated_at="2026-07-12T16:00:00Z")
    playoff = payload["framework_computed"]["playoff"]
    assert playoff["status"] == "completed"
    assert [item["delegation_id"] for item in playoff["finalists"]] == ["d01", "d03"]
    assert playoff["exclusions"][0]["reason"] == "champion_is_baseline"

    champion = payload["framework_computed"]["final_champion"]
    assert champion["consolidation_run_id"] == "20260712T150000Z"
    assert champion["consolidation_experiment_id"] == "replayed_d03"
    assert champion["source"]["delegation_id"] == "d03"

    markdown = campaign_mod.render_campaign_markdown(payload)
    assert "replayed_d03" in markdown
    assert "Replayed from delegation d03" in markdown


def test_write_campaign_report_updates_status_and_report_path(orchestrations_root):
    d01 = _delegation("d01")
    orch = _campaign(delegations=(d01,), reports={"d01": _report(d01)})
    assert orch.status == "active"

    result = write_campaign_report(orch.orchestration_id)

    assert result["json_path"].exists()
    assert result["markdown_path"].exists()
    reloaded = load_orchestration(orch.orchestration_id)
    assert reloaded.status == "completed"
    assert reloaded.campaign_report is not None
    stored = Path(reloaded.campaign_report)
    resolved = stored if stored.is_absolute() else PROJECT_ROOT / stored
    assert resolved == result["json_path"]
    assert read_json(result["json_path"])["status"] == "completed"


def test_finalized_status_respects_live_delegations_and_playoff_ownership(
    orchestrations_root,
):
    running = _campaign(
        "20260712T090001Z", delegations=(_delegation("d01", status="running"),)
    )
    assert finalized_status(running) == "active"

    consolidating = manifest_mod.Orchestration.from_dict(
        {**_campaign("20260712T090002Z").to_dict(), "status": "consolidating"}
    )
    assert finalized_status(consolidating) == "consolidating"

    abandoned = manifest_mod.Orchestration.from_dict(
        {**_campaign("20260712T090003Z").to_dict(), "status": "abandoned"}
    )
    assert finalized_status(abandoned) == "abandoned"

    done = _campaign("20260712T090004Z", delegations=(_delegation("d01"),))
    assert finalized_status(done) == "completed"


# ── orchestrate backend-stats ───────────────────────────────────────────────


def test_backend_stats_aggregates_across_campaigns_and_backends(orchestrations_root):
    first_a = _delegation("d01", backend="stub")
    first_b = _delegation("d02", backend="stub-codex")
    _campaign(
        "20260712T090000Z",
        delegations=(first_a, first_b),
        reports={
            "d01": _report(first_a, decisions=("promote", "reject"), lift=0.010, repairs=2),
            "d02": _report(
                first_b,
                cycles_used=2,
                decisions=("reject", "reject"),
                model_family="global_mean",
                distress=("all_rejected", "champion_is_baseline"),
                cost_usd=0.50,
            ),
        },
    )
    second_a = _delegation("d01", backend="stub")
    _campaign(
        "20260713T090000Z",
        delegations=(second_a,),
        reports={
            "d01": _report(
                second_a, decisions=("promote", "promote"), lift=0.020, cost_usd=2.50
            )
        },
    )

    stats = collect_backend_stats(orchestrations_root)
    assert stats["campaigns"] == 2
    assert sorted(stats["backends"]) == ["stub", "stub-codex"]

    stub = stats["backends"]["stub"]
    assert stub["campaigns"] == 2
    assert stub["delegations"] == 2
    assert stub["cycles"] == 4
    assert stub["promotions"] == 3
    assert stub["promotion_rate"] == pytest.approx(0.75)  # 3 promotions / 4 cycles
    assert stub["distress_rate"] == pytest.approx(0.0)
    assert stub["repair_attempts_per_cycle"] == pytest.approx(0.5)  # 2 requests / 4 cycles
    # (0.010 + 0.020 + 0.020) / 3
    assert stub["mean_promoted_gini_lift"] == pytest.approx(0.016667, abs=1e-6)
    assert stub["cost_usd"] == pytest.approx(4.0)
    assert stub["cost_per_cycle_usd"] == pytest.approx(1.0)
    assert stub["cost_per_promotion_usd"] == pytest.approx(4.0 / 3)

    codex = stats["backends"]["stub-codex"]
    assert codex["delegations"] == 1
    assert codex["promotions"] == 0
    assert codex["promotion_rate"] == pytest.approx(0.0)
    assert codex["distress_rate"] == pytest.approx(1.0)
    assert codex["distress_rate_by_flag"]["all_rejected"] == pytest.approx(1.0)
    assert codex["distress_rate_by_flag"]["champion_is_baseline"] == pytest.approx(1.0)
    assert codex["distress_rate_by_flag"]["crashed"] == pytest.approx(0.0)
    assert codex["mean_promoted_gini_lift"] is None
    assert codex["cost_per_promotion_usd"] is None  # no promotions: not zero cost


def test_backend_stats_skips_delegations_without_a_report(orchestrations_root):
    d01 = _delegation("d01")
    d02 = _delegation("d02")
    _campaign("20260712T090000Z", delegations=(d01, d02), reports={"d01": _report(d01)})

    stats = collect_backend_stats(orchestrations_root)
    assert stats["skipped_missing_report"] == 1
    assert stats["backends"]["stub"]["delegations"] == 1
    assert stats["backends"]["stub"]["cycles"] == 2


def test_backend_stats_reports_unknown_rather_than_zero_for_missing_telemetry(
    orchestrations_root,
):
    d01 = _delegation("d01")
    _campaign(
        "20260712T090000Z",
        delegations=(d01,),
        reports={"d01": _report(d01, cycles_used=0, decisions=(), cost_usd=None)},
    )

    stub = collect_backend_stats(orchestrations_root)["backends"]["stub"]
    assert stub["cycles"] == 0
    assert stub["promotion_rate"] is None  # a rate over zero cycles is unknown
    assert stub["repair_attempts_per_cycle"] is None
    assert stub["cost_usd"] is None
    assert stub["delegations_with_cost"] == 0
    assert stub["cost_per_cycle_usd"] is None

    rendered = stats_mod.format_backend_stats(collect_backend_stats(orchestrations_root))
    assert "promotion_rate=—" in rendered
    assert "cost_usd=not reported" in rendered


def test_backend_stats_on_an_empty_root_is_empty_not_an_error(tmp_path):
    stats = collect_backend_stats(tmp_path / "nothing-here")
    assert stats == {"campaigns": 0, "backends": {}, "skipped_missing_report": 0}
    assert "nothing to score" in stats_mod.format_backend_stats(stats)


# ── list-backends scorecard integration ─────────────────────────────────────


def test_list_backends_appends_the_scorecard_without_changing_the_registry(
    orchestrations_root, monkeypatch, capsys
):
    from autoresearch import cli
    from autoresearch.orchestration.backends import load_backends

    d01 = _delegation("d01", backend="stub")
    _campaign("20260712T090000Z", delegations=(d01,), reports={"d01": _report(d01)})

    registry = load_backends()
    assert "stub" in registry and "stub-codex" in registry

    assert cli._orchestrate_list_backends() == 0
    out = capsys.readouterr().out

    # Every registry entry is still listed, and only the registry's entries are.
    for name in registry:
        assert f"- {name}" in out
    assert "not_in_the_registry" not in out
    # The measured backend gets numbers; the untouched one is visibly untested.
    assert "scorecard: 1 delegations / 2 cycles" in out
    assert "scorecard: — (no delegations recorded yet)" in out


def test_scorecard_never_raises_when_the_campaign_root_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_mod, "ORCHESTRATIONS_DIR", tmp_path / "absent")
    assert stats_mod.scorecard() == {}


# ── CLI parsing and dispatch ────────────────────────────────────────────────


def test_note_cli_parses_and_dispatches(orchestrations_root, monkeypatch, capsys):
    from autoresearch import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        ["orchestrate", "note", "--text", "hello", "--kind", "takeover", "--delegation", "d01"]
    )
    assert (args.orchestrate_subcommand, args.kind, args.delegation) == (
        "note",
        "takeover",
        "d01",
    )
    assert parser.parse_args(["orchestrate", "note", "--text", "x"]).kind == "reflection"
    with pytest.raises(SystemExit):
        parser.parse_args(["orchestrate", "note", "--text", "x", "--kind", "rambling"])

    captured: dict = {}
    monkeypatch.setattr(
        manifest_mod, "resolve_orchestration_id", lambda value: "20260712T090000Z"
    )
    monkeypatch.setattr(
        log_mod,
        "append_note",
        lambda orchestration_id, **kwargs: captured.update(
            {"orchestration_id": orchestration_id, **kwargs}
        )
        or Note(timestamp="2026-07-12T10:00:00Z", kind=kwargs["kind"], text=kwargs["text"]),
    )
    assert cli._orchestrate_note(args) == 0
    assert captured == {
        "orchestration_id": "20260712T090000Z",
        "text": "hello",
        "kind": "takeover",
        "delegation_id": "d01",
    }
    assert "Campaign log regenerated" in capsys.readouterr().out


def test_report_cli_parses_and_dispatches(monkeypatch, capsys, tmp_path):
    from autoresearch import cli

    parser = cli.build_parser()
    assert parser.parse_args(["orchestrate", "report"]).json is False
    args = parser.parse_args(["orchestrate", "report", "--json"])
    assert args.json is True

    markdown = tmp_path / "CAMPAIGN_REPORT.md"
    markdown.write_text("# Campaign Report\n")
    captured: dict = {}
    monkeypatch.setattr(
        manifest_mod, "resolve_orchestration_id", lambda value: "20260712T090000Z"
    )
    monkeypatch.setattr(
        campaign_mod,
        "write_campaign_report",
        lambda orchestration_id: captured.update({"orchestration_id": orchestration_id})
        or {
            "payload": {"status": "completed"},
            "json_path": tmp_path / "campaign_report.json",
            "markdown_path": markdown,
        },
    )
    assert cli._orchestrate_report(args) == 0
    assert captured == {"orchestration_id": "20260712T090000Z"}
    assert '"status": "completed"' in capsys.readouterr().out

    assert cli._orchestrate_report(parser.parse_args(["orchestrate", "report"])) == 0
    assert "# Campaign Report" in capsys.readouterr().out


def test_backend_stats_cli_parses_and_dispatches(orchestrations_root, capsys):
    from autoresearch import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        ["orchestrate", "backend-stats", "--orchestrations-root", str(orchestrations_root)]
    )
    assert args.orchestrations_root == str(orchestrations_root)
    assert args.json is False

    d01 = _delegation("d01")
    _campaign("20260712T090000Z", delegations=(d01,), reports={"d01": _report(d01)})

    assert cli._orchestrate_backend_stats(args) == 0
    assert "Backend scorecard across 1 campaign(s)" in capsys.readouterr().out

    json_args = parser.parse_args(
        [
            "orchestrate",
            "backend-stats",
            "--orchestrations-root",
            str(orchestrations_root),
            "--json",
        ]
    )
    assert cli._orchestrate_backend_stats(json_args) == 0
    assert '"promotion_rate"' in capsys.readouterr().out


def test_orchestrate_dispatch_routes_the_new_subcommands(monkeypatch):
    from autoresearch import cli

    routed: list[str] = []
    for subcommand, target in (
        ("note", "_orchestrate_note"),
        ("report", "_orchestrate_report"),
        ("backend-stats", "_orchestrate_backend_stats"),
    ):
        monkeypatch.setattr(cli, target, lambda args, name=target: routed.append(name) or 0)
        args = cli.build_parser().parse_args(
            ["orchestrate", subcommand] + (["--text", "x"] if subcommand == "note" else [])
        )
        assert cli._cmd_orchestrate(None, args) == 0
    assert routed == ["_orchestrate_note", "_orchestrate_report", "_orchestrate_backend_stats"]


# ── contract and documentation presence ─────────────────────────────────────


def test_orchestrator_contract_exists_and_carries_the_safety_guidance():
    text = (PROJECT_ROOT / "ORCHESTRATOR.md").read_text(encoding="utf-8")

    # Design §6's six-part outline.
    for heading in (
        "## 1. Role",
        "## 2. Workflow",
        "## 4. Delegation heuristics",
        "## 5. Backend choice",
        "## 6. Reading reports",
        "## 9. Takeover",
        "## 10. Hard constraints",
    ):
        assert heading in text

    # The safety rules that make orchestration auditable.
    assert "never read the holdout" in text
    assert "integrity-protected evaluation files" in text
    assert "Never hand-edit a child run's artifacts" in text
    assert "Never spawn a backend that is not in `orchestrate list-backends`" in text
    assert "treat `agent_summary` as testimony" in text

    # Every distress flag has a documented response.
    from autoresearch.orchestration.report import DISTRESS_FLAGS

    for flag in DISTRESS_FLAGS:
        assert f"`{flag}`" in text


def test_orchestrated_docs_are_wired_together():
    quickstart = (PROJECT_ROOT / "docs" / "RUN_ORCHESTRATED.md").read_text(encoding="utf-8")
    for section in (
        "## Launching the orchestrator session",
        "## Sequential campaign",
        "## Parallel campaign",
        "## Distress handling",
        "## Respawn and seeding",
        "## Playoff",
        "## Takeover",
    ):
        assert section in quickstart
    assert "AUTORESEARCH_SCOPE=orchestrator" in quickstart

    cli_doc = (PROJECT_ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    for command in (
        "### `orchestrate new`",
        "### `orchestrate spawn`",
        "### `orchestrate playoff`",
        "### `orchestrate note`",
        "### `orchestrate report`",
        "### `orchestrate backend-stats`",
        "### `orchestrate finish-delegation`",
    ):
        assert command in cli_doc

    architecture = (PROJECT_ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    assert "## Orchestration" in architecture
    assert "orchestration/" in architecture

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/RUN_ORCHESTRATED.md" in readme
    assert "ORCHESTRATOR.md" in readme
