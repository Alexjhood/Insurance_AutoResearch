from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from autoresearch.config import PROJECT_ROOT, ProjectConfig
from autoresearch.comparison_runner import compare_experiments
from autoresearch.experiment_registry.registry import list_experiments
from autoresearch.experiment_runner import run_experiment


def _make_config(tmp_path: Path) -> ProjectConfig:
    """Build a minimal ProjectConfig for runner tests."""
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    artifacts = tmp_path / "artifacts"
    processed.mkdir(parents=True)
    splits.mkdir(parents=True)
    return ProjectConfig(
        root=tmp_path,
        raw_data_dir=tmp_path / "raw",
        processed_dir=processed,
        holdout_vault_dir=tmp_path / "holdout_vault",
        metadata_dir=tmp_path / "metadata",
        splits_dir=splits,
        artifacts_dir=artifacts,
        registry_path=artifacts / "registry.sqlite",
        research_log_path=tmp_path / "RESEARCH_LOG.md",
        track_id="test",
        random_seed=1,
        id_column="IDpol",
        agent_dataset_name="agent_dataset",
        claim_capping_enabled=True,
        claim_cap_threshold=100000,
        split_ratios={"train": 0.64, "search_validation": 0.16, "milestone_holdout": 0.2},
        ordinary_train_split="train",
        ordinary_eval_splits=("search_validation",),
        target_mode="burning_cost",
        primary_metric="tweedie_deviance_p15",
        tweedie_power=1.5,
        use_cv=False,
        cv_folds=4,
        cv_n_repeats=4,
        cv_seed=1,
        gate_mode="single_partition",
        gate_primary_metric="gini_weighted",
        bootstrap_per_fold=5,
        escalation_win_rate_low=0.40,
        escalation_win_rate_high=0.60,
        escalation_partitions=1,
        repeated_resamples=5,
        bootstrap_iterations=20,
        resample_fraction=1.0,
        resampling_seed=1,
        minimum_mean_lift=0.0,
        min_relative_lift=0.005,
        min_absolute_lift=0.0,
        minimum_win_rate=0.60,
        bootstrap_lower_bound=0.0,
        bootstrap_lower_bound_relative=0.0,
        confidence_level=0.9,
        max_predicted_to_actual_drift=0.05,
        require_diagnostics=True,
        bonferroni_lookback=10,
        handoff_base_dir=tmp_path / "auto_research",
        handoff_context_dir=tmp_path / "auto_research" / "context",
        handoff_proposal_inbox_dir=tmp_path / "auto_research" / "proposals" / "inbox",
        handoff_proposal_processed_dir=tmp_path / "auto_research" / "proposals" / "processed",
        handoff_results_dir=tmp_path / "auto_research" / "results",
        handoff_handoffs_dir=tmp_path / "auto_research" / "handoffs",
        proposal_inbox_file=tmp_path / "auto_research" / "proposals" / "inbox" / "manual_proposals.jsonl",
        deduplication_policy="reject",
        deduplication_lookback=25,
        search_space={
            "model_families": ["global_mean"],
            "target_strategies": ["direct_pure_premium", "frequency_severity"],
            "preprocessing": {"claim_cap_thresholds": [100000], "allow_disable_claim_capping": False},
        },
    )


def _write_fixtures(config: ProjectConfig) -> None:
    """Write minimal parquet + split_pack for runner tests."""
    frame = pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "ClaimNb": [0, 1, 0, 1, 0, 1],
        "Exposure": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "VehPower": [4, 5, 4, 5, 4, 5],
        "Region": ["a", "b", "a", "b", "a", "b"],
        "ClaimAmount": [0.0, 100.0, 0.0, 200.0, 0.0, 150.0],
        "ClaimAmountCount": [0, 1, 0, 1, 0, 1],
    })
    # Write both legacy path and new search path
    frame.to_parquet(config.processed_dir / "agent_dataset.parquet", index=False)
    frame.to_parquet(config.processed_dir / "agent_dataset_search.parquet", index=False)
    pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "split_unit": [0.1, 0.2, 0.5, 0.6, 0.7, 0.9],
        "split": ["train", "train", "train", "train", "search_validation", "search_validation"],
    }).to_csv(config.splits_dir / "split_pack.csv", index=False)


def test_run_experiment_writes_registry_and_artifacts(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)

    exp_config = tmp_path / "experiment.toml"
    exp_config.write_text(
        """
experiment_name = "test_direct"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )

    outputs = run_experiment(config, exp_config)
    rows = list_experiments(config.registry_path)

    assert outputs["metrics"].exists()
    assert outputs["diagnostics"].exists()
    assert outputs["environment_manifest"].exists()
    assert rows[0]["experiment_name"] == "test_direct"
    assert rows[0]["claim_cap_threshold"] == 100000
    assert "iterations" in outputs["metrics"].parts
    assert "experiments" not in outputs["metrics"].relative_to(config.artifacts_dir).parts


def test_run_experiment_uses_run_local_model_script(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    _write_fixtures(config)
    script = tmp_path / "scripted_model.py"
    script.write_text(
        """
import numpy as np

EXPOSURE = "Exposure"
CLAIM_COST = "ClaimAmountCapped"


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hp):
    rate = float(train[CLAIM_COST].sum() / train[EXPOSURE].sum())
    return rate * score[EXPOSURE].to_numpy(dtype=float), {"rate": rate}
""".strip(),
        encoding="utf-8",
    )

    exp_config = tmp_path / "experiment_scripted.toml"
    exp_config.write_text(
        f"""
experiment_name = "test_scripted_model"
model_family = "scripted_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
script_path = "{script.name}"
""".strip(),
        encoding="utf-8",
    )

    outputs = run_experiment(config, exp_config)

    assert outputs["metrics"].exists()
    assert outputs["model_script"] == script.resolve()


def test_run_experiment_supports_frequency_target_mode(tmp_path: Path) -> None:
    config = replace(_make_config(tmp_path), target_mode="frequency", primary_metric="poisson_deviance")
    _write_fixtures(config)

    exp_config = tmp_path / "experiment_frequency.toml"
    exp_config.write_text(
        """
experiment_name = "test_frequency"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )

    outputs = run_experiment(config, exp_config)
    predictions = pd.read_parquet(outputs["predictions"])
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))

    assert set(predictions["target_mode"]) == {"frequency"}
    assert predictions["predicted_claim_count"].notna().all()
    assert predictions["predicted_claim_cost"].isna().all()
    assert metrics["target_mode"] == "frequency"


def test_compare_experiments_writes_html_report(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config = replace(
        config,
        primary_metric="gini_weighted",
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)

    champion_config = tmp_path / "champion.toml"
    champion_config.write_text(
        """
experiment_name = "champion_report"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )
    challenger_config = tmp_path / "challenger.toml"
    challenger_config.write_text(
        """
experiment_name = "challenger_report"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )

    run_experiment(config, champion_config)
    run_experiment(config, challenger_config)
    rows = list_experiments(config.registry_path)
    champion_id = next(row["experiment_id"] for row in rows if row["experiment_name"] == "champion_report")
    challenger_id = next(row["experiment_id"] for row in rows if row["experiment_name"] == "challenger_report")
    config = replace(config, root=PROJECT_ROOT)

    outputs = compare_experiments(config, champion_id, challenger_id)

    assert outputs["html_report"].exists()
    assert "Validation Metrics" in outputs["html_report"].read_text(encoding="utf-8")


def test_compare_experiments_repeated_cv_mode(tmp_path: Path) -> None:
    """End-to-end repeated_cv gate: refits models on folds and renders report."""
    config = _make_config(tmp_path)
    config = replace(
        config,
        primary_metric="gini_weighted",
        gate_mode="repeated_cv",
        gate_primary_metric="rank_gini_weighted",
        cv_folds=2,
        cv_n_repeats=2,
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)

    # Fold assignments covering all six search rows (the CV path needs these)
    pd.DataFrame({
        "record_id": [1, 2, 3, 4, 5, 6],
        "fold": [1, 2, 1, 2, 1, 2],
    }).to_parquet(config.splits_dir / "split_pack_folds.parquet", index=False)

    for name in ("champion_cv", "challenger_cv"):
        exp_config = tmp_path / f"{name}.toml"
        exp_config.write_text(
            f"""
experiment_name = "{name}"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
            encoding="utf-8",
        )
        run_experiment(config, exp_config)

    rows = list_experiments(config.registry_path)
    champion_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == "champion_cv")
    challenger_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == "challenger_cv")
    config = replace(config, root=PROJECT_ROOT)

    outputs = compare_experiments(config, champion_id, challenger_id)

    # Report and machine-readable artifacts exist
    assert outputs["html_report"].exists()
    html = outputs["html_report"].read_text(encoding="utf-8")
    assert "Multi-Metric Comparison" in html
    assert "rank_gini_weighted" in html

    report = json.loads(outputs["promotion_report"].read_text(encoding="utf-8"))
    assert report["gate_mode"] == "repeated_cv"
    summary = report["comparison_summary"]
    # 2 folds × 2 repeats = 4 partitions
    assert summary["n_partitions"] == 4
    assert summary["gate_primary_metric"] == "rank_gini_weighted"
    assert "between_partition_std" in summary
    # Metric lift table populated with the gate metric flagged
    table = report["metric_lift_table"]
    gate_rows = [r for r in table if r.get("is_gate_metric")]
    assert len(gate_rows) == 1
    assert gate_rows[0]["metric"] == "rank_gini_weighted"


def test_compare_experiments_writes_pending_llm_decision(tmp_path: Path) -> None:
    """compare_experiments writes decision=pending_llm, not a mechanical verdict."""
    config = replace(
        _make_config(tmp_path),
        primary_metric="gini_weighted",
        gate_mode="single_partition",
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)

    for name in ("champ_pd", "chal_pd"):
        exp_config = tmp_path / f"{name}.toml"
        exp_config.write_text(
            f"""
experiment_name = "{name}"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
            encoding="utf-8",
        )
        run_experiment(config, exp_config)

    rows = list_experiments(config.registry_path)
    champ_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == "champ_pd")
    chal_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == "chal_pd")
    config = replace(config, root=PROJECT_ROOT)

    artifacts = compare_experiments(config, champ_id, chal_id)
    decision = json.loads(artifacts["promotion_decision"].read_text())
    report = json.loads(artifacts["promotion_report"].read_text())

    assert decision["decision"] == "pending_llm"
    assert "advisory_decision" in decision
    assert report["promotion_decision"]["decision"] == "pending_llm"
    assert "guardrail_result" in report


def test_record_decision_reject_persists_rationale(tmp_path: Path) -> None:
    """record_decision('reject') persists the rationale without promoting."""
    from autoresearch.comparison_runner import record_decision
    from autoresearch.experiment_registry.registry import (
        list_comparisons,
        list_proposals,
        record_proposal,
        update_proposal_status,
        upsert_research_node,
    )

    config = replace(
        _make_config(tmp_path),
        primary_metric="gini_weighted",
        gate_mode="single_partition",
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)

    for name in ("champ_rd", "chal_rd"):
        exp_config = tmp_path / f"{name}.toml"
        exp_config.write_text(
            f"""
experiment_name = "{name}"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
            encoding="utf-8",
        )
        run_experiment(config, exp_config)

    rows = list_experiments(config.registry_path)
    champ_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == "champ_rd")
    chal_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == "chal_rd")
    config = replace(config, root=PROJECT_ROOT)

    artifacts = compare_experiments(config, champ_id, chal_id)
    report = json.loads(artifacts["promotion_report"].read_text())
    comp_id = report["comparison_id"]
    record_proposal(
        config.registry_path,
        proposal_id="reject_prop",
        status="awaiting_decision",
        parent_experiment_id=champ_id,
        parent_branch_id="main",
        branch_id="main",
        experiment_name="chal_rd",
        rationale="test reject",
        change_summary="test reject",
        expected_benefit="none",
        key_risk="none",
        config={},
        validation_errors=[],
        llm_provider="test",
        llm_model=None,
        prompt_path=None,
        response_path=None,
        proposal_path=None,
        notes="Awaiting decision.",
    )
    update_proposal_status(
        config.registry_path,
        "reject_prop",
        "awaiting_decision",
        experiment_id=chal_id,
        comparison_id=comp_id,
    )
    upsert_research_node(
        config.registry_path,
        node_id="reject_prop",
        line_id="reject_line",
        proposal_id="reject_prop",
        experiment_id=chal_id,
        comparison_id=comp_id,
        status="awaiting_decision",
    )

    result = record_decision(
        config,
        comp_id,
        decision="reject",
        rationale="Insufficient evidence.",
        reason_code="noise",
        interpretation="The challenger did not establish a reliable improvement.",
        next_step="Try a materially different modelling approach.",
    )

    assert result["decision"] == "reject"
    assert result["rationale"] == "Insufficient evidence."
    assert result["decided_by"] == "llm"
    assert result["reason_code"] == "noise"

    comps = list_comparisons(config.registry_path)
    comp = next(c for c in comps if c["comparison_id"] == comp_id)
    assert comp["decision"] == "reject"
    assert comp["decision_rationale"] == "Insufficient evidence."
    assert comp["decision_reason_code"] == "noise"
    proposal = next(p for p in list_proposals(config.registry_path) if p["proposal_id"] == "reject_prop")
    assert proposal["status"] == "rejected"
    context = json.loads((config.handoff_context_dir / "latest_context.json").read_text())
    compact = next(c for c in context["recent_comparisons"] if c["comparison_id"] == comp_id)
    assert compact["final_decision"] == "reject"


def _setup_two_experiments(tmp_path: Path, prefix: str):
    """Run champion + challenger global-mean experiments; return (config, champ_id, chal_id)."""
    config = replace(
        _make_config(tmp_path),
        primary_metric="gini_weighted",
        gate_mode="single_partition",
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)
    for name in (f"{prefix}_champ", f"{prefix}_chal"):
        exp_config = tmp_path / f"{name}.toml"
        exp_config.write_text(
            f"""
experiment_name = "{name}"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
            encoding="utf-8",
        )
        run_experiment(config, exp_config)
    rows = list_experiments(config.registry_path)
    champ_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == f"{prefix}_champ")
    chal_id = next(r["experiment_id"] for r in rows if r["experiment_name"] == f"{prefix}_chal")
    return replace(config, root=PROJECT_ROOT), champ_id, chal_id


def test_compare_experiments_cv_bootstrap_end_to_end(tmp_path: Path) -> None:
    """Default cv_bootstrap path: cache build → bootstrap → guardrails → report."""
    config = replace(
        _make_config(tmp_path),
        primary_metric="gini_weighted",
        gate_mode="cv_bootstrap",
        gate_primary_metric="gini_weighted",
        cv_folds=2,
        bootstrap_per_fold=5,
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)
    for name in ("champ_cvb", "chal_cvb"):
        exp_config = tmp_path / f"{name}.toml"
        exp_config.write_text(
            f"""
experiment_name = "{name}"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
            encoding="utf-8",
        )
        run_experiment(config, exp_config)
    rows = list_experiments(config.registry_path)
    champ = next(r["experiment_id"] for r in rows if r["experiment_name"] == "champ_cvb")
    chal = next(r["experiment_id"] for r in rows if r["experiment_name"] == "chal_cvb")
    config = replace(config, root=PROJECT_ROOT)

    artifacts = compare_experiments(config, champ, chal)
    report = json.loads(artifacts["promotion_report"].read_text())

    assert report["gate_mode"] == "cv_bootstrap"
    assert report["promotion_decision"]["decision"] == "pending_llm"
    assert report["comparison_summary"]["n_samples"] == 2 * 5  # folds × bootstrap
    assert "guardrail_result" in report
    assert artifacts["html_report"].exists()
    assert len(report["metric_lift_table"]) > 0

    # Cache reuse — second comparison must not crash and reuses cached folds
    artifacts2 = compare_experiments(config, champ, chal)
    assert artifacts2["html_report"].exists()


def test_bonferroni_family_includes_current_comparison(tmp_path: Path) -> None:
    """The Bonferroni family size counts prior comparisons plus the current one,
    so a second comparison against the same champion uses n=2 (not n=1)."""
    config, champ_id, chal_id = _setup_two_experiments(tmp_path, "bonf")

    first = compare_experiments(config, champ_id, chal_id)
    first_boot = json.loads(first["bootstrap_summary"].read_text())
    assert first_boot["n_comparisons_bonferroni"] == 1

    second = compare_experiments(config, champ_id, chal_id)
    second_boot = json.loads(second["bootstrap_summary"].read_text())
    assert second_boot["n_comparisons_bonferroni"] == 2


def test_base_partition_rotates_after_five_comparisons(tmp_path: Path) -> None:
    from autoresearch.comparison_runner import _base_partition_index
    from autoresearch.experiment_registry.registry import record_comparison

    config = replace(
        _make_config(tmp_path),
        partition_rotation_interval=5,
        escalation_partitions=2,
    )
    assert _base_partition_index(config) == 0
    for index in range(5):
        record_comparison(
            config.registry_path,
            comparison_id=f"rotation_{index}",
            champion_id="champion",
            challenger_id=f"challenger_{index}",
            paired_summary={"mean_lift": 0.0, "challenger_win_rate": 0.5},
            bootstrap_summary={"interval_lower": -0.1, "interval_upper": 0.1},
            promotion_decision="pending_llm",
            promotion_rationale="pending",
            artifacts={},
        )
    assert _base_partition_index(config) == 3


def test_record_decision_promote_updates_champion(tmp_path: Path) -> None:
    """record_decision('promote') with passing guardrails updates the official champion."""
    from autoresearch.comparison_runner import record_decision
    from autoresearch.experiment_registry.registry import (
        list_champion_history, list_comparisons, get_official_champion, set_official_champion,
    )

    config, champ_id, chal_id = _setup_two_experiments(tmp_path, "rdp")
    set_official_champion(
        config.registry_path, champion_id=champ_id, branch_id="main",
        reason="seed", action="initialised", comparison_id="",
    )

    artifacts = compare_experiments(config, champ_id, chal_id)
    comp_id = json.loads(artifacts["promotion_report"].read_text())["comparison_id"]
    # This test exercises decision bookkeeping with two constant fixtures. The
    # tie-aware Gini correctly gives them zero discrimination, so explicitly
    # isolate the bookkeeping path from the separately-tested guardrail path.
    import sqlite3
    with sqlite3.connect(config.registry_path) as con:
        con.execute(
            "UPDATE comparisons SET guardrail_status = ? WHERE comparison_id = ?",
            (json.dumps({"passed": True, "failures": [], "checks": {}}), comp_id),
        )

    result = record_decision(
        config,
        comp_id,
        decision="promote",
        rationale="Clear improvement.",
        interpretation="The challenger produced a clean improvement.",
        next_step="Build the next experiment from the promoted model.",
    )

    assert result["decision"] == "promote"
    assert get_official_champion(config.registry_path)["champion_id"] == chal_id
    comp = next(c for c in list_comparisons(config.registry_path) if c["comparison_id"] == comp_id)
    assert comp["decision"] == "promote"
    assert comp["decided_by"] == "llm"

    history_count = len(list_champion_history(config.registry_path))
    repeated = record_decision(
        config,
        comp_id,
        decision="promote",
        rationale="Clear improvement.",
        interpretation="The challenger produced a clean improvement.",
        next_step="Build the next experiment from the promoted model.",
    )

    assert repeated["already_recorded"] is True
    assert len(list_champion_history(config.registry_path)) == history_count


def test_record_decision_local_promote_updates_line_only(tmp_path: Path) -> None:
    """local_promote advances a research line without replacing the official champion."""
    from autoresearch.comparison_runner import record_decision
    from autoresearch.experiment_registry.registry import (
        get_official_champion,
        get_research_line,
        list_comparisons,
        set_official_champion,
        upsert_research_line,
        upsert_research_node,
    )

    config, champ_id, chal_id = _setup_two_experiments(tmp_path, "rdlp")
    set_official_champion(
        config.registry_path, champion_id=champ_id, branch_id="main",
        reason="seed", action="initialised", comparison_id="",
    )
    upsert_research_line(
        config.registry_path,
        line_id="line_local",
        label="Local line",
        status="active",
        root_node_id="node_local",
        hypothesis="Test local progression.",
        current_experiment_id=champ_id,
    )
    upsert_research_node(
        config.registry_path,
        node_id="node_local",
        line_id="line_local",
        experiment_id=chal_id,
        status="awaiting_decision",
    )

    artifacts = compare_experiments(config, champ_id, chal_id)
    comp_id = json.loads(artifacts["promotion_report"].read_text())["comparison_id"]

    result = record_decision(
        config,
        comp_id,
        decision="local_promote",
        rationale="Good for this line only.",
        interpretation="The change is useful within this research line.",
        next_step="Continue the line without replacing the global champion.",
    )

    assert result["decision"] == "local_promote"
    assert result["research_line_id"] == "line_local"
    assert get_official_champion(config.registry_path)["champion_id"] == champ_id
    assert get_research_line(config.registry_path, "line_local")["current_experiment_id"] == chal_id
    comp = next(c for c in list_comparisons(config.registry_path) if c["comparison_id"] == comp_id)
    assert comp["decision"] == "local_promote"


def test_comparison_includes_local_incumbent_cv_evidence(tmp_path: Path) -> None:
    """A line-local verdict gets paired CV evidence against its own incumbent."""
    config, champ_id, chal_id = _setup_two_experiments(tmp_path, "localcv")
    local_id = champ_id

    artifacts = compare_experiments(
        config,
        champ_id,
        chal_id,
        local_incumbent_id=local_id,
    )
    report = json.loads(artifacts["promotion_report"].read_text())

    assert report["local_incumbent_comparison"] is None

    third_config = tmp_path / "localcv_third.toml"
    third_config.write_text(
        """
experiment_name = "localcv_third"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
        encoding="utf-8",
    )
    third_id = json.loads(
        run_experiment(config, third_config)["config_snapshot"].read_text(encoding="utf-8")
    )["experiment_id"]

    artifacts = compare_experiments(
        config,
        champ_id,
        third_id,
        local_incumbent_id=chal_id,
    )
    report = json.loads(artifacts["promotion_report"].read_text())
    local = report["local_incumbent_comparison"]

    assert local["incumbent_id"] == chal_id
    assert local["challenger_id"] == third_id
    assert local["comparison_summary"]["n_independent_units"] >= 1
    assert local["bootstrap_summary"]["uncertainty_method"] == "cluster_bootstrap"


def test_record_decision_promote_blocked_by_guardrail(tmp_path: Path) -> None:
    """A promote is blocked (raises) when the stored guardrail status is a hard fail."""
    import sqlite3
    from autoresearch.comparison_runner import record_decision
    from autoresearch.experiment_registry.registry import get_official_champion, set_official_champion

    config, champ_id, chal_id = _setup_two_experiments(tmp_path, "rdb")
    set_official_champion(
        config.registry_path, champion_id=champ_id, branch_id="main",
        reason="seed", action="initialised", comparison_id="",
    )
    artifacts = compare_experiments(config, champ_id, chal_id)
    comp_id = json.loads(artifacts["promotion_report"].read_text())["comparison_id"]

    # Force a failing guardrail status on the stored comparison row
    failing = json.dumps({"passed": False, "failures": ["gini_above_zero"], "checks": {"gini_above_zero": False}})
    with sqlite3.connect(config.registry_path) as con:
        con.execute("UPDATE comparisons SET guardrail_status = ? WHERE comparison_id = ?", (failing, comp_id))

    with pytest.raises(ValueError, match="guardrail"):
        record_decision(
            config,
            comp_id,
            decision="promote",
            rationale="trying anyway",
            interpretation="The apparent gain conflicts with a hard guardrail.",
            next_step="Fix the guardrail failure before reconsidering promotion.",
        )

    # Champion unchanged
    assert get_official_champion(config.registry_path)["champion_id"] == champ_id


def test_compare_experiments_writes_frequency_html_report(tmp_path: Path) -> None:
    config = replace(
        _make_config(tmp_path),
        target_mode="frequency",
        primary_metric="poisson_deviance",
        min_relative_lift=0.0,
        minimum_win_rate=0.0,
        bootstrap_iterations=5,
        require_diagnostics=False,
    )
    _write_fixtures(config)

    for name in ("champion_frequency", "challenger_frequency"):
        exp_config = tmp_path / f"{name}.toml"
        exp_config.write_text(
            f"""
experiment_name = "{name}"
model_family = "global_mean"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model]
""".strip(),
            encoding="utf-8",
        )
        run_experiment(config, exp_config)

    rows = list_experiments(config.registry_path)
    champion_id = next(row["experiment_id"] for row in rows if row["experiment_name"] == "champion_frequency")
    challenger_id = next(row["experiment_id"] for row in rows if row["experiment_name"] == "challenger_frequency")
    config = replace(config, root=PROJECT_ROOT)

    outputs = compare_experiments(config, champion_id, challenger_id)

    html = outputs["html_report"].read_text(encoding="utf-8")
    assert outputs["html_report"].exists()
    assert "frequency" in html.lower()
