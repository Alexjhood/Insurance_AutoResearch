from pathlib import Path
from dataclasses import replace

import pytest

from autoresearch.bootstrap import bootstrap_track
from autoresearch.experiment_registry.registry import get_official_champion, init_registry
from tests.test_handoff import _record_direct
from tests.test_runner import _make_config as _base_config


def _config(tmp_path: Path):
    """Make a config with required model identity for bootstrap tests."""
    return replace(
        _base_config(tmp_path),
        model_provider="anthropic",
        model_name="claude-sonnet-4-6",
    )


def _write_prepared_data_markers(config) -> None:
    for path in (
        config.processed_dir / f"{config.agent_dataset_name}.parquet",
        config.processed_dir / "agent_dataset_search.parquet",
        config.holdout_vault_dir / "agent_dataset_holdout.parquet",
        config.metadata_dir / "dataset_schema.json",
        config.metadata_dir / "dataset_profile.json",
        config.metadata_dir / "capping_diagnostics.json",
        config.splits_dir / "split_pack.csv",
        config.splits_dir / "split_pack_manifest.json",
        config.splits_dir / "split_pack_folds.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def test_bootstrap_track_reuses_existing_baseline_and_exports_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_prepared_data_markers(config)
    _record_direct(config)

    result = bootstrap_track(config)

    assert result["track"] == "test"
    assert result["dataset"] == config.dataset_name
    assert result["target_mode"] == config.target_mode
    assert Path(result["context"]).exists()
    assert Path(result["handoff"]).exists()
    assert get_official_champion(config.registry_path)["champion_id"] == "direct"
    assert [step["step"] for step in result["steps"]] == [
        "prepare-data",
        "test-gate",
        "init-registry",
        "run-starting-baseline",
        "init-official-champion",
        "write-proposal-template",
        "export-context",
    ]
    assert result["steps"][0]["status"] == "skipped"
    assert result["steps"][3]["status"] == "skipped"


def test_bootstrap_reruns_prepare_data_when_holdout_missing(tmp_path: Path, monkeypatch) -> None:
    """A complete-looking data plane that is missing the holdout must not be
    treated as ready — prepare-data has to re-run."""
    config = _config(tmp_path)
    _write_prepared_data_markers(config)
    _record_direct(config)
    # Remove just the holdout parquet, leaving every other artifact in place.
    (config.holdout_vault_dir / "agent_dataset_holdout.parquet").unlink()

    calls: list[bool] = []

    def fake_prepare_data(cfg):
        calls.append(True)
        return {"holdout_dataset": cfg.holdout_vault_dir / "agent_dataset_holdout.parquet"}

    monkeypatch.setattr("autoresearch.bootstrap.prepare_data", fake_prepare_data)

    result = bootstrap_track(config)

    assert calls, "prepare-data should have re-run because the holdout was missing"
    assert result["steps"][0]["status"] == "ran"


def test_bootstrap_track_runs_only_global_mean_config(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    _write_prepared_data_markers(config)
    experiment_dir = config.root / "configs" / "experiments"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "global_mean.toml").write_text("", encoding="utf-8")
    (experiment_dir / "challenger.toml").write_text("", encoding="utf-8")
    calls: list[str] = []

    def fake_run_experiment(cfg, path, *, output_dir=None):
        calls.append(path.name)
        _record_direct(cfg, "direct")
        return {"metrics": cfg.artifacts_dir / "direct" / "metrics.json"}

    monkeypatch.setattr("autoresearch.bootstrap.run_experiment", fake_run_experiment)

    result = bootstrap_track(config)

    assert result["steps"][3]["status"] == "ran"
    assert calls == ["global_mean.toml"]
    assert get_official_champion(config.registry_path)["champion_id"] == "direct"


def test_bootstrap_track_fails_when_global_mean_config_is_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_prepared_data_markers(config)
    experiment_dir = config.root / "configs" / "experiments"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "challenger.toml").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="global-mean starting baseline did not complete"):
        bootstrap_track(config)


def test_bootstrap_track_reports_global_mean_failure(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    _write_prepared_data_markers(config)
    experiment_dir = config.root / "configs" / "experiments"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "global_mean.toml").write_text("", encoding="utf-8")

    def fake_run_experiment(cfg, path, *, output_dir=None):
        raise ValueError("broken baseline")

    monkeypatch.setattr("autoresearch.bootstrap.run_experiment", fake_run_experiment)

    with pytest.raises(ValueError, match=r"global_mean\.toml: broken baseline"):
        bootstrap_track(config)


def test_bootstrap_materialises_run_before_test_gate(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    track_base = tmp_path / "tracks" / "test"
    run_dir = track_base / "runs" / "20260610T072525Z"
    config = replace(
        config,
        track_base_dir=track_base,
        artifacts_dir=run_dir,
        registry_path=run_dir / "registry.sqlite",
        research_log_path=run_dir / "RESEARCH_LOG.md",
        handoff_base_dir=run_dir,
        handoff_context_dir=run_dir / "context",
        handoff_proposal_inbox_dir=run_dir / "proposal_inbox",
        handoff_proposal_processed_dir=run_dir / "proposal_processed",
        handoff_results_dir=run_dir / "results",
        handoff_handoffs_dir=run_dir / "handoffs",
        proposal_inbox_file=run_dir / "proposal_inbox" / "manual_proposals.jsonl",
        run_id="20260610T072525Z",
    )
    _write_prepared_data_markers(config)
    monkeypatch.setattr(
        "autoresearch.bootstrap.ensure_pytest_gate",
        lambda root, cache_dir: {
            "passed": False,
            "cached": False,
            "skipped": False,
            "output": "failed",
        },
    )

    with pytest.raises(ValueError, match="Pytest gate failed"):
        bootstrap_track(config)

    assert (run_dir / "run_manifest.json").exists()
    latest = (track_base / "latest_run.json").read_text(encoding="utf-8")
    assert "20260610T072525Z" in latest


def test_bootstrap_track_requires_named_track(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(config, track_id="default")

    with pytest.raises(ValueError, match="requires --track"):
        bootstrap_track(config)
