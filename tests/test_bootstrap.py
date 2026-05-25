from pathlib import Path
from dataclasses import replace

import pytest

from autoresearch.bootstrap import bootstrap_track
from autoresearch.experiment_registry.registry import get_official_champion, init_registry
from tests.test_handoff import _record_direct
from tests.test_phase4_controller import _config


def _write_prepared_data_markers(config) -> None:
    for path in (
        config.processed_dir / f"{config.agent_dataset_name}.parquet",
        config.processed_dir / "agent_dataset_search.parquet",
        config.metadata_dir / "agent_schema.json",
        config.metadata_dir / "dataset_profile.json",
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
    assert Path(result["context"]).exists()
    assert Path(result["handoff"]).exists()
    assert get_official_champion(config.registry_path)["champion_id"] == "direct"
    assert [step["step"] for step in result["steps"]] == [
        "prepare-data",
        "init-registry",
        "run-all-baselines",
        "init-official-champion",
        "write-proposal-template",
        "export-context",
    ]
    assert result["steps"][0]["status"] == "skipped"
    assert result["steps"][2]["status"] == "skipped"


def test_bootstrap_track_keeps_successful_baseline_when_another_fails(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    _write_prepared_data_markers(config)
    (config.root / "configs" / "experiments").mkdir(parents=True)
    (config.root / "configs" / "experiments" / "01_good.toml").write_text("", encoding="utf-8")
    (config.root / "configs" / "experiments" / "02_bad.toml").write_text("", encoding="utf-8")

    def fake_run_experiment(cfg, path, *, output_dir=None):
        if path.name == "02_bad.toml":
            raise ValueError("bad baseline")
        _record_direct(cfg, "direct")
        return {"metrics": cfg.artifacts_dir / "direct" / "metrics.json"}

    monkeypatch.setattr("autoresearch.bootstrap.run_experiment", fake_run_experiment)

    result = bootstrap_track(config)

    assert result["steps"][2]["status"] == "ran_with_errors"
    assert "02_bad.toml: bad baseline" in result["steps"][2]["errors"]
    assert get_official_champion(config.registry_path)["champion_id"] == "direct"


def test_bootstrap_track_requires_named_track(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(config, track_id="default")

    with pytest.raises(ValueError, match="requires --track"):
        bootstrap_track(config)
