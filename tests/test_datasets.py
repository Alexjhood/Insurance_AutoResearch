"""Phase 1 — DatasetSpec loading/validation and config dataset resolution."""

from __future__ import annotations

import json

import pytest

from autoresearch.config import load_config
from autoresearch.datasets import (
    DATASETS_CONFIG_DIR,
    DatasetSpec,
    list_datasets,
    load_dataset_spec,
)


def test_registered_datasets_present():
    names = set(list_datasets())
    assert {"french_motor", "allstate", "allstate_full", "porto_seguro"} <= names


def test_every_registered_dataset_loads_and_validates():
    for name in list_datasets():
        spec = load_dataset_spec(name)
        assert isinstance(spec, DatasetSpec)
        assert spec.name == name
        assert spec.default_target_mode in spec.target_modes


def test_french_spec_core_fields():
    spec = load_dataset_spec("french_motor")
    assert spec.loader == "adapter"
    assert spec.adapter_module == "autoresearch.data.adapters.french_motor"
    assert spec.id_column == "IDpol"
    assert spec.weight_column == "Exposure"
    assert spec.count_column == "ClaimNb"
    assert spec.event_count_column == "ClaimAmountCount"
    assert spec.cap is not None
    assert spec.cap.column == "ClaimAmount"
    assert spec.cap.output_column == "ClaimAmountCapped"
    assert spec.cap.threshold == 100000
    assert spec.structural_gini_threshold == 0.37
    assert spec.target_modes == ("burning_cost", "frequency", "severity")


def test_allstate_sample_and_weight():
    spec = load_dataset_spec("allstate")
    assert spec.weight_column is None
    assert spec.synthesises_unit_weight is True
    assert spec.sample is not None
    assert spec.sample.unit_column == "Household_ID"
    assert spec.sample.rows == 2_000_000
    assert spec.split_unit_column == "Household_ID"
    assert spec.count_column is None  # no frequency_severity


def test_allstate_full_overrides_compute():
    spec = load_dataset_spec("allstate_full")
    assert spec.sample is None
    assert spec.overrides["compute"]["base_budget_minutes"] == 40


def test_porto_na_marker():
    spec = load_dataset_spec("porto_seguro")
    assert spec.na_marker is not None
    assert spec.na_marker.value == -1
    assert spec.na_marker.columns_matching == "_cat$"
    assert spec.structural_gini_threshold is None


def test_unknown_dataset_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset_spec("does_not_exist")


def test_missing_default_target_rejected(tmp_path, monkeypatch):
    bad = DATASETS_CONFIG_DIR / "_tmp_bad_default.toml"
    bad.write_text(
        """
[dataset]
name = "_tmp_bad_default"
default_target_mode = "a"
[source]
id_column = "id"
[[targets]]
mode = "a"
source_column = "x"
rate_label = "r"
entity_label = "e"
[[targets]]
mode = "b"
source_column = "y"
rate_label = "r"
entity_label = "e"
""",
        encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="exactly one"):
            load_dataset_spec("_tmp_bad_default")
    finally:
        bad.unlink()


# --- config-level dataset resolution ---------------------------------------


def test_load_config_defaults_to_french():
    config = load_config()
    assert config.dataset_name == "french_motor"
    assert config.id_column == "IDpol"
    assert config.claim_capping_enabled is True
    assert config.claim_cap_threshold == 100000
    assert "datasets/french_motor" in str(config.processed_dir)


def test_load_config_dataset_paths_and_cap_for_porto():
    config = load_config(dataset="porto_seguro")
    assert config.dataset_name == "porto_seguro"
    assert config.id_column == "id"
    assert config.claim_capping_enabled is False
    assert "datasets/porto_seguro" in str(config.processed_dir)
    assert "datasets/porto_seguro" in str(config.holdout_vault_dir)


def test_allstate_full_compute_override_applied():
    config = load_config(dataset="allstate_full")
    assert config.base_budget_minutes == 40
    assert config.budget_increment_minutes == 15
    assert config.preflight_sample_rows == 20000


def _isolated_config(tmp_path):
    """Write a config whose artifact paths live entirely under tmp_path."""

    from autoresearch.config import DEFAULT_CONFIG_PATH
    import tomllib

    with DEFAULT_CONFIG_PATH.open("rb") as f:
        raw = tomllib.load(f)
    artifacts = tmp_path / "artifacts"
    raw["paths"]["artifacts_dir"] = str(artifacts)
    raw["paths"]["registry_path"] = str(artifacts / "experiment_registry.sqlite")
    # tomllib has no dumper; emit the two keys we overrode plus a passthrough copy.
    cfg_path = tmp_path / "config.toml"
    _dump_toml(raw, cfg_path)
    return cfg_path


def _dump_toml(raw: dict, path) -> None:
    # Minimal round-trip good enough for load_config: re-read default.toml text
    # and splice absolute artifact paths in. Simpler: use tomli_w if available,
    # else write via json-like structure is not valid TOML — so shell out to the
    # original file text with a [paths] override appended (last section wins in
    # our loader because we read whole tables, so we instead rewrite [paths]).
    import re

    from autoresearch.config import DEFAULT_CONFIG_PATH

    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    paths = raw["paths"]
    new_paths = "[paths]\n" + "".join(
        f'{k} = "{v}"\n' for k, v in paths.items()
    )
    text = re.sub(r"\[paths\].*?(?=\n\[)", new_paths + "\n", text, count=1, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def test_manifest_dataset_mismatch_guard(tmp_path):
    # Pin a fresh timestamped run manifest to allstate in an isolated artifacts
    # tree, then a contradicting --dataset must be rejected.
    from autoresearch.config import ensure_project_dirs

    cfg = _isolated_config(tmp_path)
    config = load_config(cfg, track_id="claude", new_run=True)
    ensure_project_dirs(config)
    manifest = config.artifacts_dir / "run_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["dataset"] == "french_motor"  # written by ensure_project_dirs
    payload["dataset"] = "allstate"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_id = config.run_id
    ok = load_config(cfg, track_id="claude", run_id=run_id, dataset="allstate")
    assert ok.dataset_name == "allstate"
    with pytest.raises(ValueError, match="contradicts"):
        load_config(cfg, track_id="claude", run_id=run_id, dataset="porto_seguro")
    resolved = load_config(cfg, track_id="claude", run_id=run_id)
    assert resolved.dataset_name == "allstate"
