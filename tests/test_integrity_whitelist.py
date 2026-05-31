"""Tests for the tightened integrity-scan whitelist (Bug J fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.utils import integrity
from autoresearch.utils.integrity import (
    PROTECTED_RELATIVE_PATHS,
    check_integrity,
    compute_protected_hashes,
    scan_file_for_holdout_access,
    scan_file_for_non_predictive_feature_use,
    write_integrity_manifest,
)

_MARKER_SOURCE = "milestone_holdout = 'bad'"


@pytest.fixture
def fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect _AUTORESEARCH_ROOT to a temp tree so tests never touch src/."""
    fake = tmp_path / "autoresearch"
    (fake / "models").mkdir(parents=True)
    (fake / "data").mkdir(parents=True)
    monkeypatch.setattr(integrity, "_AUTORESEARCH_ROOT", fake.resolve())
    return fake


def test_evil_file_in_models_not_whitelisted(fake_root: Path) -> None:
    """A file named holdout_vault_evil.py inside models/ is NOT whitelisted."""
    evil = fake_root / "models" / "holdout_vault_evil.py"
    evil.write_text(_MARKER_SOURCE, encoding="utf-8")
    violations = scan_file_for_holdout_access(evil)
    assert violations, "Expected a violation for a non-whitelisted file with holdout marker"


def test_real_vault_path_is_whitelisted(fake_root: Path) -> None:
    """data/holdout_vault.py at the exact whitelisted path is not scanned."""
    vault = fake_root / "data" / "holdout_vault.py"
    vault.write_text(_MARKER_SOURCE, encoding="utf-8")
    violations = scan_file_for_holdout_access(vault)
    assert not violations, f"Unexpected violations for whitelisted file: {violations}"


def test_model_script_outside_src_always_scanned(tmp_path: Path) -> None:
    """A model script outside src/autoresearch/ is always scanned regardless of name."""
    script_dir = tmp_path / "artifacts" / "proposal"
    script_dir.mkdir(parents=True)
    # Name it like a whitelisted file to confirm substring bypass is closed.
    script = script_dir / "holdout_vault.py"
    script.write_text(_MARKER_SOURCE, encoding="utf-8")
    violations = scan_file_for_holdout_access(script)
    assert violations, "Model script outside src/autoresearch/ must be scanned regardless of name"


def test_manifest_protects_real_promotion_logic_not_the_shim() -> None:
    """The manifest must hash the real promotion-critical modules, and must NOT
    rely on the experiment_registry/registry.py re-export shim."""
    protected = set(PROTECTED_RELATIVE_PATHS)
    for rel in (
        "src/autoresearch/comparison_runner.py",
        "src/autoresearch/experiment_registry/comparisons.py",
        "src/autoresearch/experiment_registry/champions.py",
        "src/autoresearch/evaluation/diagnostics.py",
    ):
        assert rel in protected, f"{rel} should be integrity-protected"
    assert "src/autoresearch/experiment_registry/registry.py" not in protected, (
        "registry.py is a re-export shim; protecting it does not detect edits to "
        "the real submodules"
    )


def test_check_integrity_detects_edit_to_comparison_runner(tmp_path: Path) -> None:
    """Editing a protected promotion module is flagged by check_integrity."""
    rel = "src/autoresearch/comparison_runner.py"
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ORIGINAL = 1\n", encoding="utf-8")
    # Create the remaining protected files so the manifest is comprehensive.
    for other in PROTECTED_RELATIVE_PATHS:
        p = tmp_path / other
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text("PLACEHOLDER = 0\n", encoding="utf-8")

    artifacts_dir = tmp_path / "artifacts"
    write_integrity_manifest(tmp_path, artifacts_dir)
    assert check_integrity(tmp_path, artifacts_dir) == []

    # Simulate an LLM silently editing the promotion gate.
    target.write_text("ORIGINAL = 999  # tampered\n", encoding="utf-8")
    violations = check_integrity(tmp_path, artifacts_dir)
    assert any("comparison_runner.py" in v for v in violations), violations


def test_exposure_in_predictor_list_is_rejected(tmp_path: Path) -> None:
    script = tmp_path / "model.py"
    script.write_text(
        'NUMERIC = ["exposure_term_a", "driver_age_band_d"]\n'
        'EXPOSURE = "exposure_term_a"\n'
        'def fit_predict(train, score, **kw):\n'
        '    weight = train[EXPOSURE]\n'
        '    return weight.to_numpy(), {}\n',
        encoding="utf-8",
    )

    violations = scan_file_for_non_predictive_feature_use(script)

    assert violations
    assert "predictor container" in violations[0]


def test_exposure_weight_usage_is_allowed(tmp_path: Path) -> None:
    script = tmp_path / "model.py"
    script.write_text(
        'EXPOSURE = "exposure_term_a"\n'
        'def fit_predict(train, score, **kw):\n'
        '    train_exp = train[EXPOSURE].to_numpy()\n'
        '    return score[EXPOSURE].to_numpy() * train_exp.mean(), {}\n',
        encoding="utf-8",
    )

    assert scan_file_for_non_predictive_feature_use(script) == []
