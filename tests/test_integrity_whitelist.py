"""Tests for the tightened integrity-scan whitelist (Bug J fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.utils import integrity
from autoresearch.utils.integrity import (
    PROTECTED_RELATIVE_PATHS,
    check_integrity,
    ensure_pytest_gate,
    compute_protected_hashes,
    scan_file_for_holdout_access,
    scan_file_for_non_predictive_feature_use,
    write_integrity_manifest,
)

_MARKER_SOURCE = "milestone_holdout = 'bad'"


def _enable_pytest_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AUTORESEARCH_SKIP_PYTEST_GATE", raising=False)


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
        'NUMERIC = ["Exposure", "DrivAge"]\n'
        'EXPOSURE = "Exposure"\n'
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
        'EXPOSURE = "Exposure"\n'
        'def fit_predict(train, score, **kw):\n'
        '    train_exp = train[EXPOSURE].to_numpy()\n'
        '    return score[EXPOSURE].to_numpy() * train_exp.mean(), {}\n',
        encoding="utf-8",
    )

    assert scan_file_for_non_predictive_feature_use(script) == []


def test_pytest_gate_reuses_success_for_unchanged_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tests").mkdir()
    source = tmp_path / "src" / "package.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _enable_pytest_gate(monkeypatch)
    calls: list[Path] = []

    def fake_run_pytest(root: Path) -> tuple[bool, str]:
        calls.append(root)
        return True, "1 passed"

    monkeypatch.setattr(integrity, "run_pytest", fake_run_pytest)

    first = ensure_pytest_gate(tmp_path, tmp_path / "artifacts")
    second = ensure_pytest_gate(tmp_path, tmp_path / "artifacts")

    assert first["cached"] is False
    assert second["cached"] is True
    assert calls == [tmp_path]


def test_pytest_gate_invalidates_when_relevant_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tests").mkdir()
    source = tmp_path / "src" / "package.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _enable_pytest_gate(monkeypatch)
    calls = 0

    def fake_run_pytest(root: Path) -> tuple[bool, str]:
        nonlocal calls
        calls += 1
        return True, "1 passed"

    monkeypatch.setattr(integrity, "run_pytest", fake_run_pytest)

    ensure_pytest_gate(tmp_path, tmp_path / "artifacts")
    source.write_text("VALUE = 2\n", encoding="utf-8")
    result = ensure_pytest_gate(tmp_path, tmp_path / "artifacts")

    assert result["cached"] is False
    assert calls == 2


def test_pytest_gate_does_not_cache_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tests").mkdir()
    _enable_pytest_gate(monkeypatch)
    results = iter([(False, "failed"), (True, "passed")])

    monkeypatch.setattr(integrity, "run_pytest", lambda root: next(results))

    first = ensure_pytest_gate(tmp_path, tmp_path / "artifacts")
    second = ensure_pytest_gate(tmp_path, tmp_path / "artifacts")

    assert first["passed"] is False
    assert second["passed"] is True
    assert second["cached"] is False
