"""Tests for the tightened integrity-scan whitelist (Bug J fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.utils import integrity
from autoresearch.utils.integrity import scan_file_for_holdout_access

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
