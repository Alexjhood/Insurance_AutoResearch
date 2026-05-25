"""Code-integrity scan and protected-file manifest.

Two distinct checks:

1. **Holdout-access scan** — AST-walk every .py file under
   ``src/autoresearch/models/`` and ``src/autoresearch/features/`` and
   reject any that reference the holdout vault (except the vault module
   and milestone evaluator themselves, which are whitelisted).

2. **Protected-file manifest** — SHA256 hashes of core evaluation and
   registry files, written at ``init-registry`` time and verified before
   any comparison runs.  A mismatch means the LLM silently edited the
   metric or promotion gate, which is the primary reward-hacking risk.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


# ── Holdout-access scan ───────────────────────────────────────────────────────

_HOLDOUT_MARKERS = frozenset(
    {
        "milestone_holdout",
        "holdout_vault",
        "load_holdout_dataset",
        "AUTORESEARCH_MILESTONE_TOKEN",
        "agent_dataset_holdout",
    }
)

# Files that are *allowed* to reference holdout markers.
# This covers core framework files that legitimately check for the guard string
# and the vault/milestone modules themselves.
_SCAN_WHITELIST = frozenset(
    {
        "holdout_vault.py",
        "milestone.py",
        "dispatcher.py",   # contains the guard check itself
        "baselines.py",    # original framework baseline
        "integrity.py",    # this file
        "test_holdout",
        "test_split",
        "test_vault",
    }
)


def _file_is_whitelisted(path: Path) -> bool:
    return any(w in path.name for w in _SCAN_WHITELIST)


def _ast_strings(tree: ast.AST) -> list[str]:
    """Return all string literals in an AST."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            results.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            # f-strings: scan their value nodes
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    results.append(child.value)
    return results


def _ast_names(tree: ast.AST) -> list[str]:
    """Return all Name and Attribute ids in an AST."""
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            results.append(node.id)
        elif isinstance(node, ast.Attribute):
            results.append(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                results.append(node.module)
            for alias in node.names:
                results.append(alias.name)
    return results


def scan_file_for_holdout_access(path: Path) -> list[str]:
    """Return a list of violation messages for a single Python file."""

    if _file_is_whitelisted(path):
        return []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []  # let pytest catch syntax errors

    violations = []
    all_tokens = _ast_strings(tree) + _ast_names(tree)
    for token in all_tokens:
        for marker in _HOLDOUT_MARKERS:
            if marker in token:
                violations.append(
                    f"{path}: references holdout marker {marker!r} — "
                    "model and feature files must not access the holdout vault"
                )
                break
    return violations


def scan_for_holdout_access(root: Path) -> list[str]:
    """Scan all model and feature Python files for holdout references."""

    scan_dirs = [
        root / "src" / "autoresearch" / "models",
        root / "src" / "autoresearch" / "features",
    ]
    violations: list[str] = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py_file in sorted(scan_dir.rglob("*.py")):
            violations.extend(scan_file_for_holdout_access(py_file))
    return violations


# ── Protected-file manifest ───────────────────────────────────────────────────

PROTECTED_RELATIVE_PATHS = [
    "src/autoresearch/evaluation/metrics.py",
    "src/autoresearch/evaluation/resampling.py",
    "src/autoresearch/data/holdout_vault.py",
    "src/autoresearch/experiment_registry/registry.py",
]

_MANIFEST_FILENAME = "integrity_manifest.json"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def compute_protected_hashes(root: Path) -> dict[str, str]:
    """Compute SHA256 hashes of protected files."""

    result: dict[str, str] = {}
    for rel in PROTECTED_RELATIVE_PATHS:
        p = root / rel
        if p.exists():
            result[rel] = _hash_file(p)
    return result


def write_integrity_manifest(root: Path, artifacts_dir: Path) -> Path:
    """Write the integrity manifest to artifacts_dir and return its path."""

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    hashes = compute_protected_hashes(root)
    manifest: dict[str, Any] = {
        "protected_files": hashes,
        "note": (
            "SHA256 hashes of files that define the promotion gate and evaluation metrics. "
            "Any change to these files will block comparisons until "
            "`autoresearch update-integrity-manifest` is run to explicitly accept the change."
        ),
    }
    manifest_path = artifacts_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def check_integrity(root: Path, artifacts_dir: Path) -> list[str]:
    """Return violation messages if protected files have changed since manifest was written."""

    manifest_path = artifacts_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        # No manifest yet — first run or legacy project
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded: dict[str, str] = manifest.get("protected_files", {})
    current = compute_protected_hashes(root)
    violations = []
    for rel, expected_hash in recorded.items():
        actual = current.get(rel)
        if actual is None:
            violations.append(f"Protected file missing: {rel}")
        elif actual != expected_hash:
            violations.append(
                f"Protected file changed: {rel} — "
                "run `autoresearch update-integrity-manifest` to accept the change"
            )
    return violations


# ── Pytest gate ───────────────────────────────────────────────────────────────

def run_pytest(root: Path) -> tuple[bool, str]:
    """Run the test suite and return (passed, output_summary).

    Returns (True, "skipped") when already running inside pytest to prevent
    infinite recursion.  Set ``AUTORESEARCH_SKIP_PYTEST_GATE=1`` to disable
    the gate in CI or other non-interactive contexts.
    """
    import os

    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AUTORESEARCH_SKIP_PYTEST_GATE"):
        return True, "skipped (running inside test suite)"

    tests_dir = root / "tests"
    if not tests_dir.exists():
        return True, "skipped (tests/ directory not found)"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q", str(tests_dir)],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**os.environ, "AUTORESEARCH_SKIP_PYTEST_GATE": "1"},
    )
    passed = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    lines = output.splitlines()
    summary = "\n".join(lines[-60:]) if len(lines) > 60 else output
    return passed, summary
