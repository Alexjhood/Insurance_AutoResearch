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
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from autoresearch.feature_policy import NON_PREDICTIVE_COLUMNS


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

# Exact paths relative to src/autoresearch/ that may reference holdout markers.
_SCAN_WHITELIST = frozenset(
    {
        "data/holdout_vault.py",
        "milestone.py",
        "models/dispatcher.py",
        "utils/integrity.py",
    }
)

_AUTORESEARCH_ROOT = Path(__file__).resolve().parent.parent  # src/autoresearch/


def _file_is_whitelisted(path: Path) -> bool:
    """Return True only for exact-path matches under src/autoresearch/."""
    try:
        rel = path.resolve().relative_to(_AUTORESEARCH_ROOT)
        return str(rel) in _SCAN_WHITELIST
    except ValueError:
        # Path is not under src/autoresearch/ — always scan, never whitelist
        return False


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


_FEATURE_CONTAINER_NAMES = frozenset(
    {
        "features",
        "feature_columns",
        "feature_inclusions",
        "all_features",
        "all_feats",
        "base_features",
        "base_numeric",
        "numeric",
        "numeric_features",
        "numeric_feats",
        "num_features",
        "num_feats",
        "predictors",
        "predictor_columns",
        "x_columns",
        "x_cols",
    }
)


def scan_file_for_non_predictive_feature_use(path: Path) -> list[str]:
    """Flag common attempts to place reserved columns in predictor lists.

    The exposure column is allowed in scripts for weights, offsets, and
    response/rate calculations, so this deliberately scans only feature-like
    containers rather than every occurrence of the string.
    """

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    violations: list[str] = []
    for node in ast.walk(tree):
        target_names: list[str] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            target_names = [_target_name(target) for target in node.targets]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target_names = [_target_name(node.target)]
            value = node.value
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg:
                    target_names.append(keyword.arg)
                    if _is_feature_container_name(keyword.arg):
                        for column in _string_literals(keyword.value).intersection(NON_PREDICTIVE_COLUMNS):
                            violations.append(_non_predictive_message(path, column, keyword.arg))
                continue

        if value is None:
            continue
        feature_targets = [name for name in target_names if _is_feature_container_name(name)]
        if not feature_targets:
            continue
        for column in _string_literals(value).intersection(NON_PREDICTIVE_COLUMNS):
            for name in feature_targets:
                violations.append(_non_predictive_message(path, column, name))
    return sorted(set(violations))


def _target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _is_feature_container_name(name: str) -> bool:
    lowered = name.lower()
    if "non_predictive" in lowered or "reserved" in lowered or "exposure" in lowered:
        return False
    return lowered in _FEATURE_CONTAINER_NAMES or "feature" in lowered or "predictor" in lowered


def _string_literals(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }


def _non_predictive_message(path: Path, column: str, container: str) -> str:
    return (
        f"{path}: {column!r} appears in predictor container {container!r}; "
        "exposure may be used only for weights, offsets, response denominators, "
        "and converting predicted rates to target totals"
    )


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

# Files that define the evaluation metrics, the promotion gate, and the
# holdout boundary.  These are the modules an LLM could silently edit to make a
# weak challenger look promotable (reward hacking), so any change blocks
# comparisons until explicitly accepted via `update-integrity-manifest`.
#
# Note: we hash the *real* implementation modules, not the
# `experiment_registry/registry.py` re-export shim — editing a submodule such as
# `comparisons.py` does not change the shim's bytes and would otherwise slip
# through unprotected.
PROTECTED_RELATIVE_PATHS = [
    # Metrics and resampling/promotion-decision logic
    "src/autoresearch/evaluation/metrics.py",
    "src/autoresearch/evaluation/resampling.py",
    "src/autoresearch/evaluation/diagnostics.py",
    "src/autoresearch/evaluation/validation.py",
    # Comparison orchestration and the Bonferroni / integrity gate itself
    "src/autoresearch/comparison_runner.py",
    # Holdout boundary and milestone evaluation
    "src/autoresearch/data/holdout_vault.py",
    "src/autoresearch/milestone.py",
    # Registry submodules that record comparisons and set the official champion
    "src/autoresearch/experiment_registry/comparisons.py",
    "src/autoresearch/experiment_registry/champions.py",
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

_TEST_GATE_CACHE_FILENAME = "test_gate_cache.json"
_TEST_GATE_CACHE_VERSION = 1
_TEST_GATE_DIRS = ("src", "tests", "configs", "scripts")
_TEST_GATE_SUFFIXES = frozenset({".py", ".toml", ".json", ".yaml", ".yml"})
_TEST_GATE_FILES = (
    "AGENT.md",
    "pyproject.toml",
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "conftest.py",
)


def _pytest_skip_reason(root: Path) -> str | None:
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AUTORESEARCH_SKIP_PYTEST_GATE"):
        return "running inside test suite"
    if not (root / "tests").exists():
        return "tests/ directory not found"
    return None


def _test_gate_files(root: Path) -> list[Path]:
    """Return files whose contents can affect the repository pytest gate."""

    files: set[Path] = set()
    for dirname in _TEST_GATE_DIRS:
        base = root / dirname
        if base.exists():
            files.update(
                path
                for path in base.rglob("*")
                if path.is_file() and path.suffix.lower() in _TEST_GATE_SUFFIXES
            )
    files.update(path for name in _TEST_GATE_FILES if (path := root / name).is_file())
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def compute_test_gate_fingerprint(root: Path) -> str:
    """Hash test-relevant files and the active Python package environment."""

    digest = hashlib.sha256()
    digest.update(f"cache-version:{_TEST_GATE_CACHE_VERSION}\n".encode())
    for path in _test_gate_files(root):
        relative = path.relative_to(root)
        digest.update(f"file:{relative.as_posix()}\0".encode())
        digest.update(path.read_bytes())
        digest.update(b"\0")

    runtime = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": sorted(
            f"{dist.metadata.get('Name', '')}=={dist.version}"
            for dist in importlib.metadata.distributions()
        ),
    }
    digest.update(json.dumps(runtime, sort_keys=True).encode())
    return digest.hexdigest()


def ensure_pytest_gate(root: Path, cache_dir: Path) -> dict[str, Any]:
    """Run pytest once per test-relevant code and environment fingerprint."""

    skip_reason = _pytest_skip_reason(root)
    if skip_reason:
        return {
            "passed": True,
            "cached": False,
            "skipped": True,
            "output": f"skipped ({skip_reason})",
        }

    fingerprint = compute_test_gate_fingerprint(root)
    cache_path = cache_dir / _TEST_GATE_CACHE_FILENAME
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("passed") is True and cached.get("fingerprint") == fingerprint:
            return {**cached, "cached": True, "skipped": False, "cache_path": str(cache_path)}

    started = time.perf_counter()
    passed, output = run_pytest(root)
    result: dict[str, Any] = {
        "passed": passed,
        "cached": False,
        "skipped": False,
        "fingerprint": fingerprint,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "output": output,
        "cache_path": str(cache_path),
    }
    if passed:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(cache_path)
    return result


def run_pytest(root: Path) -> tuple[bool, str]:
    """Run the test suite and return (passed, output_summary).

    Returns (True, "skipped") when already running inside pytest to prevent
    infinite recursion.  Set ``AUTORESEARCH_SKIP_PYTEST_GATE=1`` to disable
    the gate in CI or other non-interactive contexts.
    """
    skip_reason = _pytest_skip_reason(root)
    if skip_reason:
        return True, f"skipped ({skip_reason})"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q", str(root / "tests")],
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
