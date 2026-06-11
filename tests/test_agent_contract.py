"""Guards for the generated agent runtime contract (``AGENT.md``).

These tests enforce the property that makes a generated contract worthwhile: it
cannot silently drift from the code. They fail if ``AGENT.md`` is stale relative
to its sources, if it names a command the CLI does not expose, or if it bloats
back toward the old 48 KB manual.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the generator as a module (it lives under scripts/, not an importable pkg).
_spec = importlib.util.spec_from_file_location(
    "generate_agent_contract", REPO_ROOT / "scripts" / "generate_agent_contract.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

from autoresearch.cli import COMMANDS  # noqa: E402

AGENT_MD = REPO_ROOT / "AGENT.md"

# Generous ceiling: guards against regression toward the old 48 KB manual while
# leaving room for accurate, code-derived content. The contract is ~11.7 KB
# today (objective, safety, workflow, repair, model interface + calibration,
# adaptive search, decision policy, proposal contract).
SIZE_CEILING_BYTES = 15_000


def test_agent_md_is_in_sync_with_sources():
    """AGENT.md must equal freshly generated output (run the generator to fix)."""
    expected = gen.render()
    actual = AGENT_MD.read_text(encoding="utf-8")
    assert actual == expected, (
        "AGENT.md is out of sync with code/config. "
        "Regenerate with: python scripts/generate_agent_contract.py"
    )


def test_harness_mirrors_match_agent_md():
    """AGENTS.md / CLAUDE.md must be byte-identical copies of AGENT.md.

    These are the harness-native auto-load files (Codex/OpenCode read AGENTS.md,
    Claude Code reads CLAUDE.md); they must not drift from the canonical contract.
    """
    expected = gen.render()
    for mirror in gen.AGENT_MIRRORS:
        assert mirror.exists(), f"{mirror.name} missing; run the generator."
        assert mirror.read_text(encoding="utf-8") == expected, (
            f"{mirror.name} is out of sync. "
            "Regenerate with: python scripts/generate_agent_contract.py"
        )


def test_workflow_commands_exist_in_cli():
    """Every command the contract renders must be a real CLI subcommand."""
    missing = [name for name, _ in gen.WORKFLOW_COMMANDS if name not in COMMANDS]
    assert not missing, f"Contract names commands absent from cli.COMMANDS: {missing}"


def test_no_unknown_autoresearch_command_in_contract():
    """No `autoresearch <token>` in AGENT.md may reference a non-existent command.

    This is the spec's "test that every command shown in it exists" check, and is
    what would have caught the stale `compare-to-champion`/auto-promote guidance.
    """
    text = AGENT_MD.read_text(encoding="utf-8")
    # Tokens that immediately follow `autoresearch`, skipping global flags.
    global_flags = {"--track", "--run-id", "--new-run", "--config", "--target-mode"}
    referenced: set[str] = set()
    for match in re.finditer(r"autoresearch((?:\s+(?:--[\w-]+|<[^>]+>))*)\s+([a-z][a-z-]+)", text):
        referenced.add(match.group(2))
    # Also treat the rendered command bullets (`- \`name\` — ...`) as references.
    for match in re.finditer(r"^- `([a-z][a-z-]+)`", text, flags=re.MULTILINE):
        referenced.add(match.group(1))
    referenced -= global_flags
    unknown = sorted(c for c in referenced if c not in COMMANDS)
    assert not unknown, f"AGENT.md references unknown commands: {unknown}"


def test_contract_stays_compact():
    size = AGENT_MD.stat().st_size
    assert size <= SIZE_CEILING_BYTES, (
        f"AGENT.md grew to {size} bytes (ceiling {SIZE_CEILING_BYTES}). "
        "Keep the runtime contract compact; move detail to docs/OPERATING_MANUAL.md."
    )
