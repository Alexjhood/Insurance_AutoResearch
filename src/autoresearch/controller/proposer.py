"""File-handoff proposal provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class Proposer(Protocol):
    """Protocol for structured experiment proposal providers."""

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        """Return raw response text and parsed proposal when available."""


class FileProposer:
    """Read the next proposal from a local JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def propose(self, context: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        if not self.path.exists():
            raise FileNotFoundError(f"Proposal file does not exist: {self.path}")
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                proposal = json.loads(line)
                return line, proposal
        raise ValueError(f"Proposal file contains no JSON proposals: {self.path}")


def proposer_from_config(config) -> Proposer:
    """Create the file-handoff proposal provider."""
    return FileProposer(config.proposal_inbox_file)
