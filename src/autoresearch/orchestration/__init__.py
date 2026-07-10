"""Orchestrated research: an orchestrator campaign over headless sub-agent runs.

A sub-agent run is a *normal tracked run*. Nothing in this package changes the
single-agent workflow: the spawner pre-bootstraps a child run, the handoff grows
an "Orchestration brief" block, and a report is generated from the child's
registry at the end. Everything else — the proposal inbox, ``run-session-cycles``,
``record-decision``, the repair flow — is the existing machinery, unchanged.

Like :mod:`autoresearch.tracks`, this package is a *caller* of public framework
entry points; it never touches the integrity-protected evaluation files.
"""

from __future__ import annotations

__all__ = [
    "brief",
    "backends",
    "manifest",
    "monitor",
    "report",
    "spawner",
]
