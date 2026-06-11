#!/usr/bin/env python3
"""Compatibility no-op for Codex sessions started before hook removal.

Current sessions can retain their startup hook configuration in memory. Keeping
this path temporarily prevents those sessions from failing on Stop; new
sessions use the shared workflow-level research-log enforcement instead.
"""

from __future__ import annotations


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
