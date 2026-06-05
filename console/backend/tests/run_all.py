"""
Run all console backend tests, each in its own subprocess for isolation.

Several tests reload the db module against a temp database (mutating the
AUTORESEARCH_CONSOLE_DB env var). Running each file in a fresh process keeps
that isolation clean. Usage: python console/backend/tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

TEST_FILES = [
    "test_claude_adapter.py",
    "test_codex_opencode_adapters.py",
    "test_db_and_jobs.py",
    "test_job_lifecycle_e2e.py",
    "test_api_integration.py",
]


def main() -> int:
    failures = 0
    for fname in TEST_FILES:
        print(f"\n{'='*50}\nRunning {fname}\n{'='*50}")
        # Fresh env per file (drop any leftover console-db override)
        import os
        env = {k: v for k, v in os.environ.items() if k != "AUTORESEARCH_CONSOLE_DB"}
        result = subprocess.run(
            [sys.executable, str(_HERE / fname)],
            cwd=str(_REPO),
            env=env,
        )
        if result.returncode != 0:
            failures += 1
    print(f"\n{'#'*50}")
    if failures == 0:
        print("ALL TEST FILES PASSED")
    else:
        print(f"{failures} TEST FILE(S) HAD FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
