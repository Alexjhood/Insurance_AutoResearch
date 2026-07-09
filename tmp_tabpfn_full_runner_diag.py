#!/usr/bin/env python3
from __future__ import annotations

import faulthandler
import os
import sys
from pathlib import Path

faulthandler.enable(file=sys.stderr, all_threads=True)
os.environ.setdefault("AUTORESEARCH_FOUNDATION_MODELS", "1")

from autoresearch.config import load_config
from autoresearch.experiment_runner import run_experiment


def main() -> int:
    cfg = load_config(
        Path("configs/frugal.toml"),
        track_id="claude",
        run_id="20260703T144352Z",
    )
    run_experiment(
        cfg,
        Path(
            "artifacts/tracks/claude/runs/20260703T144352Z/"
            "iterations/003_tabpfn_api_context_1000_20260703T144811945430Z/"
            "proposal/experiment_config_attempt_1.toml"
        ),
        output_dir=Path("/private/tmp/tabpfn_full_runner_lldb_current"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
