from __future__ import annotations

import argparse
from pathlib import Path

from .build import build_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained Flight Deck report")
    parser.add_argument("orchestration_id", nargs="?", help="Campaign to export (omit with --all)")
    parser.add_argument("--all", action="store_true", help="Embed every built snapshot in one report")
    parser.add_argument("--out", type=Path, help="Output directory (default: flightdeck/export/out/<id|all>)")
    parser.add_argument("--skip-app-build", action="store_true", help="Reuse the existing app/dist-export bundle")
    args = parser.parse_args()
    if bool(args.orchestration_id) == args.all:
        parser.error("pass exactly one of <orchestration_id> or --all")
    folder, archive = build_export(
        None if args.all else args.orchestration_id,
        out=args.out,
        build_app=not args.skip_app_build,
        log=print,
    )
    print(f"Export folder: {folder}")
    print(f"Zip archive: {archive}")


if __name__ == "__main__":
    main()
