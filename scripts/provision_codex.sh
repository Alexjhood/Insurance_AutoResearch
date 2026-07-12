#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root/tools/codex"
npm install --ignore-scripts --no-audit --no-fund
"$repo_root/tools/codex/node_modules/.bin/codex" --version
