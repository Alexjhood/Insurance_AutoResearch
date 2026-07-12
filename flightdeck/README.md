# Flight Deck

Flight Deck turns AutoResearch orchestration artifacts into an interactive,
read-only report. It supports a local FastAPI + Vite workflow and standalone
static exports that open directly from `file://`.

## Requirements

- Python 3.11 or newer with the repository development dependencies installed
- Node.js 20 or newer and npm
- An orchestration artifact under `artifacts/orchestrations/<id>`

All commands below run from the repository root.

## Setup

Install the frontend dependencies:

```bash
cd flightdeck/app
npm ci
cd ../..
```

Build one snapshot, or all available snapshots:

```bash
.venv/bin/python -m flightdeck.etl --orchestration 20260711T164959Z
.venv/bin/python -m flightdeck.etl --all
```

Snapshots are generated under `flightdeck/snapshots/` and are safe to rebuild.
Flight Deck never reads live research databases in the browser.

## Local development

Start the API and app in separate terminals:

```bash
.venv/bin/python -m uvicorn flightdeck.server.main:app --host 127.0.0.1 --port 8799
```

```bash
cd flightdeck/app
npm run dev
```

Open `http://127.0.0.1:5199`. The Vite server proxies `/api` to port 8799.

## Verification and production build

```bash
.venv/bin/python -m pytest flightdeck
cd flightdeck/app
npm run typecheck
npm run build
```

The production app is written to `flightdeck/app/dist/`.

## Static export

First build the requested snapshot(s), then run one of:

```bash
.venv/bin/python -m flightdeck.export 20260711T164959Z   # one campaign
.venv/bin/python -m flightdeck.export --all              # every built snapshot
```

This creates both:

- `flightdeck/export/out/<id|all>/index.html`
- `flightdeck/export/out/<id|all>-flightdeck.zip`

The `--all` form embeds the whole Hangar — every campaign, the league table,
and all delegation telemetry — in a single file.

Open `index.html` directly in a browser. It is a single self-contained file:
the app bundle, snapshot, and delegation telemetry are all inlined (browsers
block external module scripts from `file://`, so nothing loads from disk
beside the page itself); no API server or network connection is required.
Evidence files smaller than 512 KB are embedded. Larger files remain listed and
show a clear not-included message when opened.

Use a custom output directory or reuse an existing frontend build with:

```bash
.venv/bin/python -m flightdeck.export 20260711T164959Z --out /tmp/flightdeck-report
.venv/bin/python -m flightdeck.export 20260711T164959Z --skip-app-build
```

`--skip-app-build` reuses `flightdeck/app/dist-export/` (the single-chunk
bundle built by `npm run build:export`), not the regular `dist/`.

## Troubleshooting

**Snapshot not found:** run `.venv/bin/python -m flightdeck.etl --orchestration <id>`.

**Empty Hangar:** confirm `flightdeck/snapshots/index.json` exists and contains
the orchestration. Rebuild with `--force` if source artifacts changed without a
newer timestamp.

**API unavailable:** verify the API is listening on port 8799 and returns JSON
from `http://127.0.0.1:8799/healthz`.

**Static export opens but links fail:** open the generated export's
`index.html`, not `flightdeck/app/dist/index.html`. Export navigation uses URL
hashes intentionally for `file://` compatibility.

**npm reports a cache permission error:** use an isolated cache without changing
system ownership: `npm ci --cache /tmp/flightdeck-npm-cache`.

**A raw file is unavailable in an export:** files at or above 512 KB are
excluded by design. Use the local server workflow to inspect the full file.
