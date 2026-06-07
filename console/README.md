# AutoResearch Console

Web front-end for the Insurance AutoResearch loop. Provides run launch, live monitoring,
steering, cross-run leaderboard, and artifact inspection — all in one place.

## Architecture

```
Next.js UI (port 3000) → FastAPI orchestrator (port 8765) → autoresearch library + agents
```

See [`docs/frontend_design.md`](../docs/frontend_design.md) for the full design reference.

## Setup

### Backend (FastAPI orchestrator)

```bash
cd console/backend

# Install into the same venv as autoresearch
source ../../.venv/bin/activate
pip install fastapi uvicorn[standard] aiofiles

# Start the orchestrator
uvicorn main:app --host 127.0.0.1 --port 8765 --reload
```

The orchestrator reads from the autoresearch `src/` package. Make sure
`autoresearch` is installed (`pip install -e ../../` from the repo root).

### Frontend (Next.js)

```bash
cd console/frontend
npm install
npm run dev        # http://localhost:3000
```

The Next.js config proxies `/api/*` → `http://127.0.0.1:8765/api/*`.

### Quick start (both together)

```bash
# Terminal 1
cd console/backend && source ../../.venv/bin/activate && uvicorn main:app --port 8765 --reload

# Terminal 2
cd console/frontend && npm run dev
```

Then open http://localhost:3000.

## What each page does

| Page | Phase | Notes |
|---|---|---|
| `/` | 1 | Dashboard: track summary, top experiments |
| `/runs` | 1 | All runs with track selector |
| `/runs/[track]/[runId]` | 1 | Full run detail: LLM telemetry, experiments, comparisons, research lines, artifact browser |
| `/leaderboard` | 1 | Cross-run search-split leaderboard + gated holdout panel |
| `/launch` | 2 | Launch form — creates a git worktree and starts an agent job |
| `/monitor` | 2 | Job roster |
| `/monitor/[jobId]` | 2 | Live token stream, tool-call log, steer panel, stop button |

## Prerequisite for launching agents (Phase 2+)

Read-only pages (Phase 1) work with no agent setup. To **launch** a run, the chosen
agent's CLI must be installed *and authenticated* on the machine running the orchestrator:

- **Claude**: install Claude Code and run `claude login` (or set `ANTHROPIC_API_KEY`).
  The headless desktop-app binary by itself reports "Not logged in" until authenticated.
- **Codex**: `brew install codex` / `npm i -g @openai/codex`, then authenticate.
- **OpenCode**: install `opencode` and configure provider keys.

The Launch page greys out any surface whose CLI the orchestrator can't find, and shows
the binary status at `GET /api/health`.

## Phase status

- **Phase 1** (read-only console): ✅ complete, tested against real registry data
- **Phase 2** (launch + monitor): ✅ complete — Claude adapter parses real `stream-json`;
  worktree spawn, live WebSocket stream, pause/stop. Needs an authenticated `claude` CLI.
- **Phase 3** (steer + reattach): ✅ best-effort interrupt + between-turn fallback;
  `agent_session_id` captured for `--resume` reattach after orchestrator restart
- **Phase 4** (Codex + OpenCode adapters): ✅ rewritten against the **real** CLI event
  formats (codex-cli 0.137, opencode 1.16) and tested. Codex is **verified end-to-end live**
  (real run emits tokens/tool/turn/exit + thread id for resume). OpenCode integration is
  verified (spawns, captures session id, parses documented events, clean exit) but the
  OpenCode CLI returned sparse/empty output for several models on this machine — a
  provider/model/auth issue to resolve in OpenCode, not in the adapter.
- **Phase 5** (gated holdout + cloud seam): ✅ holdout endpoint (token never stored),
  `Runner` abstraction, `AuthMiddleware` (no-op → bearer/OIDC), configurable API base

## Known limitations

- **OpenCode CLI output is environment-dependent.** On this machine `opencode run
  --format json` returned empty/sparse output for several OpenRouter models (the adapter
  still captures the session id and exits cleanly). Use a model that reliably returns
  output, or check `opencode auth`/credits. Codex is verified working end-to-end.
- **WebSocket auth** bypasses `AuthMiddleware` (Starlette `BaseHTTPMiddleware` only wraps
  HTTP). Fine for local single-user; add token-in-query auth before any networked deploy.
- **Reattach** after orchestrator restart marks live jobs `interrupted` and can `--resume`
  Claude by session id, but cannot recover the original stdout pipe — new output only flows
  after a resume.

## Running the tests

```bash
source ../.venv/bin/activate        # from repo root: source .venv/bin/activate
python console/backend/tests/run_all.py
```

24 tests across 4 files: Claude stream-json parser, DB/steer-queue/reattach, end-to-end
job lifecycle (stub adapter), and HTTP+WebSocket integration. Frontend: `npm run build`
(from `console/frontend`) type-checks and compiles all pages.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `AUTORESEARCH_CONSOLE_PORT` | `8765` | Orchestrator port |
| `AUTORESEARCH_CONSOLE_HOST` | `127.0.0.1` | Orchestrator bind host |
| `AUTORESEARCH_CONSOLE_DB` | `console/orchestrator.db` | Orchestrator state DB |
| `AUTORESEARCH_MEMORY_DIR` | `~/.autoresearch/<project>/memory` | Cross-run memory location |
| `AUTORESEARCH_CONSOLE_CORS` | `http://localhost:3000,...` | CORS origins |
| `CLAUDE_BIN` / `CODEX_BIN` / `OPENCODE_BIN` | auto-detected | Override agent CLI paths |
| `AUTORESEARCH_BIN` | venv → PATH | Override the `autoresearch` CLI path |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8765` | Backend base URL the UI talks to (set for cloud) |
| `AUTORESEARCH_AUTH_PROVIDER` | `none` | `none` / `bearer` (set `AUTORESEARCH_AUTH_TOKEN`) |
| `AUTORESEARCH_RUNNER` | `local` | Job runner backend (local worktrees) |
