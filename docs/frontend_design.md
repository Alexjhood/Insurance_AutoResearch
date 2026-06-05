# AutoResearch Console — Design Reference

## Problem

The research loop runs in Claude Code, Codex, or OpenCode — interactive terminal sessions
that must be started manually, steered by pasting prompts, and monitored by tailing files.
As the number of tracks and concurrent runs grows, the ops burden grows with it.

This document describes the **AutoResearch Console**: a web front-end backed by a local
orchestrator service that unifies launch, live monitoring, steering, and cross-run
assessment in one place.

---

## System Layers

```
React / Next.js UI  ─── HTTP + WebSocket ───▶  FastAPI Orchestrator
                                                      │
                          ┌───────────────────────────┤
                          │                           │
                    Job Manager                  Readers (read-only)
                    + orchestrator.db            ├── per-run registry.sqlite
                          │                      ├── memory.sqlite (aggregator)
                    Agent Adapters               └── artifact files
                    ├── Claude (Agent SDK)
                    ├── Codex  (codex exec)
                    └── OpenCode (opencode serve)
```

The orchestrator is the **only** process that writes new state (to `orchestrator.db`
and the agent processes it spawns). The readers are read-only; they never write to
`registry.sqlite` or the memory aggregator.

---

## Agent Adapter Interface

```python
class AgentAdapter(Protocol):
    def launch(self, *, cwd: Path, env: dict, seed_prompt: str) -> SessionHandle
    def stream(self) -> Iterator[AgentEvent]   # token | tool_use | turn_end | exit
    def steer(self, message: str, *, interrupt: bool = True) -> SteerResult
    def stop(self) -> None
```

### Implementations

| Agent | Control surface | Best-effort interrupt |
|---|---|---|
| **Claude** | `claude -p --output-format stream-json` + Agent SDK stdin | Yes — pipe new input mid-stream |
| **Codex** | `codex exec --json` + `codex exec resume <id>` | No — stop + resume (between-turn) |
| **OpenCode** | `opencode serve` HTTP/SSE; POST to session | Partial — between-turn |

`steer(interrupt=True)` tries the live path; on failure or if unsupported, it enqueues
the message for the agent's next turn. The UI indicates which mode took effect.

---

## Concurrency — Git Worktrees

Each launched run gets its own git worktree: `git worktree add ../.worktrees/<track>-<run-id> HEAD`.
The agent process runs in that worktree, preventing collisions on `proposal_inbox/`, `src/`,
and the git index. The run-scope guard hook (already shared by all three harnesses) provides
an additional layer of confinement by binding the session to its run.

The orchestrator tracks worktree lifecycle and cleans up on stop (auto-removes if unchanged;
retains otherwise for investigation).

---

## Run Launch Controls

| Control | Maps to |
|---|---|
| Surface | Claude Code / Codex / OpenCode |
| Model | `--model-provider` + `--model-name` |
| Cycles | `autoresearch run-session-cycles N` |
| Memory access | `AUTORESEARCH_MEMORY_ACCESS` env (`none` / `own` / `all`) |
| Scope | Research (default) vs `AUTORESEARCH_SCOPE=analyst` |
| Track | `--track <name>` |
| Guidance | Free-text appended to the seed prompt template |

The seed prompt template mirrors the "First Prompt" blocks in `docs/RUN_WITH_CLAUDE_CODE.md`
and `docs/RUN_WITH_CODEX.md`. The orchestrator never sets `AUTORESEARCH_MILESTONE_TOKEN`.

---

## Data Model (`orchestrator.db`)

```sql
jobs          id, track, run_id, surface, model_provider, model_name,
              worktree_path, status, seed_prompt, env_json, created_at, updated_at

events        id, job_id, ts, event_type, payload_json
              -- event_type: token | tool_use | turn_end | agent_exit | steer_sent | steer_queued

steer_queue   id, job_id, message, interrupt, queued_at, sent_at, status
```

On orchestrator restart, jobs with `status=running` are re-checked; still-alive processes
are reattached, dead ones are marked `interrupted`.

---

## API Surface

### Phase 1 — Read-Only Console

```
GET  /api/tracks                              → list tracks
GET  /api/tracks/{track}/runs                 → list runs
GET  /api/tracks/{track}/runs/{run_id}        → run summary + registry counts
GET  /api/tracks/{track}/runs/{run_id}/experiments
GET  /api/tracks/{track}/runs/{run_id}/comparisons
GET  /api/tracks/{track}/runs/{run_id}/champion
GET  /api/tracks/{track}/runs/{run_id}/proposals
GET  /api/tracks/{track}/runs/{run_id}/sessions
GET  /api/tracks/{track}/runs/{run_id}/research-lines
GET  /api/tracks/{track}/runs/{run_id}/artifacts/{path}  → serve file
GET  /api/leaderboard                         → cross-run from memory.sqlite
GET  /api/health
```

### Phase 2 — Launch + Monitor

```
POST /api/jobs                                → launch new run → job_id
GET  /api/jobs                                → list jobs
GET  /api/jobs/{job_id}                       → job status
WS   /api/jobs/{job_id}/stream                → live event stream
POST /api/jobs/{job_id}/steer                 → send steering message
POST /api/jobs/{job_id}/pause
POST /api/jobs/{job_id}/stop
```

---

## Front-End Pages

| Route | Phase | Content |
|---|---|---|
| `/` | 1 | Dashboard: track roster, quick stats |
| `/runs` | 1 | All runs table with status + peak gini |
| `/runs/[track]/[runId]` | 1 | Run detail: experiments, comparisons, champion, research log, artifact viewer |
| `/leaderboard` | 1 | Cross-run leaderboard (search-split). Gated holdout panel (token prompt). |
| `/launch` | 2 | Launch form: surface, model, cycles, memory, guidance |
| `/monitor/[jobId]` | 2 | Live stream, steer box, pause/stop |

---

## Assessment and Exhibits

- **Leaderboard**: search-split metrics from `memory.sqlite` — safe for agent access.
  A visually distinct **milestone holdout** panel prompts for `AUTORESEARCH_MILESTONE_TOKEN`
  interactively; the token is never stored server-side and never passed to research jobs.
- **Exhibits** (Phase 1): surface artifacts the pipeline already writes per cycle —
  `comparison_report.html`, `RESEARCH_LOG.md`, `validation_report.json`, bootstrap charts.
  No new chart engine.
- **Custom exhibits** (Phase 2): steer the running agent ("build an exhibit comparing X")
  and the artifact viewer automatically picks up new files from the run folder.

---

## Cloud-Readiness

Designed locally-first, upgrading without rewrites:

| Concern | Local | Cloud swap |
|---|---|---|
| Job runner | Local subprocess | Container / remote runner abstraction |
| Persistence | `orchestrator.db` (SQLite) | Postgres via one DSN env var |
| Auth | No-op middleware | Real provider |
| Agent credentials | Host shell env | Per-user secret store |
| Worktrees | Local git | Remote repo checkout per run |

---

## Security / Integrity Guardrails

- `AUTORESEARCH_MILESTONE_TOKEN` never stored in `orchestrator.db` or env files
- Orchestrator readers are read-only against all registries and memory
- `AUTORESEARCH_SCOPE=analyst` set only for explicit analyst jobs, never research jobs
- Promotion decisions remain entirely within `autoresearch` CLI — the console never promotes
- Worktrees + run-scope guard provide two independent isolation layers per research session

---

## Build Phases

| Phase | Delivers |
|---|---|
| **1** | Read-only console: roster, run interrogation, leaderboard. Immediate value. |
| **2** | Launch + monitor: worktree spawn, live stream, pause/stop. Claude adapter. |
| **3** | Steering: best-effort interrupt + between-turn fallback; orchestrator restart/reattach. |
| **4** | Codex + OpenCode adapters behind same interface. |
| **5** | Gated holdout panel, cloud-seam hardening (Runner/auth/DB abstractions). |
