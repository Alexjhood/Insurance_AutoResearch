# Running with Codex

## Prerequisites

- Codex CLI installed and authenticated (see [Codex CLI install docs](https://github.com/openai/codex))
- Repo cloned and quickstart completed (see [README.md](../README.md))
- The chosen dataset is prepared: `autoresearch list-datasets` shows `prepared=yes` (if not, run `autoresearch --dataset <name> prepare-data`; for French synthetic data run `python scripts/generate_synthetic_data.py` first)

## One-time Setup

The `.codex/config.toml` in this repository is already configured:

```toml
model = "gpt-5.5"
sandbox_mode = "workspace-write"
network_access = true
```

`sandbox_mode = "workspace-write"` allows Codex to write files within the repo.
`network_access = true` is required if Codex calls `python scripts/fetch_fremtpl2.py`
to download the real dataset from OpenML.

Open the repository in Codex from the repo root.

## The First Prompt

Copy and paste this block (replace model name as appropriate):

```
Read AGENT.md, then bootstrap a new run under track "codex" with a
timestamped run id: use `--new-run`, pass `--dataset <name>`, and pass --model-provider openai
--model-name codex-mini-latest to bootstrap-track. Capture the run id it
prints, then open the session with `start-session main --max-cycles 3`
so the session owns the cycle budget, and run the adaptive loop
(`run-session-cycles 1` at a time) until the session reports its budget
is exhausted. The dataset is already prepared.
```

Vary the run by adding flags to the bootstrap instruction:

- **Dataset** — pass `--dataset <name>` (`french_motor`, `allstate`,
  `allstate_full`, `porto_seguro`); fresh tracked bootstraps require an explicit
  selection and the run manifest pins it thereafter.
- **Target mode** — add `--target-mode <mode>` (must be one of the dataset's
  modes, per `list-datasets`) and instruct the agent to pass it on **every**
  command; omit for the dataset's default.
- **Foundation models (TabPFN/TabFM)** — add `--enable-foundation-models`
  (requires the `[foundation]` extra) and, if desired, tell the agent to try
  the `tabpfn` recipe estimator.
- **Modelling guidance** — append free-text steer, e.g. "prioritise feature
  engineering over hyperparameter tuning".

## What Happens

- **Bootstrap**: `autoresearch --track codex --new-run --dataset french_motor bootstrap-track --model-provider openai --model-name codex-mini-latest` creates a fresh timestamped run folder, creates the registry, runs the global-mean baseline, initialises the search champion, and exports the handoff context. The `--model-provider` and `--model-name` flags are required so results can be attributed in the cross-run memory aggregator.
- **Session open**: `autoresearch --track codex --run-id <run-id> start-session main --max-cycles N` opens the supervised session and makes it own the cycle budget, so the session stops itself after N cycles instead of the agent counting.
- **Handoff read**: the agent reads the latest handoff file to understand the current champion state before proposing anything.
- **Proposal generation**: the agent writes a proposal JSON and a companion model script to the proposal inbox.
- **Experiment run**: `autoresearch run-session-cycles 1` ingests the proposal, runs one cycle, and compares the challenger to the current champion; the agent decides, then repeats one cycle at a time.
- **Promotion or rejection**: if all promotion gate checks pass, the challenger becomes the new champion; otherwise it is rejected and the research log records what was learned.

## Where to Look Afterward

- `artifacts/tracks/codex/runs/<run-id>/RESEARCH_LOG.md` — the framework-generated research log for this run
- `artifacts/tracks/codex/runs/<run-id>/iterations/` — per-cycle experiment and comparison artifacts
- `artifacts/tracks/codex/runs/<run-id>/telemetry.sqlite` — normalized Codex Desktop usage and tool telemetry
- `artifacts/tracks/codex/runs/<run-id>/LLM_USAGE.md` — per-experiment and per-user-breakpoint token ledger, including output and reasoning tokens
- The run detail page in the web Console — live token, cache, reasoning, tool, error, and framework-step summaries
- The latest `comparison_report.html` inside the most recent `comparison/` folder

Codex Desktop writes a structured rollout transcript and invokes the project
`Stop` hook after a completed turn. The hook starts a short deferred import so
the final Codex `task_complete` record is captured after the hook returns. It
imports only newly appended
records into the bound run. It does not connect to Codex Desktop's internal IPC
socket. Full prompts and tool output remain in Codex's native rollout; the run
database stores normalized metrics and compact labels only.

To inspect or recover telemetry manually:

```bash
autoresearch --track codex --run-id <run-id> telemetry report
autoresearch --track codex --run-id <run-id> telemetry sync \
  --surface codex --session-id <codex-thread-id> --finalize-turn
```

## Common Follow-up Prompts

- "Continue" — read the handoff and run 3 more cycles
- "Run 5 more cycles" — read the handoff and run 5 cycles
- "Try a GLM next" — propose a GLM-based experiment in the next cycle

## Troubleshooting

**pytest failures**: the experiment runner runs `pytest` automatically and aborts if tests fail. Fix the failing test or model script before retrying.

**Integrity manifest changes**: if a protected file was edited intentionally, run `autoresearch update-integrity-manifest` and explain why in the research log.

**Holdout token errors**: `autoresearch evaluate-milestone` requires the `AUTORESEARCH_MILESTONE_TOKEN` environment variable. This is a human-only operation; the agent should not call it.

**No telemetry after a turn**: restart or reload the Codex project so the
updated `.codex/hooks.json` is trusted and loaded, then check that the thread
has bootstrapped and is bound to a run. The telemetry hook deliberately skips
unbound analysis threads.

## Recommended Command Sequence

```bash
autoresearch --track codex --new-run --dataset french_motor bootstrap-track \
  --model-provider openai --model-name codex-mini-latest
# capture the printed run id, then let the session own the budget:
autoresearch --track codex --run-id <run-id> start-session main --max-cycles 3
autoresearch --track codex --run-id <run-id> run-session-cycles 1   # repeat until the budget is exhausted
```

## Run Isolation and Analysis Sessions

A **run-scope guard** keeps a research session inside its own run folder. It is wired into Codex as a pre-tool-use hook (`.codex/hooks.json`, sharing the same `scripts/run_scope_guard.py` used by Claude Code and OpenCode). Once you bootstrap, the session is bound to that run and Codex blocks any shell read (`cat`/`grep`) or `apply_patch` that targets another run's files under `artifacts/tracks/`; your own run, `src/`, data, and configs stay accessible. Because an un-bootstrapped session is unrestricted, **run the bootstrap command before inspecting the repository or any artifacts**.

For a build or analysis thread that needs to read across runs, set the analyst override in the environment before launching Codex:

```bash
export AUTORESEARCH_SCOPE=analyst
```

This exempts the session from confinement. See [`docs/architecture.md`](architecture.md) → *Run-Scope Guard* for details.

## Optional: Enable Cross-Run Memory Access

```bash
export AUTORESEARCH_MEMORY_ACCESS=own   # this model's history only
# or
export AUTORESEARCH_MEMORY_ACCESS=all   # all models, fully attributed
autoresearch --track codex --new-run --dataset french_motor bootstrap-track \
  --model-provider openai --model-name codex-mini-latest
```

The default (`none`) keeps runs fully isolated. See `docs/CLI.md` for `memory harvest`, `memory query`, and `memory build-playbook`.
