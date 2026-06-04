# Running with Claude Code

## Prerequisites

- Claude Code installed (`npm install -g @anthropic-ai/claude-code` or via the desktop app)
- Repo cloned and quickstart completed (see [README.md](../README.md))
- `data/processed/agent_dataset_search.parquet` exists (run `python scripts/generate_synthetic_data.py` then `autoresearch prepare-data` if not)

## One-time Setup

Open the repository in Claude Code:

```bash
cd <repo> && claude
```

Claude Code will read `AGENT.md` at the start of every session. No additional configuration is needed.

## The First Prompt

Copy and paste this block into Claude Code (replace model name/version as appropriate):

```
Read AGENT.md, then bootstrap a new run under track "claude" with
a timestamped run id and run 3 cycles. Use synthetic data — I have
already run scripts/generate_synthetic_data.py. Use `--new-run` for
the bootstrap command. Pass --model-provider anthropic
--model-name claude-sonnet-4-6 to bootstrap-track.
```

## What Happens

- **Bootstrap**: `autoresearch --track claude --new-run bootstrap-track --model-provider anthropic --model-name claude-sonnet-4-6` creates a fresh timestamped run folder, creates the registry, runs the global-mean baseline, initialises the official champion, and exports the handoff context. The `--model-provider` and `--model-name` flags are required so results can be attributed in the cross-run memory aggregator.
- **Handoff read**: the agent reads the latest handoff file to understand the current champion state before proposing anything.
- **Proposal generation**: the agent writes a proposal JSON and a companion model script to the proposal inbox.
- **Experiment run**: `autoresearch run-session-cycles N` ingests the proposal, runs the experiment, and compares the challenger to the current champion.
- **Promotion or rejection**: if all promotion gate checks pass, the challenger becomes the new champion; otherwise it is rejected and the research log records what was learned.

## Where to Look Afterward

- `artifacts/tracks/claude/runs/<run-id>/RESEARCH_LOG.md` — the agent's running research log for this run
- `artifacts/tracks/claude/runs/<run-id>/iterations/` — per-cycle experiment and comparison artifacts
- The latest `comparison_report.html` inside the most recent `comparison/` folder

## Common Follow-up Prompts

- "Continue" — read the handoff and run 3 more cycles
- "Run 5 more cycles" — read the handoff and run 5 cycles
- "Try a GLM next" — propose a GLM-based experiment in the next cycle

## Troubleshooting

**pytest failures**: the experiment runner runs `pytest` automatically and aborts if tests fail. Fix the failing test or model script before retrying.

**Integrity manifest changes**: if a protected file was edited intentionally, run `autoresearch update-integrity-manifest` and explain why in the research log.

**Holdout token errors**: `autoresearch evaluate-milestone` requires the `AUTORESEARCH_MILESTONE_TOKEN` environment variable. This is a human-only operation; the agent should not call it.

## Recommended Command Pair

```bash
autoresearch --track claude --new-run bootstrap-track \
  --model-provider anthropic --model-name claude-sonnet-4-6
autoresearch --track claude run-session-cycles 3
```

## Run Isolation and Analysis Sessions

A **run-scope guard** (a pre-tool-use hook, shared by all three harnesses) keeps a research session inside its own run folder. Once you bootstrap, the session is bound to that run and the harness blocks any attempt to read another run's files under `artifacts/tracks/`; your own run, `src/`, data, and configs stay fully accessible. Because an un-bootstrapped session is unrestricted, **run the bootstrap command before inspecting the repository or any artifacts**.

For a build or analysis thread that needs to read across runs, launch Claude Code with the analyst override in the environment:

```bash
AUTORESEARCH_SCOPE=analyst claude
```

This exempts the session from confinement. See [`docs/architecture.md`](architecture.md) → *Run-Scope Guard* for details.

## Optional: Enable Cross-Run Memory Access

To let the agent query prior runs' insights and analysis:

```bash
export AUTORESEARCH_MEMORY_ACCESS=own   # this model's history only
# or
export AUTORESEARCH_MEMORY_ACCESS=all   # all models, fully attributed
autoresearch --track claude --new-run bootstrap-track \
  --model-provider anthropic --model-name claude-sonnet-4-6
```

The default (`none`) keeps runs fully isolated with no cross-run context in the handoff. After running several sessions, populate the memory store with `autoresearch memory harvest --all` and build the playbook with `autoresearch memory build-playbook`.
