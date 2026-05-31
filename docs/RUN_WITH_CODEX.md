# Running with Codex

## Prerequisites

- Codex CLI installed and authenticated (see [Codex CLI install docs](https://github.com/openai/codex))
- Repo cloned and quickstart completed (see [README.md](../README.md))
- `data/processed/agent_dataset_search.parquet` exists (run `python scripts/generate_synthetic_data.py` then `autoresearch prepare-data` if not)

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
Read AGENT.md, then bootstrap a new run under track "codex" with
a timestamped run id and run 3 cycles. Use synthetic data — I have
already run scripts/generate_synthetic_data.py. Use `--new-run` for
the bootstrap command. Pass --model-provider openai
--model-name codex-mini-latest to bootstrap-track.
```

## What Happens

- **Bootstrap**: `autoresearch --track codex --new-run bootstrap-track --model-provider openai --model-name codex-mini-latest` creates a fresh timestamped run folder, creates the registry, runs the global-mean baseline, initialises the official champion, and exports the handoff context. The `--model-provider` and `--model-name` flags are required so results can be attributed in the cross-run memory aggregator.
- **Handoff read**: the agent reads the latest handoff file to understand the current champion state before proposing anything.
- **Proposal generation**: the agent writes a proposal JSON and a companion model script to the proposal inbox.
- **Experiment run**: `autoresearch run-session-cycles N` ingests the proposal, runs the experiment, and compares the challenger to the current champion.
- **Promotion or rejection**: if all promotion gate checks pass, the challenger becomes the new champion; otherwise it is rejected and the research log records what was learned.

## Where to Look Afterward

- `artifacts/tracks/codex/runs/<run-id>/RESEARCH_LOG.md` — the agent's running research log for this run
- `artifacts/tracks/codex/runs/<run-id>/iterations/` — per-cycle experiment and comparison artifacts
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
autoresearch --track codex --new-run bootstrap-track \
  --model-provider openai --model-name codex-mini-latest
autoresearch --track codex run-session-cycles 3
```

## Optional: Enable Cross-Run Memory Access

```bash
export AUTORESEARCH_MEMORY_ACCESS=own   # this model's history only
# or
export AUTORESEARCH_MEMORY_ACCESS=all   # all models, fully attributed
autoresearch --track codex --new-run bootstrap-track \
  --model-provider openai --model-name codex-mini-latest
```

The default (`none`) keeps runs fully isolated. See `docs/CLI.md` for `memory harvest`, `memory query`, and `memory build-playbook`.
