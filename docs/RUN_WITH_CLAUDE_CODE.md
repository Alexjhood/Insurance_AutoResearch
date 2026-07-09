# Running with Claude Code

## Prerequisites

- Claude Code installed (`npm install -g @anthropic-ai/claude-code` or via the desktop app)
- Repo cloned and quickstart completed (see [README.md](../README.md))
- The chosen dataset is prepared: `autoresearch list-datasets` shows `prepared=yes` (if not, run `autoresearch --dataset <name> prepare-data`; for French synthetic data run `python scripts/generate_synthetic_data.py` first)

## One-time Setup

Open the repository in Claude Code:

```bash
cd <repo> && claude
```

Claude Code will read `AGENT.md` at the start of every session. No additional configuration is needed.

## The First Prompt

Copy and paste this block into Claude Code (replace model name/version as appropriate):

```
Read AGENT.md, then bootstrap a new run under track "claude" with a
timestamped run id: use `--new-run` and pass --model-provider anthropic
--model-name claude-sonnet-4-6 to bootstrap-track. Capture the run id it
prints, then open the session with `start-session main --max-cycles 3`
so the session owns the cycle budget, and run the adaptive loop
(`run-session-cycles 1` at a time) until the session reports its budget
is exhausted. The dataset is already prepared.
```

Vary the run by adding flags to the bootstrap instruction: `--dataset <name>`
(`french_motor` default, `allstate`, `allstate_full`, `porto_seguro`; pinned in
the run manifest), `--target-mode <mode>` (one of the dataset's modes — also
instruct the agent to pass it on every command), and
`--enable-foundation-models` to allow the `tabpfn` recipe estimator. Append any
free-text modelling guidance after the block.

## What Happens

- **Bootstrap**: `autoresearch --track claude --new-run bootstrap-track --model-provider anthropic --model-name claude-sonnet-4-6` creates a fresh timestamped run folder, creates the registry, runs the global-mean baseline, initialises the official champion, and exports the handoff context. The `--model-provider` and `--model-name` flags are required so results can be attributed in the cross-run memory aggregator.
- **Session open**: `autoresearch --track claude --run-id <run-id> start-session main --max-cycles N` opens the supervised session and makes it own the cycle budget, so the session stops itself after N cycles instead of the agent counting.
- **Handoff read**: the agent reads the latest handoff file to understand the current champion state before proposing anything.
- **Proposal generation**: the agent writes a proposal JSON and a companion model script to the proposal inbox.
- **Experiment run**: `autoresearch run-session-cycles 1` ingests the proposal, runs one cycle, and compares the challenger to the current champion; the agent decides, then repeats one cycle at a time.
- **Promotion or rejection**: if all promotion gate checks pass, the challenger becomes the new champion; otherwise it is rejected and the research log records what was learned.

## Where to Look Afterward

- `artifacts/tracks/claude/runs/<run-id>/RESEARCH_LOG.md` — the framework-generated research log for this run
- `artifacts/tracks/claude/runs/<run-id>/iterations/` — per-cycle experiment and comparison artifacts
- `artifacts/tracks/claude/runs/<run-id>/telemetry.sqlite` — normalized Claude Code Desktop usage and tool telemetry
- `artifacts/tracks/claude/runs/<run-id>/LLM_USAGE.md` — per-experiment and per-user-breakpoint token ledger, including output and reasoning tokens
- The run detail page in the web Console — live token, cache, tool, error, and framework-step summaries
- The latest `comparison_report.html` inside the most recent `comparison/` folder

Claude Code Desktop writes a structured local transcript and invokes the
project `Stop` hook after a completed turn. The hook starts a short deferred
import after the hook returns and imports only newly appended
records into the bound run. Full prompts and tool output remain in Claude's
native transcript; the run database stores metrics, byte counts, compact tool
labels, and experiment identifiers only.

To inspect or recover telemetry manually:

```bash
autoresearch --track claude --run-id <run-id> telemetry report
autoresearch --track claude --run-id <run-id> telemetry sync \
  --surface claude --session-id <claude-session-id> --finalize-turn
```

## Common Follow-up Prompts

- "Continue" — read the handoff and run 3 more cycles
- "Run 5 more cycles" — read the handoff and run 5 cycles
- "Try a GLM next" — propose a GLM-based experiment in the next cycle

## Troubleshooting

**pytest failures**: the experiment runner runs `pytest` automatically and aborts if tests fail. Fix the failing test or model script before retrying.

**Integrity manifest changes**: if a protected file was edited intentionally, run `autoresearch update-integrity-manifest` and explain why in the research log.

**Holdout token errors**: `autoresearch evaluate-milestone` requires the `AUTORESEARCH_MILESTONE_TOKEN` environment variable. This is a human-only operation; the agent should not call it.

**No telemetry after a turn**: restart the Claude Code project so the updated
`.claude/settings.json` is loaded, then check that the session has bootstrapped
and is bound to a run. The telemetry hook deliberately skips unbound analysis
sessions.

## Recommended Command Sequence

```bash
autoresearch --track claude --new-run bootstrap-track \
  --model-provider anthropic --model-name claude-sonnet-4-6
# capture the printed run id, then let the session own the budget:
autoresearch --track claude --run-id <run-id> start-session main --max-cycles 3
autoresearch --track claude --run-id <run-id> run-session-cycles 1   # repeat until the budget is exhausted
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
