# Supervised Autonomous Session Workflow

This project now uses Codex or Claude Code as an external proposal-design agent through file handoff. The Python framework remains the lab: it validates proposals, runs deterministic experiments, compares against the official champion, applies the promotion gate, persists state, and refreshes handoff files.

## Start Or Resume

```bash
autoresearch init-official-champion
autoresearch start-session local-research --max-cycles 10
autoresearch export-context
```

If interrupted:

```bash
autoresearch session-status
autoresearch resume-session
```

## Agent Loop

1. Read `artifacts/auto_research/handoffs/latest_handoff.md`.
2. Read `artifacts/auto_research/context/latest_context.json`.
3. Read `artifacts/auto_research/handoffs/proposal_instructions.md`.
4. Write one proposal JSON file into `artifacts/auto_research/proposals/inbox/`.
5. Run:

```bash
autoresearch run-session-cycle
```

If the session returns `waiting_for_proposal`, repeat from step 1. The framework refreshes handoff artifacts after ingestion, evaluation, comparison, promotion decisions, pause/failure/completion, and session state changes.

## Pause Or Stop

```bash
autoresearch pause-session
autoresearch resume-session
autoresearch stop-session
```

`pause-session` is for temporary intervention. `stop-session` marks the session completed and stops further automatic progress.

## Monitoring

Use the dashboard pages:

- `Sessions`: session state, recent events, latest proposal outcomes
- `File Handoff`: inbox status, processed valid/invalid/duplicate counts, latest handoff and cycle summary
- `Champion`: official champion and champion history
- `Comparisons`: promotion evidence and uncertainty summaries

CLI equivalents:

```bash
autoresearch session-status
autoresearch show-proposal-inbox-status
autoresearch list-proposals
autoresearch list-champion-history
autoresearch list-promotions
```

## Duplicate And Non-Promotion Handling

The lab rejects obvious duplicates by default. A proposal is treated as duplicate when it repeats a recent executable config or identical change summary. Duplicate, invalid, failed, and inconclusive proposals get concise summaries under:

```text
artifacts/auto_research/results/non_promoted/
artifacts/auto_research/results/latest_nonpromotion_summary.md
```

External agents should read these summaries before proposing the next experiment.

## Safety Boundaries

- Do not use `milestone_holdout` during ordinary search.
- Do not redefine metrics or promotion rules in a proposal.
- Do not mutate historical artifacts.
- Official champion changes only through volatility-aware promotion.
