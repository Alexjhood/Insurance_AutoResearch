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
4. Write one proposal JSON file and one neighbouring model script into `artifacts/auto_research/proposals/inbox/`.
5. Run:

```bash
autoresearch run-session-cycle
```

If the session returns `waiting_for_proposal`, repeat from step 1. The framework refreshes handoff artifacts after ingestion, evaluation, comparison, promotion decisions, pause/failure/completion, and session state changes.

For non-`global_mean` proposals, set `experiment_config.model.script_path` in
the JSON to the script filename. The script must expose
`fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None,
**hyperparameters)` and return original-space claim-cost predictions plus a
notes dict. Do not rely on pre-existing implementations under
`src/autoresearch/models`; if a GLM, GBM, or any other method is the chosen
research direction, write that implementation in the run-local script.

If output validation fails, the session may return `waiting_for_repair` and
write `repair_request_2.json` or `repair_request_3.json` beside the proposal.
Revise the requested `model_attempt_N.py` and rerun `autoresearch
run-session-cycle`. After three failed attempts, the proposal is failed.

## Research Tree And Screening

Each run maintains its own research tree in that run's `registry.sqlite`. The
tree records proposal nodes, explicit tree-walk metadata, their optional
`research_parent_node_id`, outcome, screening metrics, and guidance for later
proposals. Context export only reads the active run's tree; it does not search
other tracks or runs.

The context also includes `research_tree.tree_policy.recommended_actions`. A
proposal must choose one recommendation through `selected_tree_action_id`, set
`tree_action`, and explain the choice in `parent_rationale`. Use
`tree_action=new_root` with `research_parent_node_id=null` only for a genuinely
new line of attack. Other tree actions must point to a valid node from this
run's tree. Cross-run memory, when enabled, may inform broad strategy but does
not provide tree parent IDs or active-run evidence.

Only one valid proposal is ingested per context refresh while the queue is
active or a decision is pending. Additional JSON files remain in the inbox with
`deferred_pending_context_refresh` in the ingest summary, so the agent can
refresh context before choosing whether to keep, rewrite, or delete them.

Required tree metadata fields are:

- `tree_action`
- `selected_tree_action_id`
- `parent_rationale`
- `exploration_axis`
- `approach_family`
- `target_framing`
- `feature_representation`
- `expected_learning`

Valid challengers pass through a cheap full `search_validation` single-split
screen before CV/bootstrap comparison. Clearly worse challengers are
auto-rejected and summarised for reflection, but the framework still writes a
diagnostic comparison report using one paired eval-split sample. That report is
not recorded as an official pending comparison. Similar or better challengers
continue to the full comparison report and LLM decision.

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
