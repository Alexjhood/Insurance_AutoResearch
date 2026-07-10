#!/usr/bin/env python3
"""Generate the compact agent runtime contract (``AGENT.md``) from code/config.

The agent loads ``AGENT.md`` into context at the start of every run (the console
seed prompt says "Read AGENT.md, then bootstrap ..."). To keep that startup cost
small *and* prevent the manual from drifting out of sync with the CLI, the
runtime contract is generated here from authoritative, importable sources:

* command names      -> ``autoresearch.cli.COMMANDS``
* proposal fields    -> ``autoresearch.controller.proposal_schema.SCIENTIFIC_PROPOSAL_FIELDS``
* column constants   -> ``autoresearch.models.dispatcher``
* protected files    -> ``autoresearch.utils.integrity.PROTECTED_RELATIVE_PATHS``
* compute/gate/target-> ``configs/default.toml``

The long, human-facing manual lives in ``docs/OPERATING_MANUAL.md`` and is the
drill-down target referenced from the contract.

Usage::

    python scripts/generate_agent_contract.py            # write AGENT.md
    python scripts/generate_agent_contract.py --check     # exit 1 if stale

``tests/test_agent_contract.py`` runs ``--check`` and also asserts every command
named in the contract exists in ``cli.COMMANDS``.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from autoresearch.cli import COMMANDS  # noqa: E402
from autoresearch.controller.proposal_schema import (  # noqa: E402
    DERIVED_PROPOSAL_FIELDS,
    SCIENTIFIC_PROPOSAL_FIELDS,
)
from autoresearch.models.recipe import enable_foundation_models, menu as recipe_menu  # noqa: E402
from autoresearch.utils.integrity import PROTECTED_RELATIVE_PATHS  # noqa: E402

AGENT_MD = REPO_ROOT / "AGENT.md"
# Harness-native auto-load copies of the same contract: Codex auto-loads
# AGENTS.md, Claude Code auto-loads CLAUDE.md, OpenCode reads AGENTS.md. Emitting
# byte-identical copies puts the contract in each harness's system region ahead
# of the dynamic seed prompt, removing the agent's start-of-run AGENT.md read.
AGENT_MIRRORS = [REPO_ROOT / "AGENTS.md", REPO_ROOT / "CLAUDE.md"]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
MANUAL_REL = "docs/OPERATING_MANUAL.md"

# Curated workflow commands shown in the contract, in the order the agent uses
# them. Each is validated against cli.COMMANDS at generation time, so renaming or
# removing a command in the CLI breaks generation (and the sync test) loudly.
WORKFLOW_COMMANDS: list[tuple[str, str]] = [
    ("bootstrap-track", "Create + prepare a fresh run (first command of a new run)."),
    ("start-session", "Open the supervised session for the run."),
    ("show-latest-handoff", "Read current champion/state before proposing."),
    ("list-experiments", "List this run's registered experiments."),
    ("list-champion-history", "Show how the champion evolved this run."),
    ("run-session-cycles", "Run N cycles; STOPS at awaiting_decision (no auto-promote)."),
    ("record-decision", "Your verdict: promote | local_promote | reject."),
    ("record-cycle-reflection", "Complete the final reflection after an auto-rejected cycle."),
    ("park-research-line", "Park an exhausted research line."),
    ("clear-line-champion", "Drop a local incumbent that looks artefactual."),
    ("export-context", "Refresh the handoff/context bundle on demand."),
]

# Column constants a model script may import. Their concrete values are bound to
# the active dataset at config load and printed in the handoff; the contract
# stays dataset-neutral and names only the constants.
_COLUMN_CONSTANTS = [
    ("EXPOSURE", "weight/offset column; weights + rate->total only, never a feature"),
    ("CLAIM_COST", "training-target total column (when present)"),
    ("CLAIM_COUNT", "claim-count target (freq/freq-sev; when present)"),
    ("RECORD_ID", "row identifier"),
]


def _load_config() -> dict:
    with DEFAULT_CONFIG.open("rb") as fh:
        return tomllib.load(fh)


def _validate_commands() -> None:
    """Fail loudly if the contract references a command the CLI does not expose."""
    missing = [c for c, _ in WORKFLOW_COMMANDS if c not in COMMANDS]
    if missing:
        raise SystemExit(
            f"generate_agent_contract: commands not found in cli.COMMANDS: {missing}"
        )


def render() -> str:
    cfg = _load_config()
    compute = cfg["compute"]
    evaluation = cfg["evaluation"]

    base = compute["base_budget_minutes"]
    incr = compute["budget_increment_minutes"]
    per = compute["experiments_per_increment"]
    gate_mode = evaluation["gate_mode"]
    gate_metric = evaluation["gate_primary_metric"]

    cmd_lines = "\n".join(
        f"- `{name}` — {desc}" for name, desc in WORKFLOW_COMMANDS
    )
    protected_lines = "\n".join(f"- `{p}`" for p in PROTECTED_RELATIVE_PATHS)
    const_lines = "\n".join(
        f"- `{name}` — {desc}" for name, desc in _COLUMN_CONSTANTS
    )
    proposal_fields = ", ".join(f"`{f}`" for f in SCIENTIFIC_PROPOSAL_FIELDS)
    derived_fields = ", ".join(f"`{f}`" for f in DERIVED_PROPOSAL_FIELDS)

    enable_foundation_models()
    rmenu = recipe_menu()
    estimator_lines = "\n".join(
        f"- **{name}** — obj {sorted(info['objectives'])}; enc {sorted(info['encodings'])}"
        + (" (early-stop)" if info["supports_early_stopping"] else "")
        for name, info in rmenu["estimators"].items()
    )
    structure_list = ", ".join(f"`{s}`" for s in rmenu["structures"])

    # Foundation estimators (TabPFN, ...) only appear here when the run enabled
    # them AND the [foundation] extra is installed; when present, the agent needs
    # the extra operating notes they carry (no early stopping, subsampled context,
    # minutes-not-seconds per fit).
    _foundation_present = "tabpfn" in rmenu["estimators"]
    foundation_note = ""
    if _foundation_present:
        foundation_note = ""

    return f"""\
<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with: python scripts/generate_agent_contract.py
     Source of truth: cli.COMMANDS, proposal_schema, dispatcher, integrity,
     configs/default.toml. Full human manual: {MANUAL_REL}. -->

# AGENT.md — Auto-Research Runtime Contract

You are the research agent for an autonomous tabular target-modelling loop on a
**per-run selected dataset**. The active dataset, target mode, column roles,
weight policy, and any fixed preprocessing (e.g. a claim cap) are printed in the
handoff's **"Active dataset"** block; read it first — those facts are binding.
Maximise **weight-weighted Gini** (`{gate_metric}`) on the search-validation
split; every promotion is re-checked on a protected holdout. Each run starts with
the `global_mean` baseline as champion — beat a flat weighted rate first.

Escalation only — most runs never need it: **{MANUAL_REL}** has the full manual
(dataset schema, metric panel, gate modes, research-line mechanics, worked examples).

## Hard safety constraints — never break

1. **Never read the holdout.** Do not import `autoresearch.data.holdout_vault`
   or reference `milestone_holdout` / `AUTORESEARCH_MILESTONE_TOKEN` in model or
   feature code. The integrity scanner fails the experiment.
2. **Never edit protected evaluation files** (comparisons block until an
   operator runs `autoresearch update-integrity-manifest`):
{protected_lines}
3. **Never change** the primary metric, gate thresholds, the fixed split
   (`split_pack.csv`, `data/datasets/<name>/`), or any dataset-fixed
   preprocessing (e.g. the claim cap) — the handoff says what is fixed.
4. **Always pass pytest** — the runner refuses to proceed on a failing suite.
5. **Always stay in your own run.** Pass `--track <your-tool-name>`
   (`claude` / `codex` / `opencode`) and `--run-id <id>` to every command. After
   your first `bootstrap-track`/`start-session`, the harness blocks reads of any
   other run's `artifacts/tracks/.../runs/<other>` folder.

## The workflow (this is the only one)

A request to "run N experiments" / "start" / "test N ideas" means a **fresh
run** unless the user says "continue"/"resume" or gives a run id. Do not inspect
existing artifacts to infer intent.

1. **Bootstrap (fresh run only).** First shell command — binds the run scope:
   ```bash
   autoresearch --track <t> --new-run bootstrap-track \\
     --model-provider <provider> --model-name <model-name> --cycles <N>
   ```
   `--cycles <N>` pins the requested experiment budget — the framework stops
   the run at N cycles so you never have to count.
   Capture the returned timestamped `run_id`; pass `--run-id <id>` thereafter.
   For an explicit continuation, skip bootstrap and resolve latest once with
   `start-session`, then pin the resolved id.
2. **Read the handoff** (`show-latest-handoff`) before forming any hypothesis.
   It is authoritative: champion, recent results and learnings, recommended tree
   actions, key constraints, and the exact proposal template are all embedded
   inline. Open this run's `RESEARCH_LOG.md` or `latest_context.json` only when
   you need older detail than the handoff already carries.
3. **Propose one idea per context refresh.** Write a proposal JSON + a
   neighbouring model script into **this run's** inbox. The handoff prints its
   exact path (`Inbox: ... ← write proposal JSON + model script here`); for a
   tracked run it is
   `artifacts/tracks/<track>/runs/<run-id>/proposal_inbox/` — **not** the
   repo-root `proposal_inbox/` (that legacy path is used only for untracked
   runs and stays empty here). The handoff prints the exact proposal template
   inline — copy it and fill the `<...>` fields; no separate schema-file read is
   needed. At most one queued proposal is ingested per refresh; extra JSON files
   stay deferred.
4. **Run + decide, looping one cycle at a time:**
   ```bash
   autoresearch --track <t> --run-id <id> run-session-cycles 1   # stops at awaiting_decision
   # review the metric summary, then:
   autoresearch --track <t> --run-id <id> record-decision <comparison_id> \\
     --decision promote|local_promote|reject --rationale "..." \\
     --reason-code clear_win|line_progress|noise|inferior|artifact_suspected|calibration|other \\
     --interpretation "what the result taught" --next "next direction"
   ```
   `run-session-cycles` **never auto-promotes** — it always stops for your
   `record-decision`. Repeat propose -> cycle -> decide. **Design the next
   experiment only after reading the current one's result** — N is a budget of
   cycles to spend adaptively, not a slate to plan in advance (see "Adaptive
   search"). `record-decision` prints the post-decision champion and the next
   command in its own output, so you do **not** need to follow it with
   `show-latest-handoff`, `list-champion-history`, or `session-status` — those
   are for the rare case you need detail the decision output did not carry. On a
   mid-run refresh, prefer `show-latest-handoff --delta` (dynamic state only;
   the template and constraints are already in this contract).
5. **Research logging is framework-owned.** The framework writes hypothesis,
   changes, outcome, and metrics from registry state. Supply interpretation and
   next direction with `record-decision`. After an auto-rejection, put
   `previous_cycle_reflection` in the next proposal; if it was the final cycle,
   run `record-cycle-reflection --interpretation "..." --next "..."`.
   `RESEARCH_LOG.md` is generated from these structured records; do not edit it.

### Workflow commands

All take `--track <your-tool-name> --run-id <id>` (omit `--run-id` only on the
first `bootstrap-track`, which prints the id):

{cmd_lines}

### Orchestrated mode

If your launch prompt names a **pre-bootstrapped run**, you are a sub-agent under
an orchestrator. Skip step 1: never run `bootstrap-track`, never start another
run, and pass the given `--track`/`--run-id` to every command. Your handoff
carries an **"Orchestration brief"** block — treat its direction, constraints,
and stop conditions as binding, on par with the Active dataset block, and spend
your cycles adaptively inside it. When the budget is exhausted (or a brief
stop-condition fires), finish with:

```bash
autoresearch --track <t> --run-id <id> orchestrate finish-delegation \\
  --summary "<3–6 sentences: what you learned, what you'd try next, anything artifactual>"
```

and then stop. Your report is built from the registry either way — the summary is
your testimony, not your score.

### When a cycle needs repair

A cycle can stop in **`needs_repair`** (instead of `awaiting_decision`) when the
model fails preflight, the compute budget, or output validation. A negative lift
is **not** a repair trigger — such models proceed to screening. The framework
writes `repair_request_<N>.json` into the proposal directory. Recover
without guessing — the file tells you what to do via its `repair_kind`:

1. Read `repair_request_<N>.json`: `repair_kind`, `failed_checks`, `instruction`.
2. `repair_kind == "recipe"` → write the corrected recipe to
   `recipe_attempt_<N>.json` (framework still owns units/calibration; only write
   `model_attempt_<N>.py` if the recipe truly can't express the fix).
   `repair_kind == "script"` → write `model_attempt_<N>.py` (same `fit_predict`).
   Never touch holdout data.
3. Rerun `run-session-cycles 1` — it picks up the new attempt automatically.

You get up to **3 attempts**. Attempt 2 should move *opposite* the failure (if a
tree was too deep, go shallower — not halfway back); if attempt 2 is still no
better, let it fail rather than spend attempt 3 on a near-duplicate.

## Model: a recipe (preferred) or a script (escape hatch)

Specify the model one of two ways. Both flow through the same framework
units/calibration stage, so **you never write exposure conversion or calibration
yourself**. Prefer a recipe for any method the registry covers; drop to a script
only for a model the recipe vocabulary cannot express.

### Option A — declarative recipe (`model.recipe`), preferred

A validated object interpreted by trusted code. Set `model_family = "recipe"`:
```json
{{"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
 "encoding": "native_categorical", "params": {{"num_leaves": 63}}, "early_stopping": 50}}
```
Champion follow-up (do not repeat the full recipe):
```json
{{"recipe_ref":"champion","recipe_overrides":{{"params":{{"num_leaves":31}}}}}}
```
The controller expands it before validation; feature selectors remain sibling fields.
Frequency × severity nests two stages (only when the dataset has a claim-count column):
`{{"structure":"frequency_severity","stages":{{"frequency":{{"estimator":"lightgbm","objective":"poisson"}},"severity":{{"estimator":"lightgbm","objective":"gamma"}}}}}}`

Structures: {structure_list}. Estimators (only these obj × enc combos are legal —
invalid ones are rejected before running):
{estimator_lines}

Target *shape* → objective (the handoff names the active target's shape):
**has zeros** (pure premium / incidence) → tweedie/poisson/squared_error;
**counts** → poisson/tweedie/squared_error; **strictly positive** (severity) →
gamma/squared_error. Features default to all eligible predictors; restrict with
`model.feature_inclusions/exclusions` using names from the handoff.{foundation_note}

### Option B — run-local script (escape hatch, for novel models)

Expose `fit_predict`; return a `Prediction` (the framework converts unit→total
and calibrates for you) **or** a raw `np.ndarray` of TOTALS (you own calibration):
```python
from autoresearch.models.prediction import Prediction

def fit_predict(train, score, *, feature_inclusions=None,
                feature_exclusions=None, **hyperparameters):
    ...  # fit on `train`
    return Prediction(values=pred_rates, unit="rate"), notes
```
If you return a raw array instead: multiply rates by the weight column (`score[EXPOSURE]`,
the name the handoff prints); gamma/log losses need `y > 0` (split freq×sev or use Tweedie); encode categoricals
(`'B12'`): lightgbm `category` dtype, xgboost/sklearn ordinal/one-hot; early-stop
on a train-internal split only; and calibration is mandatory —
`apply_training_calibration(pred_score, pred_train, actual_train_cost)` from
`autoresearch.models.calibration` (factor = Σactual/Σpred). Feature names come
from the handoff — **build features only from that named list** (never sweep
"all remaining columns" into the model; the framework strips target and id
columns from script frames, and `score` carries no targets at all). Column
constants (`from autoresearch.models.dispatcher import`):
{const_lines}

## Compute budget

Per-experiment wall-clock budget: **{base} min for the first {per} experiments,
+{incr} min every {per} thereafter** — `budget_minutes = {base} + {incr} × (N // {per})`.
The challenger is refit ~5× per comparison (1 fit + 4 CV folds), so budget your
single fit at ~1/5 of that. A close call (win rate in [0.50, 0.75]) escalates to
more folds — up to ~13× the single fit, *outside* the budget alarm — so leave
headroom. Watch `n_estimators × (1/learning_rate)`, `num_leaves`/`max_depth`,
and row count.

## Decision policy

Comparisons run under `gate_mode = {gate_mode}` and stop at `pending_llm`; the
mechanical gates are **advisory**, the verdict is yours. Before deciding, review
`{gate_metric}`, `rank_gini_weighted`, `asym_pricing_loss` (lower is better;
penalises under-pricing 4×), and the calibration ratio. Then:
- **promote** — clean win; replaces the global champion + fires holdout eval.
- **local_promote** — useful progress for its research line, not a champion.
- **reject** — insufficient/contradictory evidence; keep it as a learning.

Record a structured `--reason-code` with the free-text rationale:
`clear_win`, `line_progress`, `noise`, `inferior`, `artifact_suspected`,
`calibration`, or `other`. This keeps cross-run memory queryable without
discarding the scientific explanation.

### Adaptive search — design experiments from results, not in advance

Treat the cycle count as a **budget, not a to-do list**. Submit **one**
experiment, see its result, and let that result shape the next hypothesis. Do
**not** pre-commit to a fixed slate of N experiments at the start of a run.
After each cycle, before proposing the next, state what the last result changed
about your thinking — a surprising, degenerate, or near-miss result is a signal
to run a quick **diagnostic** on *why* (e.g. a model that scores no better than
the flat baseline is probably broken, not merely weak — investigate before
moving on), not a cue to advance to the next pre-decided idea. You **may** commit
to a short, explicit sequence when you deliberately want comprehensive coverage
of a defined set (e.g. a feature-engineering sweep, or a head-to-head of a few
model families) — name the sequence and why, still read results between steps,
and abandon it early if a result makes it moot.

**Search policy:** let breadth **emerge** from these adaptive choices rather than
from a coverage checklist written up front. Prefer breadth over depth: rotate
exploration axis after 2 same-axis experiments and cap same-model-family tuning
at 3. A plateau forces a **structural** change (new family or target framing),
not more tuning — re-tuning at a plateau is provably below the gate's noise floor.

## Proposal contract (what you must supply)

Supply only the fields that encode your scientific choice:

{proposal_fields}

plus an `experiment_config` with `model_family`, `target_strategy`, and a model
implementation — **either** `model.recipe` (preferred; set `model_family =
"recipe"`) **or** `model.script_path` (point it at your `model_<name>.py` for a
novel method).

The controller derives everything else from the champion, the recommended tree
action, and the research-line registry — you do **not** need to send:
{derived_fields}, `parent_branch_id`, `branch_action`, the fixed
`preprocessing` block, or the duplicate `experiment_config.experiment_name` /
`experiment_config.parent_experiment_id`. To deviate from a default (e.g. a
different tree parent or research line), set that field explicitly; when you
diverge from the recommended tree action, also include
`tree_policy_override_rationale`. The handoff embeds a ready-to-fill template.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if AGENT.md differs from freshly generated output.",
    )
    args = parser.parse_args()

    _validate_commands()
    content = render()
    targets = [AGENT_MD, *AGENT_MIRRORS]

    if args.check:
        stale = [
            t.relative_to(REPO_ROOT)
            for t in targets
            if (t.read_text(encoding="utf-8") if t.exists() else "") != content
        ]
        if stale:
            names = ", ".join(str(p) for p in stale)
            print(
                f"{names} out of sync with sources. "
                "Run: python scripts/generate_agent_contract.py",
                file=sys.stderr,
            )
            return 1
        print("AGENT.md (and harness mirrors) in sync.")
        return 0

    for target in targets:
        target.write_text(content, encoding="utf-8")
    written = ", ".join(str(t.relative_to(REPO_ROOT)) for t in targets)
    print(f"Wrote {written} ({len(content):,} bytes each).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
