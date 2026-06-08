#!/usr/bin/env python3
"""Generate the compact agent runtime contract (``AGENT.md``) from code/config.

The agent loads ``AGENT.md`` into context at the start of every run (the console
seed prompt says "Read AGENT.md, then bootstrap ..."). To keep that startup cost
small *and* prevent the manual from drifting out of sync with the CLI, the
runtime contract is generated here from authoritative, importable sources:

* command names      -> ``autoresearch.cli.COMMANDS``
* proposal fields    -> ``autoresearch.controller.proposal_schema.REQUIRED_PROPOSAL_TEXT_FIELDS``
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
    REQUIRED_PROPOSAL_TEXT_FIELDS,
)
from autoresearch.models import dispatcher  # noqa: E402
from autoresearch.utils.integrity import PROTECTED_RELATIVE_PATHS  # noqa: E402

AGENT_MD = REPO_ROOT / "AGENT.md"
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
    ("park-research-line", "Park an exhausted research line."),
    ("clear-line-champion", "Drop a local incumbent that looks artefactual."),
    ("export-context", "Refresh the handoff/context bundle on demand."),
]

# Column constants the model script needs, sourced from the dispatcher module.
_COLUMN_CONSTANTS = [
    ("EXPOSURE", dispatcher.EXPOSURE, "offset; weights + rate->total only, never a feature"),
    ("CLAIM_COST", dispatcher.CLAIM_COST, "training target (burning-cost mode)"),
    ("CLAIM_COUNT", dispatcher.CLAIM_COUNT, "training target (frequency mode)"),
    ("CLAIM_EVENTS", dispatcher.CLAIM_EVENTS, "alternative claim count"),
    ("RECORD_ID", dispatcher.RECORD_ID, "policy identifier"),
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
    caps = cfg["search_space"]["preprocessing"]["claim_cap_thresholds"]

    base = compute["base_budget_minutes"]
    incr = compute["budget_increment_minutes"]
    per = compute["experiments_per_increment"]
    gate_mode = evaluation["gate_mode"]
    gate_metric = evaluation["gate_primary_metric"]
    target_mode = evaluation["target_mode"]
    cap = caps[0]

    cmd_lines = "\n".join(
        f"- `{name}` — {desc}" for name, desc in WORKFLOW_COMMANDS
    )
    protected_lines = "\n".join(f"- `{p}`" for p in PROTECTED_RELATIVE_PATHS)
    const_lines = "\n".join(
        f"- `{name} = \"{value}\"` — {desc}" for name, value, desc in _COLUMN_CONSTANTS
    )
    proposal_fields = ", ".join(f"`{f}`" for f in REQUIRED_PROPOSAL_TEXT_FIELDS)

    return f"""\
<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with: python scripts/generate_agent_contract.py
     Source of truth: cli.COMMANDS, proposal_schema, dispatcher, integrity,
     configs/default.toml. Full human manual: {MANUAL_REL}. -->

# AGENT.md — Auto-Research Runtime Contract

You are the research agent for an autonomous insurance target-modelling loop on
the French Motor dataset (freMTPL2, ~678K policies). The active target is
**burning cost** (`{dispatcher.CLAIM_COST}`) unless the run sets
`target_mode = "frequency"` (current default: `{target_mode}`). Maximise
**exposure-weighted Gini** (`{gate_metric}`) on the search-validation split;
every promotion is re-checked on a protected holdout. Each run starts with the
`global_mean` baseline as champion — your first model only has to beat a flat
exposure-weighted rate.

For anything not covered here, read **{MANUAL_REL}** (full manual: dataset
schema, metric panel, gate modes, research-line mechanics, worked examples).

## Hard safety constraints — never break

1. **Never read the holdout.** Do not import `autoresearch.data.holdout_vault`
   or reference `milestone_holdout` / `AUTORESEARCH_MILESTONE_TOKEN` in model or
   feature code. The integrity scanner fails the experiment.
2. **Never edit protected evaluation files** (comparisons block until an
   operator runs `autoresearch update-integrity-manifest`):
{protected_lines}
3. **Never change** the primary metric, gate thresholds, the fixed split
   (`split_pack.csv`, `data/processed/`), or the claim cap (fixed at {cap:,}).
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
     --model-provider <provider> --model-name <model-name>
   ```
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
     --decision promote|local_promote|reject --rationale "..."
   ```
   `run-session-cycles` **never auto-promotes** — it always stops for your
   `record-decision`. Repeat propose -> cycle -> decide. **Design the next
   experiment only after reading the current one's result** — N is a budget of
   cycles to spend adaptively, not a slate to plan in advance (see "Adaptive
   search").
5. **Log** each cycle's hypothesis, outcome, and learning to this run's
   `RESEARCH_LOG.md`.

### Workflow commands

All take `--track <your-tool-name> --run-id <id>` (omit `--run-id` only on the
first `bootstrap-track`, which prints the id):

{cmd_lines}

### When a cycle needs repair

A cycle can stop in **`needs_repair`** (instead of `awaiting_decision`) when the
model script fails preflight, output validation, or the positive-lift check. The
framework writes `repair_request_<N>.json` into the proposal directory. Recover
without guessing — the file tells you what to do:

1. Read `repair_request_<N>.json`. It names the script to write
   (`write_script`, e.g. `model_attempt_2.py`), the `failed_checks`, the
   `error_type`, and an `instruction` field.
2. Write that `model_attempt_<N>.py` fixing the named checks, keeping the same
   `fit_predict` interface and never touching holdout data.
3. Rerun `run-session-cycles 1` — it automatically picks up the new attempt.

You get up to **3 attempts**. Attempt 2 should move *opposite* the failure (if a
tree was too deep, go shallower — not halfway back); if attempt 2 is still no
better, let it fail rather than spend attempt 3 on a near-duplicate.

## Model interface (compact)

A run-local model script must expose:
```python
def fit_predict(train, score, *, feature_inclusions=None,
                feature_exclusions=None, **hyperparameters) -> tuple[np.ndarray, dict]:
    ...  # fit on `train`, return active-target TOTAL predictions + notes
```
Rules that are easy to get wrong (full detail in the manual):
- **Return totals, not rates.** Multiply predicted rates by
  `score["{dispatcher.EXPOSURE}"]` before returning.
- **Target has exact zeros** — losses needing `y > 0` (gamma/log) error unless
  you split frequency×severity. Prefer Tweedie (`lightgbm`/`xgboost`/statsmodels)
  for direct pure-premium. Do **not** pass `tweedie_power` to
  `HistGradientBoostingRegressor`.
- **Encode categoricals** (string levels like `'B12'`): lightgbm `category`
  dtype; xgboost/sklearn need ordinal/one-hot.
- **Use early stopping** with a train-internal split (never search-validation or
  holdout).
- **Feature names** come from the handoff bundle's
  `allowed_search_space.feature_columns`; `non_predictive_columns` there must
  not be used as predictors. Read them from the handoff — do not assume a schema.

Column constants (`from autoresearch.models.dispatcher import ...`):
{const_lines}

**Calibration is mandatory** — apply the framework's one-parameter aggregate
calibrator just before returning. Exact signature and usage:
```python
from autoresearch.models.dispatcher import CLAIM_COST, CLAIM_COUNT
from autoresearch.models.calibration import apply_training_calibration

# pred_train and pred_score must already be TOTALS (rate × exposure), not rates.
actual = (train[CLAIM_COUNT].to_numpy()
          if hyperparameters.get("target_mode") == "frequency"
          else train[CLAIM_COST].to_numpy())
pred_score, calib_factor = apply_training_calibration(pred_score, pred_train, actual)
notes["calib_factor"] = round(float(calib_factor), 4)
notes["native_pred_to_actual_ratio"] = round(1.0 / calib_factor, 4)
return pred_score, notes
```
`apply_training_calibration(pred_score, pred_train_cost, actual_train_cost)`
returns `(calibrated_pred_score, calib_factor)` where
`calib_factor = sum(actual_train_cost) / sum(pred_train_cost)`.

## Compute budget

Per-experiment wall-clock budget: **{base} min for the first {per} experiments,
+{incr} min every {per} thereafter** — `budget_minutes = {base} + {incr} × (N // {per})`.
The challenger is refit ~5× per comparison (1 fit + 4 CV folds), so budget your
single fit at ~1/5 of that. Watch `n_estimators × (1/learning_rate)`,
`num_leaves`/`max_depth`, and row count.

## Decision policy

Comparisons run under `gate_mode = {gate_mode}` and stop at `pending_llm`; the
mechanical gates are **advisory**, the verdict is yours. Before deciding, review
`{gate_metric}`, `rank_gini_weighted`, `asym_pricing_loss` (lower is better;
penalises under-pricing 4×), and the calibration ratio. Then:
- **promote** — clean win; replaces the global champion + fires holdout eval.
- **local_promote** — useful progress for its research line, not a champion.
- **reject** — insufficient/contradictory evidence; keep it as a learning.

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

## Proposal contract (required top-level fields)

{proposal_fields}

The handoff embeds the exact proposal template, including the nested
`experiment_config` and every field's placeholder — fill those in rather than
reading a separate schema file. Slimmer JSON is rejected at ingestion and wastes
a cycle.
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

    if args.check:
        current = AGENT_MD.read_text(encoding="utf-8") if AGENT_MD.exists() else ""
        if current != content:
            print(
                "AGENT.md is out of sync with its sources. "
                "Run: python scripts/generate_agent_contract.py",
                file=sys.stderr,
            )
            return 1
        print("AGENT.md is in sync.")
        return 0

    AGENT_MD.write_text(content, encoding="utf-8")
    print(f"Wrote {AGENT_MD.relative_to(REPO_ROOT)} ({len(content):,} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
