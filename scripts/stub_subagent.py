#!/usr/bin/env python3
"""A scripted sub-agent that makes zero LLM calls.

Registered as the ``stub`` backend, this drives a real delegation through the
real CLI: it reads its handoff, drops a trivial recipe proposal, runs its cycle,
records a decision, and calls ``finish-delegation``. That exercises the whole
spawn → run → report pipeline for free, which is why it lives in the repo
permanently rather than in a test fixture.

It is deliberately dumb. Every proposal is a ``constant`` recipe — a flat
prediction that cannot beat the flat baseline — so cycles are fast and always
resolve to a rejection. What is under test is the *plumbing*, not the science.

Contract with the spawner: the run is already bootstrapped, and the launch
prompt (on stdin) names the track, run id, and cycle budget. Those also arrive
as ``AUTORESEARCH_TRACK`` / ``AUTORESEARCH_RUN_ID`` in the environment.

  --skip-finish   exit without calling finish-delegation, to provoke the
                  ``no_finish_delegation`` distress flag end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AUTORESEARCH = os.environ.get("AUTORESEARCH_CLI", "autoresearch")

# Objectives legal for a `constant` estimator on a has-zeros target. Rotating
# them gives each cycle a distinct proposal fingerprint, so the duplicate-proposal
# detector does not reject cycle 2 as a repeat of cycle 1.
_OBJECTIVES = ("tweedie", "poisson", "squared_error")

# Session states that mean "the framework is done with this cycle".
_STOP_STATES = {"waiting_for_repair", "failed", "completed", "paused"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-finish",
        action="store_true",
        help="Exit without calling finish-delegation (provokes no_finish_delegation).",
    )
    args = parser.parse_args()

    prompt = sys.stdin.read() if not sys.stdin.isatty() else ""
    track = os.environ.get("AUTORESEARCH_TRACK", "")
    run_id = os.environ.get("AUTORESEARCH_RUN_ID", "")
    if not track or not run_id:
        print("stub: AUTORESEARCH_TRACK/AUTORESEARCH_RUN_ID not set in environment", file=sys.stderr)
        return 2

    cycle_budget = _cycle_budget_from_prompt(prompt)
    print(f"stub: track={track} run={run_id} cycle_budget={cycle_budget}", flush=True)

    # Step 2 of the contract: read the handoff before forming any hypothesis.
    handoff = _run(track, run_id, "show-latest-handoff")
    if "Orchestration brief" not in handoff.stdout:
        print("stub: WARNING — handoff carries no Orchestration brief block", file=sys.stderr)

    _run(track, run_id, "start-session", "stub_delegation", "--max-cycles", str(cycle_budget))

    inbox = REPO_ROOT / "artifacts" / "tracks" / track / "runs" / run_id / "proposal_inbox"

    for cycle in range(1, cycle_budget + 1):
        print(f"stub: --- cycle {cycle}/{cycle_budget} ---", flush=True)
        _write_proposal(inbox, cycle)
        states = _run_cycle(track, run_id)
        if states is None:
            print("stub: cycle produced no parsable state; stopping", file=sys.stderr)
            return 1
        state = states[-1]
        name = state.get("state")
        result = state.get("latest_cycle_result") or {}
        print(f"stub: state={name} decision={result.get('decision')}", flush=True)

        if name == "awaiting_decision":
            _record_decision(track, run_id, result.get("comparison_id"), cycle)
        elif name == "awaiting_reflection":
            _record_reflection(track, run_id, cycle)
        elif name in _STOP_STATES:
            print(f"stub: stopping early in state {name}", flush=True)
            break

    if args.skip_finish:
        print("stub: --skip-finish set; exiting without finish-delegation", flush=True)
        return 0

    _run(
        track,
        run_id,
        "orchestrate",
        "finish-delegation",
        "--summary",
        (
            "Scripted stub delegation. Every cycle proposed a constant recipe, which "
            "cannot beat a flat baseline, so all cycles were rejected as expected. "
            "No scientific learning is claimed. This run exists to exercise the "
            "spawn/run/report pipeline. Nothing here smelled artifactual because "
            "nothing here was a real experiment."
        ),
    )
    print("stub: done", flush=True)
    return 0


def _cycle_budget_from_prompt(prompt: str) -> int:
    match = re.search(r"cycle budget\s*`?(\d+)`?", prompt)
    return int(match.group(1)) if match else 1


def _run(track: str, run_id: str, *args: str) -> subprocess.CompletedProcess:
    command = [AUTORESEARCH, "--track", track, "--run-id", run_id, *args]
    print(f"stub: $ {' '.join(command)}", flush=True)
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc


def _run_cycle(track: str, run_id: str) -> list[dict] | None:
    proc = _run(track, run_id, "run-session-cycles", "1")
    if proc.returncode != 0:
        return None
    try:
        states = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return states if isinstance(states, list) and states else None


def _write_proposal(inbox: Path, cycle: int) -> None:
    """Fill the framework-written template with a trivial, fast, losing recipe."""

    template_path = inbox / "proposal_template.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    objective = _OBJECTIVES[(cycle - 1) % len(_OBJECTIVES)]

    template["experiment_name"] = f"stub_constant_{objective}_c{cycle}"
    template["rationale"] = "Stub delegation: exercise the cycle pipeline, not the science."
    template["change_summary"] = f"Constant estimator with objective={objective}."
    template["expected_benefit"] = "None expected; a constant cannot beat a flat baseline."
    template["key_risk"] = "None; this is a plumbing test."
    template["exploration_axis"] = "model_family"
    template["approach_family"] = "constant"
    template["feature_representation"] = "raw"
    template["expected_learning"] = "Confirms the cycle machinery runs end to end."
    template["experiment_config"]["model_family"] = "recipe"
    template["experiment_config"]["model"] = {
        "recipe": {
            "structure": "direct",
            "estimator": "constant",
            "objective": objective,
            "encoding": "ordinal",
        }
    }
    # After an auto-rejection the template carries a reflection stub the framework
    # requires us to fill before it will accept the next proposal.
    if "previous_cycle_reflection" in template:
        template["previous_cycle_reflection"]["interpretation"] = (
            "The constant recipe scored at the flat baseline, as designed."
        )
        template["previous_cycle_reflection"]["next"] = (
            "Rotate the objective and run the next scripted cycle."
        )

    target = inbox / f"proposal_stub_c{cycle}.json"
    target.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"stub: wrote {target.name}", flush=True)


def _record_decision(track: str, run_id: str, comparison_id: str | None, cycle: int) -> None:
    if not comparison_id:
        print("stub: awaiting_decision without a comparison_id; skipping", file=sys.stderr)
        return
    _run(
        track,
        run_id,
        "record-decision",
        comparison_id,
        "--decision",
        "reject",
        "--rationale",
        "Constant recipe scores at the flat baseline; no lift, as designed.",
        "--reason-code",
        "inferior",
        "--interpretation",
        f"Cycle {cycle}: a constant prediction carries no signal, so it cannot beat the champion.",
        "--next",
        "Run the next scripted cycle with a rotated objective.",
    )


def _record_reflection(track: str, run_id: str, cycle: int) -> None:
    _run(
        track,
        run_id,
        "record-cycle-reflection",
        "--interpretation",
        f"Cycle {cycle}: auto-rejected, as a constant recipe should be.",
        "--next",
        "Run the next scripted cycle with a rotated objective.",
    )


if __name__ == "__main__":
    sys.exit(main())
