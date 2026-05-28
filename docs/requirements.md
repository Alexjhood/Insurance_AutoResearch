# Requirements

## Goal
Build a local auto-research platform for insurance target modelling on freMTPL2. Burning cost is the default target, and claim frequency can be selected explicitly for frequency-only runs.

## Priorities
1. Reproducibility
2. Agentic autonomy
3. Interpretability
4. Volatility-aware evaluation
5. Continuous experimentation

## Core capabilities
- Ingest freMTPL2 from local files
- Create anonymised agent-facing schema
- Persist split packs for repeatable evaluation
- Run deterministic baselines first
- Compare experiments with uncertainty-aware evaluation
- Track branch history and rationale
- Show results in a Streamlit dashboard
- Support automatic continuation of experiments
- Protect final holdout except at milestone checkpoints

## Key risks to mitigate
- Validation noise mistaken for improvement
- Public benchmark contamination / memorised heuristics
- Reward hacking against the evaluation protocol
- Non-reproducible experiment state
