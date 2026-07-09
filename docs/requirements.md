# Requirements

## Goal
Build a local auto-research platform for tabular target modelling on registered datasets (freMTPL2 by default; AllState and Porto Seguro built in). Each dataset declares its target modes; the dataset default applies unless a mode is selected explicitly.

## Priorities
1. Reproducibility
2. Agentic autonomy
3. Interpretability
4. Volatility-aware evaluation
5. Continuous experimentation

## Core capabilities
- Ingest registered datasets from local files (config-driven; freMTPL2, AllState, Porto Seguro built in)
- Create dataset schema metadata using the source column names
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
