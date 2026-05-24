# Architecture

## Main modules
- data
- features
- models
- evaluation
- experiment_registry
- controller
- reporting
- dashboard

## High-level flow
1. Load raw data
2. Create anonymised metadata and stable split packs
3. Run deterministic baseline experiments
4. Store metrics and artifacts
5. Display results in dashboard
6. Later: allow LLM-driven experiment proposals
7. Promote only evidence-backed improvements

## Constraints
- Local Python project
- Streamlit dashboard
- No final holdout access during ordinary search
- Every experiment must be resumable and reproducible