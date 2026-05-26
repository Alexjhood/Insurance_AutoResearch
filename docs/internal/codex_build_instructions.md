# Codex Build Instructions

Build in phases. Do not jump straight to the autonomous loop.

## Phase 0
Create:
- pyproject.toml
- src package structure
- tests structure
- config loading
- basic CLI entrypoint
- experiment registry skeleton
- Streamlit app skeleton

## Phase 1
Implement:
- raw dataset loader for freMTPL2
- anonymisation pipeline
- metadata/profile generation
- stable split pack generation
- saved outputs under data/metadata and data/splits

## Phase 2
Implement deterministic baselines:
- simple frequency model
- simple severity model or direct pure-premium baseline
- evaluation harness
- artifact persistence

## Phase 3
Implement uncertainty-aware comparison:
- repeated resampling
- paired comparison
- bootstrap summary
- champion vs challenger report

## Rules
- Prefer simple, inspectable code
- Keep modules small
- Add tests for core utilities
- Write clear docstrings
- Do not implement LLM-driven search until deterministic baselines work end-to-end