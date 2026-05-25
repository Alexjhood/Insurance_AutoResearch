# Auto-Research Platform — Implementation Plan

**Goal**: Transform the existing actuarial lab backbone into a true Karpathy-style autonomous research loop, where Claude Code / Codex iterates on model forms, feature engineering, and hyperparameters with minimal human involvement, producing progressive measured improvements assessed on holdout data.

---

## Assessment summary

The current project is a strong deterministic experimentation backbone but restricts the LLM to a closed JSON proposal schema with four hard-coded model families and no feature engineering surface. The changes below open the search space, harden the gate against gaming, and give the LLM a complete operating manual so a CC session can just work.

---

## Bucket 1 — Open the search space (implemented)

### 1.1 Demote JSON proposal schema to optional convenience
- The primary agent workflow is now AGENT.md-driven: CC/Codex edits code on a branch, runs `autoresearch run-experiment`, and reads the result.
- `controller/proposal_schema.py` and `controller/workflow.py` are retained for the legacy file-handoff path; they are not the primary loop.

### 1.2 Python feature-builder path in experiment TOML
- Experiment TOMLs may specify `feature_builder_module = "autoresearch.features.my_transforms"`.
- The dispatcher calls `module.build_features(frame) -> frame` before fitting and scoring.
- The LLM creates feature builder files under `src/autoresearch/features/`.
- New entry point protocol: `build_features(frame: pd.DataFrame) -> pd.DataFrame`.

### 1.3 Open model-family registry
- The dispatcher tries `importlib.import_module(f"autoresearch.models.{model_family}")` for any unknown family.
- New model files expose `fit_predict(train, score, *, feature_inclusions, feature_exclusions, **hp) -> (predictions_array, notes_dict)`.
- No validator changes needed; the integrity scan catches holdout refs automatically.
- LLM may `pip install lightgbm / xgboost / catboost` and update `pyproject.toml`.

---

## Bucket 2 — Harden the gate (implemented)

### 2.1 Code-integrity scan (`src/autoresearch/utils/integrity.py`)
- Runs in `run_experiment()` before model fitting.
- AST-walks every `.py` file under `src/autoresearch/models/` and `src/autoresearch/features/`.
- Rejects any file that references `milestone_holdout`, `holdout_vault`, `load_holdout_dataset`, or `AUTORESEARCH_MILESTONE_TOKEN` (except the vault and milestone modules themselves).
- Failure raises `ValueError`; the experiment is recorded as `failed`.

### 2.2 Protected-file integrity manifest
- `artifacts/integrity_manifest.json` stores SHA256 hashes of:
  - `src/autoresearch/evaluation/metrics.py`
  - `src/autoresearch/evaluation/resampling.py`
  - `src/autoresearch/data/holdout_vault.py`
  - `src/autoresearch/experiment_registry/registry.py`
- Written at `autoresearch init-registry`.
- Checked in `compare_experiments()`. Mismatch → comparison blocked with `decision = "blocked_integrity_violation"`.
- To accept an intentional protected-file change: `autoresearch update-integrity-manifest`.

### 2.3 Mandatory pytest
- `run_experiment()` runs `pytest --tb=short -q` before training.
- If tests fail, experiment is recorded as `failed` immediately.
- This forces the LLM's code edits to satisfy all 52 existing tests plus any new ones it writes.

### 2.4 Auto-holdout on every promotion (`src/autoresearch/milestone.py`)
- When promotion gate returns `"promote"`, `evaluate_on_holdout()` fires automatically.
- Refits the champion model on `train ∪ search_validation` (full search partition) then scores on `milestone_holdout`.
- Writes `artifacts/milestone_reports/<promotion_id>.md` with: Tweedie deviance, Gini, double-lift, calibration deciles, SV→holdout overfitting gap.
- Report is included in the next session's LLM context.

### 2.5 Default `use_cv = true`
- Reduces single-split noise; the loop competes on 5-fold CV mean rather than one lucky validation draw.
- Existing tests pass; promotion gate thresholds apply to the CV mean.

---

## Bucket 3 — Loop legibility (implemented)

### 3.1 `AGENT.md` — operating manual for CC/Codex
- Self-contained document at repo root telling CC exactly what to read, what to write, how to run experiments, how to interpret results, and how to log findings.
- Defines the new-model-family and feature-builder entry-point protocols.
- Includes safety rules: never read holdout, never edit protected files without `update-integrity-manifest`, always pass pytest.

### 3.2 `docs/RESEARCH_LOG.md` — append-only research narrative
- Structured per-cycle entries: hypothesis, changes made, outcome (metrics + promotion decision), interpretation, what to try next.
- Framework auto-appends a one-line summary after every comparison.
- LLM reads this at session start and appends a full entry after each cycle.

### 3.3 `autoresearch evaluate-milestone <experiment_id>` CLI command
- Manually trigger holdout evaluation for any experiment (not just the promoted champion).
- Useful for milestone checkpoints outside the promotion path.

### 3.4 Research log in LLM context
- `controller/context.py` includes the last N lines of `RESEARCH_LOG.md` in the JSON context bundle so the LLM always knows recent research history.

---

## What was NOT changed

- The promotion gate statistics (relative lift, win rate, Bonferroni CI) — these are correct and should not be weakened.
- The holdout token mechanism — it works and provides hard architectural separation.
- The SQLite registry and experiment artifacts structure — stable, tested.
- The mock proposer and file-handoff path — retained for backward compatibility.

---

## How to run the new loop

See `AGENT.md` for the full agent operating manual. Short version:

```bash
# First time
autoresearch prepare-data
autoresearch init-registry          # also writes integrity_manifest.json
autoresearch run-all-baselines
autoresearch init-official-champion
autoresearch export-context

# Each research cycle (agent does this)
# 1. Read RESEARCH_LOG.md, export-context output, recent experiment metrics
# 2. Form a hypothesis
# 3. Create/edit code: new model file or feature builder, plus experiment TOML
# 4. pytest  (must pass)
# 5. autoresearch run-experiment configs/experiments/<new>.toml
# 6. autoresearch compare-to-champion <experiment_id>
# 7. Read promotion_report.json and (if promoted) milestone_report
# 8. Append to RESEARCH_LOG.md
# Repeat
```
