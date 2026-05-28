# Contributing

## Local Dev Setup

Follow the quickstart in [README.md](README.md) to set up a virtual environment and install the package:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/generate_synthetic_data.py
autoresearch prepare-data
```

## Tests

`pytest` must pass before any PR is merged:

```bash
pytest --tb=short -q
```

Add a smoke test for any new model family or feature builder. The test suite lives under `tests/`.

## Protected Files

The following files define the evaluation and promotion logic. Do not edit them during a research session:

- `src/autoresearch/evaluation/metrics.py`
- `src/autoresearch/evaluation/resampling.py`
- `src/autoresearch/data/holdout_vault.py`
- `src/autoresearch/experiment_registry/registry.py`

If you intentionally change one of these files, run:

```bash
autoresearch update-integrity-manifest
```

Explain why in the PR description. The experiment runner will block comparisons until the manifest is updated.

## Adding a Model Family

There are two paths, and the right one depends on who is adding it.

**Agent experiments (recommended):** an LLM agent adds a model by writing a
*run-local* Python script alongside its proposal JSON. The proposal's
`experiment_config.model.script_path` points at the script, which is copied
into the iteration folder, integrity-scanned, and executed. No edits to
`src/` are needed. See `AGENT.md` §"Option A" for the full proposal shape.

**Permanent additions to the package (rare):** only the `global_mean`
baseline lives under `src/autoresearch/models/` (built-in model families
were intentionally removed in favour of the open scripted-model surface —
see commit `8ba9b6f`). If you have a genuine reason to add a new built-in
(e.g. a second baseline that ships with the repo), the entry point must
expose:

```python
def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters):
    return predicted_target_array, notes_dict
```

Predictions must be target totals, not rates. In the default `burning_cost`
mode, return predicted claim costs (multiply by `exposure_term_a` if you model
pure premium). In `frequency` mode, return expected claim counts (multiply by
`exposure_term_a` if you model annual claim frequency). Apply
`autoresearch.models.calibration.apply_training_calibration` before
returning. See `src/autoresearch/models/global_mean.py` for the reference
shape.

## Adding a Feature Builder

New feature builders live under `src/autoresearch/features/`. The entry point must expose:

```python
def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    ...
```

The function must not drop existing columns, must not access holdout data, and must be deterministic.

## PRs

Keep diffs focused: one logical change per PR. Reference the relevant research track and run-id if the PR is motivated by an agent experiment.
