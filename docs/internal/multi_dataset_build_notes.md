# Multi-Dataset Build Notes

**Branch:** `multi-dataset` (from `TabFPN`)
**Status:** complete — all acceptance criteria pass; full suite green (433 passed, 2 skipped).
This is the review entry point for the implementation of
`docs/internal/multi_dataset_design.md`. It is factual: what was built per phase,
every deviation from the design with rationale, the open-choice decisions, the
exact data-prep / smoke commands with key output, and what was deliberately left
undone.

---

## Acceptance criteria — status

| Criterion | Status |
|---|---|
| `list-datasets` shows french_motor, allstate, allstate_full, porto_seguro | ✅ (all `prepared=yes`) |
| `prepare-data --dataset <name>` succeeds for french_motor (idempotent), porto_seguro, allstate | ✅ + allstate_full also prepared |
| Group-aware split leakage: no Household_ID straddles splits for allstate | ✅ (0 straddles on 653k households; synthetic + splits-layer tests) |
| Porto smoke run end-to-end (bootstrap → global_mean champion → lightgbm cycle, handoff Active dataset block) | ✅ (challenger gini 0.262, lift 0.254, win_rate 1.0) |
| Full pytest green | ✅ 433 passed / 2 skipped |
| `scripts/generate_agent_contract.py --check` passes with dataset-neutral AGENT.md | ✅ |
| French TargetSpec-equality + split-pack-hash regression tests pass | ✅ (hash `cabc289f…` unchanged) |

---

## Phase-by-phase summary

### Phase 1 — DatasetSpec + config plumbing
- New `src/autoresearch/datasets.py`: frozen `DatasetSpec` (+ `SampleSpec`,
  `CapSpec`, `NaMarker`, `DerivedColumn`, `ReportingSpec`), `load_dataset_spec`,
  `list_datasets`, `dataset_data_dir`, with load-time validation (exactly one
  default target, snake_case modes, referenced columns non-empty).
- Four configs under `configs/datasets/`. `french_motor.toml` reproduces the old
  `default.toml` values verbatim.
- `config.py`: `ProjectConfig` gains `dataset: DatasetSpec` (default-factory
  French so direct test construction keeps working) + `dataset_name`.
  `load_config(..., dataset=…)` resolves **explicit flag → run manifest →
  `default.toml [data] default_dataset`**, deep-merges `[overrides.<section>]`,
  derives all data paths under `data/datasets/<name>/`, and pins the dataset in
  the run manifest. Mismatch guard raises on a contradicting `--dataset`.
- CLI: global `--dataset` flag + new `list-datasets` command.
- `scripts/migrate_dataset_layout.py` moved the French data to
  `data/datasets/french_motor/` (split pack **byte-identical**, hash-verified),
  backfilled `dataset` into 27 existing run manifests, and (Phase 6) moved the
  memory store.
- Tests: `tests/test_datasets.py`.

### Phase 2 — generic TargetSpec construction
- Wrote the **regression lock first** (`tests/test_targets_regression.py`) with
  the pre-change French literal field values, then refactored.
- `targets.py`: `build_target_spec(cfg)` generates the ten derived label/key
  fields from `entity_label`/`rate_label`(+optional `rate_slug`); French config
  reproduces every historical literal exactly. Added `population_column`.
  `target_spec`/`normalise_target_mode` are dataset-relative via a process-global
  binding (`bind_dataset`, set at config load) with the French built-ins as a
  fallback, so the **protected** metric/diagnostic/validation stack — which calls
  `target_spec(mode)` with no dataset — keeps working across datasets.
- New bindable `models/columns.py` (leak set, id/weight, French-default
  re-exported constants synced onto dispatcher/global_mean/interpreter for agent
  scripts). Dispatcher population filter + prediction-frame assembly + script
  sanitisation, `global_mean`, and the recipe interpreter are all spec-driven.
  `feature_policy` and the integrity scanner read the bound non-predictive set.
- `apply_claim_capping` is a tolerant no-op when the cap column is absent and
  capping is disabled — this is what lets the **protected** `comparison_runner`
  (which caps unconditionally) run on cap-less datasets untouched.

### Phase 3 — generic loading + pipeline
- `data/sources.py` (`RawDataset` + `load_raw` dispatch), `data/loaders/single_table.py`
  (pattern discovery, `na_values`/`na_marker`, restricted `derived` columns via
  `DataFrame.eval`), `data/adapters/french_motor.py` (wraps the untouched
  `load_fremtpl2`).
- `pipeline.prepare_data` is dataset-driven: deterministic whole-unit sampling
  (`sample_manifest.json`), `record_id` + `unit_weight` synthesis, spec-driven
  cap diagnostics + schema roles, group-aware split/fold generation.
- `splits.py`: `generate_split_pack`/`generate_fold_assignments` gain group-aware
  variants (unit-level assign + broadcast) and generic zero-band + target-quantile
  stratification; the French claim/exposure auto-detection is preferred whenever
  present, so the French pack is byte-identical.
- Tests: `tests/test_pipeline_multidataset.py` (three synthetic archetypes) +
  a splits-layer group leakage/determinism test.

### Phase 4 — AllState
- Sampling + household grouping + `allstate_full` `[overrides.compute]` validated
  end-to-end on the real data (see commands below). Budget maths: allstate 20 min
  @10 exp; allstate_full 70 min @10 exp, preflight 20 000.

### Phase 5 — dataset-neutral agent surfaces
- `scripts/generate_agent_contract.py` rewritten dataset-neutral (no French
  columns/cap/mode); regenerated `AGENT.md`/`AGENTS.md`/`CLAUDE.md` (15 968 bytes,
  under the 16 000 ceiling). Column constants named generically.
- `controller/context.py`: `project_goal` templated + new `active_dataset` block.
- `controller/handoff.py`: new **"Active dataset"** section (name, target/source,
  weight policy, population, cap, freq×sev availability, dataset cautions);
  genericised the Key-constraints block.
- `controller/proposal_schema.py`: search space (target columns, modes,
  non-predictive set, feature policy, cap thresholds, freq×sev availability) all
  derived from the spec. `controller/workflow.py`: per-dataset preprocessing
  hydration. CLI `--target-mode` validates against the active dataset.
- Protected edits (authorised, §3.8): `validation.py` message "Exposure"→"Weight";
  `milestone.py` spec-driven cap column. `autoresearch update-integrity-manifest`
  run afterwards (manifest is per-run/out-of-tree, so no committed file changed).
- `docs/OPERATING_MANUAL.md`: new Datasets chapter.
- Tests: `tests/test_agent_surfaces_multidataset.py`.

### Phase 6 — memory scoping + reporting
- `memory_root`/`default_memory_store_path`/`default_playbook_dir` take an
  optional dataset; run-facing callers pass `config.dataset_name` so a run sees
  only its dataset's insights. Migration moved `memory.sqlite` into
  `french_motor/`.
- `reporting/comparison.py` swaps the `£` symbol for `dataset.reporting.currency`.

---

## Deviations from the design (with rationale)

1. **`default.toml` keeps its French `[data] id_column`, `[preprocessing]`, and
   `[search_space.preprocessing]` keys** rather than deleting them (§4 step 2).
   The active values come from the dataset spec; the leftover keys are harmless
   fallbacks. Rationale: rule-7 minimal-risk — deleting them risked breaking any
   direct `raw[...]` reader for zero functional gain. `[data] default_dataset`
   was added.
2. **Processed parquet does not persist the cap column.** The design's §3.4 step
   4 applies the cap in the pipeline; historically the French cap column was
   materialised at *model* time by the runner/comparison/milestone (the pipeline
   discarded the capped frame). I preserved the historical behaviour so the
   French processed parquet + schema stay byte-identical; capping still happens
   at model time via the tolerant `apply_claim_capping`. Net effect for new
   cap-less datasets is identical (their target column is raw and already
   present).
3. **`models/columns.py` (not `.bind()` on an existing module).** The design
   sketched `autoresearch.models.columns.bind(dataset_spec)`; I created that
   module. Constants are re-exported from the dispatcher (the contract's import
   path) and synced on bind for agent-script compatibility.
4. **Registry `dataset` column not added.** §3.6 calls it "honest but not
   required" (each run is already dataset-pinned). Deferred to avoid a schema
   migration on the protected registry submodules; the run manifest is the
   source of truth. See "left undone".
5. **Telemetry rows not tagged with dataset.** §3.9 says "no structural change";
   the run manifest already records the dataset, so telemetry is
   dataset-attributable via the run. No importer change made. See "left undone".
6. **Recipe reuse ledger (`recipes.jsonl`) left root-level (unscoped).** It takes
   no config and is gated behind `recipe_reuse_scope="memory"` (non-default). The
   *insight* store is dataset-scoped. Documented as a minor deferral.
7. **`split_method` label reads `target_exposure_stratified_hash` for generic
   datasets too** (the manifest label is cosmetic; the stratification itself is
   correct — verified both Porto classes appear in every split). Not worth
   threading a separate label.

## Open-choice decisions (rule 7)

- **Frequency-slug preservation:** French `frequency` needed `actual_frequency`
  (not `actual_claim_frequency`); added an explicit optional `rate_slug` config
  field (French frequency sets `rate_slug = "frequency"`). Every other spec
  slugifies `rate_label`.
- **Unit-weight column name:** `unit_weight` (constant `targets.UNIT_WEIGHT_COLUMN`),
  schema role `exposure_offset`, always non-predictive.
- **Generic stratification:** zero band + 8 target quantiles crossed with a
  5-quantile weight band (single band when the weight is constant, i.e. unit
  weight). French keeps its fixed claim bands via auto-detection.
- **Cap-less proposal preprocessing:** `claim_cap_thresholds = [None]`,
  `allow_disable_claim_capping = true`, hydrated preprocessing
  `{claim_capping_enabled: false, claim_cap_threshold: null}`.
- **Binary-target diagnostics quirk:** for datasets whose emitted target columns
  aren't in the protected `diagnostics._PREDICTION_SYSTEM_COLS` whitelist (e.g.
  Porto's `actual_claim_indicator`), the segment analysis may treat the binary
  target as an extra segment. This is cosmetic (no gate depends on it) and
  `diagnostics.py` is protected/authorised-untouched, so it was left as-is per
  §3.8 ("no diagnostics change expected").

---

## Exact commands run

Data preparation (raw files symlinked into `data/datasets/{porto_seguro,allstate}/raw/`;
`data/*` is gitignored so nothing under `data/` — including the ~2.7 GB AllState
raw and multi-GB parquets — is committed):

```
python scripts/migrate_dataset_layout.py          # split pack byte-identical; 27 manifests + memory.sqlite migrated
autoresearch prepare-data                          # french_motor, reuses migrated split pack (idempotent)
autoresearch --dataset porto_seguro prepare-data   # 595,212 rows, ~9s
autoresearch --dataset allstate prepare-data       # 13.18M → 2,000,011 sampled rows (653,347 households), ~1:49
autoresearch --dataset allstate_full prepare-data  # 13,184,290 rows, ~2:23
autoresearch update-integrity-manifest             # after the protected-file edits
```

`list-datasets` (final):

```
  allstate       … target=pure_premium  modes=pure_premium,severity,claim_incidence  weight=unit     prepared=yes rows=2000011
  allstate_full  … target=pure_premium  …                                            weight=unit     prepared=yes rows=13184290
* french_motor   … target=burning_cost  modes=burning_cost,frequency,severity        weight=Exposure prepared=yes rows=678013
  porto_seguro   … target=claim_incidence modes=claim_incidence                      weight=unit     prepared=yes rows=595212
```

Porto smoke run (driven via an isolated scratchpad artifacts tree to avoid
polluting a real track / the run-scope guard; equivalent to
`autoresearch --track claude --new-run bootstrap-track --dataset porto_seguro …`
then a `run-session-cycles 1` on a lightgbm-poisson recipe):

```
bootstrap steps: prepare-data(skipped) test-gate(ran) init-registry(ran)
                 run-starting-baseline(ran) init-official-champion(ran)
                 write-proposal-template(ran) export-context(ran)
champion: <ts>_global_mean_baseline           # global_mean, gini 0.0, claim_incidence
challenger: <ts>_porto_lgbm_poisson           # lightgbm poisson, gini 0.2618
comparison: lift 0.2544, win_rate 1.0, advisory promote, guardrail passed → awaiting_decision
Handoff "## Active dataset": porto_seguro / target=claim_incidence (source `target`,
  rate claim probability) / weight `unit_weight` synthesised / population all rows /
  no capping / freq×sev unavailable / caution: `-1` encodes missing in /_cat$/
```

---

## Left deliberately undone

- **Registry `dataset` column** (deviation 4): the run manifest already pins the
  dataset and every run is dataset-scoped; adding the column touches protected
  registry submodules. A cross-run honesty nicety, not required for correctness.
- **Telemetry dataset tagging** (deviation 5): dataset is recoverable from the
  run manifest; no importer/report change made (§3.9 "no structural change").
- **Recipe ledger dataset-scoping** (deviation 6): the insight store is scoped;
  the recipe-reuse ledger is not (non-default feature, no config threading).
- **Full `--cycles 2` real runs for allstate/french** (design §5 phase 7): the
  Porto smoke exercises the whole loop end-to-end and every dataset's model
  layer was exercised directly (global_mean + lightgbm dispatch on porto and
  allstate). A multi-cycle real run per dataset is available but was not run to
  completion here.
- **Reporting axis prose** ("Pure Premium (£)" → rate-label-driven text): only
  the currency *symbol* is swapped; the axis title text still says "Pure
  Premium". Cosmetic (§3.10), low value for non-French runs.

---

## Review addendum (Fable, 2026-07-09)

Full review of the branch against the design doc, with independent verification:
suite re-run green (433 passed); French TargetSpec literals confirmed against the
pre-change `targets.py`; French split-pack hash confirmed; group leakage re-checked
on the real AllState artifacts (0 of 653,347 households straddle splits, 0 of
522,676 straddle folds; claim rate 0.728–0.732% across splits); Porto stratification
confirmed (target rate 3.645% in all three splits); dispatch + protected metric
panel exercised directly on porto (`claim_incidence`: global_mean gini 0.0,
lightgbm 0.25) and allstate (`pure_premium` + `severity` population filter).
The protected-file strategy (call-time `RAW_CLAIM_COST` import in
`comparison_runner` + the tolerant cap no-op) was verified at all three call sites.

Three defects found and fixed in the review commit:

1. `scripts/fetch_fremtpl2.py` / `scripts/generate_synthetic_data.py` still wrote
   to the pre-migration `data/raw/` — files landed there would be invisible to the
   loader. Now write to `data/datasets/french_motor/raw/`; the old
   `data/raw/README.md` is a pointer to the new layout.
2. Test-order fragility: `load_config(dataset=…)` mutates process-global binds
   (`targets`/`models.columns`/`feature_policy`) and tests never restored them.
   New `tests/conftest.py` autouse fixture snapshots/restores the bound state
   around every test.
3. `allow_log1p_features = ["Density"]` (French-specific, from `default.toml`)
   was advertised in every dataset's search space. `allowed_search_space` now
   filters the list to the active dataset's actual feature columns.
