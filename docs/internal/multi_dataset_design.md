# Multi-Dataset Support — Design

**Status:** proposed (2026-07-09)
**Scope:** make the AutoResearch loop run against any registered tabular
regression dataset, selected per run, starting with three: the existing French
Motor (freMTPL2) data plus AllState Claim Prediction and Porto Seguro Safe
Driver. New datasets must be addable by dropping a config file (plus, only when
the raw shape demands it, a small loader adapter) — no framework code changes.

Decisions already taken with the operator:

1. **Porto Seguro's binary target** is treated as generic rate regression
   (predicted claim probability as a rate, unit weight, `gini_weighted`
   primary metric — identical in spirit to the Kaggle normalized Gini). No
   classification machinery.
2. **AllState** is registered twice: a stratified ~2M-row subsample
   (`allstate`) as the default working dataset, and `allstate_full` (13.2M
   rows) with enlarged compute budgets for deliberate large-scale runs.
3. **Uniform layout with migration**: all datasets live under
   `data/datasets/<name>/…`; the French data is migrated there once
   (split pack copied byte-identical, nothing regenerated).

---

## 1. Dataset assessment

### 1.1 French Motor TPL (freMTPL2) — current dataset

| | |
|---|---|
| Rows | 678,013 policies (freq table) + 26,639 claim rows (sev table) |
| Files | `freMTPL2freq.csv` (policy features, `ClaimNb`, `Exposure`), `freMTPL2sev.csv` (`IDpol`, `ClaimAmount` per claim) |
| ID | `IDpol`; loader aggregates severity per policy (`ClaimAmount` sum, `ClaimAmountCount`) |
| Exposure | `Exposure` in policy-years (true offset/weight) |
| Features | 9 real, interpretable features (VehPower/VehAge/DrivAge/BonusMalus/Density numeric; VehBrand/VehGas/Area/Region categorical) |
| Targets | claim count + claim amount → supports **burning_cost**, **frequency**, **severity** modes; frequency×severity recipes possible |
| Preprocessing | claim amounts capped at 100,000 → `ClaimAmountCapped` |

### 1.2 AllState Claim Prediction Challenge (`train_set.csv`, 13.18M rows)

| | |
|---|---|
| Rows | 13,184,290 vehicle-years (test_set.csv is **unlabelled** — unusable here) |
| ID | `Row_ID`; grouping column `Household_ID` (~3 rows per household — a leakage unit) |
| Exposure | **none** — each row is one vehicle-year, so weight ≡ 1 |
| Target | `Claim_Amount` (bodily-injury paid, per vehicle-year). Zero-inflated: 0.81% non-zero, mean ≈ 1.38, mean|>0 ≈ 170, observed max ≈ 6.4k (no cap needed) |
| Claim count | **none** — no frequency mode; a derived `claim_occurred = (Claim_Amount > 0)` indicator enables an incidence mode and severity population selection |
| Features | 32: `Calendar_Year` (2005–07), `Model_Year`, blinded vehicle identity (`Blind_Make` 68 / `Blind_Model` ~1.2k / `Blind_Submodel` ~2.4k levels — genuinely high-cardinality), `Cat1–12` + `OrdCat` + `NVCat` categoricals, `Var1–8` + `NVVar1–4` standardized numerics |
| Missing | `?` marker; heavy in Cat7 (57%), Cat4/Cat5 (~44%), Cat2 (38%) |
| Distinct challenges vs French | scale, no exposure, household grouping, high-cardinality blinded categoricals, categorical missingness |

### 1.3 Porto Seguro Safe Driver Prediction (`train.csv`, 595k rows)

| | |
|---|---|
| Rows | 595,212 policies (test.csv unlabelled — unusable) |
| ID | `id` |
| Exposure | none — weight ≡ 1 |
| Target | `target` ∈ {0,1}: claim filed in the following year. 3.64% positive. Modelled as a rate (claim probability); Gini is the canonical metric for this data |
| Features | 57 fully anonymised: 14 `*_cat` (max cardinality 104), 17 `*_bin`, remainder continuous/ordinal; families `ps_ind_*`, `ps_reg_*`, `ps_car_*`, `ps_calc_*` (the `calc` block is famously noise — a good research-agent discovery, do **not** pre-drop it) |
| Missing | `-1` marker in 13 columns (worst: `ps_car_03_cat` 69%, `ps_car_05_cat` 45%) |
| Distinct challenges vs French | binary target, anonymised features (no domain priors), missing markers, severe class imbalance |

### 1.4 What the three datasets force the framework to generalise

| Assumption baked in today | French | AllState | Porto |
|---|---|---|---|
| Exposure column exists | ✅ `Exposure` | ❌ (unit) | ❌ (unit) |
| Claim count exists (frequency mode, freq×sev recipes) | ✅ `ClaimNb` | ❌ | ❌ |
| Target is an amount needing a 100k cap | ✅ | ❌ (uncapped) | ❌ (binary) |
| Fixed mode set {burning_cost, frequency, severity} | ✅ | partial | ❌ |
| Two-file freq/sev join loader | ✅ | ❌ single table | ❌ single table |
| `IDpol` id, row = split unit | ✅ | ❌ `Row_ID` + `Household_ID` grouping | ❌ `id` |
| Row count fits 10-min budget | ✅ 678k | ❌ 13.2M | ✅ 595k |
| Missing-value markers | none | `?` | `-1` |

---

## 2. Where the French dataset is hard-wired today (coupling audit)

The target layer is already half-abstracted: `targets.TargetSpec` carries
`source_column`, `weight_column`, `population`, label/metric keys, and the
severity mode proved the evaluation stack is weight-generic. The remaining
coupling:

| Area | Files | Coupling |
|---|---|---|
| Loading | `data/loader.py` | `load_fremtpl2` only: freq/sev discovery, `ClaimAmount` aggregation |
| Pipeline | `data/pipeline.py` | calls `load_fremtpl2`; hard-codes `Exposure` + target-column set for schema roles; capping on `ClaimAmount` |
| Preprocessing | `data/preprocessing.py` | `DEFAULT_CAPPED_COLUMN = "ClaimAmountCapped"` |
| Splits | `data/splits.py` | `_split_strata` looks for `ClaimAmount`/`Exposure`, French-specific claim bands; split unit = record |
| Target specs | `targets.py` | fixed `SPECS` dict of exactly 3 modes with French column names |
| Config | `config.py`, `configs/default.toml` | single global data dirs; `id_column = "IDpol"`; claim-cap settings global; one config for all runs |
| Model layer | `models/dispatcher.py`, `models/global_mean.py`, `models/recipe/interpreter.py` | module-level column constants (`EXPOSURE`, `CLAIM_COST`, `CLAIM_COUNT`, `CLAIM_EVENTS`, `RAW_CLAIM_COST`); `frequency_severity` recipe structure assumes `ClaimNb` |
| Feature policy | `feature_policy.py` | `NON_PREDICTIVE_COLUMNS = {Exposure, record_id, IDpol}` |
| Runner | `experiment_runner.py` | caps `RAW_CLAIM_COST` on every load; global processed/splits paths |
| Protected eval | `evaluation/validation.py` | "Exposure must be positive" check text (mechanically it reads the generic `exposure` weight column) |
| Milestone | `milestone.py` (protected) | re-applies claim capping with French column |
| Agent-facing | `controller/handoff.py`, `controller/context.py`, `controller/proposal_schema.py`, `scripts/generate_agent_contract.py`, `CLAUDE.md`/`AGENT.md` | Exposure policy text, reserved-column lists, "cap fixed at 100,000", burning-cost default, recipe legality table |
| Memory | `memory/*`, `structural_gini_threshold` | cross-run store not dataset-scoped; the 0.37 "plateau escape" threshold is French-specific |
| Reporting | `reporting/comparison.py` | "Exposure" axis labels, `£` formatting, pure-premium wording (cosmetic; partially driven by TargetSpec labels already) |

---

## 3. Design overview

One new concept — a **`DatasetSpec`**, loaded from `configs/datasets/<name>.toml`
into a frozen dataclass and threaded through `ProjectConfig` — replaces every
hard-coded column name, path, mode table, and preprocessing rule. The run
manifest pins the dataset the same way it pins the run id.

```
configs/
  default.toml                  # framework defaults (dataset-neutral)
  datasets/
    french_motor.toml           # default dataset
    allstate.toml               # ~2M-row stratified household sample
    allstate_full.toml          # 13.2M rows, enlarged budgets
    porto_seguro.toml

data/
  datasets/
    french_motor/{raw,processed,metadata,splits,holdout_vault}/
    allstate/{...}              # sample + full share raw/, differ downstream
    allstate_full/{...}
    porto_seguro/{...}
```

Selection at run time:

```bash
autoresearch --track claude --new-run bootstrap-track --dataset porto_seguro ...
```

- `--dataset` is accepted by `bootstrap-track` / `prepare-data` /
  `start-session`; it defaults to `french_motor` (from `default.toml`
  `[data] default_dataset`).
- `bootstrap-track` writes `dataset: <name>` into `run_manifest.json`. Every
  subsequent command resolves the dataset **from the run manifest** — the agent
  never needs to repeat it, and passing a mismatching `--dataset` for an
  existing run is a hard error (same philosophy as run-id pinning).
- `prepare-data --dataset <name>` builds that dataset's artifacts; bootstrap
  auto-prepares when missing, exactly as today.

### 3.1 The dataset config file

Everything dataset-specific in one TOML. Illustrated with the three concrete
files (abridged):

```toml
# configs/datasets/french_motor.toml
[dataset]
name = "french_motor"
display_name = "French Motor TPL (freMTPL2)"
default_target_mode = "burning_cost"

[source]
loader = "adapter"                                # "single_table" | "adapter"
adapter_module = "autoresearch.data.adapters.french_motor"
raw_dir = "data/datasets/french_motor/raw"        # may be absolute/external
id_column = "IDpol"

[columns]
weight = "Exposure"              # omit → framework synthesises unit weight
count = "ClaimNb"                # optional; enables frequency_severity recipes
event_count = "ClaimAmountCount" # optional
non_predictive = ["Exposure", "IDpol", "record_id"]

[preprocessing.cap]              # optional table; absent = no capping
column = "ClaimAmount"
output_column = "ClaimAmountCapped"
threshold = 100000
fixed = true                     # agents may not vary it (search-space lock)

[splits]
unit_column = "record_id"        # group-aware splitting unit
stratify_target = "ClaimAmount"  # banded via quantiles + fixed zero band
stratify_weight = "Exposure"

[[targets]]
mode = "burning_cost"
source_column = "ClaimAmountCapped"
weight = "Exposure"
population = "all"
rate_label = "pure premium"
entity_label = "claim_cost"      # drives metric/alias key generation, §3.3
default = true

[[targets]]
mode = "frequency"
source_column = "ClaimNb"
weight = "Exposure"
population = "all"
rate_label = "claim frequency"
entity_label = "claim_count"

[[targets]]
mode = "severity"
source_column = "ClaimAmountCapped"
weight = "ClaimAmountCount"
population = { positive_column = "ClaimAmountCount" }
rate_label = "severity"
entity_label = "claim_cost"

[memory]
structural_gini_threshold = 0.37

[reporting]
currency = "£"
```

```toml
# configs/datasets/allstate.toml  (the default, sampled variant)
[dataset]
name = "allstate"
display_name = "AllState Claim Prediction (2M household sample)"
default_target_mode = "pure_premium"

[source]
loader = "single_table"
raw_dir = "data/datasets/allstate/raw"
train_file_pattern = "train_set*"      # unlabelled test_set.csv is ignored
id_column = "Row_ID"
na_values = ["?"]

[source.sample]                        # optional; deterministic ingestion-time sample
rows = 2000000
unit_column = "Household_ID"           # sample whole households (no leakage)
stratify_column = "Claim_Amount"       # preserve the 0.8% claim tail
seed = 20260709

[columns]
# no weight → unit weight synthesised
non_predictive = ["Row_ID", "Household_ID", "record_id"]
derived = [{ name = "claim_occurred", expr = "Claim_Amount > 0" }]

[splits]
unit_column = "Household_ID"
stratify_target = "Claim_Amount"

[[targets]]
mode = "pure_premium"
source_column = "Claim_Amount"
weight = "unit"
population = "all"
rate_label = "pure premium"
entity_label = "claim_cost"
default = true

[[targets]]
mode = "severity"
source_column = "Claim_Amount"
weight = "unit"
population = { positive_column = "claim_occurred" }
rate_label = "severity"
entity_label = "claim_cost"

[[targets]]
mode = "claim_incidence"
source_column = "claim_occurred"
weight = "unit"
population = "all"
rate_label = "claim incidence rate"
entity_label = "claim_indicator"

[reporting]
currency = "$"
```

`allstate_full.toml` is identical minus `[source.sample]`, with
`raw_dir` pointing at the same shared raw folder and

```toml
[overrides.compute]
base_budget_minutes = 40
budget_increment_minutes = 15
preflight_sample_rows = 20000
```

```toml
# configs/datasets/porto_seguro.toml
[dataset]
name = "porto_seguro"
display_name = "Porto Seguro Safe Driver"
default_target_mode = "claim_incidence"

[source]
loader = "single_table"
raw_dir = "data/datasets/porto_seguro/raw"
train_file_pattern = "train*"
id_column = "id"
# -1 is the published missing marker; recode to NaN only in *_cat columns
# where estimator native-categorical handling benefits; leave numeric -1
# visible (its missingness is informative and models can exploit it either way)
na_marker = { value = -1, columns_matching = "_cat$" }

[columns]
non_predictive = ["id", "record_id"]

[splits]
unit_column = "record_id"
stratify_target = "target"

[[targets]]
mode = "claim_incidence"
source_column = "target"
weight = "unit"
population = "all"
rate_label = "claim probability"
entity_label = "claim_indicator"
default = true
```

**Override mechanism.** A dataset file may carry `[overrides.<section>]`
tables (`evaluation`, `compute`, `screening`, `promotion`, `resampling`,
`memory`) that are deep-merged over `default.toml` before `ProjectConfig` is
built. This is how `allstate_full` gets bigger budgets and how a future noisy
dataset could loosen `min_relative_lift` — without forking the framework
config.

### 3.2 `DatasetSpec` (new module `src/autoresearch/datasets.py`)

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    loader: str                      # "single_table" | "adapter"
    adapter_module: str | None
    raw_dir: Path
    id_column: str
    weight_column: str | None        # None → synthesised "unit_weight" (≡ 1.0)
    count_column: str | None         # None → frequency_severity structure illegal
    event_count_column: str | None
    non_predictive: frozenset[str]
    na_values: tuple[str, ...]
    na_marker: NaMarker | None
    derived_columns: tuple[DerivedColumn, ...]
    sample: SampleSpec | None
    split_unit_column: str
    stratify_target: str | None
    stratify_weight: str | None
    cap: CapSpec | None
    targets: tuple[TargetSpec, ...]  # generic TargetSpec, §3.3
    default_target_mode: str
    reporting: ReportingSpec         # currency, labels
    overrides: dict[str, dict]       # raw section overrides

def load_dataset_spec(name: str) -> DatasetSpec: ...
def list_datasets() -> list[str]:            # scans configs/datasets/*.toml
def dataset_data_dir(name: str) -> Path:     # data/datasets/<name>
```

Validation at load: exactly one default target; every referenced column name
non-empty; `frequency_severity` availability derived from `count_column`;
mode names must be lowercase snake_case (they flow into registry rows,
metric keys, and handoff text).

### 3.3 Generic `TargetSpec` (rework of `targets.py`)

Today `SPECS` is a fixed dict of three modes whose ten label/key fields are
hand-written. The dataclass **stays** (it is imported widely and is already
the right abstraction); what changes is construction:

- `TargetSpec` gains `population_column: str | None` (replaces the three
  `POPULATION_*` enum constants — `dispatcher._filter_to_population` filters
  `frame[spec.population_column] > 0` when set; the enum aliases remain as
  thin shims during transition).
- The ten derived names (`predicted_column`, `rate_actual_column`, metric
  keys, aliases…) are **generated** from `entity_label` + `rate_label` by a
  single `build_target_spec(cfg)` function:
  `predicted_{entity_label}`, `actual_{rate_slug}`, `weighted_mae_{entity_label}`,
  `mean_actual_{rate_slug}`, `total_actual_{entity_label}`, …
  Applied to the French config this reproduces today's strings **exactly**
  (`predicted_claim_cost`, `actual_pure_premium`, `weighted_mae_claim_cost`,
  `mean_actual_frequency`, …) — old registries, prediction parquets and
  reports stay readable with zero special-casing. A regression test locks
  the generated French specs to the current literal values.
- `VALID_TARGET_MODES` / `normalise_target_mode` / `target_spec()` become
  dataset-relative: `target_spec(mode, dataset=spec)`. Module-level French
  constants (`BURNING_COST` etc.) survive as plain strings for existing
  imports.
- `dispatcher._add_target_rate_columns` drops its mode-name `if/elif` and
  writes columns purely from the spec fields (it nearly does already; the
  frequency branch keys off `entity_label == "claim_count"`-style spec data,
  i.e. "does this mode predict the count entity or the cost entity" becomes
  `spec.predicted_column`/`spec.actual_alias` driven, with the companion
  column NaN-filled).

### 3.4 Loading & preparation pipeline

`data/loader.py` splits into:

- `data/loaders/single_table.py` — generic: find newest file matching
  `train_file_pattern` under `raw_dir` (CSV/parquet), apply `na_values` /
  `na_marker`, verify `id_column`, evaluate `derived` column expressions
  (restricted vocabulary: comparisons and arithmetic on existing columns via
  `pd.eval` — not arbitrary Python), return `RawDataset`.
- `data/adapters/french_motor.py` — the existing `load_fremtpl2` moved
  verbatim (freq/sev discovery + aggregation), conforming to the same
  `RawDataset` return.
- Adapter contract: `load(spec: DatasetSpec) -> RawDataset` where
  `RawDataset.provenance` is a free dict (replaces the freq/sev-specific
  `frequency_path`/`severity_path` fields; the French adapter records them in
  provenance).

`data/pipeline.py::prepare_data(config)` becomes dataset-driven:

1. Resolve loader from spec; load raw.
2. **Sample** (if `spec.sample`): deterministic — hash `unit_column` values
   with the configured seed, take whole units, stratified on banded
   `stratify_column` so the claim tail is preserved proportionally. Write
   `sample_manifest.json` (seed, unit counts, target-band composition) into
   `metadata/`.
3. Synthesise `record_id` (as today) and, when `weight_column is None`, a
   `unit_weight` column ≡ 1.0 (schema role `exposure_offset`, so it is
   automatically non-predictive).
4. Apply `spec.cap` (if any) — `apply_target_capping(frame, cap)` generalises
   `apply_claim_capping`; capping diagnostics written as today.
5. Build schema/profile with roles derived from the spec (targets =
   every `source_column`/`cap.column`/count/event columns; weight →
   `exposure_offset`; `non_predictive` respected).
6. Split pack: `generate_split_pack(frame, unit_column=spec.split_unit_column, …)`
   — grouping-aware: strata and split assignment operate on **units**
   (households for AllState) with all rows of a unit following it. The
   stratifier reads `spec.stratify_target`/`stratify_weight`; claim-band edges
   become quantile-based with a pinned zero band (French keeps its current
   fixed bands via an optional `bands` field in config so its existing split
   pack re-validates unchanged).
7. Vault write per dataset: `data/datasets/<name>/holdout_vault/` — same
   token mechanism, per-dataset files.

All artifact paths in `ProjectConfig` (`processed_dir`, `splits_dir`,
`metadata_dir`, `holdout_vault_dir`, `raw_data_dir`) resolve to
`data/datasets/<name>/…`. `load_config()` gains a `dataset` parameter; run
manifest resolution supplies it (§3.6).

### 3.5 Model layer

- **Dispatcher constants** (`EXPOSURE`, `CLAIM_COST`, `CLAIM_COUNT`,
  `CLAIM_EVENTS`, `RAW_CLAIM_COST`) stop being load-time constants. The
  dispatch path already receives `target_mode` and builds everything from the
  spec; the remaining uses (population filter, prediction assembly, id-column
  stripping) switch to `spec.*` fields plus `config.dataset`. For
  **backwards compatibility of agent scripts**, the constants remain importable
  but are set per-process at config load
  (`autoresearch.models.columns.bind(dataset_spec)`), and — more importantly —
  the handoff prints the *actual* names for the active dataset, so newly
  generated scripts are dataset-correct. The duplicated constants in
  `global_mean.py` and `recipe/interpreter.py` are deleted in favour of the
  bound module.
- **Script frames**: sanitisation (target/id stripping — `tests/test_script_frame_sanitization.py`)
  reads the strip-list from the spec (all target-role columns + ids + raw
  cap source). The "build features only from the handoff's named list" policy
  is unchanged and now automatically correct per dataset.
- **Recipes**: vocabulary unchanged. Two legality rules become dataset-aware
  in `recipe/schema.py` validation:
  - `structure = "frequency_severity"` requires `spec.count_column` (French
    only, among the three).
  - objective guidance in the handoff is generated from the active target
    (e.g. Porto incidence: poisson/tweedie/squared_error all legal on a 0/1
    rate; gamma illegal since zeros).
- **`global_mean` baseline**: already conceptually generic ("flat
  weighted rate") — rewritten to read `spec.source_column`/`spec.weight_column`
  instead of its four constants. Works untouched for all three datasets.
- **Feature policy** (`feature_policy.py`): `NON_PREDICTIVE_COLUMNS` becomes
  `spec.non_predictive ∪ {record_id, weight, ids}`; the module keeps a
  function API (`non_predictive_columns(spec)`) and the static-scan of model
  scripts (`scan_file_for_non_predictive_feature_use`) checks the active
  dataset's names.

### 3.6 Run scoping, CLI, and config plumbing

- `ProjectConfig` gains `dataset: DatasetSpec`. `load_config(..., dataset=None)`
  resolves in priority order: explicit `--dataset` flag → run manifest of the
  resolved run → `default.toml` `[data] default_dataset` (= `french_motor`).
- `bootstrap-track` stores `"dataset": name` in `run_manifest.json`;
  `_resolve_run_id`-style guard: an explicit `--dataset` that contradicts an
  existing run's manifest raises.
- `--dataset` added to: `bootstrap-track`, `prepare-data`, `start-session`
  (continuations), `init-official-champion`, and the analyst/telemetry
  commands that operate outside a run. A new `list-datasets` command prints
  registered datasets with row counts and available target modes.
- The compute-budget formula stays in `default.toml` but is overridable per
  dataset (§3.1) — this is how `allstate_full` runs at all and how
  `preflight_sample_rows` scales.
- Registry rows gain a `dataset` column (nullable; backfilled `french_motor`
  by migration). Champion history, comparisons, and telemetry all key within
  a run, which is already dataset-pinned, so no further schema change is
  needed — but the column makes cross-run queries honest.

### 3.7 Agent-facing surfaces (contract, handoff, proposals)

Principle: **the repo contract goes dataset-neutral; all dataset specifics move
into the handoff**, which is already the authoritative, dynamically generated
document the agent must read first.

- `AGENT.md` / `CLAUDE.md` (via `scripts/generate_agent_contract.py`): remove
  French specifics ("burning cost", "ClaimAmountCapped", "cap fixed at
  100,000", the Exposure policy paragraph, French column constants). Replace
  with: "the active dataset, target mode, column roles, weight policy, and any
  fixed preprocessing are printed at the top of the handoff — they are
  binding". The recipe estimator/objective table stays (it is
  dataset-neutral); the target→objective guidance paragraph becomes
  descriptive of target *shapes* (zeros present → tweedie/squared_error;
  counts → poisson; positive-only → gamma) rather than named insurance modes.
- `controller/handoff.py`: new "Active dataset" block generated from
  `DatasetSpec` + dataset schema: name, target mode + source column, weight
  column and its policy line ("`unit_weight` — every row weighs 1.0; it is
  synthesised, never a feature"), population filter, cap statement (or "no
  capping"), named feature list with roles/cardinalities (already sourced from
  `dataset_schema.json` — unchanged mechanism), and dataset-specific cautions
  generated from spec facts (e.g. AllState: "`Household_ID` is the split unit
  and is non-predictive"; Porto: "`-1` encodes missing in `*_cat`").
- `controller/proposal_schema.py`: reserved/target column lists built from the
  spec instead of literals; validation messages name the active dataset's
  columns.
- `controller/context.py`: mission text templated from spec
  (`Improve {mode} prediction on {display_name} …`); cap sentence conditional.
- `docs/OPERATING_MANUAL.md`: gains a "Datasets" chapter (registry, layout,
  how to add one); French schema section moves under it.

### 3.8 Evaluation & protected files

The metric stack is already weight-generic (the severity mode shipped without
touching `metrics.py`). Expected protected-file impact — one operator
`update-integrity-manifest` after review:

| File | Change | Size |
|---|---|---|
| `evaluation/validation.py` | reword `exposure_positive` check message to "weight" (mechanics already read the generic `exposure` prediction column) | trivial |
| `milestone.py` | replace `apply_claim_capping(..., RAW_CLAIM_COST, …)` with spec-driven `apply_target_capping(spec.cap)`; per-dataset vault path comes free via config | small |
| `data/holdout_vault.py` | none (paths injected via config already) | — |
| `evaluation/metrics.py`, `resampling.py`, `diagnostics.py`, `comparison_runner.py`, registry submodules | none expected; `diagnostics.py` label strings verified during implementation | — |

`gini_weighted` remains the default primary metric for all three datasets
(with unit weights it degenerates to the plain/normalized Gini — exactly the
Porto and AllState conventions). `asym_pricing_loss` and the Tweedie panel
remain in the metric panel; for Porto the Tweedie deviance at p=1.5 is less
meaningful but harmless as an advisory column.

### 3.9 Cross-run memory & telemetry

- The memory store roots at `~/.autoresearch/<project>/memory/…`; it becomes
  `…/memory/<dataset>/…`. Harvest/query/playbook take the dataset from config.
  Existing memory content migrates to `french_motor/`. Rationale: "lightgbm
  num_leaves 63 beats 31" is not portable across datasets, and the
  leaderboard's time-to-structural-insight is defined against a
  dataset-specific threshold.
- `structural_gini_threshold` moves into the dataset file (French keeps 0.37;
  new datasets start with a placeholder documented as "recalibrate after the
  first few runs" — for Porto the plateau is around 0.27–0.29 Gini per the
  public leaderboard; AllState around 0.55+ weighted-Gini equivalents are not
  comparable, so start `null` = leaderboard column suppressed).
- Telemetry importer/usage report: tag rows with dataset (from run manifest);
  no structural change.

### 3.10 Reporting

`reporting/comparison.py` label fixes: axis/series names use
`spec.weight_column` display name ("Exposure" vs "Policies"), currency symbol
from `reporting.currency`, and rate wording from `TargetSpec.rate_label`
(largely already threaded). Cosmetic, low-risk, done last.

---

## 4. Migration plan (one-time, scripted)

`scripts/migrate_dataset_layout.py`:

1. `git mv`/move `data/{raw,processed,metadata,splits,holdout_vault}` →
   `data/datasets/french_motor/…` (split pack **byte-identical** — verified by
   hash before/after; nothing regenerated).
2. Write `configs/datasets/french_motor.toml` (values lifted from
   `default.toml`, which loses its French-specific keys:
   `id_column`, `[preprocessing]` cap block, `[search_space.preprocessing]`
   cap lock and `allow_log1p_features` move into the dataset file).
3. Backfill `dataset = "french_motor"` into existing run manifests and the
   registry column.
4. Move cross-run memory content into the `french_motor/` subfolder.
5. New-dataset ingestion: copy (or symlink) the external raw files into
   `data/datasets/{allstate,porto_seguro}/raw/` from
   `/Users/alexhood/Documents/Insurance Datasets/…` (unlabelled test files
   deliberately not copied), then `autoresearch prepare-data --dataset <name>`
   for each. AllState raw (~5GB) is shared between `allstate` and
   `allstate_full` via a single raw dir referenced by both configs.

Compatibility guarantees:

- French runs continue with identical splits, caps, metric keys, and champion
  history (regression test: generated French `TargetSpec`s equal today's
  literals; split-pack hash unchanged post-migration).
- Untracked/legacy handoff paths and the `proposal_inbox/` fallback are
  untouched.

---

## 5. Implementation order

Phased so the suite is green after each step:

1. **`DatasetSpec` + config plumbing** — new `datasets.py`, dataset TOMLs,
   `load_config(dataset=…)`, per-dataset paths, run-manifest pinning,
   `list-datasets`. French only; behaviour identical. Migration script + run.
2. **Generic targets** — `build_target_spec`, population-column filter,
   dispatcher/global_mean/interpreter constant unbinding, feature-policy and
   sanitisation from spec. Regression-lock French key names.
3. **Generic pipeline** — single-table loader, adapter extraction, sampling,
   group-aware splits, generic capping. Prepare Porto (smallest lift: no
   sampling, no grouping) end-to-end; synthetic-data smoke test archetypes:
   (a) exposure+freq/sev, (b) single-table amount + grouping + sampling,
   (c) binary target.
4. **AllState** — sampling + household grouping + `allstate_full` overrides;
   verify budget maths with a baseline + one lightgbm run.
5. **Agent surfaces** — handoff dataset block, contract regeneration, proposal
   schema, context. Protected-file edits + `update-integrity-manifest`.
6. **Memory scoping, telemetry tag, reporting labels.**
7. **End-to-end**: one short real run per dataset (`--cycles 2`) as acceptance.

New/changed tests: dataset-spec loading/validation; French spec-equality
regression; group-aware split leakage test (no Household_ID straddles splits);
sampling determinism; unit-weight synthesis; per-dataset handoff content;
recipe legality (`frequency_severity` rejected without count column);
manifest/dataset mismatch guard.

## 6. Risks & open points

- **AllState sample representativeness**: 0.8% claim rate → a 2M-row sample
  holds ~16k claim rows; stratified sampling makes the Gini gate workable but
  holdout claim counts (~3.2k) put a noise floor on promotions — the
  per-dataset `[overrides.promotion]` hook exists if gates prove too tight or
  loose.
- **Porto Tweedie panel**: advisory metrics computed on a 0/1 target are
  statistically odd but harmless; revisit only if agents get confused by them
  (handoff can annotate "primary metric is Gini; deviance panel advisory").
- **Agent script portability**: old French scripts importing bound constants
  keep working on French runs; on other datasets the constants bind to that
  dataset's columns, which is the correct behaviour but a behavioural change
  worth a line in the contract.
- **`structural_gini_threshold` for new datasets** starts null/provisional;
  leaderboard comparability across datasets is explicitly out of scope.
