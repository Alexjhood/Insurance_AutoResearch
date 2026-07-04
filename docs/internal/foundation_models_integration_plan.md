# Foundation tabular models in auto-research: TabPFN-3 now, TabFM spike

**Status:** design approved 2026-07-03 (TabPFN first / recipe estimator / optional
extra in main env / defaults tuned for M2 Pro 32GB). **TabPFN integration built
and merged into the recipe framework on 2026-07-03** (see "Implementation as
built" below). TabFM spike still outstanding.

## ⚠️ Benchmark verdict (2026-07-03, M3 Air 16GB): TabPFN is GPU-bound, impractical here

Full auth chain cleared and the integration runs correctly end to end, **but
TabPFN is not viable on Apple Silicon for this dataset:**

- **Real score frame = 108,483 rows** (`search_validation`); a comparison refits
  ~5× (1 fit + 4 CV folds), each scoring ~90-108k rows.
- **MPS backend is unusable:** crashes at 15k context (`MPSNDArray` Metal
  assertion, aborts the process) and, even when it runs, predicts at **~85s per
  1000 rows** — it stalls on GPU dispatch/sync (low CPU *and* low GPU
  utilisation), not compute. Extrapolates to ~2.5h for one full predict.
- **CPU backend is stable but far too slow:** measured `predict` per 1000 rows —
  context 2000 → **10.6s**, 5000 → 20.7s, 8000 → 39.0s (fit is ~1s; predict
  dominates and scales ~linearly with context). At the *smallest* context (2000)
  one full 108k-row predict ≈ **19 min**; a full comparison ≈ **~95 min**.
  Budget is 10 min.
- **Not a memory problem** → the 32GB M2 Pro won't fix it (same Metal stack;
  only ~1.3× more CPU cores). TabPFN is designed for CUDA GPU inference; its
  own docs say CPU is for ≲1000-sample data.

**Consequences applied to the code:** `_select_device()` no longer auto-selects
MPS on Apple Silicon (it crashed) — CPU is the stable default; MPS only via
`AUTORESEARCH_TABPFN_DEVICE=mps`.

**Decision (2026-07-03): API backend chosen and built.** `tabpfn-client` offloads
inference to Prior Labs' GPU; MTPL data is public so egress is low-risk.

### API backend as built

- `foundation.py` gains a **backend switch** on the `tabpfn` estimator:
  `params.backend` (`local`/`api`) or `AUTORESEARCH_TABPFN_BACKEND` (default
  `local`). `api` imports `tabpfn_client`, authenticates non-interactively via
  `set_access_token(TABPFN_TOKEN)` (raises a clear error if unset), and reuses the
  same subsample/batch/clip/calibration path. API defaults: context 20000,
  predict batch 30000. Registration now fires if **either** `tabpfn` or
  `tabpfn_client` is importable.
- `_select_device()` no longer auto-picks MPS (local only; CPU default).
- `tabpfn-client>=0.1` added to the `[foundation]` extra; setup script checks it;
  docs (manual + contract note) cover the backend + credit budget.
- Tests: stubbed `tabpfn_client` fixture covers the api path (auth, dispatch,
  missing-token error, unknown-backend error) with no real credits.

### Verified against the real API (2026-07-03, M3 Air)

- Auth with the account's `TABPFN_TOKEN` works; account quota **50,000,000
  credits/day** (resets 00:00 UTC).
- **Credit cost model** (measured): ~**2.8-3.9 credits per scored row**, all-in
  (context overhead ~20k credits at ctx 5000 is negligible). The framework scores
  the **whole frame** each fit (train, for calibration, + search-validation), and
  a comparison refits ~5× (≤13× on escalation).
- **End-to-end through `dispatch_model`** (backend=api, ctx 5000, ~60k-row frame):
  16.8s, 235,128 credits, predictions finite/non-negative/calibrated.
- **Budget implication:** a real full-scale comparison (542k-row frame × ~5
  folds) ≈ **2-10M credits**, i.e. **~5-20 full comparisons/day** on the 50M
  quota; latency ~1-2 min/comparison — comfortably within the 10-min budget.
  Operator should watch <https://ux.priorlabs.ai/account/usage>.

The estimator/gating/subsample/batch/calibration code is complete and correct,
verified on real data through both backends. Local remains available for a CUDA
box; the API is the practical path on Apple Silicon.

## Pilot-run post-mortem + OpenMP root cause (2026-07-03)

A Codex pilot run (`claude/runs/20260703T144352Z`, frugal gate, `backend=api`,
`n_estimators=1`) completed both GBM baselines but **every TabPFN experiment
segfaulted (exit 139)** — API charges accrued but no experiment was ever
registered. Diagnosis (Codex + verification here):

- The crash is a **macOS dual-OpenMP-runtime clash**, in the LightGBM
  interpretation surrogate (`interpretation.py::_fit_surrogate`) that runs after
  every experiment — the crash report showed LightGBM `ConstructFromSampleData`
  under OpenMP `__kmp_*` frames.
- **`tabpfn_client` imports torch** (a guarded type-hint import that fires only
  because torch is installed); torch registers a second `libomp`. Probed matrix:
  `lgbm_only` OK, **`torch_then_lgbm` SEGV**, `lgbm_then_torch` OK,
  `client_then_lgbm` SEGV. So it is purely **import order** — whichever OpenMP
  runtime loads first wins; LightGBM tolerates first-but-not-second.
- `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` did **not** help. A clean
  venv did **not** help either while torch was installed (contradicting an early
  Codex guess) — the conflict is intrinsic to torch+LightGBM co-residence.

**Resolution — torch-free API install (default).** `tabpfn-client` declares **no
torch dependency**; standard `.predict()` needs no torch. A venv with the client
but **without torch/local-tabpfn** was verified: client imports without loading
torch, LightGBM co-load does not segfault, and a real API predict works. So:

- `pyproject.toml`: default `foundation = ["tabpfn-client"]` (torch-free, the
  crash cannot occur); `foundation-local = ["tabpfn","torch","tabpfn-client"]`
  for a CUDA box only. `lightgbm`/`xgboost` promoted to declared base deps (they
  were required but undeclared — which had muddied the clean-venv diagnosis).
- `foundation.py`: local backend preloads `lightgbm` before importing the local
  `tabpfn` (orders the OpenMP init) so the local path also survives on macOS; the
  API path is unchanged (never imports torch).
- Proven torch-free run env at `.venv-api`; `_select_device()` already never
  auto-selects MPS. Also surfaced a harness lesson: treat exit 139/134 or a
  missing `awaiting_decision`+registry row as an infrastructure failure — stop,
  don't retry/advance (each crash still spends the full API scoring cost).

## Implementation as built (2026-07-03)

- `src/autoresearch/models/recipe/foundation.py` — `tabpfn` recipe estimator:
  exposure-weighted context subsampler (`subsample_context`), device pick
  (`_select_device`: MPS→CUDA→CPU, `AUTORESEARCH_TABPFN_DEVICE` override),
  batched+clipped predict wrapper (`_BatchedRegressor`), guarded registration
  (`register_foundation_estimators`). Objective `squared_error` only; encodings
  `ordinal`/`one_hot` (default `ordinal`); no early stopping. Curated params:
  `max_context_rows` (default 40000), `subsample_strategy`, `random_state`,
  `n_estimators`, `device`, `predict_batch_size`.
- `recipe/__init__.py` — `enable_foundation_models()` (idempotent, import-order
  independent) + env auto-enable in `_bootstrap`.
- Gating: `bootstrap-track --enable-foundation-models` → `run_manifest.json`
  `foundation_models: true`; `cli.main` calls `apply_foundation_models_gate`
  (in `bootstrap.py`) for every command, which sets
  `AUTORESEARCH_FOUNDATION_MODELS=1` and registers into the live registry.
  In-process only (no subprocess in the fit path — verified), so the live-registry
  call is sufficient; env is belt-and-braces.
- Contract: `scripts/generate_agent_contract.py` renders a foundation note only
  when `tabpfn` is in the live menu. Handoff estimator menu is already dynamic.
- `pyproject.toml` `[foundation]` extra (`tabpfn>=2.5`, `torch>=2.6`).
- `scripts/setup_foundation_models.py` (token + weights + smoke) and
  `scripts/benchmark_foundation.py` (context-size sweep on the real split).
- `tests/test_foundation_estimator.py` — 10 pass + 1 skip (real-package smoke via
  `importorskip`); uses a stub `tabpfn` for the wiring paths. Full suite: 399
  pass, 1 skip.
- **Dependency resolution verified on the M3 Air (Python 3.13 / arm64):** pip
  resolves `torch 2.12.1` + `tabpfn 8.0.8` (the TabPFN-3 package line) with
  wheels for cp313, and pulls Apple `mlx`/`mlx-metal` for Metal acceleration.
  Real setup on the M3 Air (2026-07-03) surfaced a **three-layer auth/env chain**
  for local inference — all now understood and documented:
  1. **SSL:** python.org framework Python ships without a linked CA bundle, so
     TabPFN's licence check (raw `urllib`) fails with
     `CERTIFICATE_VERIFY_FAILED`. Fix once: run
     `/Applications/Python 3.13/Install Certificates.command` (symlinks
     `.../etc/openssl/cert.pem` → certifi). Verified fixed.
  2. **HuggingFace gated repo** `Prior-Labs/tabpfn_3` (weight download): accept
     terms + `hf auth login`/`HF_TOKEN`. Verified: `list_repo_files` and the
     model-card API both return 200 with the cached token.
  3. **Prior Labs licence** (`TabPFNLicenseError`, separate from HF): local
     inference needs a one-time Prior Labs licence acceptance —
     `TABPFN_TOKEN` from ux.priorlabs.ai, or an interactive-terminal browser
     acceptance that caches. Our runs are non-interactive, so `TABPFN_TOKEN` is
     the robust route. **This is the one gate still outstanding on the Air.**
  Everything up to layer 3 is verified against the real package (import, MPS
  device pick, registration, recipe validation, constructor kwargs, weight
  download reaching the licence check).

**Note on package version:** the pip package is versioned `8.x` while the *model*
is "TabPFN-3"; `8.0.8` defaults to TabPFN-3. The `>=2.5` floor is permissive; bump
it if a specific TabPFN-3 package version must be pinned.

---

### Original design (retained for reference)

## 1. What the research found

### TabPFN-3 (Prior Labs)

- **What it is:** transformer foundation model for tabular classification and
  regression via in-context learning — no gradient training; `fit()` just
  stores the context, `predict()` is one forward pass. Sklearn-compatible
  (`TabPFNRegressor`).
- **Install:** `pip install tabpfn` (Python ≥3.10; pulls PyTorch). TabPFN-3 is
  the default model in the current package.
- **Auth:** first use requires accepting the license via a Prior Labs account.
  Headless: accept once at https://ux.priorlabs.ai and set `TABPFN_TOKEN`.
  Weights cache at `~/Library/Caches/tabpfn/` (macOS); offline after download.
- **Apple Silicon:** explicitly supported. MPS backend works; flash attention
  on MPS (big memory reduction) needs PyTorch ≥2.13 (nightly as of writing —
  stable torch works, just uses more memory). `TABPFN_MPS_MEMORY_FRACTION`
  (default 0.7) caps MPS memory to avoid macOS crashes.
- **Scale:** advertised up to 1M rows × 200 features **on GPU**. On CPU only
  ~1k rows is practical; MPS sits in between. Regressor memory was cut ~60% in
  v3. `ignore_pretraining_limits=True` lifts soft caps.
- **License:** weights allow research and internal evaluation (incl.
  benchmarking on proprietary data); commercial/production use requires a paid
  enterprise license from Prior Labs.

### TabFM (Google Research)

- **What it is:** zero-shot tabular foundation model (hybrid row/column
  attention + in-context learning), classification and regression. Announced
  **2026-06-30** — three days old.
- **Install:** `git clone https://github.com/google-research/tabfm` then
  `pip install -e .[jax]` or `.[pytorch]` (Python ≥3.11; JAX 0.10.1/Flax
  0.12.7 or Torch 2.12.1). Weights auto-download from
  `google/tabfm-1.0.0-pytorch` on HF. Sklearn-style `TabFMRegressor`.
- **Apple Silicon:** undocumented. JAX backend is CPU-only on macOS; whether
  the PyTorch backend runs on MPS is untested. **This is the main risk** — it
  may be CPU-only and too slow for the cycle budget.
- **License:** code Apache 2.0; weights TabFM Non-Commercial License v1.0.

### Constraints from our side

- Search split = **542,412 rows × 11 predictors** (French MTPL). Neither model
  ingests 542k context rows on a 16–32GB Mac within the per-experiment budget
  (10 min early, challenger refit ~5× per comparison, escalation up to ~13×).
  → the wrapper must **subsample the training context** and **batch scoring**.
- Neither sklearn interface accepts `sample_weight`, but every recipe stage is
  fit on *rate* with exposure weights. → handle exposure by **weighted
  subsampling** (draw context rows with probability ∝ exposure); the
  framework's downstream calibration (Σactual/Σpred) fixes the aggregate level.
- Both weight licenses are research/non-commercial — fine for this project;
  revisit before any commercial use.

## 2. Integration design (decided)

### Dependencies — optional extra in the main env

`pyproject.toml`:

```toml
[project.optional-dependencies]
foundation = ["tabpfn>=2.5", "torch>=2.6"]
```

(TabFM is *not* added here — its spike runs in a scratch venv; see §5.)

### One-time setup command

`scripts/setup_foundation_models.py` (and a `docs/CLI.md` entry):

1. Verify `import tabpfn`, report torch version + `mps.is_available()`.
2. Check/obtain the license token (`TABPFN_TOKEN` or cached login).
3. Pre-download the regressor checkpoint(s) into the OS cache so runs are
   offline.
4. Smoke test: fit/predict on 1k synthetic rows, on MPS and CPU, print
   timings.

### Recipe estimator (the only agent-facing surface)

New module `src/autoresearch/models/recipe/foundation.py`, one
`register_estimator` call — no LLM-written code at run time:

```python
EstimatorSpec(
    name="tabpfn",
    objectives=frozenset({"squared_error"}),      # TabPFN is objective-agnostic;
                                                  # advertise the honest one
    encodings=frozenset({"native_categorical", "ordinal"}),
    default_encoding="native_categorical",        # TabPFN preprocesses cats itself
    fit=_fit_tabpfn,
    supports_early_stopping=False,
    native_categorical=True,
    allowed_params=frozenset({
        "max_context_rows", "n_estimators", "subsample_strategy", "random_state",
    }),
    description="TabPFN-3 foundation model (in-context learning; no training). "
                "Context subsampled to max_context_rows (exposure-weighted).",
)
```

`_fit_tabpfn(ctx)` behaviour:

- **Context subsample:** if `len(X_train) > max_context_rows` (default
  **40_000**, tuned for the M2 Pro 32GB), draw rows without replacement with
  probability ∝ `w_train` (exposure), seeded by `random_state` (default from
  ctx params, deterministic). Strategy `"uniform"` available as an override.
  Record `{"context_rows": n, "subsample_strategy": ...}` in the notes dict.
- **Device:** `mps` if available else `cpu`; overridable via
  `AUTORESEARCH_TABPFN_DEVICE`. Set `ignore_pretraining_limits=True`.
- **Predictor wrapper:** `predict(X)` runs in **batches** (default 50k rows)
  and `np.clip(pred, 0, None)` — the estimator contract requires non-negative
  rates and TabPFN's regression mean can dip below zero.
- No exposure/calibration code here — the framework stage handles rate→total
  and calibration exactly as for lightgbm.

This automatically works inside **`frequency_severity`** too — the severity
stage (~25k claim rows) fits TabPFN's comfort zone *without* subsampling,
which is arguably its best use here.

### Opt-in gating

Registration is conditional — the estimator appears in `menu()` (and therefore
in the generated agent contract and every handoff) **only when**:

1. `tabpfn` is importable (the extra is installed), **and**
2. the run enables it: `bootstrap-track --enable-foundation-models` writes
   `foundation_models: true` into the run manifest/config; sessions for runs
   without the flag never register it. (Env override
   `AUTORESEARCH_FOUNDATION_MODELS=1` for untracked/dev use.)

`scripts/generate_agent_contract.py` gains a short conditional paragraph
(driven by the same registry menu) covering: no early stopping, context is
subsampled, keep `max_context_rows ≤ ~60k`, single fit ≈ minutes not seconds.

### Tests (suite must stay green without the extra)

`tests/test_foundation_estimator.py`:

- `pytest.importorskip("tabpfn")` for anything touching the real model; one
  tiny (500-row synthetic) end-to-end fit/predict marked slow.
- Without the package / without the flag: registry does **not** list
  `tabpfn`; recipe naming it fails with the standard "unknown estimator" error.
- Unit tests for the subsampler (exposure weighting, determinism, no-op below
  cap) and the clipping/batching wrapper using a stub model — these run
  everywhere.

### Benchmark script → pick real defaults

`scripts/benchmark_foundation.py`: grid over context sizes (10k/20k/40k/60k)
× device, timing one fit + a 542k-row batched predict on the actual dataset,
printing peak memory. Run it on the **M2 Pro** before the first real run and
adjust `max_context_rows`/batch defaults; run on the Air to record the safe
override (`max_context_rows≈15_000` expected).

## 3. Build steps (order of work)

1. **Env prep** — add the `foundation` extra; `pip install -e .[foundation]`
   on the M2 Pro; run token setup + weight download; commit
   `setup_foundation_models.py`.
2. **Benchmark** — `benchmark_foundation.py` on M2 Pro (and Air); freeze
   default `max_context_rows` and predict batch size from the numbers.
3. **Estimator** — `foundation.py` (subsampler, device pick, batching, clip),
   conditional registration, `--enable-foundation-models` flag through
   bootstrap → manifest → session, contract paragraph.
4. **Tests + docs** — test file as above; OPERATING_MANUAL section (setup,
   flag, budget guidance, license note); CLI.md.
5. **Pilot run** — 3–5-cycle run with the flag on the M2 Pro; confirm the
   agent proposes `{"estimator": "tabpfn"}` recipes, cycles stay in budget,
   and CV escalation (~13× fits) doesn't blow the wall clock. Tune defaults.
6. **TabFM spike** (separate, after or parallel) — see §5.

## 4. Expected performance envelope

- TabPFN "fit" is trivial; cost is in `predict` over 542k rows, scaling with
  context × query. With 40k context on MPS expect single fit+score in the
  low minutes; comparison = ~5× that (escalation up to ~13×, outside the
  budget alarm). If benchmarks disagree, drop the default context to 20k
  before touching anything else.
- M3 Air 16GB: workable for dev/smoke with small contexts; real runs on the
  M2 Pro. `TABPFN_MPS_MEMORY_FRACTION=0.7` default retained.

## 5. TabFM spike (gate before any integration)

In a scratch venv (not the project env, deps are pinned and heavy):

1. Clone repo; try `.[pytorch]` first (torch 2.12; check Python 3.13 wheel —
   else use 3.11/3.12 in the venv).
2. Test whether the PyTorch backend accepts `device="mps"`; else CPU.
3. Time `TabFMRegressor` fit+predict at 5k/20k context vs 100k query rows on
   the M2 Pro; check output sanity on a rate target with many zeros.
4. **Gate:** integrate only if a 20k-context fit + full-split batched predict
   fits ~1/5 of the cycle budget. If yes, it becomes a second
  `register_estimator` call reusing the same subsample/batch/clip helpers
  (`name="tabfm"`); if no, park it and note the result in the manual.

## 6. Risks / notes

- **License:** both weight sets are research/non-commercial. Fine today;
  blocking for any commercial deployment (Prior Labs sells an enterprise
  license; Google's is non-commercial only).
- **Token dependency:** TabPFN needs the Prior Labs token once per machine;
  the setup script must fail loudly with instructions if absent so a run
  never dies mid-cycle on auth.
- **No sample weights:** exposure enters only via weighted subsampling +
  framework calibration. Note this in the estimator description so the agent
  can reason about it (e.g. it slightly underweights high-exposure rows vs a
  true weighted fit).
- **Torch nightly for MPS flash attention:** optional optimisation; do not
  make it a requirement. Revisit when torch 2.13 goes stable.
- **Determinism:** subsample is seeded, but TabPFN on MPS may have minor
  nondeterminism; comparisons already tolerate this via CV/bootstrap gates.

## Sources

- https://huggingface.co/Prior-Labs/tabpfn_3
- https://github.com/PriorLabs/tabpfn
- https://docs.priorlabs.ai/changelog/tabpfn-3
- https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/
- https://github.com/google-research/tabfm
- https://huggingface.co/google/tabfm-1.0.0-pytorch
