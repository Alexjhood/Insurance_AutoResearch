# TabFM Integration Plan (spike → recipe estimator)

Status: **Phase 1 (spike) COMPLETE, 2026-07-05 — see §6 verdict.** Accuracy
promising (gini 0.315 @ 20k context vs TabPFN 0.344 ceiling / champion 0.369);
single-GPU scoring is FLOPs-bound and slow (~50-60 min per fit+score at 20k),
so Phase 2 hinges on fan-out scoring across parallel Modal containers.
Decisions (2026-07-04, with operator): compute = **Modal serverless GPU**;
scope = **spike first, integrate only if competitive**; licence = OK
(non-commercial research use).

## 1. What TabFM is

[TabFM](https://github.com/google-research/tabfm) (Google Research, released
2026-07-01) is a zero-shot tabular foundation model: TabPFN-style row/column
attention feeding a 24-block causal in-context-learning transformer. Like
TabPFN it does **no gradient training** — the training rows are the prompt and
predictions come from a single forward pass. It ships a scikit-learn API
(`TabFMRegressor.fit/predict`), accepts mixed numeric/categorical frames, and
was reported to beat tuned GBMs zero-shot on TabArena (13 regression datasets,
700–150k rows).

Key facts that shape the design:

| Fact | Consequence |
|---|---|
| Regression checkpoint is **6.59 GB** (`google/tabfm-1.0.0-pytorch`, safetensors) — ~100× TabPFN | Needs a real GPU; M3 Air CPU is smoke-test only |
| **No hosted API** (unlike TabPFN's Prior Labs service); local `pip install tabfm[pytorch|jax]`, Python ≥ 3.11 | We must bring our own GPU → Modal |
| Memory scales with context rows (all training rows in the forward pass); ≤ 500 features | Reuse `subsample_context` exposure-weighted capping from `foundation.py` |
| Single squared-error-style regression head, no `sample_weight` | Same honest `squared_error`-only objective + exposure-weighted subsampling + framework calibration as TabPFN |
| Weights licence: **TabFM Non-Commercial License v1.0** (code Apache-2.0) | Fine for this research project; nothing built on TabFM weights may be used commercially. Flag in estimator description. |
| PyTorch backend imports torch | Never import it in the local research process (macOS torch↔LightGBM libomp segfault, see `foundation.py`). The Modal backend keeps torch entirely remote; the local client (`modal` package) is torch-free, so `.venv-api` stays safe. |

Baselines to beat (burning-cost, search split, from the 2026-07-03/04 TabPFN
benchmark): LightGBM Tweedie champion **gini_weighted 0.369**; best TabPFN
(64k context) **0.344**.

## 2. Compute: Modal serverless GPU

Chosen over RunPod/vast (idle-billed pods + hand-rolled HTTP server) and
Kaggle/Colab (notebook-only, can't be called from the harness).

- **Starter plan is $0/month and includes $30/month of free credits.**
  L4 (24 GB VRAM) ≈ $0.80/hr billed per second, no idle charge. The spike is
  well under $1; a full comparison (~5 fits × fit+542k-row score) is cents to
  a few tens of cents, so free credits cover dozens of comparisons per month.
- Functions are plain Python called from local code (`modal run` /
  `Function.from_name(...).remote(...)`) — the same shape as the TabPFN `api`
  backend, so the estimator integration mirrors it exactly.
- The 6.6 GB weights are cached in a Modal **Volume** (HF cache dir), so only
  the first container ever downloads them; warm-ish cold starts after that are
  container boot + weight load (~30–60 s, to be measured).
- Default GPU **L4 24 GB** (fp32 weights 6.6 GB + row/column-attention
  activations at 60k+ context need headroom; T4 16 GB may OOM at large
  contexts). Overridable, e.g. A100 if L4 OOMs at the biggest contexts.

### One-time setup (operator)

1. Modal account: sign up at modal.com (free Starter plan), then in
   `.venv-api`: `pip install modal && modal setup` (browser auth, writes
   `~/.modal.toml`).
2. Hugging Face: accept the TabFM licence on
   `huggingface.co/google/tabfm-1.0.0-pytorch` (if gated), then
   `modal secret create huggingface HF_TOKEN=<hf token>`.

## 3. Phase 1 — spike (`scripts/tabfm_modal_app.py`)

Mirror of `scripts/benchmark_foundation.py` (the TabPFN context sweep), same
features/encoding/split so numbers are directly comparable:

```bash
.venv-api/bin/modal run scripts/tabfm_modal_app.py            # 10k/20k/40k/60k sweep
.venv-api/bin/modal run scripts/tabfm_modal_app.py --contexts 5000 --gpu T4   # cheap smoke test
```

Per context size it: exposure-weighted-subsamples the search-split context
(seed 42, same as TabPFN benchmark), ships encoded numpy arrays to the remote
L4 (payloads ~20 MB, well within Modal limits), fits+scores remotely in
batches, calibrates locally (Σactual/Σpred on the context rows — the framework
formula), and prints the full metric panel (`gini_weighted`,
`rank_gini_weighted`, `asym_pricing_loss`, calibration ratio) plus fit/predict
wall-times and peak GPU memory. **Search split only — the holdout is never
touched.**

Spike exit questions:
1. `gini_weighted` vs TabPFN 0.344 and LightGBM 0.369 at matched contexts —
   and does it show the same weighting/objective gap TabPFN did (good
   unweighted spearman, weaker exposure-weighted gini)?
2. Wall-time per fit+full-score → is a comparison feasible inside
   `budget_minutes = 10 + 5×(N//5)` with ~5 refits (single fit ≲ budget/5)?
3. Peak VRAM by context size → default `max_context_rows` and GPU tier.
4. Cold-start cost with volume-cached weights → whether cold starts must be
   amortised (e.g. `scaledown_window`) for CV refits.

Note: `benchmark_foundation.py` targets uncapped `ClaimAmount`; this spike
targets `min(ClaimAmount, 100000)` to match the real `ClaimAmountCapped`
objective. The 0.344/0.369 references were computed by the framework on the
capped target, so this is the fairer comparison (the TabPFN sweep numbers are
close but not exactly apples-to-apples).

## 4. Phase 2 — recipe estimator (only if the spike is competitive)

Fill the `tabfm` estimator slot already reserved in the agent contract
(obj `['squared_error']`, enc `['one_hot','ordinal']`), next to `_TABPFN_SPEC`
in `src/autoresearch/models/recipe/foundation.py`:

- **Backends** (`params.backend` / `AUTORESEARCH_TABFM_BACKEND`):
  - `modal` (default on the Macs) — thin client:
    `modal.Function.from_name("tabfm-inference", ...)` → `.remote(...)`
    against a **deployed** app (`modal deploy scripts/tabfm_modal_app.py`),
    so research runs don't need `modal run`. Torch never enters the local
    process. Auth = `~/.modal.toml` (fail loudly with setup instructions,
    like `_authenticate_api()` does for TABPFN_TOKEN).
  - `local` — in-process `tabfm[pytorch]` for CUDA boxes / smoke tests;
    reuses `_select_device()` including its no-MPS-by-default rule and the
    import-lightgbm-first libomp guard.
- **Reuse the shared helpers as designed**: `subsample_context`
  (exposure-weighted context capping), `_BatchedRegressor` (batched scoring,
  clip ≥ 0), framework calibration downstream. `allowed_params`:
  `backend`, `max_context_rows`, `subsample_strategy`, `random_state`,
  `predict_batch_size`, `gpu` (modal-only), `device` (local-only). Defaults
  come from the spike measurements.
- **Registration**: `register_foundation_estimators()` registers `tabfm` when
  `modal` or `tabfm` is importable; stays behind the existing per-run opt-in
  (`bootstrap-track --enable-foundation-models`). Packaging: add a
  `foundation-modal = ["modal"]` extra (torch-free, safe next to
  `tabpfn-client` in `.venv-api`).
- **Tests**: mirror `tests/test_foundation_estimator.py` — stub the `modal`
  module, assert subsampling/param plumbing/notes, no network.
- **Docs**: OPERATING_MANUAL estimator table + a licence warning; the CLAUDE.md
  estimator list already includes tabfm.

Out of scope for now: the JAX backend (PyTorch checkpoint is what HF ships and
Modal images make CUDA-torch trivial), classification (frequency mode would
need the 6.5 GB classification checkpoint and a ≤10-class discretisation —
revisit only if regression impresses), and BigQuery's "coming soon" managed
TabFM.

## 5. Cost & risk summary

- Spike: < $1 of the $30/mo free credits. Ongoing experimentation: roughly
  $0.10–0.70 per comparison depending on measured speed → free credits ≈
  dozens of comparisons/month; the operator's $20 fallback budget is unlikely
  to be needed.
- Risks: (1) tabfm 1.0.0 API details may differ from the README snippets —
  the spike will surface this in one cheap run; (2) large-context OOM on L4 →
  drop context or bump GPU; (3) per-fit latency may not fit the compute
  budget → smaller contexts or a `[compute] enforce = false` config like
  `configs/frugal_thinking.toml`; (4) same exposure-weighting gap as TabPFN —
  the spike measures it before we spend integration effort.

### Spike findings so far (2026-07-05)

- tabfm 1.0.0 pip loader expects `regression/pytorch_model.bin` but the HF
  repo moved to safetensors (2026-07-03) → convert once into the volume and
  pass `checkpoint_path` (done in the spike script).
- `tabfm_v1_0_0_pytorch.load()` defaults to CPU **even on a GPU container**
  and `TabFMRegressor` computes on whatever device the model is on — always
  load with `device="cuda"`.
- `TabFMRegressor(n_estimators=…)` defaults to **32** ensemble members = 32
  forward passes per scored row; the default timed out an L4 at >1h for one
  5k-context fit + 542k-row score. Treat `n_estimators` as the main
  quality-vs-budget dial (like TabPFN's).
- **T4 16GB OOMs even at 5k context** — L4 24GB is the VRAM floor (measured).
- Modal pitfall: `Cls.with_options(gpu=…)` is silently ignored under
  `modal run` for same-file classes — select the GPU via the `TABFM_GPU` env
  var, which is read when the app decorator is built. Also: killing the local
  `modal run` client does NOT stop the remote app — use `modal app stop -y`.

## 6. Spike results & verdict (2026-07-05)

All runs: search split only, exposure-weighted context (seed 42), 4 ensemble
members, full 542k-row score, framework-style calibration on context rows.
Refs: LightGBM Tweedie champion gini_weighted **0.369**; best TabPFN (64k)
**0.344**.

| GPU | ctx | batch | stack | predict | peak VRAM | gini_w | rank_gini | spearman | calib |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 5k | 10k | 1 | 51.6 min | 10.3 GB | 0.2858 | 0.363 | 0.071 | 0.853 |
| A100 | 5k | 10k | 1 | 19.7 min | 10.3 GB | 0.2858 | 0.363 | 0.071 | 0.853 |
| A100 | 20k | 10k | 1 | 58.9 min | 14.4 GB | **0.3149** | 0.370 | 0.078 | 0.869 |
| A100 | 20k | 20k | 2 | 48.7 min | 29.5 GB | 0.3149 | 0.370 | 0.078 | 0.869 |
| L4 | 5k | 50k | 1 | >60 min (timeout) | — | — | | | |
| L4/A100 | 20k | 20-30k | 2-4 | OOM (22.6-42+ GB) | — | — | | | |
| A100 | 32k↑4× | 10k | 2 | 106 min | 30.7 GB | **0.3550** | 0.371 | 0.083 | 0.842 |
| H100 ens-8 | 32k↑4× | 16k | 2 | 21 min (100k rows) | 44.1 GB | 0.3636±.01 | 0.371 | 0.085 | 0.905 |
| H100 ens-8 | 32k↑4× | 16k | 2 | 81 min (full frame) | 44.1 GB | **0.3571** | 0.372 | 0.082 | 0.843 |

Full-frame ensemble rerun (operator-requested fair comparison): ensemble's
true gain over base is **+0.002 gini** (0.3550 → 0.3571; the 100k subsample
had flattered it by +0.0065) — confirmed not worth 2× compute. Panel details:
decile_lift_monotonicity 1.0, double_lift_slope 1.79 (predictions
under-spread ~1.8× — the main gap vs the GBM), apl under/over ratio 1.20,
predicted/actual 0.843 even after unbiased recalibration (spike's 50k-sample
calibration doesn't fully transfer; framework train-calibration owns this in
Phase 2). Spike total spend ≈ $28 of $30 free credits.

**Second-round findings (2026-07-05 PM, operator-directed):** (a) stratified
**4× positive upsampling** of the context (claim rows swapped in for
negatives: 19.3% vs 4.8% baseline; recalibrate on an UNBIASED exposure-weighted
sample, never the context) plus 32k context is worth **+0.040 gini** over
plain 20k — the biggest single lever found; (b) **TabFM-Ensemble**
(`n_feature_crosses="sqrt", n_svd_features="sqrt", enable_nnls=True`, the
TabArena-winning config) with 8 members adds only ~+0.009 (within 100k-row
subsample noise) — not clearly worth 2× compute vs more context/upsampling;
(c) H100 VRAM envelope: 4-stack @ 64k rows/forward OOMs even 80 GB (~97 GB
asked, 22 GB single spikes); 2-stack @ 48k rows/forward = 44 GB works;
(d) best TabFM now **0.355 (full frame) / 0.364 (subsample)** vs champion
0.369 — statistically at the champion's level, untuned.

Findings:

1. **Accuracy is real and scales with context**: 0.286 @ 5k → 0.315 @ 20k,
   well above TabPFN at comparable contexts (TabPFN ~0.20 @ 1k, 0.344 @ 64k).
   Fully deterministic across GPUs/batchings (identical panels, seed 42).
   Calibration ratio ~0.85-0.87 — framework calibration handles it.
2. **Scoring is FLOPs-bound; batching does not rescue it.** Row attention runs
   over (context+batch) jointly, so cost/forward is ~quadratic in rows: the
   50k outer batch was *slower* than 10k (timeout), and 2-member stacking +
   2× batch bought only 17% (58.9 → 48.7 min) for 2× VRAM. Optimal outer
   batch ≈ context size. A100 = 2.6× L4 speed at 2.6× price (cost/fit ≈
   $0.70 either way).
3. **VRAM envelope**: T4 16 GB unusable (OOM @ 5k ctx). L4 24 GB fits ctx 20k
   only with sequential members and ≤ ~10-20k batch. Member-stacking at 20k
   ctx needs ~27 GB (A100). 4-stack @ 20k needs > 42 GB (nothing we tried).
4. **Cold starts are solved**: volume-cached weights load in ~1 s fetch +
   model init; fit itself is ~0.1 s (all cost is predict).
5. Spike spend: ≈ $8-10 of the $30/mo free credits (incl. OOM probes and two
   timed-out hours).

**Verdict — integrate only with fan-out scoring.** Single-container scoring
(≈ 50-60 min per fit+score at 20k ctx; ~5 refits per comparison) cannot fit
the compute budget and is unpleasant even budget-exempt. But score batches are
independent: Phase 2 should split the score frame across N parallel containers
(Modal `.map()` / `.starmap()`, weights already cached) — wall-clock ÷ N at
identical GPU-seconds cost. 6-8 × L4 at 20k ctx ⇒ ~8-10 min/fit+score,
borderline budget-viable; per-comparison cost ~$3-4. Before building Phase 2,
run one decisive accuracy point: 40-60k context with fan-out scoring, to see
whether TabFM actually threatens 0.344/0.369 — if it plateaus below TabPFN,
integration is not worth it.

Sources: [HF model card](https://huggingface.co/google/tabfm-1.0.0-pytorch) ·
[GitHub](https://github.com/google-research/tabfm) ·
[Google Research blog](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/) ·
[Modal pricing](https://modal.com/pricing)
