#!/usr/bin/env python3
"""TabFM context-size benchmark on Modal serverless GPU (Phase 1 spike).

Google's TabFM regression checkpoint is 6.59 GB with no hosted API, so the
forward pass runs on a rented Modal GPU (L4 24GB by default, ~$0.80/hr billed
per second — a full sweep costs well under $1 of the Starter plan's $30/mo free
credits). The local entrypoint mirrors scripts/benchmark_foundation.py (same
features, ordinal encoding, exposure-weighted context subsample, seed 42) so
results are directly comparable to the TabPFN sweep, except the target is
capped at 100k to match the real ClaimAmountCapped objective.

One-time setup (see docs/internal/tabfm_integration_plan.md):
    .venv-api/bin/pip install modal && .venv-api/bin/modal setup
    # accept the TabFM licence on huggingface.co/google/tabfm-1.0.0-pytorch, then:
    modal secret create huggingface HF_TOKEN=<hf token>

Run (from the repo root so autoresearch metrics import):
    .venv-api/bin/modal run scripts/tabfm_modal_app.py                       # default sweep
    .venv-api/bin/modal run scripts/tabfm_modal_app.py --contexts 5000 --gpu T4  # smoke test

Reference numbers (burning-cost, search split): LightGBM Tweedie champion
gini_weighted 0.369; best TabPFN (64k context) 0.344.

NEVER point this at the holdout. It reads agent_dataset_search.parquet only.
"""

from __future__ import annotations

import os
import time

import modal

app = modal.App("tabfm-inference")

# Weights cache: HF hub downloads the 6.6 GB checkpoint into this Volume once;
# every later container boots from the cache.
CACHE_DIR = "/cache/huggingface"
weights_volume = modal.Volume.from_name("tabfm-weights", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # tabfm 1.0.0 on PyPI; [pytorch] pulls torch (stays remote — never local).
    .pip_install("tabfm[pytorch]", "numpy", "huggingface_hub", "safetensors")
    .env({
        "HF_HOME": CACHE_DIR,
        # Avoid CUDA allocator fragmentation — activations vary per batch shape.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# GPU is baked into the app at build time, so select it via env var when
# launching (Cls.with_options(gpu=...) is silently ignored under `modal run`):
#     TABFM_GPU=A100 modal run scripts/tabfm_modal_app.py ...
# T4 16GB OOMs even at 5k context (measured 2026-07-05); L4 24GB is the floor.
GPU = os.environ.get("TABFM_GPU", "L4")
# Per-call ceiling. Measured: ~50-60 min per full fit+score at 20k ctx; the
# 32-member ensemble variant runs hours. Override per launch via TABFM_TIMEOUT.
TIMEOUT = int(os.environ.get("TABFM_TIMEOUT", "7200"))


@app.cls(
    image=image,
    gpu=GPU,
    volumes={CACHE_DIR: weights_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=TIMEOUT,
    scaledown_window=300,  # keep the container warm between sweep points
)
class TabFMRunner:
    @modal.enter()
    def load_model(self) -> None:
        import os

        t0 = time.perf_counter()
        from huggingface_hub import snapshot_download
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch

        # tabfm 1.0.0's load() torch.load's regression/pytorch_model.bin, but the
        # HF repo switched to safetensors on 2026-07-03 — convert once into the
        # volume-cached snapshot, then point load() at the converted file.
        base = snapshot_download(repo_id="google/tabfm-1.0.0-pytorch")
        bin_path = os.path.join(base, "regression", "pytorch_model.bin")
        if not os.path.exists(bin_path):
            import torch
            from safetensors.torch import load_file

            st_path = os.path.join(base, "regression", "model.safetensors")
            torch.save(load_file(st_path), bin_path)
        weights_volume.commit()  # persist download + conversion for future containers

        self._regressor_cls = TabFMRegressor
        # device="cuda" matters: TabFMRegressor computes on whatever device the
        # model is on (default CPU, even on a GPU container).
        self._model = tabfm_v1_0_0_pytorch.load(
            model_type="regression", checkpoint_path=bin_path, device="cuda"
        )
        self.load_s = time.perf_counter() - t0

    @modal.method()
    def fit_predict(
        self,
        X_ctx,  # np.ndarray (n_ctx, n_features) float
        y_ctx,  # np.ndarray (n_ctx,) float — capped burning-cost rate
        X_score,  # np.ndarray (n_score, n_features) float
        predict_batch: int = 25_000,
        n_estimators: int = 8,  # library default is 32 — 32 forward passes per row
        ensemble_batch: int = 1,  # members per forward pass (library default 1 = sequential)
        ensemble_mode: bool = False,  # TabFM-Ensemble: crosses + SVD + NNLS weighting
    ) -> dict:
        """Zero-shot fit on the context, batched predict over the score frame."""
        import numpy as np
        import torch

        torch.cuda.reset_peak_memory_stats()
        kwargs: dict = {"n_estimators": n_estimators, "batch_size": ensemble_batch}
        if ensemble_mode:
            # The paper's TabFM-Ensemble configuration (their TabArena winner):
            # random cross features + SVD features per member, NNLS ensemble
            # weights learned on out-of-fold context predictions.
            kwargs.update(
                n_feature_crosses="sqrt", n_svd_features="sqrt", enable_nnls=True
            )
        reg = self._regressor_cls(model=self._model, **kwargs)

        t0 = time.perf_counter()
        reg.fit(X_ctx, y_ctx)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = np.concatenate([
            np.asarray(reg.predict(X_score[s : s + predict_batch]), dtype=np.float64)
            for s in range(0, X_score.shape[0], predict_batch)
        ])
        predict_s = time.perf_counter() - t1

        return {
            "predictions": preds,
            "model_load_s": self.load_s,
            "fit_s": fit_s,
            "predict_s": predict_s,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }


@app.local_entrypoint()
def main(
    contexts: str = "10000,20000,40000,60000",
    batch: int = 25_000,
    estimators: int = 8,
    ensemble_batch: int = 1,
    ensemble_mode: bool = False,
    upsample_positive: float = 1.0,  # context sampling weight multiplier for claim rows
    score_rows: int = 0,  # 0 = score the full frame; else a uniform subsample
    full_panel: bool = False,  # dump the entire metric panel as JSON
) -> None:
    """Runs locally: loads the search split, encodes, calls the GPU per context."""
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))

    from benchmark_foundation import FEATURES, SEARCH_PARQUET, _encode
    from autoresearch.evaluation.metrics import full_metric_panel
    from autoresearch.models.recipe.foundation import subsample_context

    if not SEARCH_PARQUET.exists():
        raise SystemExit(f"Missing {SEARCH_PARQUET}; run prepare-data first.")

    runner = TabFMRunner()

    df = pd.read_parquet(SEARCH_PARQUET)
    exposure = df["Exposure"].astype(float).to_numpy()
    # Capped burning cost (the real target); rate = total / exposure.
    actual_total = np.minimum(df["ClaimAmount"].astype(float).to_numpy(), 100_000.0)
    y_rate = actual_total / np.clip(exposure, 1e-12, None)
    X = _encode(df).astype(np.float32)

    print(f"dataset: {X.shape[0]:,} rows × {X.shape[1]} cols ({', '.join(FEATURES)})")
    print(
        f"gpu={GPU} batch={batch:,} ensemble_mode={ensemble_mode} "
        f"upsample_pos={upsample_positive}x score_rows={score_rows or 'all'} "
        f"| LightGBM ref gini 0.369, TabPFN best 0.344"
    )
    header = f"{'context':>8} {'fit_s':>7} {'pred_s':>8} {'gpu_GB':>7} {'gini_w':>7} {'rank_gini':>9} {'spearman':>9} {'apl':>9} {'calib':>7}"
    print(header)

    for ctx_rows in (int(c) for c in contexts.replace(",", " ").split()):
        idx = np.arange(len(y_rate))
        if upsample_positive != 1.0:
            # Stratified context: positives get exactly upsample× their baseline
            # exposure-weighted share of rows (swapped in for dropped negatives).
            # A plain weight multiplier under-delivers (~3× for 4×) because
            # without-replacement inclusion probabilities saturate.
            is_pos = actual_total > 0
            pos_share = exposure[is_pos].sum() / exposure.sum()
            n_pos = min(round(ctx_rows * pos_share * upsample_positive), int(is_pos.sum()))
            idx_pos, _, _, _ = subsample_context(
                idx[is_pos], y_rate[is_pos], exposure[is_pos], n_pos, "exposure", seed=42
            )
            idx_neg, _, _, _ = subsample_context(
                idx[~is_pos], y_rate[~is_pos], exposure[~is_pos], ctx_rows - n_pos, "exposure", seed=42
            )
            idx_ctx = np.sort(np.concatenate([idx_pos, idx_neg]))
            y_ctx, n = y_rate[idx_ctx], len(idx_ctx)
            print(f"  context composition: {n_pos:,} positives / {n:,} rows "
                  f"({100 * n_pos / n:.1f}%; baseline {100 * pos_share:.1f}%)")
        else:
            idx_ctx, y_ctx, _, n = subsample_context(idx, y_rate, exposure, ctx_rows, "exposure", seed=42)

        if score_rows and score_rows < len(idx):
            idx_score = np.sort(np.random.default_rng(7).choice(len(idx), size=score_rows, replace=False))
        else:
            idx_score = idx
        exp_s = exposure[idx_score]
        act_s = actual_total[idx_score]

        out = runner.fit_predict.remote(
            X[idx_ctx], y_ctx, X[idx_score], predict_batch=batch,
            n_estimators=estimators, ensemble_batch=ensemble_batch,
            ensemble_mode=ensemble_mode,
        )
        pred_rate = np.clip(out["predictions"], 0.0, None)

        # Calibration factor = Σ actual / Σ predicted on an UNBIASED
        # exposure-weighted sample of scored rows — never the context itself,
        # which is biased upward when positives are upsampled.
        rng = np.random.default_rng(123)
        n_cal = min(50_000, len(idx_score))
        p_cal = exp_s / exp_s.sum()
        idx_cal = rng.choice(len(idx_score), size=n_cal, replace=False, p=p_cal)
        factor = act_s[idx_cal].sum() / max((pred_rate[idx_cal] * exp_s[idx_cal]).sum(), 1e-12)
        pred_total = pred_rate * exp_s * factor

        panel = full_metric_panel(
            pd.Series(act_s), pd.Series(pred_total), pd.Series(exp_s)
        )
        if full_panel:
            import json

            print(json.dumps(
                {k: v for k, v in panel.items() if isinstance(v, (int, float, str))},
                indent=2, default=str,
            ))
        print(
            f"{n:>8,} {out['fit_s']:>7.1f} {out['predict_s']:>8.1f} {out['peak_gpu_gb']:>7.2f} "
            f"{panel['gini_weighted']:>7.4f} {panel['rank_gini_weighted']:>9.4f} "
            f"{panel['spearman_rho']:>9.4f} {panel['asym_pricing_loss']:>9.2f} "
            f"{panel['predicted_to_actual_ratio']:>7.3f}"
        )

    print("\nDecide per docs/internal/tabfm_integration_plan.md §3 exit questions.")
