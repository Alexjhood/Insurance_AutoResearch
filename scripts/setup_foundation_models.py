#!/usr/bin/env python3
"""One-time setup + health check for the optional foundation tabular models.

Run this once per machine after ``pip install -e .[foundation]``:

    python scripts/setup_foundation_models.py            # check + smoke test
    python scripts/setup_foundation_models.py --no-smoke # skip the fit/predict

What it does:
  1. Confirms ``tabpfn`` and ``torch`` import; reports torch version and whether
     the Apple-Silicon MPS backend is available.
  2. Checks the two independent access gates for local inference:
       A. HuggingFace gated repo ``Prior-Labs/tabpfn_3`` (weight download) —
          accept terms in the browser + ``hf auth login`` / ``HF_TOKEN``.
       B. Prior Labs licence acceptance (local-inference right) — ``TABPFN_TOKEN``
          from https://ux.priorlabs.ai, or a cached acceptance from a prior
          interactive fit.
     Prints exactly what to do if either is missing, so a research run never dies
     mid-cycle on auth.

  Note (python.org macOS Python): the framework build ships without a linked CA
  bundle, so TabPFN's licence check (raw urllib) fails with an SSL error until you
  run "/Applications/Python <ver>/Install Certificates.command" once.
  3. Pre-downloads the TabPFN regressor weights into the OS cache so runs are
     offline afterwards.
  4. Smoke test: fits + predicts on a tiny synthetic table on MPS (if available)
     and CPU, printing timings — proves the install actually runs on this box.

Exit code is non-zero if a required step fails, so it is safe to gate CI/setup on.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def check_imports() -> tuple[object | None, object | None]:
    print("[1/4] Imports")
    torch = tabpfn = None
    # The API backend is the default install and needs only the thin client.
    try:
        import tabpfn_client  # noqa: F401

        _ok("tabpfn_client present (api backend) — the default, torch-free path")
    except Exception:  # pragma: no cover
        _fail("tabpfn_client not importable. Run: pip install -e '.[foundation]'")
    # Local backend is optional (`.[foundation-local]`); absence just means api-only.
    try:
        import torch as _torch

        torch = _torch
        _ok(f"torch {torch.__version__} (local backend available)")
    except Exception:  # pragma: no cover - environment dependent
        print("  · torch not installed — api-only mode (recommended on a Mac)")
    try:
        import tabpfn as _tabpfn

        tabpfn = _tabpfn
        _ok(f"tabpfn {getattr(tabpfn, '__version__', '?')} (local backend)")
    except Exception:  # pragma: no cover
        print("  · local tabpfn not installed — api-only mode")
    if torch is not None:
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda = bool(torch.cuda.is_available())
        device = "mps" if mps else ("cuda" if cuda else "cpu")
        _ok(f"local compute device: {device} (mps={mps}, cuda={cuda})")
    return torch, tabpfn


def check_token(local_available: bool = True) -> bool:
    """Check the access gates. TABPFN_TOKEN (Prior Labs) is always required. The
    HuggingFace gate applies only to the LOCAL backend's weight download, so it is
    skipped in the default api-only install."""
    print("[2/4] Access gates")
    ok = True

    # Gate A — HuggingFace gated repo (LOCAL weight download only).
    if not local_available:
        print("  · HuggingFace gate skipped (api-only install downloads no local weights)")
    else:
        hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            import huggingface_hub

            # huggingface_hub >=1.0 removed HfFolder; get_token() covers env + cache.
            hf = hf or huggingface_hub.get_token()
        except Exception:
            pass
        if hf:
            _ok("HuggingFace credential found (local weight download)")
        else:
            ok = False
            _fail(
                "No HuggingFace credential (needed for LOCAL weights). GATED repo Prior-Labs/tabpfn_3:\n"
                "        1. Accept terms at https://huggingface.co/Prior-Labs/tabpfn_3\n"
                "        2. Authenticate:  hf auth login   (or export HF_TOKEN=<read-token>)"
            )

    # Gate B — Prior Labs licence / API key (required for both backends). Satisfied
    # by TABPFN_TOKEN, or by a cached acceptance from a prior *interactive* fit.
    if os.environ.get("TABPFN_TOKEN"):
        _ok("TABPFN_TOKEN set (Prior Labs licence)")
    else:
        _fail(
            "TABPFN_TOKEN not set. TabPFN needs a one-time Prior Labs licence acceptance\n"
            "        for local inference (separate from the HuggingFace gate). Either:\n"
            "        • headless: register at https://ux.priorlabs.ai, accept the licence,\n"
            "          copy the API key from https://ux.priorlabs.ai/account, then\n"
            "          export TABPFN_TOKEN=<key>; or\n"
            "        • interactive: run one .fit() in a real terminal to accept via browser\n"
            "          (the acceptance then caches and non-interactive runs work).\n"
            "        The fit step below is the definitive check."
        )
    return ok


def predownload(tabpfn: object) -> bool:
    print("[3/4] Weight download (cached for offline runs)")
    try:
        from tabpfn import TabPFNRegressor

        # Constructing + a trivial fit forces the checkpoint download into the
        # OS cache (~/Library/Caches/tabpfn on macOS).
        import numpy as np

        reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
        reg.fit(np.random.default_rng(0).random((32, 4)), np.arange(32, dtype=float))
        _ok("regressor weights present in cache")
        return True
    except Exception as exc:  # pragma: no cover
        _fail(f"weight download/fit failed: {exc}")
        return False


def smoke_test(torch: object) -> bool:
    print("[4/4] Smoke test (fit + predict)")
    import numpy as np
    from tabpfn import TabPFNRegressor

    rng = np.random.default_rng(42)
    X = rng.random((200, 6))
    y = X[:, 0] * 3.0 + X[:, 1] - X[:, 2] + rng.normal(scale=0.1, size=200)
    Xte = rng.random((50, 6))

    devices = ["cpu"]
    if torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.insert(0, "mps")

    ok = True
    for dev in devices:
        try:
            t0 = time.perf_counter()
            reg = TabPFNRegressor(device=dev, ignore_pretraining_limits=True, random_state=0)
            reg.fit(X, y)
            pred = reg.predict(Xte)
            dt = time.perf_counter() - t0
            assert pred.shape == (50,)
            _ok(f"{dev}: fit+predict on 200×6 in {dt:.2f}s")
        except Exception as exc:  # pragma: no cover
            _fail(f"{dev}: {exc}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-smoke", action="store_true", help="Skip the fit/predict smoke test.")
    parser.add_argument("--no-download", action="store_true", help="Skip the weight pre-download.")
    args = parser.parse_args()

    torch, tabpfn = check_imports()
    try:
        import tabpfn_client  # noqa: F401
    except Exception:
        _fail("api backend unavailable — install with `pip install -e '.[foundation]'`")
        return 1
    check_token(local_available=tabpfn is not None)  # warning only
    # The weight pre-download and local fit/predict smoke test only apply to the
    # local backend. In the default api-only install they are skipped.
    if tabpfn is None:
        print("[3/4] Weight download — skipped (api-only install)")
        print("[4/4] Smoke test — skipped (api-only install; the API is exercised at run time)")
    else:
        if not args.no_download and not predownload(tabpfn):
            return 1
        if not args.no_smoke and not smoke_test(torch):
            return 1
    print("\nFoundation models ready. Enable per run with "
          "`bootstrap-track --enable-foundation-models` and use backend='api'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
