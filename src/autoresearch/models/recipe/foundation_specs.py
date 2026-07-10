"""Static metadata for foundation tabular estimators (dependency-free).

This module is the **single, environment-independent declaration** of every
foundation estimator the framework knows about — its name, its legal
objective × encoding combinations, and the caveat the agent contract prints for
it. It imports **no** optional dependency (no ``tabpfn``/``tabpfn_client``, no
``numpy``): importing it is safe on any machine, whether or not the
``[foundation]`` extra is installed.

Why this exists — the "tabpfn drift" fix:

The agent contract used to derive its estimator table from the *live* recipe
registry, which only lists ``tabpfn`` when the ``[foundation]`` extra happens to
be importable. Different machines therefore regenerated the contract differently
and the pytest gate flip-flopped the working tree between the two. The generator
now reads foundation *metadata* from here (always the same everywhere) and only
the *implementation* registration (``foundation.register_foundation_estimators``)
keeps requiring the packages. Metadata is static; execution stays gated.

Anything that needs "what foundation estimators exist and what shapes are legal"
— the contract generator, recipe validation's specific error messages — reads
:data:`FOUNDATION_ESTIMATOR_SPECS` from here. The heavy
``foundation.py`` module re-exports these and builds its runtime
:class:`~autoresearch.models.recipe.registry.EstimatorSpec` objects from them, so
the declaration and the registered estimator can never diverge.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationEstimatorSpec:
    """Environment-independent declaration of one foundation estimator.

    Carries only metadata — no ``fit`` callable, no package import. The runtime
    :class:`EstimatorSpec` in ``foundation.py`` is built from this record, so the
    legal objective/encoding matrix shown in the contract is exactly the one the
    registered estimator enforces.
    """

    name: str
    objectives: frozenset[str]
    encodings: frozenset[str]
    default_encoding: str
    supports_early_stopping: bool
    native_categorical: bool
    # Any one of these packages being importable makes the estimator *executable*
    # (the extra is installed). The declaration is present regardless.
    required_packages: tuple[str, ...]
    # Short parenthetical shown after the estimator's obj/enc line in the contract.
    caveat: str
    # Longer prose used as the registered estimator's ``description``.
    description: str


# ── the declarations (extend this tuple to add a foundation estimator) ────────

TABPFN_SPEC = FoundationEstimatorSpec(
    name="tabpfn",
    # TabPFN has a single regression head; advertise the honest objective only.
    objectives=frozenset({"squared_error"}),
    encodings=frozenset({"ordinal", "one_hot"}),
    default_encoding="ordinal",
    supports_early_stopping=False,
    native_categorical=False,
    required_packages=("tabpfn", "tabpfn_client"),
    caveat="requires `bootstrap-track --enable-foundation-models` + the `[foundation]` extra",
    description=(
        "TabPFN-3 foundation model (in-context learning, no gradient training). "
        "Training context is subsampled to max_context_rows (exposure-weighted by "
        "default) and scoring is batched; exposure enters via the subsample, not a "
        "sample weight, with framework calibration fixing the level. backend='api' "
        "offloads to Prior Labs' GPU (needs TABPFN_TOKEN) — the practical choice on "
        "Apple Silicon; backend='local' runs on this machine (CPU is slow at scale). "
        "Requires the [foundation] extra and a per-run opt-in."
    ),
)

FOUNDATION_ESTIMATOR_SPECS: tuple[FoundationEstimatorSpec, ...] = (TABPFN_SPEC,)

FOUNDATION_ESTIMATOR_NAMES: frozenset[str] = frozenset(
    spec.name for spec in FOUNDATION_ESTIMATOR_SPECS
)

_SPECS_BY_NAME: dict[str, FoundationEstimatorSpec] = {
    spec.name: spec for spec in FOUNDATION_ESTIMATOR_SPECS
}


def foundation_spec(name: str) -> FoundationEstimatorSpec | None:
    """Return the static declaration for ``name``, or ``None`` if it is not a
    known foundation estimator."""
    return _SPECS_BY_NAME.get(name)


def _importable(module_name: str) -> bool:
    if module_name in sys.modules:  # already imported (incl. test stubs)
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def foundation_packages_available(spec: FoundationEstimatorSpec) -> bool:
    """True if any of ``spec``'s packages is importable — i.e. the estimator is
    *executable* in this environment (the ``[foundation]`` extra is installed)."""
    return any(_importable(pkg) for pkg in spec.required_packages)
