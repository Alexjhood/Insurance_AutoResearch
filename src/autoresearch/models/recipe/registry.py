"""Extensible registries for recipe building blocks.

A *recipe* is assembled from four kinds of building block:

* **estimators**  — the prediction engine (lightgbm, xgboost, GLMs, ...).
* **objectives**  — the loss/error each engine minimises, with a label domain.
* **encodings**   — how categorical levels become model inputs.
* **structures**  — how one or more stage models combine into one prediction.

Each block type has its own registry plus a registration decorator, so adding a
new method is a small, self-contained code change (a curated registry, by
design — anything not registered here must use the Python escape hatch). The
recipe interpreter reads only from these registries, so it never needs editing
when a block is added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class RecipeError(ValueError):
    """Raised for an invalid or unsupported recipe."""


# ── Objectives ───────────────────────────────────────────────────────────────

# label domain: "nonneg" | "positive" | "any"
@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    label_domain: str
    description: str = ""


_OBJECTIVES: dict[str, ObjectiveSpec] = {}


def register_objective(spec: ObjectiveSpec) -> ObjectiveSpec:
    _OBJECTIVES[spec.name] = spec
    return spec


def get_objective(name: str) -> ObjectiveSpec:
    try:
        return _OBJECTIVES[name]
    except KeyError:
        raise RecipeError(
            f"Unknown objective {name!r}. Registered objectives: {sorted(_OBJECTIVES)}"
        ) from None


def list_objectives() -> list[str]:
    return sorted(_OBJECTIVES)


# ── Encodings ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EncodingSpec:
    name: str
    # builder(numeric_cols, categorical_cols) -> sklearn transformer, OR None for
    # native handling (the estimator consumes a category-typed DataFrame directly).
    builder: Callable[[list[str], list[str]], Any] | None
    native: bool = False
    description: str = ""


_ENCODINGS: dict[str, EncodingSpec] = {}


def register_encoding(spec: EncodingSpec) -> EncodingSpec:
    _ENCODINGS[spec.name] = spec
    return spec


def get_encoding(name: str) -> EncodingSpec:
    try:
        return _ENCODINGS[name]
    except KeyError:
        raise RecipeError(
            f"Unknown encoding {name!r}. Registered encodings: {sorted(_ENCODINGS)}"
        ) from None


def list_encodings() -> list[str]:
    return sorted(_ENCODINGS)


# ── Estimators ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FitContext:
    """Everything an estimator needs to fit one stage."""

    objective: str
    params: dict[str, Any]
    X_train: Any
    y_train: Any
    w_train: Any
    X_val: Any | None
    y_val: Any | None
    w_val: Any | None
    categorical_features: list[str] | None
    early_stopping: int | None
    # Name → post-encoding column index, only for encodings that keep one
    # column per feature (ordinal). None on native/one-hot paths. Lets
    # estimators that take positional categorical indices (e.g. TabPFN)
    # locate named features in the encoded matrix.
    encoded_column_indices: dict[str, int] | None = None


@dataclass(frozen=True)
class EstimatorSpec:
    """A registered prediction engine.

    ``fit`` receives a :class:`FitContext` and must return a fitted object
    exposing ``predict(X) -> np.ndarray`` of non-negative rate predictions, plus
    a notes dict: ``fit(ctx) -> tuple[predictor, dict]``.
    """

    name: str
    objectives: frozenset[str]
    encodings: frozenset[str]
    default_encoding: str
    fit: Callable[[FitContext], tuple[Any, dict[str, Any]]]
    supports_early_stopping: bool = False
    # True when the estimator manages its own internal validation split for early
    # stopping (e.g. sklearn HistGradientBoosting), so the interpreter must NOT
    # also carve out an external holdout.
    self_validates_early_stopping: bool = False
    native_categorical: bool = False
    description: str = ""
    # Curated set of accepted hyperparameter names (after alias) for this estimator.
    # Empty means "do not restrict" (no name validation).
    allowed_params: frozenset[str] = field(default_factory=frozenset)
    # Groups of mutually-exclusive param names, e.g. ("power", "tweedie_variance_power").
    param_conflicts: tuple[tuple[str, ...], ...] = ()
    # Well-known library-native spellings mapped to the curated canonical name,
    # e.g. xgboost's "eta" -> "learning_rate". Canonicalised before validation.
    param_aliases: tuple[tuple[str, str], ...] = ()


_ESTIMATORS: dict[str, EstimatorSpec] = {}


def register_estimator(spec: EstimatorSpec) -> EstimatorSpec:
    _ESTIMATORS[spec.name] = spec
    return spec


def get_estimator(name: str) -> EstimatorSpec:
    try:
        return _ESTIMATORS[name]
    except KeyError:
        raise RecipeError(
            f"Unknown estimator {name!r}. Registered estimators: {sorted(_ESTIMATORS)}. "
            "To use a method not in the curated registry, write a run-local Python "
            "model script (the escape hatch) instead of a recipe."
        ) from None


def list_estimators() -> list[str]:
    return sorted(_ESTIMATORS)


# ── Structures ───────────────────────────────────────────────────────────────

_STRUCTURES: dict[str, str] = {}


def register_structure(name: str, description: str = "") -> None:
    _STRUCTURES[name] = description


def list_structures() -> list[str]:
    return sorted(_STRUCTURES)


def menu() -> dict[str, Any]:
    """Return the full recipe vocabulary for contracts, validation, and docs."""

    return {
        "structures": list_structures(),
        "estimators": {
            name: {
                "objectives": sorted(spec.objectives),
                "encodings": sorted(spec.encodings),
                "default_encoding": spec.default_encoding,
                "supports_early_stopping": spec.supports_early_stopping,
                "native_categorical": spec.native_categorical,
                "description": spec.description,
            }
            for name, spec in sorted(_ESTIMATORS.items())
        },
        "objectives": {
            name: {"label_domain": spec.label_domain, "description": spec.description}
            for name, spec in sorted(_OBJECTIVES.items())
        },
        "encodings": {
            name: {"native": spec.native, "description": spec.description}
            for name, spec in sorted(_ENCODINGS.items())
        },
    }
