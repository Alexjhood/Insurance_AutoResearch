"""Declarative model recipes with a Python escape hatch (#6).

A recipe is a small, validated description of a common model (estimator,
objective, encoding, params, structure) that trusted framework code interprets —
replacing ~100 lines of hand-written, error-prone scaffolding for the common
case. Genuinely novel models still use a run-local Python script (the escape
hatch), and both paths share the framework's units/exposure/calibration stage.

Public API:
    fit_predict(...)            — the model hook (used by the dispatcher).
    validate_recipe(recipe)     — validity-matrix validation.
    recipe_summary(recipe)      — one-line description.
    menu()                      — full building-block vocabulary (contracts/docs).
    list_estimators/objectives/encodings/structures()
"""

from __future__ import annotations

import os

from autoresearch.models.recipe.encoders import register_builtin_encodings
from autoresearch.models.recipe.estimators import register_builtin_estimators
from autoresearch.models.recipe.foundation import register_foundation_estimators
from autoresearch.models.recipe.registry import (
    RecipeError,
    list_encodings,
    list_estimators,
    list_objectives,
    list_structures,
    menu,
    register_estimator,
    register_objective,
    register_structure,
)
from autoresearch.models.recipe.registry import ObjectiveSpec


def _register_builtin_objectives() -> None:
    register_objective(ObjectiveSpec("tweedie", "nonneg", "Compound Poisson-Gamma; zero-inflated cost."))
    register_objective(ObjectiveSpec("poisson", "nonneg", "Counts / frequency."))
    register_objective(ObjectiveSpec("gamma", "positive", "Strictly positive severity."))
    register_objective(ObjectiveSpec("squared_error", "any", "Plain least-squares regression."))


def _register_builtin_structures() -> None:
    register_structure("direct", "One model predicts the active-target rate directly.")
    register_structure("frequency_severity", "Frequency × severity two-stage model (burning cost only).")


def _foundation_env_enabled() -> bool:
    return os.environ.get("AUTORESEARCH_FOUNDATION_MODELS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def enable_foundation_models() -> list[str]:
    """Opt a run into foundation estimators (TabPFN, ...).

    Registers each foundation estimator whose optional package is installed into
    the live registry, so ``menu()``, ``validate_recipe`` and the interpreter all
    pick them up immediately. Idempotent and import-order independent: the CLI
    calls this once it resolves that a run enabled foundation models, regardless
    of whether the recipe package was already imported. Returns the newly
    available estimator names (empty if the extra is not installed).
    """
    return register_foundation_estimators()


def _bootstrap() -> None:
    _register_builtin_objectives()
    register_builtin_encodings()
    register_builtin_estimators()
    _register_builtin_structures()
    # Dev / untracked convenience: honour the env var at import time. The CLI also
    # calls enable_foundation_models() explicitly for tracked runs (see gating).
    if _foundation_env_enabled():
        register_foundation_estimators()


_bootstrap()

# Imported after bootstrap so the registries are populated.
from autoresearch.models.recipe.interpreter import fit_predict  # noqa: E402
from autoresearch.models.recipe.schema import recipe_summary, validate_recipe  # noqa: E402

__all__ = [
    "fit_predict",
    "validate_recipe",
    "recipe_summary",
    "menu",
    "list_estimators",
    "list_objectives",
    "list_encodings",
    "list_structures",
    "register_estimator",
    "register_objective",
    "register_structure",
    "enable_foundation_models",
    "RecipeError",
]
