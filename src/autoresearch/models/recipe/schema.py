"""Recipe schema and validity-matrix validation.

A recipe describes a model declaratively. Two shapes are supported:

direct (one stage)::

    {
      "structure": "direct",
      "estimator": "lightgbm",
      "objective": "tweedie",
      "encoding": "native_categorical",   # optional; estimator default if omitted
      "params": {"num_leaves": 63, "learning_rate": 0.05},
      "early_stopping": 50                  # optional
    }

frequency × severity (two nested stages)::

    {
      "structure": "frequency_severity",
      "stages": {
        "frequency": {"estimator": "lightgbm", "objective": "poisson"},
        "severity":  {"estimator": "lightgbm", "objective": "gamma"}
      }
    }

Validation rejects illegal building-block combinations *before* anything runs, so
crash-prone proposals (gamma-on-zero labels, native categoricals on xgboost,
Tweedie on a model that has no Tweedie loss) never reach the execution path.
"""

from __future__ import annotations

from typing import Any

from autoresearch.models.recipe.registry import (
    RecipeError,
    get_encoding,
    get_estimator,
    get_objective,
    list_structures,
)
from autoresearch.targets import BURNING_COST, FREQUENCY, SEVERITY, normalise_target_mode


STAGE_KEYS = {"estimator", "objective", "encoding", "params", "early_stopping", "target"}

# Param names the framework controls; an experiment must not set them directly.
RESERVED_PARAM_NAMES = frozenset({"objective", "loss", "target_mode", "recipe"})


def _canonicalise_param_aliases(
    params: dict[str, Any], spec: Any, where: str, errors: list[str]
) -> None:
    """Rewrite well-known library-native param spellings to the curated name.

    Mutates ``params`` in place so the canonical name is what validation checks,
    what gets persisted, and what the estimator builder receives (agents
    routinely write ``eta`` for xgboost or ``bagging_fraction`` for lightgbm).
    """

    for alias, canonical in getattr(spec, "param_aliases", ()) or ():
        if alias not in params:
            continue
        if canonical in params:
            errors.append(
                f"{where}.params sets both {alias!r} and its canonical name "
                f"{canonical!r}; keep only {canonical!r}"
            )
            continue
        params[canonical] = params.pop(alias)


def _validate_params(params: dict[str, Any], spec: Any, where: str, errors: list[str]) -> None:
    """Catch bad param names/types at proposal time, not as paid runtime repairs."""
    _canonicalise_param_aliases(params, spec, where, errors)
    for name, value in params.items():
        if name in RESERVED_PARAM_NAMES:
            errors.append(f"{where}.params.{name} is framework-controlled and must not be set")
            continue
        if isinstance(value, dict):
            errors.append(f"{where}.params.{name} must be a scalar or list, not an object")
        if spec.allowed_params and name not in spec.allowed_params:
            errors.append(
                f"{where}.params.{name} is not a recognised parameter for estimator "
                f"{spec.name!r} (allowed: {sorted(spec.allowed_params)}). Use the script "
                "escape hatch for parameters outside the curated set."
            )
    for group in getattr(spec, "param_conflicts", ()):
        present = [n for n in group if n in params]
        if len(present) > 1:
            errors.append(f"{where}.params sets mutually-exclusive parameters {present}; keep only one")

# Which objectives are scientifically valid for each target the framework knows.
TARGET_OBJECTIVES = {
    # Pure premium is non-negative with a point mass at zero: Poisson and Tweedie
    # (p=1 and 1<p<2) are both valid quasi-likelihoods; only gamma (needs y>0) is
    # excluded by the zeros.
    "pure_premium": {"tweedie", "poisson", "squared_error"},
    "frequency": {"poisson", "tweedie", "squared_error"},
    "severity": {"gamma", "poisson", "squared_error"},  # strictly positive by construction
}


def _stage_target(structure: str, stage_name: str | None, target_mode: str) -> str:
    if structure == "frequency_severity":
        return "frequency" if stage_name == "frequency" else "severity"
    # direct
    if target_mode == FREQUENCY:
        return "frequency"
    if target_mode == SEVERITY:
        return "severity"
    return "pure_premium"


def _validate_stage(
    stage: dict[str, Any],
    *,
    structure: str,
    stage_name: str | None,
    target_mode: str,
    errors: list[str],
) -> None:
    where = f"stages.{stage_name}" if stage_name else "recipe"
    if not isinstance(stage, dict):
        errors.append(f"{where} must be an object")
        return

    allowed_keys = STAGE_KEYS | ({"structure"} if stage_name is None else set())
    unknown = set(stage) - allowed_keys
    if unknown:
        errors.append(f"{where} has unknown keys: {sorted(unknown)}")

    estimator_name = stage.get("estimator")
    if not isinstance(estimator_name, str) or not estimator_name:
        errors.append(f"{where}.estimator is required")
        return
    try:
        spec = get_estimator(estimator_name)
    except RecipeError as exc:
        errors.append(f"{where}.estimator: {exc}")
        return

    objective = stage.get("objective")
    if not isinstance(objective, str) or not objective:
        errors.append(f"{where}.objective is required")
    else:
        try:
            get_objective(objective)
        except RecipeError as exc:
            errors.append(f"{where}.objective: {exc}")
        if objective not in spec.objectives:
            errors.append(
                f"{where}: estimator {estimator_name!r} does not support objective "
                f"{objective!r} (supported: {sorted(spec.objectives)})"
            )
        target = _stage_target(structure, stage_name, target_mode)
        allowed = TARGET_OBJECTIVES.get(target, set())
        if objective not in allowed:
            errors.append(
                f"{where}: objective {objective!r} is invalid for target {target!r} "
                f"(allowed: {sorted(allowed)})"
            )
        # The interpreter derives the target from structure + mode; an explicit
        # `target` must agree (it is otherwise silently ignored).
        declared_target = stage.get("target")
        if declared_target is not None and declared_target != target:
            errors.append(
                f"{where}: target {declared_target!r} does not match the derived target "
                f"{target!r} for this structure/mode"
            )

    encoding = stage.get("encoding") or spec.default_encoding
    try:
        get_encoding(encoding)
    except RecipeError as exc:
        errors.append(f"{where}.encoding: {exc}")
    else:
        if encoding not in spec.encodings:
            errors.append(
                f"{where}: estimator {estimator_name!r} does not support encoding "
                f"{encoding!r} (supported: {sorted(spec.encodings)})"
            )

    params = stage.get("params", {})
    if not isinstance(params, dict):
        errors.append(f"{where}.params must be an object")
    else:
        _validate_params(params, spec, where, errors)

    es = stage.get("early_stopping")
    if es is not None and (not isinstance(es, int) or isinstance(es, bool) or es < 0):
        errors.append(f"{where}.early_stopping must be a non-negative integer")


def validate_recipe(recipe: dict[str, Any], *, target_mode: str = BURNING_COST) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""

    errors: list[str] = []
    if not isinstance(recipe, dict):
        return ["recipe must be an object"]

    target_mode = normalise_target_mode(target_mode)
    structure = recipe.get("structure", "direct")
    if structure not in list_structures():
        errors.append(f"structure must be one of {list_structures()}")
        return errors

    if structure == "frequency_severity":
        if target_mode != BURNING_COST:
            errors.append("frequency_severity structure is only valid in burning_cost target mode")
        stages = recipe.get("stages")
        if not isinstance(stages, dict):
            errors.append("frequency_severity requires a 'stages' object with 'frequency' and 'severity'")
            return errors
        for name in ("frequency", "severity"):
            if name not in stages:
                errors.append(f"stages.{name} is required for frequency_severity")
            else:
                _validate_stage(stages[name], structure=structure, stage_name=name,
                                target_mode=target_mode, errors=errors)
    else:  # direct
        _validate_stage(recipe, structure=structure, stage_name=None,
                        target_mode=target_mode, errors=errors)

    return errors


def recipe_summary(recipe: dict[str, Any]) -> str:
    """One-line human/agent summary of a recipe for logs and notes."""

    structure = recipe.get("structure", "direct")
    if structure == "frequency_severity":
        stages = recipe.get("stages", {})
        parts = []
        for name in ("frequency", "severity"):
            s = stages.get(name, {})
            parts.append(f"{name}={s.get('estimator')}/{s.get('objective')}")
        return "freq_sev[" + ", ".join(parts) + "]"
    return f"{recipe.get('estimator')}/{recipe.get('objective')}/{recipe.get('encoding', 'default')}"
