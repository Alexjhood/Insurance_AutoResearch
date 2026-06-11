"""Recipe library with a run-local / cross-run reuse switch (#6, Options 1 & 2).

Every validly-run recipe is recorded with its outcome so a run can learn from
both winners and dead-ends. Two reuse scopes are supported, chosen by
``[recipes] reuse_scope`` in config:

* **Option 1 — "run" (default):** a run sees only recipes tried earlier in *this*
  run (a run-local JSONL ledger). Fully isolated; preserves an independent
  discovery benchmark with no extra mechanism.
* **Option 2 — "memory":** a run may also see recipes from *other* runs/threads,
  via a cross-run ledger in the memory store. This is part of the memory feature
  and only engages when memory access is granted
  (``AUTORESEARCH_MEMORY_ACCESS`` = ``own`` | ``all``). With access ``own`` only
  the same model identity's recipes are visible; with ``all`` every recipe is.
  When access is ``none`` it transparently falls back to run scope, so an
  uncontaminated benchmark stays uncontaminated even if config asks for memory.

Recording always writes the run-local ledger; it additionally appends to the
cross-run ledger only when a model identity is known (mirroring how the memory
harvester behaves). Reading is what the access gate protects.
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig
from autoresearch.models.recipe.schema import recipe_summary

logger = logging.getLogger(__name__)

_RUN_LEDGER_NAME = "recipe_ledger.jsonl"
_GLOBAL_LEDGER_NAME = "recipes.jsonl"

# Terminal outcomes rank above interim ones when the same recipe recurs.
_OUTCOME_RANK = {
    "promoted": 5,
    "local_promoted": 4,
    "rejected": 3,
    "inconclusive": 2,
    "failed": 1,
    "completed": 0,
}
_TERMINAL_OUTCOMES = frozenset(
    {"promoted", "local_promoted", "rejected", "inconclusive", "failed"}
)

# The decision/workflow layers speak in verbs (promote/reject/auto_reject); the
# ledger and ranking use the canonical past-tense enum. Normalising here is what
# keeps a promoted champion from being out-ranked by a merely "completed" run.
_OUTCOME_ALIASES = {
    "promote": "promoted",
    "local_promote": "local_promoted",
    "reject": "rejected",
    "auto_reject": "rejected",
}
CANONICAL_OUTCOMES = frozenset(_OUTCOME_RANK)


def canonical_outcome(outcome: str) -> str:
    """Map a decision verb (or already-canonical value) to the canonical enum."""
    o = (outcome or "").strip().lower()
    return _OUTCOME_ALIASES.get(o, o)


def _run_ledger_path(config: ProjectConfig) -> Path:
    return config.artifacts_dir / _RUN_LEDGER_NAME


def _global_ledger_path() -> Path | None:
    try:
        from autoresearch.memory.store import memory_root

        return memory_root() / _GLOBAL_LEDGER_NAME
    except Exception:  # pragma: no cover - memory subsystem optional
        return None


def _model_identity(config: ProjectConfig) -> dict[str, str] | None:
    try:
        manifest_path = config.artifacts_dir / "run_manifest.json"
        if not manifest_path.exists():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ident = manifest.get("model_identity")
        if ident and ident.get("provider") and ident.get("name"):
            return {"provider": str(ident["provider"]), "name": str(ident["name"])}
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _normalise_model_spec(
    recipe: dict[str, Any],
    *,
    target_strategy: str | None = None,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    model_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the complete reusable model specification for fingerprinting."""

    if isinstance(model_spec, dict) and model_spec:
        result = json.loads(json.dumps(model_spec))
        result["recipe"] = recipe
    else:
        result = {"recipe": recipe}
        if target_strategy:
            result["target_strategy"] = target_strategy
        if feature_inclusions:
            result["feature_inclusions"] = sorted(set(feature_inclusions))
        if feature_exclusions:
            result["feature_exclusions"] = sorted(set(feature_exclusions))
    for key in ("feature_inclusions", "feature_exclusions"):
        if isinstance(result.get(key), list):
            result[key] = sorted(set(result[key]))
    return result


def _row_model_spec(row: dict[str, Any]) -> dict[str, Any] | None:
    model_spec = row.get("model_spec")
    if isinstance(model_spec, dict) and isinstance(model_spec.get("recipe"), dict):
        return model_spec
    recipe = row.get("recipe")
    if isinstance(recipe, dict):
        return {"recipe": recipe}
    return None


def _fingerprint(model_spec: dict[str, Any]) -> str:
    return json.dumps(model_spec, sort_keys=True, separators=(",", ":"))


def _model_summary(
    recipe: dict[str, Any],
    model_spec: dict[str, Any],
    *,
    variant_id: str,
) -> str:
    summary = recipe_summary(recipe)
    details = _recipe_details(recipe)
    if details:
        summary += f" ({details})"
    included = model_spec.get("feature_inclusions")
    excluded = model_spec.get("feature_exclusions")
    if isinstance(included, list) and included:
        summary += f" features=only[{','.join(included)}]"
    if isinstance(excluded, list) and excluded:
        summary += f" features=except[{','.join(excluded)}]"
    return f"{summary} [variant={variant_id}]"


def _recipe_details(recipe: dict[str, Any]) -> str:
    aliases = {
        "learning_rate": "lr",
        "num_leaves": "leaves",
        "n_estimators": "trees",
        "max_depth": "depth",
        "min_child_samples": "min_child",
        "reg_alpha": "l1",
        "reg_lambda": "l2",
        "colsample_bytree": "colsample",
    }

    def stage_details(stage: dict[str, Any], prefix: str = "") -> list[str]:
        details = [
            f"{prefix}{aliases.get(str(key), key)}={value}"
            for key, value in sorted((stage.get("params") or {}).items())
        ]
        if stage.get("early_stopping") is not None:
            details.append(f"{prefix}early_stop={stage['early_stopping']}")
        return details

    if recipe.get("structure", "direct") == "frequency_severity":
        details: list[str] = []
        for stage_name in ("frequency", "severity"):
            stage = (recipe.get("stages") or {}).get(stage_name) or {}
            details.extend(stage_details(stage, prefix=f"{stage_name[:4]}."))
    else:
        details = stage_details(recipe)
    if len(details) > 5:
        hidden = len(details) - 5
        details = details[:5] + [f"+{hidden} more"]
    return ", ".join(details)


def _append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def record_recipe(
    config: ProjectConfig,
    recipe: dict[str, Any],
    *,
    experiment_id: str,
    outcome: str,
    score: float | None = None,
    target_strategy: str | None = None,
    feature_inclusions: list[str] | None = None,
    feature_exclusions: list[str] | None = None,
    model_spec: dict[str, Any] | None = None,
    comparison_score: float | None = None,
    comparison_lift: float | None = None,
) -> None:
    """Record a recipe and its outcome. Never raises into the caller."""

    if not isinstance(recipe, dict) or not recipe:
        return
    try:
        from datetime import datetime, timezone

        reusable_spec = _normalise_model_spec(
            recipe,
            target_strategy=target_strategy,
            feature_inclusions=feature_inclusions,
            feature_exclusions=feature_exclusions,
            model_spec=model_spec,
        )
        fingerprint = _fingerprint(reusable_spec)
        variant_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:8]
        entry: dict[str, Any] = {
            "recipe": recipe,
            "model_spec": reusable_spec,
            "summary": _model_summary(recipe, reusable_spec, variant_id=variant_id),
            "variant_id": variant_id,
            "experiment_id": experiment_id,
            "outcome": canonical_outcome(outcome),
            "screen_score": None if score is None else round(float(score), 6),
            "comparison_score": (
                None if comparison_score is None else round(float(comparison_score), 6)
            ),
            "comparison_lift": (
                None if comparison_lift is None else round(float(comparison_lift), 6)
            ),
            "track_id": config.track_id,
            "run_id": config.run_id,
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _append(_run_ledger_path(config), entry)

        identity = _model_identity(config)
        if identity:
            global_path = _global_ledger_path()
            if global_path is not None:
                _append(global_path, {**entry, **{f"model_{k}": v for k, v in identity.items()}})
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("record_recipe failed (non-fatal): %s", exc)


def effective_reuse_scope(config: ProjectConfig) -> str:
    """Resolve the scope actually in force, after memory-access gating.

    Returns ``"run"`` or ``"memory"``. ``"memory"`` only survives when memory
    access is ``own`` or ``all``; otherwise it degrades to ``"run"``.
    """

    requested = (getattr(config, "recipe_reuse_scope", "run") or "run").strip().lower()
    if requested != "memory":
        return "run"
    try:
        from autoresearch.memory import resolve_memory_access

        access = resolve_memory_access(config)
    except Exception:  # pragma: no cover
        access = "none"
    return "memory" if access in {"own", "all"} else "run"


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                if "screen_score" not in row and "score" in row:
                    row["screen_score"] = row.get("score")
                if not row.get("variant_id"):
                    model_spec = _row_model_spec(row)
                    if model_spec is not None:
                        fingerprint = _fingerprint(model_spec)
                        row["variant_id"] = hashlib.sha256(
                            fingerprint.encode("utf-8")
                        ).hexdigest()[:8]
                model_spec = _row_model_spec(row)
                recipe = model_spec.get("recipe") if model_spec else None
                if isinstance(recipe, dict) and row.get("variant_id"):
                    row["summary"] = _model_summary(
                        recipe,
                        model_spec,
                        variant_id=str(row["variant_id"]),
                    )
                rows.append(row)
    except (OSError, json.JSONDecodeError):
        return rows
    return rows


def _dedup_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated model specs, preferring terminal state over completion."""

    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_spec = _row_model_spec(row)
        if model_spec is None:
            continue
        key = _fingerprint(model_spec)
        current = best.get(key)
        if current is None:
            best[key] = row
            continue

        row_terminal = str(row.get("outcome", "")) in _TERMINAL_OUTCOMES
        current_terminal = str(current.get("outcome", "")) in _TERMINAL_OUTCOMES
        replace = row_terminal and not current_terminal
        if row_terminal == current_terminal:
            replace = _entry_rank(row) > _entry_rank(current)
        if replace:
            replacement = dict(row)
            for metric in ("screen_score", "comparison_score", "comparison_lift"):
                if replacement.get(metric) is None and isinstance(current.get(metric), (int, float)):
                    replacement[metric] = current[metric]
            best[key] = replacement
        else:
            merged = dict(current)
            changed = False
            for metric in ("screen_score", "comparison_score", "comparison_lift"):
                if merged.get(metric) is None and isinstance(row.get(metric), (int, float)):
                    merged[metric] = row[metric]
                    changed = True
            if changed:
                best[key] = merged
    return list(best.values())


def _entry_rank(row: dict[str, Any]) -> tuple[int, float]:
    outcome_rank = _OUTCOME_RANK.get(str(row.get("outcome", "")), 0)
    score = row.get("comparison_score")
    if not isinstance(score, (int, float)):
        score = row.get("screen_score", row.get("score"))
    return (outcome_rank, float(score) if isinstance(score, (int, float)) else float("-inf"))


def list_recipes(config: ProjectConfig, *, limit: int = 20) -> list[dict[str, Any]]:
    """Return ranked prior recipes visible under the effective reuse scope."""

    scope = effective_reuse_scope(config)
    rows = _read_ledger(_run_ledger_path(config))

    if scope == "memory":
        global_path = _global_ledger_path()
        if global_path is not None:
            global_rows = _read_ledger(global_path)
            from autoresearch.memory import resolve_memory_access

            access = resolve_memory_access(config)
            if access == "own":
                identity = _model_identity(config)
                if identity:
                    global_rows = [
                        r for r in global_rows
                        if r.get("model_provider") == identity["provider"]
                        and r.get("model_name") == identity["name"]
                    ]
            rows = rows + global_rows

    deduped = _dedup_best(rows)
    champion_id = None
    try:
        from autoresearch.experiment_registry.registry import get_official_champion

        champion = get_official_champion(config.registry_path)
        champion_id = champion.get("champion_id") if champion else None
    except Exception:
        pass
    annotated = []
    for row in deduped:
        item = dict(row)
        item["is_current_champion"] = bool(
            champion_id
            and item.get("experiment_id") == champion_id
            and item.get("track_id") == config.track_id
            and item.get("run_id") == config.run_id
        )
        annotated.append(item)
    ranked = sorted(
        annotated,
        key=lambda row: (bool(row.get("is_current_champion")), *_entry_rank(row)),
        reverse=True,
    )
    return ranked[:limit]
