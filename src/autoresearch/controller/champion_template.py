"""Champion template generation (#8).

After every promotion the framework writes a ready-to-tweak template into the
run's proposal inbox, so a one-parameter follow-up is a tiny diff instead of a
rewritten model:

* ``champion_recipe.json`` — the promoted champion's recipe + metadata (when the
  champion is recipe-based).
* ``champion_template.py``  — a runnable escape-hatch model script. For a
  recipe champion it reconstructs the model through the recipe interpreter and
  exposes a ``PARAM_OVERRIDES`` dict for trivial follow-ups. For a script
  champion it copies the winning script as the starting point.

This fulfils the long-standing manual promise of a ``champion_template.py`` that
the framework previously never generated.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch.config import ProjectConfig

logger = logging.getLogger(__name__)

# Champion artifacts written into the proposal inbox. These are framework-owned
# reference files, NOT agent proposals — the proposal ingester must skip them
# (champion_template.py is a .py so it is already invisible to the *.json scan,
# but champion_recipe.json must be excluded by name).
CHAMPION_RECIPE_FILENAME = "champion_recipe.json"
CHAMPION_TEMPLATE_FILENAME = "champion_template.py"
RESERVED_INBOX_FILENAMES = frozenset({CHAMPION_RECIPE_FILENAME, CHAMPION_TEMPLATE_FILENAME})


_RECIPE_TEMPLATE = '''\
"""Champion template — promoted experiment {experiment_id} ({recorded_at}).

This run's current champion as a ready-to-tweak recipe. To run a small follow-up
experiment, edit PARAM_OVERRIDES (top-level recipe params) and/or RECIPE, point a
proposal's model.script_path at this file, and submit. For anything structural,
write a fresh recipe or model script instead.
"""

from __future__ import annotations

import copy

from autoresearch.models.recipe.interpreter import fit_predict as _recipe_fit_predict

# The promoted champion recipe.
RECIPE = {recipe!r}

# One-parameter follow-ups: merged into the top-level recipe params at run time.
# (For frequency_severity champions, edit RECIPE['stages'][...] directly instead.)
PARAM_OVERRIDES: dict = {{}}


def fit_predict(train, score, *, feature_inclusions=None, feature_exclusions=None, **hyperparameters):
    recipe = copy.deepcopy(RECIPE)
    if PARAM_OVERRIDES:
        recipe.setdefault("params", {{}}).update(PARAM_OVERRIDES)
    hyperparameters["recipe"] = recipe
    return _recipe_fit_predict(
        train, score,
        feature_inclusions=feature_inclusions,
        feature_exclusions=feature_exclusions,
        **hyperparameters,
    )
'''


def _load_experiment_model_cfg(config: ProjectConfig, experiment_id: str) -> dict[str, Any] | None:
    """Return the promoted experiment's ``model`` config block, if recoverable."""

    try:
        from autoresearch.experiment_registry.registry import get_experiment
        from autoresearch.utils.io import read_json

        row = get_experiment(config.registry_path, experiment_id)
        snapshot_path = row.get("config_snapshot_path")
        if not snapshot_path:
            return None
        snapshot = read_json(Path(snapshot_path))
        experiment = snapshot.get("experiment", {})
        model = experiment.get("model", {})
        return {
            "model": model if isinstance(model, dict) else {},
            "target_strategy": experiment.get("target_strategy"),
            "model_script_path": snapshot.get("model_script_path"),
        }
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("Could not load experiment model cfg for template: %s", exc)
        return None


def generate_champion_template(config: ProjectConfig, experiment_id: str) -> dict[str, Path]:
    """Write champion template artifacts into the run inbox. Best-effort."""

    written: dict[str, Path] = {}
    try:
        info = _load_experiment_model_cfg(config, experiment_id)
        if info is None:
            return written
        inbox = config.handoff_proposal_inbox_dir
        inbox.mkdir(parents=True, exist_ok=True)
        recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        model = info["model"]
        recipe = model.get("recipe")
        template_path = inbox / CHAMPION_TEMPLATE_FILENAME

        if isinstance(recipe, dict) and recipe:
            recipe_path = inbox / CHAMPION_RECIPE_FILENAME
            recipe_path.write_text(
                json.dumps(
                    {
                        "experiment_id": experiment_id,
                        "recorded_at": recorded_at,
                        "target_strategy": info.get("target_strategy"),
                        "recipe": recipe,
                        "model": model,
                    },
                    indent=2, sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
            written["champion_recipe"] = recipe_path
            template_path.write_text(
                _RECIPE_TEMPLATE.format(
                    experiment_id=experiment_id, recorded_at=recorded_at, recipe=recipe
                ),
                encoding="utf-8",
            )
            written["champion_template"] = template_path
        else:
            # Script-based champion: a recipe override no longer applies — remove any
            # stale champion_recipe.json left by a previous recipe champion.
            stale_recipe = inbox / CHAMPION_RECIPE_FILENAME
            if stale_recipe.exists():
                try:
                    stale_recipe.unlink()
                except OSError:
                    pass
            # Copy the winning script as the starting point.
            script_path = info.get("model_script_path") or model.get("script_path")
            if script_path and Path(script_path).exists():
                header = (
                    f'"""Champion template — promoted experiment {experiment_id} ({recorded_at}).\n'
                    "Copied from the promoted model script. Edit and submit a follow-up;\n"
                    "for trivial changes prefer a recipe (see champion_recipe.json when present).\n"
                    '"""\n\n'
                )
                template_path.write_text(
                    header + Path(script_path).read_text(encoding="utf-8"), encoding="utf-8"
                )
                written["champion_template"] = template_path
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("generate_champion_template failed (non-fatal): %s", exc)
    return written
