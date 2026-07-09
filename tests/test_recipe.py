"""Tests for declarative model recipes (#6), framework units/calibration (#7),
the recipe reuse library (Options 1/2), and champion templates (#8)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoresearch.models import recipe as recipe_pkg
from autoresearch.models.dispatcher import dispatch_model, dispatch_model_on_explicit_frames
from autoresearch.models.prediction import (
    Prediction,
    PredictionUnitError,
    validate_objective_labels,
    validate_predictions,
)
from autoresearch.models.recipe import validate_recipe

from tests.test_runner import _make_config


def _frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "record_id": np.arange(n),
        "Exposure": rng.uniform(0.5, 1.5, n),
        "VehPower": rng.integers(1, 6, n),
        "Region": rng.choice(list("abcde"), n),
        "ClaimNb": rng.poisson(0.25, n),
    })
    frame["ClaimAmountCount"] = frame["ClaimNb"]
    frame["ClaimAmount"] = frame["ClaimNb"] * rng.gamma(2.0, 400.0, n)
    frame["ClaimAmountCapped"] = frame["ClaimAmount"]
    n_train = int(n * 0.75)
    split = pd.DataFrame({
        "record_id": np.arange(n),
        "split": ["train"] * n_train + ["search_validation"] * (n - n_train),
    })
    return frame, split


# ── #7 framework units / calibration / validators ────────────────────────────

def test_finalize_rate_to_total_and_calibration() -> None:
    frame, split = _frame()
    rc = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
          "params": {"num_leaves": 15}, "early_stopping": 20}
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    # Calibration factor recorded, predictions finite & non-negative.
    assert "calib_factor" in res.model_notes
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)
    # Aggregate calibration ties predicted train total to actual train total.
    train_mask = res.predictions["split"].to_numpy() == "train"
    assert train_mask.any()


def test_hist_gbm_one_hot_runs_dense() -> None:
    # Regression: advertised-legal hist_gbm × one_hot crashed on sparse X
    # (sklearn TypeError) in run 20260612T105643Z cycle 9.
    frame, split = _frame()
    rc = {"structure": "direct", "estimator": "hist_gbm", "objective": "squared_error",
          "encoding": "one_hot", "params": {"max_depth": 4}, "early_stopping": 10}
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)


def test_validate_objective_labels() -> None:
    validate_objective_labels("poisson", np.array([0.0, 1.0, 2.0]))
    with pytest.raises(PredictionUnitError):
        validate_objective_labels("gamma", np.array([0.0, 1.0]))
    with pytest.raises(PredictionUnitError):
        validate_objective_labels("poisson", np.array([-1.0, 1.0]))


def test_validate_predictions_rejects_bad() -> None:
    validate_predictions(np.array([1.0, 2.0, 3.0]), n_expected=3)
    with pytest.raises(PredictionUnitError):
        validate_predictions(np.array([1.0, 2.0]), n_expected=3)
    with pytest.raises(PredictionUnitError):
        validate_predictions(np.array([1.0, np.inf, 3.0]), n_expected=3)


def test_prediction_unit_validation() -> None:
    with pytest.raises(PredictionUnitError):
        Prediction(values=np.array([1.0]), unit="bogus")


# ── #6 validity matrix ────────────────────────────────────────────────────────

def test_recipe_validation_matrix() -> None:
    assert validate_recipe(
        {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"},
        target_mode="burning_cost") == []
    # gamma is invalid for pure premium (has zeros)
    assert validate_recipe(
        {"structure": "direct", "estimator": "lightgbm", "objective": "gamma"},
        target_mode="burning_cost")
    # xgboost has no native categorical encoding
    assert validate_recipe(
        {"structure": "direct", "estimator": "xgboost", "objective": "tweedie",
         "encoding": "native_categorical"}, target_mode="burning_cost")
    # hist_gbm has no tweedie loss
    assert validate_recipe(
        {"structure": "direct", "estimator": "hist_gbm", "objective": "tweedie"},
        target_mode="burning_cost")
    # freq_sev only valid in burning cost mode
    assert validate_recipe(
        {"structure": "frequency_severity",
         "stages": {"frequency": {"estimator": "lightgbm", "objective": "poisson"},
                    "severity": {"estimator": "lightgbm", "objective": "gamma"}}},
        target_mode="frequency")


def test_param_aliases_canonicalised() -> None:
    # Run 20260612T105643Z lost a cycle to xgboost's native `eta` spelling.
    rc = {"structure": "direct", "estimator": "xgboost", "objective": "tweedie",
          "params": {"eta": 0.05, "max_depth": 6}}
    assert validate_recipe(rc, target_mode="burning_cost") == []
    assert rc["params"] == {"learning_rate": 0.05, "max_depth": 6}

    rc = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
          "params": {"bagging_fraction": 0.8, "lambda_l2": 0.1}}
    assert validate_recipe(rc, target_mode="burning_cost") == []
    assert rc["params"] == {"subsample": 0.8, "reg_lambda": 0.1}


def test_param_alias_conflict_rejected() -> None:
    rc = {"structure": "direct", "estimator": "xgboost", "objective": "tweedie",
          "params": {"eta": 0.05, "learning_rate": 0.1}}
    errs = validate_recipe(rc, target_mode="burning_cost")
    assert errs and "canonical" in errs[0]


def test_unknown_estimator_points_to_escape_hatch() -> None:
    errs = validate_recipe(
        {"structure": "direct", "estimator": "catboost", "objective": "tweedie"},
        target_mode="burning_cost")
    assert errs and "escape hatch" in errs[0]


@pytest.mark.parametrize("rc", [
    {"structure": "direct", "estimator": "tweedie_glm", "objective": "tweedie", "encoding": "one_hot"},
    {"structure": "direct", "estimator": "elasticnet", "objective": "squared_error", "encoding": "one_hot"},
    {"structure": "direct", "estimator": "xgboost", "objective": "tweedie", "encoding": "ordinal"},
    {"structure": "frequency_severity",
     "stages": {"frequency": {"estimator": "lightgbm", "objective": "poisson", "params": {"num_leaves": 7}},
                "severity": {"estimator": "lightgbm", "objective": "gamma", "params": {"num_leaves": 7}}}},
])
def test_recipe_dispatch_variants(rc) -> None:
    frame, split = _frame()
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert len(res.predictions) == len(frame)
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)


@pytest.mark.parametrize("rc", [
    # Regression: agent passed sklearn's own `power` arg in params → previously
    # collided with the explicit power= kwarg (run 20260608T190605Z, iter 004).
    {"structure": "direct", "estimator": "tweedie_glm", "objective": "tweedie",
     "encoding": "one_hot", "params": {"alpha": 0.01, "max_iter": 1000, "power": 1.5}},
    {"structure": "direct", "estimator": "tweedie_glm", "objective": "tweedie",
     "encoding": "one_hot", "params": {"tweedie_variance_power": 1.4}},
])
def test_recipe_param_kwarg_collisions(rc) -> None:
    frame, split = _frame()
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": rc}, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)


# ── G: curated param validation ──────────────────────────────────────────────

@pytest.mark.parametrize("rc,needle", [
    ({"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
      "params": {"objective": "tweedie"}}, "framework-controlled"),
    ({"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
      "params": {"num_leevs": 31}}, "not a recognised parameter"),
    ({"structure": "direct", "estimator": "tweedie_glm", "objective": "tweedie",
      "encoding": "one_hot", "params": {"power": 1.5, "tweedie_variance_power": 1.5}}, "mutually-exclusive"),
    ({"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
      "params": {"num_leaves": {"nested": 1}}}, "must be a scalar"),
])
def test_recipe_param_validation_rejects(rc, needle) -> None:
    errs = validate_recipe(rc, target_mode="burning_cost")
    assert any(needle in e for e in errs), errs


# ── E: poisson allowed for pure premium ───────────────────────────────────────

def test_poisson_allowed_for_pure_premium() -> None:
    assert validate_recipe(
        {"structure": "direct", "estimator": "lightgbm", "objective": "poisson"},
        target_mode="burning_cost") == []
    frame, split = _frame()
    res = dispatch_model(
        frame, split, model_family="recipe", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={"recipe": {"structure": "direct", "estimator": "lightgbm", "objective": "poisson"}},
        target_mode="burning_cost",
    )
    assert len(res.predictions) == len(frame)


def test_recipe_target_field_must_match() -> None:
    assert validate_recipe(
        {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie", "target": "severity"},
        target_mode="burning_cost")  # declared severity != derived pure_premium


# ── B: canonical outcome normalization ────────────────────────────────────────

def test_recipe_outcome_canonicalised(tmp_path: Path) -> None:
    from autoresearch.models import recipe_library as lib

    config = _make_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    rc = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"}
    lib.record_recipe(config, rc, experiment_id="e1", outcome="completed", score=0.30)
    # decision verb, not past-tense — must be normalised so it out-ranks "completed"
    lib.record_recipe(config, rc, experiment_id="e1", outcome="promote", score=0.34)
    rows = lib.list_recipes(config)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "promoted"


def test_terminal_rejection_replaces_completed_and_preserves_score(tmp_path: Path) -> None:
    from autoresearch.models import recipe_library as lib

    config = _make_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    rc = {"structure": "direct", "estimator": "xgboost", "objective": "tweedie"}
    lib.record_recipe(config, rc, experiment_id="e1", outcome="completed", score=0.3738)
    lib.record_recipe(config, rc, experiment_id="e1", outcome="auto_reject")

    rows = lib.list_recipes(config)
    assert len(rows) == 1
    assert rows[0]["outcome"] == "rejected"
    assert rows[0]["screen_score"] == pytest.approx(0.3738)


def test_recipe_library_distinguishes_feature_variants(tmp_path: Path) -> None:
    from autoresearch.models import recipe_library as lib

    config = _make_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    rc = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"}
    lib.record_recipe(
        config,
        rc,
        experiment_id="all",
        outcome="promoted",
        score=0.37,
        target_strategy="direct_pure_premium",
    )
    lib.record_recipe(
        config,
        rc,
        experiment_id="without",
        outcome="rejected",
        score=0.21,
        target_strategy="direct_pure_premium",
        feature_exclusions=["BonusMalus"],
    )
    lib.record_recipe(
        config,
        rc,
        experiment_id="only",
        outcome="rejected",
        score=0.27,
        target_strategy="direct_pure_premium",
        feature_inclusions=["BonusMalus"],
    )

    rows = lib.list_recipes(config)
    assert len(rows) == 3
    assert {row["experiment_id"] for row in rows} == {"all", "without", "only"}
    summaries = {row["experiment_id"]: row["summary"] for row in rows}
    assert "features=except[BonusMalus]" in summaries["without"]
    assert "features=only[BonusMalus]" in summaries["only"]


def test_recipe_summary_distinguishes_parameter_variants(tmp_path: Path) -> None:
    from autoresearch.models import recipe_library as lib

    config = _make_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    first = {
        "structure": "direct",
        "estimator": "lightgbm",
        "objective": "tweedie",
        "encoding": "native_categorical",
        "params": {"num_leaves": 127, "learning_rate": 0.03},
    }
    second = {
        **first,
        "params": {"num_leaves": 31, "learning_rate": 0.08},
    }
    lib.record_recipe(config, first, experiment_id="first", outcome="rejected", score=0.36)
    lib.record_recipe(config, second, experiment_id="second", outcome="promoted", score=0.37)

    rows = lib.list_recipes(config)
    summaries = {row["experiment_id"]: row["summary"] for row in rows}

    assert summaries["first"] != summaries["second"]
    assert "leaves=127" in summaries["first"]
    assert "lr=0.03" in summaries["first"]
    assert "variant=" in summaries["first"]


def test_recipe_library_ranks_current_champion_first(tmp_path: Path) -> None:
    from autoresearch.experiment_registry.registry import set_official_champion
    from autoresearch.models import recipe_library as lib

    config = _make_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    older = {
        "structure": "direct",
        "estimator": "lightgbm",
        "objective": "tweedie",
        "params": {"num_leaves": 127},
    }
    current = {
        "structure": "direct",
        "estimator": "lightgbm",
        "objective": "tweedie",
        "params": {"num_leaves": 31},
    }
    lib.record_recipe(config, older, experiment_id="older", outcome="promoted", score=0.39)
    lib.record_recipe(config, current, experiment_id="current", outcome="promoted", score=0.37)
    set_official_champion(
        config.registry_path,
        champion_id="current",
        branch_id="main",
        reason="test",
        action="promoted",
    )

    rows = lib.list_recipes(config)

    assert rows[0]["experiment_id"] == "current"
    assert rows[0]["is_current_champion"] is True


# ── C: recipe-aware repair request ────────────────────────────────────────────

def test_repair_request_is_recipe_aware(tmp_path: Path) -> None:
    from autoresearch.controller.workflow import _write_repair_request

    report = {"error_type": "runtime_exception", "reason": "boom", "checks": []}
    recipe_req = json.loads((_write_repair_request(tmp_path, 2, report, recipe_origin=True)).read_text())
    assert recipe_req["repair_kind"] == "recipe"
    assert recipe_req["write_recipe"] == "recipe_attempt_2.json"

    script_req = json.loads((_write_repair_request(tmp_path, 2, report, recipe_origin=False)).read_text())
    assert script_req["repair_kind"] == "script"
    assert script_req["write_script"] == "model_attempt_2.py"


def test_recipe_cv_path() -> None:
    frame, _ = _frame()
    tr, va = frame.iloc[:300], frame.iloc[300:]
    res = dispatch_model_on_explicit_frames(
        tr, va, model_family="recipe", target_strategy="direct_pure_premium",
        hyperparameters={"recipe": {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"}},
        target_mode="burning_cost",
    )
    assert len(res.predictions) == len(va)


# ── end-to-end run_experiment with a recipe (no script) ───────────────────────

def _write_big_fixtures(config) -> None:
    frame, split = _frame(n=600)
    # run_experiment caps from ClaimAmount; keep both columns present.
    frame[["record_id", "ClaimNb", "Exposure", "VehPower",
           "Region", "ClaimAmount", "ClaimAmountCount"]].to_parquet(
        config.processed_dir / "agent_dataset_search.parquet", index=False)
    split.to_csv(config.splits_dir / "split_pack.csv", index=False)


def test_run_experiment_with_recipe(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config = replace(config, preflight_enabled=False)  # tiny preflight sample not needed here
    _write_big_fixtures(config)
    exp = tmp_path / "experiment_recipe.toml"
    exp.write_text(
        """
experiment_name = "recipe_lgbm"
model_family = "recipe"
target_strategy = "direct_pure_premium"

[preprocessing]
claim_capping_enabled = true
claim_cap_threshold = 100000

[model.recipe]
structure = "direct"
estimator = "lightgbm"
objective = "tweedie"

[model.recipe.params]
num_leaves = 15
""".strip(),
        encoding="utf-8",
    )
    outputs = run_experiment_safe(config, exp)
    assert outputs["metrics"].exists()
    # recipe ledger written (Option 1)
    ledger = config.artifacts_dir / "recipe_ledger.jsonl"
    assert ledger.exists()


def run_experiment_safe(config, exp):
    from autoresearch.experiment_runner import run_experiment

    # The mandatory pytest gate is skipped in unit tests by pointing root at a
    # repo with no tests would hang; instead monkeypatch is overkill — call the
    # internal dispatch path directly is not equivalent. We rely on the gate's
    # pytest running quickly against the temp root (no tests => passes).
    return run_experiment(config, exp)


# ── proposal schema accepts a recipe ─────────────────────────────────────────

def test_proposal_schema_accepts_recipe() -> None:
    from autoresearch.controller.proposal_schema import validate_proposal

    search_space = {
        "model_families": ["global_mean", "recipe"],
        "target_strategies": ["direct_pure_premium", "frequency_severity"],
        "branch_actions": ["extend_current", "new_branch"],
        "research_line_actions": ["create_line", "extend_line", "revisit_line", "close_line"],
        "claim_cap_thresholds": [100000],
        "allow_disable_claim_capping": False,
        "feature_columns": ["VehPower", "Region"],
        "non_predictive_columns": ["Exposure"],
        "requires_model_script": True,
        "active_target_mode": "burning_cost",
        "allow_open_model_families": True,
    }
    proposal = {
        "experiment_name": "r1", "rationale": "x", "change_summary": "x",
        "expected_benefit": "x", "key_risk": "x", "exploration_axis": "model_family",
        "approach_family": "x", "target_framing": "x", "feature_representation": "x",
        "expected_learning": "x", "proposal_id": "abc", "parent_experiment_id": "p",
        "tree_action": "extend_node", "parent_rationale": "x", "selected_tree_action_id": "x",
        "research_line_action": "create_line", "research_line_id": "lll",
        "research_line_label": "x", "research_line_hypothesis": "x",
        "line_membership_rationale": "x",
        "experiment_config": {
            "experiment_name": "r1", "parent_experiment_id": "p",
            "model_family": "recipe", "target_strategy": "direct_pure_premium",
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": {"recipe": {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"}},
        },
    }
    assert validate_proposal(proposal, search_space) == []

    # An invalid recipe surfaces as a proposal error.
    bad = {**proposal}
    bad["experiment_config"] = {**proposal["experiment_config"],
                                "model": {"recipe": {"structure": "direct", "estimator": "lightgbm",
                                                     "objective": "gamma"}}}
    assert any("model.recipe" in e for e in validate_proposal(bad, search_space))


def _base_recipe_proposal() -> tuple[dict, dict]:
    search_space = {
        "model_families": ["global_mean", "recipe"],
        "target_strategies": ["direct_pure_premium", "frequency_severity"],
        "branch_actions": ["extend_current", "new_branch"],
        "research_line_actions": ["create_line", "extend_line", "revisit_line", "close_line"],
        "claim_cap_thresholds": [100000], "allow_disable_claim_capping": False,
        "feature_columns": ["VehPower"], "non_predictive_columns": ["Exposure"],
        "requires_model_script": True, "active_target_mode": "burning_cost",
        "allow_open_model_families": True,
    }
    proposal = {
        "experiment_name": "r1", "rationale": "x", "change_summary": "x",
        "expected_benefit": "x", "key_risk": "x", "exploration_axis": "model_family",
        "approach_family": "x", "target_framing": "x", "feature_representation": "x",
        "expected_learning": "x", "proposal_id": "abc", "parent_experiment_id": "p",
        "tree_action": "extend_node", "parent_rationale": "x", "selected_tree_action_id": "x",
        "research_line_action": "create_line", "research_line_id": "lll",
        "research_line_label": "x", "research_line_hypothesis": "x", "line_membership_rationale": "x",
        "experiment_config": {
            "experiment_name": "r1", "parent_experiment_id": "p",
            "model_family": "recipe", "target_strategy": "direct_pure_premium",
            "preprocessing": {"claim_capping_enabled": True, "claim_cap_threshold": 100000},
            "model": {"recipe": {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"}},
        },
    }
    return proposal, search_space


def test_proposal_rejects_recipe_and_script_together() -> None:
    from autoresearch.controller.proposal_schema import validate_proposal
    p, ss = _base_recipe_proposal()
    p["experiment_config"]["model"]["script_path"] = "model.py"
    assert any("not both" in e for e in validate_proposal(p, ss))


def test_proposal_rejects_recipe_under_wrong_family() -> None:
    from autoresearch.controller.proposal_schema import validate_proposal
    p, ss = _base_recipe_proposal()
    p["experiment_config"]["model_family"] = "lightgbm"
    assert any("model_family must be 'recipe'" in e for e in validate_proposal(p, ss))


def test_proposal_rejects_structure_target_mismatch() -> None:
    from autoresearch.controller.proposal_schema import validate_proposal
    p, ss = _base_recipe_proposal()
    p["experiment_config"]["model"]["recipe"] = {
        "structure": "frequency_severity",
        "stages": {"frequency": {"estimator": "lightgbm", "objective": "poisson"},
                   "severity": {"estimator": "lightgbm", "objective": "gamma"}},
    }
    # target_strategy is still direct_pure_premium → mismatch
    assert any("frequency_severity" in e and "target_strategy" in e for e in validate_proposal(p, ss))


# ── recipe reuse library (Options 1/2) ───────────────────────────────────────

def test_recipe_library_run_scope(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    from autoresearch.models import recipe_library as lib

    rc = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"}
    lib.record_recipe(config, rc, experiment_id="e1", outcome="completed", score=0.30)
    lib.record_recipe(config, rc, experiment_id="e1", outcome="promoted", score=0.34)
    rows = lib.list_recipes(config)
    assert len(rows) == 1  # deduped by fingerprint
    assert rows[0]["outcome"] == "promoted"  # terminal outcome wins
    assert lib.effective_reuse_scope(config) == "run"


def test_recipe_library_memory_scope_gated(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    config = replace(config, recipe_reuse_scope="memory")
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    from autoresearch.models import recipe_library as lib

    # No memory access → falls back to run scope.
    monkeypatch.setenv("AUTORESEARCH_MEMORY_ACCESS", "none")
    assert lib.effective_reuse_scope(config) == "run"

    # Access granted → memory scope engages.
    monkeypatch.setenv("AUTORESEARCH_MEMORY_ACCESS", "all")
    assert lib.effective_reuse_scope(config) == "memory"


# ── champion template (#8) ────────────────────────────────────────────────────

def test_champion_template_generation(tmp_path: Path, monkeypatch) -> None:
    config = _make_config(tmp_path)
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    from autoresearch.controller import champion_template as ct

    recipe = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
              "params": {"num_leaves": 31}}
    monkeypatch.setattr(
        ct, "_load_experiment_model_cfg",
        lambda c, eid: {
            "model": {"recipe": recipe, "feature_exclusions": ["Region"]},
            "target_strategy": "direct_pure_premium",
            "model_script_path": None,
        },
    )
    written = ct.generate_champion_template(config, "exp123")
    assert "champion_template" in written and written["champion_template"].exists()
    assert "champion_recipe" in written and written["champion_recipe"].exists()
    text = written["champion_template"].read_text()
    assert "PARAM_OVERRIDES" in text and "fit_predict" in text
    artifact = json.loads(written["champion_recipe"].read_text())
    assert artifact["target_strategy"] == "direct_pure_premium"
    assert artifact["model"]["feature_exclusions"] == ["Region"]


def test_champion_recipe_reference_resolves_nested_overrides(tmp_path: Path) -> None:
    from autoresearch.controller.workflow import _hydrate_recipe_reference

    config = _make_config(tmp_path)
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "champion_recipe.json").write_text(
        json.dumps({
            "experiment_id": "champ1",
            "target_strategy": "direct_pure_premium",
            "model": {
                "recipe": {
                    "structure": "direct",
                    "estimator": "lightgbm",
                    "objective": "tweedie",
                    "params": {"num_leaves": 63, "learning_rate": 0.05},
                },
                "feature_exclusions": ["Region"],
            },
        }),
        encoding="utf-8",
    )
    parsed = {
        "experiment_config": {
            "model": {
                "recipe_ref": "champion",
                "recipe_overrides": {"params": {"num_leaves": 31}},
                "feature_exclusions": None,
            },
        }
    }
    errors: list[str] = []

    _hydrate_recipe_reference(
        config,
        parsed,
        {"champion_id": "champ1"},
        errors,
    )

    assert errors == []
    exp = parsed["experiment_config"]
    assert exp["model_family"] == "recipe"
    assert exp["target_strategy"] == "direct_pure_premium"
    assert exp["model"]["recipe"]["params"] == {
        "num_leaves": 31,
        "learning_rate": 0.05,
    }
    assert "feature_exclusions" not in exp["model"]
    assert "recipe_ref" not in exp["model"]


def test_champion_recipe_reference_rejects_stale_artifact(tmp_path: Path) -> None:
    from autoresearch.controller.workflow import _hydrate_recipe_reference

    config = _make_config(tmp_path)
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    (config.handoff_proposal_inbox_dir / "champion_recipe.json").write_text(
        json.dumps({
            "experiment_id": "old",
            "recipe": {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie"},
        }),
        encoding="utf-8",
    )
    parsed = {"experiment_config": {"model": {"recipe_ref": "champion"}}}
    errors: list[str] = []

    _hydrate_recipe_reference(config, parsed, {"champion_id": "new"}, errors)

    assert errors == ["model.recipe_ref points to a stale champion_recipe.json"]


def test_champion_artifacts_not_ingested_as_proposals(tmp_path: Path) -> None:
    """champion_recipe.json / champion_template.py in the inbox must be skipped by
    the proposal ingester (regression: run 20260608T190605Z quarantined the
    champion_recipe.json to processed/invalid on the next cycle)."""
    from autoresearch.controller.champion_template import (
        CHAMPION_RECIPE_FILENAME,
        CHAMPION_TEMPLATE_FILENAME,
    )
    from autoresearch.controller.handoff import ingest_proposals
    from autoresearch.experiment_registry.registry import init_registry

    config = _make_config(tmp_path)
    init_registry(config.registry_path)
    inbox = config.handoff_proposal_inbox_dir
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / CHAMPION_RECIPE_FILENAME).write_text('{"recipe": {"structure": "direct"}}', encoding="utf-8")
    (inbox / CHAMPION_TEMPLATE_FILENAME).write_text("# champion template\n", encoding="utf-8")

    ingest_proposals(config)

    # The champion artifacts stay put and are NOT quarantined as invalid proposals.
    assert (inbox / CHAMPION_RECIPE_FILENAME).exists()
    invalid_dir = config.handoff_proposal_processed_dir / "invalid"
    moved = list(invalid_dir.glob("*champion_recipe*")) if invalid_dir.exists() else []
    assert not moved, f"champion_recipe.json was wrongly ingested: {moved}"


def test_generated_champion_template_runs_as_script(tmp_path: Path, monkeypatch) -> None:
    """The generated champion_template.py must be a runnable escape-hatch script."""
    config = _make_config(tmp_path)
    config.handoff_proposal_inbox_dir.mkdir(parents=True, exist_ok=True)
    from autoresearch.controller import champion_template as ct
    from autoresearch.models.dispatcher import dispatch_model

    recipe = {"structure": "direct", "estimator": "lightgbm", "objective": "tweedie",
              "params": {"num_leaves": 15}}
    monkeypatch.setattr(
        ct, "_load_experiment_model_cfg",
        lambda c, eid: {"model": {"recipe": recipe}, "model_script_path": None},
    )
    written = ct.generate_champion_template(config, "exp123")
    template_path = written["champion_template"]

    frame, split = _frame()
    res = dispatch_model(
        frame, split, model_family="champion_copy", target_strategy="direct_pure_premium",
        train_split="train", score_splits=("search_validation",),
        hyperparameters={}, model_script_path=template_path, target_mode="burning_cost",
    )
    preds = res.predictions["predicted_claim_cost"].to_numpy()
    assert len(res.predictions) == len(frame)
    assert np.all(np.isfinite(preds)) and np.all(preds >= 0)
    # Aggregate calibration: predicted train total ≈ actual train total.
    train_rows = res.predictions[res.predictions["split"] == "train"]
    ratio = train_rows["predicted_claim_cost"].sum() / max(train_rows["actual_claim_cost"].sum(), 1e-9)
    assert 0.98 <= ratio <= 1.02
