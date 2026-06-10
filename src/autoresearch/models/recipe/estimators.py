"""Built-in estimator building blocks (curated registry).

Each ``fit`` takes a :class:`FitContext` and returns ``(predictor, notes)`` where
``predictor`` exposes ``predict(X) -> np.ndarray`` of non-negative **rate**
predictions (per unit exposure). Stages are fit on the target *rate* with
``sample_weight`` = exposure (frequency/severity use their own weights), so the
framework can convert rate → total and calibrate uniformly downstream (#7).

Adding a new estimator is a single ``register_estimator`` call — no other module
needs to change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autoresearch.models.recipe.registry import (
    EstimatorSpec,
    FitContext,
    RecipeError,
    register_estimator,
)


def _params(ctx: FitContext) -> dict[str, Any]:
    return dict(ctx.params or {})


def _pop_conflicts(params: dict[str, Any], *names: str) -> None:
    """Drop keys the builder sets explicitly, so a user-supplied param of the same
    name cannot collide with an explicit keyword argument (``got multiple values``).
    """
    for name in names:
        params.pop(name, None)


# ── LightGBM ─────────────────────────────────────────────────────────────────

_LGBM_OBJ = {"tweedie": "tweedie", "poisson": "poisson", "gamma": "gamma", "squared_error": "regression"}


def _fit_lightgbm(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    import lightgbm as lgb

    params = _params(ctx)
    _pop_conflicts(params, "objective")
    objective = _LGBM_OBJ[ctx.objective]
    if ctx.objective == "tweedie":
        params.setdefault("tweedie_variance_power", 1.5)
    params.setdefault("n_estimators", 500 if ctx.early_stopping else 200)
    params.setdefault("verbosity", -1)
    model = lgb.LGBMRegressor(objective=objective, **params)

    fit_kwargs: dict[str, Any] = {"sample_weight": ctx.w_train}
    if ctx.categorical_features:
        fit_kwargs["categorical_feature"] = ctx.categorical_features
    if ctx.early_stopping and ctx.X_val is not None:
        fit_kwargs["eval_set"] = [(ctx.X_val, ctx.y_val)]
        fit_kwargs["eval_sample_weight"] = [ctx.w_val]
        fit_kwargs["callbacks"] = [lgb.early_stopping(int(ctx.early_stopping), verbose=False),
                                   lgb.log_evaluation(0)]
    model.fit(ctx.X_train, ctx.y_train, **fit_kwargs)
    notes = {"estimator": "lightgbm", "objective": ctx.objective,
             "best_iteration": getattr(model, "best_iteration_", None)}
    return model, notes


# ── XGBoost ──────────────────────────────────────────────────────────────────

_XGB_OBJ = {"tweedie": "reg:tweedie", "poisson": "count:poisson",
            "gamma": "reg:gamma", "squared_error": "reg:squarederror"}


def _fit_xgboost(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    import xgboost as xgb

    params = _params(ctx)
    _pop_conflicts(params, "objective")
    params.setdefault("n_estimators", 500 if ctx.early_stopping else 200)
    if ctx.objective == "tweedie":
        params.setdefault("tweedie_variance_power", 1.5)
    use_es = bool(ctx.early_stopping and ctx.X_val is not None)
    if use_es:
        params["early_stopping_rounds"] = int(ctx.early_stopping)
    model = xgb.XGBRegressor(objective=_XGB_OBJ[ctx.objective], **params)

    fit_kwargs: dict[str, Any] = {"sample_weight": ctx.w_train, "verbose": False}
    if use_es:
        fit_kwargs["eval_set"] = [(ctx.X_val, ctx.y_val)]
        fit_kwargs["sample_weight_eval_set"] = [ctx.w_val]
    model.fit(ctx.X_train, ctx.y_train, **fit_kwargs)
    # best_iteration is only meaningful when early stopping ran.
    best_it = getattr(model, "best_iteration", None) if use_es else None
    notes = {"estimator": "xgboost", "objective": ctx.objective, "best_iteration": best_it}
    return model, notes


# ── sklearn HistGradientBoosting ─────────────────────────────────────────────

_HGB_LOSS = {"poisson": "poisson", "gamma": "gamma", "squared_error": "squared_error"}


def _fit_hist_gbm(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = _params(ctx)
    _pop_conflicts(params, "loss")
    if ctx.early_stopping:
        params.setdefault("early_stopping", True)
        params.setdefault("n_iter_no_change", int(ctx.early_stopping))
    model = HistGradientBoostingRegressor(loss=_HGB_LOSS[ctx.objective], **params)
    # HistGBR manages its own internal validation split for early stopping.
    model.fit(ctx.X_train, ctx.y_train, sample_weight=ctx.w_train)
    notes = {"estimator": "hist_gbm", "objective": ctx.objective,
             "n_iter": getattr(model, "n_iter_", None)}
    return model, notes


# ── sklearn TweedieRegressor (GLM) ───────────────────────────────────────────

def _fit_tweedie_glm(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    from sklearn.linear_model import TweedieRegressor

    params = _params(ctx)
    # Accept the sklearn name (`power`) or the GBM-style alias
    # (`tweedie_variance_power`); pop both so neither collides with the explicit
    # `power=` keyword below. poisson/gamma objectives pin the power.
    explicit_power = params.pop("power", None)
    explicit_power = params.pop("tweedie_variance_power", explicit_power)
    power = {"poisson": 1.0, "gamma": 2.0}.get(
        ctx.objective, float(explicit_power) if explicit_power is not None else 1.5
    )
    params.setdefault("max_iter", 10000)
    model = TweedieRegressor(power=power, **params)
    model.fit(ctx.X_train, ctx.y_train, sample_weight=ctx.w_train)
    notes = {"estimator": "tweedie_glm", "objective": ctx.objective, "power": power}
    return model, notes


# ── sklearn ElasticNet ───────────────────────────────────────────────────────

def _fit_elasticnet(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    from sklearn.linear_model import ElasticNet

    params = _params(ctx)
    params.setdefault("alpha", 0.1)
    params.setdefault("l1_ratio", 0.5)
    params.setdefault("max_iter", 10000)
    model = ElasticNet(**params)
    model.fit(ctx.X_train, ctx.y_train, sample_weight=ctx.w_train)
    notes = {"estimator": "elasticnet", "objective": ctx.objective}
    return model, notes


# ── Constant baseline ────────────────────────────────────────────────────────

class _Constant:
    def __init__(self, rate: float) -> None:
        self.rate = float(rate)

    def predict(self, X: Any) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.rate, dtype=float)


def _fit_constant(ctx: FitContext) -> tuple[Any, dict[str, Any]]:
    y = np.asarray(ctx.y_train, dtype=float)
    w = np.asarray(ctx.w_train, dtype=float)
    total_w = float(w.sum())
    if total_w <= 0:
        raise RecipeError("constant estimator requires positive total weight")
    rate = float((y * w).sum() / total_w)
    return _Constant(rate), {"estimator": "constant", "rate": rate}


_GBM_TREE_PARAMS = frozenset({
    "n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree",
    "reg_alpha", "reg_lambda", "random_state", "n_jobs", "min_child_weight",
})


def register_builtin_estimators() -> None:
    register_estimator(EstimatorSpec(
        name="lightgbm",
        objectives=frozenset({"tweedie", "poisson", "gamma", "squared_error"}),
        encodings=frozenset({"native_categorical", "one_hot", "ordinal"}),
        default_encoding="native_categorical",
        fit=_fit_lightgbm,
        supports_early_stopping=True,
        native_categorical=True,
        description="Gradient-boosted trees with native categorical support and Tweedie loss.",
        allowed_params=_GBM_TREE_PARAMS | frozenset({
            "num_leaves", "min_child_samples", "min_split_gain", "subsample_freq",
            "colsample_bynode", "max_bin", "min_data_in_leaf", "tweedie_variance_power",
            "verbosity", "force_col_wise",
        }),
    ))
    register_estimator(EstimatorSpec(
        name="xgboost",
        objectives=frozenset({"tweedie", "poisson", "gamma", "squared_error"}),
        encodings=frozenset({"one_hot", "ordinal"}),
        default_encoding="ordinal",
        fit=_fit_xgboost,
        supports_early_stopping=True,
        native_categorical=False,
        description="Gradient-boosted trees; needs encoded categoricals.",
        allowed_params=_GBM_TREE_PARAMS | frozenset({
            "gamma", "colsample_bylevel", "colsample_bynode", "max_bin", "max_leaves",
            "grow_policy", "tree_method", "min_split_loss", "tweedie_variance_power",
        }),
    ))
    register_estimator(EstimatorSpec(
        name="hist_gbm",
        objectives=frozenset({"poisson", "gamma", "squared_error"}),
        encodings=frozenset({"one_hot", "ordinal"}),
        default_encoding="ordinal",
        fit=_fit_hist_gbm,
        supports_early_stopping=True,
        self_validates_early_stopping=True,
        native_categorical=False,
        description="sklearn histogram GBM (no Tweedie loss).",
        allowed_params=frozenset({
            "learning_rate", "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf",
            "l2_regularization", "max_bins", "early_stopping", "n_iter_no_change",
            "validation_fraction", "max_features", "random_state",
        }),
    ))
    register_estimator(EstimatorSpec(
        name="tweedie_glm",
        objectives=frozenset({"tweedie", "poisson", "gamma"}),
        encodings=frozenset({"one_hot"}),
        default_encoding="one_hot",
        fit=_fit_tweedie_glm,
        supports_early_stopping=False,
        native_categorical=False,
        description="Interpretable Tweedie/Poisson/Gamma GLM (sklearn TweedieRegressor).",
        allowed_params=frozenset({
            "alpha", "max_iter", "tol", "power", "tweedie_variance_power",
            "fit_intercept", "solver", "warm_start", "link",
        }),
        param_conflicts=(("power", "tweedie_variance_power"),),
    ))
    register_estimator(EstimatorSpec(
        name="elasticnet",
        objectives=frozenset({"squared_error"}),
        encodings=frozenset({"one_hot"}),
        default_encoding="one_hot",
        fit=_fit_elasticnet,
        supports_early_stopping=False,
        native_categorical=False,
        description="Sparse linear regression with L1/L2 penalty.",
        allowed_params=frozenset({
            "alpha", "l1_ratio", "max_iter", "tol", "fit_intercept",
            "selection", "positive", "warm_start", "random_state",
        }),
    ))
    register_estimator(EstimatorSpec(
        name="constant",
        objectives=frozenset({"squared_error", "tweedie", "poisson", "gamma"}),
        encodings=frozenset({"native_categorical", "one_hot", "ordinal"}),
        default_encoding="native_categorical",
        fit=_fit_constant,
        supports_early_stopping=False,
        native_categorical=True,
        description="Weighted-mean flat-rate baseline (ignores features).",
    ))
