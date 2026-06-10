"""Built-in encoding building blocks.

Each encoding turns the selected numeric + categorical columns into model inputs.
``native_categorical`` is special: it has no transformer because the estimator
(LightGBM) consumes a category-typed DataFrame directly.
"""

from __future__ import annotations

from autoresearch.models.recipe.registry import EncodingSpec, register_encoding


def _one_hot(numeric: list[str], categorical: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]),
            categorical,
        ))
    # sparse_threshold>0 lets the high-cardinality one-hot block stay sparse;
    # the sklearn linear estimators (ElasticNet/TweedieRegressor) accept sparse X.
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)


def _ordinal(numeric: list[str], categorical: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def register_builtin_encodings() -> None:
    register_encoding(EncodingSpec(
        name="native_categorical",
        builder=None,
        native=True,
        description="Pass category-typed columns straight to a gradient-boosted tree (LightGBM).",
    ))
    register_encoding(EncodingSpec(
        name="one_hot",
        builder=_one_hot,
        description="One column per level (+ median-impute & standard-scale numerics). Linear models.",
    ))
    register_encoding(EncodingSpec(
        name="ordinal",
        builder=_ordinal,
        description="Integer code per level (median-impute numerics). Tree models.",
    ))
