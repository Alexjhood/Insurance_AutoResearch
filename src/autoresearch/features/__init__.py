"""Feature engineering modules.

Each module in this package must expose::

    def build_features(frame: pd.DataFrame) -> pd.DataFrame:
        ...

The function receives the full modelling frame (train or score partition),
must return a DataFrame with at least all the original columns, and must
be pure (no side effects, no I/O, no access to the holdout vault).

New modules are discovered automatically by the model dispatcher when
``feature_builder_module`` is set in an experiment TOML.
"""
