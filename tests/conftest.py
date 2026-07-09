"""Shared test fixtures.

The multi-dataset support binds the active dataset into process-global state at
config load (``targets.bind_dataset``, ``models.columns.bind``,
``feature_policy.bind``). A test that calls ``load_config(dataset=...)`` would
otherwise leave that dataset bound for every later test, making the suite
test-order dependent. This autouse fixture snapshots the bound state before each
test and restores it after, so each test starts from whatever the module
defaults are (French) regardless of what ran before it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _restore_dataset_binding():
    import autoresearch.feature_policy as feature_policy
    import autoresearch.targets as targets
    from autoresearch.models import columns

    targets_state = (targets._ACTIVE_SPECS, targets._ACTIVE_MODES)
    columns_state = dict(columns._STATE)
    columns_consts = {
        name: getattr(columns, name)
        for name in ("EXPOSURE", "CLAIM_COUNT", "CLAIM_EVENTS", "CLAIM_COST", "RAW_CLAIM_COST")
    }
    policy_state = (feature_policy.NON_PREDICTIVE_COLUMNS, feature_policy.EXPOSURE_COLUMN)

    yield

    targets._ACTIVE_SPECS, targets._ACTIVE_MODES = targets_state
    columns._STATE.clear()
    columns._STATE.update(columns_state)
    for name, value in columns_consts.items():
        setattr(columns, name, value)
    columns._sync_exports()
    feature_policy.NON_PREDICTIVE_COLUMNS, feature_policy.EXPOSURE_COLUMN = policy_state
