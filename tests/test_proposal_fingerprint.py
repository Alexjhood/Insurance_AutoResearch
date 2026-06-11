"""Guards for duplicate-detection fingerprinting of proposals."""

from __future__ import annotations

from autoresearch.controller.proposal_schema import proposal_fingerprint


def test_nested_param_float_jitter_collapses():
    """Float jitter inside model.params must not evade duplicate detection.

    The fingerprint rounds floats to 6 dp; before the recursive fix only
    top-level ``model`` floats were rounded, so jittered nested params slipped
    past dedup.
    """
    a = {"config": {"model": {"params": {"learning_rate": 0.05000000001, "num_leaves": 63}}}}
    b = {"config": {"model": {"params": {"learning_rate": 0.05, "num_leaves": 63}}}}
    assert proposal_fingerprint(a) == proposal_fingerprint(b)


def test_meaningful_param_difference_still_distinct():
    """Genuinely different params must produce different fingerprints."""
    a = {"config": {"model": {"params": {"learning_rate": 0.05}}}}
    b = {"config": {"model": {"params": {"learning_rate": 0.10}}}}
    assert proposal_fingerprint(a) != proposal_fingerprint(b)
