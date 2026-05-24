import pandas as pd
import pytest

from autoresearch.data.splits import generate_split_pack, stable_unit, validate_split_ratios


def test_stable_unit_is_reproducible() -> None:
    assert stable_unit(123, 42) == stable_unit(123, 42)
    assert stable_unit(123, 42) != stable_unit(123, 43)


def test_generate_split_pack_is_reproducible_and_uses_expected_splits() -> None:
    frame = pd.DataFrame({"IDpol": range(100)})
    ratios = {
        "train": 0.64,
        "search_validation": 0.16,
        "milestone_holdout": 0.2,
    }

    first, first_manifest = generate_split_pack(frame, "IDpol", ratios, seed=7)
    second, second_manifest = generate_split_pack(frame, "IDpol", ratios, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first_manifest == second_manifest
    assert set(first["split"]).issubset(set(ratios))
    assert first_manifest["holdout_policy"].startswith("milestone_holdout")
    assert first_manifest["milestone_holdout_ratio"] == 0.2


def test_validate_split_ratios_rejects_bad_total() -> None:
    with pytest.raises(ValueError):
        validate_split_ratios(
            {
                "train": 0.6,
                "search_validation": 0.2,
                "milestone_holdout": 0.3,
            }
        )
