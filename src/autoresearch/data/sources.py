"""Raw-dataset loading dispatch.

A :class:`RawDataset` is the uniform return of every loader: a modelling frame
keyed by the dataset's id column, plus a free-form ``provenance`` dict recorded
into the dataset profile. :func:`load_raw` routes to the ``single_table`` loader
or a named ``adapter`` module per the :class:`~autoresearch.datasets.DatasetSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any

import pandas as pd

from autoresearch.datasets import DatasetSpec


@dataclass(frozen=True)
class RawDataset:
    """Loaded raw dataset plus provenance needed for metadata."""

    frame: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)


def load_raw(spec: DatasetSpec, *, nrows: int | None = None) -> RawDataset:
    """Load a dataset's raw modelling frame via its configured loader.

    ``nrows`` (single-table only) caps the read for fast iteration/testing.
    """

    if spec.loader == "adapter":
        if not spec.adapter_module:
            raise ValueError(f"Dataset {spec.name!r} uses loader='adapter' but sets no adapter_module")
        module = importlib.import_module(spec.adapter_module)
        if not hasattr(module, "load"):
            raise ValueError(f"Adapter module {spec.adapter_module!r} must expose load(spec)")
        return module.load(spec)
    if spec.loader == "single_table":
        from autoresearch.data.loaders.single_table import load as load_single_table

        return load_single_table(spec, nrows=nrows)
    raise ValueError(f"Dataset {spec.name!r} has unknown loader {spec.loader!r}")
