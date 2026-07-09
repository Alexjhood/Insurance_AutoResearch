"""French Motor (freMTPL2) loader adapter.

Wraps the historical two-file frequency/severity discovery + aggregation
(``autoresearch.data.loader.load_fremtpl2``) into the uniform
:class:`~autoresearch.data.sources.RawDataset` contract, recording the source
paths under ``provenance``.
"""

from __future__ import annotations

from typing import Any

from autoresearch.data.loader import load_fremtpl2
from autoresearch.data.sources import RawDataset


def load(spec: Any) -> RawDataset:
    loaded = load_fremtpl2(spec.raw_dir, id_column=spec.id_column)
    return RawDataset(
        frame=loaded.frame,
        provenance={
            "loader": "adapter:french_motor",
            "frequency_path": str(loaded.frequency_path),
            "severity_path": str(loaded.severity_path) if loaded.severity_path else None,
        },
    )
