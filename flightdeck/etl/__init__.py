"""Flight Deck ETL — offline snapshot builder.

Public entry points::

    python -m flightdeck.etl --orchestration <id>
    python -m flightdeck.etl --all

or programmatically via :func:`flightdeck.etl.build.build`.
"""

from .build import build, build_orchestration, discover_orchestrations, main

__all__ = ["build", "build_orchestration", "discover_orchestrations", "main"]
