from pathlib import Path

from autoresearch.experiment_registry.registry import init_registry, registry_counts


def test_init_registry_creates_tables(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.sqlite"

    init_registry(registry_path)

    assert registry_path.exists()
    assert registry_counts(registry_path) == {
        "experiments": 0,
        "artifacts": 0,
        "comparisons": 0,
        "proposals": 0,
        "branches": 0,
        "sessions": 0,
        "research_nodes": 0,
    }
