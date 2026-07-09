"""Dataset registry: per-dataset configuration loaded from ``configs/datasets``.

A :class:`DatasetSpec` captures everything that used to be hard-coded French
freMTPL2 assumptions — column roles, the loader, missing-value handling, the
split unit, capping, target modes, and reporting labels — so the same research
loop can run against any registered tabular regression dataset.

New datasets are added by dropping a ``configs/datasets/<name>.toml`` file (plus,
only when the raw shape needs it, a small loader adapter). No framework code
change is required.

The concrete generic ``TargetSpec`` objects are built lazily from the raw target
tables (see :func:`autoresearch.targets.build_target_spec`); this module keeps
the raw target configuration so it has no import dependency on target-key
generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import tomllib
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_CONFIG_DIR = PROJECT_ROOT / "configs" / "datasets"
DATASETS_DATA_ROOT = PROJECT_ROOT / "data" / "datasets"

_MODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class NaMarker:
    """A sentinel value that means "missing" within selected columns."""

    value: Any
    columns_matching: str | None = None  # regex matched against column names; None → all


@dataclass(frozen=True)
class DerivedColumn:
    """A column computed from existing columns via a restricted expression."""

    name: str
    expr: str  # evaluated with pandas.eval over existing columns only


@dataclass(frozen=True)
class SampleSpec:
    """Deterministic ingestion-time subsample specification."""

    rows: int
    unit_column: str          # sample whole units (e.g. households) — no leakage
    stratify_column: str | None
    seed: int


@dataclass(frozen=True)
class CapSpec:
    """Fixed capping rule applied to a source column."""

    column: str
    output_column: str
    threshold: float
    fixed: bool = True


@dataclass(frozen=True)
class ReportingSpec:
    """Cosmetic reporting labels for a dataset."""

    currency: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    """Frozen description of a registered dataset.

    Path fields (``raw_dir``, ``data_dir``) are resolved absolute paths. Column
    fields carry ``None`` when the dataset lacks that concept (e.g. no exposure
    weight, no claim count).
    """

    name: str
    display_name: str
    default_target_mode: str
    # source / loading
    loader: str                       # "single_table" | "adapter"
    adapter_module: str | None
    raw_dir: Path
    train_file_pattern: str | None
    id_column: str
    na_values: tuple[str, ...]
    na_marker: NaMarker | None
    derived_columns: tuple[DerivedColumn, ...]
    sample: SampleSpec | None
    # columns
    weight_column: str | None         # None → framework synthesises unit weight
    count_column: str | None          # None → frequency_severity structure illegal
    event_count_column: str | None
    non_predictive: frozenset[str]
    # splits
    split_unit_column: str
    stratify_target: str | None
    stratify_weight: str | None
    stratify_bands: tuple[float, ...] | None
    # preprocessing
    cap: CapSpec | None
    # targets (raw config; concrete TargetSpecs are built in targets.build_target_spec)
    target_configs: tuple[dict[str, Any], ...]
    # memory / reporting / overrides
    structural_gini_threshold: float | None
    reporting: ReportingSpec
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    data_dir: Path = DATASETS_DATA_ROOT

    # ---- derived views -------------------------------------------------

    @property
    def target_modes(self) -> tuple[str, ...]:
        return tuple(str(t["mode"]) for t in self.target_configs)

    @property
    def weight_display_name(self) -> str:
        return self.weight_column or "unit"

    @property
    def synthesises_unit_weight(self) -> bool:
        return self.weight_column is None

    def target_config(self, mode: str) -> dict[str, Any]:
        for t in self.target_configs:
            if str(t["mode"]) == mode:
                return dict(t)
        raise KeyError(
            f"Dataset {self.name!r} has no target mode {mode!r}; "
            f"available: {list(self.target_modes)}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def dataset_config_path(name: str) -> Path:
    return DATASETS_CONFIG_DIR / f"{name}.toml"


def dataset_data_dir(name: str) -> Path:
    """Return the per-dataset data root ``data/datasets/<name>``."""

    return DATASETS_DATA_ROOT / name


def list_datasets() -> list[str]:
    """Return the names of every registered dataset (scans configs/datasets)."""

    if not DATASETS_CONFIG_DIR.exists():
        return []
    return sorted(p.stem for p in DATASETS_CONFIG_DIR.glob("*.toml"))


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_na_marker(raw: dict[str, Any] | None) -> NaMarker | None:
    if not raw:
        return None
    return NaMarker(value=raw["value"], columns_matching=raw.get("columns_matching"))


def _parse_sample(raw: dict[str, Any] | None) -> SampleSpec | None:
    if not raw:
        return None
    return SampleSpec(
        rows=int(raw["rows"]),
        unit_column=str(raw["unit_column"]),
        stratify_column=(str(raw["stratify_column"]) if raw.get("stratify_column") else None),
        seed=int(raw["seed"]),
    )


def _parse_cap(raw: dict[str, Any] | None) -> CapSpec | None:
    if not raw:
        return None
    return CapSpec(
        column=str(raw["column"]),
        output_column=str(raw["output_column"]),
        threshold=float(raw["threshold"]),
        fixed=bool(raw.get("fixed", True)),
    )


def _parse_derived(raw: list[dict[str, Any]] | None) -> tuple[DerivedColumn, ...]:
    if not raw:
        return ()
    return tuple(DerivedColumn(name=str(d["name"]), expr=str(d["expr"])) for d in raw)


def load_dataset_spec(name: str) -> DatasetSpec:
    """Load and validate a dataset spec from ``configs/datasets/<name>.toml``."""

    path = dataset_config_path(name)
    if not path.exists():
        available = ", ".join(list_datasets()) or "(none)"
        raise FileNotFoundError(
            f"Unknown dataset {name!r}: no config at {path}. Registered datasets: {available}"
        )
    with path.open("rb") as f:
        raw = tomllib.load(f)

    dataset = raw.get("dataset", {})
    source = raw.get("source", {})
    columns = raw.get("columns", {})
    splits = raw.get("splits", {})
    preprocessing = raw.get("preprocessing", {})
    targets = raw.get("targets", [])
    memory = raw.get("memory", {})
    reporting = raw.get("reporting", {})
    overrides = raw.get("overrides", {})

    if str(dataset.get("name", name)) != name:
        raise ValueError(
            f"Dataset config {path} declares name={dataset.get('name')!r} "
            f"but the file is {name}.toml"
        )

    raw_dir_value = source.get("raw_dir") or str(dataset_data_dir(name) / "raw")

    spec = DatasetSpec(
        name=name,
        display_name=str(dataset.get("display_name", name)),
        default_target_mode=str(dataset["default_target_mode"]),
        loader=str(source.get("loader", "single_table")),
        adapter_module=(str(source["adapter_module"]) if source.get("adapter_module") else None),
        raw_dir=_resolve_path(raw_dir_value),
        train_file_pattern=(str(source["train_file_pattern"]) if source.get("train_file_pattern") else None),
        id_column=str(source["id_column"]),
        na_values=tuple(str(v) for v in source.get("na_values", [])),
        na_marker=_parse_na_marker(source.get("na_marker")),
        derived_columns=_parse_derived(columns.get("derived")),
        sample=_parse_sample(source.get("sample")),
        weight_column=(str(columns["weight"]) if columns.get("weight") else None),
        count_column=(str(columns["count"]) if columns.get("count") else None),
        event_count_column=(str(columns["event_count"]) if columns.get("event_count") else None),
        non_predictive=frozenset(str(c) for c in columns.get("non_predictive", [])),
        split_unit_column=str(splits.get("unit_column", "record_id")),
        stratify_target=(str(splits["stratify_target"]) if splits.get("stratify_target") else None),
        stratify_weight=(str(splits["stratify_weight"]) if splits.get("stratify_weight") else None),
        stratify_bands=(tuple(float(b) for b in splits["bands"]) if splits.get("bands") else None),
        cap=_parse_cap(preprocessing.get("cap")),
        target_configs=tuple(dict(t) for t in targets),
        structural_gini_threshold=(
            float(memory["structural_gini_threshold"])
            if memory.get("structural_gini_threshold") is not None
            else None
        ),
        reporting=ReportingSpec(currency=str(reporting.get("currency", ""))),
        overrides={str(k): dict(v) for k, v in overrides.items()},
        data_dir=dataset_data_dir(name),
    )
    _validate_spec(spec, path)
    return spec


def _validate_spec(spec: DatasetSpec, path: Path) -> None:
    if not spec.target_configs:
        raise ValueError(f"Dataset config {path} defines no [[targets]]")
    modes = [str(t["mode"]) for t in spec.target_configs]
    for mode in modes:
        if not _MODE_RE.match(mode):
            raise ValueError(
                f"Dataset {spec.name}: target mode {mode!r} must be lowercase snake_case"
            )
    if len(set(modes)) != len(modes):
        raise ValueError(f"Dataset {spec.name}: duplicate target modes {modes}")
    defaults = [t for t in spec.target_configs if t.get("default")]
    if len(defaults) != 1:
        raise ValueError(
            f"Dataset {spec.name}: exactly one [[targets]] must set default=true "
            f"(found {len(defaults)})"
        )
    if spec.default_target_mode not in modes:
        raise ValueError(
            f"Dataset {spec.name}: default_target_mode={spec.default_target_mode!r} "
            f"is not among target modes {modes}"
        )
    default_mode = str(defaults[0]["mode"])
    if default_mode != spec.default_target_mode:
        raise ValueError(
            f"Dataset {spec.name}: default target [[targets]] mode {default_mode!r} "
            f"does not match [dataset] default_target_mode {spec.default_target_mode!r}"
        )
    # frequency_severity availability is derived from the count column.
    for t in spec.target_configs:
        for key in ("mode", "source_column", "rate_label", "entity_label"):
            if not str(t.get(key, "")).strip():
                raise ValueError(
                    f"Dataset {spec.name}: target {t.get('mode')!r} is missing {key!r}"
                )
