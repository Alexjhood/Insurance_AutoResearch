"""Configuration loading for local runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import re
import tomllib

from autoresearch.datasets import DatasetSpec, dataset_data_dir, load_dataset_spec
from autoresearch.targets import BURNING_COST, normalise_target_mode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"
AGENT_TRACK_IDS = {"codex", "claude", "opencode"}
RUN_ID_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
DEFAULT_DATASET = "french_motor"


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved project configuration with paths rooted at the repository."""

    root: Path
    raw_data_dir: Path
    processed_dir: Path
    holdout_vault_dir: Path
    metadata_dir: Path
    splits_dir: Path
    artifacts_dir: Path
    registry_path: Path
    research_log_path: Path
    track_id: str
    random_seed: int
    id_column: str
    agent_dataset_name: str
    claim_capping_enabled: bool
    claim_cap_threshold: float
    split_ratios: dict[str, float]
    ordinary_train_split: str
    ordinary_eval_splits: tuple[str, ...]
    # new evaluation settings
    target_mode: str
    primary_metric: str
    tweedie_power: float
    use_cv: bool
    cv_folds: int
    cv_n_repeats: int
    cv_seed: int
    gate_mode: str          # "cv_bootstrap" | "repeated_cv" | "single_partition"
    gate_primary_metric: str
    bootstrap_per_fold: int
    escalation_win_rate_low: float
    escalation_win_rate_high: float
    escalation_partitions: int
    repeated_resamples: int
    bootstrap_iterations: int
    resample_fraction: float
    resampling_seed: int
    # promotion
    minimum_mean_lift: float
    min_relative_lift: float
    min_absolute_lift: float
    minimum_win_rate: float
    bootstrap_lower_bound: float
    bootstrap_lower_bound_relative: float
    confidence_level: float
    max_predicted_to_actual_drift: float
    require_diagnostics: bool
    bonferroni_lookback: int
    # handoff dirs
    handoff_base_dir: Path
    handoff_context_dir: Path
    handoff_proposal_inbox_dir: Path
    handoff_proposal_processed_dir: Path
    handoff_results_dir: Path
    handoff_handoffs_dir: Path
    proposal_inbox_file: Path
    # dedup
    deduplication_policy: str
    deduplication_lookback: int
    # search space (raw dict for flexibility)
    search_space: dict[str, object]
    run_id: str = "default"
    track_base_dir: Path | None = None
    # compute budget
    base_budget_minutes: int = 10
    budget_increment_minutes: int = 5
    experiments_per_increment: int = 5
    compute_enforce: bool = True
    preflight_enabled: bool = True
    preflight_sample_rows: int = 5000
    # repair
    repair_noise_floor_eps: float = 0.002
    repair_auto_abandon_enabled: bool = True
    # screening
    screening_enabled: bool = True
    screening_min_absolute_lift: float = -0.001
    screening_min_relative_lift: float = -0.002
    screening_bootstrap_iterations: int = 200
    screening_confidence_level: float = 0.90
    screening_min_bootstrap_rows: int = 30
    partition_rotation_interval: int = 5
    # handoff
    running_stale_minutes: int = 30
    # model identity (required for memory harvest; optional at config-load time)
    model_provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    model_harness: str | None = None
    # cross-run memory
    structural_gini_threshold: float = 0.37
    # declarative model recipes (#6): "run" (Option 1) | "memory" (Option 2)
    recipe_reuse_scope: str = "run"
    # Whether this command should move artifacts/tracks/<track>/latest_run.json.
    update_latest_run: bool = True
    # Active dataset spec. Defaults to French so ProjectConfig objects built
    # directly in tests (with French columns) keep working unchanged.
    dataset: DatasetSpec = field(default_factory=lambda: load_dataset_spec(DEFAULT_DATASET))

    @property
    def dataset_name(self) -> str:
        return self.dataset.name


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a deep-merged copy of *base* with *override* applied on top."""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_manifest_dataset(manifest_path: Path) -> str | None:
    """Return the dataset pinned in a run manifest, if any."""

    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("dataset")
    return str(value) if value else None


def _read_orchestration_target_mode(manifest_path: Path) -> str | None:
    """Return an orchestration-pinned target mode, else preserve legacy loading."""

    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not payload.get("orchestration_id"):
        return None
    value = payload.get("target_mode")
    return str(value) if value else None


def load_config(
    config_path: str | Path | None = None,
    track_id: str | None = None,
    run_id: str | None = None,
    new_run: bool = False,
    dataset: str | None = None,
) -> "ProjectConfig":
    """Load TOML config and resolve all project paths.

    When *track_id* is supplied every mutable artifact path is scoped under
    ``artifacts/tracks/<track_id>/`` so that parallel research runs (e.g. one
    for Claude and one for Codex) are fully isolated from each other.  Shared
    read-only paths (raw data, processed data, split pack, holdout vault) are
    unchanged.
    """

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("rb") as f:
        raw = tomllib.load(f)

    paths = raw["paths"]
    data = raw["data"]
    preprocessing = raw["preprocessing"]
    splits = raw["splits"]
    evaluation = raw["evaluation"]
    resampling = raw["resampling"]
    promotion = raw["promotion"]
    handoff = raw["handoff"]
    deduplication = raw["deduplication"]
    search_space = raw["search_space"]
    compute_cfg = raw.get("compute", {})
    repair_cfg = raw.get("repair", {})
    screening_cfg = raw.get("screening", {})
    memory_cfg = raw.get("memory", {})
    recipes_cfg = raw.get("recipes", {})

    resolved_track = track_id or "default"

    base_artifacts = _resolve(PROJECT_ROOT, paths["artifacts_dir"])
    resolved_run = run_id or "default"
    track_base: Path | None = None
    update_latest_run = True

    if track_id:
        if track_id in AGENT_TRACK_IDS and run_id and not RUN_ID_TIMESTAMP_RE.fullmatch(run_id):
            raise ValueError(
                f"run_id for agent track '{track_id}' must be a UTC timestamp "
                "in YYYYMMDDTHHMMSSZ form. Use --new-run to create one, or "
                "omit --run-id to continue the latest timestamped run."
            )
        track_base = base_artifacts / "tracks" / track_id
        latest_path = track_base / "latest_run.json"
        update_latest_run = new_run or run_id is None or not latest_path.exists()
        resolved_run = _resolve_run_id(track_base, run_id, new_run=new_run)
        if track_id in AGENT_TRACK_IDS and not RUN_ID_TIMESTAMP_RE.fullmatch(resolved_run):
            raise ValueError(
                f"latest run_id for agent track '{track_id}' is not timestamp-shaped: "
                f"{resolved_run!r}. Pass --new-run to create a timestamped run."
            )
        run_base = track_base / "runs" / resolved_run
        artifacts_dir = run_base
        registry_path = run_base / "registry.sqlite"
        research_log_path = run_base / "RESEARCH_LOG.md"
        handoff_base_dir = run_base
        handoff_context_dir = run_base / "context"
        handoff_proposal_inbox_dir = run_base / "proposal_inbox"
        handoff_proposal_processed_dir = run_base / "proposal_processed"
        handoff_results_dir = run_base / "results"
        handoff_handoffs_dir = run_base / "handoffs"
        run_manifest_path = run_base / "run_manifest.json"
        manifest_dataset = _read_manifest_dataset(run_manifest_path)
        manifest_target_mode = _read_orchestration_target_mode(run_manifest_path)
    else:
        manifest_dataset = None
        manifest_target_mode = None
        artifacts_dir = base_artifacts
        registry_path = _resolve(PROJECT_ROOT, paths["registry_path"])
        research_log_path = PROJECT_ROOT / "docs" / "RESEARCH_LOG.md"
        handoff_base_dir = _resolve(PROJECT_ROOT, handoff["base_dir"])
        handoff_context_dir = _resolve(PROJECT_ROOT, handoff["context_dir"])
        handoff_proposal_inbox_dir = _resolve(PROJECT_ROOT, handoff["proposal_inbox_dir"])
        handoff_proposal_processed_dir = _resolve(PROJECT_ROOT, handoff["proposal_processed_dir"])
        handoff_results_dir = _resolve(PROJECT_ROOT, handoff["results_dir"])
        handoff_handoffs_dir = _resolve(PROJECT_ROOT, handoff["handoffs_dir"])

    # --- Resolve the active dataset: explicit flag > run manifest > default ---
    default_dataset = str(data.get("default_dataset", DEFAULT_DATASET))
    if dataset and manifest_dataset and dataset != manifest_dataset:
        raise ValueError(
            f"--dataset {dataset!r} contradicts run {resolved_run!r} which is pinned "
            f"to dataset {manifest_dataset!r}. Omit --dataset to use the pinned dataset."
        )
    dataset_name = dataset or manifest_dataset or default_dataset
    dataset_spec = load_dataset_spec(dataset_name)
    # Bind the active dataset so targets.target_spec()/normalise_target_mode()
    # — called with no dataset argument by the protected evaluation stack —
    # resolve this dataset's modes/columns. The model-layer column constants and
    # the feature policy bind here too.
    from autoresearch import feature_policy as _feature_policy
    from autoresearch import targets as _targets
    from autoresearch.models import columns as _model_columns

    _targets.bind_dataset(dataset_spec)
    _model_columns.bind(dataset_spec)
    _feature_policy.bind(dataset_spec)

    # Deep-merge the dataset's [overrides.<section>] tables over framework defaults.
    for section, override in dataset_spec.overrides.items():
        if section == "compute":
            compute_cfg = _deep_merge(compute_cfg, override)
        elif section == "evaluation":
            evaluation = _deep_merge(evaluation, override)
        elif section == "screening":
            screening_cfg = _deep_merge(screening_cfg, override)
        elif section == "promotion":
            promotion = _deep_merge(promotion, override)
        elif section == "resampling":
            resampling = _deep_merge(resampling, override)
        elif section == "memory":
            memory_cfg = _deep_merge(memory_cfg, override)

    # Per-dataset data paths under data/datasets/<name>/…
    data_dir = dataset_data_dir(dataset_name)
    dataset_processed_dir = data_dir / "processed"
    dataset_metadata_dir = data_dir / "metadata"
    dataset_splits_dir = data_dir / "splits"
    dataset_holdout_vault_dir = data_dir / "holdout_vault"

    # Capping and cross-run threshold come from the dataset spec.
    cap = dataset_spec.cap
    claim_capping_enabled = cap is not None
    claim_cap_threshold = float(cap.threshold) if cap is not None else float(
        preprocessing.get("claim_cap_threshold", 100000)
    )
    structural_threshold = (
        dataset_spec.structural_gini_threshold
        if dataset_spec.structural_gini_threshold is not None
        else float(memory_cfg.get("structural_gini_threshold", 0.37))
    )

    return ProjectConfig(
        root=PROJECT_ROOT,
        raw_data_dir=dataset_spec.raw_dir,
        processed_dir=dataset_processed_dir,
        holdout_vault_dir=dataset_holdout_vault_dir,
        metadata_dir=dataset_metadata_dir,
        splits_dir=dataset_splits_dir,
        artifacts_dir=artifacts_dir,
        registry_path=registry_path,
        research_log_path=research_log_path,
        track_id=resolved_track,
        run_id=resolved_run,
        track_base_dir=track_base,
        random_seed=int(data["random_seed"]),
        id_column=dataset_spec.id_column,
        agent_dataset_name=str(data["agent_dataset_name"]),
        claim_capping_enabled=claim_capping_enabled,
        claim_cap_threshold=claim_cap_threshold,
        split_ratios={key: float(value) for key, value in splits.items()},
        ordinary_train_split=str(evaluation["ordinary_train_split"]),
        ordinary_eval_splits=tuple(str(value) for value in evaluation["ordinary_eval_splits"]),
        target_mode=normalise_target_mode(
            manifest_target_mode or dataset_spec.default_target_mode, dataset_spec
        ),
        primary_metric=str(evaluation.get("primary_metric", "tweedie_deviance_p15")),
        tweedie_power=float(evaluation.get("tweedie_power", 1.5)),
        use_cv=bool(evaluation.get("use_cv", False)),
        cv_folds=int(evaluation.get("cv_folds", 4)),
        cv_n_repeats=int(evaluation.get("cv_n_repeats", 1)),
        cv_seed=int(evaluation.get("cv_seed", int(data["random_seed"]))),
        gate_mode=str(evaluation.get("gate_mode", "cv_bootstrap")),
        gate_primary_metric=str(evaluation.get("gate_primary_metric", "gini_weighted")),
        bootstrap_per_fold=int(resampling.get("bootstrap_per_fold", 20)),
        escalation_win_rate_low=float(resampling.get("escalation_win_rate_low", 0.50)),
        escalation_win_rate_high=float(resampling.get("escalation_win_rate_high", 0.75)),
        escalation_partitions=int(resampling.get("escalation_partitions", 2)),
        partition_rotation_interval=int(resampling.get("partition_rotation_interval", 5)),
        repeated_resamples=int(resampling["repeated_resamples"]),
        bootstrap_iterations=int(resampling["bootstrap_iterations"]),
        resample_fraction=float(resampling["resample_fraction"]),
        resampling_seed=int(resampling["random_seed"]),
        minimum_mean_lift=float(promotion["minimum_mean_lift"]),
        min_relative_lift=float(promotion.get("min_relative_lift", 0.005)),
        min_absolute_lift=float(promotion.get("min_absolute_lift", 0.0)),
        minimum_win_rate=float(promotion["minimum_win_rate"]),
        bootstrap_lower_bound=float(promotion["bootstrap_lower_bound"]),
        bootstrap_lower_bound_relative=float(promotion.get("bootstrap_lower_bound_relative", 0.0)),
        confidence_level=float(promotion["confidence_level"]),
        max_predicted_to_actual_drift=float(promotion.get("max_predicted_to_actual_drift", 0.05)),
        require_diagnostics=bool(promotion.get("require_diagnostics", True)),
        bonferroni_lookback=int(promotion.get("bonferroni_lookback", 10)),
        handoff_base_dir=handoff_base_dir,
        handoff_context_dir=handoff_context_dir,
        handoff_proposal_inbox_dir=handoff_proposal_inbox_dir,
        handoff_proposal_processed_dir=handoff_proposal_processed_dir,
        handoff_results_dir=handoff_results_dir,
        handoff_handoffs_dir=handoff_handoffs_dir,
        proposal_inbox_file=handoff_proposal_inbox_dir / "manual_proposals.jsonl",
        deduplication_policy=str(deduplication["policy"]),
        deduplication_lookback=int(deduplication["lookback"]),
        search_space=dict(search_space),
        base_budget_minutes=int(compute_cfg.get("base_budget_minutes", 10)),
        budget_increment_minutes=int(compute_cfg.get("budget_increment_minutes", 5)),
        experiments_per_increment=int(compute_cfg.get("experiments_per_increment", 5)),
        compute_enforce=bool(compute_cfg.get("enforce", True)),
        preflight_enabled=bool(compute_cfg.get("preflight_enabled", True)),
        preflight_sample_rows=int(compute_cfg.get("preflight_sample_rows", 5000)),
        repair_noise_floor_eps=float(repair_cfg.get("noise_floor_eps", 0.002)),
        repair_auto_abandon_enabled=bool(repair_cfg.get("auto_abandon_enabled", True)),
        screening_enabled=bool(screening_cfg.get("enabled", True)),
        screening_min_absolute_lift=float(screening_cfg.get("min_absolute_lift", -0.001)),
        screening_min_relative_lift=float(screening_cfg.get("min_relative_lift", -0.002)),
        screening_bootstrap_iterations=int(screening_cfg.get("bootstrap_iterations", 200)),
        screening_confidence_level=float(screening_cfg.get("confidence_level", 0.90)),
        screening_min_bootstrap_rows=int(screening_cfg.get("min_bootstrap_rows", 30)),
        running_stale_minutes=int(raw.get("handoff", {}).get("running_stale_minutes", 30)),
        structural_gini_threshold=structural_threshold,
        recipe_reuse_scope=str(recipes_cfg.get("reuse_scope", "run")).strip().lower(),
        update_latest_run=update_latest_run,
        dataset=dataset_spec,
    )


def ensure_project_dirs(config: ProjectConfig) -> None:
    """Create configured output directories if they are missing."""

    for path in (
        config.processed_dir,
        config.holdout_vault_dir,
        config.metadata_dir,
        config.splits_dir,
        config.artifacts_dir,
        config.handoff_base_dir,
        config.handoff_context_dir,
        config.handoff_proposal_inbox_dir,
        config.handoff_proposal_processed_dir,
        config.handoff_results_dir,
        config.handoff_handoffs_dir,
        config.research_log_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    if config.track_id != "default" and config.track_base_dir is not None:
        latest_path = config.track_base_dir / "latest_run.json"
        if config.update_latest_run:
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(
                json.dumps(
                    {
                        "track_id": config.track_id,
                        "run_id": config.run_id,
                        "run_dir": str(config.artifacts_dir),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        manifest_path = config.artifacts_dir / "run_manifest.json"
        if not manifest_path.exists():
            import os
            _raw_access = os.environ.get("AUTORESEARCH_MEMORY_ACCESS", "").strip().lower()
            _memory_access = _raw_access if _raw_access in {"none", "own", "all"} else "none"
            manifest_body: dict = {
                "track_id": config.track_id,
                "run_id": config.run_id,
                "run_dir": str(config.artifacts_dir),
                "dataset": config.dataset_name,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "memory_access": _memory_access,
            }
            if config.model_provider and config.model_name:
                manifest_body["model_identity"] = {
                    "provider": config.model_provider.lower().strip(),
                    "name": config.model_name.lower().strip(),
                    "version": config.model_version or "",
                    "harness": config.model_harness or "",
                }
            manifest_path.write_text(
                json.dumps(manifest_body, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif config.model_provider and config.model_name:
            # Patch identity into an existing manifest that lacks it.
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if "model_identity" not in existing:
                existing["model_identity"] = {
                    "provider": config.model_provider.lower().strip(),
                    "name": config.model_name.lower().strip(),
                    "version": config.model_version or "",
                    "harness": config.model_harness or "",
                }
                manifest_path.write_text(
                    json.dumps(existing, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )


def _resolve_run_id(track_base: Path, requested_run_id: str | None, *, new_run: bool = False) -> str:
    if requested_run_id and new_run:
        raise ValueError("--run-id and --new-run are mutually exclusive")
    if requested_run_id:
        return _safe_run_id(requested_run_id)

    if new_run:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    latest_path = track_base / "latest_run.json"
    if latest_path.exists():
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
            latest = str(payload.get("run_id", "")).strip()
            if latest:
                return _safe_run_id(latest)
        except Exception:
            pass

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_run_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    if not safe:
        raise ValueError("run_id must contain at least one letter, number, hyphen, or underscore")
    return safe
