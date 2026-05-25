"""Configuration loading for local runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"


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
    primary_metric: str
    tweedie_power: float
    use_cv: bool
    cv_folds: int
    cv_n_repeats: int
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
    # llm
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_proposal_file: Path
    # handoff dirs
    handoff_base_dir: Path
    handoff_context_dir: Path
    handoff_proposal_inbox_dir: Path
    handoff_proposal_processed_dir: Path
    handoff_results_dir: Path
    handoff_handoffs_dir: Path
    # dedup
    deduplication_policy: str
    deduplication_lookback: int
    # search space (raw dict for flexibility)
    search_space: dict[str, object]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(
    config_path: str | Path | None = None,
    track_id: str | None = None,
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
    llm = raw["llm"]
    handoff = raw["handoff"]
    deduplication = raw["deduplication"]
    search_space = raw["search_space"]

    resolved_track = track_id or "default"

    base_artifacts = _resolve(PROJECT_ROOT, paths["artifacts_dir"])

    if track_id:
        track_base = base_artifacts / "tracks" / track_id
        artifacts_dir = track_base
        registry_path = track_base / "registry.sqlite"
        research_log_path = track_base / "RESEARCH_LOG.md"
        handoff_base_dir = track_base / "auto_research"
        handoff_context_dir = handoff_base_dir / "context"
        handoff_proposal_inbox_dir = handoff_base_dir / "proposals" / "inbox"
        handoff_proposal_processed_dir = handoff_base_dir / "proposals" / "processed"
        handoff_results_dir = handoff_base_dir / "results"
        handoff_handoffs_dir = handoff_base_dir / "handoffs"
        llm_proposal_file = handoff_proposal_inbox_dir / "manual_proposals.jsonl"
    else:
        artifacts_dir = base_artifacts
        registry_path = _resolve(PROJECT_ROOT, paths["registry_path"])
        research_log_path = PROJECT_ROOT / "docs" / "RESEARCH_LOG.md"
        handoff_base_dir = _resolve(PROJECT_ROOT, handoff["base_dir"])
        handoff_context_dir = _resolve(PROJECT_ROOT, handoff["context_dir"])
        handoff_proposal_inbox_dir = _resolve(PROJECT_ROOT, handoff["proposal_inbox_dir"])
        handoff_proposal_processed_dir = _resolve(PROJECT_ROOT, handoff["proposal_processed_dir"])
        handoff_results_dir = _resolve(PROJECT_ROOT, handoff["results_dir"])
        handoff_handoffs_dir = _resolve(PROJECT_ROOT, handoff["handoffs_dir"])
        llm_proposal_file = _resolve(PROJECT_ROOT, llm["proposal_file"])

    return ProjectConfig(
        root=PROJECT_ROOT,
        raw_data_dir=_resolve(PROJECT_ROOT, paths["raw_data_dir"]),
        processed_dir=_resolve(PROJECT_ROOT, paths["processed_dir"]),
        holdout_vault_dir=_resolve(PROJECT_ROOT, paths.get("holdout_vault_dir", "data/holdout_vault")),
        metadata_dir=_resolve(PROJECT_ROOT, paths["metadata_dir"]),
        splits_dir=_resolve(PROJECT_ROOT, paths["splits_dir"]),
        artifacts_dir=artifacts_dir,
        registry_path=registry_path,
        research_log_path=research_log_path,
        track_id=resolved_track,
        random_seed=int(data["random_seed"]),
        id_column=str(data["id_column"]),
        agent_dataset_name=str(data["agent_dataset_name"]),
        claim_capping_enabled=bool(preprocessing["claim_capping_enabled"]),
        claim_cap_threshold=float(preprocessing["claim_cap_threshold"]),
        split_ratios={key: float(value) for key, value in splits.items()},
        ordinary_train_split=str(evaluation["ordinary_train_split"]),
        ordinary_eval_splits=tuple(str(value) for value in evaluation["ordinary_eval_splits"]),
        primary_metric=str(evaluation.get("primary_metric", "tweedie_deviance_p15")),
        tweedie_power=float(evaluation.get("tweedie_power", 1.5)),
        use_cv=bool(evaluation.get("use_cv", False)),
        cv_folds=int(evaluation.get("cv_folds", 5)),
        cv_n_repeats=int(evaluation.get("cv_n_repeats", 1)),
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
        llm_provider=str(llm["provider"]),
        llm_model=str(llm["model"]),
        llm_temperature=float(llm["temperature"]),
        llm_proposal_file=llm_proposal_file,
        handoff_base_dir=handoff_base_dir,
        handoff_context_dir=handoff_context_dir,
        handoff_proposal_inbox_dir=handoff_proposal_inbox_dir,
        handoff_proposal_processed_dir=handoff_proposal_processed_dir,
        handoff_results_dir=handoff_results_dir,
        handoff_handoffs_dir=handoff_handoffs_dir,
        deduplication_policy=str(deduplication["policy"]),
        deduplication_lookback=int(deduplication["lookback"]),
        search_space=dict(search_space),
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
