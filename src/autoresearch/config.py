"""Configuration loading for local runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

from autoresearch.targets import BURNING_COST, normalise_target_mode


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
    # handoff
    running_stale_minutes: int = 30


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(
    config_path: str | Path | None = None,
    track_id: str | None = None,
    run_id: str | None = None,
    new_run: bool = False,
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

    resolved_track = track_id or "default"

    base_artifacts = _resolve(PROJECT_ROOT, paths["artifacts_dir"])
    resolved_run = run_id or "default"
    track_base: Path | None = None

    if track_id:
        track_base = base_artifacts / "tracks" / track_id
        resolved_run = _resolve_run_id(track_base, run_id, new_run=new_run)
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
        run_id=resolved_run,
        track_base_dir=track_base,
        random_seed=int(data["random_seed"]),
        id_column=str(data["id_column"]),
        agent_dataset_name=str(data["agent_dataset_name"]),
        claim_capping_enabled=bool(preprocessing["claim_capping_enabled"]),
        claim_cap_threshold=float(preprocessing["claim_cap_threshold"]),
        split_ratios={key: float(value) for key, value in splits.items()},
        ordinary_train_split=str(evaluation["ordinary_train_split"]),
        ordinary_eval_splits=tuple(str(value) for value in evaluation["ordinary_eval_splits"]),
        target_mode=normalise_target_mode(evaluation.get("target_mode", BURNING_COST)),
        primary_metric=str(evaluation.get("primary_metric", "tweedie_deviance_p15")),
        tweedie_power=float(evaluation.get("tweedie_power", 1.5)),
        use_cv=bool(evaluation.get("use_cv", False)),
        cv_folds=int(evaluation.get("cv_folds", 4)),
        cv_n_repeats=int(evaluation.get("cv_n_repeats", 1)),
        cv_seed=int(evaluation.get("cv_seed", int(data["random_seed"]))),
        gate_mode=str(evaluation.get("gate_mode", "cv_bootstrap")),
        gate_primary_metric=str(evaluation.get("gate_primary_metric", "gini_weighted")),
        bootstrap_per_fold=int(resampling.get("bootstrap_per_fold", 20)),
        escalation_win_rate_low=float(resampling.get("escalation_win_rate_low", 0.40)),
        escalation_win_rate_high=float(resampling.get("escalation_win_rate_high", 0.60)),
        escalation_partitions=int(resampling.get("escalation_partitions", 2)),
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
        running_stale_minutes=int(raw.get("handoff", {}).get("running_stale_minutes", 30)),
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
            manifest_path.write_text(
                json.dumps(
                    {
                        "track_id": config.track_id,
                        "run_id": config.run_id,
                        "run_dir": str(config.artifacts_dir),
                        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
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
