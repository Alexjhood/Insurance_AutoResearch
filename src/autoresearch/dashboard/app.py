"""Phase 0 Streamlit dashboard skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from autoresearch.config import load_config
from autoresearch.controller.handoff import inbox_status
from autoresearch.experiment_registry.registry import (
    get_official_champion,
    list_artifacts,
    list_branches,
    list_champion_history,
    list_comparisons,
    list_experiments,
    list_proposals,
    list_session_events,
    list_sessions,
    registry_counts,
)


st.set_page_config(page_title="Insurance AutoResearch", layout="wide")


def _load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_home() -> None:
    config = load_config()
    st.title("Insurance AutoResearch")
    st.caption("Phase 0-4 controlled auto-research backbone")

    profile = _load_json(config.metadata_dir / "dataset_profile.json")
    split_manifest = _load_json(config.splits_dir / "split_pack_manifest.json")
    capping = _load_json(config.metadata_dir / "capping_diagnostics.json")
    counts = registry_counts(config.registry_path)

    cols = st.columns(6)
    cols[0].metric("Registry experiments", counts["experiments"])
    cols[1].metric("Registry artifacts", counts["artifacts"])
    cols[2].metric("Comparisons", counts["comparisons"])
    cols[3].metric("Proposals", counts["proposals"])
    cols[4].metric("Rows", profile["row_count"] if profile else "Not prepared")
    cols[5].metric("Columns", profile["column_count"] if profile else "Not prepared")

    st.subheader("Current Phase")
    st.write(
        "The LLM proposes structured challengers, while deterministic evaluation and "
        "promotion gates remain responsible for champion changes."
    )

    if split_manifest:
        st.subheader("Split Pack")
        st.json(split_manifest)
    if capping:
        st.subheader("Default Claim Capping")
        st.json(capping)


def render_data_profile() -> None:
    config = load_config()
    profile = _load_json(config.metadata_dir / "dataset_profile.json")
    st.title("Data Profile")
    if not profile:
        st.info("Run `autoresearch prepare-data` to generate profile metadata.")
        return
    st.json({key: profile[key] for key in ("profile_version", "row_count", "column_count", "source_paths")})
    st.dataframe(pd.DataFrame(profile["columns"]), use_container_width=True)


def render_experiments() -> None:
    config = load_config()
    st.title("Experiments")
    rows = list_experiments(config.registry_path)
    if not rows:
        st.info("Run `autoresearch run-all-baselines` to create baseline experiment results.")
        st.json(registry_counts(config.registry_path))
        return

    table = pd.DataFrame(rows)
    visible_columns = [
        "experiment_id",
        "experiment_name",
        "target_mode",
        "target_strategy",
        "model_family",
        "mean_score",
        "std_score",
        "primary_metric",
        "claim_cap_threshold",
        "status",
        "metrics_path",
    ]
    st.subheader("Experiment Table")
    st.dataframe(table[[column for column in visible_columns if column in table.columns]], use_container_width=True)

    scored = table.dropna(subset=["mean_score"]).copy()
    if not scored.empty:
        st.subheader("Best Point-Estimate Experiments")
        ascending = config.primary_metric != "gini_weighted"
        best = scored.sort_values("mean_score", ascending=ascending).head(5)
        st.dataframe(best[[column for column in visible_columns if column in best.columns]], use_container_width=True)

        st.subheader("Direct vs Frequency-Severity")
        score_agg = "min" if ascending else "max"
        summary = (
            scored.groupby("target_strategy", as_index=False)
            .agg(best_mean_score=("mean_score", score_agg), runs=("experiment_id", "count"))
            .sort_values("best_mean_score", ascending=ascending)
        )
        st.dataframe(summary, use_container_width=True)

        st.subheader("Stability by Family and Strategy")
        stability = (
            scored.groupby(["model_family", "target_strategy"], as_index=False)
            .agg(mean_score=("mean_score", "mean"), mean_std_score=("std_score", "mean"), runs=("experiment_id", "count"))
            .sort_values("mean_score", ascending=ascending)
        )
        st.dataframe(stability, use_container_width=True)

    selected = st.selectbox("Experiment details", table["experiment_id"].tolist())
    selected_row = table[table["experiment_id"] == selected].iloc[0].to_dict()
    metrics = _load_json(Path(selected_row["metrics_path"])) if selected_row.get("metrics_path") else None
    if metrics:
        st.subheader("Split-Level Metrics")
        st.dataframe(pd.DataFrame(metrics["split_metrics"]), use_container_width=True)
        st.subheader("Preprocessing")
        st.json(metrics.get("preprocessing", {}))
    artifacts = list_artifacts(config.registry_path, selected)
    if artifacts:
        st.subheader("Artifacts")
        st.dataframe(pd.DataFrame(artifacts), use_container_width=True)


def render_champion() -> None:
    config = load_config()
    st.title("Official Champion")
    state = get_official_champion(config.registry_path)
    experiments = pd.DataFrame(list_experiments(config.registry_path))
    if state is None:
        st.info("Run `autoresearch init-official-champion` to initialise official champion state.")
    else:
        st.subheader("Official Champion State")
        st.json(state)

    if not experiments.empty and "mean_score" in experiments.columns:
        scored = experiments.dropna(subset=["mean_score"])
        if not scored.empty:
            best = scored.sort_values("mean_score", ascending=True).iloc[0].to_dict()
            st.subheader("Official vs Best Point Estimate")
            st.write(
                {
                    "official_champion_id": state["champion_id"] if state else None,
                    "best_point_estimate_id": best["experiment_id"],
                    "best_point_estimate_score": best["mean_score"],
                    "distinction": "Official champion changes only through the promotion gate.",
                }
            )

    history = list_champion_history(config.registry_path)
    st.subheader("Champion History")
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.write("No champion history yet.")


def render_auto_research() -> None:
    config = load_config()
    st.title("Auto Research Queue")

    proposals = list_proposals(config.registry_path)
    st.subheader("Proposal Queue")
    if proposals:
        table = pd.DataFrame(proposals)
        visible = [
            "proposal_id",
            "status",
            "parent_experiment_id",
            "branch_id",
            "experiment_name",
            "change_summary",
            "experiment_id",
            "comparison_id",
            "notes",
        ]
        st.dataframe(table[[column for column in visible if column in table.columns]], use_container_width=True)

        selected = st.selectbox("Proposal details", table["proposal_id"].tolist())
        row = table[table["proposal_id"] == selected].iloc[0].to_dict()
        st.subheader("Rationale and Change Summary")
        st.write(row.get("rationale"))
        st.write(row.get("change_summary"))
        st.subheader("Expected Benefit / Key Risk")
        st.write({"expected_benefit": row.get("expected_benefit"), "key_risk": row.get("key_risk")})
        st.subheader("Structured Config")
        st.json(row.get("config", {}))
        if row.get("validation_errors"):
            st.subheader("Validation Errors")
            st.json(row.get("validation_errors"))
    else:
        st.info("Write a proposal JSON to the inbox and run `autoresearch run-session-cycle` to process it.")

    st.subheader("Branch Lineage")
    branches = list_branches(config.registry_path)
    if branches:
        st.dataframe(pd.DataFrame(branches), use_container_width=True)
    else:
        st.write("No branches registered.")


def render_handoff() -> None:
    config = load_config()
    st.title("File Handoff")
    status = inbox_status(config)
    st.subheader("Workflow Status")
    cols = st.columns(5)
    cols[0].metric("Mode", status["mode"])
    cols[1].metric("Inbox JSON", status["inbox_json_count"])
    cols[2].metric("Processed Valid", status["processed_valid_count"])
    cols[3].metric("Processed Invalid", status["processed_invalid_count"])
    cols[4].metric("Processed Duplicate", status["processed_duplicate_count"])

    st.subheader("Handoff Artifacts")
    st.json(
        {
            "inbox_dir": status["inbox_dir"],
            "latest_context": status["latest_context"],
            "latest_handoff": status["latest_handoff"],
            "latest_cycle_result": status["latest_cycle_result"],
        }
    )

    latest_context = Path(status["latest_context"])
    latest_handoff = Path(status["latest_handoff"])
    latest_cycle = Path(status["latest_cycle_result"])
    st.subheader("Latest Exported Context")
    if latest_context.exists():
        st.write({"path": str(latest_context), "updated": latest_context.stat().st_mtime})
    else:
        st.info("Run `autoresearch export-context` to create the handoff context bundle.")

    st.subheader("Latest Handoff Summary")
    if latest_handoff.exists():
        st.markdown(latest_handoff.read_text(encoding="utf-8"))
    else:
        st.write("No handoff summary yet.")

    st.subheader("Latest Cycle Result")
    if latest_cycle.exists():
        st.markdown(latest_cycle.read_text(encoding="utf-8"))
    else:
        st.write("No file-handoff cycle result yet.")


def render_sessions() -> None:
    config = load_config()
    st.title("Autonomous Sessions")
    sessions = list_sessions(config.registry_path)
    if not sessions:
        st.info("Run `autoresearch start-session NAME` to create a supervised autonomous session.")
        return

    table = pd.DataFrame(sessions)
    visible = [
        "session_id",
        "name",
        "state",
        "current_cycle",
        "max_cycles",
        "stop_requested",
        "updated_at",
        "summary_path",
        "notes",
    ]
    st.subheader("Session State")
    st.dataframe(table[[column for column in visible if column in table.columns]], use_container_width=True)

    selected = st.selectbox("Session details", table["session_id"].tolist())
    row = table[table["session_id"] == selected].iloc[0].to_dict()
    summary_path = Path(row.get("summary_path") or "")
    if summary_path.exists():
        st.subheader("Latest Session Summary")
        st.markdown(summary_path.read_text(encoding="utf-8"))

    st.subheader("Recent Session Events")
    events = list_session_events(config.registry_path, selected, limit=50)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    else:
        st.write("No events recorded.")

    st.subheader("Latest Proposal Outcomes")
    proposals = list_proposals(config.registry_path)
    if proposals:
        outcomes = pd.DataFrame(proposals)
        visible_props = ["proposal_id", "status", "experiment_name", "comparison_id", "notes", "updated_at"]
        st.dataframe(outcomes[[column for column in visible_props if column in outcomes.columns]], use_container_width=True)
    else:
        st.write("No proposals recorded.")


def render_memory() -> None:
    """Memory & Leaderboard page — reads from artifacts/memory/memory.sqlite."""
    import sqlite3

    config = load_config()
    memory_path = config.root / config.memory_store_relpath
    threshold = config.structural_gini_threshold

    st.title("Memory & Leaderboard")
    if not memory_path.exists():
        st.info(
            "No memory store found. Run `autoresearch memory harvest --all` to build it."
        )
        return

    try:
        with sqlite3.connect(memory_path) as con:
            exp_df = pd.read_sql_query(
                "SELECT e.run_uid, e.cycle_index, e.gini_weighted, e.status,"
                "       r.model_id, m.provider"
                " FROM experiments e"
                " JOIN runs r ON r.run_uid = e.run_uid"
                " JOIN models m ON m.model_id = r.model_id"
                " WHERE e.status = 'completed' AND e.gini_weighted IS NOT NULL",
                con,
            )
            runs_df = pd.read_sql_query(
                "SELECT r.run_uid, r.model_id, r.n_experiments, r.n_promotions,"
                "       r.peak_gini, r.final_champion_id, m.provider"
                " FROM runs r JOIN models m ON m.model_id = r.model_id",
                con,
            )
            cmp_df = pd.read_sql_query(
                "SELECT c.run_uid, c.decision, c.mean_lift, c.std_lift, r.model_id"
                " FROM comparisons c JOIN runs r ON r.run_uid = c.run_uid",
                con,
            )
    except Exception as exc:
        st.error(f"Error reading memory store: {exc}")
        return

    if exp_df.empty:
        st.info("Memory store is empty. Run `autoresearch memory harvest --all`.")
        return

    # ---- Score-trace chart: running-max gini_weighted per model_id ----
    st.subheader("Score-trace (running peak gini per model)")
    trace_df = (
        exp_df.sort_values(["model_id", "run_uid", "cycle_index"])
        .groupby(["model_id", "cycle_index"], as_index=False)["gini_weighted"]
        .max()
    )
    trace_df = trace_df.sort_values(["model_id", "cycle_index"])
    trace_df["running_max"] = trace_df.groupby("model_id")["gini_weighted"].cummax()

    model_ids = sorted(trace_df["model_id"].unique())
    selected_models = st.multiselect("Models to display", model_ids, default=model_ids)
    filtered_trace = trace_df[trace_df["model_id"].isin(selected_models)]

    if not filtered_trace.empty:
        pivot = (
            filtered_trace.pivot_table(
                index="cycle_index", columns="model_id", values="running_max", aggfunc="max"
            )
            .sort_index()
            .ffill()
        )
        st.line_chart(pivot)
        st.caption(f"Threshold line (structural insight): {threshold:.3f}")

    st.markdown("---")

    # ---- Leaderboard tables ----
    if runs_df.empty:
        st.info("No run data yet.")
        return

    # Aggregate per model
    model_agg = (
        runs_df.groupby(["model_id", "provider"], as_index=False)
        .agg(
            n_runs=("run_uid", "nunique"),
            n_experiments=("n_experiments", "sum"),
            peak_gini=("peak_gini", "max"),
            total_fit_seconds=("n_experiments", "sum"),  # placeholder; replaced below
        )
    )

    # Efficiency: peak_gini / n_experiments
    model_agg["efficiency_per_exp"] = model_agg["peak_gini"] / model_agg["n_experiments"].replace(0, float("nan"))

    # Time-to-structural-insight: first cycle_index where running_max >= threshold, per model
    crossed = (
        trace_df[trace_df["running_max"] >= threshold]
        .groupby("model_id", as_index=False)["cycle_index"]
        .min()
        .rename(columns={"cycle_index": "first_threshold_cycle"})
    )
    model_agg = model_agg.merge(crossed, on="model_id", how="left")
    model_agg["first_threshold_cycle"] = model_agg["first_threshold_cycle"].fillna(float("inf"))

    # Decision quality: fraction of promotions not later reverted (champion_history not
    # available in aggregator; proxy via comparing n_promotions rows).
    # Sub-noise thrash: comparisons where abs(mean_lift) < 2*std_lift
    if not cmp_df.empty and "std_lift" in cmp_df.columns:
        thrash = cmp_df.dropna(subset=["mean_lift", "std_lift"])
        thrash = thrash.copy()
        thrash["is_thrash"] = thrash["mean_lift"].abs() < 2 * thrash["std_lift"]
        thrash_rate = (
            thrash.groupby("model_id", as_index=False)
            .agg(total_cmps=("is_thrash", "count"), thrash_cmps=("is_thrash", "sum"))
        )
        thrash_rate["thrash_rate"] = thrash_rate["thrash_cmps"] / thrash_rate["total_cmps"].replace(0, float("nan"))
        model_agg = model_agg.merge(thrash_rate[["model_id", "thrash_rate"]], on="model_id", how="left")
    else:
        model_agg["thrash_rate"] = float("nan")

    tab_quality, tab_efficiency, tab_insight, tab_decision = st.tabs(
        ["Peak Quality", "Efficiency", "Time-to-Insight", "Decision Quality"]
    )

    with tab_quality:
        st.subheader("Peak Quality — max gini_weighted per model")
        board = (
            model_agg[["model_id", "provider", "n_runs", "n_experiments", "peak_gini"]]
            .sort_values("peak_gini", ascending=False)
            .reset_index(drop=True)
        )
        board.index += 1
        st.dataframe(board, use_container_width=True)

    with tab_efficiency:
        st.subheader("Efficiency — peak gini per experiment")
        board = (
            model_agg[["model_id", "provider", "n_runs", "n_experiments", "peak_gini", "efficiency_per_exp"]]
            .sort_values("efficiency_per_exp", ascending=False)
            .reset_index(drop=True)
        )
        board.index += 1
        st.dataframe(board, use_container_width=True)

    with tab_insight:
        st.subheader(f"Time-to-Structural-Insight — first cycle to reach gini >= {threshold:.3f}")
        board = (
            model_agg[["model_id", "provider", "n_runs", "n_experiments", "first_threshold_cycle"]]
            .sort_values("first_threshold_cycle", ascending=True)
            .reset_index(drop=True)
        )
        board["first_threshold_cycle"] = board["first_threshold_cycle"].replace(
            float("inf"), "never"
        )
        board.index += 1
        st.dataframe(board, use_container_width=True)

    with tab_decision:
        st.subheader("Decision Quality — sub-noise thrash rate (lower is better)")
        cols_show = ["model_id", "provider", "n_runs", "n_experiments", "thrash_rate"]
        board = (
            model_agg[cols_show]
            .sort_values("thrash_rate", ascending=True)
            .reset_index(drop=True)
        )
        board.index += 1
        st.dataframe(board, use_container_width=True)
        st.caption(
            "Thrash rate = fraction of comparisons where |mean_lift| < 2 * std_lift. "
            "Lower means the model made fewer sub-noise promotion attempts."
        )


def render_comparisons() -> None:
    config = load_config()
    st.title("Promotion Comparisons")
    rows = list_comparisons(config.registry_path)
    if not rows:
        st.info("Run `autoresearch compare-experiments CHAMPION_ID CHALLENGER_ID` to create promotion evidence.")
        return

    table = pd.DataFrame(rows)
    visible_columns = [
        "comparison_id",
        "champion_id",
        "challenger_id",
        "mean_lift",
        "challenger_win_rate",
        "bootstrap_interval_lower",
        "bootstrap_interval_upper",
        "probability_challenger_outperforms",
        "promotion_decision",
        "promotion_rationale",
    ]
    st.subheader("Champion vs Challenger")
    st.dataframe(table[[column for column in visible_columns if column in table.columns]], use_container_width=True)

    selected = st.selectbox("Comparison details", table["comparison_id"].tolist())
    selected_row = table[table["comparison_id"] == selected].iloc[0].to_dict()
    st.subheader("Promotion Decision")
    decision_label = "Likely real improvement" if selected_row["promotion_decision"] == "promote" else "Likely noise or inconclusive"
    st.metric("Decision", selected_row["promotion_decision"])
    st.write(decision_label)
    st.write(selected_row["promotion_rationale"])

    st.subheader("Uncertainty Summary")
    st.json(selected_row["bootstrap_summary"])

    paired_path = Path(selected_row["paired_scores_path"])
    if paired_path.exists():
        per_resample = pd.read_csv(paired_path)
        st.subheader("Per-Resample Lift")
        st.dataframe(per_resample, use_container_width=True)
        st.line_chart(per_resample.set_index("resample_id")[["lift"]])


page = st.sidebar.radio(
    "Page",
    [
        "Home", "Champion", "Data Profile", "Experiments", "Comparisons",
        "Auto Research", "File Handoff", "Sessions", "Memory & Leaderboard",
    ],
)
if page == "Home":
    render_home()
elif page == "Champion":
    render_champion()
elif page == "Data Profile":
    render_data_profile()
elif page == "Experiments":
    render_experiments()
elif page == "Comparisons":
    render_comparisons()
elif page == "Memory & Leaderboard":
    render_memory()
else:
    if page == "Auto Research":
        render_auto_research()
    elif page == "File Handoff":
        render_handoff()
    else:
        render_sessions()
