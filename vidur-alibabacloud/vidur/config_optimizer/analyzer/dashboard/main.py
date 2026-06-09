import argparse
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vidur Demo", layout="wide")

from vidur.config_optimizer.analyzer.constants import AXIS_COLS, PRETTY_NAMES
from vidur.config_optimizer.analyzer.dashboard.best_config_page import (
    render_best_config_selection_page,
)
from vidur.config_optimizer.analyzer.dashboard.config_compare_page import (
    render_config_comparison_page,
)
from vidur.config_optimizer.analyzer.dashboard.cost_analysis_page import (
    render_cost_analysis_page,
)
from vidur.config_optimizer.analyzer.dashboard.intro_page import render_intro_page
from vidur.config_optimizer.analyzer.dashboard.pareto_curve_page import (
    render_pareto_curve_page,
)
from vidur.config_optimizer.analyzer.dashboard.search_analysis_page import (
    render_search_analysis_page,
)
from vidur.config_optimizer.analyzer.dashboard.utils import (
    get_config_name,
    hide_anchor_link,
)
from vidur.config_optimizer.analyzer.utils import get_trace_name


def select_best_run(
    df: pd.DataFrame,
    scheduling_delay_slo_percentile: float,
    scheduling_delay_slo_value: float,
):
    metric_names = f"request_scheduling_delay_{scheduling_delay_slo_percentile}%"
    mask = df[metric_names] < scheduling_delay_slo_value
    if len(df[mask]) > 0:
        # multiple points satisfy the SLO
        # select the point with the highest qps
        return (
            df[mask]
            .sort_values("poisson_request_interval_generator_qps", ascending=False)
            .iloc[0]
        )
    else:
        # no point satisfy the SLO
        # select the point with the lowest scheduling delay
        return df.iloc[df[metric_names].argmin()]


@st.cache_data
def prepare_subset_dfs(_args):
    analysis_dir = f"{_args.sim_results_dir}/analysis"

    stats_file = f"{analysis_dir}/stats.csv"

    if not os.path.exists(stats_file):
        raise ValueError(f"Stats file {stats_file} does not exist")

    df = pd.read_csv(stats_file)

    df = df[df["replica_device"] != "a40"]

    OLD_GPU_COSTS = {
        "h100": 4.25,
        "a100": 2.21,
        "a40": 1.28,
    }

    NEW_GPU_COSTS = {
        "h100": 6.98,
        "a100": 3.72,
        "a40": 1.28,
    }

    # update the "cost" and "capacity_per_dollar" columns accourding to the new costs use "replica_device" as the key
    def update_cost(row):
        if row["replica_device"] == "a40":
            return row["cost"]
        return (
            NEW_GPU_COSTS[row["replica_device"]]
            * row["cost"]
            / OLD_GPU_COSTS[row["replica_device"]]
        )

    def update_capacity_per_dollar(row):
        if row["replica_device"] == "a40":
            return row["capacity_per_dollar"]
        return (
            row["capacity_per_dollar"]
            * OLD_GPU_COSTS[row["replica_device"]]
            / NEW_GPU_COSTS[row["replica_device"]]
        )

    df["cost"] = df.apply(update_cost, axis=1)
    df["capacity_per_dollar"] = df.apply(update_capacity_per_dollar, axis=1)

    # map the trace file to a human readable name
    df["Trace"] = df["trace_request_length_generator_trace_file"].apply(get_trace_name)
    # map axis columns with human readable names
    df = df.rename(columns=AXIS_COLS)
    # replace scheduler and SKU names with human readable names
    df["Scheduler"] = df["Scheduler"].replace(PRETTY_NAMES)
    df["SKU"] = df["SKU"].replace(PRETTY_NAMES)

    df["Model"] = df["replica_model_name"].str.split("/").str[1]
    df["Model"] = df["Model"].str.replace("-Instruct-hf", "")
    df["Model"] = df["Model"].str.replace("-hf", "")

    # rename all cols starting with tbt_ and ttft_ to TBT_ and TTFT_
    df = df.rename(
        columns=lambda x: x.replace("tbt_", "TBT_").replace("ttft_", "TTFT_")
    )
    # create str version of capacity_per_dollar
    df["capacity_per_dollar_str"] = df["capacity_per_dollar"].apply(
        lambda x: str(round(x, 3))
    )
    # rename capacity_per_dollar to QPS per Dollar
    df = df.rename(columns={"capacity_per_dollar": "QPS per Dollar"})
    # convert TBT_ cols to ms from seconds
    tbt_cols = [col for col in df.columns if "TBT_" in col]
    # remove tbt_cdf
    tbt_cols.remove("TBT_cdf")
    df[tbt_cols] = df[tbt_cols] * 1000

    # if scheduler is not sarathi then set sarathi chunk size to -
    df.loc[df["Scheduler"] != "Sarathi-Serve", "Sarathi Chunk Size"] = "-"

    # assign a unique name to each config
    df["Name"] = df.apply(get_config_name, axis=1)

    filtered_subset_dfs = {}

    for (model_name, trace_name), group in df.groupby(["Model", "Trace"]):
        filtered_subset_df = (
            group.groupby(list(AXIS_COLS.values()))
            .apply(
                lambda x: select_best_run(
                    x,
                    _args.scheduling_delay_slo_percentile,
                    _args.scheduling_delay_slo_value,
                )
            )
            .reset_index(drop=True)
        )
        filtered_subset_dfs[(model_name, trace_name)] = filtered_subset_df

    # combine all the filtered subset dfs
    filtered_df = pd.concat(filtered_subset_dfs.values())

    return df, filtered_df, filtered_subset_dfs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results-dir", type=str, required=True)
    parser.add_argument("--scheduling-delay-slo-percentile", type=float, default=95)
    parser.add_argument("--scheduling-delay-slo-value", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    df, filtered_df, filtered_subset_dfs = prepare_subset_dfs(args)

    hide_anchor_link()

    if "comparison_points" not in st.session_state:
        st.session_state.comparison_points = []

    page_names_to_funcs = {
        "-": render_intro_page,
        "Best Config Selection": lambda: render_best_config_selection_page(filtered_df),
        "Config Comparison": lambda: render_config_comparison_page(filtered_df),
        "Pareto Curve Analysis": lambda: render_pareto_curve_page(filtered_subset_dfs),
        "Cost Analysis": lambda: render_cost_analysis_page(filtered_subset_dfs),
    }

    if args.debug:
        page_names_to_funcs["Search Analysis"] = lambda: render_search_analysis_page(
            filtered_df
        )

    demo_name = st.sidebar.selectbox("Choose analysis mode", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()


if __name__ == "__main__":
    main()
