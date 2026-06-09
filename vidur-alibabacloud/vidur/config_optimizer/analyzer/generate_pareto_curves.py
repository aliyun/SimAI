import argparse
import os

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from paretoset import paretoset

from vidur.config_optimizer.analyzer.constants import (
    AXIS_COLS,
    CONFIG_KEY,
    METRICS,
    PRETTY_NAMES,
)
from vidur.config_optimizer.analyzer.utils import get_trace_name


def plot_pareto_curve_under_slos(
    df,
    model_name: str,
    trace_file: str,
    results_dir: str,
    metric_1: str,
    percentile_1: str,
    slo_1: float,
    metric_2: str,
    percentile_2: str,
    slo_2: float,
):
    metric_1_col = f"{metric_1}_{percentile_1}%"
    metric_2_col = f"{metric_2}_{percentile_2}%"

    # filter out the rows that are not in the pareto frontier
    metric_1_paretoset_mask = paretoset(
        df[["capacity_per_dollar", metric_1_col]], sense=["max", "min"]
    )
    metric_2_paretoset_mask = paretoset(
        df[["capacity_per_dollar", metric_2_col]], sense=["max", "min"]
    )

    metric_1_pareto_df = df[metric_1_paretoset_mask]
    metric_2_pareto_df = df[metric_2_paretoset_mask]

    # filter based on the SLOs
    slo_mask = (df[metric_1_col] <= slo_1) & (df[metric_2_col] <= slo_2)
    slo_df = df[slo_mask]

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # Legends for plots
    red_line = mlines.Line2D([], [], color="red", label="SLO Limit")
    green_patch = mpatches.Patch(color="green", alpha=0.8, label="SLO Compliant Region")
    orange_scatter = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="o",
        linestyle="None",
        markersize=10,
        alpha=0.95,
        label="Near Pareto Configs",
    )
    green_scatter = mlines.Line2D(
        [],
        [],
        color="green",
        marker="o",
        linestyle="None",
        markersize=10,
        alpha=0.95,
        label="SLO Compliant Configs",
    )
    green_star = mlines.Line2D(
        [],
        [],
        color="green",
        marker="*",
        linestyle="None",
        markersize=15,
        alpha=0.95,
        label="Best Config",
    )

    # plot the pareto curve for metric_1
    axs[0].scatter(df[metric_1_col], df["capacity_per_dollar"], color="blue", alpha=0.8)
    # draw the SLO line
    axs[0].axvline(x=slo_1, color="red", alpha=0.7)
    # add light green background for the points that satisfy the SLO
    axs[0].axvspan(0, slo_1, alpha=0.1, color="green")
    # draw pareto curve
    metric_1_pareto_df = metric_1_pareto_df.sort_values(metric_1_col)
    axs[0].plot(
        metric_1_pareto_df[metric_1_col],
        metric_1_pareto_df["capacity_per_dollar"],
        color="orange",
    )
    # draw the points that satisfy the SLO
    axs[0].scatter(
        slo_df[metric_1_col], slo_df["capacity_per_dollar"], color="green", alpha=0.8
    )

    # plot the pareto curve for metric_2
    axs[1].scatter(df[metric_2_col], df["capacity_per_dollar"], color="blue", alpha=0.8)
    # draw the SLO line
    axs[1].axvline(x=slo_2, color="red", alpha=0.7)
    # add light green background for the points that satisfy the SLO
    axs[1].axvspan(0, slo_2, alpha=0.1, color="green")
    # draw pareto curve
    metric_2_pareto_df = metric_2_pareto_df.sort_values(metric_2_col)
    axs[1].plot(
        metric_2_pareto_df[metric_2_col],
        metric_2_pareto_df["capacity_per_dollar"],
        color="orange",
    )
    # draw the points that satisfy the SLO
    axs[1].scatter(
        slo_df[metric_2_col], slo_df["capacity_per_dollar"], color="green", alpha=0.8
    )

    # plot the pareto curve for metric_1 and metric_2
    # plot all the points
    norm = mcolors.Normalize(
        vmin=slo_df["capacity_per_dollar"].min(),
        vmax=slo_df["capacity_per_dollar"].max(),
    )
    blue_to_orange_cmap = LinearSegmentedColormap.from_list("", ["blue", "orange"])
    axs[2].scatter(
        df[metric_1_col],
        df[metric_2_col],
        alpha=0.95,
        c=df["capacity_per_dollar"],
        norm=norm,
        cmap=blue_to_orange_cmap,
    )

    # add colorbar
    cbar = plt.colorbar(axs[2].collections[0], ax=axs[2])
    cbar.set_label("QPS per Dollar", rotation=270, labelpad=20, fontsize=14)

    # plot the SLOs
    axs[2].axvline(x=slo_1, color="red", alpha=0.7)
    axs[2].axhline(y=slo_2, color="red", alpha=0.7)

    # axs[2].set_xlim(0, min(1.5 * slo_1, metric_1_pareto_df[metric_1_col].max() * 1.25))
    # axs[2].set_ylim(0, min(1.5 * slo_2, metric_2_pareto_df[metric_2_col].max() * 1.25))
    axs[2].set_xlim(
        max(0.25 * slo_1, metric_1_pareto_df[metric_1_col].min() * 1.2),
        min(2 * slo_1, metric_1_pareto_df[metric_1_col].max() * 1.5),
    )
    axs[2].set_ylim(0, slo_2 * 1.5)

    # add light green background for the region that satisfies both the SLOs
    axs[2].axvspan(
        0, slo_1, alpha=0.15, color="green", ymin=0, ymax=slo_2 / axs[2].get_ylim()[1]
    )

    # highlight the best point i.e. max capacity per dollar under the SLOs
    best_points = slo_df.sort_values("capacity_per_dollar", ascending=False)
    if best_points.shape[0] > 0:
        best_point = best_points.iloc[0]
        # get the star color based on the capacity per dollar
        star_color = blue_to_orange_cmap(norm(best_point["capacity_per_dollar"]))
        # update the color in legend
        green_star.set_color(star_color)
        axs[2].scatter(
            best_point[metric_1_col],
            best_point[metric_2_col],
            color=star_color,
            s=800,
            marker="*",
            alpha=0.95,
        )

    axs[0].legend(
        handles=[red_line, green_patch, green_scatter, green_star],
        # loc="upper left",
        loc="upper right",
        fontsize=14,
    )

    # set the x and y labels
    axs[0].set_xlabel(f"{METRICS[metric_1]} - P{percentile_1} (s)")
    axs[0].set_ylabel("QPS per Dollar")
    axs[1].set_xlabel(f"{METRICS[metric_2]} - P{percentile_2} (s)")
    axs[1].set_ylabel("QPS per Dollar")
    axs[2].set_xlabel(f"{METRICS[metric_1]} - P{percentile_1} (s)")
    axs[2].set_ylabel(f"{METRICS[metric_2]} - P{percentile_2} (s)")
    # increase the font size of the labels
    for ax in axs:
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        # increase tick font size
        ax.tick_params(axis="both", which="major", labelsize=14)

    # set the x and y limits
    axs[0].set_xlim(
        max(0.25 * slo_1, metric_1_pareto_df[metric_1_col].min() * 1.2),
        min(2 * slo_1, metric_1_pareto_df[metric_1_col].max() * 1.5),
    )
    axs[1].set_xlim(0.1, slo_2 * 1.25)

    # write the best config axis values at the bottom
    if best_points.shape[0] > 0:
        best_point_str = ""
        for axis_col, axis_name in AXIS_COLS.items():
            if (
                axis_col == "sarathi_scheduler_chunk_size"
                and best_point["replica_scheduler_provider"] != "sarathi"
            ):
                continue
            value = best_point[axis_col]
            value = str(value) if not value in PRETTY_NAMES else PRETTY_NAMES[value]
            best_point_str += f"{axis_name}: {value}, "

        best_point_str = best_point_str[:-2]
        best_point_str += f"\nQPS per Dollar: {best_point['capacity_per_dollar']:.2f}"
        fig.suptitle(f"Best Config: {best_point_str}", fontsize=20)

    # tight plot layout
    plt.tight_layout()

    # save the plot
    output_dir = f"{results_dir}/pareto_curves/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        f"{output_dir}/{trace_file}_{metric_1}_{metric_2}_{percentile_1}_{percentile_2}_{slo_1}_{slo_2}.png"
    )
    plt.savefig(
        f"{output_dir}/{trace_file}_{metric_1}_{metric_2}_{percentile_1}_{percentile_2}_{slo_1}_{slo_2}.pdf"
    )


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


def process_sim_results(args: argparse.Namespace):
    analysis_dir = f"{args.sim_results_dir}/analysis"

    stats_file = f"{analysis_dir}/stats.csv"

    if not os.path.exists(stats_file):
        raise ValueError(f"Stats file {stats_file} does not exist")

    df = pd.read_csv(stats_file)
    df = df[df["sarathi_scheduler_chunk_size"] != 256]
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

    for (model_name, trace_file), group in df.groupby(
        ["replica_model_name", "trace_request_length_generator_trace_file"]
    ):
        subset_df = (
            group.groupby(CONFIG_KEY)
            .apply(
                lambda x: select_best_run(
                    x,
                    args.scheduling_delay_slo_percentile,
                    args.scheduling_delay_slo_value,
                )
            )
            .reset_index(drop=True)
        )

        trace_name = get_trace_name(trace_file)

        plot_pareto_curve_under_slos(
            subset_df,
            model_name,
            trace_name,
            analysis_dir,
            "ttft",
            args.ttft_slo_percentile,
            args.ttft_slo_value,
            "tbt",
            args.tbt_slo_percentile,
            args.tbt_slo_value,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-results-dir", type=str, required=True)
    parser.add_argument("--scheduling-delay-slo-percentile", type=float, default=99)
    parser.add_argument("--scheduling-delay-slo-value", type=float, default=5.0)
    parser.add_argument("--ttft-slo-percentile", type=float, default=90)
    parser.add_argument("--ttft-slo-value", type=float, default=2.0)
    parser.add_argument("--tbt-slo-percentile", type=float, default=99)
    parser.add_argument("--tbt-slo-value", type=float, default=0.2)
    args = parser.parse_args()

    process_sim_results(args)


if __name__ == "__main__":
    main()
