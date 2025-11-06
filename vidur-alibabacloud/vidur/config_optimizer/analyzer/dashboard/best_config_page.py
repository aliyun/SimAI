import numpy as np
import plotly.graph_objs as go
import streamlit as st

from vidur.config_optimizer.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from vidur.config_optimizer.analyzer.constants import AXIS_COLS, AXIS_COLS_LONG_TO_SHORT
from vidur.config_optimizer.analyzer.dashboard.utils import (
    add_advanced_filters,
    add_small_divider,
    convert_config_row_to_comparison_point,
    plot_cdf,
)


def get_best_configs(
    df,
    metric_1: str,
    percentile_1: str,
    slo_1: float,
    metric_2: str,
    percentile_2: str,
    slo_2: float,
    pick_min_if_no_match: bool = True,
):
    metric_1_col = f"{metric_1}_{percentile_1}%"
    metric_2_col = f"{metric_2}_{percentile_2}%"

    slo_df = df[(df[metric_1_col] <= slo_1) & (df[metric_2_col] <= slo_2)]

    if len(slo_df) == 0 and pick_min_if_no_match:
        min_metric_1 = df[metric_1_col].min()
        slo_df = df[(df[metric_1_col] == min_metric_1) & (df[metric_2_col] < slo_2)]
        if len(slo_df) == 0:
            min_metric_2 = df[metric_2_col].min()
            slo_df = df[
                (df[metric_1_col] == min_metric_1) & (df[metric_2_col] == min_metric_2)
            ]

    # group by config key and get row with max capacity_per_dollar
    best_configs_df = (
        slo_df.sort_values("QPS per Dollar", ascending=False)
        .groupby(["Model", "Trace"])
        .first()
        .reset_index()
    )

    return best_configs_df


def plot_parallel_coordinates(best_configs_df):
    best_configs_df["trace"] = (
        best_configs_df["Model"]
        + "<br>"
        + best_configs_df["Trace"]
        + "<br>"
        + best_configs_df["capacity_per_dollar_str"]
        + " QPS/$"
    )
    best_configs_df = best_configs_df.sort_values(["Model", "cost"])
    best_configs_df["trace_id"] = (
        best_configs_df.groupby(["Model", "cost"]).ngroup() + 1
    )
    labels = {**AXIS_COLS_LONG_TO_SHORT, "trace": "Trace"}

    dimensions = []

    for label_col, label_name in labels.items():
        if label_name == "Trace":
            dimension = go.parcats.Dimension(
                values=best_configs_df[label_col],
                label=label_name,
                categoryorder="array",
                categoryarray=best_configs_df["trace"].to_list(),
            )
        else:
            dimension = go.parcats.Dimension(
                values=best_configs_df[label_col],
                label=label_name,
                categoryorder="category ascending",
            )
        dimensions.append(dimension)

    # Create parcats trace
    color = np.log(best_configs_df["trace_id"])

    fig = go.Figure(
        data=[
            go.Parcats(
                dimensions=dimensions,
                line={
                    "color": color,
                    "colorscale": "agsunset",
                },
                hoverinfo="skip",
                labelfont={
                    "size": 18,
                },
                tickfont={
                    "size": 16,
                },
                arrangement="freeform",
            )
        ]
    )
    # reduce the width of the plot
    fig.update_layout(width=1100, height=100 * len(best_configs_df))
    # remove padding from left and add to right
    fig.update_layout(margin=dict(l=0, r=50, t=30, b=20))

    st.plotly_chart(fig, use_container_width=True)


def render_config_panels(df, bottleneck_analyzer):
    cols = st.columns(len(df))

    for idx, config_row in df.iterrows():
        col = cols[idx]
        with col:
            container = st.container(border=True)
            with container:
                st.markdown(f"### {config_row['Model']} - {config_row['Trace']}")

                add_small_divider()

                st.markdown("##### Configuration Details")
                for axis_col, value in config_row[AXIS_COLS.values()].items():
                    st.write(f"**{axis_col}**: {value}")

                add_small_divider()

                st.markdown("##### Performance Metrics")

                st.write(f"**TTFT (P95)**: {config_row['TTFT_95%']:.2f} s")
                st.write(f"**TBT (P99)**: {config_row['TBT_99%']:.2f} ms")
                st.write(f"**QPS per Dollar**: {config_row['QPS per Dollar']:.2f}")
                st.write(f"**MFU**: {config_row['mfu_mean']:.2f}")
                st.write(f"**Memory Usage**: {config_row['memory_usage_mean']:.2f}%")
                st.write(f"**Busy Time**: {config_row['busy_time_percent_mean']:.2f}%")

                bottleneck_message = bottleneck_analyzer.analyze(config_row)
                st.write(f"**Bottleneck**: {bottleneck_message}")

                add_small_divider()

                # add a button to add this config to comparison
                add_to_comparison = st.button(
                    "Add to Comparison", key=f"add_to_comparison_{idx}"
                )

                if add_to_comparison:
                    st.session_state.comparison_points.append(
                        convert_config_row_to_comparison_point(config_row)
                    )

    add_small_divider()

    df["Model-Trace"] = df["Model"] + "-" + df["Trace"]

    st.markdown("### CDFs")
    plot_cdf(df, y="TTFT_cdf", y_name="TTFT", color="Model-Trace")
    plot_cdf(df, y="TBT_cdf", y_name="TBT", color="Model-Trace")
    plot_cdf(df, y="batch_size_cdf", y_name="Batch Size", color="Model-Trace")
    plot_cdf(
        df, y="batch_num_tokens_cdf", y_name="Batch Num Tokens", color="Model-Trace"
    )


def render_best_config_selection_page(df):
    # create a subpage for best config selection
    st.markdown("## Best Config Selection")
    st.markdown(
        "This tool helps you find and visualize variations in best configuration for a given model model under different workloads."
    )

    add_small_divider()
    st.markdown("### Input Parameters")

    models = df["Model"].unique()

    selected_model = st.selectbox("Select Model", models)

    col1, col2 = st.columns(2)

    with col1:
        ttft_slo_percentile = st.selectbox(
            "TTFT SLO Percentile:", [50, 75, 90, 95, 99], index=2
        )
        ttft_slo_value = st.slider(
            "TTFT SLO Value (s)", min_value=0.1, max_value=10.0, value=2.0
        )

    with col2:
        tbt_slo_percentile = st.selectbox(
            "TBT SLO Percentile:", [50, 75, 90, 95, 99], index=4
        )
        tbt_slo_value = st.slider(
            "TBT SLO Value (ms)", min_value=10, max_value=2000, value=200
        )

    df = df[(df["Model"] == selected_model)]

    filtered_df = add_advanced_filters(df)

    best_configs_df = get_best_configs(
        filtered_df,
        "TTFT",
        ttft_slo_percentile,
        ttft_slo_value,
        "TBT",
        tbt_slo_percentile,
        tbt_slo_value,
    )

    bottleneck_analyzer = BottleneckAnalyzer(
        ttft_slo_percentile,
        ttft_slo_value,
        tbt_slo_percentile,
        tbt_slo_value,
    )

    st.markdown("### Results")

    add_small_divider()

    if len(best_configs_df) > 0:
        plot_parallel_coordinates(best_configs_df)
        render_config_panels(best_configs_df, bottleneck_analyzer)
    else:
        st.markdown("No config satisfies the SLOs")
