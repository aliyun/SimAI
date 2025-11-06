import plotly.express as px
import streamlit as st

from vidur.config_optimizer.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from vidur.config_optimizer.analyzer.constants import AXIS_COLS
from vidur.config_optimizer.analyzer.dashboard.utils import (
    add_advanced_filters,
    add_model_trace_selector,
    add_small_divider,
    convert_config_row_to_comparison_point,
    write_best_config,
)


def render_axis_comparison_bar_chart(
    df, metric_1, percentile_1, slo_1, metric_2, percentile_2, slo_2, axis_col
):
    if axis_col == "Sarathi Chunk Size":
        df = df[df["Scheduler"] == "Sarathi-Serve"]

    best_configs_df = (
        df[
            (df[f"{metric_1}_{percentile_1}%"] <= slo_1)
            & (df[f"{metric_2}_{percentile_2}%"] <= slo_2)
        ]
        .sort_values("QPS", ascending=False)
        .groupby(axis_col)
        .first()
        .reset_index()
    )

    # treat each x axis as a category
    best_configs_df[f"{axis_col}_str"] = best_configs_df[axis_col].astype(str)

    fig = px.bar(
        best_configs_df,
        x=f"{axis_col}_str",
        y="QPS",
        color=f"{axis_col}_str",
        hover_data=list(AXIS_COLS.values()),
        labels={f"{axis_col}_str": f"{axis_col}"},
        width=300,
        height=300,
    )

    fig.update_xaxes(type="category")

    st.plotly_chart(fig)

    add_small_divider()

    add_to_comparison_button = st.button(
        f"Add to Comparison", key=f"{axis_col}_add_to_comparison"
    )

    if add_to_comparison_button:
        points_to_add = best_configs_df.apply(
            convert_config_row_to_comparison_point, axis=1
        ).to_list()
        for point in points_to_add:
            if point not in st.session_state.comparison_points:
                st.session_state.comparison_points.append(point)


def render_cost_analysis(
    bottleneck_analyzer,
    df,
    metric_1,
    percentile_1,
    slo_1,
    metric_2,
    percentile_2,
    slo_2,
    cost_budget,
):
    num_replicas = cost_budget // df["hour_cost_per_replica"]
    capacity_per_replica = (
        df["poisson_request_interval_generator_qps"] / df["cluster_num_replicas"]
    )
    df["QPS"] = num_replicas * capacity_per_replica

    write_best_config(
        bottleneck_analyzer,
        df,
        metric_1,
        percentile_1,
        slo_1,
        metric_2,
        percentile_2,
        slo_2,
        use_qps=True,
    )
    cols = st.columns(3)

    for i, axis_col in enumerate(AXIS_COLS.values()):
        with cols[i % 3]:
            with st.container(border=True):
                render_axis_comparison_bar_chart(
                    df,
                    metric_1,
                    percentile_1,
                    slo_1,
                    metric_2,
                    percentile_2,
                    slo_2,
                    axis_col,
                )


def render_cost_analysis_page(subset_dfs):
    # create a subpage for best config selection
    st.markdown("## Cost Analysis")
    st.markdown(
        "This tool helps you select the best configuration based on your SLOs and cost budget. It also provides additional visualizations which allow to see the performance as a function of different config options."
    )

    add_small_divider()
    st.markdown("### Input Parameters")

    model_select_box, trace_select_box = add_model_trace_selector(subset_dfs)

    subset_df = subset_dfs[(model_select_box, trace_select_box)]

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

    cost_budget = st.slider(
        "Per Hour Cost Budget ($)", min_value=0, max_value=1000, value=100
    )

    filtered_df = add_advanced_filters(subset_df)

    bottleneck_analyzer = BottleneckAnalyzer(
        ttft_slo_percentile,
        ttft_slo_value,
        tbt_slo_percentile,
        tbt_slo_value,
    )

    st.markdown("### Results")
    add_small_divider()

    render_cost_analysis(
        bottleneck_analyzer,
        filtered_df,
        "TTFT",
        ttft_slo_percentile,
        ttft_slo_value,
        "TBT",
        tbt_slo_percentile,
        tbt_slo_value,
        cost_budget,
    )
