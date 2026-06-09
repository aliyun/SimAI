import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from paretoset import paretoset

from vidur.config_optimizer.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from vidur.config_optimizer.analyzer.constants import AXIS_COLS
from vidur.config_optimizer.analyzer.dashboard.utils import (
    add_advanced_filters,
    add_model_trace_selector,
    add_small_divider,
    write_best_config,
)


def plot_pareto_curve(df, metric, percentile, slo):
    metric_col = f"{metric}_{percentile}%"

    # filter out points which are more than 1.5x away from the SLO limit
    df = df[(df[metric_col] <= slo * 1.5)]

    paretoset_mask = paretoset(df[[metric_col, "QPS per Dollar"]], sense=["min", "max"])
    pareto_df = df[paretoset_mask]

    fig = go.Figure()

    # Scatter plot for all configurations
    configs_trace = px.scatter(
        df,
        x=metric_col,
        y="QPS per Dollar",
        hover_data=list(AXIS_COLS.values()),
    )
    # set the color of the scatter plot to blue
    configs_trace["data"][0]["marker"]["color"] = "blue"
    fig.add_trace(configs_trace["data"][0])

    # Pareto frontier
    # sort the pareto_df by metric_col
    pareto_df = pareto_df.sort_values(metric_col)

    pareto_trace = px.line(
        pareto_df,
        x=metric_col,
        y="QPS per Dollar",
        hover_data=list(AXIS_COLS.values()),
    )
    pareto_trace["data"][0]["line"]["color"] = "orange"
    fig.add_trace(pareto_trace["data"][0])

    # SLO line
    fig.add_vline(
        x=slo,
        line=dict(
            color="Red",
        ),
        name="SLO Limit",
    )

    # add a vrect for the SLO compliant region
    fig.add_vrect(
        x0=0, x1=slo, fillcolor="green", opacity=0.05, layer="below", line_width=0
    )

    # set x limit to slo * 1.25 times
    fig.update_xaxes(range=[0, slo * 1.25])

    # Layout
    fig.update_layout(
        title=f"Pareto Curve for {metric} vs. QPS per Dollar",
        xaxis_title=f"{metric} - P{percentile}",
        yaxis_title="QPS per Dollar",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
    )

    # Display
    st.plotly_chart(fig, use_container_width=True)


def plot_metrics_scatter(
    df, metric_1, percentile_1, slo_1, metric_2, percentile_2, slo_2
):
    metric_1_col = f"{metric_1}_{percentile_1}%"
    metric_2_col = f"{metric_2}_{percentile_2}%"

    slo_compliant_df = df[(df[metric_1_col] <= slo_1) & (df[metric_2_col] <= slo_2)]
    best_config = slo_compliant_df.sort_values("QPS per Dollar", ascending=False).iloc[
        0
    ]
    # convert the best config to a dataframe
    best_config = pd.DataFrame([best_config])

    fig = go.Figure()

    configs_trace = px.scatter(
        df,
        x=metric_1_col,
        y=metric_2_col,
        hover_data=list(AXIS_COLS.values()),
        color="QPS per Dollar",
    )
    fig.add_trace(configs_trace["data"][0])

    # Best configuration
    best_config_trace = px.scatter(
        best_config,
        x=metric_1_col,
        y=metric_2_col,
        hover_data=list(AXIS_COLS.values()),
    )
    # set the color of the best config to orange
    best_config_trace["data"][0]["marker"]["color"] = "orange"
    # set maker to star
    best_config_trace["data"][0]["marker"]["symbol"] = "star"
    # enlarge the size of the marker
    best_config_trace["data"][0]["marker"]["size"] = 12
    fig.add_trace(best_config_trace["data"][0])

    # SLO lines
    fig.add_shape(
        type="line",
        x0=slo_1,
        y0=0,
        y1=slo_2 * 1.25,
        x1=slo_1,
        line=dict(color="Red"),
        name=f"SLO Limit for {metric_1}",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=slo_1 * 1.25,
        y0=slo_2,
        y1=slo_2,
        line=dict(color="Red"),
        name=f"SLO Limit for {metric_2}",
    )

    # set x and y limits
    fig.update_xaxes(range=[0, slo_1 * 1.25])
    fig.update_yaxes(range=[0, slo_2 * 1.25])

    # Highlighting SLO compliant area
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=slo_1,
        y1=slo_2,
        fillcolor="green",
        opacity=0.05,
        layer="below",
        line_width=0,
    )

    # Layout
    fig.update_layout(
        title=f"{metric_1} vs. {metric_2} Colored by QPS per Dollar",
        xaxis_title=f"{metric_1} - P{percentile_1}",
        yaxis_title=f"{metric_2} - P{percentile_2}",
        coloraxis_colorbar=dict(
            title="QPS per Dollar",
            title_side="right",
        ),
    )

    # Display
    st.plotly_chart(fig, use_container_width=True)


def plot_pareto_curve_under_slos(
    bottleneck_analyzer,
    df,
    metric_1,
    percentile_1,
    slo_1,
    metric_2,
    percentile_2,
    slo_2,
):
    write_best_config(
        bottleneck_analyzer,
        df,
        metric_1,
        percentile_1,
        slo_1,
        metric_2,
        percentile_2,
        slo_2,
    )

    add_small_divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        plot_pareto_curve(df, metric_1, percentile_1, slo_1)
    with col2:
        plot_pareto_curve(df, metric_2, percentile_2, slo_2)
    with col3:
        plot_metrics_scatter(
            df, metric_1, percentile_1, slo_1, metric_2, percentile_2, slo_2
        )


def render_pareto_curve_page(subset_dfs):
    # create a subpage for best config selection
    st.markdown("## Pareto Curve Analysis")
    st.markdown(
        "This tool helps you visualize the trade-offs between different performance metrics and helps you identify the best configurations."
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

    filtered_df = add_advanced_filters(subset_df)

    bottleneck_analyzer = BottleneckAnalyzer(
        ttft_slo_percentile,
        ttft_slo_value,
        tbt_slo_percentile,
        tbt_slo_value,
    )

    st.markdown("### Results")
    add_small_divider()

    plot_pareto_curve_under_slos(
        bottleneck_analyzer,
        filtered_df,
        "TTFT",
        ttft_slo_percentile,
        ttft_slo_value,
        "TBT",
        tbt_slo_percentile,
        tbt_slo_value,
    )
