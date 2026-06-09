import plotly.express as px
import randomname
import streamlit as st

from vidur.config_optimizer.analyzer.constants import AXIS_COLS, PRETTY_NAMES


def hide_anchor_link():
    st.markdown(
        """
        <style>
        .st-emotion-cache-15zrgzn.e1nzilvr3 {display: none}
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_small_divider():
    st.markdown(
        """<hr style="height:1px;border:none;color:#333;background-color:#333;margin-top: 3px;margin-bottom: 20px;" /> """,
        unsafe_allow_html=True,
    )


def get_model_trace_names(subset_dfs):
    model_trace_names = list(subset_dfs.keys())
    model_names = set([model for model, _ in model_trace_names])
    trace_names = set([trace for _, trace in model_trace_names])
    return model_names, trace_names


def add_model_trace_selector(subset_dfs):
    model_names, trace_names = get_model_trace_names(subset_dfs)

    col1, col2 = st.columns(2)

    with col1:
        model_select_box = st.selectbox("Model:", model_names)
    with col2:
        trace_select_box = st.selectbox("Trace:", trace_names)

    return model_select_box, trace_select_box


def get_config_name(config_row):
    axis_cols = ["Model", "Trace"] + list(AXIS_COLS.values())
    axis_dict = {col: config_row[col] for col in axis_cols}
    randomname.util.rng.seed(hash(str(axis_dict)))
    return randomname.core.generate(
        randomname.util.prefix("a", ("music_theory", "sound", "size")),
        randomname.util.prefix("n", ("cats", "food", "apex_predators")),
        sep="-",
    )


def add_advanced_filters(df):
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)

        config_filters = {}
        for i, axis_col in enumerate(AXIS_COLS.values()):
            col = col1 if i % 2 == 0 else col2
            with col:
                config_filters[axis_col] = st.multiselect(
                    axis_col, df[axis_col].unique(), key=f"{axis_col}_selector"
                )

    # apply filters
    filtered_df = df
    for axis_col, values in config_filters.items():
        if len(values) == 0:
            continue
        filtered_df = filtered_df[filtered_df[axis_col].isin(values)]

    return filtered_df


def plot_cdf(config_df, y, y_name, color):
    chart_col, _ = st.columns([2.5, 1])
    # split the cdf_series from str to float list
    config_df["cdf_series"] = config_df[y].apply(
        lambda x: list(map(float, x[1:-1].split(",")))
    )
    # create a separate row for each cdf value and add a new column cdf_x
    config_df = config_df.explode("cdf_series").reset_index(drop=True)
    config_df["cdf_x"] = config_df.groupby(
        ["Name", "poisson_request_interval_generator_qps"]
    ).cumcount()

    with chart_col:
        fig = px.line(
            config_df,
            x="cdf_x",
            y="cdf_series",
            color=color,
            labels={"cdf_series": y_name, "cdf_x": "CDF"},
            title=f"{y_name}",
        )

        fig.update_layout(
            font=dict(size=14),
            legend=dict(
                title_font=dict(size=16),
                font=dict(size=14),
            ),
            xaxis_title=y_name,
            yaxis_title="CDF",
        )

        st.plotly_chart(fig, use_container_width=True)


def write_best_config(
    bottleneck_analyzer,
    df,
    metric_1,
    percentile_1,
    slo_1,
    metric_2,
    percentile_2,
    slo_2,
    use_qps=False,
):
    metric_1_col = f"{metric_1}_{percentile_1}%"
    metric_2_col = f"{metric_2}_{percentile_2}%"

    target_col = "QPS" if use_qps else "QPS per Dollar"

    slo_compliant_df = df[(df[metric_1_col] <= slo_1) & (df[metric_2_col] <= slo_2)]
    best_config = slo_compliant_df.sort_values(target_col, ascending=False)

    if len(best_config) == 0 or best_config[target_col].iloc[0] == 0:
        st.markdown("No config satisfies the SLOs")
        return

    best_config = best_config.iloc[0]

    best_point_str = ""
    for _, axis_name in AXIS_COLS.items():
        if (
            axis_name == "Sarathi Chunk Size"
            and best_config["Scheduler"] != "Sarathi-Serve"
        ):
            continue
        value = best_config[axis_name]
        value = str(value) if not value in PRETTY_NAMES else PRETTY_NAMES[value]
        best_point_str += f"**{axis_name}**: {value}, "

    best_point_str = best_point_str[:-2]
    best_point_str += f"\n\n"

    if use_qps:
        best_point_str += f"**QPS**: {best_config['QPS']:.2f}, "
    best_point_str += f"**QPS per Dollar**: {best_config['QPS per Dollar']:.2f}, "
    best_point_str += f"**{metric_1_col}**: {best_config[metric_1_col]:.2f}, "
    best_point_str += f"**{metric_2_col}**: {best_config[metric_2_col]:.2f}"

    best_point_str += f"\n\n**MFU**: {best_config['mfu_mean']:.2f}, "
    best_point_str += f"**Memory Usage**: {best_config['memory_usage_mean']:.2f}%, "
    best_point_str += f"**Busy Time**: {best_config['busy_time_percent_mean']:.2f}%"

    bottleneck_message = bottleneck_analyzer.analyze(best_config)
    best_point_str += f"\n\n**Bottleneck**: {bottleneck_message}"

    st.markdown(f"#### Best Config \n\n{best_point_str}")


def convert_config_row_to_comparison_point(config_row):
    COMPARE_COLS = ["Model", "Trace"] + list(AXIS_COLS.values())
    return {col: config_row[col] for col in COMPARE_COLS}
