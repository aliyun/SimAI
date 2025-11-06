from itertools import product

import pandas as pd
import streamlit as st

from vidur.config_optimizer.analyzer.constants import AXIS_COLS
from vidur.config_optimizer.analyzer.dashboard.utils import add_small_divider, plot_cdf


def render_comparison_points(df):
    configs_to_remove = []
    rows_to_render = []
    COMPARE_COLS = ["Model", "Trace"] + list(AXIS_COLS.values())
    for idx, point in enumerate(st.session_state.comparison_points):
        config_row = df[(df[COMPARE_COLS] == pd.Series(point)).all(axis=1)]
        if config_row.empty:
            st.warning(f"Configuration {point} not found in the dataset")
            configs_to_remove.append(point)
            continue

        rows_to_render.append((idx, point, config_row))

    for point in configs_to_remove:
        st.session_state.comparison_points.remove(point)

    if not rows_to_render:
        st.warning("No configurations to compare")
        return

    cols = st.columns([1] * len(rows_to_render))
    num_cols = len(rows_to_render)

    for idx, (idx, point, config_row) in enumerate(rows_to_render):
        col = cols[idx]
        with col:
            container = st.container(border=True)
            with container:
                title_col, cancel_col, _ = st.columns(
                    [100 - 5 * num_cols, 5, 5 * num_cols]
                )
                with title_col:
                    st.markdown(f"### {config_row['Name'].values[0]}")
                with cancel_col:
                    remove_button = st.button("✖️", key=f"remove_{idx}")

                add_small_divider()

                st.markdown("##### Configuration Details")
                for axis_col, value in point.items():
                    st.write(f"**{axis_col}**: {value}")

                add_small_divider()

                st.markdown("##### Performance Metrics")

                st.write(f"**TTFT (P95)**: {config_row['TTFT_95%'].values[0]:.2f} s")
                st.write(f"**TBT (P99)**: {config_row['TBT_99%'].values[0]:.2f} ms")
                st.write(
                    f"**QPS per Dollar**: {config_row['QPS per Dollar'].values[0]:.2f}"
                )
                st.write(f"**MFU**: {config_row['mfu_mean'].values[0]:.2f}%")
                st.write(
                    f"**Memory Usage**: {config_row['memory_usage_mean'].values[0]:.2f}%"
                )
                st.write(
                    f"**Busy Time**: {config_row['busy_time_percent_mean'].values[0]:.2f}%"
                )

                add_small_divider()

            if remove_button:
                st.session_state.comparison_points.remove(point)
                st.experimental_rerun()

    add_small_divider()
    st.markdown("### CDFs")
    config_rows = [config_row for _, _, config_row in rows_to_render]
    config_df = pd.concat(config_rows)
    plot_cdf(config_df, y="TTFT_cdf", y_name="TTFT", color="Name")
    plot_cdf(config_df, y="TBT_cdf", y_name="TBT", color="Name")
    plot_cdf(config_df, y="batch_size_cdf", y_name="Batch Size", color="Name")
    plot_cdf(
        config_df, y="batch_num_tokens_cdf", y_name="Batch Num Tokens", color="Name"
    )


def render_config_comparison_page(df):
    st.markdown("## Config Comparison")
    st.markdown(
        'This tool enabled one-on-one performance comparison between different configurations. You can either manually add configuration on this page or use the **"Add to Configuration"** button on other pages to quickly populate the comparison.'
    )
    add_small_divider()
    st.markdown("### Select configurations to compare")

    COMPARISON_COLS = ["Model", "Trace"] + list(AXIS_COLS.values())

    comparison_features = {}
    with st.form(key="comparison_point_form"):
        col1, col2 = st.columns(2)

        for i, axis_col in enumerate(COMPARISON_COLS):
            col = col1 if i % 2 == 0 else col2
            with col:
                comparison_features[axis_col] = st.multiselect(
                    axis_col, df[axis_col].unique(), key=f"{axis_col}_selector"
                )

        # Submit button for the form
        submit_button = st.form_submit_button(label="Add Comparison Points")

    comparison_points = []
    # take the product of all the selected features
    for point in product(*comparison_features.values()):
        comparison_points.append(dict(zip(COMPARISON_COLS, point)))

    if submit_button:
        for comparison_point in comparison_points:
            if comparison_point not in st.session_state.comparison_points:
                st.session_state.comparison_points.append(comparison_point)
            else:
                st.warning(f"{comparison_point} is already added for comparison.")

    if st.session_state.comparison_points:
        title_col, clear_col = st.columns([0.9, 0.1])
        st.markdown("--- \n\n")
        with title_col:
            st.markdown("### Comparison Results")
        with clear_col:
            clear_button = st.button("Clear All", key="clear_all")
            if clear_button:
                st.session_state.comparison_points = []
                return

        render_comparison_points(df)
