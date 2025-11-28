import streamlit as st

from vidur.config_optimizer.analyzer.constants import AXIS_COLS
from vidur.config_optimizer.analyzer.dashboard.utils import add_small_divider, plot_cdf


def render_search_instances(df):
    cols = st.columns([1] * len(df))

    for idx, (_, config_row) in enumerate(df.iterrows()):
        col = cols[idx]
        with col:
            container = st.container(border=True)
            with container:
                config_name = (
                    f"QPS: {config_row['poisson_request_interval_generator_qps']:.2f}"
                )
                st.markdown(f"### {config_name}")

                add_small_divider()

                st.markdown("##### Configuration Details")
                for axis_col, value in config_row[AXIS_COLS.values()].items():
                    st.write(f"**{axis_col}**: {value}")

                add_small_divider()

                st.markdown("##### Performance Metrics")

                # extract config hash - it is part of the output dir
                # .*/runs/<config_hash>/.*
                # first find /runs/ and then take the next part
                config_hash = config_row["output_dir"].split("/runs/")[1].split("/")[0]

                st.write(f"**Config Hash**: {config_hash}")
                st.write(
                    f"**Scheduling Delay (P95)**: {config_row['request_scheduling_delay_95%']:.2f} s"
                )
                st.write(f"**TTFT (P95)**: {config_row['TTFT_95%']:.2f} s")
                st.write(f"**TBT (P99)**: {config_row['TBT_99%']:.2f} ms")
                st.write(f"**QPS per Dollar**: {config_row['QPS per Dollar']:.2f}")
                st.write(f"**MFU**: {config_row['mfu_mean']:.2f}%")
                st.write(f"**Memory Usage**: {config_row['memory_usage_mean']:.2f}%")
                st.write(f"**Busy Time**: {config_row['busy_time_percent_mean']:.2f}%")

    add_small_divider()

    df["QPS"] = df["poisson_request_interval_generator_qps"]

    st.markdown("### CDFs")
    plot_cdf(df, y="TTFT_cdf", y_name="TTFT", color="QPS")
    plot_cdf(df, y="TBT_cdf", y_name="TBT", color="QPS")
    plot_cdf(df, y="batch_size_cdf", y_name="Batch Size", color="QPS")
    plot_cdf(df, y="batch_num_tokens_cdf", y_name="Batch Num Tokens", color="QPS")


def render_search_analysis_page(df):
    st.markdown("## Search Analysis")
    st.markdown("---")
    st.markdown("This tool helps in analysis of search sequence.")

    st.markdown("--- \n\n ### Select configurations to analyze")

    COMPARISON_COLS = ["Model", "Trace"] + list(AXIS_COLS.values())

    with st.form(key="search_analysis_form"):
        col1, col2 = st.columns(2)

        config_filters = {}
        for i, axis_col in enumerate(COMPARISON_COLS):
            col = col1 if i % 2 == 0 else col2
            with col:
                config_filters[axis_col] = st.selectbox(
                    axis_col, df[axis_col].unique(), key=f"{axis_col}_selector"
                )

        # apply filters
        filtered_df = df
        for axis_col, value in config_filters.items():
            filtered_df = filtered_df[filtered_df[axis_col] == value]

        # Submit button for the form
        submit_button = st.form_submit_button(label="Add Search Runs")

    if submit_button:
        st.markdown("--- \n\n")
        st.markdown("### Results")

        render_search_instances(filtered_df)
