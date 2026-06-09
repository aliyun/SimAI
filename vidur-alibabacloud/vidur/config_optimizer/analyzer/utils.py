from paretoset import paretoset


def get_near_pareto_mask(df, metric_col, percentile, tolerance=0.1):
    metric_col = f"{metric_col}_{percentile}"

    paretoset_mask = paretoset(
        df[["capacity_per_dollar", metric_col]], sense=["max", "min"]
    )
    pareto_df = df[paretoset_mask]

    # iterate over all the rows in pareto_df and find the rows that are within tolerance of the pareto frontier
    near_pareto_mask = df["cost"] == -1

    for _, row in pareto_df.iterrows():
        # find all the rows in df that are within tolerance of the pareto frontier
        near_pareto_mask = near_pareto_mask | (
            (df["capacity_per_dollar"] >= row["capacity_per_dollar"] * (1 - tolerance))
            & (df[metric_col] <= row[metric_col] * (1 + tolerance))
        )

    return near_pareto_mask


def get_trace_name(trace_file: str) -> str:
    if "arxiv_summarization" in trace_file:
        return "ArxivSum"
    elif "lmsys_chat_1m" in trace_file:
        return "Chat1M"
    elif "bwb" in trace_file:
        return "BWB"
    else:
        return trace_file.split("/")[-1].split(".")[0]
