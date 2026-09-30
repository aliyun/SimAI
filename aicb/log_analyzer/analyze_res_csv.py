import pandas as pd
from log_analyzer.utils import convert_msg_to_size, convert_size_to_msg
import sys

def analyze_csv(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['busbw'])

    df['busbw'] = pd.to_numeric(df['busbw'], errors='coerce')

    df = df.dropna(subset=['busbw'])

    def exclude_min(group):
        if len(group) > 1:
            group = group.sort_values(by='busbw')
            return group.iloc[2:]
        return group

    df_excluded_min = df.groupby(['comm_type', 'comm_group', 'msg_size']).apply(exclude_min).reset_index(drop=True)
    grouped = df_excluded_min.groupby(['comm_type', 'comm_group', 'msg_size']).agg(
        busbw_mean=('busbw', 'mean'),
        busbw_max=('busbw', 'max'),
        busbw_min=('busbw', 'min'),
        busbw_std=('busbw', 'std'),
        occurrence_count=('busbw', 'size')
    ).reset_index()
    grouped['msg_size'] = grouped['msg_size'].apply(convert_size_to_msg)
    return grouped


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m log_analyzer.analyze_res_csv <path_to_csv>")
        sys.exit(1)
    grouped = analyze_csv(sys.argv[1])
    print(grouped)

