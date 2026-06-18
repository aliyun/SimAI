"""Merge per-rank CSVs into a single file. Only the ranks column varies —
instead of one file per rank, output one file where each row shows ALL groups.

Usage:
    cd aicb && python3 scripts/merge_per_rank_csv.py \
        --csv results/mocked_workload/megatron_xxx_workload.csv
"""

import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import RankGenerator, CommGroup
from utils.rank_mapper import _COMM_GROUP_TOKEN_MAP


def _detect_from_sidecar(sidecar_path):
    if not os.path.exists(sidecar_path):
        return {}
    info = {}
    with open(sidecar_path, newline='') as f:
        for row in csv.reader(f):
            if not row or row[0] == 'group':
                continue
            name, size_str = row[0], row[1]
            size = int(size_str)
            if name == 'all_nodes': info['world_size'] = size
            elif name == 'tp_group': info['tp'] = size
            elif name == 'pp_group': info['pp'] = size
            elif name == 'dp_group': info['dp_full'] = size
            elif name == 'ep_group':
                rank_groups = row[2].split()
                info['ep'] = size if len(rank_groups) > 0 and size > 1 else 1
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--pp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=None)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--order", default="tp-cp-ep-dp-pp")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sidecar = args.csv.replace('_workload.csv', '_rank_mapping.csv')
    info = _detect_from_sidecar(sidecar)
    world_size = info.get('world_size')
    tp = args.tp or info.get('tp', 1)
    pp = args.pp or info.get('pp', 1)
    ep = args.ep or info.get('ep', 1)
    if world_size is None:
        world_size = max(tp * pp, 1)
    dp = world_size // (tp * pp)

    rg = RankGenerator(tp=tp, ep=ep, dp=dp, pp=pp, cp=args.cp, order=args.order)

    # Read the base CSV (rank 0)
    with open(args.csv, newline='') as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    header = all_rows[0]
    data_rows = all_rows[1:]
    ranks_col = len(header) - 1

    # Replace 'ranks' header with 'all_groups'
    new_header = list(header)
    new_header[ranks_col] = 'all_groups'

    out_file = args.out or args.csv.replace('_workload.csv', '_merged.csv')
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)

        for row in data_rows:
            cg_raw = row[1] if len(row) > 1 else ''
            new_row = list(row)
            while len(new_row) <= ranks_col:
                new_row.append('')

            if cg_raw in ('None', '', 'CommGroup.None'):
                new_row[ranks_col] = ''
            else:
                cg_name = cg_raw.split('.')[-1]
                try:
                    cg = CommGroup(cg_name)
                except ValueError:
                    new_row[ranks_col] = ''
                    writer.writerow(new_row)
                    continue

                # Get ALL groups for this CommGroup type
                if cg == CommGroup.all:
                    all_groups_str = f"[{','.join(str(r) for r in range(world_size))}]"
                elif cg in _COMM_GROUP_TOKEN_MAP:
                    token, indep = _COMM_GROUP_TOKEN_MAP[cg]
                    groups = rg.get_ranks(token, independent_ep=indep)
                    all_groups_str = ' '.join(
                        '[' + ','.join(str(r) for r in g) + ']' for g in groups
                    )
                else:
                    all_groups_str = ''

                new_row[ranks_col] = '"' + all_groups_str + '"'

            writer.writerow(new_row)

    print(f"Merged CSV: {out_file}")
    print(f"  rows={len(data_rows)}, groups_col shows all {world_size}-rank decomposition")

    # Preview
    with open(out_file, newline='') as f:
        rows_out = list(csv.reader(f))
    print(f"\n=== Preview (first 5 comm rows) ===")
    count = 0
    for row in rows_out:
        if row[ranks_col] and row[ranks_col] != '""':
            display = row[ranks_col][:150] + ('...' if len(row[ranks_col]) > 150 else '')
            print(f"  {row[4][:35]:<35} {row[1].split('.')[-1]:<15} {display}")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
