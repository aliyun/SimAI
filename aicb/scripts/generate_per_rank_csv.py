"""Generate per-rank CSV files from a rank-0 workload CSV.

Auto-detects tp/pp/ep/world_size from the sidecar _rank_mapping.csv.
Can also override via CLI arguments.

Usage:
    cd aicb && python3 scripts/generate_per_rank_csv.py --csv results/.../xxx_workload.csv
"""

import argparse, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import RankGenerator, CommGroup
from utils.rank_mapper import get_rank_list_for_comm_group


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
    parser = argparse.ArgumentParser(description="Generate per-rank workload CSVs")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--pp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=None)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--order", default="tp-cp-ep-dp-pp")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    sidecar = args.csv.replace('_workload.csv', '_rank_mapping.csv')
    info = _detect_from_sidecar(sidecar)
    world_size = info.get('world_size')
    tp = args.tp or info.get('tp', 1)
    pp = args.pp or info.get('pp', 1)
    ep = args.ep or info.get('ep', 1)

    if world_size is None:
        with open(args.csv, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            ranks_col = len(header) - 1
            max_rank = 0
            for row in reader:
                r = row[ranks_col] if len(row) > ranks_col else ''
                if r:
                    for x in r.strip('"').split(','):
                        try: max_rank = max(max_rank, int(x))
                        except: pass
            world_size = max(max_rank + 1, tp * pp)

    dp = world_size // (tp * pp)
    rg = RankGenerator(tp=tp, ep=ep, dp=dp, pp=pp, cp=args.cp, order=args.order)

    print(f"World: {rg.world_size} | tp={tp} dp={dp} pp={pp} ep={ep}")
    if info: print(f"Auto-detected from: {sidecar}")

    with open(args.csv, newline='') as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    header = all_rows[0]
    data_rows = all_rows[1:]
    ranks_col = len(header) - 1
    print(f"Input: {args.csv} ({len(data_rows)} rows)")

    outdir = args.outdir or os.path.dirname(args.csv) or '.'
    os.makedirs(outdir, exist_ok=True)
    base_name = os.path.basename(args.csv).replace('_workload.csv', '')

    for ref_rank in range(rg.world_size):
        out_file = os.path.join(outdir, f"{base_name}_rank{ref_rank}_workload.csv")
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in data_rows:
                cg_raw = row[1] if len(row) > 1 else ''
                if cg_raw in ('None', '', 'CommGroup.None'):
                    writer.writerow(row)
                    continue
                cg_name = cg_raw.split('.')[-1]
                try: cg = CommGroup(cg_name)
                except ValueError: writer.writerow(row); continue
                new_ranks = get_rank_list_for_comm_group(rg, cg, ref_rank=ref_rank)
                new_row = list(row)
                while len(new_row) <= ranks_col:
                    new_row.append('')
                new_row[ranks_col] = '"' + ','.join(str(r) for r in new_ranks) + '"'
                writer.writerow(new_row)
        tag = f"  [{ref_rank}] {out_file}"
        if ref_rank < 2 or ref_rank >= rg.world_size - 1: print(tag)
        elif ref_rank == 2: print("  ...")

    print(f"\nDone. {rg.world_size} rank CSVs written to {outdir}/")


if __name__ == "__main__":
    main()
