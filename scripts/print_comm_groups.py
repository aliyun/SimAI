#!/usr/bin/env python3
"""解析 SimAI workload 文件，打印每个阶段每个通信 group 的具体卡号和通信操作。
正确处理 PP (Pipeline Parallelism) 分组。"""

import sys
import re


def parse_header(line):
    """解析 workload 文件头"""
    info = {}
    info['tp'] = int(re.search(r'model_parallel_NPU_group:\s*(\d+)', line).group(1))
    info['ep'] = int(re.search(r'ep:\s*(\d+)', line).group(1))
    info['pp'] = int(re.search(r'pp:\s*(\d+)', line).group(1))
    info['all_gpus'] = int(re.search(r'all_gpus:\s*(\d+)', line).group(1))
    info['ga'] = int(re.search(r'ga:\s*(\d+)', line).group(1))

    m = re.search(r'pp_comm[:\s]+(\d+)', line)
    info['pp_comm_size'] = int(m.group(1)) if m else 0

    n = info['all_gpus']
    tp = info['tp']
    ep = info['ep']
    pp = info['pp']
    info['dp'] = n // (tp * ep * pp)
    info['dp_ep'] = info['dp'] * ep
    return info


def build_groups(info):
    """构建所有通信 group 的具体卡号。

    Megatron rank 布局: [TP, PP, DP]
    rank = tp_rank + pp_rank * TP + dp_rank * TP * PP
    """
    n = info['all_gpus']
    tp = info['tp']
    pp = info['pp']
    dp = info['dp']
    ep = info['ep']
    dp_ep = info['dp_ep']

    groups = {}

    # TP groups: 连续的 tp 张卡 (same pp_rank, same dp_rank)
    groups['TP'] = []
    for dp_r in range(dp):
        for pp_r in range(pp):
            base = pp_r * tp + dp_r * tp * pp
            groups['TP'].append([base + t for t in range(tp)])

    # PP groups: 同一 dp_rank 内, 跨 pp stages, 同一 tp_rank
    # 每个 PP group = tp 个 rank (每个 stage 的对应位置)
    groups['PP'] = []
    for dp_r in range(dp):
        for tp_r in range(tp):
            groups['PP'].append([
                tp_r + pp_r * tp + dp_r * tp * pp
                for pp_r in range(pp)
            ])

    # DP groups: 同一 pp_rank + 同一 tp_rank, 跨 dp_ranks
    groups['DP'] = []
    for pp_r in range(pp):
        for tp_r in range(tp):
            groups['DP'].append([
                tp_r + pp_r * tp + dp_r * tp * pp
                for dp_r in range(dp)
            ])

    # DP_EP groups: 与 DP 相同 (EP=1 时)
    groups['DP_EP'] = groups['DP'][:]

    # PP stages: 每个 stage 包含哪些卡
    groups['PP_STAGE'] = []
    for pp_r in range(pp):
        stage_ranks = []
        for dp_r in range(dp):
            for tp_r in range(tp):
                stage_ranks.append(tp_r + pp_r * tp + dp_r * tp * pp)
        groups['PP_STAGE'].append(sorted(stage_ranks))

    return groups


def determine_domain(comm_type, phase, info):
    """根据通信类型和阶段确定通信域"""
    if comm_type == 'NONE':
        return None, 0

    if 'DP_EP' in comm_type:
        return 'DP_EP', info['dp_ep']
    if '_EP' in comm_type:
        return 'EP', info['ep']

    if phase == 'wg':
        return 'DP', info['dp']
    else:
        return 'TP', info['tp']


def fmt_size(size_bytes):
    if size_bytes == 0:
        return "0"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def fmt_ranks(ranks, max_show=20):
    if len(ranks) <= max_show:
        return str(ranks)
    return f"[{ranks[0]}, {ranks[1]}, ..., {ranks[-1]}] ({len(ranks)} ranks)"


def parse_workload_line(line):
    parts = line.strip().split('\t')
    if len(parts) < 12:
        parts = line.strip().split()
    if len(parts) < 12:
        return None
    return {
        'name': parts[0],
        'fwd_comm': parts[3],
        'fwd_size': int(parts[4]),
        'ig_comm': parts[6],
        'ig_size': int(parts[7]),
        'wg_comm': parts[9],
        'wg_size': int(parts[10]),
    }


def print_groups_for_domain(domain, groups, info, indent="      "):
    """打印某个域的 group 列表, 按 PP stage 分组显示"""
    group_list = groups.get(domain, [])
    tp, pp, dp = info['tp'], info['pp'], info['dp']

    if domain == 'TP' and pp > 1:
        # 按 PP stage 分组显示 TP groups
        for pp_r in range(pp):
            stage_tp_groups = []
            for dp_r in range(dp):
                gi = dp_r * pp + pp_r
                stage_tp_groups.append((gi, group_list[gi]))
            print(f"{indent}PP Stage {pp_r}:")
            for gi, g in stage_tp_groups:
                print(f"{indent}  TP Group {gi:2d}: {fmt_ranks(g)}")
    elif domain == 'DP' and pp > 1:
        # 按 PP stage 分组显示 DP groups
        for pp_r in range(pp):
            stage_dp_groups = []
            for tp_r in range(tp):
                gi = pp_r * tp + tp_r
                stage_dp_groups.append((gi, group_list[gi]))
            if len(stage_dp_groups) > 4:
                print(f"{indent}PP Stage {pp_r}: (showing first 2 of {len(stage_dp_groups)} groups)")
                for gi, g in stage_dp_groups[:2]:
                    print(f"{indent}  DP Group {gi:2d}: {fmt_ranks(g)}")
                print(f"{indent}  ...")
            else:
                print(f"{indent}PP Stage {pp_r}:")
                for gi, g in stage_dp_groups:
                    print(f"{indent}  DP Group {gi:2d}: {fmt_ranks(g)}")
    else:
        for gi, g in enumerate(group_list):
            print(f"{indent}Group {gi:2d}: {fmt_ranks(g)}")


def print_layer_comms(layer_idx, layer, info, groups):
    """打印一个 layer 的所有通信操作"""
    phases = [
        ('fwd', '前向', layer['fwd_comm'], layer['fwd_size']),
        ('ig', '反向', layer['ig_comm'], layer['ig_size']),
        ('wg', '梯度同步', layer['wg_comm'], layer['wg_size']),
    ]

    has_comm = any(p[2] != 'NONE' for p in phases)
    if not has_comm:
        return False

    pp = info['pp']
    print(f"\n  Layer {layer_idx}: {layer['name']}")

    for phase_key, phase_name, comm_type, data_size in phases:
        if comm_type == 'NONE':
            continue

        domain, group_size = determine_domain(comm_type, phase_key, info)
        if domain is None:
            continue

        group_list = groups.get(domain, [])
        n_groups = len(group_list)
        link = "NVLink 机内" if domain == 'TP' else "NIC 跨机"

        if pp > 1 and domain == 'TP':
            groups_per_stage = n_groups // pp
            print(f"    [{phase_name}] {comm_type} | 域: {domain} ({group_size}卡/组, 每 PP Stage {groups_per_stage}组, 共{n_groups}组) | {fmt_size(data_size)} | {link}")
        else:
            print(f"    [{phase_name}] {comm_type} | 域: {domain} ({group_size}卡/组, {n_groups}组并行) | {fmt_size(data_size)} | {link}")

        print_groups_for_domain(domain, groups, info)

    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python3 print_comm_groups.py <workload_file> [--full]")
        print("  --full: 打印所有层 (默认只打印不重复的关键层)")
        sys.exit(1)

    workload_file = sys.argv[1]
    full_mode = '--full' in sys.argv

    with open(workload_file) as f:
        lines = f.readlines()

    header = lines[0].strip()
    info = parse_header(header)
    n_layers = int(lines[1].strip())
    tp, pp, dp = info['tp'], info['pp'], info['dp']

    print("=" * 80)
    print("SimAI 通信调用顺序表")
    print("=" * 80)
    print(f"\n配置: {info['all_gpus']} GPUs | TP={tp} PP={pp} DP={dp} EP={info['ep']} | GA={info['ga']}")
    if pp > 1:
        print(f"PP 通信消息大小: {fmt_size(info['pp_comm_size'])}")
        layers_per_stage = 40 // pp  # num_layers from model
        print(f"每 PP Stage 处理: {layers_per_stage} 层 (40层 / {pp} stages)")
    print(f"总 workload 层数: {n_layers}")

    groups = build_groups(info)

    # ═══════════════ 通信域定义 ═══════════════
    print("\n" + "═" * 80)
    print("通信域定义")
    print("═" * 80)

    # PP Stages
    if pp > 1:
        print(f"\n--- PP Stages ({pp} stages, 每 stage {tp * dp} 卡) ---")
        for si, stage_ranks in enumerate(groups['PP_STAGE']):
            tp_groups_in_stage = []
            for dp_r in range(dp):
                gi = dp_r * pp + si
                tp_groups_in_stage.append(groups['TP'][gi])
            print(f"  Stage {si}: {fmt_ranks(stage_ranks)}")
            for tgi, tg in enumerate(tp_groups_in_stage):
                print(f"    TP Group: {tg}")

        print(f"\n--- PP Groups (流水线连接, {len(groups['PP'])}组, 每组{pp}卡跨stage) ---")
        if len(groups['PP']) > 8:
            for gi in range(4):
                print(f"  PP Group {gi:2d}: {groups['PP'][gi]}  (Stage 0→1→2→3)")
            print(f"  ... ({len(groups['PP']) - 8} groups omitted)")
            for gi in range(len(groups['PP']) - 4, len(groups['PP'])):
                print(f"  PP Group {gi:2d}: {groups['PP'][gi]}  (Stage 0→1→2→3)")
        else:
            for gi, g in enumerate(groups['PP']):
                print(f"  PP Group {gi:2d}: {g}  (Stage 0→1→...→{pp-1})")

    # TP Groups
    print(f"\n--- TP Groups ({tp}卡/组, {len(groups['TP'])}组, NVLink 机内) ---")
    if pp > 1:
        for pp_r in range(pp):
            stage_groups = []
            for dp_r in range(dp):
                gi = dp_r * pp + pp_r
                stage_groups.append((gi, groups['TP'][gi]))
            print(f"  PP Stage {pp_r}:")
            for gi, g in stage_groups:
                print(f"    Group {gi:2d}: {g}")
    else:
        for gi, g in enumerate(groups['TP']):
            print(f"  Group {gi:2d}: {fmt_ranks(g)}")

    # DP Groups
    print(f"\n--- DP Groups ({dp}卡/组, {len(groups['DP'])}组, NIC 跨机) ---")
    if len(groups['DP']) > 12:
        for gi in range(4):
            print(f"  Group {gi:2d}: {fmt_ranks(groups['DP'][gi])}")
        print(f"  ... ({len(groups['DP']) - 8} groups omitted)")
        for gi in range(len(groups['DP']) - 4, len(groups['DP'])):
            print(f"  Group {gi:2d}: {fmt_ranks(groups['DP'][gi])}")
    else:
        for gi, g in enumerate(groups['DP']):
            print(f"  Group {gi:2d}: {fmt_ranks(g)}")

    # ═══════════════ PP 流水线调度 ═══════════════
    if pp > 1:
        print("\n" + "═" * 80)
        print("PP 流水线调度 (1F1B Schedule)")
        print("═" * 80)
        print(f"""
  PP=4 时, 4 个 Stage 不是同时做同一层, 而是流水线交错:

  时间步 →   1    2    3    4    5    6    7    8   ...
  Stage 0:  F0   F1   F2   F3   B0   B1   B2   B3  ...
  Stage 1:  --   F0   F1   F2   F3   B0   B1   B2  ...
  Stage 2:  --   --   F0   F1   F2   F3   B0   B1  ...
  Stage 3:  --   --   --   F0   F1   F2   F3   B0  ...

  F=前向  B=反向  数字=microbatch编号  --=等待(bubble)

  Stage 间通过 PP Send/Recv 传递激活值, 消息大小: {fmt_size(info['pp_comm_size'])}

  具体卡参与:""")
        for si in range(pp):
            stage = groups['PP_STAGE'][si]
            print(f"    Stage {si}: {fmt_ranks(stage)}")
            if si < pp - 1:
                next_stage = groups['PP_STAGE'][si + 1]
                print(f"      ↓ PP Send/Recv ({fmt_size(info['pp_comm_size'])}) ↓")

    # ═══════════════ 通信调用序列 ═══════════════
    all_layers = []
    for i in range(2, min(2 + n_layers, len(lines))):
        layer = parse_workload_line(lines[i])
        if layer:
            all_layers.append(layer)

    print("\n" + "═" * 80)
    print("通信调用序列 (按训练阶段, 重复层已折叠)")
    print("═" * 80)

    if pp > 1:
        print(f"\n  注意: workload 描述的是单个 PP rank 视角。")
        print(f"  实际执行时, 4 个 Stage 各自处理自己的 10 层, 通过 PP Send/Recv 串联。")
        print(f"  下面每层的 TP 通信只在该层所属的 PP Stage 内发生。\n")

    # 识别阶段
    phases = []
    mb_start = None
    for i, l in enumerate(all_layers):
        if l['name'] == 'embedding_layer':
            mb_start = i
            break

    if mb_start and mb_start > 0:
        phases.append(("阶段 0: 梯度处理 (iteration 开始)", 0, mb_start))

    if mb_start is not None:
        mb_end = mb_start
        for i in range(mb_start + 1, len(all_layers)):
            if all_layers[i]['name'] in ('embedding_layer', 'cross_entropy1'):
                mb_end = i
                break
        else:
            mb_end = len(all_layers)

        mb_size = mb_end - mb_start
        phases.append((f"阶段 1: Microbatch 前向+反向 (×{info['ga']}次, 每次{mb_size}层)", mb_start, mb_end))

    for i in range(len(all_layers) - 1, -1, -1):
        if all_layers[i]['name'] == 'cross_entropy1':
            phases.append(("阶段 2: Loss 计算 + 优化器 (iteration 结尾)", i, len(all_layers)))
            break

    for phase_name, start, end in phases:
        print(f"\n{'━' * 80}")
        print(f"  {phase_name}")
        print(f"{'━' * 80}")

        prev_pattern = None
        repeat_count = 0

        for idx in range(start, end):
            layer = all_layers[idx]
            pattern = (layer['name'], layer['fwd_comm'], layer['ig_comm'], layer['wg_comm'])

            if pattern == prev_pattern:
                repeat_count += 1
                continue

            if repeat_count > 0:
                print(f"\n    ... 以上 pattern 再重复 {repeat_count} 次 ...")
                repeat_count = 0

            print_layer_comms(idx, layer, info, groups)
            prev_pattern = pattern

        if repeat_count > 0:
            print(f"\n    ... 以上 pattern 再重复 {repeat_count} 次 ...")

    # ═══════════════ 统计 ═══════════════
    print("\n" + "═" * 80)
    print("通信统计")
    print("═" * 80)

    stats = {}
    for layer in all_layers:
        for phase_key, comm_type, data_size in [
            ('fwd', layer['fwd_comm'], layer['fwd_size']),
            ('ig', layer['ig_comm'], layer['ig_size']),
            ('wg', layer['wg_comm'], layer['wg_size']),
        ]:
            if comm_type == 'NONE':
                continue
            domain, _ = determine_domain(comm_type, phase_key, info)
            key = (domain, comm_type)
            if key not in stats:
                stats[key] = {'count': 0, 'total_bytes': 0}
            stats[key]['count'] += 1
            stats[key]['total_bytes'] += data_size

    if pp > 1 and info['pp_comm_size'] > 0:
        # PP Send/Recv 次数 = microbatches × (pp-1) stages × 2 (fwd+bwd)
        pp_send_count = info['ga'] * (pp - 1) * 2
        stats[('PP', 'SEND/RECV')] = {
            'count': pp_send_count,
            'total_bytes': pp_send_count * info['pp_comm_size'],
        }

    print(f"\n{'域':<8} {'操作':<25} {'次数':>6} {'总数据量':>12}  {'链路'}")
    print("─" * 70)
    for (domain, comm_type), s in sorted(stats.items(), key=lambda x: -x[1]['total_bytes']):
        link = "NVLink" if domain == 'TP' else ("PP P2P" if domain == 'PP' else "NIC")
        print(f"{domain:<8} {comm_type:<25} {s['count']:>6} {fmt_size(s['total_bytes']):>12}  {link}")


if __name__ == '__main__':
    main()
