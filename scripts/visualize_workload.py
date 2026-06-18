"""Workload Timeline Visualizer — parse SimAI workload file and render training timeline as HTML.

Usage:
    python scripts/visualize_workload.py example/workload_analytical.txt
    python scripts/visualize_workload.py example/workload_analytical.txt --max-layers 50
"""

import re
import sys
import os
import html
import math

# --- Workload parser ---

COMM_TYPE_TO_GROUP = {
    "ALLREDUCE": "TP", "ALLGATHER": "TP", "REDUCESCATTER": "TP", "ALLTOALL": "TP",
    "ALLREDUCEALLTOALL": "TP",
    "ALLREDUCE_EP": "EP", "ALLGATHER_EP": "EP", "REDUCESCATTER_EP": "EP",
    "ALLTOALL_EP": "EP", "ALLREDUCEALLTOALL_EP": "EP",
    "ALLREDUCE_DP_EP": "DP_EP", "ALLGATHER_DP_EP": "DP_EP",
    "REDUCESCATTER_DP_EP": "DP_EP", "ALLTOALL_DP_EP": "DP_EP",
    "ALLREDUCEALLTOALL_DP_EP": "DP_EP",
    "NONE": "NONE",
}

# For wg phase, bare names (ALLREDUCE, REDUCESCATTER etc) default to DP domain
COMM_TYPE_TO_GROUP_WG = dict(COMM_TYPE_TO_GROUP)
COMM_TYPE_TO_GROUP_WG.update({
    "ALLREDUCE": "DP", "ALLGATHER": "DP", "REDUCESCATTER": "DP",
    "ALLTOALL": "DP", "ALLREDUCEALLTOALL": "DP",
})

def base_comm_type(comm_type_s: str) -> str:
    """Strip domain suffix: ALLTOALL_EP → ALLTOALL, REDUCESCATTER_DP_EP → REDUCESCATTER."""
    for suffix in ("_DP_EP", "_EP"):
        if comm_type_s.endswith(suffix):
            return comm_type_s[:-len(suffix)]
    return comm_type_s


def parse_workload(path: str) -> dict:
    """Parse a SimAI workload file into structured data."""
    with open(path) as f:
        header_line = f.readline().strip()
        count_line = f.readline().strip()
        layer_lines = [l.strip() for l in f if l.strip()]

    # Parse header
    header = {}
    tokens = header_line.split()
    header["policy"] = tokens[0]
    for i in range(1, len(tokens) - 1, 2):
        key = tokens[i].rstrip(":")
        header[key] = tokens[i + 1]

    tp_size = int(header.get("model_parallel_NPU_group", 1))
    ep_size = int(header.get("ep", 1))
    pp_size = int(header.get("pp", 1))
    all_gpus = int(header.get("all_gpus", 0))
    # dp_size computed below after all fields are ready

    num_layers = int(count_line)

    layers = []
    for line in layer_lines[:num_layers]:
        parts = line.split()
        name = parts[0]
        # depen = int(parts[1])
        fp_compute = int(parts[2])
        fp_comm_type = parts[3]
        fp_comm_size = int(parts[4])
        ig_compute = int(parts[5])
        ig_comm_type = parts[6]
        ig_comm_size = int(parts[7])
        wg_compute = int(parts[8])
        wg_comm_type = parts[9]
        wg_comm_size = int(parts[10])

        layers.append({
            "name": name,
            "fp_compute": fp_compute,
            "fp_comm_type": fp_comm_type,
            "fp_comm_size": fp_comm_size,
            "fp_group": COMM_TYPE_TO_GROUP.get(fp_comm_type, "NONE"),
            "ig_compute": ig_compute,
            "ig_comm_type": ig_comm_type,
            "ig_comm_size": ig_comm_size,
            "ig_group": COMM_TYPE_TO_GROUP.get(ig_comm_type, "NONE"),
            "wg_compute": wg_compute,
            "wg_comm_type": wg_comm_type,
            "wg_comm_size": wg_comm_size,
            "wg_group": COMM_TYPE_TO_GROUP_WG.get(wg_comm_type, "NONE"),
        })

    # C++ constraint: TP_size * DP_size * PP_size = ngpus; EP_size * DP_EP_size = DP_size
    dp_full = all_gpus // (tp_size * pp_size) if all_gpus > 0 and tp_size * pp_size > 0 else 1
    dp_ep_size = dp_full // ep_size if ep_size > 0 else 1

    return {
        "header": header,
        "tp_size": tp_size,
        "dp_size": dp_full,      # full DP group size (= EP * DP_EP)
        "dp_ep_size": dp_ep_size, # DP_EP group size
        "ep_size": ep_size,
        "pp_size": pp_size,
        "all_gpus": all_gpus,
        "layers": layers,
    }


def format_bytes(n: int) -> str:
    if n == 0:
        return "0"
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.1f}GB"
    if n >= 1024 * 1024:
        return f"{n / (1024**2):.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n}B"


# --- Compute communication group ranks (mirrors MockNcclGroup.cc logic) ---

def compute_group_ranks(rank: int, tp_size: int, dp_size: int, ep_size: int,
                        dp_ep_size: int, all_gpus: int) -> dict:
    """For a given rank, compute its TP/DP/EP/DP_EP group members.

    Logic matches MockNcclGroup::MockNcclGroup() in MockNcclGroup.cc:
      - TP: consecutive blocks of tp_size  →  rank = group_idx * tp_size + j
      - DP: strided by dp_nums             →  rank = pos + j * dp_nums
      - EP: across ep_size consecutive TP groups, same intra-TP position
      - DP_EP: across dp_ep_size TP groups with stride ep_size, same position
    """
    groups = {}

    # TP group: consecutive
    if tp_size > 1:
        tp_group_idx = rank // tp_size
        groups["TP"] = [tp_group_idx * tp_size + j for j in range(tp_size)]
    else:
        groups["TP"] = [rank]

    # DP group: strided
    dp_nums = all_gpus // dp_size  # == tp_size * pp_size
    if dp_size > 1:
        pos = rank % dp_nums
        groups["DP"] = [pos + j * dp_nums for j in range(dp_size)]
    else:
        groups["DP"] = [rank]

    # EP group: based on TP groups
    tp_group_idx = rank // tp_size
    intra_tp_pos = rank % tp_size
    tp_nums = all_gpus // tp_size
    if ep_size > 1:
        ep_block_start = (tp_group_idx // ep_size) * ep_size
        groups["EP"] = [
            (ep_block_start + l) * tp_size + intra_tp_pos
            for l in range(ep_size)
        ]
    else:
        groups["EP"] = [rank]

    # DP_EP group: stride by ep_size across TP groups
    if dp_ep_size > 1:
        tp_base = tp_group_idx % ep_size  # position within EP block
        groups["DP_EP"] = [
            (tp_base + l * ep_size) * tp_size + intra_tp_pos
            for l in range(dp_ep_size)
        ]
    else:
        groups["DP_EP"] = [rank]

    return groups





def build_groups_from_cpp(tp_size: int, dp_size: int, ep_size: int,
                          dp_ep_size: int, all_gpus: int) -> dict:
    """Build communication groups exactly mirroring MockNcclGroup.cc constructor.

    Faithfully reproduces the nested loop structure from MockNcclGroup::MockNcclGroup(),
    including GroupIndex overwrite behavior for overlapping DP_EP groups.

    Returns dict: {"TP": [[ranks], ...], "DP": [[ranks], ...], ...}
    Each value is a list of unique groups (deduplicated, ordered by first rank).
    """
    tp_nums = all_gpus // tp_size
    dp_nums = all_gpus // dp_size

    # Build TP group rank lists first (EP and DP_EP reference them)
    tp_grp = {}  # tp_group_index -> [ranks]
    for i in range(tp_nums):
        tp_grp[i] = [i * tp_size + j for j in range(tp_size)]

    # Track final group assignment per (rank, domain).
    # C++ uses GroupIndex[{rank, type}] = group_idx which gets overwritten
    # by later loop iterations. We track the group members directly.
    final = {}  # (rank, domain) -> tuple(ranks)

    # === TP: consecutive blocks (MockNcclGroup.cc L51-71) ===
    # for(int i=0; i<TP_nums; i++) { rank = i*TP_size+j; }
    if tp_size > 1:
        for i in range(tp_nums):
            key = tuple(tp_grp[i])
            for r in key:
                final[(r, "TP")] = key

    # === DP: strided (MockNcclGroup.cc L73-92) ===
    # for(int i=0; i<DP_nums; i++) { rank = i+j*DP_nums; }
    if dp_size > 1:
        for i in range(dp_nums):
            ranks = tuple(i + j * dp_nums for j in range(dp_size))
            for r in ranks:
                final[(r, "DP")] = ranks

    # === EP: based on TP groups (MockNcclGroup.cc L98-131) ===
    # Outer: i over EP blocks (TP_nums/EP_size)
    # Inner j loop is redundant (creates identical overwrites), skipped.
    # k iterates intra-TP positions.
    # l iterates TP groups within the EP block.
    if ep_size > 1:
        for i in range(tp_nums // ep_size):
            tp_idx = i * ep_size
            for k in range(tp_size):
                ranks = tuple(
                    tp_grp[l][k]
                    for l in range(tp_idx, tp_idx + ep_size)
                )
                for r in ranks:
                    final[(r, "EP")] = ranks

    # === DP_EP: stride by EP_size (MockNcclGroup.cc L132-158) ===
    # for(int i=0; i<TP_nums/DP_EP_size; i++) { TP_idx = i; }
    # Inner j loop is redundant (creates identical overwrites), skipped.
    # l strides by EP_size: l = TP_idx, TP_idx+EP_size, ..., TP_idx+(DP_EP-1)*EP_size
    # NOTE: Later i values may overwrite GroupIndex set by earlier i values,
    # causing some ranks to change groups. This faithfully mirrors the C++ behavior.
    if dp_ep_size > 1:
        for i in range(tp_nums // dp_ep_size):
            tp_idx = i
            for k in range(tp_size):
                ranks = tuple(
                    tp_grp[l][k]
                    for l in range(tp_idx, tp_idx + dp_ep_size * ep_size, ep_size)
                )
                for r in ranks:
                    final[(r, "DP_EP")] = ranks

    # Deduplicate: collect unique groups per domain, ordered by first rank
    result = {}
    for domain in ("TP", "DP", "EP", "DP_EP"):
        seen = set()
        groups = []
        for r in range(all_gpus):
            g = final.get((r, domain))
            if g is not None and g not in seen:
                seen.add(g)
                groups.append(list(g))
        result[domain] = groups

    return result


def format_ranks(ranks: list, max_show: int = 16) -> str:
    """Format rank list for display, truncating if too many."""
    if len(ranks) <= max_show:
        return "[" + ", ".join(str(r) for r in ranks) + "]"
    shown = ranks[:max_show // 2] + ranks[-max_show // 2:]
    return ("[" + ", ".join(str(r) for r in ranks[:max_show // 2])
            + f", ... ({len(ranks) - max_show} more) ..., "
            + ", ".join(str(r) for r in ranks[-max_show // 2:]) + "]")


def print_groups(workload: dict, rank: int = 0, max_layers: int = 0):
    """Print communication domain groups (mirrors MockNcclGroup.cc exactly)."""
    tp = workload["tp_size"]
    dp = workload["dp_size"]
    ep = workload["ep_size"]
    dp_ep = workload["dp_ep_size"]
    pp = workload["pp_size"]
    all_gpus = workload["all_gpus"]
    layers = workload["layers"]
    if max_layers > 0:
        layers = layers[:max_layers]

    # Build ALL groups from C++ logic
    all_domain_groups = build_groups_from_cpp(tp, dp, ep, dp_ep, all_gpus)

    # Header
    print(f"\n{'='*100}")
    print(f"SimAI Communication Domain Groups  (mirroring MockNcclGroup.cc)")
    print(f"{'='*100}")
    print(f"  all_gpus={all_gpus}  TP={tp}  DP={dp}  EP={ep}  DP_EP={dp_ep}  PP={pp}")
    print(f"  Constraint check: TP×DP×PP = {tp}×{dp}×{pp} = {tp*dp*pp}"
          f" {'✓' if tp*dp*pp == all_gpus else '✗'}"
          f"  EP×DP_EP = {ep}×{dp_ep} = {ep*dp_ep}"
          f" {'✓' if ep*dp_ep == dp else '✗'}")

    # Print domain group listings
    for domain in ("TP", "DP", "EP", "DP_EP"):
        groups_list = all_domain_groups[domain]
        if not groups_list:
            print(f"\n--- {domain}: (disabled, size=1) ---")
            continue
        group_size = len(groups_list[0])
        num_groups = len(groups_list)
        total_covered = num_groups * group_size
        coverage = f"covers {total_covered}/{all_gpus} ranks"

        print(f"\n--- {domain}: {num_groups} groups × {group_size} ranks  ({coverage}) ---")

        # Show first N and last M groups
        show_first = min(5, num_groups)
        show_last = min(3, max(0, num_groups - show_first))
        for idx in range(show_first):
            print(f"  Group {idx:>5d}: {format_ranks(groups_list[idx], max_show=20)}")
        if num_groups > show_first + show_last:
            print(f"  ... ({num_groups - show_first - show_last} more groups) ...")
        for idx in range(num_groups - show_last, num_groups):
            if idx >= show_first:
                print(f"  Group {idx:>5d}: {format_ranks(groups_list[idx], max_show=20)}")

    # AllGroups Registry (sequential numbering across all domains, like C++ all_group_idx)
    print(f"\n{'='*100}")
    print(f"AllGroups Registry  (mirrors C++ AllGroups[all_group_idx])")
    print(f"{'='*100}")
    all_group_idx = 0
    for domain in ("TP", "DP", "EP", "DP_EP"):
        groups_list = all_domain_groups[domain]
        if not groups_list:
            continue
        domain_start = all_group_idx
        for idx, g in enumerate(groups_list):
            all_group_idx += 1
        domain_end = all_group_idx - 1
        print(f"\n  {domain}: groups [{domain_start} .. {domain_end}]  "
              f"({len(groups_list)} groups × {len(groups_list[0])} ranks)")
        # Show first 10 + last 5
        show_first = min(10, len(groups_list))
        show_last = min(5, max(0, len(groups_list) - show_first))
        for idx in range(show_first):
            print(f"    Group {domain_start + idx:>6d} [{domain}]: "
                  f"{format_ranks(groups_list[idx], max_show=24)}")
        if len(groups_list) > show_first + show_last:
            print(f"    ... ({len(groups_list) - show_first - show_last} more groups) ...")
        for idx in range(len(groups_list) - show_last, len(groups_list)):
            if idx >= show_first:
                print(f"    Group {domain_start + idx:>6d} [{domain}]: "
                      f"{format_ranks(groups_list[idx], max_show=24)}")
    print(f"\n  Total: {all_group_idx} groups across all domains")

    # Viewing rank's specific groups
    print(f"\n{'='*100}")
    print(f"Rank {rank}'s Group Membership")
    print(f"{'='*100}")
    for domain in ("TP", "DP", "EP", "DP_EP"):
        groups_list = all_domain_groups[domain]
        if not groups_list:
            print(f"  {domain}: (disabled, size=1)")
            continue
        # Find which group this rank belongs to
        found_idx = None
        found_group = None
        for idx, g in enumerate(groups_list):
            if rank in g:
                found_idx = idx
                found_group = g
                break
        if found_group:
            print(f"  {domain}: Group #{found_idx} — {format_ranks(found_group, max_show=24)}")
        else:
            print(f"  {domain}: rank {rank} not found in any group")

    # Per-layer communication info
    print(f"\n{'='*100}")
    print(f"Per-Layer Communication  (showing {len(layers)} layers, from rank {rank}'s perspective)")
    print(f"{'='*100}")
    print(f"{'Layer':<30s} {'Phase':<6s} {'CommType':<20s} {'Domain':<7s} "
          f"{'Size':<10s} {'Rank '+str(rank)+"'s Group"}")
    print("-" * 100)

    # Pre-build rank's group lookup per domain
    rank_group_lookup = {}
    for domain in ("TP", "DP", "EP", "DP_EP"):
        for g in all_domain_groups.get(domain, []):
            if rank in g:
                rank_group_lookup[domain] = g
                break

    for i, layer in enumerate(layers):
        phases = [
            ("fwd", layer["fp_comm_type"], layer["fp_group"], layer["fp_comm_size"]),
            ("ig",  layer["ig_comm_type"], layer["ig_group"], layer["ig_comm_size"]),
            ("wg",  layer["wg_comm_type"], layer["wg_group"], layer["wg_comm_size"]),
        ]
        for phase, comm_type, group, comm_size in phases:
            if comm_type == "NONE" or comm_size == 0:
                continue
            name_display = f"[{i}] {layer['name']}"
            rg = rank_group_lookup.get(group)
            if rg:
                grp_str = format_ranks(rg, max_show=16)
            else:
                grp_str = f"[{rank}]"
            print(f"{name_display:<30s} {phase:<6s} {comm_type:<20s} {group:<7s} "
                  f"{format_bytes(comm_size):<10s} {grp_str}")


# --- Groups HTML visualization ---

def render_groups_html(workload: dict, rank: int = 0, max_layers: int = 0) -> str:
    """Render an interactive HTML page showing GPU topology and per-layer comm groups."""
    tp = workload["tp_size"]
    dp = workload["dp_size"]
    ep = workload["ep_size"]
    dp_ep = workload["dp_ep_size"]
    pp = workload["pp_size"]
    all_gpus = workload["all_gpus"]
    layers = workload["layers"]
    if max_layers > 0:
        layers = layers[:max_layers]

    groups = compute_group_ranks(rank, tp, dp, ep, dp_ep, all_gpus)
    gpus_per_server = tp  # TP group is intra-server in most configs
    # Heuristic: gpus_per_server = tp if tp <= 8, else 8
    if tp > 8:
        gpus_per_server = 8
    elif tp < 1:
        gpus_per_server = 1
    num_servers = all_gpus // gpus_per_server

    # For the GPU grid, limit display to a manageable number
    max_display_gpus = min(all_gpus, 256)
    max_display_servers = max_display_gpus // gpus_per_server
    truncated = all_gpus > max_display_gpus

    # Build layer comm entries for the table
    layer_entries = []
    import json as _json
    for i, layer in enumerate(layers):
        phases = [
            ("fwd", layer["fp_comm_type"], layer["fp_group"], layer["fp_comm_size"]),
            ("ig",  layer["ig_comm_type"], layer["ig_group"], layer["ig_comm_size"]),
            ("wg",  layer["wg_comm_type"], layer["wg_group"], layer["wg_comm_size"]),
        ]
        for phase, comm_type, group, comm_size in phases:
            if comm_type == "NONE" or comm_size == 0:
                continue
            rank_list = groups.get(group, [rank])
            layer_entries.append({
                "idx": i,
                "name": layer["name"],
                "phase": phase,
                "comm_type": base_comm_type(comm_type),
                "group": group,
                "comm_size": comm_size,
                "comm_size_fmt": format_bytes(comm_size),
                "ranks": rank_list,
            })

    # Prepare JSON data for JavaScript
    groups_json = _json.dumps({k: v for k, v in groups.items()})
    entries_json = _json.dumps([
        {"idx": e["idx"], "name": e["name"], "phase": e["phase"],
         "comm_type": e["comm_type"], "group": e["group"],
         "comm_size_fmt": e["comm_size_fmt"],
         "ranks": e["ranks"]}
        for e in layer_entries
    ])

    group_colors_json = _json.dumps(GROUP_COLORS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SimAI Communication Groups — Rank {rank}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=DM+Sans:wght@400;500;600;700&display=swap');

  :root {{
    --bg: #06080d;
    --surface: #0d1117;
    --surface2: #161b22;
    --border: #21262d;
    --border-bright: #30363d;
    --text: #e6edf3;
    --text2: #8b949e;
    --text3: #484f58;
    --tp: #58a6ff;
    --dp: #d29922;
    --ep: #3fb950;
    --dp-ep: #a371f7;
    --pp: #f85149;
    --inactive: #0d1117;
    --active-glow: rgba(88, 166, 255, 0.15);
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', system-ui, sans-serif;
    min-height: 100vh;
  }}

  .header {{
    padding: 32px 40px 24px;
    border-bottom: 1px solid var(--border);
  }}

  .header h1 {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
  }}

  .header h1 span {{
    color: var(--tp);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
  }}

  .config-chips {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }}

  .chip {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text2);
  }}

  .chip b {{ color: var(--text); font-weight: 600; }}

  .main {{
    display: grid;
    grid-template-columns: 1fr 420px;
    min-height: calc(100vh - 100px);
  }}

  /* --- GPU Grid --- */
  .gpu-panel {{
    padding: 28px 32px;
    border-right: 1px solid var(--border);
    overflow-y: auto;
  }}

  .section-title {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }}

  .group-summary {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
  }}

  .group-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.2s;
  }}

  .group-card:hover, .group-card.active {{
    border-color: var(--card-color, var(--border-bright));
    box-shadow: 0 0 20px var(--card-glow, transparent);
  }}

  .group-card .grp-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 2px;
  }}

  .group-card .grp-size {{
    font-size: 11px;
    color: var(--text3);
  }}

  .gpu-grid {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}

  .server-row {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}

  .server-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--text3);
    width: 48px;
    text-align: right;
    flex-shrink: 0;
  }}

  .gpu-cells {{
    display: flex;
    gap: 2px;
  }}

  .gpu-cell {{
    width: 20px;
    height: 16px;
    border-radius: 2px;
    background: var(--surface2);
    border: 1px solid var(--border);
    transition: all 0.15s;
    cursor: default;
    position: relative;
  }}

  .gpu-cell.highlighted {{
    border-color: var(--hl-color, var(--tp));
    background: var(--hl-color, var(--tp));
    opacity: 0.85;
    box-shadow: 0 0 6px var(--hl-color, var(--tp));
  }}

  .gpu-cell.is-rank {{
    border: 2px solid #fff;
    z-index: 2;
  }}

  .gpu-cell .gpu-tip {{
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface2);
    border: 1px solid var(--border-bright);
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--text);
    white-space: nowrap;
    z-index: 100;
    pointer-events: none;
  }}

  .gpu-cell:hover .gpu-tip {{ display: block; }}

  .truncation-notice {{
    font-size: 11px;
    color: var(--text3);
    font-style: italic;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
  }}

  /* --- Layer List --- */
  .layer-panel {{
    padding: 28px 24px;
    overflow-y: auto;
    max-height: calc(100vh - 100px);
  }}

  .layer-list {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}

  .layer-row {{
    display: grid;
    grid-template-columns: 28px 1fr 42px 70px 80px;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 6px;
    background: var(--surface);
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.15s;
    font-size: 12px;
  }}

  .layer-row:hover {{
    border-color: var(--border-bright);
    background: var(--surface2);
  }}

  .layer-row.selected {{
    border-color: var(--row-color, var(--tp));
    background: var(--surface2);
    box-shadow: inset 3px 0 0 var(--row-color, var(--tp));
  }}

  .layer-row .row-idx {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text3);
  }}

  .layer-row .row-name {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .layer-row .row-phase {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    text-align: center;
    border-radius: 3px;
    padding: 1px 4px;
  }}

  .phase-fwd {{ background: rgba(88,166,255,0.12); color: var(--tp); }}
  .phase-ig  {{ background: rgba(248,81,73,0.12); color: var(--pp); }}
  .phase-wg  {{ background: rgba(210,153,34,0.12); color: var(--dp); }}

  .layer-row .row-group {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
  }}

  .layer-row .row-size {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: var(--text2);
    text-align: right;
  }}

  .legend-bar {{
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: var(--text2);
  }}

  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }}

  .detail-box {{
    margin-top: 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    line-height: 1.7;
    color: var(--text2);
    min-height: 60px;
  }}

  .detail-box .detail-label {{ color: var(--text3); }}
  .detail-box .detail-value {{ color: var(--text); }}
  .detail-box .detail-ranks {{
    margin-top: 8px;
    max-height: 120px;
    overflow-y: auto;
    word-break: break-all;
    color: var(--text2);
    font-size: 10px;
    line-height: 1.5;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Communication Groups — Rank <span>{rank}</span></h1>
  <div class="config-chips">
    <div class="chip">GPUs: <b>{all_gpus}</b></div>
    <div class="chip">TP: <b>{tp}</b></div>
    <div class="chip">DP: <b>{dp}</b></div>
    <div class="chip">EP: <b>{ep}</b></div>
    <div class="chip">DP_EP: <b>{dp_ep}</b></div>
    <div class="chip">PP: <b>{pp}</b></div>
    <div class="chip">Layers: <b>{len(layers)}</b></div>
    <div class="chip">GPUs/Server: <b>{gpus_per_server}</b></div>
  </div>
</div>

<div class="main">
  <div class="gpu-panel">
    <div class="section-title">Group Membership</div>
    <div class="group-summary" id="groupCards">
      <div class="group-card" data-group="TP"
           style="--card-color:var(--tp);--card-glow:rgba(88,166,255,0.2)">
        <div class="grp-name" style="color:var(--tp)">TP</div>
        <div class="grp-size">size {len(groups['TP'])}</div>
      </div>
      <div class="group-card" data-group="DP"
           style="--card-color:var(--dp);--card-glow:rgba(210,153,34,0.2)">
        <div class="grp-name" style="color:var(--dp)">DP</div>
        <div class="grp-size">size {len(groups['DP'])}</div>
      </div>
      <div class="group-card" data-group="EP"
           style="--card-color:var(--ep);--card-glow:rgba(63,185,80,0.2)">
        <div class="grp-name" style="color:var(--ep)">EP</div>
        <div class="grp-size">size {len(groups['EP'])}</div>
      </div>
      <div class="group-card" data-group="DP_EP"
           style="--card-color:var(--dp-ep);--card-glow:rgba(163,113,247,0.2)">
        <div class="grp-name" style="color:var(--dp-ep)">DP_EP</div>
        <div class="grp-size">size {len(groups['DP_EP'])}</div>
      </div>
    </div>

    <div class="section-title">GPU Topology</div>
    <div class="gpu-grid" id="gpuGrid"></div>
    {'<div class="truncation-notice">Showing first ' + str(max_display_gpus) + ' of ' + str(all_gpus) + ' GPUs (' + str(max_display_servers) + ' of ' + str(num_servers) + ' servers)</div>' if truncated else ''}
  </div>

  <div class="layer-panel">
    <div class="section-title">Layer Communications</div>
    <div class="legend-bar">
      <div class="legend-item"><div class="legend-dot" style="background:var(--tp)"></div>TP</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--dp)"></div>DP</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--ep)"></div>EP</div>
      <div class="legend-item"><div class="legend-dot" style="background:var(--dp-ep)"></div>DP_EP</div>
    </div>
    <div class="layer-list" id="layerList"></div>
    <div class="detail-box" id="detailBox">
      <span style="color:var(--text3)">Click a layer to see its communication group details.</span>
    </div>
  </div>
</div>

<script>
const RANK = {rank};
const GPUS_PER_SERVER = {gpus_per_server};
const ALL_GPUS = {all_gpus};
const MAX_DISPLAY = {max_display_gpus};
const GROUPS = {groups_json};
const ENTRIES = {entries_json};
const COLORS = {group_colors_json};
const CSS_COLORS = {{ TP: 'var(--tp)', DP: 'var(--dp)', EP: 'var(--ep)', DP_EP: 'var(--dp-ep)' }};

// Build GPU grid
const gridEl = document.getElementById('gpuGrid');
const gpuCells = {{}};
const numServers = Math.min(Math.ceil(MAX_DISPLAY / GPUS_PER_SERVER), Math.ceil(ALL_GPUS / GPUS_PER_SERVER));

for (let s = 0; s < numServers; s++) {{
  const row = document.createElement('div');
  row.className = 'server-row';
  const label = document.createElement('div');
  label.className = 'server-label';
  label.textContent = 'S' + s;
  row.appendChild(label);

  const cells = document.createElement('div');
  cells.className = 'gpu-cells';
  for (let g = 0; g < GPUS_PER_SERVER; g++) {{
    const r = s * GPUS_PER_SERVER + g;
    if (r >= ALL_GPUS) break;
    const cell = document.createElement('div');
    cell.className = 'gpu-cell' + (r === RANK ? ' is-rank' : '');
    cell.dataset.rank = r;

    const tip = document.createElement('div');
    tip.className = 'gpu-tip';
    tip.textContent = 'Rank ' + r;
    cell.appendChild(tip);

    cells.appendChild(cell);
    if (r < MAX_DISPLAY) gpuCells[r] = cell;
  }}
  row.appendChild(cells);
  gridEl.appendChild(row);
}}

// Highlight function
function highlightRanks(ranks, group) {{
  // Clear all
  Object.values(gpuCells).forEach(c => {{
    c.classList.remove('highlighted');
    c.style.removeProperty('--hl-color');
  }});

  const color = COLORS[group] || '#58a6ff';
  ranks.forEach(r => {{
    if (gpuCells[r]) {{
      gpuCells[r].classList.add('highlighted');
      gpuCells[r].style.setProperty('--hl-color', color);
    }}
  }});
}}

// Group cards click
document.querySelectorAll('.group-card').forEach(card => {{
  card.addEventListener('click', () => {{
    document.querySelectorAll('.group-card').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.layer-row').forEach(r => r.classList.remove('selected'));
    card.classList.add('active');
    const grp = card.dataset.group;
    highlightRanks(GROUPS[grp], grp);
    const ranks = GROUPS[grp];
    document.getElementById('detailBox').innerHTML =
      '<span class="detail-label">Group:</span> <span class="detail-value">' + grp + '</span><br>' +
      '<span class="detail-label">Size:</span> <span class="detail-value">' + ranks.length + ' ranks</span>' +
      '<div class="detail-ranks">[' + ranks.join(', ') + ']</div>';
  }});
}});

// Build layer list
const listEl = document.getElementById('layerList');
ENTRIES.forEach((e, idx) => {{
  const row = document.createElement('div');
  row.className = 'layer-row';
  row.style.setProperty('--row-color', CSS_COLORS[e.group] || 'var(--tp)');

  row.innerHTML =
    '<div class="row-idx">' + e.idx + '</div>' +
    '<div class="row-name">' + e.name + '</div>' +
    '<div class="row-phase phase-' + e.phase + '">' + e.phase.toUpperCase() + '</div>' +
    '<div class="row-group" style="color:' + (COLORS[e.group] || '#8b949e') + '">' + e.group + '</div>' +
    '<div class="row-size">' + e.comm_size_fmt + '</div>';

  row.addEventListener('click', () => {{
    document.querySelectorAll('.layer-row').forEach(r => r.classList.remove('selected'));
    document.querySelectorAll('.group-card').forEach(c => c.classList.remove('active'));
    row.classList.add('selected');
    highlightRanks(e.ranks, e.group);
    document.getElementById('detailBox').innerHTML =
      '<span class="detail-label">Layer:</span> <span class="detail-value">[' + e.idx + '] ' + e.name + '</span><br>' +
      '<span class="detail-label">Phase:</span> <span class="detail-value">' + e.phase.toUpperCase() + '</span><br>' +
      '<span class="detail-label">Collective:</span> <span class="detail-value">' + e.comm_type + '</span><br>' +
      '<span class="detail-label">Domain:</span> <span class="detail-value" style="color:' + (COLORS[e.group]||'#fff') + '">' + e.group + '</span><br>' +
      '<span class="detail-label">Data:</span> <span class="detail-value">' + e.comm_size_fmt + '</span><br>' +
      '<span class="detail-label">Ranks:</span> <span class="detail-value">' + e.ranks.length + '</span>' +
      '<div class="detail-ranks">[' + e.ranks.join(', ') + ']</div>';
  }});

  listEl.appendChild(row);
}});

// Default: highlight TP
document.querySelector('.group-card[data-group="TP"]').click();
</script>
</body>
</html>"""


# --- Identify Transformer block boundaries ---

def identify_transformer_blocks(layers: list) -> list:
    """Group consecutive layers into Transformer blocks based on name patterns."""
    blocks = []
    i = 0
    n = len(layers)

    while i < n:
        name = layers[i]["name"]

        # Detect attention_column + attention_row + mlp_* pattern
        if name == "attention_column" and i + 1 < n and layers[i + 1]["name"] == "attention_row":
            block_start = i
            i += 2  # skip attention pair
            # Collect following mlp layers
            while i < n and layers[i]["name"] in ("mlp_moelayer", "mlp_layer"):
                i += 1
            blocks.append({"type": "transformer", "start": block_start, "end": i - 1})
        else:
            blocks.append({"type": "other", "start": i, "end": i})
            i += 1

    return blocks


# --- Parse EndToEnd.csv for real timing data ---

def parse_endtoend_csv(filepath: str) -> list:
    """Parse EndToEnd.csv and return per-layer timing data.

    Returns list of dicts, one per layer (excluding SUM row), with keys:
      layer_name, fwd_compute, wg_compute, ig_compute,
      fwd_exposed_comm, wg_exposed_comm, ig_exposed_comm,
      fwd_total_comm, wg_total_comm, ig_total_comm
    """
    with open(filepath) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Skip dimension rows (lines starting with "File name")
    header_idx = 0
    if lines[0].startswith("File name"):
        header_idx = 2

    layers = []
    for i in range(header_idx + 1, len(lines)):
        parts = [p.strip() for p in lines[i].split(",")]
        if not parts or parts[0] in ("total exposed comm", "SUM"):
            continue
        if len(parts) < 8:
            continue

        def _val(s):
            s = s.strip()
            if not s or s.upper() in ("NONE", "NAN"):
                return 0.0
            try:
                return float(s)
            except ValueError:
                return 0.0

        layers.append({
            "layer_name": parts[0],
            "fwd_compute": _val(parts[2]) if len(parts) > 2 else 0,
            "wg_compute": _val(parts[3]) if len(parts) > 3 else 0,
            "ig_compute": _val(parts[4]) if len(parts) > 4 else 0,
            "fwd_exposed_comm": _val(parts[5]) if len(parts) > 5 else 0,
            "wg_exposed_comm": _val(parts[6]) if len(parts) > 6 else 0,
            "ig_exposed_comm": _val(parts[7]) if len(parts) > 7 else 0,
            "fwd_total_comm": _val(parts[8]) if len(parts) > 8 else 0,
            "wg_total_comm": _val(parts[11]) if len(parts) > 11 else 0,
            "ig_total_comm": _val(parts[14]) if len(parts) > 14 else 0,
        })

    return layers


# --- Build simulated timeline events ---

def build_timeline(workload: dict, max_layers: int = 0, endtoend_layers: list = None) -> list:
    """Simulate the forward → backward(ig) → backward(wg) execution order.

    If endtoend_layers is provided (from parse_endtoend_csv), uses real timing
    from EndToEnd.csv for both compute and comm durations. Otherwise falls back
    to workload-only estimation (comm durations estimated from data size).

    Returns a list of events: {phase, layer_idx, name, type(compute|comm),
                               group, comm_type, comm_size, start, duration}
    """
    layers = workload["layers"]
    if max_layers > 0:
        layers = layers[:max_layers]
    use_real = endtoend_layers is not None and len(endtoend_layers) >= len(layers)

    events = []
    t = 0.0

    # Forward pass: layers[0] → layers[N-1]
    for i, layer in enumerate(layers):
        e2e = endtoend_layers[i] if use_real else None

        # Compute
        fwd_comp = e2e["fwd_compute"] if e2e else layer["fp_compute"]
        if fwd_comp > 0:
            events.append({
                "phase": "fwd", "layer_idx": i, "name": layer["name"],
                "type": "compute", "group": "NONE", "comm_type": "NONE",
                "comm_size": 0, "start": t, "duration": fwd_comp,
            })
            t += fwd_comp

        # Comm (blocking in fwd)
        fwd_comm = e2e["fwd_exposed_comm"] if e2e else 0
        if e2e:
            # Use real exposed comm duration from EndToEnd.csv
            if fwd_comm > 0:
                events.append({
                    "phase": "fwd", "layer_idx": i, "name": layer["name"],
                    "type": "comm", "group": layer["fp_group"],
                    "comm_type": base_comm_type(layer["fp_comm_type"]),
                    "comm_size": layer["fp_comm_size"],
                    "start": t, "duration": fwd_comm,
                })
                t += fwd_comm
        else:
            # Fallback: estimate from data size
            if layer["fp_comm_type"] != "NONE" and layer["fp_comm_size"] > 0:
                comm_dur = max(layer["fp_comm_size"] / 5e6, 100)
                events.append({
                    "phase": "fwd", "layer_idx": i, "name": layer["name"],
                    "type": "comm", "group": layer["fp_group"],
                    "comm_type": base_comm_type(layer["fp_comm_type"]),
                    "comm_size": layer["fp_comm_size"],
                    "start": t, "duration": comm_dur,
                })
                t += comm_dur

    # Backward pass: layers[N-1] → layers[0]
    for i in range(len(layers) - 1, -1, -1):
        layer = layers[i]
        e2e = endtoend_layers[i] if use_real else None

        # Input gradient compute
        ig_comp = e2e["ig_compute"] if e2e else layer["ig_compute"]
        if ig_comp > (0 if e2e else 1):
            events.append({
                "phase": "bwd_ig", "layer_idx": i, "name": layer["name"],
                "type": "compute", "group": "NONE", "comm_type": "NONE",
                "comm_size": 0, "start": t, "duration": ig_comp,
            })
            t += ig_comp

        # Input gradient comm (blocking)
        ig_comm = e2e["ig_exposed_comm"] if e2e else 0
        if e2e:
            if ig_comm > 0:
                events.append({
                    "phase": "bwd_ig", "layer_idx": i, "name": layer["name"],
                    "type": "comm", "group": layer["ig_group"],
                    "comm_type": base_comm_type(layer["ig_comm_type"]),
                    "comm_size": layer["ig_comm_size"],
                    "start": t, "duration": ig_comm,
                })
                t += ig_comm
        else:
            if layer["ig_comm_type"] != "NONE" and layer["ig_comm_size"] > 0:
                comm_dur = max(layer["ig_comm_size"] / 5e6, 100)
                events.append({
                    "phase": "bwd_ig", "layer_idx": i, "name": layer["name"],
                    "type": "comm", "group": layer["ig_group"],
                    "comm_type": base_comm_type(layer["ig_comm_type"]),
                    "comm_size": layer["ig_comm_size"],
                    "start": t, "duration": comm_dur,
                })
                t += comm_dur

        # Weight gradient compute
        wg_comp = e2e["wg_compute"] if e2e else layer["wg_compute"]
        if wg_comp > (0 if e2e else 1):
            events.append({
                "phase": "bwd_wg", "layer_idx": i, "name": layer["name"],
                "type": "compute", "group": "NONE", "comm_type": "NONE",
                "comm_size": 0, "start": t, "duration": wg_comp,
            })
            t += wg_comp

        # Weight gradient comm
        # With EndToEnd data: exposed_comm already accounts for overlap, so it advances t.
        # Without EndToEnd: estimated comm is non-blocking (overlaps with next ig).
        wg_comm = e2e["wg_exposed_comm"] if e2e else 0
        if e2e:
            if wg_comm > 0:
                events.append({
                    "phase": "bwd_wg", "layer_idx": i, "name": layer["name"],
                    "type": "comm", "group": layer["wg_group"],
                    "comm_type": base_comm_type(layer["wg_comm_type"]),
                    "comm_size": layer["wg_comm_size"],
                    "start": t, "duration": wg_comm,
                })
                t += wg_comm
        else:
            if layer["wg_comm_type"] != "NONE" and layer["wg_comm_size"] > 0:
                comm_dur = max(layer["wg_comm_size"] / 5e6, 100)
                events.append({
                    "phase": "bwd_wg", "layer_idx": i, "name": layer["name"],
                    "type": "comm", "group": layer["wg_group"],
                    "comm_type": base_comm_type(layer["wg_comm_type"]),
                    "comm_size": layer["wg_comm_size"],
                    "start": t, "duration": comm_dur,
                    "non_blocking": True,
                })

    return events


# --- HTML renderer ---

GROUP_COLORS = {
    "TP": "#3b82f6",      # blue
    "DP": "#f59e0b",      # amber
    "EP": "#10b981",      # emerald
    "DP_EP": "#8b5cf6",   # violet
    "PP": "#ef4444",      # red
    "NONE": "#6b7280",    # gray
}

COMM_TYPE_PATTERNS = {
    "ALLREDUCE": "solid",
    "ALLGATHER": "solid",
    "REDUCESCATTER": "stripe",
    "ALLTOALL": "dot",
    "ALLREDUCEALLTOALL": "solid",
    "NONE": "solid",
}

PHASE_LABELS = {
    "fwd": "Forward Pass",
    "bwd_ig": "Backward (Input Grad)",
    "bwd_wg": "Backward (Weight Grad)",
}


def render_html(workload: dict, events: list, max_layers: int = 0, rank: int = 0) -> str:
    import json as _json

    layers = workload["layers"]
    if max_layers > 0:
        layers = layers[:max_layers]

    blocks = identify_transformer_blocks(layers)
    total_time = max((e["start"] + e["duration"] for e in events), default=1)

    row_labels = ["Compute", "TP Comm", "DP Comm", "EP Comm", "DP_EP Comm"]
    group_to_row = {"NONE": 0, "TP": 1, "DP": 2, "EP": 3, "DP_EP": 4}

    header_info = workload["header"]

    # Build all groups from C++ logic
    all_domain_groups = build_groups_from_cpp(
        workload["tp_size"], workload["dp_size"],
        workload["ep_size"], workload["dp_ep_size"], workload["all_gpus"])

    # Build global domain map data for JS rendering
    # Serialize ALL groups per domain so JS can render + search any rank
    import json as _json_groups
    domain_map_js = {}
    for domain in ("TP", "DP", "EP", "DP_EP"):
        groups_list = all_domain_groups[domain]
        if not groups_list:
            continue
        domain_map_js[domain] = {
            "groups": groups_list,
            "n": len(groups_list),
            "gs": len(groups_list[0]),
        }
    # Build reverse lookup: rank -> {domain: group_idx}
    rank_to_group_js = {}
    for domain, info in domain_map_js.items():
        for gidx, g in enumerate(info["groups"]):
            for r in g:
                if r not in rank_to_group_js:
                    rank_to_group_js[r] = {}
                rank_to_group_js[r][domain] = gidx
    # For large GPU counts, compress: store only the formula pattern + sample groups
    # But for correctness, we serialize all groups (they compress well in gzip)
    domain_map_json = _json_groups.dumps(domain_map_js, separators=(',', ':'))

    # Compute PP stage → group index mapping
    # In Megatron layout: rank = tp_rank + tp_size * pp_rank + tp_size * pp_size * dp_rank
    # So tp_group_idx = rank // tp_size, and pp_rank = tp_group_idx % pp_size
    # PP stage s contains TP groups where (group_idx % pp_size) == s
    pp_size = workload["pp_size"]
    tp_size = workload["tp_size"]
    num_layers = len(layers)

    # pp_stage_groups: {pp_stage: {domain: [group_indices]}}
    pp_stage_groups = {}
    if pp_size > 1:
        # Build TP group → PP stage mapping
        tp_groups = all_domain_groups.get("TP", [])
        tp_num = len(tp_groups)
        # For each PP stage, collect which groups (across all domains) are active
        for s in range(pp_size):
            stage_tp_indices = [i for i in range(tp_num) if i % pp_size == s]
            # Collect ranks in this PP stage
            stage_ranks = set()
            for gi in stage_tp_indices:
                stage_ranks.update(tp_groups[gi])

            stage_groups = {}
            for domain in ("TP", "DP", "EP", "DP_EP"):
                groups_list = all_domain_groups.get(domain, [])
                if not groups_list:
                    continue
                # Find groups that contain at least one rank from this PP stage
                active = [i for i, g in enumerate(groups_list)
                          if any(r in stage_ranks for r in g)]
                stage_groups[domain] = active
            pp_stage_groups[s] = stage_groups

    # Layer → PP stage mapping
    # Simple: layers_per_stage = num_layers // pp_size (remainder goes to last stage)
    layers_per_stage = num_layers // pp_size if pp_size > 1 and num_layers >= pp_size else max(num_layers, 1)
    def layer_to_pp_stage(layer_idx):
        if pp_size <= 1 or layers_per_stage <= 0:
            return 0
        s = layer_idx // layers_per_stage
        return min(s, pp_size - 1)

    # Serialize pp_stage_groups for JS
    pp_stage_groups_json = _json_groups.dumps(pp_stage_groups, separators=(',', ':'))

    # Classify events
    compute_events = [e for e in events if e["type"] == "compute"]
    comm_events = [e for e in events if e["type"] == "comm"]
    tp_events = [e for e in comm_events if e["group"] == "TP"]
    dp_events = [e for e in comm_events if e["group"] == "DP"]
    ep_events = [e for e in comm_events if e["group"] == "EP"]
    dpep_events = [e for e in comm_events if e["group"] == "DP_EP"]
    total_compute = sum(e["duration"] for e in compute_events)
    total_comm = sum(e["duration"] for e in comm_events)

    # Phase separator position
    fwd_events = [e for e in events if e["phase"] == "fwd"]
    phase_sep_pct = 0
    if fwd_events:
        fwd_end = max(e["start"] + e["duration"] for e in fwd_events)
        phase_sep_pct = fwd_end / total_time

    # Transformer block annotations
    block_annotations = []
    transformer_idx = 0
    for block in blocks:
        if block["type"] == "transformer":
            transformer_idx += 1
            block_annotations.append({
                "label": f"TF-{transformer_idx}",
                "start": block["start"],
                "end": block["end"],
            })

    # Prepare events for JSON — compute row assignment server-side
    events_js = []
    for e in events:
        row = group_to_row.get(e["group"], 0) if e["type"] == "comm" else 0
        ct = e.get("comm_type", "")
        pattern = COMM_TYPE_PATTERNS.get(ct, "solid")
        events_js.append({
            "s": e["start"],
            "d": e["duration"],
            "r": row,
            "p": e["phase"],
            "t": e["type"],
            "g": e.get("group", "NONE"),
            "ct": ct,
            "n": e["name"],
            "li": e["layer_idx"],
            "cs": e.get("comm_size", 0),
            "csf": format_bytes(e.get("comm_size", 0)),
            "nb": e.get("non_blocking", False),
            "pat": pattern,
            "ps": layer_to_pp_stage(e["layer_idx"]),
        })

    # Map layer indices to x positions for block annotations
    layer_start_frac = {}
    layer_end_frac = {}
    for e in events:
        li = e["layer_idx"]
        sf = e["start"] / total_time
        ef = (e["start"] + e["duration"]) / total_time
        if li not in layer_start_frac or sf < layer_start_frac[li]:
            layer_start_frac[li] = sf
        if li not in layer_end_frac or ef > layer_end_frac[li]:
            layer_end_frac[li] = ef

    annos_js = []
    for anno in block_annotations:
        s = layer_start_frac.get(anno["start"], 0)
        e_end = layer_end_frac.get(anno["end"], s + 0.001)
        annos_js.append({"l": anno["label"], "s": s, "e": e_end})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SimAI Workload Timeline</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;600;700&display=swap');
  :root {{
    --bg: #06080d; --surface: #0d1117; --surface2: #161b22;
    --border: #21262d; --text: #e6edf3; --text2: #8b949e; --text3: #484f58;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'DM Sans',system-ui,sans-serif; }}
  .header {{ padding:24px 32px 16px; border-bottom:1px solid var(--border); }}
  .header h1 {{ font-size:20px; font-weight:700; margin-bottom:10px; letter-spacing:-0.02em; }}
  .chips {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .chip {{ font-family:'JetBrains Mono',monospace; font-size:11px; padding:3px 10px;
           border-radius:16px; background:var(--surface2); border:1px solid var(--border); color:var(--text2); }}
  .chip b {{ color:var(--text); font-weight:600; }}
  .summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
              gap:10px; padding:16px 32px; border-bottom:1px solid var(--border); }}
  .scard {{ background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:10px 14px; }}
  .scard .sl {{ font-size:10px; color:var(--text3); margin-bottom:2px; }}
  .scard .sv {{ font-family:'JetBrains Mono',monospace; font-size:15px; font-weight:600; }}
  .controls {{ display:flex; align-items:center; gap:12px; padding:12px 32px;
               border-bottom:1px solid var(--border); }}
  .controls button {{
    font-family:'JetBrains Mono',monospace; font-size:12px; padding:4px 12px;
    background:var(--surface2); border:1px solid var(--border); border-radius:4px;
    color:var(--text); cursor:pointer; transition:background .15s;
  }}
  .controls button:hover {{ background:var(--border); }}
  .zoom-info {{ font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--text2); min-width:80px; text-align:center; }}
  .legend {{ display:flex; gap:14px; margin-left:auto; align-items:center; }}
  .legend-item {{ display:flex; align-items:center; gap:5px; font-size:11px; color:var(--text2); }}
  .legend-dot {{ width:10px; height:10px; border-radius:2px; }}
  .ldiv {{ width:1px; height:14px; background:var(--border); }}

  .timeline-wrap {{
    position:relative; overflow:hidden; cursor:grab;
    border-bottom:1px solid var(--border);
  }}
  .timeline-wrap:active {{ cursor:grabbing; }}
  canvas {{ display:block; }}
  .minimap-wrap {{
    position:relative; height:40px; background:var(--surface);
    border-bottom:1px solid var(--border); cursor:pointer;
  }}
  .minimap-wrap canvas {{ display:block; }}
  .minimap-viewport {{
    position:absolute; top:0; bottom:0;
    background:rgba(88,166,255,0.08); border:1px solid rgba(88,166,255,0.3);
    pointer-events:none; transition:left 50ms,width 50ms;
  }}
  .tooltip {{
    display:none; position:fixed; background:#1e293b; border:1px solid #475569;
    border-radius:6px; padding:10px 14px; font-family:'JetBrains Mono',monospace;
    font-size:11px; color:var(--text); z-index:9999; pointer-events:none;
    line-height:1.6; box-shadow:0 8px 24px rgba(0,0,0,.5); max-width:350px;
  }}
</style>
</head>
<body>
<div class="header">
  <h1>SimAI Workload Timeline — Rank {rank}</h1>
  <div class="chips">
    <div class="chip">Policy: <b>{header_info.get('policy','N/A')}</b></div>
    <div class="chip">GPUs: <b>{workload['all_gpus']}</b></div>
    <div class="chip">Rank: <b>{rank}</b></div>
    <div class="chip">TP=<b>{workload['tp_size']}</b></div>
    <div class="chip">DP=<b>{workload['dp_size']}</b></div>
    <div class="chip">EP=<b>{workload['ep_size']}</b></div>
    <div class="chip">PP=<b>{workload['pp_size']}</b></div>
    <div class="chip">Layers: <b>{len(layers)}</b></div>
    <div class="chip">Events: <b>{len(events)}</b></div>
  </div>
</div>
<div class="summary">
  <div class="scard"><div class="sl">Total Compute</div><div class="sv" style="color:#6b7280">{total_compute:.0f}</div></div>
  <div class="scard"><div class="sl">Total Comm (est.)</div><div class="sv">{total_comm:.0f}</div></div>
  <div class="scard"><div class="sl">TP Ops</div><div class="sv" style="color:#3b82f6">{len(tp_events)}</div></div>
  <div class="scard"><div class="sl">DP Ops</div><div class="sv" style="color:#f59e0b">{len(dp_events)}</div></div>
  <div class="scard"><div class="sl">EP Ops</div><div class="sv" style="color:#10b981">{len(ep_events)}</div></div>
  <div class="scard"><div class="sl">DP_EP Ops</div><div class="sv" style="color:#8b5cf6">{len(dpep_events)}</div></div>
</div>
<div class="controls">
  <button id="zoomIn">Zoom In (+)</button>
  <button id="zoomOut">Zoom Out (-)</button>
  <button id="zoomFit">Fit All</button>
  <div class="zoom-info" id="zoomInfo">1.0x</div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#6b7280"></div>Compute</div>
    <div class="ldiv"></div>
    <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>TP</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>DP</div>
    <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>EP</div>
    <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>DP_EP</div>
    <div class="ldiv"></div>
    <div class="legend-item" style="font-size:10px">Scroll to zoom, drag to pan</div>
  </div>
</div>
<div class="timeline-wrap" id="timelineWrap">
  <canvas id="timeline"></canvas>
</div>
<div class="minimap-wrap" id="minimapWrap">
  <canvas id="minimap"></canvas>
  <div class="minimap-viewport" id="minimapVP"></div>
</div>
<div style="padding:12px 32px;border-bottom:1px solid var(--border)" id="domainMapSection">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <div style="font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em">
      全局通信域分组 — {workload["all_gpus"]} GPUs 的完整分组视图
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-left:auto">
      <label style="font-size:11px;color:#484f58" for="rankSearch">查找 Rank:</label>
      <input id="rankSearch" type="number" min="0" max="{workload['all_gpus']-1}" placeholder="输入 rank 编号"
        style="width:120px;padding:3px 8px;font-family:JetBrains Mono,monospace;font-size:11px;
        background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);outline:none">
      <button id="rankSearchBtn" style="font-family:JetBrains Mono,monospace;font-size:11px;padding:3px 10px;
        background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);cursor:pointer">
        搜索</button>
    </div>
  </div>
  <div style="font-size:11px;color:#484f58;margin-bottom:10px">
    每个通信域将 {workload["all_gpus"]} 张 GPU 划分为多个并行组。执行 collective 时，同一域内所有组同时通信。
  </div>
  <div id="rankSearchResult" style="display:none;margin-bottom:10px;padding:8px 12px;
    background:rgba(88,166,255,0.06);border:1px solid rgba(88,166,255,0.2);border-radius:4px;
    font-family:JetBrains Mono,monospace;font-size:11px"></div>
  <div id="domainMapContainer"></div>
</div>
<div style="padding:12px 32px;border-bottom:1px solid var(--border)" id="rankViewSection">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:8px">
    <div style="font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.08em">
      Rank 通信视图 — 每张卡的通信行为
    </div>
    <div style="display:flex;align-items:center;gap:6px;margin-left:auto">
      <label style="font-size:11px;color:#484f58" for="rankViewInput">Rank 范围:</label>
      <input id="rankViewInput" type="text" placeholder="例: 0-7 或 0,2,4,6"
        style="width:160px;padding:3px 8px;font-family:JetBrains Mono,monospace;font-size:11px;
        background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);outline:none">
      <button id="rankViewBtn" style="font-family:JetBrains Mono,monospace;font-size:11px;padding:3px 10px;
        background:var(--surface2);border:1px solid var(--border);border-radius:4px;color:var(--text);cursor:pointer">
        展示</button>
    </div>
  </div>
  <div style="font-size:11px;color:#484f58;margin-bottom:10px">
    输入 rank 范围查看每张卡在每个 layer 中的通信行为。横轴与上方 timeline 同步缩放/平移。颜色表示通信域：
    <span style="color:#3b82f6">TP</span>
    <span style="color:#f59e0b">DP</span>
    <span style="color:#10b981">EP</span>
    <span style="color:#8b5cf6">DP_EP</span>
  </div>
  <div id="rankViewError" style="display:none;margin-bottom:8px;padding:6px 12px;
    background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:4px;
    font-family:JetBrains Mono,monospace;font-size:11px;color:#ef4444"></div>
  <div id="rankViewCanvasWrap" style="position:relative;overflow:hidden;display:none">
    <canvas id="rankViewCanvas"></canvas>
  </div>
  <div class="tooltip" id="rankViewTooltip" style="display:none;position:fixed;background:#1e293b;border:1px solid #475569;
    border-radius:6px;padding:10px 14px;font-family:JetBrains Mono,monospace;
    font-size:11px;color:var(--text);z-index:9999;pointer-events:none;
    line-height:1.6;box-shadow:0 8px 24px rgba(0,0,0,.5);max-width:400px"></div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const EVENTS = {_json.dumps(events_js)};
const ANNOS = {_json.dumps(annos_js)};
const TOTAL_TIME = {total_time};
const PHASE_SEP = {phase_sep_pct};
const ROW_LABELS = {_json.dumps(row_labels)};
const RANK = {rank};
const DOMAIN_MAP = {domain_map_json};
const ALL_GPUS = {workload["all_gpus"]};
const PP_SIZE = {pp_size};
const LAYERS_PER_STAGE = {layers_per_stage};
const PP_STAGE_GROUPS = {pp_stage_groups_json};
let searchedRank = -1;  // Global: set by domain map search, used by tooltip
const COLORS = {{ NONE:'#6b7280', TP:'#3b82f6', DP:'#f59e0b', EP:'#10b981', DP_EP:'#8b5cf6', PP:'#ef4444' }};
const PHASE_LABELS = {{ fwd:'Forward Pass', bwd_ig:'Backward (Input Grad)', bwd_wg:'Backward (Weight Grad)' }};

// Utility: find which group a rank belongs to in a domain
function findGroupForRank(domain, r) {{
  const info = DOMAIN_MAP[domain];
  if (!info) return null;
  for (let i = 0; i < info.groups.length; i++) {{
    if (info.groups[i].includes(r)) return {{ idx: i, ranks: info.groups[i], n: info.n, gs: info.gs }};
  }}
  return null;
}}

// Utility: get sample groups spanning the full range for a domain
function sampleGroupsForDomain(domain, count) {{
  const info = DOMAIN_MAP[domain];
  if (!info) return [];
  const n = info.n;
  if (n <= count) return info.groups.map((g, i) => ({{ idx: i, ranks: g }}));
  const step = (n - 1) / (count - 1);
  const samples = [];
  for (let i = 0; i < count; i++) {{
    const idx = Math.round(step * i);
    samples.push({{ idx, ranks: info.groups[idx] }});
  }}
  return samples;
}}

// === Domain Map: Global view of all groups ===
(function renderDomainMap() {{
  const container = document.getElementById('domainMapContainer');
  const DCOLORS = {{ TP:'#3b82f6', DP:'#f59e0b', EP:'#10b981', DP_EP:'#8b5cf6' }};
  const DOMAIN_ORDER = ['TP','DP','EP','DP_EP'];

  // Find which group a rank belongs to in a domain
  function findRankGroup(domain, searchRank) {{
    const info = DOMAIN_MAP[domain];
    if (!info) return -1;
    for (let i = 0; i < info.groups.length; i++) {{
      if (info.groups[i].includes(searchRank)) return i;
    }}
    return -1;
  }}

  // Build sample indices spanning the full range
  function sampleIndices(n, maxShow) {{
    if (n <= maxShow) return Array.from({{length: n}}, (_, i) => i);
    // Show first 5, then evenly spaced samples, then last 3
    const first = 5, last = 3;
    const mid = maxShow - first - last;
    const indices = [];
    for (let i = 0; i < first; i++) indices.push(i);
    const step = (n - first - last) / (mid + 1);
    for (let i = 1; i <= mid; i++) {{
      indices.push(Math.round(first + step * i));
    }}
    for (let i = n - last; i < n; i++) indices.push(i);
    return [...new Set(indices)].sort((a, b) => a - b);
  }}

  function renderDomain(domain, highlightRank) {{
    const info = DOMAIN_MAP[domain];
    if (!info) return '';
    const color = DCOLORS[domain];
    const n = info.n, gs = info.gs;
    const covered = n * gs;
    const maxShowGroups = 20;
    const maxRankCells = 32;

    const indices = sampleIndices(n, maxShowGroups);
    // If highlightRank >= 0, ensure its group is included
    let hlGroupIdx = -1;
    if (highlightRank >= 0) {{
      hlGroupIdx = findRankGroup(domain, highlightRank);
      if (hlGroupIdx >= 0 && !indices.includes(hlGroupIdx)) {{
        indices.push(hlGroupIdx);
        indices.sort((a, b) => a - b);
      }}
    }}

    let rows = '';
    let prevIdx = -1;
    for (const gidx of indices) {{
      // Insert gap indicator if indices are not consecutive
      if (prevIdx >= 0 && gidx > prevIdx + 1) {{
        const skipped = gidx - prevIdx - 1;
        rows += '<tr><td colspan="100" style="text-align:center;color:#484f58;font-size:9px;padding:1px 0">'
          + '⋮ ' + skipped + ' groups ⋮</td></tr>';
      }}
      prevIdx = gidx;

      const g = info.groups[gidx];
      const isHl = (gidx === hlGroupIdx);
      const rowBg = isHl ? 'rgba(88,166,255,0.08)' : 'transparent';
      const labelStyle = isHl ? 'color:' + color + ';font-weight:700' : 'color:#484f58';
      const marker = isHl ? ' ◀ Rank ' + highlightRank : '';

      let cells = '';
      const showRanks = g.length > maxRankCells ? g.slice(0, maxRankCells) : g;
      for (const r of showRanks) {{
        const isMe = (highlightRank >= 0 && r === highlightRank);
        const bg = isMe ? color : '#161b22';
        const fg = isMe ? '#fff' : '#8b949e';
        const border = isMe ? '2px solid ' + color : '1px solid #21262d';
        const fw = isMe ? '700' : '400';
        cells += '<td style="background:' + bg + ';color:' + fg + ';border:' + border
          + ';padding:1px 4px;font-size:9px;text-align:center;font-weight:' + fw
          + ';border-radius:2px;min-width:28px">' + r + '</td>';
      }}
      if (g.length > maxRankCells) {{
        cells += '<td style="color:#484f58;font-size:9px;padding:1px 4px">...+'
          + (g.length - maxRankCells) + '</td>';
      }}

      rows += '<tr style="background:' + rowBg + '">'
        + '<td style="' + labelStyle + ';font-size:10px;padding:2px 6px;white-space:nowrap">'
        + 'G' + gidx + marker + '</td>' + cells + '</tr>';
    }}

    const isOpen = n <= 30 ? ' open' : '';
    return '<details style="margin-bottom:6px"' + isOpen + '>'
      + '<summary style="cursor:pointer;font-family:JetBrains Mono,monospace;font-size:12px;color:'
      + color + ';padding:4px 0;user-select:none">'
      + domain + ': ' + n + ' 个并行组 × ' + gs + ' GPUs/组'
      + ' (覆盖 ' + covered + '/' + ALL_GPUS + ' GPUs)'
      + (hlGroupIdx >= 0 ? ' — Rank ' + highlightRank + ' 在 Group #' + hlGroupIdx : '')
      + '</summary>'
      + '<div style="overflow-x:auto;margin:4px 0 8px 8px">'
      + '<table style="border-collapse:separate;border-spacing:2px;font-family:JetBrains Mono,monospace">'
      + rows + '</table>'
      + '<div style="font-size:10px;color:#484f58;margin-top:4px">'
      + '显示 ' + indices.length + ' / ' + n + ' 个组（采样覆盖全范围 G0 ~ G' + (n-1) + '）。'
      + '每次 ' + domain + ' collective 时，所有 ' + n + ' 个组同时执行。'
      + '</div></div></details>';
  }}

  function renderAll(highlightRank) {{
    let html = '';
    for (const d of DOMAIN_ORDER) {{
      html += renderDomain(d, highlightRank);
    }}
    container.innerHTML = html;
  }}

  // Initial render with no highlight
  renderAll(-1);

  // Search functionality
  const searchInput = document.getElementById('rankSearch');
  const searchBtn = document.getElementById('rankSearchBtn');
  const searchResult = document.getElementById('rankSearchResult');

  function doSearch() {{
    const val = parseInt(searchInput.value);
    if (isNaN(val) || val < 0 || val >= ALL_GPUS) {{
      searchResult.style.display = 'block';
      searchResult.innerHTML = '<span style="color:#ef4444">请输入有效的 rank 编号 (0 ~ ' + (ALL_GPUS - 1) + ')</span>';
      searchedRank = -1;
      renderAll(-1);
      return;
    }}
    searchedRank = val;  // Update global for timeline tooltip
    // Show search result summary
    let summary = '<b style="color:#58a6ff">Rank ' + val + ' 的通信组 (时间轴 tooltip 已同步更新):</b><br>';
    for (const d of DOMAIN_ORDER) {{
      const gidx = findRankGroup(d, val);
      if (gidx < 0) {{ summary += d + ': (未启用)<br>'; continue; }}
      const info = DOMAIN_MAP[d];
      const g = info.groups[gidx];
      const rankStr = g.length <= 32 ? '[' + g.join(', ') + ']'
        : '[' + g.slice(0, 8).join(', ') + ', ..., ' + g.slice(-4).join(', ') + ']';
      summary += '<span style="color:' + DCOLORS[d] + '">' + d + '</span>'
        + ' Group #' + gidx + ' (' + g.length + ' GPUs): '
        + '<span style="color:#8b949e">' + rankStr + '</span><br>';
    }}
    searchResult.style.display = 'block';
    searchResult.innerHTML = summary;
    renderAll(val);
  }}

  searchBtn.addEventListener('click', doSearch);
  searchInput.addEventListener('keydown', (e) => {{ if (e.key === 'Enter') doSearch(); }});
}})();
const ROW_H = 36, ROW_GAP = 4, LABEL_W = 100, ANNO_H = 28;
const NUM_ROWS = ROW_LABELS.length;
const CANVAS_H = NUM_ROWS * (ROW_H + ROW_GAP) + ANNO_H + 20;

// State
let zoom = 1, panX = 0;
let dragging = false, dragStartX = 0, dragStartPan = 0;

const wrap = document.getElementById('timelineWrap');
const canvas = document.getElementById('timeline');
const ctx = canvas.getContext('2d');
const miniCanvas = document.getElementById('minimap');
const miniCtx = miniCanvas.getContext('2d');
const minimapVP = document.getElementById('minimapVP');
const tip = document.getElementById('tooltip');
const zoomInfo = document.getElementById('zoomInfo');

let W = 0; // canvas width

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  W = wrap.clientWidth;
  canvas.width = W * dpr; canvas.height = CANVAS_H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = CANVAS_H + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const mw = wrap.clientWidth;
  miniCanvas.width = mw * dpr; miniCanvas.height = 40 * dpr;
  miniCanvas.style.width = mw + 'px'; miniCanvas.style.height = '40px';
  miniCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function clampPan() {{
  const contentW = W * zoom;
  const maxPan = contentW - W;
  if (panX < 0) panX = 0;
  if (panX > maxPan) panX = maxPan;
  if (zoom <= 1) panX = 0;
}}

function drawTimeline() {{
  ctx.clearRect(0, 0, W, CANVAS_H);
  const drawW = W - LABEL_W;
  const contentW = drawW * zoom;

  // Row labels (fixed)
  ctx.font = '600 11px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ROW_LABELS.forEach((label, i) => {{
    const y = i * (ROW_H + ROW_GAP);
    ctx.fillStyle = '#161b22';
    ctx.globalAlpha = 0.5;
    ctx.fillRect(LABEL_W, y, drawW, ROW_H);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#8b949e';
    ctx.fillText(label, LABEL_W - 8, y + ROW_H / 2);
  }});

  // Clip for scrollable area
  ctx.save();
  ctx.beginPath();
  ctx.rect(LABEL_W, 0, drawW, CANVAS_H);
  ctx.clip();

  // Phase separator
  if (PHASE_SEP > 0) {{
    const sx = LABEL_W + PHASE_SEP * contentW - panX;
    if (sx >= LABEL_W && sx <= W) {{
      ctx.strokeStyle = 'rgba(248,81,73,0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, NUM_ROWS * (ROW_H + ROW_GAP));
      ctx.stroke();
      ctx.font = '600 10px JetBrains Mono, monospace';
      ctx.fillStyle = '#f85149';
      ctx.textAlign = 'center';
      ctx.fillText('\\u2190 FWD | BWD \\u2192', sx, NUM_ROWS * (ROW_H + ROW_GAP) + 12);
    }}
  }}

  // Events
  EVENTS.forEach(e => {{
    const x = LABEL_W + (e.s / TOTAL_TIME) * contentW - panX;
    const w = Math.max((e.d / TOTAL_TIME) * contentW, 1);
    if (x + w < LABEL_W || x > W) return; // off-screen
    const y = e.r * (ROW_H + ROW_GAP) + 2;
    const h = ROW_H - 4;

    let color;
    if (e.t === 'compute') {{
      color = e.p === 'fwd' ? '#4b5563' : '#374151';
    }} else {{
      color = COLORS[e.g] || '#6b7280';
    }}

    ctx.globalAlpha = e.nb ? 0.6 : 0.9;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);

    // Stripe pattern for ReduceScatter
    if (e.pat === 'stripe') {{
      ctx.globalAlpha = 0.15;
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      for (let sx = x - h; sx < x + w; sx += 6) {{
        ctx.beginPath();
        ctx.moveTo(sx, y + h);
        ctx.lineTo(sx + h, y);
        ctx.stroke();
      }}
      ctx.restore();
    }}
    // Dot pattern for AllToAll
    if (e.pat === 'dot') {{
      ctx.globalAlpha = 0.2;
      ctx.fillStyle = '#fff';
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      for (let dx = x; dx < x + w; dx += 6) {{
        for (let dy = y + 3; dy < y + h; dy += 6) {{
          ctx.beginPath();
          ctx.arc(dx + 3, dy, 1, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
      ctx.restore();
    }}

    ctx.globalAlpha = 1;

    // Label if wide enough
    if (w > 30) {{
      ctx.font = '400 9px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const label = e.t === 'compute' ? e.n : (e.ct + ' ' + e.csf);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x + 2, y, w - 4, h);
      ctx.clip();
      ctx.fillText(label, x + 4, y + h / 2);
      ctx.restore();
    }}
  }});

  // Transformer block annotations
  const annoY = NUM_ROWS * (ROW_H + ROW_GAP) + 2;
  ANNOS.forEach(a => {{
    const ax = LABEL_W + a.s * contentW - panX;
    const aw = (a.e - a.s) * contentW;
    if (ax + aw < LABEL_W || ax > W) return;
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(ax, annoY, aw, ANNO_H - 4);
    if (aw > 20) {{
      ctx.font = '400 8px JetBrains Mono, monospace';
      ctx.fillStyle = '#484f58';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(a.l, ax + aw / 2, annoY + (ANNO_H - 4) / 2);
    }}
  }});

  ctx.restore();
}}

function drawMinimap() {{
  const mw = miniCanvas.width / (window.devicePixelRatio || 1);
  const mh = 40;
  miniCtx.clearRect(0, 0, mw, mh);
  miniCtx.fillStyle = '#0d1117';
  miniCtx.fillRect(0, 0, mw, mh);

  // Draw events compactly
  const rowH = mh / NUM_ROWS;
  EVENTS.forEach(e => {{
    const x = (e.s / TOTAL_TIME) * mw;
    const w = Math.max((e.d / TOTAL_TIME) * mw, 0.5);
    const y = e.r * rowH;
    miniCtx.fillStyle = e.t === 'compute' ? '#374151' : (COLORS[e.g] || '#6b7280');
    miniCtx.globalAlpha = 0.7;
    miniCtx.fillRect(x, y, w, rowH - 1);
  }});
  miniCtx.globalAlpha = 1;

  // Phase separator
  if (PHASE_SEP > 0) {{
    miniCtx.strokeStyle = 'rgba(248,81,73,0.5)';
    miniCtx.lineWidth = 1;
    miniCtx.beginPath();
    miniCtx.moveTo(PHASE_SEP * mw, 0);
    miniCtx.lineTo(PHASE_SEP * mw, mh);
    miniCtx.stroke();
  }}

  // Viewport indicator
  const drawW = W - LABEL_W;
  const vpLeft = (panX / (drawW * zoom)) * mw;
  const vpW = (1 / zoom) * mw;
  minimapVP.style.left = vpLeft + 'px';
  minimapVP.style.width = Math.min(vpW, mw) + 'px';
}}

function render() {{
  clampPan();
  drawTimeline();
  drawMinimap();
  zoomInfo.textContent = zoom.toFixed(1) + 'x';
}}

// Zoom
function zoomAt(factor, centerX) {{
  const drawW = W - LABEL_W;
  const relX = (centerX - LABEL_W + panX) / (drawW * zoom);
  zoom = Math.max(1, Math.min(zoom * factor, 500));
  panX = relX * drawW * zoom - (centerX - LABEL_W);
  render();
}}

// Mouse wheel zoom
wrap.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.25 : 0.8;
  const rect = canvas.getBoundingClientRect();
  zoomAt(factor, e.clientX - rect.left);
}}, {{ passive: false }});

// Drag pan
wrap.addEventListener('mousedown', (e) => {{
  dragging = true;
  dragStartX = e.clientX;
  dragStartPan = panX;
}});
window.addEventListener('mousemove', (e) => {{
  if (dragging) {{
    panX = dragStartPan - (e.clientX - dragStartX);
    render();
  }}
}});
window.addEventListener('mouseup', () => {{ dragging = false; }});

// Minimap click to jump
document.getElementById('minimapWrap').addEventListener('click', (e) => {{
  const rect = miniCanvas.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  const drawW = W - LABEL_W;
  panX = frac * drawW * zoom - drawW / 2;
  render();
}});

// Buttons
document.getElementById('zoomIn').addEventListener('click', () => zoomAt(1.5, W / 2));
document.getElementById('zoomOut').addEventListener('click', () => zoomAt(0.67, W / 2));
document.getElementById('zoomFit').addEventListener('click', () => {{ zoom = 1; panX = 0; render(); }});

// Tooltip on hover
canvas.addEventListener('mousemove', (e) => {{
  if (dragging) {{ tip.style.display = 'none'; return; }}
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const drawW = W - LABEL_W;
  const contentW = drawW * zoom;

  let found = null;
  for (let i = EVENTS.length - 1; i >= 0; i--) {{
    const ev = EVENTS[i];
    const x = LABEL_W + (ev.s / TOTAL_TIME) * contentW - panX;
    const w = Math.max((ev.d / TOTAL_TIME) * contentW, 1);
    const y = ev.r * (ROW_H + ROW_GAP) + 2;
    const h = ROW_H - 4;
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) {{
      found = ev;
      break;
    }}
  }}

  if (found) {{
    let html = '<b>Layer:</b> ' + found.n + ' (#' + found.li + ')';
    if (PP_SIZE > 1) html += ' <span style="color:#ef4444">[PP Stage ' + found.ps + ']</span>';
    html += '<br>';
    html += '<b>Phase:</b> ' + (PHASE_LABELS[found.p] || found.p) + '<br>';
    if (found.t === 'comm') {{
      html += '<b>Collective:</b> ' + found.ct + '<br>';
      if (found.cs > 0) html += '<b>Size:</b> ' + found.csf + '<br>';

      // Show the active groups for this layer's PP stage + domain
      const domain = found.g;
      const color = COLORS[domain] || '#fff';
      const dInfo = DOMAIN_MAP[domain];
      const stageGroups = PP_STAGE_GROUPS[found.ps];
      const activeIndices = (stageGroups && stageGroups[domain]) ? stageGroups[domain] : [];
      const total = dInfo ? dInfo.n : 0;
      const count = activeIndices.length || total;

      if (dInfo && count > 0) {{
        html += '<div style="margin-top:4px;padding-top:4px;border-top:1px solid #334155">';
        html += '<span style="color:' + color + '"><b>' + domain + '</b></span>';
        if (PP_SIZE > 1) {{
          html += ' PP Stage ' + found.ps + ' 激活 <b>' + count + '</b>/' + total + ' 组';
        }} else {{
          html += ' 共 <b>' + total + '</b> 个并行组';
        }}
        html += '，每组 ' + dInfo.gs + ' GPUs<br>';

        // Sample active groups: first, 25%, 50%, 75%, last
        const indices = activeIndices.length > 0 ? activeIndices : Array.from({{length: total}}, (_, i) => i);
        const picks = indices.length <= 6 ? indices
          : [indices[0], indices[Math.floor(indices.length*0.25)],
             indices[Math.floor(indices.length*0.5)],
             indices[Math.floor(indices.length*0.75)],
             indices[indices.length-1]];
        const uniquePicks = [...new Set(picks)];
        html += '<div style="font-size:10px;color:#8b949e;margin:2px 0;line-height:1.5">';
        for (const gi of uniquePicks) {{
          const g = dInfo.groups[gi];
          const rStr = g.length <= 10 ? '[' + g.join(', ') + ']'
            : '[' + g.slice(0,4).join(', ') + ', ..., ' + g.slice(-3).join(', ') + ']';
          html += '<span style="color:' + color + '">G' + gi + '</span> ' + rStr + '<br>';
        }}
        if (indices.length > 6) {{
          html += '<span style="color:#484f58">... 共 ' + count + ' 个组同时执行</span>';
        }}
        html += '</div></div>';
      }}

      // If user searched a rank, also show that rank's specific group
      if (searchedRank >= 0 && dInfo) {{
        const gInfo = findGroupForRank(domain, searchedRank);
        if (gInfo) {{
          const isActive = activeIndices.length === 0 || activeIndices.includes(gInfo.idx);
          const r = gInfo.ranks;
          const rStr = r.length <= 16 ? '[' + r.join(', ') + ']'
            : '[' + r.slice(0,6).join(', ') + ', ..., ' + r.slice(-4).join(', ') + ']';
          const sColor = isActive ? '#58a6ff' : '#ef4444';
          html += '<div style="margin-top:3px;font-size:11px">';
          html += '<span style="color:' + sColor + '">Rank ' + searchedRank + ' → Group #' + gInfo.idx + '</span>';
          if (!isActive) html += ' <span style="color:#ef4444;font-size:10px">(不在此 PP Stage)</span>';
          html += '<br><span style="color:#8b949e">' + rStr + '</span>';
          html += '</div>';
        }}
      }}
    }} else {{
      html += '<b>Type:</b> COMPUTE<br>';
    }}
    tip.innerHTML = html;
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY + 14) + 'px';
  }} else {{
    tip.style.display = 'none';
  }}
}});
canvas.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {{
  if (e.key === '+' || e.key === '=') zoomAt(1.5, W / 2);
  else if (e.key === '-') zoomAt(0.67, W / 2);
  else if (e.key === '0') {{ zoom = 1; panX = 0; render(); }}
  else if (e.key === 'ArrowLeft') {{ panX -= W * 0.2; render(); }}
  else if (e.key === 'ArrowRight') {{ panX += W * 0.2; render(); }}
}});

// === Rank Communication View ===
(function() {{
  const RV_ROW_H = 30, RV_ROW_GAP = 2, RV_LABEL_W = LABEL_W;
  const rvInput = document.getElementById('rankViewInput');
  const rvBtn = document.getElementById('rankViewBtn');
  const rvError = document.getElementById('rankViewError');
  const rvWrap = document.getElementById('rankViewCanvasWrap');
  const rvCanvas = document.getElementById('rankViewCanvas');
  const rvCtx = rvCanvas.getContext('2d');
  const rvTip = document.getElementById('rankViewTooltip');
  let rvRanks = [];  // selected ranks to display

  function parseRankRange(input) {{
    const result = [];
    const parts = input.split(',').map(s => s.trim()).filter(Boolean);
    for (const part of parts) {{
      if (part.includes('-')) {{
        const [a, b] = part.split('-').map(Number);
        if (isNaN(a) || isNaN(b) || a < 0 || b < 0 || a >= ALL_GPUS || b >= ALL_GPUS) return null;
        const lo = Math.min(a, b), hi = Math.max(a, b);
        for (let i = lo; i <= hi; i++) result.push(i);
      }} else {{
        const n = Number(part);
        if (isNaN(n) || n < 0 || n >= ALL_GPUS) return null;
        result.push(n);
      }}
    }}
    return [...new Set(result)].sort((a, b) => a - b);
  }}

  function rvResize() {{
    if (rvRanks.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const w = wrap.clientWidth;
    const h = rvRanks.length * (RV_ROW_H + RV_ROW_GAP) + 8;
    rvCanvas.width = w * dpr;
    rvCanvas.height = h * dpr;
    rvCanvas.style.width = w + 'px';
    rvCanvas.style.height = h + 'px';
    rvCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}

  function rvDraw() {{
    if (rvRanks.length === 0) return;
    const w = wrap.clientWidth;
    const h = rvRanks.length * (RV_ROW_H + RV_ROW_GAP) + 8;
    const drawW = w - RV_LABEL_W;
    const contentW = drawW * zoom;
    rvCtx.clearRect(0, 0, w, h);

    // Row backgrounds and labels
    rvCtx.font = '600 10px JetBrains Mono, monospace';
    rvCtx.textAlign = 'right';
    rvCtx.textBaseline = 'middle';
    rvRanks.forEach((rank, ri) => {{
      const y = ri * (RV_ROW_H + RV_ROW_GAP);
      rvCtx.fillStyle = '#161b22';
      rvCtx.globalAlpha = 0.5;
      rvCtx.fillRect(RV_LABEL_W, y, drawW, RV_ROW_H);
      rvCtx.globalAlpha = 1;
      rvCtx.fillStyle = '#8b949e';
      rvCtx.fillText('Rank ' + rank, RV_LABEL_W - 8, y + RV_ROW_H / 2);
    }});

    // Clip for scrollable area
    rvCtx.save();
    rvCtx.beginPath();
    rvCtx.rect(RV_LABEL_W, 0, drawW, h);
    rvCtx.clip();

    // Phase separator
    if (PHASE_SEP > 0) {{
      const sx = RV_LABEL_W + PHASE_SEP * contentW - panX;
      if (sx >= RV_LABEL_W && sx <= w) {{
        rvCtx.strokeStyle = 'rgba(248,81,73,0.3)';
        rvCtx.lineWidth = 1;
        rvCtx.beginPath();
        rvCtx.moveTo(sx, 0);
        rvCtx.lineTo(sx, h);
        rvCtx.stroke();
      }}
    }}

    // Draw comm events for each rank
    EVENTS.forEach(ev => {{
      if (ev.t !== 'comm') return;
      const domain = ev.g;
      if (domain === 'NONE') return;
      const color = COLORS[domain] || '#6b7280';
      const x = RV_LABEL_W + (ev.s / TOTAL_TIME) * contentW - panX;
      const bw = Math.max((ev.d / TOTAL_TIME) * contentW, 2);
      if (x + bw < RV_LABEL_W || x > w) return;

      rvRanks.forEach((rank, ri) => {{
        const y = ri * (RV_ROW_H + RV_ROW_GAP) + 1;
        const bh = RV_ROW_H - 2;
        rvCtx.globalAlpha = ev.nb ? 0.55 : 0.85;
        rvCtx.fillStyle = color;
        rvCtx.fillRect(x, y, bw, bh);

        // Stripe for ReduceScatter
        if (ev.pat === 'stripe') {{
          rvCtx.globalAlpha = 0.15;
          rvCtx.strokeStyle = '#000';
          rvCtx.lineWidth = 1;
          rvCtx.save();
          rvCtx.beginPath();
          rvCtx.rect(x, y, bw, bh);
          rvCtx.clip();
          for (let sx = x - bh; sx < x + bw; sx += 6) {{
            rvCtx.beginPath();
            rvCtx.moveTo(sx, y + bh);
            rvCtx.lineTo(sx + bh, y);
            rvCtx.stroke();
          }}
          rvCtx.restore();
        }}

        rvCtx.globalAlpha = 1;

        // Label if wide enough
        if (bw > 40) {{
          rvCtx.font = '400 8px JetBrains Mono, monospace';
          rvCtx.fillStyle = 'rgba(255,255,255,0.85)';
          rvCtx.textAlign = 'left';
          rvCtx.textBaseline = 'middle';
          const label = ev.ct + ' ' + ev.csf;
          rvCtx.save();
          rvCtx.beginPath();
          rvCtx.rect(x + 2, y, bw - 4, bh);
          rvCtx.clip();
          rvCtx.fillText(label, x + 3, y + bh / 2);
          rvCtx.restore();
        }}
      }});
    }});

    // Also draw compute events as thin gray bars
    EVENTS.forEach(ev => {{
      if (ev.t !== 'compute') return;
      const x = RV_LABEL_W + (ev.s / TOTAL_TIME) * contentW - panX;
      const bw = Math.max((ev.d / TOTAL_TIME) * contentW, 1);
      if (x + bw < RV_LABEL_W || x > w) return;

      rvRanks.forEach((rank, ri) => {{
        const y = ri * (RV_ROW_H + RV_ROW_GAP) + 1;
        const bh = RV_ROW_H - 2;
        rvCtx.globalAlpha = 0.25;
        rvCtx.fillStyle = ev.p === 'fwd' ? '#4b5563' : '#374151';
        rvCtx.fillRect(x, y, bw, bh);
        rvCtx.globalAlpha = 1;
      }});
    }});

    rvCtx.restore();
  }}

  // Tooltip for rank view
  rvCanvas.addEventListener('mousemove', (e) => {{
    if (rvRanks.length === 0) return;
    const rect = rvCanvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const drawW = wrap.clientWidth - RV_LABEL_W;
    const contentW = drawW * zoom;

    // Determine which rank row
    const ri = Math.floor(my / (RV_ROW_H + RV_ROW_GAP));
    if (ri < 0 || ri >= rvRanks.length) {{ rvTip.style.display = 'none'; return; }}
    const rank = rvRanks[ri];

    // Find event under cursor
    let found = null;
    for (let i = EVENTS.length - 1; i >= 0; i--) {{
      const ev = EVENTS[i];
      const x = RV_LABEL_W + (ev.s / TOTAL_TIME) * contentW - panX;
      const bw = Math.max((ev.d / TOTAL_TIME) * contentW, ev.t === 'comm' ? 2 : 1);
      if (mx >= x && mx <= x + bw) {{
        found = ev;
        break;
      }}
    }}

    if (found) {{
      let html = '<b style="color:#58a6ff">Rank ' + rank + '</b><br>';
      html += '<b>Layer:</b> ' + found.n + ' (#' + found.li + ')';
      if (PP_SIZE > 1) html += ' <span style="color:#ef4444">[PP Stage ' + found.ps + ']</span>';
      html += '<br>';
      html += '<b>Phase:</b> ' + (PHASE_LABELS[found.p] || found.p) + '<br>';

      if (found.t === 'comm') {{
        const domain = found.g;
        const color = COLORS[domain] || '#fff';
        html += '<b>Collective:</b> ' + found.ct + '<br>';
        if (found.cs > 0) html += '<b>Size:</b> ' + found.csf + '<br>';
        html += '<b>Domain:</b> <span style="color:' + color + '">' + domain + '</span><br>';

        // Show this rank's specific group for this domain
        const gInfo = findGroupForRank(domain, rank);
        if (gInfo) {{
          const r = gInfo.ranks;
          const rStr = r.length <= 16 ? '[' + r.join(', ') + ']'
            : '[' + r.slice(0, 6).join(', ') + ', ..., ' + r.slice(-4).join(', ') + ']';
          html += '<div style="margin-top:4px;padding-top:4px;border-top:1px solid #334155">';
          html += '<span style="color:' + color + '"><b>' + domain + ' Group #' + gInfo.idx + '</b></span>';
          html += ' (' + r.length + ' GPUs)<br>';
          html += '<span style="color:#8b949e;font-size:10px">通信伙伴: ' + rStr + '</span>';
          html += '</div>';
        }}
      }} else {{
        html += '<b>Type:</b> COMPUTE<br>';
      }}

      rvTip.innerHTML = html;
      rvTip.style.display = 'block';
      rvTip.style.left = (e.clientX + 14) + 'px';
      rvTip.style.top = (e.clientY + 14) + 'px';
    }} else {{
      rvTip.style.display = 'none';
    }}
  }});
  rvCanvas.addEventListener('mouseleave', () => {{ rvTip.style.display = 'none'; }});

  // Sync zoom/pan: hook into the main render cycle
  const origRender = window.render || render;
  // We'll patch render to also draw rank view
  const _origRender = render;

  // Wheel zoom on rank view canvas
  rvWrap.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.25 : 0.8;
    const rect = rvCanvas.getBoundingClientRect();
    zoomAt(factor, e.clientX - rect.left);
  }}, {{ passive: false }});

  // Drag pan on rank view canvas
  let rvDragging = false, rvDragStartX = 0, rvDragStartPan = 0;
  rvWrap.style.cursor = 'grab';
  rvWrap.addEventListener('mousedown', (e) => {{
    rvDragging = true;
    rvDragStartX = e.clientX;
    rvDragStartPan = panX;
    rvWrap.style.cursor = 'grabbing';
  }});
  window.addEventListener('mousemove', (e) => {{
    if (rvDragging) {{
      panX = rvDragStartPan - (e.clientX - rvDragStartX);
      render();
    }}
  }});
  window.addEventListener('mouseup', () => {{
    if (rvDragging) {{
      rvDragging = false;
      rvWrap.style.cursor = 'grab';
    }}
  }});

  function showRankView() {{
    const val = rvInput.value.trim();
    if (!val) {{
      rvError.style.display = 'block';
      rvError.textContent = '请输入 rank 范围，例如 0-7 或 0,2,4,6';
      rvWrap.style.display = 'none';
      rvRanks = [];
      return;
    }}
    const parsed = parseRankRange(val);
    if (!parsed || parsed.length === 0) {{
      rvError.style.display = 'block';
      rvError.textContent = '无效的 rank 范围。有效范围: 0 ~ ' + (ALL_GPUS - 1) + '。格式: 0-7 或 0,2,4,6';
      rvWrap.style.display = 'none';
      rvRanks = [];
      return;
    }}
    if (parsed.length > 64) {{
      rvError.style.display = 'block';
      rvError.textContent = '最多显示 64 个 rank，当前选择了 ' + parsed.length + ' 个';
      rvWrap.style.display = 'none';
      rvRanks = [];
      return;
    }}
    rvError.style.display = 'none';
    rvRanks = parsed;
    rvWrap.style.display = 'block';
    rvResize();
    rvDraw();
  }}

  rvBtn.addEventListener('click', showRankView);
  rvInput.addEventListener('keydown', (e) => {{ if (e.key === 'Enter') showRankView(); }});

  // Expose rvDraw and rvResize so main render can call them
  window._rvDraw = rvDraw;
  window._rvResize = rvResize;
  window._rvActive = () => rvRanks.length > 0;
}})();

// Init
window.addEventListener('resize', () => {{ resize(); if (window._rvActive && window._rvActive()) window._rvResize(); render(); }});
resize();

// Patch render to also draw rank view
const _baseRender = render;
render = function() {{
  _baseRender();
  if (window._rvDraw && window._rvActive && window._rvActive()) window._rvDraw();
}};
render();
</script>
</body>
</html>"""


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <workload_file> [--result EndToEnd.csv] [--max-layers N] [--print-groups] [--rank R]")
        sys.exit(1)

    workload_path = sys.argv[1]
    max_layers = 0
    show_groups = "--print-groups" in sys.argv
    viz_groups = "--visualize-groups" in sys.argv
    rank = 0
    result_path = None

    if "--max-layers" in sys.argv:
        idx = sys.argv.index("--max-layers")
        max_layers = int(sys.argv[idx + 1])
    if "--rank" in sys.argv:
        idx = sys.argv.index("--rank")
        rank = int(sys.argv[idx + 1])
    if "--result" in sys.argv:
        idx = sys.argv.index("--result")
        result_path = sys.argv[idx + 1]

    workload = parse_workload(workload_path)

    if show_groups:
        print_groups(workload, rank=rank, max_layers=max_layers)
        return

    if viz_groups:
        html_content = render_groups_html(workload, rank=rank, max_layers=max_layers)
        out_name = os.path.splitext(os.path.basename(workload_path))[0] + "_groups.html"
        out_path = os.path.join(os.path.dirname(workload_path) or ".", out_name)
        with open(out_path, "w") as f:
            f.write(html_content)
        print(f"Groups visualization written to: {out_path}")
        return

    # Parse EndToEnd.csv if provided
    endtoend_layers = None
    if result_path:
        endtoend_layers = parse_endtoend_csv(result_path)
        print(f"  EndToEnd: {result_path} ({len(endtoend_layers)} layers)")

    events = build_timeline(workload, max_layers, endtoend_layers=endtoend_layers)
    html_content = render_html(workload, events, max_layers, rank=rank)

    out_name = os.path.splitext(os.path.basename(workload_path))[0] + "_timeline.html"
    out_path = os.path.join(os.path.dirname(workload_path) or ".", out_name)
    with open(out_path, "w") as f:
        f.write(html_content)

    print(f"Timeline written to: {out_path}")
    print(f"  Layers: {len(workload['layers'])}" + (f" (showing {max_layers})" if max_layers else ""))
    print(f"  Events: {len(events)}")
    if endtoend_layers:
        compute_events = [e for e in events if e["type"] == "compute"]
        comm_events = [e for e in events if e["type"] == "comm"]
        total_time = max((e["start"] + e["duration"] for e in events), default=0)
        total_comp = sum(e["duration"] for e in compute_events)
        total_comm = sum(e["duration"] for e in comm_events)
        print(f"  Total time: {total_time:.0f}  Compute: {total_comp:.0f} ({total_comp/total_time*100:.1f}%)  Comm: {total_comm:.0f} ({total_comm/total_time*100:.1f}%)")


if __name__ == "__main__":
    main()
