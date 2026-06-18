#!/usr/bin/env python3
"""
read_flow_then_inject.py — Parse SimAI decoupled-replay flow files and inject
flows into the NS3 replay binary.

Flow file format (21 fields per line, header line = total flow count):
  flow_id src dest flow_size channel_id chunk_id chunk_count conn_type
  start_time pg maxPacketCount port dport
  np prev[0..np-1]
  npar parent_flow_id[0..npar-1] nchi child_flow_id[0..nchi-1]
  layer_num group_type op loopstate relative_delay_ns

Usage:
  # Parse and dump statistics
  python3 read_flow_then_inject.py -f flow_file.txt --stats

  # Inject via decoupled replay binary
  python3 read_flow_then_inject.py -f flow_file.txt -t /path/to/topo_dir --inject

  # Validate only (check all prev[] references are valid)
  python3 read_flow_then_inject.py -f flow_file.txt --validate

  # Inject with custom arguments
  python3 read_flow_then_inject.py -f flow_file.txt -t Spectrum-X_32g_8gps_400Gbps_A5 \\
      -c SimAI.conf -o /tmp/fct.txt -s 2000000000 --dump-layer-stats
"""

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class FlowFileRecord:
    """Complete parsed record with all 21 fields."""
    __slots__ = (
        "flow_id", "src", "dst", "flow_size",
        "channel_id", "chunk_id", "chunk_count", "conn_type",
        "start_time", "pg", "max_packet_count", "port", "dport",
        "prev", "parent_flow_id", "child_flow_id",
        "layer_num", "group_type", "op", "loopstate",
        "relative_delay_ns",
    )

    def __init__(self):
        self.flow_id: int = 0
        self.src: int = 0
        self.dst: int = 0
        self.flow_size: int = 0
        self.channel_id: int = 0
        self.chunk_id: int = 0
        self.chunk_count: int = 0
        self.conn_type: str = ""
        self.start_time: float = 0.0
        self.pg: int = 0
        self.max_packet_count: int = 0
        self.port: int = 0
        self.dport: int = 0
        self.prev: List[int] = []
        self.parent_flow_id: List[int] = []
        self.child_flow_id: List[int] = []
        self.layer_num: int = 0
        self.group_type: int = 0
        self.op: int = 0
        self.loopstate: int = 0
        self.relative_delay_ns: int = 0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_flows(flow_file_path: str) -> Tuple[Optional[List[FlowFileRecord]], Optional[str]]:
    """
    Parse a flow file. Returns (records, error_message).
    On success error_message is None; on failure records is None.
    """
    try:
        with open(flow_file_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        return None, f"Cannot open flow file: {flow_file_path}"
    except OSError as e:
        return None, f"OS error reading {flow_file_path}: {e}"

    if not lines:
        return None, f"Flow file is empty: {flow_file_path}"

    # Header: expected flow count
    header = lines[0]
    try:
        expected = int(header)
    except ValueError:
        return None, f"Bad header in {flow_file_path}: expected integer count, got '{header}'"

    if expected == 0:
        return None, f"Flow file header says 0 flows (empty or placeholder)"

    records: List[FlowFileRecord] = []
    for line_num, line in enumerate(lines[1:], start=2):
        rec = FlowFileRecord()
        parts = line.split()
        idx = 0  # cursor into parts

        try:
            # Fields 1-8: fixed scalars
            rec.flow_id        = int(parts[idx]); idx += 1
            rec.src            = int(parts[idx]); idx += 1
            rec.dst            = int(parts[idx]); idx += 1
            rec.flow_size      = int(parts[idx]); idx += 1
            rec.channel_id     = int(parts[idx]); idx += 1
            rec.chunk_id       = int(parts[idx]); idx += 1
            rec.chunk_count    = int(parts[idx]); idx += 1
            rec.conn_type      = parts[idx];       idx += 1

            # Fields 9-13: start_time, pg, max_packet_count, port, dport, np
            rec.start_time       = float(parts[idx]); idx += 1
            rec.pg               = int(parts[idx]);   idx += 1
            rec.max_packet_count = int(parts[idx]);   idx += 1
            rec.port             = int(parts[idx]);   idx += 1
            rec.dport            = int(parts[idx]);   idx += 1
            np = int(parts[idx]); idx += 1

            # Field 14: prev[] (variable length)
            if np > 0:
                if idx + np > len(parts):
                    raise ValueError(f"prev[]: expected {np} entries, only {len(parts) - idx} available")
                rec.prev = [int(x) for x in parts[idx:idx + np]]
                idx += np

            # Fields 15-16: parent_flow_id[], child_flow_id[] (variable length)
            npar = int(parts[idx]); idx += 1
            if npar > 0:
                if idx + npar > len(parts):
                    raise ValueError(f"parent_flow_id[]: expected {npar} entries")
                rec.parent_flow_id = [int(x) for x in parts[idx:idx + npar]]
                idx += npar

            nchi = int(parts[idx]); idx += 1
            if nchi > 0:
                if idx + nchi > len(parts):
                    raise ValueError(f"child_flow_id[]: expected {nchi} entries")
                rec.child_flow_id = [int(x) for x in parts[idx:idx + nchi]]
                idx += nchi

            # Fields 17-20: layer_num, group_type, op, loopstate
            rec.layer_num  = int(parts[idx]); idx += 1
            rec.group_type = int(parts[idx]); idx += 1
            rec.op         = int(parts[idx]); idx += 1
            rec.loopstate  = int(parts[idx]); idx += 1

            # Field 21 (NEW): relative_delay_ns with legacy fallback
            if idx < len(parts):
                rec.relative_delay_ns = int(parts[idx])
            else:
                rec.relative_delay_ns = 0  # legacy format fallback

        except (ValueError, IndexError) as e:
            return None, f"Parse error at line {line_num}: {e}"

        records.append(rec)

    if len(records) != expected:
        print(f"[WARNING] Expected {expected} flows, parsed {len(records)}", file=sys.stderr)

    return records, None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_flows(records: List[FlowFileRecord]) -> List[str]:
    """Validate flow records. Returns list of error messages (empty = valid)."""
    errors: List[str] = []
    flow_ids = {r.flow_id for r in records}

    # Check prev[] references
    for r in records:
        for pid in r.prev:
            if pid not in flow_ids:
                errors.append(
                    f"flow {r.flow_id}: prev[] references non-existent flow {pid}"
                )

    # Check parent_flow_id[] references are valid or -1 (no parent)
    for r in records:
        for pid in r.parent_flow_id:
            if pid != -1 and pid not in flow_ids:
                errors.append(
                    f"flow {r.flow_id}: parent_flow_id[] references unknown flow {pid}"
                )

    # Check child_flow_id[] references
    for r in records:
        for cid in r.child_flow_id:
            if cid != -1 and cid not in flow_ids:
                errors.append(
                    f"flow {r.flow_id}: child_flow_id[] references unknown flow {cid}"
                )

    # Check duplicate flow IDs
    seen = set()
    for r in records:
        if r.flow_id in seen:
            errors.append(f"Duplicate flow_id: {r.flow_id}")
        seen.add(r.flow_id)

    # Check basic sanity
    for r in records:
        if r.flow_size == 0 and r.src == r.dst:
            errors.append(f"flow {r.flow_id}: both flow_size=0 and src==dst (likely invalid)")

    return errors


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def dump_stats(records: List[FlowFileRecord]):
    """Print flow statistics to stdout."""
    print(f"Total flows: {len(records)}")

    # By layer
    layers = defaultdict(list)
    for r in records:
        layers[r.layer_num].append(r)
    print(f"Layers: {len(layers)}")
    for ln in sorted(layers):
        print(f"  Layer {ln}: {len(layers[ln])} flows")

    # By group_type
    groups = defaultdict(list)
    for r in records:
        groups[r.group_type].append(r)
    group_names = {0: "TP", 1: "DP", 2: "EP", 3: "DP_EP"}
    print("By group type:")
    for gt in sorted(groups):
        name = group_names.get(gt, f"UNKNOWN({gt})")
        print(f"  {name}: {len(groups[gt])} flows")

    # By op (ComType)
    ops = defaultdict(list)
    op_names = {0: "None", 1: "Reduce_Scatter", 2: "All_Gather", 3: "All_Reduce",
                4: "All_to_All", 5: "All_Reduce_All_to_All"}
    for r in records:
        ops[r.op].append(r)
    print("By operation:")
    for op in sorted(ops):
        name = op_names.get(op, f"UNKNOWN({op})")
        print(f"  {name}: {len(ops[op])} flows")

    # By conn_type
    conns = defaultdict(list)
    for r in records:
        conns[r.conn_type].append(r)
    print("By connection type:")
    for ct in sorted(conns):
        print(f"  {ct}: {len(conns[ct])} flows")

    # relative_delay_ns distribution
    delays = [r.relative_delay_ns for r in records]
    print(f"\nrelative_delay_ns stats:")
    print(f"  min:        {min(delays):>12} ns")
    print(f"  max:        {max(delays):>12} ns")
    print(f"  mean:       {sum(delays) / len(delays):>12.0f} ns")
    sorted_delays = sorted(delays)
    n = len(sorted_delays)
    print(f"  median:     {sorted_delays[n // 2]:>12} ns")
    print(f"  p95:        {sorted_delays[int(n * 0.95)]:>12} ns")
    print(f"  p99:        {sorted_delays[int(n * 0.99)]:>12} ns")
    zero_count = sum(1 for d in delays if d == 0)
    print(f"  zero-delay: {zero_count:>12} flows ({100 * zero_count / n:.1f}%)")

    # Flow size distribution
    sizes = [r.flow_size for r in records]
    print(f"\nflow_size stats:")
    print(f"  min:   {min(sizes):>12} bytes")
    print(f"  max:   {max(sizes):>12} bytes")
    print(f"  total: {sum(sizes):>12} bytes ({sum(sizes) / 1e9:.2f} GB)")

    # Dependency count distribution
    prev_counts = defaultdict(int)
    for r in records:
        prev_counts[len(r.prev)] += 1
    print(f"\nDependency count distribution (prev[] size):")
    for sz in sorted(prev_counts):
        print(f"  {sz} deps: {prev_counts[sz]} flows")
    has_parent = sum(1 for r in records if len(r.parent_flow_id) > 0)
    has_child = sum(1 for r in records if len(r.child_flow_id) > 0)
    print(f"  Flows with parent(s):  {has_parent}")
    print(f"  Flows with child(ren): {has_child}")

    # Layer-level timing
    print(f"\nLayer FCT estimates (from relative_delay_ns + max flow size / bandwidth):")
    for ln in sorted(layers):
        layer_recs = layers[ln]
        total_delay = sum(r.relative_delay_ns for r in layer_recs)
        max_size = max(r.flow_size for r in layer_recs)
        print(f"  Layer {ln}: {len(layer_recs)} flows, "
              f"total delay {total_delay / 1e6:.2f} ms, "
              f"max flow size {max_size} bytes")


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def find_binary() -> str:
    """Find the decoupled replay binary."""
    # Check standard locations
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "simulation", "build", "scratch_decoupled_replay"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "build", "scratch_decoupled_replay"),
        "scratch_decoupled_replay",
        "./scratch_decoupled_replay",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)
    return "scratch_decoupled_replay"  # fallback to PATH


def inject_flows(args: argparse.Namespace, flow_file: str):
    """Run the decoupled replay binary with the given flow file."""
    binary = args.binary or find_binary()
    cmd = [binary]

    cmd.extend(["-f", flow_file])

    if args.topo_dir:
        cmd.extend(["-t", args.topo_dir])
    if args.config:
        cmd.extend(["-c", args.config])
    if args.fct_output:
        cmd.extend(["-o", args.fct_output])
    if args.stop_time is not None:
        cmd.extend(["-s", str(args.stop_time)])
    if args.verify_dag:
        cmd.append("--verify-dag")
    if args.dump_layer_stats:
        cmd.append("--dump-layer-stats")

    print(f"[INJECT] Running: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[INJECT] Binary exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print("[INJECT] Complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parse SimAI decoupled-replay flow files and inject into NS3 reply binary."
    )
    parser.add_argument("-f", "--flow-file", required=True,
                        help="Path to flow file (required)")
    parser.add_argument("--stats", action="store_true",
                        help="Print flow statistics")
    parser.add_argument("--validate", action="store_true",
                        help="Validate flow records (check references)")
    parser.add_argument("--inject", action="store_true",
                        help="Inject flows via the decoupled replay binary")

    # Injection options
    parser.add_argument("-t", "--topo-dir",
                        help="Topology directory for replay")
    parser.add_argument("-c", "--config",
                        help="SimAI.conf path (default: topo-dir/SimAI.conf)")
    parser.add_argument("-o", "--fct-output",
                        help="FCT output file path")
    parser.add_argument("-s", "--stop-time", type=float,
                        help="Simulator stop time in seconds")
    parser.add_argument("--verify-dag", action="store_true",
                        help="Run DAG cycle detection before simulation")
    parser.add_argument("--dump-layer-stats", action="store_true",
                        help="Output per-layer timing statistics")
    parser.add_argument("--binary",
                        help="Path to decoupled replay binary")

    args = parser.parse_args()

    # Must specify at least one action
    if not (args.stats or args.validate or args.inject):
        parser.error("Specify at least one action: --stats, --validate, or --inject")

    # Parse
    records, error = load_flows(args.flow_file)
    if error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)

    # Validate
    if args.validate:
        errors = validate_flows(records)
        if errors:
            print(f"VALIDATION FAILED ({len(errors)} errors):", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            sys.exit(1)
        print(f"VALIDATION PASSED ({len(records)} flows)")

    # Stats
    if args.stats:
        dump_stats(records)

    # Inject
    if args.inject:
        inject_flows(args, args.flow_file)


if __name__ == "__main__":
    main()
