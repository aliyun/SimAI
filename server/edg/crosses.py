"""OXC cross-connect state management: parse, patch, merge."""

from typing import Any, Dict, FrozenSet, List, Set, Tuple

Cross = Tuple[str, str, str]  # (oxc_ip, port_a, port_b) — ports always sorted


def _normalize(oxc_ip: str, a: str, b: str) -> Cross:
    pa, pb = sorted([str(a), str(b)])
    return (oxc_ip, pa, pb)


def parse_add_list(entries: List[Dict[str, Any]]) -> Set[Cross]:
    result: Set[Cross] = set()
    for e in entries:
        ip = e.get("node_ip", "")
        a = e.get("a_port_id", "")
        b = e.get("b_port_id", "")
        if ip and a and b:
            result.add(_normalize(ip, a, b))
    return result


def parse_del_list(entries: List[Dict[str, Any]]) -> Set[Cross]:
    return parse_add_list(entries)


def apply_patch(
    base: Set[Cross],
    del_list: List[Dict[str, Any]],
    add_list: List[Dict[str, Any]],
) -> Set[Cross]:
    to_del = parse_del_list(del_list)
    to_add = parse_add_list(add_list)
    return (base - to_del) | to_add


def apply_batches(
    base: Set[Cross],
    batches: Any,
) -> Set[Cross]:
    """Apply multiple batches of del/add operations sequentially.

    `batches` can be:
    - a list of dicts (multi-batch from NOTIFY_NODE_MATRIX)
    - a single dict (from IMPORT_FULL_TOPO)
    """
    if isinstance(batches, dict):
        batches = [batches]

    current = set(base)
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        del_list = batch.get("del_oper_info_list", [])
        add_list = batch.get("add_oper_info_list", [])
        # Filter out empty dicts
        del_list = [e for e in del_list if e and e.get("node_ip")]
        add_list = [e for e in add_list if e and e.get("node_ip")]
        current = apply_patch(current, del_list, add_list)
    return current
