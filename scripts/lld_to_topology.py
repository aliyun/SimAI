"""Convert lld.json to draw.io XML files for the dashboard topology directory.

Output structure:
  topology/overview/all_pod.xml   — multi-POD overview
  topology/pods/<pod_id>.xml      — single POD internal topology

Usage:
  python3 scripts/lld_to_topology.py <lld.json> [--pod-id POD#3] [--output-dir topology]
  python3 scripts/lld_to_topology.py lld1.json lld2.json  # multiple PODs
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET


def _make_node_id(node, ntype):
    nid = node['node_id']
    # Use full node_id with dots/hashes replaced for uniqueness across ODCs
    safe_id = nid.replace('.', '-').replace('#', '-')
    return f"{ntype}_{safe_id}"


def _node_label(node, ntype):
    nid = node['node_id']
    if ntype == 'SERVER':
        chassis = node.get('chassis_topo', 'SERVER')
        name = f"{chassis.split('_')[0]}_{nid.split('.')[-1]}"
    else:
        name = f"{ntype}_{nid.split('.')[-1]}"
    return f"{name}\n{ntype}\n{nid}"


# --- Layout ---

LAYER_Y = {'OXC': 940, 'SPINE': 1100, 'LEAF': 1300, 'SERVER': 1550}
NODE_SIZES = {
    'OXC':    (770, 80),
    'SPINE':  (770, 80),
    'LEAF':   (350, 80),
    'SERVER': (350, 80),
}
BASE_X = 580
PAIR_SPACING = 420


def _spread_positions(count, ntype):
    w, _ = NODE_SIZES[ntype]
    y = LAYER_Y[ntype]
    if count == 1:
        return [(BASE_X, y)]
    total_width = (count - 1) * PAIR_SPACING
    start_x = BASE_X + 385 - total_width // 2
    return [(round(start_x + i * PAIR_SPACING), y) for i in range(count)]


# --- Aggregate edges ---

def _aggregate_edges(raw_edges):
    groups = {}
    for edge in raw_edges:
        src_ip, tgt_ip = edge['a_node_id'], edge['b_node_id']
        key = tuple(sorted([src_ip, tgt_ip]))
        if key not in groups:
            groups[key] = {'a_ports': [], 'b_ports': [], 'a_ip': src_ip, 'b_ip': tgt_ip}
        groups[key]['a_ports'].append(edge['a_node_port_id'])
        groups[key]['b_ports'].append(edge['b_node_port_id'])
    return groups


# --- XML generation ---

def _pretty_xml(root):
    ET.indent(root, space='  ')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)


def generate_pod_xml(lld_data, pod_id='POD#1'):
    topo = lld_data['topology']
    all_nodes = []
    for ntype_key, ntype_label in [('oxc_nodes', 'OXC'), ('spine_nodes', 'SPINE'),
                                    ('leaf_nodes', 'LEAF'), ('server_nodes', 'SERVER')]:
        nodes = topo.get(ntype_key, [])
        positions = _spread_positions(len(nodes), ntype_label)
        for i, node in enumerate(nodes):
            all_nodes.append((node, ntype_label, positions[i]))

    ip_to_cell_id = {}
    for node, node_type, _ in all_nodes:
        cell_id = _make_node_id(node, node_type).lower().replace('_', '-')
        ip_to_cell_id[node["node_id"]] = cell_id

    mxfile = ET.Element('mxfile', host='Electron')
    diagram = ET.SubElement(mxfile, 'diagram', name='LLD Topology', id='lld-topo')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1477', dy='864',
                          grid='1', gridSize='10', guides='1', tooltips='1',
                          connect='1', arrows='1', fold='1', page='1',
                          pageScale='1', pageWidth='1920', pageHeight='2400')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    bb_x = BASE_X - 30
    bb_y = min(LAYER_Y.values()) - 40
    bb_w = max(len(topo.get(k, [])) for k in ['oxc_nodes', 'spine_nodes', 'leaf_nodes', 'server_nodes']) * PAIR_SPACING + 100
    bb_h = max(LAYER_Y.values()) - bb_y + 130
    boundary = ET.SubElement(root, 'mxCell', id='pod-boundary',
                             value=f'para_plane_id: {pod_id}',
                             style='rounded=0;whiteSpace=wrap;fontSize=12;fillColor=none;dashed=1;verticalAlign=top;align=left;fontStyle=1;',
                             vertex='1', parent='1')
    ET.SubElement(boundary, 'mxGeometry', x=str(bb_x), y=str(bb_y),
                  width=str(bb_w), height=str(bb_h)).set('as', 'geometry')

    for node, ntype, (x, y) in all_nodes:
        cell_id = ip_to_cell_id[node["node_id"]]
        w, h = NODE_SIZES[ntype]
        cell = ET.SubElement(root, 'mxCell', id=cell_id,
                             value=_node_label(node, ntype),
                             style='rounded=0;whiteSpace=wrap;align=center;fontStyle=1;fontSize=16;',
                             vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y),
                      width=str(w), height=str(h)).set('as', 'geometry')

    edge_groups = _aggregate_edges(topo.get('edges', []))
    for (ip_a, ip_b), group in edge_groups.items():
        src_id = ip_to_cell_id.get(group['a_ip'], group['a_ip'])
        tgt_id = ip_to_cell_id.get(group['b_ip'], group['b_ip'])
        edge_id = f'e-{src_id}-{tgt_id}'

        edge_cell = ET.SubElement(root, 'mxCell', id=edge_id,
                                  style='edgeStyle=none;rounded=0;orthogonalLoop=1;jettySize=auto;fontStyle=1;',
                                  edge='1', parent='1', source=src_id, target=tgt_id)
        ET.SubElement(edge_cell, 'mxGeometry', relative='1').set('as', 'geometry')

        up_cell = ET.SubElement(root, 'mxCell', id=f'{edge_id}-up',
                                value='0.00%~0.00%',
                                style='edgeLabel;align=center;verticalAlign=middle;resizable=0;points=[];fontSize=12;fontStyle=1;',
                                vertex='1', connectable='0', parent=edge_id)
        geom_up = ET.SubElement(up_cell, 'mxGeometry', relative='1')
        geom_up.set('as', 'geometry')
        geom_up.set('x', '-0.5')

        down_cell = ET.SubElement(root, 'mxCell', id=f'{edge_id}-down',
                                  value='0.00%~0.00%',
                                  style='edgeLabel;align=center;verticalAlign=middle;resizable=0;points=[];fontSize=12;fontStyle=1;',
                                  vertex='1', connectable='0', parent=edge_id)
        geom_down = ET.SubElement(down_cell, 'mxGeometry', relative='1')
        geom_down.set('as', 'geometry')
        geom_down.set('x', '0.5')

    return _pretty_xml(mxfile)


def generate_overview_xml(pod_ids):
    mxfile = ET.Element('mxfile', host='Electron')
    diagram = ET.SubElement(mxfile, 'diagram', name='Overview', id='overview')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='800', dy='600',
                          grid='1', gridSize='10', guides='1', tooltips='1',
                          connect='1', arrows='1', fold='1', page='1',
                          pageScale='1', pageWidth='827', pageHeight='1169')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    spacing = 250
    total_w = (len(pod_ids) - 1) * spacing
    start_x = 500 - total_w // 2

    for i, pid in enumerate(pod_ids):
        cell = ET.SubElement(root, 'mxCell', id=pid.lower().replace('#', ''),
                             value=pid,
                             style='ellipse;whiteSpace=wrap;aspect=fixed;fontSize=16;fillColor=default;fontStyle=1;',
                             vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(start_x + i * spacing), y='780',
                      width='60', height='60').set('as', 'geometry')

    for i in range(len(pod_ids)):
        for j in range(i + 1, len(pod_ids)):
            src = pod_ids[i].lower().replace('#', '')
            tgt = pod_ids[j].lower().replace('#', '')
            edge_id = f'e-{src}-{tgt}'
            edge = ET.SubElement(root, 'mxCell', id=edge_id,
                                 style='edgeStyle=none;rounded=0;fontSize=16;startArrow=classic;startFill=1;endArrow=classic;endFill=1;fontStyle=1;',
                                 edge='1', parent='1', source=src, target=tgt)
            ET.SubElement(edge, 'mxGeometry', relative='1').set('as', 'geometry')

    return _pretty_xml(mxfile)


def _resolve_cross_to_leaves(lld_data, crosses_list):
    """Resolve OXC crosses to leaf-leaf pairs.

    Args:
        lld_data: parsed lld.json
        crosses_list: list of {node_ip, a_port_id, b_port_id}

    Returns:
        list of (leaf_a_ip, leaf_b_ip, oxc_ip, port_a, port_b)
    """
    topo = lld_data.get('topology', {})
    oxc_ips = {n["node_id"] for n in topo.get('oxc_nodes', [])}
    edges = topo.get('edges', [])

    oxc_port_to_leaf = {}
    for e in edges:
        a_ip, b_ip = e["a_node_id"], e["b_node_id"]
        a_port, b_port = str(e['a_node_port_id']), str(e['b_node_port_id'])
        if a_ip in oxc_ips:
            oxc_port_to_leaf[(a_ip, a_port)] = b_ip
        elif b_ip in oxc_ips:
            oxc_port_to_leaf[(b_ip, b_port)] = a_ip

    result = []
    for c in crosses_list:
        oxc_ip = c.get('node_ip', '')
        pa, pb = str(c.get('a_port_id', '')), str(c.get('b_port_id', ''))
        leaf_a = oxc_port_to_leaf.get((oxc_ip, pa))
        leaf_b = oxc_port_to_leaf.get((oxc_ip, pb))
        if leaf_a and leaf_b and leaf_a != leaf_b:
            result.append((leaf_a, leaf_b, oxc_ip, pa, pb))
    return result


def generate_pod_xml_with_crosses(lld_data, crosses_list, participating_server_ips, task_id, pod_id='POD#1'):
    """Generate pod XML with OXC cross highlights and task server markers.

    Args:
        lld_data: parsed lld.json
        crosses_list: list of {node_ip, a_port_id, b_port_id}
        participating_server_ips: set of server IPs in the task
        task_id: task identifier for labeling
        pod_id: POD identifier

    Returns:
        XML string
    """
    topo = lld_data['topology']
    all_nodes = []
    for ntype_key, ntype_label in [('oxc_nodes', 'OXC'), ('spine_nodes', 'SPINE'),
                                    ('leaf_nodes', 'LEAF'), ('server_nodes', 'SERVER')]:
        nodes = topo.get(ntype_key, [])
        positions = _spread_positions(len(nodes), ntype_label)
        for i, node in enumerate(nodes):
            all_nodes.append((node, ntype_label, positions[i]))

    ip_to_cell_id = {}
    for node, node_type, _ in all_nodes:
        cell_id = _make_node_id(node, node_type).lower().replace('_', '-')
        ip_to_cell_id[node["node_id"]] = cell_id

    mxfile = ET.Element('mxfile', host='Electron')
    diagram = ET.SubElement(mxfile, 'diagram', name='LLD Topology', id='lld-topo')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1477', dy='864',
                          grid='1', gridSize='10', guides='1', tooltips='1',
                          connect='1', arrows='1', fold='1', page='1',
                          pageScale='1', pageWidth='1920', pageHeight='2400')
    root = ET.SubElement(model, 'root')
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    bb_x = BASE_X - 30
    bb_y = min(LAYER_Y.values()) - 40
    bb_w = max(len(topo.get(k, [])) for k in ['oxc_nodes', 'spine_nodes', 'leaf_nodes', 'server_nodes']) * PAIR_SPACING + 100
    bb_h = max(LAYER_Y.values()) - bb_y + 130
    boundary = ET.SubElement(root, 'mxCell', id='pod-boundary',
                             value=f'para_plane_id: {pod_id}',
                             style='rounded=0;whiteSpace=wrap;fontSize=12;fillColor=none;dashed=1;verticalAlign=top;align=left;fontStyle=1;',
                             vertex='1', parent='1')
    ET.SubElement(boundary, 'mxGeometry', x=str(bb_x), y=str(bb_y),
                  width=str(bb_w), height=str(bb_h)).set('as', 'geometry')

    participating_ips = set(participating_server_ips) if participating_server_ips else set()

    for node, ntype, (x, y) in all_nodes:
        cell_id = ip_to_cell_id[node["node_id"]]
        w, h = NODE_SIZES[ntype]
        node_ip = node["node_id"]

        label = _node_label(node, ntype)
        style = 'rounded=0;whiteSpace=wrap;align=center;fontStyle=1;fontSize=16;'

        if ntype == 'SERVER' and node_ip in participating_ips:
            style = 'rounded=0;whiteSpace=wrap;align=center;fontStyle=1;fontSize=16;fillColor=#d5e8d4;strokeColor=#82b366;'
            label += f'\n[{task_id}]'

        cell = ET.SubElement(root, 'mxCell', id=cell_id,
                             value=label, style=style,
                             vertex='1', parent='1')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y),
                      width=str(w), height=str(h)).set('as', 'geometry')

    # Physical edges (same as baseline)
    edge_groups = _aggregate_edges(topo.get('edges', []))
    for (ip_a, ip_b), group in edge_groups.items():
        src_id = ip_to_cell_id.get(group['a_ip'], group['a_ip'])
        tgt_id = ip_to_cell_id.get(group['b_ip'], group['b_ip'])
        edge_id = f'e-{src_id}-{tgt_id}'

        edge_cell = ET.SubElement(root, 'mxCell', id=edge_id,
                                  style='edgeStyle=none;rounded=0;orthogonalLoop=1;jettySize=auto;fontStyle=1;',
                                  edge='1', parent='1', source=src_id, target=tgt_id)
        ET.SubElement(edge_cell, 'mxGeometry', relative='1').set('as', 'geometry')

        up_cell = ET.SubElement(root, 'mxCell', id=f'{edge_id}-up',
                                value='0.00%~0.00%',
                                style='edgeLabel;align=center;verticalAlign=middle;resizable=0;points=[];fontSize=12;fontStyle=1;',
                                vertex='1', connectable='0', parent=edge_id)
        geom_up = ET.SubElement(up_cell, 'mxGeometry', relative='1')
        geom_up.set('as', 'geometry')
        geom_up.set('x', '-0.5')

        down_cell = ET.SubElement(root, 'mxCell', id=f'{edge_id}-down',
                                  value='0.00%~0.00%',
                                  style='edgeLabel;align=center;verticalAlign=middle;resizable=0;points=[];fontSize=12;fontStyle=1;',
                                  vertex='1', connectable='0', parent=edge_id)
        geom_down = ET.SubElement(down_cell, 'mxGeometry', relative='1')
        geom_down.set('as', 'geometry')
        geom_down.set('x', '0.5')

    # OXC cross-activated leaf-leaf logical edges — disabled per user request.
    # The monitoring dashboard should match the original XML exactly,
    # showing only physical edges (OXC→Leaf, Leaf→Server).
    # To re-enable, uncomment the block below.
    # leaf_leaf_pairs = _resolve_cross_to_leaves(lld_data, crosses_list)
    # seen_ll = set()
    # for (leaf_a, leaf_b, oxc_ip, pa, pb) in leaf_leaf_pairs:
    #     pair_key = tuple(sorted([leaf_a, leaf_b]))
    #     if pair_key in seen_ll:
    #         continue
    #     seen_ll.add(pair_key)
    #     src_id = ip_to_cell_id.get(leaf_a)
    #     tgt_id = ip_to_cell_id.get(leaf_b)
    #     if not src_id or not tgt_id:
    #         continue
    #     ll_edge_id = f'e-ll-{src_id}-{tgt_id}'
    #     ll_edge = ET.SubElement(root, 'mxCell', id=ll_edge_id,
    #                             style='edgeStyle=none;rounded=0;dashed=1;strokeColor=#FF6600;strokeWidth=2;fontStyle=1;',
    #                             edge='1', parent='1', source=src_id, target=tgt_id)
    #     ET.SubElement(ll_edge, 'mxGeometry', relative='1').set('as', 'geometry')
    #     ll_label = ET.SubElement(root, 'mxCell', id=f'{ll_edge_id}-label',
    #                              value=f'OXC:{pa}↔{pb}',
    #                              style='edgeLabel;align=center;verticalAlign=middle;resizable=0;points=[];fontSize=11;fontColor=#FF6600;fontStyle=1;',
    #                              vertex='1', connectable='0', parent=ll_edge_id)
    #     geom_ll = ET.SubElement(ll_label, 'mxGeometry', relative='1')
    #     geom_ll.set('as', 'geometry')

    return _pretty_xml(mxfile)


def _detect_group_ids(lld_data):
    """Return sorted unique group_ids from server_nodes and leaf_nodes.

    Falls back to a single '0' group if no group_id fields are present.
    """
    topo = lld_data.get('topology', {})
    groups = set()
    for node in topo.get('server_nodes', []):
        gid = node.get('group_id')
        if gid is not None:
            groups.add(str(gid))
    for node in topo.get('leaf_nodes', []):
        gid = node.get('group_id')
        if gid is not None:
            groups.add(str(gid))
    if not groups:
        return ['0']
    return sorted(groups, key=lambda x: int(x) if x.isdigit() else x)


def _split_topology_by_group(lld_data, group_id):
    """Return a subset of the topology containing only nodes/edges for the given group_id.

    Servers and leaves are filtered by group_id. OXC and spine nodes are included in
    every group since they represent shared fabric. Edges are included when at least
    one endpoint belongs to the group.
    """
    topo = lld_data.get('topology', {})

    shared_ntypes = ('oxc_nodes', 'spine_nodes')
    grouped_ntypes = ('server_nodes', 'leaf_nodes')

    group_node_ids = set()
    filtered = {'edges': []}

    for ntype in shared_ntypes:
        filtered[ntype] = topo.get(ntype, [])
        for n in filtered[ntype]:
            group_node_ids.add(n['node_id'])

    for ntype in grouped_ntypes:
        filtered[ntype] = [
            n for n in topo.get(ntype, [])
            if str(n.get('group_id', '')) == group_id
        ]
        for n in filtered[ntype]:
            group_node_ids.add(n['node_id'])

    for edge in topo.get('edges', []):
        a_id = edge.get('a_node_id', '')
        b_id = edge.get('b_node_id', '')
        if a_id in group_node_ids or b_id in group_node_ids:
            filtered['edges'].append(edge)

    return {'topology': filtered}


def main():
    parser = argparse.ArgumentParser(description='Convert lld.json to dashboard topology XML')
    parser.add_argument('inputs', nargs='+', help='lld.json file(s)')
    parser.add_argument('--pod-id', nargs='*', help='POD ID(s), default: auto-detect from group_id')
    parser.add_argument('--output-dir', default='topology', help='Output directory (default: topology)')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(os.path.join(output_dir, 'overview'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pods'), exist_ok=True)

    if args.pod_id:
        pod_ids = args.pod_id
        input_groups = [(lld_path, lld_path) for lld_path in args.inputs]
        if len(pod_ids) < len(args.inputs):
            pod_ids.extend(f'POD#{i+1}' for i in range(len(pod_ids), len(args.inputs)))
    else:
        if len(args.inputs) == 1:
            lld_data = json.loads(open(args.inputs[0]).read())
            group_ids = _detect_group_ids(lld_data)
            pod_ids = [f'POD#{gid}' for gid in group_ids]
            input_groups = [(args.inputs[0], gid) for gid in group_ids]
        else:
            pod_ids = [f'POD#{i+1}' for i in range(len(args.inputs))]
            input_groups = [(lp, lp) for lp in args.inputs]

    for (lld_path, group_id), pod_id in zip(input_groups, pod_ids):
        lld_data = json.loads(open(lld_path).read())
        if len(args.inputs) == 1 and not args.pod_id:
            pod_lld = _split_topology_by_group(lld_data, group_id)
        else:
            pod_lld = lld_data
        pod_xml = generate_pod_xml(pod_lld, pod_id)
        pod_file = os.path.join(output_dir, 'pods', f'{pod_id}.xml')
        with open(pod_file, 'w', encoding='utf-8') as f:
            f.write(pod_xml)
        print(f"  {lld_path} (group={group_id}) -> {pod_file}")

    overview_xml = generate_overview_xml(pod_ids)
    overview_file = os.path.join(output_dir, 'overview', 'all_pod.xml')
    with open(overview_file, 'w', encoding='utf-8') as f:
        f.write(overview_xml)
    print(f"  Overview -> {overview_file} ({len(pod_ids)} PODs)")


if __name__ == '__main__':
    main()
