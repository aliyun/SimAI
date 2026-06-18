"""Unit tests for server.edg.crosses — patch math."""

from server.edg.crosses import apply_patch, apply_batches, parse_add_list


def test_parse_add_list_basic():
    entries = [
        {"node_ip": "10.1.1.1", "a_port_id": "1", "b_port_id": "16"},
        {"node_ip": "10.1.1.1", "a_port_id": "2", "b_port_id": "12"},
    ]
    result = parse_add_list(entries)
    assert result == {("10.1.1.1", "1", "16"), ("10.1.1.1", "12", "2")}


def test_parse_add_list_normalizes_port_order():
    entries = [{"node_ip": "10.1.1.1", "a_port_id": "16", "b_port_id": "1"}]
    result = parse_add_list(entries)
    assert result == {("10.1.1.1", "1", "16")}


def test_parse_add_list_skips_empty():
    entries = [{}]
    result = parse_add_list(entries)
    assert result == set()


def test_apply_patch_add_and_delete():
    base = {("10.1.1.1", "1", "16"), ("10.1.1.1", "12", "2")}
    del_list = [{"node_ip": "10.1.1.1", "a_port_id": "1", "b_port_id": "16"}]
    add_list = [{"node_ip": "10.1.1.1", "a_port_id": "3", "b_port_id": "7"}]
    result = apply_patch(base, del_list, add_list)
    assert ("10.1.1.1", "1", "16") not in result
    assert ("10.1.1.1", "12", "2") in result
    assert ("10.1.1.1", "3", "7") in result


def test_apply_patch_idempotent_add():
    base = {("10.1.1.1", "1", "16")}
    add_list = [{"node_ip": "10.1.1.1", "a_port_id": "1", "b_port_id": "16"}]
    result = apply_patch(base, [], add_list)
    assert result == base


def test_apply_patch_idempotent_delete():
    base = {("10.1.1.1", "1", "16")}
    del_list = [{"node_ip": "10.1.1.1", "a_port_id": "99", "b_port_id": "100"}]
    result = apply_patch(base, del_list, [])
    assert result == base


def test_apply_batches_single_dict():
    orders = {
        "del_oper_info_list": [{}],
        "add_oper_info_list": [
            {"node_ip": "10.1.1.1", "a_port_id": "1", "b_port_id": "16"},
        ],
    }
    result = apply_batches(set(), orders)
    assert result == {("10.1.1.1", "1", "16")}


def test_apply_batches_multi_batch():
    base = {("10.1.1.1", "1", "16")}
    batches = [
        {
            "del_oper_info_list": [{"node_ip": "10.1.1.1", "a_port_id": "1", "b_port_id": "16"}],
            "add_oper_info_list": [{"node_ip": "10.1.1.1", "a_port_id": "3", "b_port_id": "7"}],
        },
        {
            "del_oper_info_list": [],
            "add_oper_info_list": [{"node_ip": "10.1.1.1", "a_port_id": "5", "b_port_id": "9"}],
        },
    ]
    result = apply_batches(base, batches)
    assert ("10.1.1.1", "1", "16") not in result
    assert ("10.1.1.1", "3", "7") in result
    assert ("10.1.1.1", "5", "9") in result


def test_apply_batches_order_independent_within_batch():
    add1 = [
        {"node_ip": "10.1.1.1", "a_port_id": "1", "b_port_id": "16"},
        {"node_ip": "10.1.1.1", "a_port_id": "2", "b_port_id": "12"},
    ]
    add2 = list(reversed(add1))
    r1 = apply_batches(set(), {"del_oper_info_list": [], "add_oper_info_list": add1})
    r2 = apply_batches(set(), {"del_oper_info_list": [], "add_oper_info_list": add2})
    assert r1 == r2
