from pybindgen import param, retval


def post_register_types(root_module):
    root_module.add_include('"ns3/propagation-module.h"')

    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
            param("std::string &", "param5_name"),
            param("ns3::AttributeValue const &", "param5_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetType",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
            param("std::string &", "param5_name"),
            param("ns3::AttributeValue const &", "param5_value"),
            param("std::string &", "param6_name"),
            param("ns3::AttributeValue const &", "param6_value"),
        ],
    )

    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
            param("std::string &", "param5_name"),
            param("ns3::AttributeValue const &", "param5_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetProtectionManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
            param("std::string &", "param5_name"),
            param("ns3::AttributeValue const &", "param5_value"),
            param("std::string &", "param6_name"),
            param("ns3::AttributeValue const &", "param6_value"),
        ],
    )

    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
            param("std::string &", "param5_name"),
            param("ns3::AttributeValue const &", "param5_value"),
        ],
    )
    root_module["WifiMacHelper"].add_method(
        "SetAckManager",
        retval("void"),
        [
            param("std::string", "type"),
            param("std::string &", "param1_name"),
            param("ns3::AttributeValue const &", "param1_value"),
            param("std::string &", "param2_name"),
            param("ns3::AttributeValue const &", "param2_value"),
            param("std::string &", "param3_name"),
            param("ns3::AttributeValue const &", "param3_value"),
            param("std::string &", "param4_name"),
            param("ns3::AttributeValue const &", "param4_value"),
            param("std::string &", "param5_name"),
            param("ns3::AttributeValue const &", "param5_value"),
            param("std::string &", "param6_name"),
            param("ns3::AttributeValue const &", "param6_value"),
        ],
    )
