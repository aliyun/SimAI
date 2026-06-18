/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Blake Hurd
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 * Author: Blake Hurd  <naimorai@gmail.com>
 */

// An essential include is test.h
#include "ns3/test.h"

#include "ns3/openflow-switch-net-device.h"
#include "ns3/openflow-interface.h"

// Do not put your test classes in namespace ns3.  You may find it useful
// to use the using directive to access the ns3 namespace directly
using namespace ns3;


/**
 * \ingroup openflow-tests
 *
 * \brief OpenFlow Test
 */
class SwitchFlowTableTestCase : public TestCase
{
public:
  SwitchFlowTableTestCase () : TestCase ("Switch test case")
  {
    m_chain = chain_create ();
  }

  virtual ~SwitchFlowTableTestCase ()
  {
    chain_destroy (m_chain);
  }

private:
  virtual void DoRun (void);

  sw_chain* m_chain; //!< OpenFlow service function chain
};

void
SwitchFlowTableTestCase::DoRun (void)
{
  // Flow Table implementation is used by the OpenFlowSwitchNetDevice under the chain_ methods
  // we should test its implementation to verify the flow table works.

  // Initialization
  time_init (); // OFSI requires this, otherwise we crash before we can do anything.

  size_t actions_len = 0; // Flow is created with 0 actions.
  int output_port = 0; // Flow will be modified later with an action to output on port 0.

  Mac48Address dl_src ("00:00:00:00:00:00"), dl_dst ("00:00:00:00:00:01");
  Ipv4Address nw_src ("192.168.1.1"), nw_dst ("192.168.1.2");
  int tp_src = 5000, tp_dst = 80;

  // Create an sw_flow_key; in actual usage this is generated from the received packet's headers.
  sw_flow_key key;
  key.wildcards = 0;

  key.flow.in_port = htons (0);

  key.flow.dl_vlan = htons (OFP_VLAN_NONE);
  key.flow.dl_type = htons (ETH_TYPE_IP);
  key.flow.nw_proto = htons (IP_TYPE_UDP);

  key.flow.reserved = 0;
  key.flow.mpls_label1 = htonl (MPLS_INVALID_LABEL);
  key.flow.mpls_label2 = htonl (MPLS_INVALID_LABEL);

  // Set Mac Addresses
  dl_src.CopyTo (key.flow.dl_src);
  dl_dst.CopyTo (key.flow.dl_dst);

  // Set IP Addresses
  key.flow.nw_src = htonl (nw_src.Get ());
  key.flow.nw_dst = htonl (nw_dst.Get ());

  // Set TCP/UDP Ports
  key.flow.tp_src = htonl (tp_src);
  key.flow.tp_dst = htonl (tp_dst);

  // Create flow
  ofp_flow_mod ofm;
  ofm.header.version = OFP_VERSION;
  ofm.header.type = OFPT_FLOW_MOD;
  ofm.header.length = htons (sizeof (ofp_flow_mod) + actions_len);
  ofm.command = htons (OFPFC_ADD);
  ofm.idle_timeout = htons (OFP_FLOW_PERMANENT);
  ofm.hard_timeout = htons (OFP_FLOW_PERMANENT);
  ofm.buffer_id = htonl (-1);
  ofm.priority = OFP_DEFAULT_PRIORITY;

  ofm.match.wildcards = key.wildcards;                                 // Wildcard fields
  ofm.match.in_port = key.flow.in_port;                                // Input switch port
  memcpy (ofm.match.dl_src, key.flow.dl_src, sizeof ofm.match.dl_src); // Ethernet source address.
  memcpy (ofm.match.dl_dst, key.flow.dl_dst, sizeof ofm.match.dl_dst); // Ethernet destination address.
  ofm.match.dl_vlan = key.flow.dl_vlan;                                // Input VLAN OFP_VLAN_NONE;
  ofm.match.dl_type = key.flow.dl_type;                                // Ethernet frame type ETH_TYPE_IP;
  ofm.match.nw_proto = key.flow.nw_proto;                              // IP Protocol
  ofm.match.nw_src = key.flow.nw_src;                                  // IP source address
  ofm.match.nw_dst = key.flow.nw_dst;                                  // IP destination address
  ofm.match.tp_src = key.flow.tp_src;                                  // TCP/UDP source port
  ofm.match.tp_dst = key.flow.tp_dst;                                  // TCP/UDP destination port
  ofm.match.mpls_label1 = key.flow.mpls_label1;                        // Top of label stack
  ofm.match.mpls_label2 = key.flow.mpls_label1;                        // Second label (if available)

  // Build a sw_flow from the ofp_flow_mod
  sw_flow *flow = flow_alloc (actions_len);
  NS_TEST_ASSERT_MSG_NE (flow, 0, "Cannot allocate memory for the flow.");

  flow_extract_match (&flow->key, &ofm.match);

  // Fill out flow.
  flow->priority = flow->key.wildcards ? ntohs (ofm.priority) : -1;
  flow->idle_timeout = ntohs (ofm.idle_timeout);
  flow->hard_timeout = ntohs (ofm.hard_timeout);
  flow->used = flow->created = time_now ();
  flow->sf_acts->actions_len = actions_len;
  flow->byte_count = 0;
  flow->packet_count = 0;
  memcpy (flow->sf_acts->actions, ofm.actions, actions_len);

  // Insert the flow into the Flow Table
  NS_TEST_ASSERT_MSG_EQ (chain_insert (m_chain, flow), 0, "Flow table failed to insert Flow.");

  // Use key to match the flow to verify we created it correctly.
  NS_TEST_ASSERT_MSG_NE (chain_lookup (m_chain, &key), 0, "Key provided doesn't match to the flow that was created from it.");

  // Modify key to make sure the flow doesn't match it.
  dl_dst.CopyTo (key.flow.dl_src);
  dl_src.CopyTo (key.flow.dl_dst);
  key.flow.nw_src = htonl (nw_dst.Get ());
  key.flow.nw_dst = htonl (nw_src.Get ());
  key.flow.tp_src = htonl (tp_dst);
  key.flow.tp_dst = htonl (tp_src);

  NS_TEST_ASSERT_MSG_EQ (chain_lookup (m_chain, &key), 0, "Key provided shouldn't match the flow but it does.");

  // Modify key back to matching the flow so we can test flow modification.
  dl_dst.CopyTo (key.flow.dl_dst);
  dl_src.CopyTo (key.flow.dl_src);
  key.flow.nw_src = htonl (nw_src.Get ());
  key.flow.nw_dst = htonl (nw_dst.Get ());
  key.flow.tp_src = htonl (tp_src);
  key.flow.tp_dst = htonl (tp_dst);

  // Testing Flow Modification; chain_modify should return 1, for 1 flow modified.
  // Create output-to-port action
  ofp_action_output acts[1];
  acts[0].type = htons (OFPAT_OUTPUT);
  acts[0].len = htons (sizeof (ofp_action_output));
  acts[0].port = output_port;

  uint16_t priority = key.wildcards ? ntohs (ofm.priority) : -1;
  NS_TEST_ASSERT_MSG_EQ (chain_modify (m_chain, &key, priority, false, (const ofp_action_header*)acts, sizeof (acts)), 1, "Flow table failed to modify Flow.");

  // Testing Flow Deletion; chain_delete should return 1, for 1 flow deleted.
  // Note: By providing chain_delete with output_port, the flow must have an action that outputs on that port in order to delete the flow.
  // This is how we verify that our action was truly added via the flow modification.
  NS_TEST_ASSERT_MSG_EQ (chain_delete (m_chain, &key, output_port, 0, 0), 1, "Flow table failed to delete Flow.");
  NS_TEST_ASSERT_MSG_EQ (chain_lookup (m_chain, &key), 0, "Key provided shouldn't match the flow but it does.");
}

/**
 * \ingroup openflow-tests
 *
 * \brief OpenFlow TestSuite
 */
class SwitchTestSuite : public TestSuite
{
public:
  SwitchTestSuite ();
};

SwitchTestSuite::SwitchTestSuite () : TestSuite ("openflow", UNIT)
{
  AddTestCase (new SwitchFlowTableTestCase, TestCase::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite
static SwitchTestSuite switchTestSuite;

