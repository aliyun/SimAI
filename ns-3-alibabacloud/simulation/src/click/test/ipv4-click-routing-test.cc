/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Lalith Suresh
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Lalith Suresh <suresh.lalith@gmail.com>
 */

#ifdef NS3_CLICK

#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/ipv4-l3-protocol.h"
#include "ns3/simple-net-device.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/click-internet-stack-helper.h"

#include <click/simclick.h>

using namespace ns3;

static void
AddClickInternetStack (Ptr<Node> node)
{
  ClickInternetStackHelper internet;
  internet.SetClickFile (node, "src/click/test/nsclick-test-lan-single-interface.click");
  internet.Install (node);
}

static void
AddNetworkDevice (Ptr<Node> node, Mac48Address macaddr, Ipv4Address ipv4addr, Ipv4Mask ipv4mask)
{
  Ptr<SimpleNetDevice> rxDev1;

  rxDev1 = CreateObject<SimpleNetDevice> ();
  rxDev1->SetAddress (Mac48Address (macaddr));
  node->AddDevice (rxDev1);

  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  uint32_t netdev_idx = ipv4->AddInterface (rxDev1);
  Ipv4InterfaceAddress ipv4Addr = Ipv4InterfaceAddress (ipv4addr, ipv4mask);
  ipv4->AddAddress (netdev_idx, ipv4Addr);
  ipv4->SetUp (netdev_idx);
}

class ClickIfidFromNameTest : public TestCase
{
public:
  ClickIfidFromNameTest ();
  virtual void DoRun ();
};

ClickIfidFromNameTest::ClickIfidFromNameTest ()
  : TestCase ("Test SIMCLICK_IFID_FROM_NAME")
{
}

void
ClickIfidFromNameTest::DoRun ()
{
  Ptr<Node> node = CreateObject<Node> ();
  AddClickInternetStack (node);
  AddNetworkDevice (node, Mac48Address ("00:00:00:00:00:01"), Ipv4Address ("10.1.1.1"), Ipv4Mask ("255.255.255.0"));
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  Ptr<Ipv4ClickRouting> click = DynamicCast<Ipv4ClickRouting> (ipv4->GetRoutingProtocol ());
  click->DoInitialize ();

  int ret;

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IFID_FROM_NAME, "tap0");
  NS_TEST_EXPECT_MSG_EQ (ret, 0, "tap0 is interface 0");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IFID_FROM_NAME, "tun0");
  NS_TEST_EXPECT_MSG_EQ (ret, 0, "tun0 is interface 0");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IFID_FROM_NAME, "eth0");
  NS_TEST_EXPECT_MSG_EQ (ret, 1, "Eth0 is interface 1");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IFID_FROM_NAME, "tap1");
  NS_TEST_EXPECT_MSG_EQ (ret, 0, "tap1 is interface 0");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IFID_FROM_NAME, "tun1");
  NS_TEST_EXPECT_MSG_EQ (ret, 0, "tun1 is interface 0");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IFID_FROM_NAME, "eth1");
  NS_TEST_EXPECT_MSG_EQ (ret, -1, "No eth1 on node");
}

class ClickIpMacAddressFromNameTest : public TestCase
{
public:
  ClickIpMacAddressFromNameTest ();
  virtual void DoRun ();
};

ClickIpMacAddressFromNameTest::ClickIpMacAddressFromNameTest ()
  : TestCase ("Test SIMCLICK_IPADDR_FROM_NAME")
{
}

void
ClickIpMacAddressFromNameTest::DoRun ()
{
  Ptr<Node> node = CreateObject<Node> ();
  AddClickInternetStack (node);
  AddNetworkDevice (node, Mac48Address ("00:00:00:00:00:01"), Ipv4Address ("10.1.1.1"), Ipv4Mask ("255.255.255.0"));
  AddNetworkDevice (node, Mac48Address ("00:00:00:00:00:02"), Ipv4Address ("10.1.1.2"), Ipv4Mask ("255.255.255.0"));
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  Ptr<Ipv4ClickRouting> click = DynamicCast<Ipv4ClickRouting> (ipv4->GetRoutingProtocol ());
  click->DoInitialize ();

  char *buf = NULL;
  buf = new char [255];

  simclick_sim_command (click->m_simNode, SIMCLICK_IPADDR_FROM_NAME, "eth0", buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "10.1.1.1"), 0, "eth0 has IP 10.1.1.1");

  simclick_sim_command (click->m_simNode, SIMCLICK_MACADDR_FROM_NAME, "eth0", buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "00:00:00:00:00:01"), 0, "eth0 has Mac Address 00:00:00:00:00:01");

  simclick_sim_command (click->m_simNode, SIMCLICK_IPADDR_FROM_NAME, "eth1", buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "10.1.1.2"), 0, "eth1 has IP 10.1.1.2");

  simclick_sim_command (click->m_simNode, SIMCLICK_MACADDR_FROM_NAME, "eth1", buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "00:00:00:00:00:02"), 0, "eth0 has Mac Address 00:00:00:00:00:02");

  // Not sure how to test the below case, because the Ipv4ClickRouting code is to ASSERT for such inputs
  // simclick_sim_command (click->m_simNode, SIMCLICK_IPADDR_FROM_NAME, "eth2", buf, 255);
  // NS_TEST_EXPECT_MSG_EQ (buf, NULL, "No eth2");

  simclick_sim_command (click->m_simNode, SIMCLICK_IPADDR_FROM_NAME, "tap0", buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "127.0.0.1"), 0, "tun0 has IP 127.0.0.1");

  simclick_sim_command (click->m_simNode, SIMCLICK_MACADDR_FROM_NAME, "tap0", buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "00:00:00:00:00:00"), 0, "tun0 has IP 127.0.0.1");

  delete [] buf;
}

class ClickTrivialTest : public TestCase
{
public:
  ClickTrivialTest ();
  virtual void DoRun ();
};

ClickTrivialTest::ClickTrivialTest ()
  : TestCase ("Test SIMCLICK_GET_NODE_NAME and SIMCLICK_IF_READY")
{
}

void
ClickTrivialTest::DoRun ()
{
  Ptr<Node> node = CreateObject<Node> ();
  AddClickInternetStack (node);
  AddNetworkDevice (node, Mac48Address ("00:00:00:00:00:01"), Ipv4Address ("10.1.1.1"), Ipv4Mask ("255.255.255.0"));
  Ptr<Ipv4> ipv4 = node->GetObject<Ipv4> ();
  Ptr<Ipv4ClickRouting> click = DynamicCast<Ipv4ClickRouting> (ipv4->GetRoutingProtocol ());
  click->SetNodeName ("myNode");
  click->DoInitialize ();

  int ret = 0;
  char *buf = NULL;
  buf = new char [255];

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_GET_NODE_NAME, buf, 255);
  NS_TEST_EXPECT_MSG_EQ (strcmp (buf, "myNode"), 0, "Node name is Node");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IF_READY, 0);
  NS_TEST_EXPECT_MSG_EQ (ret, 1, "tap0 is ready");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IF_READY, 1);
  NS_TEST_EXPECT_MSG_EQ (ret, 1, "eth0 is ready");

  ret = simclick_sim_command (click->m_simNode, SIMCLICK_IF_READY, 2);
  NS_TEST_EXPECT_MSG_EQ (ret, 0, "eth1 does not exist, so return 0");

  delete [] buf;
}

class ClickIfidFromNameTestSuite : public TestSuite
{
public:
  ClickIfidFromNameTestSuite () : TestSuite ("routing-click", UNIT)
  {
    AddTestCase (new ClickTrivialTest, TestCase::QUICK);
    AddTestCase (new ClickIfidFromNameTest, TestCase::QUICK);
    AddTestCase (new ClickIpMacAddressFromNameTest, TestCase::QUICK);
  }
} g_ipv4ClickRoutingTestSuite;

#endif // NS3_CLICK
