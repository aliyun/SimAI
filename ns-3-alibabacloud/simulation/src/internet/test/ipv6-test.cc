/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 Strasbourg University
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
 * Authors: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 *         Faker Moatamri <faker.moatamri@sophia.inria.fr>
 */
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/boolean.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/node.h"
#include "ns3/simple-net-device.h"

#include "ns3/ipv6-interface.h"
#include "ns3/ipv6-l3-protocol.h"
#include "ns3/icmpv6-l4-protocol.h"

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 Test
 */
class Ipv6L3ProtocolTestCase : public TestCase
{
public:
  Ipv6L3ProtocolTestCase ();

  virtual ~Ipv6L3ProtocolTestCase ();
  virtual void DoRun ();
};

Ipv6L3ProtocolTestCase::Ipv6L3ProtocolTestCase () :
  TestCase ("Verify the IPv6 layer 3 protocol")
{
}

Ipv6L3ProtocolTestCase::~Ipv6L3ProtocolTestCase ()
{
}

void
Ipv6L3ProtocolTestCase::DoRun ()
{
  Ptr<Node> node = CreateObject<Node> ();
  Ptr<Ipv6L3Protocol> ipv6 = CreateObject<Ipv6L3Protocol> ();
  Ptr<Icmpv6L4Protocol> icmpv6 = CreateObject<Icmpv6L4Protocol> ();
  Ptr<Ipv6Interface> interface = CreateObject<Ipv6Interface> ();
  Ptr<Ipv6Interface> interface2 = CreateObject<Ipv6Interface> ();
  Ptr<SimpleNetDevice> device = CreateObject<SimpleNetDevice> ();
  Ptr<SimpleNetDevice> device2 = CreateObject<SimpleNetDevice> ();
  uint32_t index = 0;

  /* init */
  icmpv6->SetAttribute ("DAD", BooleanValue (false));
  node->AggregateObject (ipv6);
  node->AggregateObject (icmpv6);
  ipv6->Insert (icmpv6);

  /* first real interface (loopback is also installed) */
  node->AddDevice (device);
  interface->SetDevice (device);
  interface->SetNode (node);
  index = ipv6->AddIpv6Interface (interface);
  NS_TEST_ASSERT_MSG_EQ (index, 1,"The index is not 1??");

  /* second interface */
  node->AddDevice (device2);
  interface2->SetDevice (device2);
  interface2->SetNode (node);
  index = ipv6->AddIpv6Interface (interface2);
  NS_TEST_ASSERT_MSG_EQ (index, 2, "The index is not 2??");

  interface->SetUp ();
  interface2->SetUp ();

  Ipv6InterfaceAddress ifaceAddr = interface->GetLinkLocalAddress ();
  NS_TEST_ASSERT_MSG_EQ (ifaceAddr.GetAddress ().IsLinkLocal (), true,
                         "Should be link local??");

  NS_TEST_ASSERT_MSG_EQ (interface->GetNAddresses (), 1,
                         "interface has always a link-local address"); /* interface has always a link-local address */

  Ipv6InterfaceAddress ifaceAddr1 = Ipv6InterfaceAddress (
      "2001:1234:5678:9000::1", Ipv6Prefix (64));
  interface->AddAddress (ifaceAddr1);
  Ipv6InterfaceAddress ifaceAddr2 = Ipv6InterfaceAddress (
      "2001:ffff:5678:9000::1", Ipv6Prefix (64));
  interface->AddAddress (ifaceAddr2);

  Ipv6InterfaceAddress ifaceAddr3 = Ipv6InterfaceAddress (
      "2001:ffff:5678:9001::2", Ipv6Prefix (64));
  interface2->AddAddress (ifaceAddr3);

  uint32_t num = interface->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 3, "Number of addresses should be 3??"); /* 2 global addresses + link-local ones */

  num = interface2->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 2, "1 global addresses + link-local ones"); /* 1 global addresses + link-local ones */

  interface->RemoveAddress (2);
  num = interface->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 2, "Number of addresses should be 2??");

  Ipv6InterfaceAddress output = interface->GetAddress (1);
  NS_TEST_ASSERT_MSG_EQ (ifaceAddr1, output,
                         "Should be the interface address 1?");

  index = ipv6->GetInterfaceForPrefix ("2001:1234:5678:9000::0",
                                       Ipv6Prefix (64));
  NS_TEST_ASSERT_MSG_EQ (index, 1, "We should get one address??"); /* link-local address is always index 0 */

  index = ipv6->GetInterfaceForAddress ("2001:ffff:5678:9001::2");
  NS_TEST_ASSERT_MSG_EQ (index, 2, "Number of addresses should be 2??");

  index = ipv6->GetInterfaceForAddress ("2001:ffff:5678:9000::1"); /* address we just remove */
  NS_TEST_ASSERT_MSG_EQ (index, (uint32_t) -1, "Address should not be found??");

  /* Test Ipv6Interface()::RemoveAddress(address) */
  output = interface->RemoveAddress (Ipv6Address ("2001:1234:5678:9000::1"));
  NS_TEST_ASSERT_MSG_EQ (ifaceAddr1, output, "Wrong Interface Address Removed??");
  num = interface->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 1, "Number of addresses should be 1??");
  
  /* Remove a non-existent Address */
  output = interface->RemoveAddress (Ipv6Address ("2001:1234:5678:9000::1"));
  NS_TEST_ASSERT_MSG_EQ (Ipv6InterfaceAddress (), output, 
                         "Removed non-existent address??");
  num = interface->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 1, "Number of addresses should be 1??");

  /* Remove a loopback Address */
  output = interface->RemoveAddress (Ipv6Address::GetLoopback ());
  NS_TEST_ASSERT_MSG_EQ (Ipv6InterfaceAddress (), output, 
                         "Able to remove loopback address??");
  num = interface->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 1, "Number of addresses should be 1??");

  /* Test Ipv6Address::RemoveAddress(index, address) */
  index = ipv6->GetInterfaceForAddress ("2001:ffff:5678:9001::2");
  bool result = ipv6->RemoveAddress (index, Ipv6Address 
                                     ("2001:ffff:5678:9001::2"));
  NS_TEST_ASSERT_MSG_EQ (result, true, "Unable to remove Address??");
  num = interface2->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 1, "Number of addresses should be 1??");

  /* Remove a non-existent Address */
  result = ipv6->RemoveAddress (index, Ipv6Address  
                                ("2001:ffff:5678:9001::2"));
  NS_TEST_ASSERT_MSG_EQ (result, false, "Removed Non-existent address??");
  num = interface2->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 1, "Number of addresses should be 1??");
 
  /* Remove a loopback Address */
  result = ipv6->RemoveAddress (index, Ipv6Address::GetLoopback ());
  NS_TEST_ASSERT_MSG_EQ (result, false, "Able to remove loopback address??");
  num = interface2->GetNAddresses ();
  NS_TEST_ASSERT_MSG_EQ (num, 1, "Number of addresses should be 1??");

  Simulator::Destroy ();
} //end DoRun


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 TestSuite
 */
class IPv6L3ProtocolTestSuite : public TestSuite
{
public:
  IPv6L3ProtocolTestSuite () :
    TestSuite ("ipv6-protocol", UNIT)
  {
    AddTestCase (new Ipv6L3ProtocolTestCase (), TestCase::QUICK);
  }
};

static IPv6L3ProtocolTestSuite g_ipv6protocolTestSuite; //!< Static variable for test initialization
