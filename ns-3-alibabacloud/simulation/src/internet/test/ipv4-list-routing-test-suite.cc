/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#include "ns3/test.h"
#include "ns3/ipv4-list-routing.h"
#include "ns3/ipv4-routing-protocol.h"

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 dummy routing class (A)
 */
class Ipv4ARouting : public Ipv4RoutingProtocol {
public:
  Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p, const Ipv4Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)  { return 0; }
  bool RouteInput  (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
                    UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                    LocalDeliverCallback lcb, ErrorCallback ecb) { return false; }
  void NotifyInterfaceUp (uint32_t interface) {}
  void NotifyInterfaceDown (uint32_t interface) {}
  void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address) {}
  void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address) {}
  void SetIpv4 (Ptr<Ipv4> ipv4) {}
  void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const {}
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 dummy routing class (B)
 */
class Ipv4BRouting : public Ipv4RoutingProtocol {
public:
  Ptr<Ipv4Route> RouteOutput (Ptr<Packet> p, const Ipv4Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)  { return 0; }
  bool RouteInput  (Ptr<const Packet> p, const Ipv4Header &header, Ptr<const NetDevice> idev,
                    UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                    LocalDeliverCallback lcb, ErrorCallback ecb) { return false; }
  void NotifyInterfaceUp (uint32_t interface) {}
  void NotifyInterfaceDown (uint32_t interface) {}
  void NotifyAddAddress (uint32_t interface, Ipv4InterfaceAddress address) {}
  void NotifyRemoveAddress (uint32_t interface, Ipv4InterfaceAddress address) {}
  void SetIpv4 (Ptr<Ipv4> ipv4) {}
  void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const {}
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 ListRouting negative test.
 */
class Ipv4ListRoutingNegativeTestCase : public TestCase
{
public:
  Ipv4ListRoutingNegativeTestCase();
  virtual void DoRun (void);
};

Ipv4ListRoutingNegativeTestCase::Ipv4ListRoutingNegativeTestCase()
  : TestCase ("Check negative priorities")
{
}
void
Ipv4ListRoutingNegativeTestCase::DoRun (void)
{
  Ptr<Ipv4ListRouting> lr = CreateObject<Ipv4ListRouting> ();
  Ptr<Ipv4RoutingProtocol> aRouting = CreateObject<Ipv4ARouting> ();
  Ptr<Ipv4RoutingProtocol> bRouting = CreateObject<Ipv4BRouting> ();
  // The Ipv4BRouting should be added with higher priority (larger integer value)
  lr->AddRoutingProtocol (aRouting, -10);
  lr->AddRoutingProtocol (bRouting, -5);
  int16_t first = 3;
  uint32_t num = lr->GetNRoutingProtocols ();
  NS_TEST_ASSERT_MSG_EQ (num, 2, "100");
  Ptr<Ipv4RoutingProtocol> firstRp = lr->GetRoutingProtocol (0, first);
  NS_TEST_ASSERT_MSG_EQ (-5, first, "101");
  NS_TEST_ASSERT_MSG_EQ (firstRp, bRouting, "102");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 ListRouting positive test.
 */
class Ipv4ListRoutingPositiveTestCase : public TestCase
{
public:
  Ipv4ListRoutingPositiveTestCase();
  virtual void DoRun (void);
};

Ipv4ListRoutingPositiveTestCase::Ipv4ListRoutingPositiveTestCase()
  : TestCase ("Check positive priorities")
{
}
void
Ipv4ListRoutingPositiveTestCase::DoRun (void)
{
  Ptr<Ipv4ListRouting> lr = CreateObject<Ipv4ListRouting> ();
  Ptr<Ipv4RoutingProtocol> aRouting = CreateObject<Ipv4ARouting> ();
  Ptr<Ipv4RoutingProtocol> bRouting = CreateObject<Ipv4BRouting> ();
  // The Ipv4ARouting should be added with higher priority (larger integer
  // value) and will be fetched first below
  lr->AddRoutingProtocol (aRouting, 10);
  lr->AddRoutingProtocol (bRouting, 5);
  int16_t first = 3;
  int16_t second = 3;
  uint32_t num = lr->GetNRoutingProtocols ();
  NS_TEST_ASSERT_MSG_EQ (num, 2, "200");
  Ptr<Ipv4RoutingProtocol> firstRp = lr->GetRoutingProtocol (0, first);
  NS_TEST_ASSERT_MSG_EQ (10, first, "201");
  NS_TEST_ASSERT_MSG_EQ (firstRp, aRouting, "202");
  Ptr<Ipv4RoutingProtocol> secondRp = lr->GetRoutingProtocol (1, second);
  NS_TEST_ASSERT_MSG_EQ (5, second, "203");
  NS_TEST_ASSERT_MSG_EQ (secondRp, bRouting, "204");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv4 ListRouting TestSuite
 */
class Ipv4ListRoutingTestSuite : public TestSuite
{
public:
  Ipv4ListRoutingTestSuite()
    : TestSuite ("ipv4-list-routing", UNIT)
  {
    AddTestCase (new Ipv4ListRoutingPositiveTestCase (), TestCase::QUICK);
    AddTestCase (new Ipv4ListRoutingNegativeTestCase (), TestCase::QUICK);
  }
};

static Ipv4ListRoutingTestSuite g_ipv4ListRoutingTestSuite; //!< Static variable for test initialization
