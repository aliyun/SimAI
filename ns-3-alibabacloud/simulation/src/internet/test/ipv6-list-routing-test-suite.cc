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
#include "ns3/ipv6-list-routing.h"
#include "ns3/ipv6-route.h"
#include "ns3/ipv6-routing-protocol.h"

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 dummy routing class (A)
 */
class Ipv6ARouting : public Ipv6RoutingProtocol {
public:
  Ptr<Ipv6Route> RouteOutput (Ptr<Packet> p, const Ipv6Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)  { return 0; }
  bool RouteInput  (Ptr<const Packet> p, const Ipv6Header &header, Ptr<const NetDevice> idev,
                    UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                    LocalDeliverCallback lcb, ErrorCallback ecb) { return false; }
  void NotifyInterfaceUp (uint32_t interface) {}
  void NotifyInterfaceDown (uint32_t interface) {}
  void NotifyAddAddress (uint32_t interface, Ipv6InterfaceAddress address) {}
  void NotifyRemoveAddress (uint32_t interface, Ipv6InterfaceAddress address) {}
  void NotifyAddRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address::
                         GetZero ()) {}
  void NotifyRemoveRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse) {}
  void SetIpv6 (Ptr<Ipv6> ipv6) {}
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const {};
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 dummy routing class (B)
 */
class Ipv6BRouting : public Ipv6RoutingProtocol {
public:
  Ptr<Ipv6Route> RouteOutput (Ptr<Packet> p, const Ipv6Header &header, Ptr<NetDevice> oif, Socket::SocketErrno &sockerr)  { return 0; }
  bool RouteInput  (Ptr<const Packet> p, const Ipv6Header &header, Ptr<const NetDevice> idev,
                    UnicastForwardCallback ucb, MulticastForwardCallback mcb,
                    LocalDeliverCallback lcb, ErrorCallback ecb) { return false; }
  void NotifyInterfaceUp (uint32_t interface) {}
  void NotifyInterfaceDown (uint32_t interface) {}
  void NotifyAddAddress (uint32_t interface, Ipv6InterfaceAddress address) {}
  void NotifyRemoveAddress (uint32_t interface, Ipv6InterfaceAddress address) {}
  void NotifyAddRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse = Ipv6Address::
                         GetZero ()) {}
  void NotifyRemoveRoute (Ipv6Address dst, Ipv6Prefix mask, Ipv6Address nextHop, uint32_t interface, Ipv6Address prefixToUse) {}
  void SetIpv6 (Ptr<Ipv6> ipv6) {}
  virtual void PrintRoutingTable (Ptr<OutputStreamWrapper> stream, Time::Unit unit) const {};
};

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 ListRouting negative test.
 */
class Ipv6ListRoutingNegativeTestCase : public TestCase
{
public:
  Ipv6ListRoutingNegativeTestCase();
  virtual void DoRun (void);
};

Ipv6ListRoutingNegativeTestCase::Ipv6ListRoutingNegativeTestCase()
  : TestCase ("Check negative priorities")
{
}
void
Ipv6ListRoutingNegativeTestCase::DoRun (void)
{
  Ptr<Ipv6ListRouting> lr = CreateObject<Ipv6ListRouting> ();
  Ptr<Ipv6RoutingProtocol> aRouting = CreateObject<Ipv6ARouting> ();
  Ptr<Ipv6RoutingProtocol> bRouting = CreateObject<Ipv6BRouting> ();
  // The Ipv6BRouting should be added with higher priority (larger integer value)
  lr->AddRoutingProtocol (aRouting, -10);
  lr->AddRoutingProtocol (bRouting, -5);
  int16_t first = 3;
  uint32_t num = lr->GetNRoutingProtocols ();
  NS_TEST_ASSERT_MSG_EQ (num, 2, "100");
  Ptr<Ipv6RoutingProtocol> firstRp = lr->GetRoutingProtocol (0, first);
  NS_TEST_ASSERT_MSG_EQ (-5, first, "101");
  NS_TEST_ASSERT_MSG_EQ (firstRp, bRouting, "102");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 ListRouting positive test.
 */
class Ipv6ListRoutingPositiveTestCase : public TestCase
{
public:
  Ipv6ListRoutingPositiveTestCase();
  virtual void DoRun (void);
};

Ipv6ListRoutingPositiveTestCase::Ipv6ListRoutingPositiveTestCase()
  : TestCase ("Check positive priorities")
{
}
void
Ipv6ListRoutingPositiveTestCase::DoRun (void)
{
  Ptr<Ipv6ListRouting> lr = CreateObject<Ipv6ListRouting> ();
  Ptr<Ipv6RoutingProtocol> aRouting = CreateObject<Ipv6ARouting> ();
  Ptr<Ipv6RoutingProtocol> bRouting = CreateObject<Ipv6BRouting> ();
  // The Ipv6ARouting should be added with higher priority (larger integer
  // value) and will be fetched first below
  lr->AddRoutingProtocol (aRouting, 10);
  lr->AddRoutingProtocol (bRouting, 5);
  int16_t first = 3;
  int16_t second = 3;
  uint32_t num = lr->GetNRoutingProtocols ();
  NS_TEST_ASSERT_MSG_EQ (num, 2, "200");
  Ptr<Ipv6RoutingProtocol> firstRp = lr->GetRoutingProtocol (0, first);
  NS_TEST_ASSERT_MSG_EQ (10, first, "201");
  NS_TEST_ASSERT_MSG_EQ (firstRp, aRouting, "202");
  Ptr<Ipv6RoutingProtocol> secondRp = lr->GetRoutingProtocol (1, second);
  NS_TEST_ASSERT_MSG_EQ (5, second, "203");
  NS_TEST_ASSERT_MSG_EQ (secondRp, bRouting, "204");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief IPv6 ListRouting TestSuite
 */
class Ipv6ListRoutingTestSuite : public TestSuite
{
public:
  Ipv6ListRoutingTestSuite()
    : TestSuite ("ipv6-list-routing", UNIT)
  {
    AddTestCase (new Ipv6ListRoutingPositiveTestCase (), TestCase::QUICK);
    AddTestCase (new Ipv6ListRoutingNegativeTestCase (), TestCase::QUICK);
  }

};

static Ipv6ListRoutingTestSuite g_ipv6ListRoutingTestSuite; //!< Static variable for test initialization
