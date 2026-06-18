/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 Universita' di Firenze, Italy
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
 * Authors: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 *
 */

#include "ns3/data-rate.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/ipv6-routing-helper.h"
#include "ns3/ipv6-route.h"
#include "ns3/radvd-helper.h"
#include "ns3/test.h"

using namespace ns3;

/**
 * \ingroup radvd
 * \defgroup radvd-test radvd tests
 */


/**
 * \ingroup radvd-test
 * \ingroup tests
 *
 * \brief radvd basic tests
 */
class RadvdTestCase : public TestCase
{
public:
  RadvdTestCase ();
  virtual ~RadvdTestCase ();

private:
  virtual void DoRun (void);

  /**
   * Checks the addresses on the selected NetDevices.
   * \param n0Dev node 0 device
   * \param r0Dev router device toward node 0
   * \param r1Dev router device toward node 1
   * \param n1Dev node 1 device
   */
  void CheckAddresses (Ptr<NetDevice> n0Dev, Ptr<NetDevice> r0Dev, Ptr<NetDevice> r1Dev, Ptr<NetDevice> n1Dev);
  /**
   * Checks the routing between the selected NetDevices.
   * \param n0Dev node 0 device
   * \param r0Dev router device toward node 0
   * \param r1Dev router device toward node 1
   * \param n1Dev node 1 device
   */
  void CheckRouting (Ptr<NetDevice> n0Dev, Ptr<NetDevice> r0Dev, Ptr<NetDevice> r1Dev, Ptr<NetDevice> n1Dev);

  std::vector<Ipv6Address> m_addresses; //!< Addresses on the nodes
  std::vector<Socket::SocketErrno> m_routingResults; //!< Routing call return values
  std::vector<Ptr<Ipv6Route> > m_routes; //!< Routing call results
};

RadvdTestCase::RadvdTestCase ()
  : TestCase ("Radvd test case ")
{
}

RadvdTestCase::~RadvdTestCase ()
{
}

void
RadvdTestCase::CheckAddresses (Ptr<NetDevice> n0Dev, Ptr<NetDevice> r0Dev, Ptr<NetDevice> r1Dev, Ptr<NetDevice> n1Dev)
{
  Ptr<Ipv6L3Protocol> ipv6;

  ipv6 = n0Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  m_addresses.push_back (ipv6->GetAddress (ipv6->GetInterfaceForDevice (n0Dev), 1).GetAddress ());

  ipv6 = r0Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  m_addresses.push_back (ipv6->GetAddress (ipv6->GetInterfaceForDevice (r0Dev), 1).GetAddress ());

  ipv6 = r1Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  m_addresses.push_back (ipv6->GetAddress (ipv6->GetInterfaceForDevice (r1Dev), 1).GetAddress ());

  ipv6 = n1Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  m_addresses.push_back (ipv6->GetAddress (ipv6->GetInterfaceForDevice (n1Dev), 1).GetAddress ());
}

void
RadvdTestCase::CheckRouting (Ptr<NetDevice> n0Dev, Ptr<NetDevice> r0Dev, Ptr<NetDevice> r1Dev, Ptr<NetDevice> n1Dev)
{
  Ptr<Ipv6L3Protocol> ipv6;
  Ptr<Packet> p = Create<Packet> ();
  Ipv6Header ipHdr;
  Socket::SocketErrno sockerr;
  Ptr<Ipv6Route> route;

  ipv6 = n0Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  ipHdr.SetSource (m_addresses[0]);
  ipHdr.SetDestination (m_addresses[1]);
  route = ipv6->GetRoutingProtocol ()->RouteOutput (p, ipHdr, n0Dev, sockerr);
  m_routingResults.push_back (sockerr);
  m_routes.push_back (route);

  ipv6 = r0Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  ipHdr.SetSource (m_addresses[1]);
  ipHdr.SetDestination (m_addresses[0]);
  route = ipv6->GetRoutingProtocol ()->RouteOutput (p, ipHdr, r0Dev, sockerr);
  m_routingResults.push_back (sockerr);
  m_routes.push_back (route);

  ipv6 = r1Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  ipHdr.SetSource (m_addresses[2]);
  ipHdr.SetDestination (m_addresses[3]);
  route = ipv6->GetRoutingProtocol ()->RouteOutput (p, ipHdr, r1Dev, sockerr);
  m_routingResults.push_back (sockerr);
  m_routes.push_back (route);

  ipv6 = n1Dev->GetNode ()->GetObject<Ipv6L3Protocol> ();
  ipHdr.SetSource (m_addresses[3]);
  ipHdr.SetDestination (m_addresses[2]);
  route = ipv6->GetRoutingProtocol ()->RouteOutput (p, ipHdr, n1Dev, sockerr);
  m_routingResults.push_back (sockerr);
  m_routes.push_back (route);
}

void
RadvdTestCase::DoRun (void)
{
   // Create nodes
  Ptr<Node> n0 = CreateObject<Node> ();
  Ptr<Node> r = CreateObject<Node> ();
  Ptr<Node> n1 = CreateObject<Node> ();

  NodeContainer net1 (n0, r);
  NodeContainer net2 (r, n1);
  NodeContainer all (n0, r, n1);

  // Create IPv6 Internet Stack
  InternetStackHelper internetv6;
  internetv6.Install (all);

  // Create channels
  SimpleNetDeviceHelper simpleNetDevice;
  simpleNetDevice.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  simpleNetDevice.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("5Mbps")));

  NetDeviceContainer d1 = simpleNetDevice.Install (net1); /* n0 - R */
  NetDeviceContainer d2 = simpleNetDevice.Install (net2); /* R - n1 */

  // Create networks and assign IPv6 Addresses
  Ipv6AddressHelper ipv6;

  /* first subnet */
  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  NetDeviceContainer tmp;
  tmp.Add (d1.Get (0)); /* n0 */
  Ipv6InterfaceContainer iic1 = ipv6.AssignWithoutAddress (tmp); /* n0 interface */

  NetDeviceContainer tmp2;
  tmp2.Add (d1.Get (1)); /* R */
  Ipv6InterfaceContainer iicr1 = ipv6.AssignWithoutOnLink (tmp2); /* R interface to the first subnet is just statically assigned */
  iicr1.SetForwarding (0, true);
  iic1.Add (iicr1);

  /* second subnet R - n1 */
  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  NetDeviceContainer tmp3;
  tmp3.Add (d2.Get (0)); /* R */
  Ipv6InterfaceContainer iicr2 = ipv6.Assign (tmp3); /* R interface */
  iicr2.SetForwarding (0, true);

  NetDeviceContainer tmp4;
  tmp4.Add (d2.Get (1)); /* n1 */
  Ipv6InterfaceContainer iic2 = ipv6.AssignWithoutAddress (tmp4);
  iic2.Add (iicr2);

  /* radvd configuration */
  RadvdHelper radvdHelper;

  /* R interface (n0 - R) */
  /* n0 will receive unsolicited (periodic) RA */
  radvdHelper.AddAnnouncedPrefix (iic1.GetInterfaceIndex (1), Ipv6Address("2001:1::0"), 64);
  Ptr<RadvdPrefix> prefix = *(radvdHelper.GetRadvdInterface (iic1.GetInterfaceIndex (1))->GetPrefixes ().begin ());
  prefix->SetOnLinkFlag (false);

  /* R interface (R - n1) */
  /* n1 will have to use RS, as RA are not sent automatically */
  radvdHelper.AddAnnouncedPrefix(iic2.GetInterfaceIndex (1), Ipv6Address("2001:2::0"), 64);
  radvdHelper.GetRadvdInterface (iic2.GetInterfaceIndex (1))->SetSendAdvert (false);

  ApplicationContainer radvdApps = radvdHelper.Install (r);
  radvdApps.Start (Seconds (1.0));
  radvdApps.Stop (Seconds (10.0));

  Simulator::Schedule (Seconds (2), &RadvdTestCase::CheckAddresses, this, d1.Get (0), d1.Get (1), d2.Get (0), d2.Get (1));
  Simulator::Schedule (Seconds (3), &RadvdTestCase::CheckRouting, this, d1.Get (0), d1.Get (1), d2.Get (0), d2.Get (1));


  Simulator::Stop (Seconds (10.0));

  Simulator::Run ();

  // Address assignment checks
  NS_TEST_ASSERT_MSG_EQ (m_addresses[0], Ipv6Address ("2001:1::200:ff:fe00:1"),
                         m_addresses[0] << " instead of " << "2001:1::200:ff:fe00:1");

  NS_TEST_ASSERT_MSG_EQ (m_addresses[1], Ipv6Address ("2001:1::200:ff:fe00:2"),
                         m_addresses[1] << " instead of " << "2001:1::200:ff:fe00:2");

  NS_TEST_ASSERT_MSG_EQ (m_addresses[2], Ipv6Address ("2001:2::200:ff:fe00:3"),
                         m_addresses[2] << " instead of " << "2001:2::200:ff:fe00:3");

  NS_TEST_ASSERT_MSG_EQ (m_addresses[3], Ipv6Address ("2001:2::200:ff:fe00:4"),
                         m_addresses[3] << " instead of " << "2001:2::200:ff:fe00:4");

  // Routes checks
  NS_TEST_ASSERT_MSG_EQ (m_routingResults[0], Socket::ERROR_NOTERROR,
                         (int) m_routingResults[0] << " instead of Socket::ERROR_NOTERROR");

  NS_TEST_ASSERT_MSG_EQ (m_routes[0]->GetGateway (), Ipv6Address ("fe80::200:ff:fe00:2"),
                         m_routes[0]->GetGateway () << " instead of " << "fe80::200:ff:fe00:2");

  NS_TEST_ASSERT_MSG_EQ (m_routingResults[1], Socket::ERROR_NOROUTETOHOST,
                         (int) m_routingResults[1] << " instead of Socket::ERROR_NOROUTETOHOST");

  NS_TEST_ASSERT_MSG_EQ (m_routingResults[2], Socket::ERROR_NOTERROR,
                         (int) m_routingResults[2] << " instead of Socket::ERROR_NOTERROR");

  NS_TEST_ASSERT_MSG_EQ (m_routes[2]->GetGateway (), Ipv6Address ("::"),
                         m_routes[2]->GetGateway () << " instead of " << "::");

  NS_TEST_ASSERT_MSG_EQ (m_routingResults[3], Socket::ERROR_NOTERROR,
                         (int) m_routingResults[3] << " instead of Socket::ERROR_NOTERROR");

  NS_TEST_ASSERT_MSG_EQ (m_routes[3]->GetGateway (), Ipv6Address ("::"),
                         m_routes[3]->GetGateway () << " instead of " << "::");


  Simulator::Destroy ();
}

/**
 * \ingroup radvd-test
 * \ingroup tests
 *
 * \brief radvd TestSuite
 */
class RadvdTestSuite : public TestSuite
{
public:
  RadvdTestSuite ();
};

RadvdTestSuite::RadvdTestSuite ()
  : TestSuite ("radvd", UNIT)
{
  AddTestCase (new RadvdTestCase, TestCase::QUICK);
}

static RadvdTestSuite radvdTestSuite; //!< Static variable for test initialization

