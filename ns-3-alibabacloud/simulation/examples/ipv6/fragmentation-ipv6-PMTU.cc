/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008-2009 Strasbourg University
 * Copyright (c) 2013 Universita' di Firenze
 * Copyright (c) 2022 Jadavpur University
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
 * Author: David Gross <gdavid.devel@gmail.com>
 *         Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 * Modified by Akash Mondal <a98mondal@gmail.com>
 */

// Network topology
// //
// //      Src  n0        r1     n1      r2        n2
// //           |         _      |       _         |
// //           =========|_|============|_|=========
// //      MTU   5000           2000          1500
// //
// // - Tracing of queues and packet receptions to file "fragmentation-ipv6-PMTU.tr"

#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/csma-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/ipv6-static-routing-helper.h"

#include "ns3/ipv6-routing-table-entry.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FragmentationIpv6PmtuExample");

int
main (int argc, char **argv)
{
  bool verbose = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("verbose", "turn on log components", verbose);
  cmd.Parse (argc, argv);

  if (verbose)
    {
      LogComponentEnable ("Ipv6L3Protocol", LOG_LEVEL_ALL);
      LogComponentEnable ("Icmpv6L4Protocol", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6StaticRouting", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6Interface", LOG_LEVEL_ALL);
      LogComponentEnable ("Ping6Application", LOG_LEVEL_ALL);
    }

  NS_LOG_INFO ("Create nodes.");
  Ptr<Node> n0 = CreateObject<Node> ();
  Ptr<Node> r1 = CreateObject<Node> ();
  Ptr<Node> n1 = CreateObject<Node> ();
  Ptr<Node> r2 = CreateObject<Node> ();
  Ptr<Node> n2 = CreateObject<Node> ();

  NodeContainer net1 (n0, r1);
  NodeContainer net2 (r1, n1, r2);
  NodeContainer net3 (r2, n2);
  NodeContainer all (n0, r1, n1, r2, n2);

  NS_LOG_INFO ("Create IPv6 Internet Stack");
  InternetStackHelper internetv6;
  internetv6.Install (all);

  NS_LOG_INFO ("Create channels.");
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("Mtu", UintegerValue (2000));
  NetDeviceContainer d2 = csma.Install (net2);  // CSMA Network with MTU 2000

  csma.SetDeviceAttribute ("Mtu", UintegerValue (5000));
  NetDeviceContainer d1 = csma.Install (net1);  // CSMA Network with MTU 5000

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", DataRateValue (5000000));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
  pointToPoint.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  NetDeviceContainer d3 = pointToPoint.Install (net3);  // P2P Network with MTU 1500

  NS_LOG_INFO ("Create networks and assign IPv6 Addresses.");
  Ipv6AddressHelper ipv6;

  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i1 = ipv6.Assign (d1);
  i1.SetForwarding (1, true);
  i1.SetDefaultRouteInAllNodes (1);

  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i2 = ipv6.Assign (d2);
  i2.SetForwarding (0, true);
  i2.SetDefaultRouteInAllNodes (0);
  i2.SetForwarding (1, true);
  i2.SetDefaultRouteInAllNodes (0);
  i2.SetForwarding (2, true);
  i2.SetDefaultRouteInAllNodes (2);

  ipv6.SetBase (Ipv6Address ("2001:3::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i3 = ipv6.Assign (d3);
  i3.SetForwarding (0, true);
  i3.SetDefaultRouteInAllNodes (0);

  Ipv6StaticRoutingHelper routingHelper;
  Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> (&std::cout);
  routingHelper.PrintRoutingTableAt (Seconds (0), r1, routingStream);

  /* Create a Ping6 application to send ICMPv6 echo request from r to n2 */
  uint32_t packetSize = 1600; // Packet should fragment as intermediate link MTU is 1500
  uint32_t maxPacketCount = 5;
  Time interPacketInterval = Seconds (1.0);
  Ping6Helper ping6;

  ping6.SetLocal (i2.GetAddress (1, 1));
  ping6.SetRemote (i3.GetAddress (1, 1));

  ping6.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  ping6.SetAttribute ("Interval", TimeValue (interPacketInterval));
  ping6.SetAttribute ("PacketSize", UintegerValue (packetSize));
  ApplicationContainer apps = ping6.Install (n1);
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (10.0));

  /* Create a Ping6 application to send ICMPv6 echo request from n0 to n2 */
  packetSize = 4000;
  ping6.SetAttribute ("PacketSize", UintegerValue (packetSize));

  ping6.SetLocal (i1.GetAddress (0, 1));
  ping6.SetRemote (i3.GetAddress (1, 1));
  apps = ping6.Install (n0);
  apps.Start (Seconds (11.0));
  apps.Stop (Seconds (20.0));

  AsciiTraceHelper ascii;
  csma.EnableAsciiAll (ascii.CreateFileStream ("fragmentation-ipv6-PMTU.tr"));
  csma.EnablePcapAll (std::string ("fragmentation-ipv6-PMTU"), true);
  pointToPoint.EnablePcapAll (std::string ("fragmentation-ipv6-PMTU"), true);

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}
