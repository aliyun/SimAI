/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
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
 * Author: David Gross <gdavid.devel@gmail.com>
 */

// Network topology
// //
// //
// //                               +------------+
// //  +------------+           |---|  Router 1  |---|
// //  |   Host 0   |--|        |   [------------]   |
// //  [------------]  |        |                    |
// //                  |  +------------+             |
// //                  +--|            |       +------------+
// //                     |  Router 0  |       |  Router 2  |
// //                  +--|            |       [------------]
// //                  |  [------------]             |
// //  +------------+  |        |                    |
// //  |   Host 1   |--|        |   +------------+   |
// //  [------------]           |---|  Router 3  |---|
// //                               [------------]
// //
// //
// // - Tracing of queues and packet receptions to file "loose-routing-ipv6.tr"

#include <fstream>
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/ipv6-header.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("LooseRoutingIpv6Example");

int main (int argc, char **argv)
{
  bool verbose = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("verbose", "turn on log components", verbose);
  cmd.Parse (argc, argv);

  if (verbose)
    {
      LogComponentEnable ("Ipv6ExtensionLooseRouting", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6Extension", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6L3Protocol", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6StaticRouting", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6Interface", LOG_LEVEL_ALL);
      LogComponentEnable ("Ipv6Interface", LOG_LEVEL_ALL);
      LogComponentEnable ("NdiscCache", LOG_LEVEL_ALL);
    }

  NS_LOG_INFO ("Create nodes.");
  Ptr<Node> h0 = CreateObject<Node> ();
  Ptr<Node> h1 = CreateObject<Node> ();
  Ptr<Node> r0 = CreateObject<Node> ();
  Ptr<Node> r1 = CreateObject<Node> ();
  Ptr<Node> r2 = CreateObject<Node> ();
  Ptr<Node> r3 = CreateObject<Node> ();

  NodeContainer net1 (h0, r0);
  NodeContainer net2 (h1, r0);
  NodeContainer net3 (r0, r1);
  NodeContainer net4 (r1, r2);
  NodeContainer net5 (r2, r3);
  NodeContainer net6 (r3, r0);
  NodeContainer all;
  all.Add (h0);
  all.Add (h1);
  all.Add (r0);
  all.Add (r1);
  all.Add (r2);
  all.Add (r3);

  NS_LOG_INFO ("Create IPv6 Internet Stack");
  InternetStackHelper internetv6;
  internetv6.Install (all);

  NS_LOG_INFO ("Create channels.");
  CsmaHelper csma;
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1500));
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  NetDeviceContainer d1 = csma.Install (net1);
  NetDeviceContainer d2 = csma.Install (net2);
  NetDeviceContainer d3 = csma.Install (net3);
  NetDeviceContainer d4 = csma.Install (net4);
  NetDeviceContainer d5 = csma.Install (net5);
  NetDeviceContainer d6 = csma.Install (net6);

  NS_LOG_INFO ("Create networks and assign IPv6 Addresses.");
  Ipv6AddressHelper ipv6;

  ipv6.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i1 = ipv6.Assign (d1);
  i1.SetForwarding (1, true);
  i1.SetDefaultRouteInAllNodes (1);

  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i2 = ipv6.Assign (d2);
  i2.SetForwarding (1, true);
  i2.SetDefaultRouteInAllNodes (1);

  ipv6.SetBase (Ipv6Address ("2001:3::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i3 = ipv6.Assign (d3);
  i3.SetForwarding (0, true);
  i3.SetDefaultRouteInAllNodes (0);
  i3.SetForwarding (1, true);
  i3.SetDefaultRouteInAllNodes (1);

  ipv6.SetBase (Ipv6Address ("2001:4::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i4 = ipv6.Assign (d4);
  i4.SetForwarding (0, true);
  i4.SetDefaultRouteInAllNodes (0);
  i4.SetForwarding (1, true);
  i4.SetDefaultRouteInAllNodes (1);

  ipv6.SetBase (Ipv6Address ("2001:5::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i5 = ipv6.Assign (d5);
  i5.SetForwarding (0, true);
  i5.SetDefaultRouteInAllNodes (0);
  i5.SetForwarding (1, true);
  i5.SetDefaultRouteInAllNodes (1);

  ipv6.SetBase (Ipv6Address ("2001:6::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i6 = ipv6.Assign (d6);
  i6.SetForwarding (0, true);
  i6.SetDefaultRouteInAllNodes (0);
  i6.SetForwarding (1, true);
  i6.SetDefaultRouteInAllNodes (1);

  NS_LOG_INFO ("Create Applications.");

  /**
   * ICMPv6 Echo from h0 to h1 port 7
   */
  uint32_t packetSize = 1024;
  uint32_t maxPacketCount = 1;
  Time interPacketInterval = Seconds (1.0);

  std::vector<Ipv6Address> routersAddress;
  routersAddress.push_back (i3.GetAddress (1, 1));
  routersAddress.push_back (i4.GetAddress (1, 1));
  routersAddress.push_back (i5.GetAddress (1, 1));
  routersAddress.push_back (i6.GetAddress (1, 1));
  routersAddress.push_back (i2.GetAddress (0, 1));

  Ping6Helper client;
  /* remote address is first routers in RH0 => source routing */
  client.SetRemote (i1.GetAddress (1, 1));
  client.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  client.SetAttribute ("Interval", TimeValue (interPacketInterval));
  client.SetAttribute ("PacketSize", UintegerValue (packetSize));
  client.SetRoutersAddress (routersAddress);
  ApplicationContainer apps = client.Install (h0);
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  AsciiTraceHelper ascii;
  csma.EnableAsciiAll (ascii.CreateFileStream ("loose-routing-ipv6.tr"));
  csma.EnablePcapAll ("loose-routing-ipv6", true);

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}

