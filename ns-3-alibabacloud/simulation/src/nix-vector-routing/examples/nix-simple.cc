/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 NITK Surathkal
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
 * Modified By: Ameya Deshpande <ameyanrd@outlook.com>
 *              Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/ipv6-static-routing-helper.h"
#include "ns3/ipv6-list-routing-helper.h"
#include "ns3/nix-vector-helper.h"

/**
 * This program demonstrates two types of
 * trace output that are available:
 * 1) Print Routing Table for all the nodes.
 * 2) Print Routing Path, given source and destination.
 *
 * Simple point to point links:
 * \verbatim
       ________
      /        \
    n0 -- n1 -- n2 -- n3

    Using IPv4:
    n0 IP: 10.1.1.1, 10.1.4.1
    n1 IP: 10.1.1.2, 10.1.2.1
    n2 IP: 10.1.2.2, 10.1.3.1, 10.1.4.2
    n3 IP: 10.1.3.2

    Using IPv6: (parenthesis mentions the link for node
                 interface associated)
    n0 IP: 2001:1::200:ff:fe00:1 (Global Unicast on n0 -- n1)
           2001:4::200:ff:fe00:7 (Global Unicast on n0 -- n2)
           fe80::200:ff:fe00:1 (Link-local on n0 -- n1)
           fe80::200:ff:fe00:7 (Link-local on n0 -- n2)
    n1 IP: 2001:1::200:ff:fe00:2 (Global Unicast on n0 -- n1)
           2001:2::200:ff:fe00:3 (Global Unicast on n1 -- n2)
           fe80::200:ff:fe00:2 (Link-local on n0 -- n1)
           fe80::200:ff:fe00:3 (Link-local on n0 -- n2)
    n2 IP: 2001:2::200:ff:fe00:4 (Global Unicast on n1 -- n2)
           2001:3::200:ff:fe00:5 (Global Unicast on n2 -- n3)
           2001:4::200:ff:fe00:8 (Global Unicast on n0 -- n2)
           fe80::200:ff:fe00:4 (Link-local on n1 -- n2)
           fe80::200:ff:fe00:5 (Link-local on n2 -- n3)
           fe80::200:ff:fe00:8 (Link-local on n0 -- n2)
    n3 IP: 2001:3::200:ff:fe00:6 (Global Unicast on n2 -- n3)
           fe80::200:ff:fe00:6 (Link-local on n2 -- n3)
   \endverbatim
 *
 * Route Path for considered cases:
 * - Source (n0) and Destination (n3)
 *   It goes from n0 -> n2 -> n3
 * - Source (n1) and Destination (n3)
 *   It goes from n1 -> n2 -> n3
 * - Source (n2) and Destination (n0)
 *   It goes from n2 -> n0
 * - Source (n1) and Destination (n1)
 *   It goes from n1 -> n1
 * .
 * 
 * Expected IPv4 Routing Path output for above
 * cases (in the output stream):
 * \verbatim
  Time: +3s, Nix Routing
  Route path from Node 0 to Node 3, Nix Vector: 101 (3 bits left)
  10.1.4.1                 (Node 0)  ---->   10.1.4.2                 (Node 2)
  10.1.3.1                 (Node 2)  ---->   10.1.3.2                 (Node 3)

  Time: +5s, Nix Routing
  Route path from Node 1 to Node 3, Nix Vector: 101 (3 bits left)
  10.1.2.1                 (Node 1)  ---->   10.1.2.2                 (Node 2)
  10.1.3.1                 (Node 2)  ---->   10.1.3.2                 (Node 3)

  Time: +6s, Nix Routing
  Route path from Node 2 to Node 0, Nix Vector: 10 (2 bits left)
  10.1.4.2                 (Node 2)  ---->   10.1.1.1                 (Node 0)

  Time: +7s, Nix Routing
  Route path from Node 1 to Node 1, Nix Vector: 
  10.1.1.2                 (Node 1)  ---->   10.1.1.2                 (Node 1)

  Node: 0, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  Destination                   NixVector
  10.1.3.2                      101 (3 bits left)
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.3.2                      10.1.4.2                      10.1.4.1                        1

  Node: 1, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  IpRouteCache:

  Node: 2, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.3.2                      10.1.3.2                      10.1.3.1                        1
  10.1.4.1                      10.1.4.1                      10.1.4.2                        2

  Node: 3, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  Destination                   NixVector
  10.1.4.1                      010 (3 bits left)
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.4.1                      10.1.3.1                      10.1.3.2                        0
   \endverbatim
 *
 * Expected IPv6 Routing Path output for above
 * cases (in the output stream):
 * \verbatim
  Time: +3s, Nix Routing
  Route path from Node 0 to Node 3, Nix Vector: 101 (3 bits left)
  2001:4::200:ff:fe00:7    (Node 0)  ---->   fe80::200:ff:fe00:8      (Node 2)
  fe80::200:ff:fe00:5      (Node 2)  ---->   2001:3::200:ff:fe00:6    (Node 3)

  Time: +5s, Nix Routing
  Route path from Node 1 to Node 3, Nix Vector: 101 (3 bits left)
  2001:2::200:ff:fe00:3    (Node 1)  ---->   fe80::200:ff:fe00:4      (Node 2)
  fe80::200:ff:fe00:5      (Node 2)  ---->   2001:3::200:ff:fe00:6    (Node 3)

  Time: +6s, Nix Routing
  Route path from Node 2 to Node 0, Nix Vector: 10 (2 bits left)
  2001:4::200:ff:fe00:8    (Node 2)  ---->   2001:1::200:ff:fe00:1    (Node 0)

  Time: +7s, Nix Routing
  Route path from Node 1 to Node 1, Nix Vector: 
  2001:1::200:ff:fe00:2    (Node 1)  ---->   2001:1::200:ff:fe00:2    (Node 1)

  Node: 0, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  Destination                   NixVector
  2001:3::200:ff:fe00:6         101 (3 bits left)
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  2001:3::200:ff:fe00:6         fe80::200:ff:fe00:8           2001:4::200:ff:fe00:7           1

  Node: 1, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  IpRouteCache:

  Node: 2, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  2001:3::200:ff:fe00:6         fe80::200:ff:fe00:6           fe80::200:ff:fe00:5             1
  2001:4::200:ff:fe00:7         fe80::200:ff:fe00:7           fe80::200:ff:fe00:8             2

  Node: 3, Time: +8s, Local time: +8s, Nix Routing
  NixCache:
  Destination                   NixVector
  2001:4::200:ff:fe00:7         010 (3 bits left)
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  2001:4::200:ff:fe00:7         fe80::200:ff:fe00:5           2001:3::200:ff:fe00:6           0
  \endverbatim
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NixSimpleExample");

int
main (int argc, char *argv[])
{
  bool useIpv6 = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("useIPv6", "Use IPv6 instead of IPv4", useIpv6);
  cmd.Parse (argc, argv);

  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer nodes;
  nodes.Create (4);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer devices01;
  NetDeviceContainer devices12;
  NetDeviceContainer devices23;
  NetDeviceContainer devices02;
  devices01 = pointToPoint.Install (NodeContainer (nodes.Get (0), nodes.Get (1)));
  devices12 = pointToPoint.Install (NodeContainer (nodes.Get (1), nodes.Get (2)));
  devices23 = pointToPoint.Install (NodeContainer (nodes.Get (2), nodes.Get (3)));
  devices02 = pointToPoint.Install (NodeContainer (nodes.Get (0), nodes.Get (2)));

  Address udpServerAddress;

  if (!useIpv6)
    {
      // NixHelper to install nix-vector routing
      // on all nodes
      Ipv4NixVectorHelper nixRouting;
      InternetStackHelper stack;
      stack.SetRoutingHelper (nixRouting); // has effect on the next Install ()
      stack.Install (nodes);

      Ipv4AddressHelper address1;
      address1.SetBase ("10.1.1.0", "255.255.255.0");
      Ipv4AddressHelper address2;
      address2.SetBase ("10.1.2.0", "255.255.255.0");
      Ipv4AddressHelper address3;
      address3.SetBase ("10.1.3.0", "255.255.255.0");
      Ipv4AddressHelper address4;
      address4.SetBase ("10.1.4.0", "255.255.255.0");

      Ipv4InterfaceContainer interfaces01 = address1.Assign (devices01);
      Ipv4InterfaceContainer interfaces12 = address2.Assign (devices12);
      Ipv4InterfaceContainer interfaces23 = address3.Assign (devices23);
      Ipv4InterfaceContainer interfaces02 = address4.Assign (devices02);

      udpServerAddress = interfaces23.GetAddress (1);

      // Trace routing paths for different source and destinations.
      Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("nix-simple-ipv4.routes", std::ios::out);
      nixRouting.PrintRoutingPathAt (Seconds (3), nodes.Get (0), interfaces23.GetAddress (1), routingStream);
      nixRouting.PrintRoutingPathAt (Seconds (5), nodes.Get (1), interfaces23.GetAddress (1), routingStream);
      nixRouting.PrintRoutingPathAt (Seconds (6), nodes.Get (2), interfaces01.GetAddress (0), routingStream);
      nixRouting.PrintRoutingPathAt (Seconds (7), nodes.Get (1), interfaces01.GetAddress (1), routingStream);
      // Trace routing tables
      nixRouting.PrintRoutingTableAllAt (Seconds (8), routingStream);
    }
  else
    {
      // NixHelper to install nix-vector routing
      // on all nodes
      Ipv6NixVectorHelper nixRouting;
      InternetStackHelper stack;
      stack.SetRoutingHelper (nixRouting); // has effect on the next Install ()
      stack.Install (nodes);

      Ipv6AddressHelper address1;
      address1.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
      Ipv6AddressHelper address2;
      address2.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
      Ipv6AddressHelper address3;
      address3.SetBase (Ipv6Address ("2001:3::"), Ipv6Prefix (64));
      Ipv6AddressHelper address4;
      address4.SetBase (Ipv6Address ("2001:4::"), Ipv6Prefix (64));

      Ipv6InterfaceContainer interfaces01 = address1.Assign (devices01);
      Ipv6InterfaceContainer interfaces12 = address2.Assign (devices12);
      Ipv6InterfaceContainer interfaces23 = address3.Assign (devices23);
      Ipv6InterfaceContainer interfaces02 = address4.Assign (devices02);

      udpServerAddress = interfaces23.GetAddress (1, 1);

      // Trace routing paths for different source and destinations.
      Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("nix-simple-ipv6.routes", std::ios::out);
      nixRouting.PrintRoutingPathAt (Seconds (3), nodes.Get (0), interfaces23.GetAddress (1, 1), routingStream);
      nixRouting.PrintRoutingPathAt (Seconds (5), nodes.Get (1), interfaces23.GetAddress (1, 1), routingStream);
      nixRouting.PrintRoutingPathAt (Seconds (6), nodes.Get (2), interfaces01.GetAddress (0, 1), routingStream);
      nixRouting.PrintRoutingPathAt (Seconds (7), nodes.Get (1), interfaces01.GetAddress (1, 1), routingStream);
      // Trace routing tables
      nixRouting.PrintRoutingTableAllAt (Seconds (8), routingStream);
    }
  
  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (nodes.Get (3));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (udpServerAddress, 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (nodes.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
