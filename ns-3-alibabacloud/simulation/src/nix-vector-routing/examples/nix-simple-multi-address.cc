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
 * Author: Ameya Deshpande <ameyanrd@outlook.com>
 *
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/nix-vector-helper.h"

/**
 * This program demonstrates a routing and
 * PrintRoutingPath example for multiple interface
 * addresses.
 *
 * Simple point-to-point links:
 * \verbatim
    n0 -- n1 -- n2 -- n3

    From t = 0s onwards,
    n0 (right) IP: 10.1.1.1, 10.2.1.1
    n1 (left)  IP: 10.1.1.2, 10.2.1.2
    n1 (right) IP: 10.1.2.1
    n2 (left)  IP: 10.1.2.2
    n2 (right) IP: 10.1.3.1
    n3 (left)  IP: 10.1.3.2

    From t = +5s onwards (new address subnet
    added for n2 and n3),
    n0 (right) IP: 10.1.1.1, 10.2.1.1
    n1 (left)  IP: 10.1.1.2, 10.2.1.2
    n1 (right) IP: 10.1.2.1
    n2 (left)  IP: 10.1.2.2
    n2 (right) IP: 10.1.3.1, 10.2.3.1
    n3 (left)  IP: 10.1.3.2, 10.2.3.2
   \endverbatim
 *
 * Cases considered:
 * 1. For UDP Echo Application:
 *  - At t = +2s, Path from n0 to n3 (10.1.3.2).
 * 2. For PrintRoutingPath, 10.1.1.3
 *  - At t = +3s, Path from n0 to n1 (10.2.1.2).
 *  - At t = +7s, Path from n0 to n3 (10.2.3.2).
 *
 * Logging output:
 * \verbatim
   At time +2s client sent 1024 bytes to 10.1.3.2 port 9
   At time +2.01106s server received 1024 bytes from 10.1.1.1 port 49153
   At time +2.01106s server sent 1024 bytes to 10.1.1.1 port 49153
   At time +2.02212s client received 1024 bytes from 10.1.3.2 port 9
  \endverbatim
 *
 * Output in nix-simple-multi-address.routes:
 * \verbatim
  Time: +3s, Nix Routing
  Route path from Node 0 to Node 1, Nix Vector: 0 (1 bits left)
  10.2.1.1                 (Node 0)  ---->   10.2.1.2                 (Node 1)

  Node: 0, Time: +4s, Local time: +4s, Nix Routing
  NixCache:
  Destination                   NixVector
  10.1.3.2                      011 (3 bits left)
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.3.2                      10.1.1.2                      10.1.1.1                        1

  Node: 1, Time: +4s, Local time: +4s, Nix Routing
  NixCache:
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.1.1                      10.1.1.1                      10.1.1.2                        1
  10.1.3.2                      10.1.2.2                      10.1.2.1                        2

  Node: 2, Time: +4s, Local time: +4s, Nix Routing
  NixCache:
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.1.1                      10.1.2.1                      10.1.2.2                        1
  10.1.3.2                      10.1.3.2                      10.1.3.1                        2

  Node: 3, Time: +4s, Local time: +4s, Nix Routing
  NixCache:
  Destination                   NixVector
  10.1.1.1                      000 (3 bits left)
  IpRouteCache:
  Destination                   Gateway                       Source                        OutputDevice
  10.1.1.1                      10.1.3.1                      10.1.3.2                        1

  Node: 0, Time: +6s, Local time: +6s, Nix Routing
  NixCache:
  IpRouteCache:

  Node: 1, Time: +6s, Local time: +6s, Nix Routing
  NixCache:
  IpRouteCache:

  Node: 2, Time: +6s, Local time: +6s, Nix Routing
  NixCache:
  IpRouteCache:

  Node: 3, Time: +6s, Local time: +6s, Nix Routing
  NixCache:
  IpRouteCache:

  Time: +7s, Nix Routing
  Route path from Node 0 to Node 3, Nix Vector: 011 (3 bits left)
  10.1.1.1                 (Node 0)  ---->   10.1.1.2                 (Node 1)
  10.1.2.1                 (Node 1)  ---->   10.1.2.2                 (Node 2)
  10.1.3.1                 (Node 2)  ---->   10.2.3.2                 (Node 3)
 \endverbatim
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NixSimpleMultiAddressExample");

int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);

  NodeContainer nodes12;
  nodes12.Create (2);

  NodeContainer nodes23;
  nodes23.Add (nodes12.Get (1));
  nodes23.Create (1);

  NodeContainer nodes34;
  nodes34.Add (nodes23.Get (1));
  nodes34.Create (1);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NodeContainer allNodes = NodeContainer (nodes12, nodes23.Get (1), nodes34.Get (1));

  // NixHelper to install nix-vector routing
  // on all nodes
  Ipv4NixVectorHelper nixRouting;
  InternetStackHelper stack;
  stack.SetRoutingHelper (nixRouting); // has effect on the next Install ()
  stack.Install (allNodes);

  NetDeviceContainer devices12;
  NetDeviceContainer devices23;
  NetDeviceContainer devices34;
  devices12 = pointToPoint.Install (nodes12);
  devices23 = pointToPoint.Install (nodes23);
  devices34 = pointToPoint.Install (nodes34);

  Ipv4AddressHelper address1;
  address1.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4AddressHelper address2;
  address2.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4AddressHelper address3;
  address3.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4AddressHelper address4;
  address4.SetBase ("10.2.1.0", "255.255.255.0");
  Ipv4AddressHelper address5;
  address5.SetBase ("10.2.3.0", "255.255.255.0");

  address1.Assign (devices12);
  address2.Assign (devices23);
  Ipv4InterfaceContainer interfaces34 = address3.Assign (devices34);
  Ipv4InterfaceContainer interfaces12 = address4.Assign (devices12);

  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (nodes34.Get (1));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  // Set the destination address as 10.1.3.2
  UdpEchoClientHelper echoClient (interfaces34.GetAddress (1), 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (nodes12.Get (0));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  // Trace routing paths for different source and destinations.
  Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("nix-simple-multi-address.routes", std::ios::out);
  // Check the path from n0 to n1 (10.1.3.2).
  nixRouting.PrintRoutingPathAt (Seconds (3), nodes12.Get (0), interfaces12.GetAddress (1, 1), routingStream);
  // Trace routing tables.
  nixRouting.PrintRoutingTableAllAt (Seconds (4), routingStream);
  // Assign address5 addresses to n2 and n3.
  // n2 gets 10.2.3.1 and n3 gets 10.2.3.2 on the same interface.
  Simulator::Schedule(Seconds (5), &Ipv4AddressHelper::Assign, &address5, devices34);

  // Trace routing tables.
  // Notice that NixCache and Ipv4RouteCache becomes empty for each node.
  // This happens because new addresses are added at t = +5s, due to which
  // existing caches get flushed.
  nixRouting.PrintRoutingTableAllAt (Seconds (6), routingStream);

  // Check the path from n0 to n3 (10.2.3.2).
  nixRouting.PrintRoutingPathAt (Seconds (7), nodes12.Get (0), Ipv4Address ("10.2.3.2"), routingStream);

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}