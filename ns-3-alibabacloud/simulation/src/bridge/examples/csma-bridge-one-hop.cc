/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 */

// Network topology
//
//         bridge1         The node named bridge1 (node 5 in the nodelist)
//   ------------------        has three CMSA net devices that are bridged
//   CSMA   CSMA   CSMA        together using a BridgeNetDevice.
//     |      |      |
//     |      |      |     The bridge node talks over three CSMA channels
//     |      |      |
//   CSMA   CSMA   CSMA    to three other CSMA net devices
//   ----   ----   ----
//    n0     n1     n2     Node two acts as a router and talks to another
//                 ----        bridge that connects the remaining nodes.
//                 CSMA
//                   |
//    n3     n4      |
//   ----   ----     |
//   CSMA   CSMA     |
//     |      |      |
//     |      |      |
//     |      |      |
//   CSMA   CSMA   CSMA    The node named bridge2 (node 6 in the nodelist)
//   ------------------        has three CMSA net devices that are bridged
//        bridge2              together using a BridgeNetDevice.
//
// Or, more abstractly, recognizing that bridge 1 and bridge 2 are nodes 
// with three net devices:
//
//        n0     n1                (n0 = 10.1.1.2)
//        |      |                 (n1 = 10.1.1.3)  Note odd addressing
//       -----------               (n2 = 10.1.1.1)
//       | bridge1 | <- n5
//       -----------
//           |
//         router    <- n2
//           |
//       -----------
//       | bridge2 | <- n6
//       -----------               (n2 = 10.1.2.1)
//        |      |                 (n3 = 10.1.2.2)
//        n3     n4                (n4 = 10.1.2.3)
//
// So, this example shows two broadcast domains, each interconnected by a bridge
// with a router node (n2) interconnecting the layer-2 broadcast domains
// 
// It is meant to mirror somewhat the csma-bridge example but adds another
// bridged link separated by a router.
// 
// - CBR/UDP flows from n0 (10.1.1.2) to n1 (10.1.1.3) and from n3 (10.1.2.2) to n0 (10.1.1.3)
// - DropTail queues 
// - Global static routing
// - Tracing of queues and packet receptions to file "csma-bridge-one-hop.tr"

#include <iostream>
#include <fstream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/bridge-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"

/**
 * \file
 * \ingroup bridge
 * Bridge example connecting two broadcast domains.
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("CsmaBridgeOneHopExample");

int 
main (int argc, char *argv[])
{
  //
  // Users may find it convenient to turn on explicit debugging
  // for selected modules; the below lines suggest how to do this
  //
#if 0 
  LogComponentEnable ("CsmaBridgeOneHopExample", LOG_LEVEL_INFO);
#endif

  //
  // Allow the user to override any of the defaults and the above Bind() at
  // run-time, via command-line arguments
  //
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  //
  // Explicitly create the nodes required by the topology (shown above).
  //
  NS_LOG_INFO ("Create nodes.");

  Ptr<Node> n0 = CreateObject<Node> ();
  Ptr<Node> n1 = CreateObject<Node> ();
  Ptr<Node> n2 = CreateObject<Node> ();
  Ptr<Node> n3 = CreateObject<Node> ();
  Ptr<Node> n4 = CreateObject<Node> ();

  Ptr<Node> bridge1 = CreateObject<Node> ();
  Ptr<Node> bridge2 = CreateObject<Node> ();

  NS_LOG_INFO ("Build Topology");
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));

  // Create the csma links, from each terminal to the bridge
  // This will create six network devices; we'll keep track separately
  // of the devices on and off the bridge respectively, for later configuration 
  NetDeviceContainer topLanDevices;
  NetDeviceContainer topBridgeDevices;

  // It is easier to iterate the nodes in C++ if we put them into a container
  NodeContainer topLan (n2, n0, n1);

  for (int i = 0; i < 3; i++)
    {
      // install a csma channel between the ith toplan node and the bridge node
      NetDeviceContainer link = csma.Install (NodeContainer (topLan.Get (i), bridge1));
      topLanDevices.Add (link.Get (0));
      topBridgeDevices.Add (link.Get (1));
    }

  //
  // Now, Create the bridge netdevice, which will do the packet switching.  The
  // bridge lives on the node bridge1 and bridges together the topBridgeDevices
  // which are the three CSMA net devices on the node in the diagram above.
  //
  BridgeHelper bridge;
  bridge.Install (bridge1, topBridgeDevices);

  // Add internet stack to the router nodes
  NodeContainer routerNodes (n0, n1, n2, n3, n4);
  InternetStackHelper internet;
  internet.Install (routerNodes);

  // Repeat for bottom bridged LAN
  NetDeviceContainer bottomLanDevices;
  NetDeviceContainer bottomBridgeDevices;
  NodeContainer bottomLan (n2, n3, n4);
  for (int i = 0; i < 3; i++)
    {
      NetDeviceContainer link = csma.Install (NodeContainer (bottomLan.Get (i), bridge2));
      bottomLanDevices.Add (link.Get (0));
      bottomBridgeDevices.Add (link.Get (1));
    }
  bridge.Install (bridge2, bottomBridgeDevices);

  // We've got the "hardware" in place.  Now we need to add IP addresses.
  NS_LOG_INFO ("Assign IP Addresses.");
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  ipv4.Assign (topLanDevices);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  ipv4.Assign (bottomLanDevices);

  // 
  // Create router nodes, initialize routing database and set up the routing
  // tables in the nodes.  We excuse the bridge nodes from having to serve as
  // routers, since they don't even have internet stacks on them.
  //
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  //
  // Create an OnOff application to send UDP datagrams from node zero to node 1.
  //
  NS_LOG_INFO ("Create Applications.");
  uint16_t port = 9;   // Discard port (RFC 863)

  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                     Address (InetSocketAddress (Ipv4Address ("10.1.1.3"), port)));
  onoff.SetConstantRate (DataRate ("500kb/s"));

  ApplicationContainer app = onoff.Install (n0);
  // Start the application
  app.Start (Seconds (1.0));
  app.Stop (Seconds (10.0));

  // Create an optional packet sink to receive these packets
  PacketSinkHelper sink ("ns3::UdpSocketFactory",
                         Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
  ApplicationContainer sink1 = sink.Install (n1);
  sink1.Start (Seconds (1.0));
  sink1.Stop (Seconds (10.0));

  // 
  // Create a similar flow from n3 to n0, starting at time 1.1 seconds
  //
  onoff.SetAttribute ("Remote", 
                      AddressValue (InetSocketAddress (Ipv4Address ("10.1.1.2"), port)));
  ApplicationContainer app2 = onoff.Install (n3);
  app2.Start (Seconds (1.1));
  app2.Stop (Seconds (10.0));

  ApplicationContainer sink2 = sink.Install (n0);
  sink2.Start (Seconds (1.1));
  sink2.Stop (Seconds (10.0));

  NS_LOG_INFO ("Configure Tracing.");

  //
  // Configure tracing of all enqueue, dequeue, and NetDevice receive events.
  // Trace output will be sent to the file "csma-bridge-one-hop.tr"
  //
  AsciiTraceHelper ascii;
  csma.EnableAsciiAll (ascii.CreateFileStream ("csma-bridge-one-hop.tr"));

  //
  // Also configure some tcpdump traces; each interface will be traced.
  // The output files will be named:
  //     csma-bridge-one-hop-<nodeId>-<interfaceId>.pcap
  // and can be read by the "tcpdump -r" command (use "-tt" option to
  // display timestamps correctly)
  //
  csma.EnablePcapAll ("csma-bridge-one-hop", false);

  //
  // Now, do the actual simulation.
  //
  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}
