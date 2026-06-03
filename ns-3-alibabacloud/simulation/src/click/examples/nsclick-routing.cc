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
 *
 * Authors: Lalith Suresh <suresh.lalith@gmail.com>
 */

// Network topology
//
//
//              172.16.1.0/24
//        (1.1)  (1.2)  (2.1)  (2.2)
//
//         eth0   eth0  eth1    eth0
//       n0 ========= n1 ========= n2
//            LAN 1       LAN 2
//
// - UDP flows from n0 to n2 via n1.
// - All nodes are Click based.
//

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/ipv4-l3-click-protocol.h"
#include "ns3/click-internet-stack-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NsclickRouting");

int
main (int argc, char *argv[])
{
#ifdef NS3_CLICK
  std::string clickConfigFolder = "src/click/examples";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("clickConfigFolder", "Base folder for click configuration files", clickConfigFolder);
  cmd.Parse (argc, argv);

//
// Explicitly create the nodes required by the topology (shown above).
//
  NS_LOG_INFO ("Create nodes.");
  NodeContainer n;
  n.Create (3);

//
// Install Click on the nodes
//
  ClickInternetStackHelper clickinternet;
  clickinternet.SetClickFile (n.Get (0), clickConfigFolder + "/nsclick-routing-node0.click");
  clickinternet.SetClickFile (n.Get (1), clickConfigFolder + "/nsclick-ip-router.click");
  clickinternet.SetClickFile (n.Get (2), clickConfigFolder + "/nsclick-routing-node2.click");
  clickinternet.SetRoutingTableElement (n.Get (0), "kernel/rt");
  clickinternet.SetRoutingTableElement (n.Get (1), "u/rt");
  clickinternet.SetRoutingTableElement (n.Get (2), "kernel/rt");
  clickinternet.Install (n);

  NS_LOG_INFO ("Create channels.");
//
// Explicitly create the channels required by the topology (shown above).
//
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1400));
  NetDeviceContainer d01 = csma.Install (NodeContainer (n.Get (0), n.Get (1)));
  NetDeviceContainer d12 = csma.Install (NodeContainer (n.Get (1), n.Get (2)));

  Ipv4AddressHelper ipv4;
//
// We've got the "hardware" in place.  Now we need to add IP addresses.
//
  NS_LOG_INFO ("Assign IP Addresses.");
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i01 = ipv4.Assign (d01);

  ipv4.SetBase ("172.16.2.0", "255.255.255.0");
  Ipv4InterfaceContainer i12 = ipv4.Assign (d12);

  NS_LOG_INFO ("Create Applications.");
//
// Create one udpServer applications on node one.
//
  uint16_t port = 4000;
  UdpServerHelper server (port);
  ApplicationContainer apps = server.Install (n.Get (2));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

//
// Create one UdpClient application to send UDP datagrams from node zero to
// node one.
//
  uint32_t MaxPacketSize = 1024;
  Time interPacketInterval = Seconds (0.05);
  uint32_t maxPacketCount = 320;
  UdpClientHelper client (i12.GetAddress (1), port);
  client.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  client.SetAttribute ("Interval", TimeValue (interPacketInterval));
  client.SetAttribute ("PacketSize", UintegerValue (MaxPacketSize));
  apps = client.Install (NodeContainer (n.Get (0)));
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (10.0));

  csma.EnablePcap ("nsclick-routing", d01, false);
  csma.EnablePcap ("nsclick-routing", d12, false);

//
// Now, do the actual simulation.
//
  NS_LOG_INFO ("Run Simulation.");
  Simulator::Stop (Seconds (20.0));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
#else
  NS_FATAL_ERROR ("Can't use ns-3-click without NSCLICK compiled in");
#endif
}
