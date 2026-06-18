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

// Adaptation of examples/udp/udp-client-server.cc for
// Click based nodes running wifi.
//
// Network topology:
//
//               (1.4)
//             (( n4 ))
//
//          172.16.1.0/24
//
//   (1.1)      (1.2)       (1.3)
//     n0 ))   (( n1 ))   (( n2
//               WLAN
//
// - UDP flows from n0 to n1 and n2 to n1.
// - All nodes are Click based.
// - The single ethernet interface that each node
//   uses is named 'eth0' in the Click file.
// - Node 4 is running in promiscuous mode and can listen in on
//   the packets being exchanged between n0-n1 and n2-n1.
//

#include <fstream>
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-click-routing.h"
#include "ns3/click-internet-stack-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NsclickUdpClientServerWifi");

#ifdef NS3_CLICK
void
ReadArp (Ptr<Ipv4ClickRouting> clickRouter)
{
  // Access the handlers
  NS_LOG_INFO (clickRouter->ReadHandler ("wifi/arpquerier", "table"));
  NS_LOG_INFO (clickRouter->ReadHandler ("wifi/arpquerier", "stats"));
}

void
WriteArp (Ptr<Ipv4ClickRouting> clickRouter)
{
  // Access the handler
  NS_LOG_INFO (clickRouter->WriteHandler ("wifi/arpquerier", "insert", "172.16.1.2 00:00:00:00:00:02"));
}
#endif

int
main (int argc, char *argv[])
{
#ifdef NS3_CLICK
  std::string clickConfigFolder = "src/click/examples";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("clickConfigFolder", "Base folder for click configuration files", clickConfigFolder);
  cmd.Parse (argc, argv);

  //
  // Enable logging
  //
  LogComponentEnable ("NsclickUdpClientServerWifi", LOG_LEVEL_INFO);

  //
  // Explicitly create the nodes required by the topology (shown above).
  //
  NS_LOG_INFO ("Create nodes.");
  NodeContainer n;
  n.Create (4);

  NS_LOG_INFO ("Create channels.");
  //
  // Explicitly create the channels required by the topology (shown above).
  //
  std::string phyMode ("DsssRate1Mbps");

  // disable fragmentation for frames below 2200 bytes
  Config::SetDefault ("ns3::WifiRemoteStationManager::FragmentationThreshold", StringValue ("2200"));
  // turn off RTS/CTS for frames below 2200 bytes
  Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue ("2200"));
  // Fix non-unicast data rate to be the same as that of unicast
  Config::SetDefault ("ns3::WifiRemoteStationManager::NonUnicastMode",
                      StringValue (phyMode));

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211b);

  YansWifiPhyHelper wifiPhy;
  // This is one parameter that matters when using FixedRssLossModel
  // set it to zero; otherwise, gain will be added
  wifiPhy.Set ("RxGain", DoubleValue (0) );
  // ns-3 supports RadioTap and Prism tracing extensions for 802.11b
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  // The below FixedRssLossModel will cause the rss to be fixed regardless
  // of the distance between the two stations, and the transmit power
  wifiChannel.AddPropagationLoss ("ns3::FixedRssLossModel","Rss",DoubleValue (-80));
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add an upper mac and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));
  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer d = wifi.Install (wifiPhy, wifiMac, n);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (10.0, 0.0, 0.0));
  positionAlloc->Add (Vector (20.0, 0.0, 0.0));
  positionAlloc->Add (Vector (0.0, 10.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (n);

  //
  // Install Click on the nodes
  //
  ClickInternetStackHelper clickinternet;
  clickinternet.SetClickFile (n.Get (0), clickConfigFolder + "/nsclick-wifi-single-interface.click");
  clickinternet.SetClickFile (n.Get (1), clickConfigFolder + "/nsclick-wifi-single-interface.click");
  clickinternet.SetClickFile (n.Get (2), clickConfigFolder + "/nsclick-wifi-single-interface.click");

  // Node 4 is to run in promiscuous mode. This can be verified
  // from the pcap trace Node4_in_eth0.pcap generated after running
  // this script.
  clickinternet.SetClickFile (n.Get (3), clickConfigFolder + "/nsclick-wifi-single-interface-promisc.click");
  clickinternet.SetRoutingTableElement (n, "rt");
  clickinternet.Install (n);
  Ipv4AddressHelper ipv4;
  //
  // We've got the "hardware" in place.  Now we need to add IP addresses.
  //
  NS_LOG_INFO ("Assign IP Addresses.");
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = ipv4.Assign (d);

  NS_LOG_INFO ("Create Applications.");
  //
  // Create one udpServer applications on node one.
  //
  uint16_t port = 4000;
  UdpServerHelper server (port);
  ApplicationContainer apps = server.Install (n.Get (1));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  //
  // Create one UdpClient application to send UDP datagrams from node zero to
  // node one.
  //
  uint32_t MaxPacketSize = 1024;
  Time interPacketInterval = Seconds (0.5);
  uint32_t maxPacketCount = 320;
  UdpClientHelper client (i.GetAddress (1), port);
  client.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  client.SetAttribute ("Interval", TimeValue (interPacketInterval));
  client.SetAttribute ("PacketSize", UintegerValue (MaxPacketSize));
  apps = client.Install (NodeContainer (n.Get (0), n.Get (2)));
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (10.0));

  wifiPhy.EnablePcap ("nsclick-udp-client-server-wifi", d);

  // Force the MAC address of the second node: The current ARP
  // implementation of Click sends only one ARP request per incoming
  // packet for an unknown destination and does not retransmit if no
  // response is received. With the scenario of this example, all ARP
  // requests of node 3 are lost due to interference from node
  // 1. Hence, we fill in the ARP table of node 2 before at the
  // beginning of the simulation
  Simulator::Schedule (Seconds (0.5), &ReadArp, n.Get (2)->GetObject<Ipv4ClickRouting> ());
  Simulator::Schedule (Seconds (0.6), &WriteArp, n.Get (2)->GetObject<Ipv4ClickRouting> ());
  Simulator::Schedule (Seconds (0.7), &ReadArp, n.Get (2)->GetObject<Ipv4ClickRouting> ());

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
