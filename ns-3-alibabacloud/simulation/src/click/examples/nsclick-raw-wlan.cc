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

// Scenario: node A (using Click) sends packets to node B (not using
// Click)
//
//  (Click)             (non-Click)
//    A )))     WLAN      ((( B
// (172.16.1.1)         (172.16.1.2)
//    (eth0)
//

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/click-internet-stack-helper.h"
#include "ns3/log.h"
#include "ns3/mobility-helper.h"

using namespace ns3;

void ReceivePacket (Ptr<Socket> socket)
{
  NS_LOG_UNCOND ("Received one packet!");
}

int main (int argc, char *argv[])
{
#ifdef NS3_CLICK
  double rss = -80;
  std::string clickConfigFolder = "src/click/examples";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("clickConfigFolder", "Base folder for click configuration files", clickConfigFolder);
  cmd.Parse (argc, argv);

  // Setup nodes
  NodeContainer wifiNodes;
  wifiNodes.Create (2);

  // Get Wifi devices installed on both nodes.
  // Adapted from examples/wireless/wifi-simple-adhoc.cc
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
  wifiChannel.AddPropagationLoss ("ns3::FixedRssLossModel","Rss",DoubleValue (rss));
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add an upper mac and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));
  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer wifiDevices = wifi.Install (wifiPhy, wifiMac, wifiNodes);

  // Setup mobility models
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiNodes);

  // Install normal internet stack on node B
  InternetStackHelper internet;
  internet.Install (wifiNodes.Get (1));

  // Install Click on node A
  ClickInternetStackHelper clickinternet;
  clickinternet.SetClickFile (wifiNodes.Get (0), clickConfigFolder + "/nsclick-wifi-single-interface.click");
  clickinternet.SetRoutingTableElement (wifiNodes.Get (0), "rt");
  clickinternet.Install (wifiNodes.Get (0));

  // Configure IP addresses
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  ipv4.Assign (wifiDevices);

  // Setup traffic application and sockets
  Address LocalAddress (InetSocketAddress (Ipv4Address::GetAny (), 50000));
  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", LocalAddress);
  ApplicationContainer recvapp = packetSinkHelper.Install (wifiNodes.Get (1));
  recvapp.Start (Seconds (5.0));
  recvapp.Stop (Seconds (10.0));

  OnOffHelper onOffHelper ("ns3::TcpSocketFactory", Address ());
  onOffHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  onOffHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

  ApplicationContainer appcont;

  AddressValue remoteAddress (InetSocketAddress (Ipv4Address ("172.16.1.2"), 50000));
  onOffHelper.SetAttribute ("Remote", remoteAddress);
  appcont.Add (onOffHelper.Install (wifiNodes.Get (0)));

  appcont.Start (Seconds (5.0));
  appcont.Stop (Seconds (10.0));

  // For tracing
  wifiPhy.EnablePcap ("nsclick-raw-wlan", wifiDevices);

  Simulator::Stop (Seconds (20.0));
  Simulator::Run ();

  Simulator::Destroy ();
  return 0;
#else
  NS_FATAL_ERROR ("Can't use ns-3-click without NSCLICK compiled in");
#endif
}
