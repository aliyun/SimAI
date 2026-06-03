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
 * This example is inspired from examples/tutorial/third.cc by
 * substituting the CSMA network with another WiFi network.
 *
 * Author: Ameya Deshpande <ameyanrd@outlook.com>
 */

#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/ssid.h"
#include "ns3/nix-vector-routing-module.h"

/**
 * This example demonstrates how Nix works with
 * two Wifi networks on the same channel.
 *
 * IPv4 Network Topology
 * \verbatim
    Wifi 10.1.1.0/24
                   AP
    *    *    *    *
    |    |    |    |   10.1.2.0/24
   n5   n6   n7   n0 -------------- n1   n2   n3   n4
                     point-to-point  |    |    |    |
                                     *    *    *    *
                                    AP
                                      Wifi 10.1.3.0/24
   \endverbatim
 *
 * \verbatim
    Wifi 2001:1::/64
                   AP
    *    *    *    *
    |    |    |    |   2001:2::/64
   n5   n6   n7   n0 -------------- n1   n2   n3   n4
                     point-to-point  |    |    |    |
                                     *    *    *    *
                                    AP
                                      Wifi 2001:3::/64
   \endverbatim
 *
 * Expected Outputs:
 * IPv4:
 * \verbatim
   Time: +7s, Nix Routing
   Route path from Node 4 to Node 7, Nix Vector: 100011 (6 bits left)
   10.1.1.3                 (Node 4)  ---->   10.1.1.4                 (Node 0)
   10.1.2.1                 (Node 0)  ---->   10.1.2.2                 (Node 1)
   10.1.3.4                 (Node 1)  ---->   10.1.3.3                 (Node 7)
   \endverbatim
 *
 * IPv6:
 * \verbatim
   Time: +7s, Nix Routing
   Route path from Node 4 to Node 7, Nix Vector: 100011 (6 bits left)
   2001:1::200:ff:fe00:5    (Node 4)  ---->   fe80::200:ff:fe00:6      (Node 0)
   fe80::200:ff:fe00:1      (Node 0)  ---->   fe80::200:ff:fe00:2      (Node 1)
   fe80::200:ff:fe00:a      (Node 1)  ---->   2001:3::200:ff:fe00:9    (Node 7)
   \endverbatim
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("NixDoubleWifiExample");

int
main (int argc, char *argv[])
{
  bool useIpv6 = false;
  bool enableNixLog = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("useIPv6", "Use IPv6 instead of IPv4", useIpv6);
  cmd.AddValue ("enableNixLog", "Enable NixVectorRouting logging", enableNixLog);
  cmd.Parse (argc,argv);

  LogComponentEnable ("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable ("UdpEchoServerApplication", LOG_LEVEL_INFO);
  if (enableNixLog)
    {
      LogComponentEnable ("NixVectorRouting", LOG_LEVEL_LOGIC);
    }

  NodeContainer p2pNodes;
  p2pNodes.Create (2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer p2pDevices;
  p2pDevices = pointToPoint.Install (p2pNodes);

  NodeContainer wifiStaNodes1;
  wifiStaNodes1.Create (3);
  NodeContainer wifiApNode1 = p2pNodes.Get (0);

  NodeContainer wifiStaNodes2;
  wifiStaNodes2.Create (3);
  NodeContainer wifiApNode2 = p2pNodes.Get (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid-first");
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevices1;
  staDevices1 = wifi.Install (phy, mac, wifiStaNodes1);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevices1;
  apDevices1 = wifi.Install (phy, mac, wifiApNode1);

  ssid = Ssid ("ns-3-ssid-second");
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevices2;
  staDevices2 = wifi.Install (phy, mac, wifiStaNodes2);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevices2;
  apDevices2 = wifi.Install (phy, mac, wifiApNode2);

  MobilityHelper mobility;

  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (0.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (5.0),
                                 "DeltaY", DoubleValue (10.0),
                                 "GridWidth", UintegerValue (3),
                                 "LayoutType", StringValue ("RowFirst"));

  mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue (Rectangle (-50, 50, -50, 50)));
  mobility.Install (wifiStaNodes1);
  mobility.Install (wifiStaNodes2);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode1);
  mobility.Install (wifiApNode2);

  Address udpServerAddress;

  if (!useIpv6)
    {
      InternetStackHelper stack;
      Ipv4NixVectorHelper nixRouting;
      stack.SetRoutingHelper (nixRouting);
      stack.Install (wifiApNode1);
      stack.Install (wifiStaNodes1);
      stack.Install (wifiApNode2);
      stack.Install (wifiStaNodes2);

      Ipv4AddressHelper address;

      address.SetBase ("10.1.1.0", "255.255.255.0");
      address.Assign (staDevices1);
      address.Assign (apDevices1);

      address.SetBase ("10.1.2.0", "255.255.255.0");
      Ipv4InterfaceContainer p2pInterfaces;
      p2pInterfaces = address.Assign (p2pDevices);

      address.SetBase ("10.1.3.0", "255.255.255.0");
      Ipv4InterfaceContainer staDevicesInterfaces2;
      staDevicesInterfaces2 = address.Assign (staDevices2);
      address.Assign (apDevices2);

      udpServerAddress = staDevicesInterfaces2.GetAddress (2);

      Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("nix-double-wifi-ipv4.routes", std::ios::out);
      nixRouting.PrintRoutingPathAt (Seconds (7), wifiStaNodes1.Get (2), staDevicesInterfaces2.GetAddress (2), routingStream);
    }
  else
    {
      InternetStackHelper stack;
      Ipv6NixVectorHelper nixRouting;
      stack.SetRoutingHelper (nixRouting);
      stack.Install (wifiApNode1);
      stack.Install (wifiStaNodes1);
      stack.Install (wifiApNode2);
      stack.Install (wifiStaNodes2);

      Ipv6AddressHelper address;

      address.SetBase (Ipv6Address ("2001:1::"), Ipv6Prefix (64));
      address.Assign (staDevices1);
      address.Assign (apDevices1);

      address.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
      Ipv6InterfaceContainer p2pInterfaces;
      p2pInterfaces = address.Assign (p2pDevices);

      address.SetBase (Ipv6Address ("2001:3::"), Ipv6Prefix (64));
      Ipv6InterfaceContainer staDevicesInterfaces2;
      staDevicesInterfaces2 = address.Assign (staDevices2);
      address.Assign (apDevices2);

      udpServerAddress = staDevicesInterfaces2.GetAddress (2, 1);

      Ptr<OutputStreamWrapper> routingStream = Create<OutputStreamWrapper> ("nix-double-wifi-ipv6.routes", std::ios::out);
      nixRouting.PrintRoutingPathAt (Seconds (7), wifiStaNodes1.Get (2), staDevicesInterfaces2.GetAddress (2, 1), routingStream);
    }

  UdpEchoServerHelper echoServer (9);

  ApplicationContainer serverApps = echoServer.Install (wifiStaNodes2.Get (2));
  serverApps.Start (Seconds (1.0));
  serverApps.Stop (Seconds (10.0));

  UdpEchoClientHelper echoClient (udpServerAddress, 9);
  echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
  echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
  echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps = echoClient.Install (wifiStaNodes1.Get (2));
  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (10.0));

  Simulator::Stop (Seconds (10.0));

  Simulator::Run ();
  Simulator::Destroy ();
  return 0;
}
