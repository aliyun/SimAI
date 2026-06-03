/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017
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
 * Author: Sebastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/enum.h"
#include "ns3/tuple.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/yans-wifi-channel.h"

// This is an example to show how to configure an IEEE 802.11 Wi-Fi
// network where the AP and the station use different 802.11 standards.
//
// It outputs the throughput for a given configuration: user can specify
// the 802.11 versions for the AP and the station as well as their rate
// adaptation algorithms. It also allows to decide whether the station,
// the AP or both has/have traffic to send.
//
// Example for an IEEE 802.11ac station sending traffic to an 802.11a AP using Ideal rate adaptation algorithm:
// ./ns3 run "wifi-backward-compatibility --apVersion=80211a --staVersion=80211ac --staRaa=Ideal"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("wifi-backward-compatibility");

std::pair<WifiStandard, WifiPhyBand> ConvertStringToStandardAndBand (std::string version)
{
  WifiStandard standard = WIFI_STANDARD_80211a;
  WifiPhyBand band = WIFI_PHY_BAND_5GHZ;
  if (version == "80211a")
    {
      standard = WIFI_STANDARD_80211a;
      band = WIFI_PHY_BAND_5GHZ;
    }
  else if (version == "80211b")
    {
      standard = WIFI_STANDARD_80211b;
      band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (version == "80211g")
    {
      standard = WIFI_STANDARD_80211g;
      band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (version == "80211p")
    {
      standard = WIFI_STANDARD_80211p;
      band = WIFI_PHY_BAND_5GHZ;
    }
  else if (version == "80211n_2_4GHZ")
    {
      standard = WIFI_STANDARD_80211n;
      band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (version == "80211n_5GHZ")
    {
      standard = WIFI_STANDARD_80211n;
      band = WIFI_PHY_BAND_5GHZ;
    }
  else if (version == "80211ac")
    {
      standard = WIFI_STANDARD_80211ac;
      band = WIFI_PHY_BAND_5GHZ;
    }
  else if (version == "80211ax_2_4GHZ")
    {
      standard = WIFI_STANDARD_80211ax;
      band = WIFI_PHY_BAND_2_4GHZ;
    }
  else if (version == "80211ax_5GHZ")
    {
      standard = WIFI_STANDARD_80211ax;
      band = WIFI_PHY_BAND_5GHZ;
    }
  return {standard, band};
}

int main (int argc, char *argv[])
{
  uint32_t payloadSize = 1472; //bytes
  double simulationTime = 10; //seconds
  std::string apVersion = "80211a";
  std::string staVersion = "80211n_5GHZ";
  std::string apRaa = "Minstrel";
  std::string staRaa = "MinstrelHt";
  bool apHasTraffic = false;
  bool staHasTraffic = true;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("simulationTime", "Simulation time in seconds", simulationTime);
  cmd.AddValue ("apVersion", "The standard version used by the AP: 80211a, 80211b, 80211g, 80211p, 80211n_2_4GHZ, 80211n_5GHZ, 80211ac, 80211ax_2_4GHZ or 80211ax_5GHZ", apVersion);
  cmd.AddValue ("staVersion", "The standard version used by the station: 80211a, 80211b, 80211g, 80211_10MHZ, 80211_5MHZ, 80211n_2_4GHZ, 80211n_5GHZ, 80211ac, 80211ax_2_4GHZ or 80211ax_5GHZ", staVersion);
  cmd.AddValue ("apRaa", "Rate adaptation algorithm used by the AP", apRaa);
  cmd.AddValue ("staRaa", "Rate adaptation algorithm used by the station", staRaa);
  cmd.AddValue ("apHasTraffic", "Enable/disable traffic on the AP", apHasTraffic);
  cmd.AddValue ("staHasTraffic", "Enable/disable traffic on the station", staHasTraffic);
  cmd.Parse (argc,argv);

  NodeContainer wifiStaNode;
  wifiStaNode.Create (1);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiMacHelper mac;
  WifiHelper wifi;
  Ssid ssid = Ssid ("ns3");
  TupleValue<UintegerValue, UintegerValue, EnumValue, UintegerValue> channelValue;

  const auto& [staStandard, staBand] = ConvertStringToStandardAndBand (staVersion);
  wifi.SetStandard (staStandard);
  wifi.SetRemoteStationManager ("ns3::" + staRaa + "WifiManager");

  mac.SetType ("ns3::StaWifiMac",
               "QosSupported", BooleanValue (true),
               "Ssid", SsidValue (ssid));

  //Workaround needed as long as we do not fully support channel bonding
  uint16_t width = (staVersion == "80211ac" ? 20 : 0);
  channelValue.Set (WifiPhy::ChannelTuple {0, width, staBand, 0});
  phy.Set ("ChannelSettings", channelValue);

  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  const auto& [apStandard, apBand] = ConvertStringToStandardAndBand (apVersion);
  wifi.SetStandard (apStandard);
  wifi.SetRemoteStationManager ("ns3::" + apRaa + "WifiManager");

  mac.SetType ("ns3::ApWifiMac",
               "QosSupported", BooleanValue (true),
               "Ssid", SsidValue (ssid));

  //Workaround needed as long as we do not fully support channel bonding
  width = (apVersion == "80211ac" ? 20 : 0);
  channelValue.Set (WifiPhy::ChannelTuple {0, width, apBand, 0});
  phy.Set ("ChannelSettings", channelValue);

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNode);

  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNode);

  Ipv4AddressHelper address;
  address.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodeInterface;
  Ipv4InterfaceContainer apNodeInterface;

  staNodeInterface = address.Assign (staDevice);
  apNodeInterface = address.Assign (apDevice);

  UdpServerHelper apServer (9);
  ApplicationContainer apServerApp = apServer.Install (wifiApNode.Get (0));
  apServerApp.Start (Seconds (0.0));
  apServerApp.Stop (Seconds (simulationTime + 1));

  UdpServerHelper staServer (5001);
  ApplicationContainer staServerApp = staServer.Install (wifiStaNode.Get (0));
  staServerApp.Start (Seconds (0.0));
  staServerApp.Stop (Seconds (simulationTime + 1));

  if (apHasTraffic)
    {
      UdpClientHelper apClient (staNodeInterface.GetAddress (0), 5001);
      apClient.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
      apClient.SetAttribute ("Interval", TimeValue (Time ("0.00001"))); //packets/s
      apClient.SetAttribute ("PacketSize", UintegerValue (payloadSize)); //bytes
      ApplicationContainer apClientApp = apClient.Install (wifiApNode.Get (0));
      apClientApp.Start (Seconds (1.0));
      apClientApp.Stop (Seconds (simulationTime + 1));
    }

  if (staHasTraffic)
    {
      UdpClientHelper staClient (apNodeInterface.GetAddress (0), 9);
      staClient.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
      staClient.SetAttribute ("Interval", TimeValue (Time ("0.00001"))); //packets/s
      staClient.SetAttribute ("PacketSize", UintegerValue (payloadSize)); //bytes
      ApplicationContainer staClientApp = staClient.Install (wifiStaNode.Get (0));
      staClientApp.Start (Seconds (1.0));
      staClientApp.Stop (Seconds (simulationTime + 1));
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Simulator::Stop (Seconds (simulationTime + 1));
  Simulator::Run ();

  uint64_t rxBytes;
  double throughput;
  bool error = false;
  if (apHasTraffic)
    {
      rxBytes = payloadSize * DynamicCast<UdpServer> (staServerApp.Get (0))->GetReceived ();
      throughput = (rxBytes * 8) / (simulationTime * 1000000.0); //Mbit/s
      std::cout << "AP Throughput: " << throughput << " Mbit/s" << std::endl;
      if (throughput == 0)
      {
        error = true;
      }
    }
  if (staHasTraffic)
    {
      rxBytes = payloadSize * DynamicCast<UdpServer> (apServerApp.Get (0))->GetReceived ();
      throughput = (rxBytes * 8) / (simulationTime * 1000000.0); //Mbit/s
      std::cout << "STA Throughput: " << throughput << " Mbit/s" << std::endl;
      if (throughput == 0)
      {
        error = true;
      }
    }

  Simulator::Destroy ();

  if (error)
    {
      NS_LOG_ERROR ("No traffic received!");
      exit (1);
    }

  return 0;
}
