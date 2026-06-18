/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 SEBASTIEN DERONNE
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
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/yans-wifi-channel.h"

// This is a simple example in order to show how to configure an IEEE 802.11ac Wi-Fi network.
//
// It outputs the UDP or TCP goodput for every VHT MCS value, which depends on the MCS value (0 to 9, where 9 is
// forbidden when the channel width is 20 MHz), the channel width (20, 40, 80 or 160 MHz) and the guard interval (long
// or short). The PHY bitrate is constant over all the simulation run. The user can also specify the distance between
// the access point and the station: the larger the distance the smaller the goodput.
//
// The simulation assumes a single station in an infrastructure network:
//
//  STA     AP
//    *     *
//    |     |
//   n1     n2
//
//Packets in this simulation belong to BestEffort Access Class (AC_BE).

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("vht-wifi-network");

int main (int argc, char *argv[])
{
  bool udp = true;
  bool useRts = false;
  double simulationTime = 10; //seconds
  double distance = 1.0; //meters
  int mcs = -1; // -1 indicates an unset value
  double minExpectedThroughput = 0;
  double maxExpectedThroughput = 0;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("distance", "Distance in meters between the station and the access point", distance);
  cmd.AddValue ("simulationTime", "Simulation time in seconds", simulationTime);
  cmd.AddValue ("udp", "UDP if set to 1, TCP otherwise", udp);
  cmd.AddValue ("useRts", "Enable/disable RTS/CTS", useRts);
  cmd.AddValue ("mcs", "if set, limit testing to a specific MCS (0-9)", mcs);
  cmd.AddValue ("minExpectedThroughput", "if set, simulation fails if the lowest throughput is below this value", minExpectedThroughput);
  cmd.AddValue ("maxExpectedThroughput", "if set, simulation fails if the highest throughput is above this value", maxExpectedThroughput);
  cmd.Parse (argc,argv);

  if (useRts)
    {
      Config::SetDefault ("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue ("0"));
    }

  double prevThroughput [8];
  for (uint32_t l = 0; l < 8; l++)
    {
      prevThroughput[l] = 0;
    }
  std::cout << "MCS value" << "\t\t" << "Channel width" << "\t\t" << "short GI" << "\t\t" << "Throughput" << '\n';
  int minMcs = 0;
  int maxMcs = 9;
  if (mcs >= 0 && mcs <= 9)
    {
      minMcs = mcs;
      maxMcs = mcs;
    }
  for (int mcs = minMcs; mcs <= maxMcs; mcs++)
    {
      uint8_t index = 0;
      double previous = 0;
      for (int channelWidth = 20; channelWidth <= 160; )
        {
          if (mcs == 9 && channelWidth == 20)
            {
              channelWidth *= 2;
              continue;
            }
          for (int sgi = 0; sgi < 2; sgi++)
            {
              uint32_t payloadSize; //1500 byte IP packet
              if (udp)
                {
                  payloadSize = 1472; //bytes
                }
              else
                {
                  payloadSize = 1448; //bytes
                  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (payloadSize));
                }

              NodeContainer wifiStaNode;
              wifiStaNode.Create (1);
              NodeContainer wifiApNode;
              wifiApNode.Create (1);

              YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
              YansWifiPhyHelper phy;
              phy.SetChannel (channel.Create ());

              phy.Set ("ChannelSettings", StringValue ("{0, " + std::to_string (channelWidth)
                                                       + ", BAND_5GHZ, 0}"));

              WifiHelper wifi;
              wifi.SetStandard (WIFI_STANDARD_80211ac);
              WifiMacHelper mac;

              std::ostringstream oss;
              oss << "VhtMcs" << mcs;
              wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", StringValue (oss.str ()),
                                            "ControlMode", StringValue (oss.str ()));

              Ssid ssid = Ssid ("ns3-80211ac");

              mac.SetType ("ns3::StaWifiMac",
                           "Ssid", SsidValue (ssid));

              NetDeviceContainer staDevice;
              staDevice = wifi.Install (phy, mac, wifiStaNode);

              mac.SetType ("ns3::ApWifiMac",
                           "EnableBeaconJitter", BooleanValue (false),
                           "Ssid", SsidValue (ssid));

              NetDeviceContainer apDevice;
              apDevice = wifi.Install (phy, mac, wifiApNode);

              // Set guard interval
              Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (sgi));

              // mobility.
              MobilityHelper mobility;
              Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

              positionAlloc->Add (Vector (0.0, 0.0, 0.0));
              positionAlloc->Add (Vector (distance, 0.0, 0.0));
              mobility.SetPositionAllocator (positionAlloc);

              mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");

              mobility.Install (wifiApNode);
              mobility.Install (wifiStaNode);

              /* Internet stack*/
              InternetStackHelper stack;
              stack.Install (wifiApNode);
              stack.Install (wifiStaNode);

              Ipv4AddressHelper address;
              address.SetBase ("192.168.1.0", "255.255.255.0");
              Ipv4InterfaceContainer staNodeInterface;
              Ipv4InterfaceContainer apNodeInterface;

              staNodeInterface = address.Assign (staDevice);
              apNodeInterface = address.Assign (apDevice);

              /* Setting applications */
              ApplicationContainer serverApp;
              if (udp)
                {
                  //UDP flow
                  uint16_t port = 9;
                  UdpServerHelper server (port);
                  serverApp = server.Install (wifiStaNode.Get (0));
                  serverApp.Start (Seconds (0.0));
                  serverApp.Stop (Seconds (simulationTime + 1));

                  UdpClientHelper client (staNodeInterface.GetAddress (0), port);
                  client.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
                  client.SetAttribute ("Interval", TimeValue (Time ("0.00002"))); //packets/s
                  client.SetAttribute ("PacketSize", UintegerValue (payloadSize));
                  ApplicationContainer clientApp = client.Install (wifiApNode.Get (0));
                  clientApp.Start (Seconds (1.0));
                  clientApp.Stop (Seconds (simulationTime + 1));
                }
              else
                {
                  //TCP flow
                  uint16_t port = 50000;
                  Address localAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
                  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", localAddress);
                  serverApp = packetSinkHelper.Install (wifiStaNode.Get (0));
                  serverApp.Start (Seconds (0.0));
                  serverApp.Stop (Seconds (simulationTime + 1));

                  OnOffHelper onoff ("ns3::TcpSocketFactory", Ipv4Address::GetAny ());
                  onoff.SetAttribute ("OnTime",  StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
                  onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
                  onoff.SetAttribute ("PacketSize", UintegerValue (payloadSize));
                  onoff.SetAttribute ("DataRate", DataRateValue (1000000000)); //bit/s
                  AddressValue remoteAddress (InetSocketAddress (staNodeInterface.GetAddress (0), port));
                  onoff.SetAttribute ("Remote", remoteAddress);
                  ApplicationContainer clientApp = onoff.Install (wifiApNode.Get (0));
                  clientApp.Start (Seconds (1.0));
                  clientApp.Stop (Seconds (simulationTime + 1));
                }

              Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

              Simulator::Stop (Seconds (simulationTime + 1));
              Simulator::Run ();

              uint64_t rxBytes = 0;
              if (udp)
                {
                  rxBytes = payloadSize * DynamicCast<UdpServer> (serverApp.Get (0))->GetReceived ();
                }
              else
                {
                  rxBytes = DynamicCast<PacketSink> (serverApp.Get (0))->GetTotalRx ();
                }
              double throughput = (rxBytes * 8) / (simulationTime * 1000000.0); //Mbit/s

              Simulator::Destroy ();

              std::cout << mcs << "\t\t\t" << channelWidth << " MHz\t\t\t" << sgi << "\t\t\t" << throughput << " Mbit/s" << std::endl;

              //test first element
              if (mcs == 0 && channelWidth == 20 && sgi == 0)
                {
                  if (throughput < minExpectedThroughput)
                    {
                      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not expected!");
                      exit (1);
                    }
                }
              //test last element
              if (mcs == 9 && channelWidth == 160 && sgi == 1)
                {
                  if (maxExpectedThroughput > 0 && throughput > maxExpectedThroughput)
                    {
                      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not expected!");
                      exit (1);
                    }
                }
              //test previous throughput is smaller (for the same mcs)
              if (throughput > previous)
                {
                  previous = throughput;
                }
              else
                {
                  NS_LOG_ERROR ("Obtained throughput " << throughput << " is not expected!");
                  exit (1);
                }
              //test previous throughput is smaller (for the same channel width and GI)
              if (throughput > prevThroughput [index])
                {
                  prevThroughput [index] = throughput;
                }
              else
                {
                  NS_LOG_ERROR ("Obtained throughput " << throughput << " is not expected!");
                  exit (1);
                }
              index++;
            }
          channelWidth *= 2;
        }
    }
  return 0;
}
