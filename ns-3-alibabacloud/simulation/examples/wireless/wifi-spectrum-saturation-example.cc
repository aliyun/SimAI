/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 MIRKO BANCHI
 * Copyright (c) 2015 University of Washington
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
 * Authors: Mirko Banchi <mk.banchi@gmail.com>
 *          Sebastien Deronne <sebastien.deronne@gmail.com>
 *          Tom Henderson <tomhend@u.washington.edu>
 *
 * Adapted from wifi-ht-network.cc example
 */

#include <iomanip>
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/propagation-loss-model.h"

// This is a simple example of an IEEE 802.11n Wi-Fi network.
//
// The main use case is to enable and test SpectrumWifiPhy vs YansWifiPhy
// under saturation conditions (for max throughput).
//
// Network topology:
//
//  Wi-Fi 192.168.1.0
//
//   STA                  AP
//    * <-- distance -->  *
//    |                   |
//    n1                  n2
//
// Users may vary the following command-line arguments in addition to the
// attributes, global values, and default values typically available:
//
//    --simulationTime:  Simulation time in seconds [10]
//    --distance:        meters separation between nodes [1]
//    --index:           restrict index to single value between 0 and 31 [256]
//    --wifiType:        select ns3::SpectrumWifiPhy or ns3::YansWifiPhy [ns3::SpectrumWifiPhy]
//    --errorModelType:  select ns3::NistErrorRateModel or ns3::YansErrorRateModel [ns3::NistErrorRateModel]
//    --enablePcap:      enable pcap output [false]
//
// By default, the program will step through 64 index values, corresponding
// to the following MCS, channel width, and guard interval combinations:
//   index 0-7:    MCS 0-7, long guard interval, 20 MHz channel
//   index 8-15:   MCS 0-7, short guard interval, 20 MHz channel
//   index 16-23:  MCS 0-7, long guard interval, 40 MHz channel
//   index 24-31:  MCS 0-7, short guard interval, 40 MHz channel
//   index 32-39:    MCS 8-15, long guard interval, 20 MHz channel
//   index 40-47:   MCS 8-15, short guard interval, 20 MHz channel
//   index 48-55:  MCS 8-15, long guard interval, 40 MHz channel
//   index 56-63:  MCS 8-15, short guard interval, 40 MHz channel
// and send packets at a high rate using each MCS, using the SpectrumWifiPhy
// and the NistErrorRateModel, at a distance of 1 meter.  The program outputs
// results such as:
//
// wifiType: ns3::SpectrumWifiPhy distance: 1m
// index   MCS   width Rate (Mb/s) Tput (Mb/s) Received
//     0     0      20       6.5     5.96219    5063
//     1     1      20        13     11.9491   10147
//     2     2      20      19.5     17.9184   15216
//     3     3      20        26     23.9253   20317
//     ...
//
// selection of index values 32-63 will result in MCS selection 8-15
// involving two spatial streams

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("WifiSpectrumSaturationExample");

int main (int argc, char *argv[])
{
  double distance = 1;
  double simulationTime = 10; //seconds
  uint16_t index = 256;
  uint32_t channelWidth = 0;
  std::string wifiType = "ns3::SpectrumWifiPhy";
  std::string errorModelType = "ns3::NistErrorRateModel";
  bool enablePcap = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("simulationTime", "Simulation time in seconds", simulationTime);
  cmd.AddValue ("distance", "meters separation between nodes", distance);
  cmd.AddValue ("index", "restrict index to single value between 0 and 63", index);
  cmd.AddValue ("wifiType", "select ns3::SpectrumWifiPhy or ns3::YansWifiPhy", wifiType);
  cmd.AddValue ("errorModelType", "select ns3::NistErrorRateModel or ns3::YansErrorRateModel", errorModelType);
  cmd.AddValue ("enablePcap", "enable pcap output", enablePcap);
  cmd.Parse (argc,argv);

  uint16_t startIndex = 0;
  uint16_t stopIndex = 63;
  if (index < 64)
    {
      startIndex = index;
      stopIndex = index;
    }

  std::cout << "wifiType: " << wifiType << " distance: " << distance << "m" << std::endl;
  std::cout << std::setw (5) << "index" <<
    std::setw (6) << "MCS" <<
    std::setw (8) << "width" <<
    std::setw (12) << "Rate (Mb/s)" <<
    std::setw (12) << "Tput (Mb/s)" <<
    std::setw (10) << "Received " <<
    std::endl;
  for (uint16_t i = startIndex; i <= stopIndex; i++)
    {
      uint32_t payloadSize;
      payloadSize = 1472; // 1500 bytes IPv4

      NodeContainer wifiStaNode;
      wifiStaNode.Create (1);
      NodeContainer wifiApNode;
      wifiApNode.Create (1);

      YansWifiPhyHelper phy;
      SpectrumWifiPhyHelper spectrumPhy;
      if (wifiType == "ns3::YansWifiPhy")
        {
          YansWifiChannelHelper channel;
          channel.AddPropagationLoss ("ns3::FriisPropagationLossModel");
          channel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
          phy.SetChannel (channel.Create ());
          phy.Set ("TxPowerStart", DoubleValue (1));
          phy.Set ("TxPowerEnd", DoubleValue (1));

          if (i > 31 && i <= 39)
            {
              phy.Set ("Antennas", UintegerValue (2));
              phy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              phy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
          else if (i > 39 && i <= 47)
            {
              phy.Set ("Antennas", UintegerValue (2));
              phy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              phy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
          else if (i > 47 && i <= 55)
            {
              phy.Set ("Antennas", UintegerValue (2));
              phy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              phy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
          else if (i > 55 && i <= 63)
            {
              phy.Set ("Antennas", UintegerValue (2));
              phy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              phy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
        }
      else if (wifiType == "ns3::SpectrumWifiPhy")
        {
          Ptr<MultiModelSpectrumChannel> spectrumChannel
            = CreateObject<MultiModelSpectrumChannel> ();
          Ptr<FriisPropagationLossModel> lossModel
            = CreateObject<FriisPropagationLossModel> ();
          spectrumChannel->AddPropagationLossModel (lossModel);

          Ptr<ConstantSpeedPropagationDelayModel> delayModel
            = CreateObject<ConstantSpeedPropagationDelayModel> ();
          spectrumChannel->SetPropagationDelayModel (delayModel);

          spectrumPhy.SetChannel (spectrumChannel);
          spectrumPhy.SetErrorRateModel (errorModelType);
          spectrumPhy.Set ("TxPowerStart", DoubleValue (1));
          spectrumPhy.Set ("TxPowerEnd", DoubleValue (1));

          if (i > 31 && i <= 39)
            {
              spectrumPhy.Set ("Antennas", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
          else if (i > 39 && i <= 47)
            {
              spectrumPhy.Set ("Antennas", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
          else if (i > 47 && i <= 55)
            {
              spectrumPhy.Set ("Antennas", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
          else if (i > 55 && i <= 63)
            {
              spectrumPhy.Set ("Antennas", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
              spectrumPhy.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));
            }
        }
      else
        {
          NS_FATAL_ERROR ("Unsupported WiFi type " << wifiType);
        }

      WifiHelper wifi;
      wifi.SetStandard (WIFI_STANDARD_80211n);
      WifiMacHelper mac;

      Ssid ssid = Ssid ("ns380211n");

      double datarate = 0;
      StringValue DataRate;
      if (i == 0)
        {
          DataRate = StringValue ("HtMcs0");
          datarate = 6.5;
        }
      else if (i == 1)
        {
          DataRate = StringValue ("HtMcs1");
          datarate = 13;
        }
      else if (i == 2)
        {
          DataRate = StringValue ("HtMcs2");
          datarate = 19.5;
        }
      else if (i == 3)
        {
          DataRate = StringValue ("HtMcs3");
          datarate = 26;
        }
      else if (i == 4)
        {
          DataRate = StringValue ("HtMcs4");
          datarate = 39;
        }
      else if (i == 5)
        {
          DataRate = StringValue ("HtMcs5");
          datarate = 52;
        }
      else if (i == 6)
        {
          DataRate = StringValue ("HtMcs6");
          datarate = 58.5;
        }
      else if (i == 7)
        {
          DataRate = StringValue ("HtMcs7");
          datarate = 65;
        }
      else if (i == 8)
        {
          DataRate = StringValue ("HtMcs0");
          datarate = 7.2;
        }
      else if (i == 9)
        {
          DataRate = StringValue ("HtMcs1");
          datarate = 14.4;
        }
      else if (i == 10)
        {
          DataRate = StringValue ("HtMcs2");
          datarate = 21.7;
        }
      else if (i == 11)
        {
          DataRate = StringValue ("HtMcs3");
          datarate = 28.9;
        }
      else if (i == 12)
        {
          DataRate = StringValue ("HtMcs4");
          datarate = 43.3;
        }
      else if (i == 13)
        {
          DataRate = StringValue ("HtMcs5");
          datarate = 57.8;
        }
      else if (i == 14)
        {
          DataRate = StringValue ("HtMcs6");
          datarate = 65;
        }
      else if (i == 15)
        {
          DataRate = StringValue ("HtMcs7");
          datarate = 72.2;
        }
      else if (i == 16)
        {
          DataRate = StringValue ("HtMcs0");
          datarate = 13.5;
        }
      else if (i == 17)
        {
          DataRate = StringValue ("HtMcs1");
          datarate = 27;
        }
      else if (i == 18)
        {
          DataRate = StringValue ("HtMcs2");
          datarate = 40.5;
        }
      else if (i == 19)
        {
          DataRate = StringValue ("HtMcs3");
          datarate = 54;
        }
      else if (i == 20)
        {
          DataRate = StringValue ("HtMcs4");
          datarate = 81;
        }
      else if (i == 21)
        {
          DataRate = StringValue ("HtMcs5");
          datarate = 108;
        }
      else if (i == 22)
        {
          DataRate = StringValue ("HtMcs6");
          datarate = 121.5;
        }
      else if (i == 23)
        {
          DataRate = StringValue ("HtMcs7");
          datarate = 135;
        }
      else if (i == 24)
        {
          DataRate = StringValue ("HtMcs0");
          datarate = 15;
        }
      else if (i == 25)
        {
          DataRate = StringValue ("HtMcs1");
          datarate = 30;
        }
      else if (i == 26)
        {
          DataRate = StringValue ("HtMcs2");
          datarate = 45;
        }
      else if (i == 27)
        {
          DataRate = StringValue ("HtMcs3");
          datarate = 60;
        }
      else if (i == 28)
        {
          DataRate = StringValue ("HtMcs4");
          datarate = 90;
        }
      else if (i == 29)
        {
          DataRate = StringValue ("HtMcs5");
          datarate = 120;
        }
      else if (i == 30)
        {
          DataRate = StringValue ("HtMcs6");
          datarate = 135;
        }
      else if (i == 31)
        {
          DataRate = StringValue ("HtMcs7");
          datarate = 150;
        }
      else if (i == 32)
        {
          DataRate = StringValue ("HtMcs8");
          datarate = 13;
        }
      else if (i == 33)
        {
          DataRate = StringValue ("HtMcs9");
          datarate = 26;
        }
      else if (i == 34)
        {
          DataRate = StringValue ("HtMcs10");
          datarate = 39;
        }
      else if (i == 35)
        {
          DataRate = StringValue ("HtMcs11");
          datarate = 52;
        }
      else if (i == 36)
        {
          DataRate = StringValue ("HtMcs12");
          datarate = 78;
        }
      else if (i == 37)
        {
          DataRate = StringValue ("HtMcs13");
          datarate = 104;
        }
      else if (i == 38)
        {
          DataRate = StringValue ("HtMcs14");
          datarate = 117;
        }
      else if (i == 39)
        {
          DataRate = StringValue ("HtMcs15");
          datarate = 130;
        }
      else if (i == 40)
        {
          DataRate = StringValue ("HtMcs8");
          datarate = 14.4;
        }
      else if (i == 41)
        {
          DataRate = StringValue ("HtMcs9");
          datarate = 28.9;
        }
      else if (i == 42)
        {
          DataRate = StringValue ("HtMcs10");
          datarate = 43.3;
        }
      else if (i == 43)
        {
          DataRate = StringValue ("HtMcs11");
          datarate = 57.8;
        }
      else if (i == 44)
        {
          DataRate = StringValue ("HtMcs12");
          datarate = 86.7;
        }
      else if (i == 45)
        {
          DataRate = StringValue ("HtMcs13");
          datarate = 115.6;
        }
      else if (i == 46)
        {
          DataRate = StringValue ("HtMcs14");
          datarate = 130.3;
        }
      else if (i == 47)
        {
          DataRate = StringValue ("HtMcs15");
          datarate = 144.4;
        }
      else if (i == 48)
        {
          DataRate = StringValue ("HtMcs8");
          datarate = 27;
        }
      else if (i == 49)
        {
          DataRate = StringValue ("HtMcs9");
          datarate = 54;
        }
      else if (i == 50)
        {
          DataRate = StringValue ("HtMcs10");
          datarate = 81;
        }
      else if (i == 51)
        {
          DataRate = StringValue ("HtMcs11");
          datarate = 108;
        }
      else if (i == 52)
        {
          DataRate = StringValue ("HtMcs12");
          datarate = 162;
        }
      else if (i == 53)
        {
          DataRate = StringValue ("HtMcs13");
          datarate = 216;
        }
      else if (i == 54)
        {
          DataRate = StringValue ("HtMcs14");
          datarate = 243;
        }
      else if (i == 55)
        {
          DataRate = StringValue ("HtMcs15");
          datarate = 270;
        }
      else if (i == 56)
        {
          DataRate = StringValue ("HtMcs8");
          datarate = 30;
        }
      else if (i == 57)
        {
          DataRate = StringValue ("HtMcs9");
          datarate = 60;
        }
      else if (i == 58)
        {
          DataRate = StringValue ("HtMcs10");
          datarate = 90;
        }
      else if (i == 59)
        {
          DataRate = StringValue ("HtMcs11");
          datarate = 120;
        }
      else if (i == 60)
        {
          DataRate = StringValue ("HtMcs12");
          datarate = 180;
        }
      else if (i == 61)
        {
          DataRate = StringValue ("HtMcs13");
          datarate = 240;
        }
      else if (i == 62)
        {
          DataRate = StringValue ("HtMcs14");
          datarate = 270;
        }
      else if (i == 63)
        {
          DataRate = StringValue ("HtMcs15");
          datarate = 300;
        }
      else
        {
          NS_FATAL_ERROR ("Illegal index i " << i);
        }

      wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", DataRate,
                                    "ControlMode", DataRate);

      NetDeviceContainer staDevice;
      NetDeviceContainer apDevice;

      channelWidth = (i <= 15 || (i > 31 && i <= 47) ? 20 : 40);
      std::string channelStr = "{0, " + std::to_string (channelWidth) + ", BAND_5GHZ, 0}";

      if (wifiType == "ns3::YansWifiPhy")
        {
          mac.SetType ("ns3::StaWifiMac",
                       "Ssid", SsidValue (ssid));
          phy.Set ("ChannelSettings", StringValue (channelStr));

          staDevice = wifi.Install (phy, mac, wifiStaNode);
          mac.SetType ("ns3::ApWifiMac",
                       "Ssid", SsidValue (ssid));
          phy.Set ("ChannelSettings", StringValue (channelStr));
          apDevice = wifi.Install (phy, mac, wifiApNode);

        }
      else if (wifiType == "ns3::SpectrumWifiPhy")
        {
          mac.SetType ("ns3::StaWifiMac",
                       "Ssid", SsidValue (ssid));
          phy.Set ("ChannelSettings", StringValue (channelStr));
          staDevice = wifi.Install (spectrumPhy, mac, wifiStaNode);
          mac.SetType ("ns3::ApWifiMac",
                       "Ssid", SsidValue (ssid));
          phy.Set ("ChannelSettings", StringValue (channelStr));
          apDevice = wifi.Install (spectrumPhy, mac, wifiApNode);
        }

     if ((i <= 7) || (i > 31 && i <= 39))
        {
          Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (false));
        }
      else if ((i > 7 && i <= 15) || (i > 39 && i <= 47))
        {
          Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (true));
        }
      else if ((i > 15 && i <= 23) || (i > 47 && i <= 55))
        {
          Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (false));
        }
      else
        {
          Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (true));
        }

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
      uint16_t port = 9;
      UdpServerHelper server (port);
      ApplicationContainer serverApp = server.Install (wifiStaNode.Get (0));
      serverApp.Start (Seconds (0.0));
      serverApp.Stop (Seconds (simulationTime + 1));

      UdpClientHelper client (staNodeInterface.GetAddress (0), port);
      client.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
      client.SetAttribute ("Interval", TimeValue (Time ("0.0001"))); //packets/s
      client.SetAttribute ("PacketSize", UintegerValue (payloadSize));
      ApplicationContainer clientApp = client.Install (wifiApNode.Get (0));
      clientApp.Start (Seconds (1.0));
      clientApp.Stop (Seconds (simulationTime + 1));

      if (enablePcap)
        {
          phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
          std::stringstream ss;
          ss << "wifi-spectrum-saturation-example-" << i;
          phy.EnablePcap (ss.str (), apDevice);
        }

      Simulator::Stop (Seconds (simulationTime + 1));
      Simulator::Run ();

      double throughput;
      uint64_t totalPacketsThrough;
      totalPacketsThrough = DynamicCast<UdpServer> (serverApp.Get (0))->GetReceived ();
      throughput = totalPacketsThrough * payloadSize * 8 / (simulationTime * 1000000.0); //Mbit/s
      std::cout << std::setw (5) << i <<
        std::setw (6) << (i % 8) + 8 * (i / 32) <<
        std::setw (8) << channelWidth <<
        std::setw (10) << datarate <<
        std::setw (12) << throughput <<
        std::setw (8) << totalPacketsThrough <<
        std::endl;
      Simulator::Destroy ();
    }
  return 0;
}
