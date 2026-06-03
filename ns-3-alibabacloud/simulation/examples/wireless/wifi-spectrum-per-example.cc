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
#include "ns3/packet-sink-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/propagation-loss-model.h"

// This is a simple example of an IEEE 802.11n Wi-Fi network.
//
// The main use case is to enable and test SpectrumWifiPhy vs YansWifiPhy
// for packet error ratio
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
//    --udp:             UDP if set to 1, TCP otherwise [true]
//    --distance:        meters separation between nodes [50]
//    --index:           restrict index to single value between 0 and 31 [256]
//    --wifiType:        select ns3::SpectrumWifiPhy or ns3::YansWifiPhy [ns3::SpectrumWifiPhy]
//    --errorModelType:  select ns3::NistErrorRateModel or ns3::YansErrorRateModel [ns3::NistErrorRateModel]
//    --enablePcap:      enable pcap output [false]
//
// By default, the program will step through 32 index values, corresponding
// to the following MCS, channel width, and guard interval combinations:
//   index 0-7:    MCS 0-7, long guard interval, 20 MHz channel
//   index 8-15:   MCS 0-7, short guard interval, 20 MHz channel
//   index 16-23:  MCS 0-7, long guard interval, 40 MHz channel
//   index 24-31:  MCS 0-7, short guard interval, 40 MHz channel
// and send UDP for 10 seconds using each MCS, using the SpectrumWifiPhy and the
// NistErrorRateModel, at a distance of 50 meters.  The program outputs
// results such as:
//
// wifiType: ns3::SpectrumWifiPhy distance: 50m; time: 10; TxPower: 1 dBm (1.3 mW)
// index   MCS  Rate (Mb/s) Tput (Mb/s) Received Signal (dBm) Noise (dBm) SNR (dB)
//     0     0      6.50        5.77    7414      -79.71      -93.97       14.25
//     1     1     13.00       11.58   14892      -79.71      -93.97       14.25
//     2     2     19.50       17.39   22358      -79.71      -93.97       14.25
//     3     3     26.00       22.96   29521      -79.71      -93.97       14.25
//   ...
//

using namespace ns3;

// Global variables for use in callbacks.
double g_signalDbmAvg;
double g_noiseDbmAvg;
uint32_t g_samples;

void MonitorSniffRx (Ptr<const Packet> packet,
                     uint16_t channelFreqMhz,
                     WifiTxVector txVector,
                     MpduInfo aMpdu,
                     SignalNoiseDbm signalNoise,
                     uint16_t staId)

{
  g_samples++;
  g_signalDbmAvg += ((signalNoise.signal - g_signalDbmAvg) / g_samples);
  g_noiseDbmAvg += ((signalNoise.noise - g_noiseDbmAvg) / g_samples);
}

NS_LOG_COMPONENT_DEFINE ("WifiSpectrumPerExample");

int main (int argc, char *argv[])
{
  bool udp = true;
  double distance = 50;
  double simulationTime = 10; //seconds
  uint16_t index = 256;
  std::string wifiType = "ns3::SpectrumWifiPhy";
  std::string errorModelType = "ns3::NistErrorRateModel";
  bool enablePcap = false;
  const uint32_t tcpPacketSize = 1448;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("simulationTime", "Simulation time in seconds", simulationTime);
  cmd.AddValue ("udp", "UDP if set to 1, TCP otherwise", udp);
  cmd.AddValue ("distance", "meters separation between nodes", distance);
  cmd.AddValue ("index", "restrict index to single value between 0 and 31", index);
  cmd.AddValue ("wifiType", "select ns3::SpectrumWifiPhy or ns3::YansWifiPhy", wifiType);
  cmd.AddValue ("errorModelType", "select ns3::NistErrorRateModel or ns3::YansErrorRateModel", errorModelType);
  cmd.AddValue ("enablePcap", "enable pcap output", enablePcap);
  cmd.Parse (argc,argv);

  uint16_t startIndex = 0;
  uint16_t stopIndex = 31;
  if (index < 32)
    {
      startIndex = index;
      stopIndex = index;
    }

  std::cout << "wifiType: " << wifiType << " distance: " << distance << "m; time: " << simulationTime << "; TxPower: 1 dBm (1.3 mW)" << std::endl;
  std::cout << std::setw (5) << "index" <<
    std::setw (6) << "MCS" <<
    std::setw (13) << "Rate (Mb/s)" <<
    std::setw (12) << "Tput (Mb/s)" <<
    std::setw (10) << "Received " <<
    std::setw (12) << "Signal (dBm)" <<
    std::setw (12) << "Noise (dBm)" <<
    std::setw (9) << "SNR (dB)" <<
    std::endl;
  for (uint16_t i = startIndex; i <= stopIndex; i++)
    {
      uint32_t payloadSize;
      if (udp)
        {
          payloadSize = 972; // 1000 bytes IPv4
        }
      else
        {
          payloadSize = 1448; // 1500 bytes IPv6
          Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (payloadSize));
        }

      NodeContainer wifiStaNode;
      wifiStaNode.Create (1);
      NodeContainer wifiApNode;
      wifiApNode.Create (1);

      YansWifiPhyHelper phy;
      SpectrumWifiPhyHelper spectrumPhy;
      if (wifiType == "ns3::YansWifiPhy")
        {
          YansWifiChannelHelper channel;
          channel.AddPropagationLoss ("ns3::FriisPropagationLossModel",
                                      "Frequency", DoubleValue (5.180e9));
          channel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
          phy.SetChannel (channel.Create ());
          phy.Set ("TxPowerStart", DoubleValue (1)); // dBm (1.26 mW)
          phy.Set ("TxPowerEnd", DoubleValue (1));
        }
      else if (wifiType == "ns3::SpectrumWifiPhy")
        {
          Ptr<MultiModelSpectrumChannel> spectrumChannel
            = CreateObject<MultiModelSpectrumChannel> ();
          Ptr<FriisPropagationLossModel> lossModel
            = CreateObject<FriisPropagationLossModel> ();
          lossModel->SetFrequency (5.180e9);
          spectrumChannel->AddPropagationLossModel (lossModel);

          Ptr<ConstantSpeedPropagationDelayModel> delayModel
            = CreateObject<ConstantSpeedPropagationDelayModel> ();
          spectrumChannel->SetPropagationDelayModel (delayModel);

          spectrumPhy.SetChannel (spectrumChannel);
          spectrumPhy.SetErrorRateModel (errorModelType);
          spectrumPhy.Set ("TxPowerStart", DoubleValue (1)); // dBm  (1.26 mW)
          spectrumPhy.Set ("TxPowerEnd", DoubleValue (1));
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
      else
        {
          DataRate = StringValue ("HtMcs7");
          datarate = 150;
        }

      wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager","DataMode", DataRate,
                                    "ControlMode", DataRate);

      NetDeviceContainer staDevice;
      NetDeviceContainer apDevice;

      if (wifiType == "ns3::YansWifiPhy")
        {
          mac.SetType ("ns3::StaWifiMac",
                       "Ssid", SsidValue (ssid));
          phy.Set ("ChannelSettings", StringValue (std::string ("{0, ") + (i <= 15 ? "20" : "40")
                                                   + ", BAND_5GHZ, 0}"));
          staDevice = wifi.Install (phy, mac, wifiStaNode);
          mac.SetType ("ns3::ApWifiMac",
                       "Ssid", SsidValue (ssid));
          apDevice = wifi.Install (phy, mac, wifiApNode);

        }
      else if (wifiType == "ns3::SpectrumWifiPhy")
        {
          mac.SetType ("ns3::StaWifiMac",
                       "Ssid", SsidValue (ssid));
          phy.Set ("ChannelSettings", StringValue (std::string ("{0, ") + (i <= 15 ? "20" : "40")
                                                   + ", BAND_5GHZ, 0}"));
          staDevice = wifi.Install (spectrumPhy, mac, wifiStaNode);
          mac.SetType ("ns3::ApWifiMac",
                       "Ssid", SsidValue (ssid));
          apDevice = wifi.Install (spectrumPhy, mac, wifiApNode);
        }

      if (i <= 7)
        {
          Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (false));
        }
      else if (i > 7 && i <= 15)
        {
          Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (true));
        }
      else if (i > 15 && i <= 23)
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
          client.SetAttribute ("Interval", TimeValue (Time ("0.0001"))); //packets/s
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

      Config::ConnectWithoutContext ("/NodeList/0/DeviceList/*/Phy/MonitorSnifferRx", MakeCallback (&MonitorSniffRx));

      if (enablePcap)
        {
          phy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
          std::stringstream ss;
          ss << "wifi-spectrum-per-example-" << i;
          phy.EnablePcap (ss.str (), apDevice);
        }
      g_signalDbmAvg = 0;
      g_noiseDbmAvg = 0;
      g_samples = 0;

      Simulator::Stop (Seconds (simulationTime + 1));
      Simulator::Run ();

      double throughput = 0;
      uint64_t totalPacketsThrough = 0;
      if (udp)
        {
          //UDP
          totalPacketsThrough = DynamicCast<UdpServer> (serverApp.Get (0))->GetReceived ();
          throughput = totalPacketsThrough * payloadSize * 8 / (simulationTime * 1000000.0); //Mbit/s
        }
      else
        {
          //TCP
          uint64_t totalBytesRx = DynamicCast<PacketSink> (serverApp.Get (0))->GetTotalRx ();
          totalPacketsThrough = totalBytesRx / tcpPacketSize;
          throughput = totalBytesRx * 8 / (simulationTime * 1000000.0); //Mbit/s
        }
      std::cout << std::setw (5) << i <<
        std::setw (6) << (i % 8) <<
        std::setprecision (2) << std::fixed <<
        std::setw (10) << datarate <<
        std::setw (12) << throughput <<
        std::setw (8) << totalPacketsThrough;
      if (totalPacketsThrough > 0)
        {
          std::cout << std::setw (12) << g_signalDbmAvg <<
            std::setw (12) << g_noiseDbmAvg <<
            std::setw (12) << (g_signalDbmAvg - g_noiseDbmAvg) <<
            std::endl;
        }
      else
        {
          std::cout << std::setw (12) << "N/A" <<
            std::setw (12) << "N/A" <<
            std::setw (12) << "N/A" <<
            std::endl;
        }
      Simulator::Destroy ();
    }
  return 0;
}
