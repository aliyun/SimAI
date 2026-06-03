/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/wifi-net-device.h"
#include "ns3/qos-txop.h"
#include "ns3/wifi-mac.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/ht-configuration.h"

// This example shows how to configure mixed networks (i.e. mixed b/g and HT/non-HT) and how are performance in several scenarios.
//
// The example compares first g only and mixed b/g cases with various configurations depending on the following parameters:
// - protection mode that is configured on the AP;
// - whether short PPDU format is supported by the 802.11b station;
// - whether short slot time is supported by both the 802.11g station and the AP.
//
// The example then compares HT only and mixed HT/non-HT cases.
//
// The output results show that the presence of an 802.11b station strongly affects 802.11g performance.
// Protection mechanisms ensure that the NAV value of 802.11b stations is set correctly in case of 802.11g transmissions.
// In practice, those protection mechanism add a lot of overhead, resulting in reduced performance. CTS-To-Self introduces
// less overhead than Rts-Cts, but is not heard by hidden stations (and is thus generally only recommended as a protection
// mechanism for access points). Since short slot time is disabled once an 802.11b station enters the network, benefits from
// short slot time are only observed in a g only configuration.
//
// The user can also select the payload size and can choose either an UDP or a TCP connection.
// Example: ./ns3 run "wifi-mixed-network --isUdp=1"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("MixedNetwork");

/** Parameters */
struct Parameters
{
  std::string testName;             //!< Test name
  bool enableErpProtection;         //!< True to enable ERP protection
  std::string erpProtectionMode;    //!< ERP protection mode
  bool enableShortSlotTime;         //!< True to enable short slot time
  bool enableShortPhyPreamble;      //!< True to enable short PHY preamble
  WifiStandard apType;              //!< Wifi standard for AP
  uint32_t nWifiB;                  //!< Number of 802.11b stations
  bool bHasTraffic;                 //!< True if 802.11b stations generate traffic
  uint32_t nWifiG;                  //!< Number of 802.11g stations
  bool gHasTraffic;                 //!< True if 802.11g stations generate traffic
  uint32_t nWifiN;                  //!< Number of 802.11n stations
  bool nHasTraffic;                 //!< True if 802.11n stations generate traffic
  bool isUdp;                       //!< True to generate UDP traffic
  uint32_t payloadSize;             //!< Payload size in bytes
  double simulationTime;            //!< Simulation time in seconds
};

class Experiment
{
public:
  Experiment ();
  /**
   * Run an experiment with the given parameters
   * \param params the given parameters
   * \return the throughput
   */
  double Run (Parameters params);
};

Experiment::Experiment ()
{
}

double
Experiment::Run (Parameters params)
{
  std::string apTypeString;
  if (params.apType == WIFI_STANDARD_80211g)
    {
      apTypeString = "WIFI_STANDARD_80211g";
    }
  else if (params.apType == WIFI_STANDARD_80211n)
    {
      apTypeString = "WIFI_STANDARD_80211n_2_4GHZ";
    }

  std::cout << "Run: " << params.testName
            << "\n\t enableErpProtection=" << params.enableErpProtection
            << "\n\t erpProtectionMode=" << params.erpProtectionMode
            << "\n\t enableShortSlotTime=" << params.enableShortSlotTime
            << "\n\t enableShortPhyPreamble=" << params.enableShortPhyPreamble
            << "\n\t apType=" << apTypeString
            << "\n\t nWifiB=" << params.nWifiB
            << "\n\t bHasTraffic=" << params.bHasTraffic
            << "\n\t nWifiG=" << params.nWifiG
            << "\n\t gHasTraffic=" << params.gHasTraffic
            << "\n\t nWifiN=" << params.nWifiN
            << "\n\t nHasTraffic=" << params.nHasTraffic
            << std::endl;

  Config::SetDefault ("ns3::WifiRemoteStationManager::ErpProtectionMode", StringValue (params.erpProtectionMode));

  double throughput = 0;
  uint32_t nWifiB = params.nWifiB;
  uint32_t nWifiG = params.nWifiG;
  uint32_t nWifiN = params.nWifiN;
  double simulationTime = params.simulationTime;
  uint32_t payloadSize = params.payloadSize;

  NodeContainer wifiBStaNodes;
  wifiBStaNodes.Create (nWifiB);
  NodeContainer wifiGStaNodes;
  wifiGStaNodes.Create (nWifiG);
  NodeContainer wifiNStaNodes;
  wifiNStaNodes.Create (nWifiN);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  channel.AddPropagationLoss ("ns3::RangePropagationLossModel");

  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiHelper wifi;
  wifi.SetRemoteStationManager ("ns3::IdealWifiManager");

  // 802.11b STA
  wifi.SetStandard (WIFI_STANDARD_80211b);

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");

  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ShortSlotTimeSupported", BooleanValue (params.enableShortSlotTime));

  // Configure the PHY preamble type: long or short
  phy.Set ("ShortPlcpPreambleSupported", BooleanValue (params.enableShortPhyPreamble));

  NetDeviceContainer bStaDevice;
  bStaDevice = wifi.Install (phy, mac, wifiBStaNodes);

  // 802.11b/g STA
  wifi.SetStandard (WIFI_STANDARD_80211g);
  NetDeviceContainer gStaDevice;
  gStaDevice = wifi.Install (phy, mac, wifiGStaNodes);

  // 802.11b/g/n STA
  wifi.SetStandard (WIFI_STANDARD_80211n);
  NetDeviceContainer nStaDevice;
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "BE_BlockAckThreshold", UintegerValue (2),
               "ShortSlotTimeSupported", BooleanValue (params.enableShortSlotTime));
  nStaDevice = wifi.Install (phy, mac, wifiNStaNodes);

  // AP
  NetDeviceContainer apDevice;
  wifi.SetStandard (params.apType);
  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid),
               "EnableBeaconJitter", BooleanValue (false),
               "BE_BlockAckThreshold", UintegerValue (2),
               "EnableNonErpProtection", BooleanValue (params.enableErpProtection),
               "ShortSlotTimeSupported", BooleanValue (params.enableShortSlotTime));
  apDevice = wifi.Install (phy, mac, wifiApNode);

  // Set TXOP limit
  if (params.apType == WIFI_STANDARD_80211n)
    {
      Ptr<NetDevice> dev = wifiApNode.Get (0)->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
      PointerValue ptr;
      wifi_mac->GetAttribute ("BE_Txop", ptr);
      Ptr<QosTxop> edca = ptr.Get<QosTxop> ();
      edca->SetTxopLimit (MicroSeconds (3008));
    }
  if (nWifiN > 0)
    {
      Ptr<NetDevice> dev = wifiNStaNodes.Get (0)->GetDevice (0);
      Ptr<WifiNetDevice> wifi_dev = DynamicCast<WifiNetDevice> (dev);
      Ptr<WifiMac> wifi_mac = wifi_dev->GetMac ();
      PointerValue ptr;
      wifi_mac->GetAttribute ("BE_Txop", ptr);
      Ptr<QosTxop> edca = ptr.Get<QosTxop> ();
      edca->SetTxopLimit (MicroSeconds (3008));
    }

  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/BE_MaxAmpduSize", UintegerValue (0)); //Disable A-MPDU

  // Define mobility model
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();

  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  for (uint32_t i = 0; i < nWifiB; i++)
    {
      positionAlloc->Add (Vector (5.0, 0.0, 0.0));
    }
  for (uint32_t i = 0; i < nWifiG; i++)
    {
      positionAlloc->Add (Vector (0.0, 5.0, 0.0));
    }
  for (uint32_t i = 0; i < nWifiN; i++)
    {
      positionAlloc->Add (Vector (0.0, 0.0, 5.0));
    }

  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiBStaNodes);
  mobility.Install (wifiGStaNodes);
  mobility.Install (wifiNStaNodes);

  // Internet stack
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiBStaNodes);
  stack.Install (wifiGStaNodes);
  stack.Install (wifiNStaNodes);

  Ipv4AddressHelper address;
  address.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer bStaInterface;
  bStaInterface = address.Assign (bStaDevice);
  Ipv4InterfaceContainer gStaInterface;
  gStaInterface = address.Assign (gStaDevice);
  Ipv4InterfaceContainer nStaInterface;
  nStaInterface = address.Assign (nStaDevice);
  Ipv4InterfaceContainer ApInterface;
  ApInterface = address.Assign (apDevice);

  // Setting applications
  if (params.isUdp)
    {
      uint16_t port = 9;
      UdpServerHelper server (port);
      ApplicationContainer serverApp = server.Install (wifiApNode);
      serverApp.Start (Seconds (0.0));
      serverApp.Stop (Seconds (simulationTime + 1));

      UdpClientHelper client (ApInterface.GetAddress (0), port);
      client.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
      client.SetAttribute ("Interval", TimeValue (Time ("0.0002"))); //packets/s
      client.SetAttribute ("PacketSize", UintegerValue (payloadSize));

      ApplicationContainer clientApps;
      if (params.bHasTraffic)
        {
          clientApps.Add (client.Install (wifiBStaNodes));
        }
      if (params.gHasTraffic)
        {
          clientApps.Add (client.Install (wifiGStaNodes));
        }
      if (params.nHasTraffic)
        {
          clientApps.Add (client.Install (wifiNStaNodes));
        }
      clientApps.Start (Seconds (1.0));
      clientApps.Stop (Seconds (simulationTime + 1));

      Simulator::Stop (Seconds (simulationTime + 1));
      Simulator::Run ();

      uint64_t totalPacketsThrough = DynamicCast<UdpServer> (serverApp.Get (0))->GetReceived ();
      throughput = totalPacketsThrough * payloadSize * 8 / (simulationTime * 1000000.0);
    }
  else
    {
      uint16_t port = 50000;
      Address localAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
      PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", localAddress);

      ApplicationContainer serverApp = packetSinkHelper.Install (wifiApNode.Get (0));
      serverApp.Start (Seconds (0.0));
      serverApp.Stop (Seconds (simulationTime + 1));

      OnOffHelper onoff ("ns3::TcpSocketFactory", Ipv4Address::GetAny ());
      onoff.SetAttribute ("OnTime",  StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
      onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
      onoff.SetAttribute ("PacketSize", UintegerValue (payloadSize));
      onoff.SetAttribute ("DataRate", DataRateValue (150000000)); //bit/s

      AddressValue remoteAddress (InetSocketAddress (ApInterface.GetAddress (0), port));
      onoff.SetAttribute ("Remote", remoteAddress);

      ApplicationContainer clientApps;
      if (params.bHasTraffic)
        {
          clientApps.Add (onoff.Install (wifiBStaNodes));
        }
      if (params.gHasTraffic)
        {
          clientApps.Add (onoff.Install (wifiGStaNodes));
        }
      if (params.nHasTraffic)
        {
          clientApps.Add (onoff.Install (wifiNStaNodes));
        }
      clientApps.Start (Seconds (1.0));
      clientApps.Stop (Seconds (simulationTime + 1));

      Simulator::Stop (Seconds (simulationTime + 1));
      Simulator::Run ();

      uint64_t totalPacketsThrough = DynamicCast<PacketSink> (serverApp.Get (0))->GetTotalRx ();
      throughput += totalPacketsThrough * 8 / (simulationTime * 1000000.0);
    }
  Simulator::Destroy ();
  return throughput;
}

int main (int argc, char *argv[])
{
  Parameters params;
  params.testName = "";
  params.enableErpProtection = false;
  params.erpProtectionMode = "Cts-To-Self";
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = false;
  params.apType = WIFI_STANDARD_80211g;
  params.nWifiB = 0;
  params.bHasTraffic = false;
  params.nWifiG = 1;
  params.gHasTraffic = true;
  params.nWifiN = 0;
  params.nHasTraffic = false;
  params.isUdp = true;
  params.payloadSize = 1472; //bytes
  params.simulationTime = 10; //seconds

  bool verifyResults = 0; //used for regression

  CommandLine cmd (__FILE__);
  cmd.AddValue ("payloadSize", "Payload size in bytes", params.payloadSize);
  cmd.AddValue ("simulationTime", "Simulation time in seconds", params.simulationTime);
  cmd.AddValue ("isUdp", "UDP if set to 1, TCP otherwise", params.isUdp);
  cmd.AddValue ("verifyResults", "Enable/disable results verification at the end of the simulation", verifyResults);
  cmd.Parse (argc, argv);

  Experiment experiment;
  double throughput = 0;

  params.testName = "g only with all g features disabled";
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 22.5 || throughput > 23.5))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "g only with short slot time enabled";
  params.enableErpProtection = false;
  params.enableShortSlotTime = true;
  params.enableShortPhyPreamble = false;
  params.nWifiB = 0;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 29 || throughput > 30))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed b/g with all g features disabled";
  params.enableErpProtection = false;
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = false;
  params.nWifiB = 1;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 22.5 || throughput > 23.5))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed b/g with short plcp preamble enabled";
  params.enableErpProtection = false;
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = true;
  params.nWifiB = 1;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 22.5 || throughput > 23.5))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed b/g with short slot time enabled using RTS-CTS protection";
  params.enableErpProtection = true;
  params.erpProtectionMode = "Rts-Cts";
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = false;
  params.nWifiB = 1;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 19 || throughput > 20))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed b/g with short plcp preamble enabled using RTS-CTS protection";
  params.enableErpProtection = true;
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = true;
  params.nWifiB = 1;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 19 || throughput > 20))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed b/g with short slot time enabled using CTS-TO-SELF protection";
  params.enableErpProtection = true;
  params.erpProtectionMode = "Cts-To-Self";
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = false;
  params.nWifiB = 1;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 20.5 || throughput > 21.5))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed b/g with short plcp preamble enabled using CTS-TO-SELF protection";
  params.enableErpProtection = true;
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = true;
  params.nWifiB = 1;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 20.5 || throughput > 21.5))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "HT only";
  params.enableErpProtection = false;
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = false;
  params.apType = WIFI_STANDARD_80211n;
  params.nWifiB = 0;
  params.bHasTraffic = false;
  params.nWifiG = 0;
  params.gHasTraffic = false;
  params.nWifiN = 1;
  params.nHasTraffic = true;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 44 || throughput > 45))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  params.testName = "Mixed HT/non-HT";
  params.enableErpProtection = false;
  params.enableShortSlotTime = false;
  params.enableShortPhyPreamble = false;
  params.apType = WIFI_STANDARD_80211n;
  params.nWifiB = 0;
  params.bHasTraffic = false;
  params.nWifiG = 1;
  params.gHasTraffic = false;
  params.nWifiN = 1;
  params.nHasTraffic = true;
  throughput = experiment.Run (params);
  if (verifyResults && (throughput < 44 || throughput > 45))
    {
      NS_LOG_ERROR ("Obtained throughput " << throughput << " is not in the expected boundaries!");
      exit (1);
    }
  std::cout << "Throughput: " << throughput << " Mbit/s \n" << std::endl;

  return 0;
}
