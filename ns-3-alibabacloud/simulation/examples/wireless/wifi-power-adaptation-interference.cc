/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Universidad de la República - Uruguay
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
 * Author: Matias Richart <mrichart@fing.edu.uy>
 */

/**
 * This example program is designed to illustrate the behavior of three
 * power/rate-adaptive WiFi rate controls; namely, ns3::ParfWifiManager,
 * ns3::AparfWifiManager and ns3::RrpaaWifiManager.
 *
 * This simulation consist of 4 nodes, two APs and two STAs.
 * The APs generates UDP traffic with a CBR of 54 Mbps to the STAs.
 * The APa use any power and rate control mechanism, and the STAs use only
 * Minstrel rate control.
 * The STAs can be configured to be at any distance from the APs.
 *
 * The objective is to test power and rate control in the links with
 * interference from the other link.
 *
 * The output consists of:
 * - A plot of average throughput vs. time.
 * - A plot of average transmit power vs. time.
 * - Plots for the percentage of time the APs are in each MAC state (IDLE, TX, RX, BUSY)
 * - If enabled, the changes of power and rate to standard output.
 * - If enabled, the average throughput, delay, jitter and tx opportunity for the total simulation time.
 *
 * Example usage:
 * \code{.sh}
 *   ./ns3 run "wifi-power-adaptation-interference --manager=ns3::AparfWifiManager --outputFileName=aparf"
 * \endcode
 *
 * Another example (changing STAs position):
 * \code{.sh}
 *   ./ns3 run "wifi-power-adaptation-interference --manager=ns3::AparfWifiManager --outputFileName=aparf --STA1_x=5 --STA2_x=205"
 * \endcode
 *
 * To enable the log of rate and power changes:
 * \code{.sh}
 *   export NS_LOG=PowerAdaptationInterference=level_info
 * \endcode
 */

#include "ns3/gnuplot.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/uinteger.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"

using namespace ns3;
using namespace std;

NS_LOG_COMPONENT_DEFINE ("PowerAdaptationInterference");

/// Packet size generated at the AP.
static const uint32_t packetSize = 1420;

/**
 * \brief Class to collect node statistics.
 */
class NodeStatistics
{
public:
  /**
   * \brief Constructor.
   * 
   * \param aps Access points
   * \param stas WiFi Stations.
   */
  NodeStatistics (NetDeviceContainer aps, NetDeviceContainer stas);

  /**
   * \brief Collects the statistics at a given time.
   * 
   * \param time Time at which the statistics are collected.
   */
  void CheckStatistics (double time);

  /**
   * \brief Callback called by WifiNetDevice/Phy/PhyTxBegin. 
   * 
   * \param path The trace path.
   * \param packet The sent packet.
   * \param powerW The Tx power.
   */
  void PhyCallback (std::string path, Ptr<const Packet> packet, double powerW);
  /**
   * \brief Callback called by PacketSink/Rx.
   * 
   * \param path The trace path.
   * \param packet The received packet.
   * \param from The sender address.
   */
  void RxCallback (std::string path, Ptr<const Packet> packet, const Address &from);
  /**
   * \brief Callback called by WifiNetDevice/RemoteStationManager/x/PowerChange.
   * 
   * \param path The trace path.
   * \param oldPower Old Tx power.
   * \param newPower Actual Tx power.
   * \param dest Destination of the transmission.
   */
  void PowerCallback (std::string path, double oldPower, double newPower, Mac48Address dest);
  /**
   * \brief Callback called by WifiNetDevice/RemoteStationManager/x/RateChange.
   * 
   * \param path The trace path.
   * \param oldRate Old rate.
   * \param newRate Actual rate.
   * \param dest Destination of the transmission.
   */
  void RateCallback (std::string path, DataRate oldRate, DataRate newRate, Mac48Address dest);
  /**
   * \brief Callback called by YansWifiPhy/State/State.
   * 
   * \param path The trace path.
   * \param init Time when the state started.
   * \param duration Amount of time we've been in (or will be in) the state.
   * \param state The state.
   */
  void StateCallback (std::string path, Time init, Time duration, WifiPhyState state);

  /**
   * \brief Get the Throughput output data
   * 
   * \return the Throughput output data.
   */
  Gnuplot2dDataset GetDatafile ();
  /**
   * \brief Get the Power output data.
   * 
   * \return the Power output data.
   */
  Gnuplot2dDataset GetPowerDatafile ();
  /**
   * \brief Get the IDLE state output data.
   * 
   * \return the IDLE state output data.
   */
  Gnuplot2dDataset GetIdleDatafile ();
  /**
   * \brief Get the BUSY state output data.
   * 
   * \return the BUSY state output data.
   */
  Gnuplot2dDataset GetBusyDatafile ();
  /**
   * \brief Get the TX state output data.
   * 
   * \return the TX state output data.
   */
  Gnuplot2dDataset GetTxDatafile ();
  /**
   * \brief Get the RX state output data.
   * 
   * \return the RX state output data.
   */
  Gnuplot2dDataset GetRxDatafile ();

  /**
   * \brief Get the Busy time.
   * 
   * \return the busy time. 
   */
  double GetBusyTime ();

private:
  /// Time, DataRate pair vector.
  typedef std::vector<std::pair<Time, DataRate> > TxTime;
  /**
   * \brief Setup the WifiPhy object.
   * 
   * \param phy The WifiPhy to setup.
   */
  void SetupPhy (Ptr<WifiPhy> phy);
  /**
   * \brief Get the time at which a given datarate has been recorded.
   * 
   * \param rate The datarate to search.
   * \return the time.
   */
  Time GetCalcTxTime (DataRate rate);

  std::map<Mac48Address, double> m_currentPower;  //!< Current Tx power for each sender.
  std::map<Mac48Address, DataRate> m_currentRate; //!< Current Tx rate for each sender.
  uint32_t m_bytesTotal;  //!< Number of received bytes.
  double m_totalEnergy;   //!< Energy used.
  double m_totalTime;     //!< Time spent.
  double busyTime;        //!< BUSY time.
  double idleTime;        //!< IDLE time.
  double txTime;          //!< TX time.
  double rxTime;          //!< RX time.
  double m_totalBusyTime; //!< Total time in BUSY state.
  double m_totalIdleTime; //!< Total time in IDLE state.
  double m_totalTxTime;   //!< Total time in TX state.
  double m_totalRxTime;   //!< Total time in RX state.
  TxTime m_timeTable;     //!< Time, DataRate table.
  Gnuplot2dDataset m_output;        //!< Throughput output data.
  Gnuplot2dDataset m_output_power;  //!< Power output data.
  Gnuplot2dDataset m_output_idle;   //!< IDLE output data.
  Gnuplot2dDataset m_output_busy;   //!< BUSY output data.
  Gnuplot2dDataset m_output_rx;     //!< RX output data.
  Gnuplot2dDataset m_output_tx;     //!< TX output data.
};

NodeStatistics::NodeStatistics (NetDeviceContainer aps, NetDeviceContainer stas)
{
  Ptr<NetDevice> device = aps.Get (0);
  Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice> (device);
  Ptr<WifiPhy> phy = wifiDevice->GetPhy ();
  SetupPhy (phy);
  DataRate dataRate = DataRate (phy->GetDefaultMode ().GetDataRate (phy->GetChannelWidth ()));
  double power = phy->GetTxPowerEnd ();
  for (uint32_t j = 0; j < stas.GetN (); j++)
    {
      Ptr<NetDevice> staDevice = stas.Get (j);
      Ptr<WifiNetDevice> wifiStaDevice = DynamicCast<WifiNetDevice> (staDevice);
      Mac48Address addr = wifiStaDevice->GetMac ()->GetAddress ();
      m_currentPower[addr] = power;
      m_currentRate[addr] = dataRate;
    }
  m_currentRate[Mac48Address ("ff:ff:ff:ff:ff:ff")] = dataRate;
  m_totalEnergy = 0;
  m_totalTime = 0;
  busyTime = 0;
  idleTime = 0;
  txTime = 0;
  rxTime = 0;
  m_totalBusyTime = 0;
  m_totalIdleTime = 0;
  m_totalTxTime = 0;
  m_totalRxTime = 0;
  m_bytesTotal = 0;
  m_output.SetTitle ("Throughput Mbits/s");
  m_output_idle.SetTitle ("Idle Time");
  m_output_busy.SetTitle ("Busy Time");
  m_output_rx.SetTitle ("RX Time");
  m_output_tx.SetTitle ("TX Time");
}

void
NodeStatistics::SetupPhy (Ptr<WifiPhy> phy)
{
  for (const auto & mode : phy->GetModeList ())
    {
      WifiTxVector txVector;
      txVector.SetMode (mode);
      txVector.SetPreambleType (WIFI_PREAMBLE_LONG);
      txVector.SetChannelWidth (phy->GetChannelWidth ());
      DataRate dataRate = DataRate (mode.GetDataRate (phy->GetChannelWidth ()));
      Time time = phy->CalculateTxDuration (packetSize, txVector, phy->GetPhyBand ());
      NS_LOG_DEBUG (mode.GetUniqueName () << " " << time.GetSeconds () << " " << dataRate);
      m_timeTable.push_back (std::make_pair (time, dataRate));
    }
}

Time
NodeStatistics::GetCalcTxTime (DataRate rate)
{
  for (TxTime::const_iterator i = m_timeTable.begin (); i != m_timeTable.end (); i++)
    {
      if (rate == i->second)
        {
          return i->first;
        }
    }
  NS_ASSERT (false);
  return Seconds (0);
}

void
NodeStatistics::PhyCallback (std::string path, Ptr<const Packet> packet, double powerW)
{
  WifiMacHeader head;
  packet->PeekHeader (head);
  Mac48Address dest = head.GetAddr1 ();

  if (head.GetType () == WIFI_MAC_DATA)
    {
      m_totalEnergy += pow (10.0, m_currentPower[dest] / 10.0) * GetCalcTxTime (m_currentRate[dest]).GetSeconds ();
      m_totalTime += GetCalcTxTime (m_currentRate[dest]).GetSeconds ();
    }
}

void
NodeStatistics::PowerCallback (std::string path, double oldPower, double newPower, Mac48Address dest)
{
  m_currentPower[dest] = newPower;
}

void
NodeStatistics::RateCallback (std::string path, DataRate oldRate, DataRate newRate, Mac48Address dest)
{
  m_currentRate[dest] = newRate;
}

void
NodeStatistics::StateCallback (std::string path, Time init, Time duration, WifiPhyState state)
{
  if (state == WifiPhyState::CCA_BUSY)
    {
      busyTime += duration.GetSeconds ();
      m_totalBusyTime += duration.GetSeconds ();
    }
  else if (state == WifiPhyState::IDLE)
    {
      idleTime += duration.GetSeconds ();
      m_totalIdleTime += duration.GetSeconds ();
    }
  else if (state == WifiPhyState::TX)
    {
      txTime += duration.GetSeconds ();
      m_totalTxTime += duration.GetSeconds ();
    }
  else if (state == WifiPhyState::RX)
    {
      rxTime += duration.GetSeconds ();
      m_totalRxTime += duration.GetSeconds ();
    }
}

void
NodeStatistics::RxCallback (std::string path, Ptr<const Packet> packet, const Address &from)
{
  m_bytesTotal += packet->GetSize ();
}

void
NodeStatistics::CheckStatistics (double time)
{
  double mbs = ((m_bytesTotal * 8.0) / (1000000 * time));
  m_bytesTotal = 0;
  double atp = m_totalEnergy / time;
  m_totalEnergy = 0;
  m_totalTime = 0;
  m_output_power.Add ((Simulator::Now ()).GetSeconds (), atp);
  m_output.Add ((Simulator::Now ()).GetSeconds (), mbs);

  m_output_idle.Add ((Simulator::Now ()).GetSeconds (), idleTime * 100);
  m_output_busy.Add ((Simulator::Now ()).GetSeconds (), busyTime * 100);
  m_output_tx.Add ((Simulator::Now ()).GetSeconds (), txTime * 100);
  m_output_rx.Add ((Simulator::Now ()).GetSeconds (), rxTime * 100);
  busyTime = 0;
  idleTime = 0;
  txTime = 0;
  rxTime = 0;

  Simulator::Schedule (Seconds (time), &NodeStatistics::CheckStatistics, this, time);
}

Gnuplot2dDataset
NodeStatistics::GetDatafile ()
{
  return m_output;
}

Gnuplot2dDataset
NodeStatistics::GetPowerDatafile ()
{
  return m_output_power;
}

Gnuplot2dDataset
NodeStatistics::GetIdleDatafile ()
{
  return m_output_idle;
}

Gnuplot2dDataset
NodeStatistics::GetBusyDatafile ()
{
  return m_output_busy;
}

Gnuplot2dDataset
NodeStatistics::GetRxDatafile ()
{
  return m_output_rx;
}

Gnuplot2dDataset
NodeStatistics::GetTxDatafile ()
{
  return m_output_tx;
}

double
NodeStatistics::GetBusyTime ()
{
  return m_totalBusyTime + m_totalRxTime;
}

void PowerCallback (std::string path, double oldPower, double newPower, Mac48Address dest)
{
  NS_LOG_INFO ((Simulator::Now ()).GetSeconds () << " " << dest << " Old power=" << oldPower << " New power=" << newPower);
}

void RateCallback (std::string path, DataRate oldRate, DataRate newRate, Mac48Address dest)
{
  NS_LOG_INFO ((Simulator::Now ()).GetSeconds () << " " << dest << " Old rate=" << oldRate << " New rate=" <<  newRate);
}

int main (int argc, char *argv[])
{
  //LogComponentEnable("ConstantRateWifiManager", LOG_LEVEL_FUNCTION);

  double maxPower = 17;
  double minPower = 0;
  uint32_t powerLevels = 18;

  uint32_t rtsThreshold = 2346;
  std::string manager = "ns3::ParfWifiManager";
  std::string outputFileName = "parf";
  int ap1_x = 0;
  int ap1_y = 0;
  int sta1_x = 10;
  int sta1_y = 0;
  int ap2_x = 200;
  int ap2_y = 0;
  int sta2_x = 180;
  int sta2_y = 0;
  uint32_t simuTime = 100;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("manager", "PRC Manager", manager);
  cmd.AddValue ("rtsThreshold", "RTS threshold", rtsThreshold);
  cmd.AddValue ("outputFileName", "Output filename", outputFileName);
  cmd.AddValue ("simuTime", "Total simulation time (sec)", simuTime);
  cmd.AddValue ("maxPower", "Maximum available transmission level (dbm).", maxPower);
  cmd.AddValue ("minPower", "Minimum available transmission level (dbm).", minPower);
  cmd.AddValue ("powerLevels", "Number of transmission power levels available between "
                "TxPowerStart and TxPowerEnd included.", powerLevels);
  cmd.AddValue ("AP1_x", "Position of AP1 in x coordinate", ap1_x);
  cmd.AddValue ("AP1_y", "Position of AP1 in y coordinate", ap1_y);
  cmd.AddValue ("STA1_x", "Position of STA1 in x coordinate", sta1_x);
  cmd.AddValue ("STA1_y", "Position of STA1 in y coordinate", sta1_y);
  cmd.AddValue ("AP2_x", "Position of AP2 in x coordinate", ap2_x);
  cmd.AddValue ("AP2_y", "Position of AP2 in y coordinate", ap2_y);
  cmd.AddValue ("STA2_x", "Position of STA2 in x coordinate", sta2_x);
  cmd.AddValue ("STA2_y", "Position of STA2 in y coordinate", sta2_y);
  cmd.Parse (argc, argv);

  //Define the APs
  NodeContainer wifiApNodes;
  wifiApNodes.Create (2);

  //Define the STAs
  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (2);

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211a);
  WifiMacHelper wifiMac;
  YansWifiPhyHelper wifiPhy;
  YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();

  wifiPhy.SetChannel (wifiChannel.Create ());

  NetDeviceContainer wifiApDevices;
  NetDeviceContainer wifiStaDevices;
  NetDeviceContainer wifiDevices;

  //Configure the STA nodes
  wifi.SetRemoteStationManager ("ns3::AarfWifiManager", "RtsCtsThreshold", UintegerValue (rtsThreshold));
  wifiPhy.Set ("TxPowerStart", DoubleValue (maxPower));
  wifiPhy.Set ("TxPowerEnd", DoubleValue (maxPower));

  Ssid ssid = Ssid ("AP0");
  wifiMac.SetType ("ns3::StaWifiMac",
                   "Ssid", SsidValue (ssid),
                   "MaxMissedBeacons", UintegerValue (1000));
  wifiStaDevices.Add (wifi.Install (wifiPhy, wifiMac, wifiStaNodes.Get (0)));

  ssid = Ssid ("AP1");
  wifiMac.SetType ("ns3::StaWifiMac",
                   "Ssid", SsidValue (ssid));
  wifiStaDevices.Add (wifi.Install (wifiPhy, wifiMac, wifiStaNodes.Get (1)));

  //Configure the AP nodes
  wifi.SetRemoteStationManager (manager, "DefaultTxPowerLevel", UintegerValue (powerLevels - 1), "RtsCtsThreshold", UintegerValue (rtsThreshold));
  wifiPhy.Set ("TxPowerStart", DoubleValue (minPower));
  wifiPhy.Set ("TxPowerEnd", DoubleValue (maxPower));
  wifiPhy.Set ("TxPowerLevels", UintegerValue (powerLevels));

  ssid = Ssid ("AP0");
  wifiMac.SetType ("ns3::ApWifiMac",
                   "Ssid", SsidValue (ssid));
  wifiApDevices.Add (wifi.Install (wifiPhy, wifiMac, wifiApNodes.Get (0)));

  ssid = Ssid ("AP1");
  wifiMac.SetType ("ns3::ApWifiMac",
                   "Ssid", SsidValue (ssid),
                   "BeaconInterval", TimeValue (MicroSeconds (103424))); //for avoiding collisions);
  wifiApDevices.Add (wifi.Install (wifiPhy, wifiMac, wifiApNodes.Get (1)));

  wifiDevices.Add (wifiStaDevices);
  wifiDevices.Add (wifiApDevices);

  //Configure the mobility.
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (ap1_x, ap1_y, 0.0));
  positionAlloc->Add (Vector (sta1_x, sta1_y, 0.0));
  positionAlloc->Add (Vector (ap2_x, ap2_y, 0.0));
  positionAlloc->Add (Vector (sta2_x, sta2_y, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNodes.Get (0));
  mobility.Install (wifiStaNodes.Get (0));
  mobility.Install (wifiApNodes.Get (1));
  mobility.Install (wifiStaNodes.Get (1));


  //Configure the IP stack
  InternetStackHelper stack;
  stack.Install (wifiApNodes);
  stack.Install (wifiStaNodes);
  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = address.Assign (wifiDevices);
  Ipv4Address sinkAddress = i.GetAddress (0);
  Ipv4Address sinkAddress1 = i.GetAddress (1);
  uint16_t port = 9;

  //Configure the CBR generator
  PacketSinkHelper sink ("ns3::UdpSocketFactory", InetSocketAddress (sinkAddress, port));
  ApplicationContainer apps_sink = sink.Install (wifiStaNodes.Get (0));

  OnOffHelper onoff ("ns3::UdpSocketFactory", InetSocketAddress (sinkAddress, port));
  onoff.SetConstantRate (DataRate ("54Mb/s"), packetSize);
  onoff.SetAttribute ("StartTime", TimeValue (Seconds (0.0)));
  onoff.SetAttribute ("StopTime", TimeValue (Seconds (100.0)));
  ApplicationContainer apps_source = onoff.Install (wifiApNodes.Get (0));

  PacketSinkHelper sink1 ("ns3::UdpSocketFactory", InetSocketAddress (sinkAddress1, port));
  apps_sink.Add (sink1.Install (wifiStaNodes.Get (1)));

  OnOffHelper onoff1 ("ns3::UdpSocketFactory", InetSocketAddress (sinkAddress1, port));
  onoff1.SetConstantRate (DataRate ("54Mb/s"), packetSize);
  onoff1.SetAttribute ("StartTime", TimeValue (Seconds (0.0)));
  onoff1.SetAttribute ("StopTime", TimeValue (Seconds (100.0)));
  apps_source.Add (onoff1.Install (wifiApNodes.Get (1)));

  apps_sink.Start (Seconds (0.5));
  apps_sink.Stop (Seconds (simuTime));

  //------------------------------------------------------------
  //-- Setup stats and data collection
  //--------------------------------------------

  //Statistics counters
  NodeStatistics statisticsAp0 = NodeStatistics (wifiApDevices, wifiStaDevices);
  NodeStatistics statisticsAp1 = NodeStatistics (wifiApDevices, wifiStaDevices);

  //Register packet receptions to calculate throughput
  Config::Connect ("/NodeList/2/ApplicationList/*/$ns3::PacketSink/Rx",
                   MakeCallback (&NodeStatistics::RxCallback, &statisticsAp0));
  Config::Connect ("/NodeList/3/ApplicationList/*/$ns3::PacketSink/Rx",
                   MakeCallback (&NodeStatistics::RxCallback, &statisticsAp1));

  //Register power and rate changes to calculate the Average Transmit Power
  Config::Connect ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$" + manager + "/PowerChange",
                   MakeCallback (&NodeStatistics::PowerCallback, &statisticsAp0));
  Config::Connect ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$" + manager + "/RateChange",
                   MakeCallback (&NodeStatistics::RateCallback, &statisticsAp0));
  Config::Connect ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$" + manager + "/PowerChange",
                   MakeCallback (&NodeStatistics::PowerCallback, &statisticsAp1));
  Config::Connect ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$" + manager + "/RateChange",
                   MakeCallback (&NodeStatistics::RateCallback, &statisticsAp1));

  Config::Connect ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                   MakeCallback (&NodeStatistics::PhyCallback, &statisticsAp0));
  Config::Connect ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin",
                   MakeCallback (&NodeStatistics::PhyCallback, &statisticsAp1));

  //Register States
  Config::Connect ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/State/State",
                   MakeCallback (&NodeStatistics::StateCallback, &statisticsAp0));
  Config::Connect ("/NodeList/1/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::YansWifiPhy/State/State",
                   MakeCallback (&NodeStatistics::StateCallback, &statisticsAp1));

  statisticsAp0.CheckStatistics (1);
  statisticsAp1.CheckStatistics (1);

  //Callbacks to print every change of power and rate
  Config::Connect ("/NodeList/[0-1]/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$" + manager + "/PowerChange",
                   MakeCallback (PowerCallback));
  Config::Connect ("/NodeList/[0-1]/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/$" + manager + "/RateChange",
                   MakeCallback (RateCallback));


  //Calculate Throughput using Flowmonitor

  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll ();

  Simulator::Stop (Seconds (simuTime));
  Simulator::Run ();

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier> (flowmon.GetClassifier ());
  std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();
  for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin (); i != stats.end (); ++i)
    {
      Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow (i->first);
      if ((t.sourceAddress == "10.1.1.3" && t.destinationAddress == "10.1.1.1"))
        {
          NS_LOG_INFO ("Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n");
          NS_LOG_INFO ("  Tx Bytes:   " << i->second.txBytes << "\n");
          NS_LOG_INFO ("  Rx Bytes:   " << i->second.rxBytes << "\n");
          NS_LOG_UNCOND ("  Throughput to 10.1.1.1: " << i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket.GetSeconds () - i->second.timeFirstTxPacket.GetSeconds ()) / 1024 / 1024  << " Mbps\n");
          NS_LOG_INFO ("  Mean delay:   " << i->second.delaySum.GetSeconds () / i->second.rxPackets << "\n");
          NS_LOG_INFO ("  Mean jitter:   " << i->second.jitterSum.GetSeconds () / (i->second.rxPackets - 1) << "\n");
          NS_LOG_INFO ("  Tx Opp: " << 1 - (statisticsAp0.GetBusyTime () / simuTime));
        }
      if ((t.sourceAddress == "10.1.1.4" && t.destinationAddress == "10.1.1.2"))
        {
          NS_LOG_INFO ("Flow " << i->first  << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\n");
          NS_LOG_INFO ("  Tx Bytes:   " << i->second.txBytes << "\n");
          NS_LOG_INFO ("  Rx Bytes:   " << i->second.rxBytes << "\n");
          NS_LOG_UNCOND ("  Throughput to 10.1.1.2: " << i->second.rxBytes * 8.0 / (i->second.timeLastRxPacket.GetSeconds () - i->second.timeFirstTxPacket.GetSeconds ()) / 1024 / 1024  << " Mbps\n");
          NS_LOG_INFO ("  Mean delay:   " << i->second.delaySum.GetSeconds () / i->second.rxPackets << "\n");
          NS_LOG_INFO ("  Mean jitter:   " << i->second.jitterSum.GetSeconds () / (i->second.rxPackets - 1) << "\n");
          NS_LOG_INFO ("  Tx Opp: " << 1 - (statisticsAp1.GetBusyTime () / simuTime));
        }
    }

  //Plots for AP0
  std::ofstream outfileTh0 (("throughput-" + outputFileName + "-0.plt").c_str ());
  Gnuplot gnuplot = Gnuplot (("throughput-" + outputFileName + "-0.eps").c_str (), "Throughput");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Throughput (Mb/s)");
  gnuplot.SetTitle ("Throughput (AP0 to STA) vs time");
  gnuplot.AddDataset (statisticsAp0.GetDatafile ());
  gnuplot.GenerateOutput (outfileTh0);

  if (manager.compare ("ns3::ParfWifiManager") == 0
      || manager.compare ("ns3::AparfWifiManager") == 0
      || manager.compare ("ns3::RrpaaWifiManager") == 0)
    {
      std::ofstream outfilePower0 (("power-" + outputFileName + "-0.plt").c_str ());
      gnuplot = Gnuplot (("power-" + outputFileName + "-0.eps").c_str (), "Average Transmit Power");
      gnuplot.SetTerminal ("post eps color enhanced");
      gnuplot.SetLegend ("Time (seconds)", "Power (mW)");
      gnuplot.SetTitle ("Average transmit power (AP0 to STA) vs time");
      gnuplot.AddDataset (statisticsAp0.GetPowerDatafile ());
      gnuplot.GenerateOutput (outfilePower0);
    }

  std::ofstream outfileTx0 (("tx-" + outputFileName + "-0.plt").c_str ());
  gnuplot = Gnuplot (("tx-" + outputFileName + "-0.eps").c_str (), "Time in TX State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP0 in TX state vs time");
  gnuplot.AddDataset (statisticsAp0.GetTxDatafile ());
  gnuplot.GenerateOutput (outfileTx0);

  std::ofstream outfileRx0 (("rx-" + outputFileName + "-0.plt").c_str ());
  gnuplot = Gnuplot (("rx-" + outputFileName + "-0.eps").c_str (), "Time in RX State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP0 in RX state vs time");
  gnuplot.AddDataset (statisticsAp0.GetRxDatafile ());
  gnuplot.GenerateOutput (outfileRx0);

  std::ofstream outfileBusy0 (("busy-" + outputFileName + "-0.plt").c_str ());
  gnuplot = Gnuplot (("busy-" + outputFileName + "-0.eps").c_str (), "Time in Busy State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP0 in Busy state vs time");
  gnuplot.AddDataset (statisticsAp0.GetBusyDatafile ());
  gnuplot.GenerateOutput (outfileBusy0);

  std::ofstream outfileIdle0 (("idle-" + outputFileName + "-0.plt").c_str ());
  gnuplot = Gnuplot (("idle-" + outputFileName + "-0.eps").c_str (), "Time in Idle State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP0 in Idle state vs time");
  gnuplot.AddDataset (statisticsAp0.GetIdleDatafile ());
  gnuplot.GenerateOutput (outfileIdle0);

  //Plots for AP1
  std::ofstream outfileTh1 (("throughput-" + outputFileName + "-1.plt").c_str ());
  gnuplot = Gnuplot (("throughput-" + outputFileName + "-1.eps").c_str (), "Throughput");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Throughput (Mb/s)");
  gnuplot.SetTitle ("Throughput (AP1 to STA) vs time");
  gnuplot.AddDataset (statisticsAp1.GetDatafile ());
  gnuplot.GenerateOutput (outfileTh1);

  if (manager.compare ("ns3::ParfWifiManager") == 0
      || manager.compare ("ns3::AparfWifiManager") == 0
      || manager.compare ("ns3::RrpaaWifiManager") == 0)
    {
      std::ofstream outfilePower1 (("power-" + outputFileName + "-1.plt").c_str ());
      gnuplot = Gnuplot (("power-" + outputFileName + "-1.eps").c_str (), "Average Transmit Power");
      gnuplot.SetTerminal ("post eps color enhanced");
      gnuplot.SetLegend ("Time (seconds)", "Power (mW)");
      gnuplot.SetTitle ("Average transmit power (AP1 to STA) vs time");
      gnuplot.AddDataset (statisticsAp1.GetPowerDatafile ());
      gnuplot.GenerateOutput (outfilePower1);
    }

  std::ofstream outfileTx1 (("tx-" + outputFileName + "-1.plt").c_str ());
  gnuplot = Gnuplot (("tx-" + outputFileName + "-1.eps").c_str (), "Time in TX State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP1 in TX state vs time");
  gnuplot.AddDataset (statisticsAp1.GetTxDatafile ());
  gnuplot.GenerateOutput (outfileTx1);

  std::ofstream outfileRx1 (("rx-" + outputFileName + "-1.plt").c_str ());
  gnuplot = Gnuplot (("rx-" + outputFileName + "-1.eps").c_str (), "Time in RX State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP1 in RX state vs time");
  gnuplot.AddDataset (statisticsAp1.GetRxDatafile ());
  gnuplot.GenerateOutput (outfileRx1);

  std::ofstream outfileBusy1 (("busy-" + outputFileName + "-1.plt").c_str ());
  gnuplot = Gnuplot (("busy-" + outputFileName + "-1.eps").c_str (), "Time in Busy State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP1 in Busy state vs time");
  gnuplot.AddDataset (statisticsAp1.GetBusyDatafile ());
  gnuplot.GenerateOutput (outfileBusy1);

  std::ofstream outfileIdle1 (("idle-" + outputFileName + "-1.plt").c_str ());
  gnuplot = Gnuplot (("idle-" + outputFileName + "-1.eps").c_str (), "Time in Idle State");
  gnuplot.SetTerminal ("post eps color enhanced");
  gnuplot.SetLegend ("Time (seconds)", "Percent");
  gnuplot.SetTitle ("Percentage time AP1 in Idle state vs time");
  gnuplot.AddDataset (statisticsAp1.GetIdleDatafile ());
  gnuplot.GenerateOutput (outfileIdle1);

  Simulator::Destroy ();

  return 0;
}
