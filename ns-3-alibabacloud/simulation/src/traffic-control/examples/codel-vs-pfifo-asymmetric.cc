/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 ResiliNets, ITTC, University of Kansas
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
 * Author: Truc Anh N Nguyen <trucanh524@gmail.com>
 * Modified by:   Pasquale Imputato <p.imputato@gmail.com>
 *
 */

/*
 * This is an example that compares CoDel and PfifoFast queues using a
 * typical cable modem topology and delay
 * (total RTT 37 ms as measured by Measuring Broadband America)
 *
 *          10gigE         22 Mb/s         gigE
 *           15 ms          1 ms           0.1 ms
 *  --------       ------- (1)    --------        -------
 *  |      |------>|      |------>|      |------->|     |
 *  |server|       |CMTS  |       |Router|        |Host |
 *  |      |<------|      |<------|      |<-------|     |
 *  --------       --------    (2)--------        -------
 *          10gigE         5 Mb/s          gigE
 *           15 ms         6 ms            0.1 ms
 *
 * (1) PfifoFast queue , 256K bytes
 * (2) PfifoFast, CoDel
 *
 * The server initiates a bulk send TCP transfer to the host.
 * The host initiates a bulk send TCP transfer to the server.
 * Also, isochronous traffic (VoIP-like) between server and host
 * The default TCP version in ns-3, TcpNewReno, is used as the transport-layer
 * protocol.
 * Packets transmitted during a simulation run are captured into a .pcap file,
 * and congestion window values are also traced.
 */

#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/config-store-module.h"
#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/udp-header.h"
#include "ns3/enum.h"
#include "ns3/event-id.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("CoDelPfifoFastAsymmetricTest");

static void
CwndTracer (Ptr<OutputStreamWrapper>stream, uint32_t oldval, uint32_t newval)
{
  *stream->GetStream () << oldval << " " << newval << std::endl;
}

static void
TraceCwnd (std::string cwndTrFileName)
{
  AsciiTraceHelper ascii;
  if (cwndTrFileName.compare ("") == 0)
    {
      NS_LOG_DEBUG ("No trace file for cwnd provided");
      return;
    }
  else
    {
      Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream (cwndTrFileName.c_str ());
      Config::ConnectWithoutContext ("/NodeList/0/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",MakeBoundCallback (&CwndTracer, stream));
    }
}

static void
SojournTracer (Ptr<OutputStreamWrapper>stream, Time newval)
{
  *stream->GetStream () << newval << std::endl;
}

static void
TraceSojourn (std::string sojournTrFileName)
{
  AsciiTraceHelper ascii;
  if (sojournTrFileName.compare ("") == 0)
    {
      NS_LOG_DEBUG ("No trace file for sojourn provided");
      return;
    }
  else
    {
      Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream (sojournTrFileName.c_str ());
      Config::ConnectWithoutContext ("/NodeList/2/$ns3::TrafficControlLayer/RootQueueDiscList/0/$ns3::CoDelQueueDisc/SojournTime", MakeBoundCallback (&SojournTracer, stream));
    }
}

static void
QueueLengthTracer (Ptr<OutputStreamWrapper>stream, uint32_t oldval, uint32_t newval)
{
  *stream->GetStream () << oldval << " " << newval << std::endl;
}

static void
TraceQueueLength (std::string queueLengthTrFileName)
{
  AsciiTraceHelper ascii;
  if (queueLengthTrFileName.compare ("") == 0)
    {
      NS_LOG_DEBUG ("No trace file for queue length provided");
      return;
    }
  else
    {
      Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream (queueLengthTrFileName.c_str ());
      Config::ConnectWithoutContext ("/NodeList/2/$ns3::TrafficControlLayer/RootQueueDiscList/0/BytesInQueue", MakeBoundCallback (&QueueLengthTracer, stream));
    }
}

static void
EveryDropTracer (Ptr<OutputStreamWrapper>stream, Ptr<const QueueDiscItem> item)
{
  *stream->GetStream () << Simulator::Now ().GetSeconds () << " " << item << std::endl;
}

static void
TraceEveryDrop (std::string everyDropTrFileName)
{
  AsciiTraceHelper ascii;
  if (everyDropTrFileName.compare ("") == 0)
    {
      NS_LOG_DEBUG ("No trace file for every drop event provided");
      return;
    }
  else
    {
      Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream (everyDropTrFileName.c_str ());
      Config::ConnectWithoutContext ("/NodeList/2/$ns3::TrafficControlLayer/RootQueueDiscList/0/Drop", MakeBoundCallback (&EveryDropTracer, stream));
    }
}

static void
DroppingStateTracer (Ptr<OutputStreamWrapper>stream, bool oldVal, bool newVal)
{
  if (oldVal == false && newVal == true)
    {
      NS_LOG_INFO ("Entering the dropping state");
      *stream->GetStream () << Simulator::Now ().GetSeconds () << " ";
    }
  else if (oldVal == true && newVal == false)
    {
      NS_LOG_INFO ("Leaving the dropping state");
      *stream->GetStream () << Simulator::Now ().GetSeconds ()  << std::endl;
    }
}

static void
TraceDroppingState (std::string dropStateTrFileName)
{
  AsciiTraceHelper ascii;
  if (dropStateTrFileName.compare ("") == 0)
    {
      NS_LOG_DEBUG ("No trace file for dropping state provided");
      return;
    }
  else
    {
      Ptr<OutputStreamWrapper> stream = ascii.CreateFileStream (dropStateTrFileName.c_str ());
      Config::ConnectWithoutContext ("/NodeList/2/$ns3::TrafficControlLayer/RootQueueDiscList/0/$ns3::CoDelQueueDisc/DropState", MakeBoundCallback (&DroppingStateTracer, stream));
    }
}

void
CreateBulkFlow (AddressValue remoteAddress, Ptr<Node> sender, uint32_t pktSize, float stopTime)
{
  BulkSendHelper sourceHelper ("ns3::TcpSocketFactory", Address ());
  sourceHelper.SetAttribute ("Remote", remoteAddress);
  sourceHelper.SetAttribute ("SendSize", UintegerValue (pktSize));
  sourceHelper.SetAttribute ("MaxBytes", UintegerValue (0));
  ApplicationContainer sourceApp = sourceHelper.Install (sender);
  sourceApp.Start (Seconds (0));
  sourceApp.Stop (Seconds (stopTime - 3));
}

void
CreateOnOffFlow (AddressValue remoteAddress, Ptr<Node> sender, float stopTime)
{
  OnOffHelper sourceHelper ("ns3::UdpSocketFactory", Address ());
  sourceHelper.SetAttribute ("PacketSize", UintegerValue (280));
  sourceHelper.SetAttribute ("Remote", remoteAddress);
  ApplicationContainer sourceApp = sourceHelper.Install (sender);
  sourceApp.Start (Seconds (0));
  sourceApp.Stop (Seconds (stopTime - 3));
}

int main (int argc, char *argv[])
{
  std::string serverCmtsDelay = "15ms";
  std::string cmtsRouterDelay = "6ms";
  std::string routerHostDelay = "0.1ms";
  std::string serverLanDataRate = "10Gbps";
  std::string cmtsLanDataRate = "10Gbps";
  std::string cmtsWanDataRate = "22Mbps";
  std::string routerWanDataRate = "5Mbps";
  std::string routerLanDataRate = "10Gbps";
  std::string hostLanDataRate = "10Gbps";

  std::string routerWanQueueDiscType = "CoDel";           // outbound cable router queue
  uint32_t pktSize = 1458;                // in bytes. 1458 to prevent fragments
  uint32_t queueSize = 1000;              // in packets
  uint32_t numOfUpLoadBulkFlows = 1;      // # of upload bulk transfer flows
  uint32_t numOfDownLoadBulkFlows = 1;    // # of download bulk transfer flows
  uint32_t numOfUpLoadOnOffFlows = 1;     // # of upload onoff flows
  uint32_t numOfDownLoadOnOffFlows = 1;   // # of download onoff flows
  bool isPcapEnabled = true;

  float startTime = 0.1f;
  float simDuration = 60;        //in seconds

  std::string fileNamePrefix = "codel-vs-pfifo-fast-asymmetric";
  bool logging = true;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("serverCmtsDelay", "Link delay between server and CMTS", serverCmtsDelay);
  cmd.AddValue ("cmtsRouterDelay", "Link delay between CMTS and rounter", cmtsRouterDelay);
  cmd.AddValue ("routerHostDelay", "Link delay between router and host", routerHostDelay);
  cmd.AddValue ("serverLanDataRate", "Server LAN net device data rate", serverLanDataRate);
  cmd.AddValue ("cmtsLanDataRate", "CMTS LAN net device data rate", cmtsLanDataRate);
  cmd.AddValue ("cmtsWanDataRate", "CMTS WAN net device data rate", cmtsWanDataRate);
  cmd.AddValue ("routerWanDataRate", "Router WAN net device data rate", routerWanDataRate);
  cmd.AddValue ("routerLanDataRate", "Router LAN net device data rate", routerLanDataRate);
  cmd.AddValue ("hostLanDataRate", "Host LAN net device data rate", hostLanDataRate);
  cmd.AddValue ("routerWanQueueDiscType", "Router WAN queue disc type: "
                "PfifoFast, CoDel", routerWanQueueDiscType);
  cmd.AddValue ("queueSize", "Queue size in packets", queueSize);
  cmd.AddValue ("pktSize", "Packet size in bytes", pktSize);
  cmd.AddValue ("numOfUpLoadBulkFlows", "Number of upload bulk transfer flows", numOfUpLoadBulkFlows);
  cmd.AddValue ("numOfDownLoadBulkFlows", "Number of download bulk transfer flows", numOfDownLoadBulkFlows);
  cmd.AddValue ("numOfUpLoadOnOffFlows", "Number of upload OnOff flows", numOfUpLoadOnOffFlows);
  cmd.AddValue ("numOfDownLoadOnOffFlows", "Number of download OnOff flows", numOfDownLoadOnOffFlows);
  cmd.AddValue ("startTime", "Simulation start time", startTime);
  cmd.AddValue ("simDuration", "Simulation duration in seconds", simDuration);
  cmd.AddValue ("isPcapEnabled", "Flag to enable/disable pcap", isPcapEnabled);
  cmd.AddValue ("logging", "Flag to enable/disable logging", logging);
  cmd.Parse (argc, argv);

  float stopTime = startTime + simDuration;

  std::string pcapFileName = fileNamePrefix + "-" + routerWanQueueDiscType;
  std::string cwndTrFileName = fileNamePrefix + "-" + routerWanQueueDiscType + "-cwnd" + ".tr";
  std::string attributeFileName = fileNamePrefix + "-" + routerWanQueueDiscType + ".attr";
  std::string sojournTrFileName = fileNamePrefix + "-" + routerWanQueueDiscType + "-sojourn" + ".tr";
  std::string queueLengthTrFileName = fileNamePrefix + "-" + routerWanQueueDiscType + "-length" + ".tr";
  std::string everyDropTrFileName = fileNamePrefix + "-" + routerWanQueueDiscType + "-drop" + ".tr";
  std::string dropStateTrFileName = fileNamePrefix + "-" + routerWanQueueDiscType + "-drop-state" + ".tr";
  if (logging)
    {
      //LogComponentEnable ("CoDelPfifoFastAsymmetricTest", LOG_LEVEL_ALL);
      //LogComponentEnable ("BulkSendApplication", LOG_LEVEL_INFO);
      //LogComponentEnable ("PfifoFastQueue", LOG_LEVEL_ALL);
      LogComponentEnable ("CoDelQueueDisc", LOG_LEVEL_FUNCTION);
    }

  // Queue defaults
  Config::SetDefault ("ns3::PfifoFastQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, queueSize)));
  Config::SetDefault ("ns3::CoDelQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, queueSize)));

  // Create the nodes
  NS_LOG_INFO ("Create nodes");
  NodeContainer nodes;
  nodes.Create (4);
  // Descriptive names
  Names::Add ("server", nodes.Get (0));
  Names::Add ("cmts", nodes.Get (1));
  Names::Add ("router", nodes.Get (2));
  Names::Add ("host", nodes.Get (3));
  NodeContainer serverCmts;
  serverCmts = NodeContainer (nodes.Get (0), nodes.Get (1));
  NodeContainer cmtsRouter;
  cmtsRouter = NodeContainer (nodes.Get (1), nodes.Get (2));
  NodeContainer routerHost;
  routerHost = NodeContainer (nodes.Get (2), nodes.Get (3));

  // Enable checksum
  if (isPcapEnabled)
    {
      GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));
    }

  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (pktSize));

  NS_LOG_INFO ("Create channels and install net devices on nodes");
  PointToPointHelper p2p;

  p2p.SetChannelAttribute ("Delay", StringValue (serverCmtsDelay));
  NetDeviceContainer serverCmtsDev = p2p.Install (serverCmts);
  Names::Add ("server/lan", serverCmtsDev.Get (0));
  Names::Add ("cmts/lan", serverCmtsDev.Get (1));
  Ptr<PointToPointNetDevice> serverLanDev = DynamicCast<PointToPointNetDevice> (serverCmtsDev.Get (0));
  serverLanDev->SetAttribute ("DataRate", StringValue (serverLanDataRate));
  Ptr<PointToPointNetDevice> cmtsLanDev = DynamicCast<PointToPointNetDevice> (serverCmtsDev.Get (1));
  cmtsLanDev->SetAttribute ("DataRate", StringValue (cmtsLanDataRate));

  p2p.SetChannelAttribute ("Delay", StringValue (cmtsRouterDelay));
  NetDeviceContainer cmtsRouterDev = p2p.Install (cmtsRouter);
  Names::Add ("cmts/wan", cmtsRouterDev.Get (0));
  Names::Add ("router/wan", cmtsRouterDev.Get (1));
  Ptr<PointToPointNetDevice> cmtsWanDev = DynamicCast<PointToPointNetDevice> (cmtsRouterDev.Get (0));
  cmtsWanDev->SetAttribute ("DataRate", StringValue (cmtsWanDataRate));
  Ptr<PointToPointNetDevice> routerWanDev = DynamicCast<PointToPointNetDevice> (cmtsRouterDev.Get (1));
  routerWanDev->SetAttribute ("DataRate", StringValue (routerWanDataRate));

  p2p.SetChannelAttribute ("Delay", StringValue (routerHostDelay));
  NetDeviceContainer routerHostDev = p2p.Install (routerHost);
  Names::Add ("router/lan", routerHostDev.Get (0));
  Names::Add ("host/lan", routerHostDev.Get (1));
  Ptr<PointToPointNetDevice> routerLanDev = DynamicCast<PointToPointNetDevice> (routerHostDev.Get (0));
  routerLanDev->SetAttribute ("DataRate", StringValue (routerLanDataRate));
  Ptr<PointToPointNetDevice> hostLanDev = DynamicCast<PointToPointNetDevice> (routerHostDev.Get (1));
  hostLanDev->SetAttribute ("DataRate", StringValue (hostLanDataRate));

  NS_LOG_INFO ("Install Internet stack on all nodes");
  InternetStackHelper stack;
  stack.InstallAll ();

  TrafficControlHelper tchPfifo;
  tchPfifo.SetRootQueueDisc ("ns3::PfifoFastQueueDisc");

  TrafficControlHelper tchCoDel;
  tchCoDel.SetRootQueueDisc ("ns3::CoDelQueueDisc");

  tchPfifo.Install (serverCmtsDev);
  tchPfifo.Install (cmtsWanDev);
  if (routerWanQueueDiscType.compare ("PfifoFast") == 0)
    {
      tchPfifo.Install (routerWanDev);
    }
  else if (routerWanQueueDiscType.compare ("CoDel") == 0)
    {
      tchCoDel.Install (routerWanDev);
    }
  else
    {
      NS_LOG_DEBUG ("Invalid router WAN queue disc type");
      exit (1);
    }  
  tchPfifo.Install (routerHostDev);

  NS_LOG_INFO ("Assign IP Addresses");
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer serverCmtsInterface = ipv4.Assign (serverCmtsDev);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer cmtsRouterInterface = ipv4.Assign (cmtsRouterDev);
  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer routerHostInterface = ipv4.Assign (routerHostDev);

  NS_LOG_INFO ("Initialize Global Routing");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  NS_LOG_INFO ("Configure downstream");
  uint16_t port1 = 50000;
  Address sinkLocalAddress1 (InetSocketAddress (Ipv4Address::GetAny (), port1));
  PacketSinkHelper sinkHelper1 ("ns3::TcpSocketFactory", sinkLocalAddress1);
  ApplicationContainer sinkApp1 = sinkHelper1.Install (routerHost.Get (1));
  sinkApp1.Start (Seconds (0));
  sinkApp1.Stop (Seconds (stopTime));
  AddressValue remoteAddress1 (InetSocketAddress (routerHostInterface.GetAddress (1), port1));
  while (numOfDownLoadBulkFlows)
    {
      CreateBulkFlow (remoteAddress1, serverCmts.Get (0), pktSize, stopTime);
      numOfDownLoadBulkFlows--;
    }

  while (numOfDownLoadOnOffFlows)
    {
      CreateOnOffFlow (remoteAddress1, serverCmts.Get (0), stopTime);
      numOfDownLoadOnOffFlows--;
    }

  NS_LOG_INFO ("Configure upstream");
  uint16_t port2 = 50001;
  Address sinkLocalAddress2 (InetSocketAddress (Ipv4Address::GetAny (), port2));
  PacketSinkHelper sinkHelper2 ("ns3::TcpSocketFactory", sinkLocalAddress2);
  ApplicationContainer sinkApp2 = sinkHelper2.Install (serverCmts.Get (0));
  sinkApp2.Start (Seconds (0));
  sinkApp2.Stop (Seconds (stopTime));
  AddressValue remoteAddress2 (InetSocketAddress (serverCmtsInterface.GetAddress (0), port2));
  while (numOfUpLoadBulkFlows)
    {
      CreateBulkFlow (remoteAddress2, routerHost.Get (1), pktSize, stopTime);
      numOfUpLoadBulkFlows--;
    }

  while (numOfUpLoadOnOffFlows)
    {
      CreateOnOffFlow (remoteAddress2, routerHost.Get (1), stopTime);
      numOfUpLoadOnOffFlows--;
    }

  Simulator::Schedule (Seconds (0.00001), &TraceCwnd, cwndTrFileName);
  TraceEveryDrop (everyDropTrFileName);
  if (routerWanQueueDiscType.compare ("CoDel") == 0)
    {
      TraceSojourn (sojournTrFileName);
      TraceQueueLength (queueLengthTrFileName);
      TraceDroppingState (dropStateTrFileName);
    }
  if (isPcapEnabled)
    {
      p2p.EnablePcapAll (pcapFileName);
    }

  // Output config store to txt format
  Config::SetDefault ("ns3::ConfigStore::Filename", StringValue (attributeFileName));
  Config::SetDefault ("ns3::ConfigStore::FileFormat", StringValue ("RawText"));
  Config::SetDefault ("ns3::ConfigStore::Mode", StringValue ("Save"));
  ConfigStore outputConfig;
  outputConfig.ConfigureDefaults ();
  outputConfig.ConfigureAttributes ();

  Simulator::Stop (Seconds (stopTime));
  Simulator::Run ();

  Simulator::Destroy ();
  return 0;
}
