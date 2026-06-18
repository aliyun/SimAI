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

#include "mpi-test-fixtures.h"

#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/mpi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"

#include <iomanip>

/**
 * \file
 * \ingroup mpi
 *
 * Distributed version of third.cc from the tutorial.
 *
 *  Default Network Topology
 *
 * (same as third.cc from tutorial)
 * Distributed simulation, split across the p2p link
 *                          |
 *                 Rank 0   |   Rank 1
 * -------------------------|----------------------------
 *   Wifi 10.1.3.0
 *                 AP
 *  *    *    *    *
 *  |    |    |    |    10.1.1.0
 * n5   n6   n7   n0 -------------- n1   n2   n3   n4
 *                   point-to-point  |    |    |    |
 *                                   ================
 *                                    LAN 10.1.2.0
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ThirdExampleDistributed");

int 
main (int argc, char *argv[])
{
  bool verbose = false;
  uint32_t nCsma = 3;
  uint32_t nWifi = 3;
  bool tracing = false;
  bool nullmsg = false;
  bool testing = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nCsma", "Number of \"extra\" CSMA nodes/devices", nCsma);
  cmd.AddValue ("nWifi", "Number of wifi STA devices", nWifi);
  cmd.AddValue ("verbose", "Tell echo applications to log if true", verbose);
  cmd.AddValue ("tracing", "Enable pcap tracing", tracing);
  cmd.AddValue ("nullmsg", "Enable the use of null-message synchronization", nullmsg);
  cmd.AddValue ("test", "Enable regression test output", testing);

  cmd.Parse (argc,argv);

  // The underlying restriction of 18 is due to the grid position
  // allocator's configuration; the grid layout will exceed the
  // bounding box if more than 18 nodes are provided.
  if (nWifi > 18)
    {
      std::cout << "nWifi should be 18 or less; otherwise grid layout exceeds the bounding box" << std::endl;
      return 1;
    }

  if (verbose)
    {
      LogComponentEnable ("UdpEchoClientApplication", (LogLevel)(LOG_LEVEL_INFO | LOG_PREFIX_NODE | LOG_PREFIX_TIME));
      LogComponentEnable ("UdpEchoServerApplication", (LogLevel)(LOG_LEVEL_INFO | LOG_PREFIX_NODE | LOG_PREFIX_TIME));
    }

  // Sequential fallback values
  uint32_t systemId = 0;
  uint32_t systemCount = 1;

  // Distributed simulation setup; by default use granted time window algorithm.
  if(nullmsg) 
    {
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::NullMessageSimulatorImpl"));
    } 
  else 
    {
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::DistributedSimulatorImpl"));
    }

  MpiInterface::Enable (&argc, &argv);

  SinkTracer::Init ();

  systemId = MpiInterface::GetSystemId ();
  systemCount = MpiInterface::GetSize ();

  // Check for valid distributed parameters.
  // Must have 2 and only 2 Logical Processors (LPs)
  if (systemCount != 2)
    {
      std::cout << "This simulation requires 2 and only 2 logical processors." << std::endl;
      return 1;
    }

  // System id of Wifi side
  uint32_t systemWifi = 0;
  
  // System id of CSMA side
  uint32_t systemCsma = systemCount - 1;
  
  NodeContainer p2pNodes;
  // Create each end of the P2P link on a separate system (rank)
  Ptr<Node> p2pNode1 = CreateObject<Node> (systemWifi);
  Ptr<Node> p2pNode2 = CreateObject<Node> (systemCsma);
  p2pNodes.Add (p2pNode1);
  p2pNodes.Add (p2pNode2);

  PointToPointHelper pointToPoint;
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer p2pDevices;
  p2pDevices = pointToPoint.Install (p2pNodes);

  NodeContainer csmaNodes;
  csmaNodes.Add (p2pNodes.Get (1));
  // Create the csma nodes on one system (rank)
  csmaNodes.Create (nCsma, systemCsma);

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));

  NetDeviceContainer csmaDevices;
  csmaDevices = csma.Install (csmaNodes);

  NodeContainer wifiStaNodes;
  // Create the wifi nodes on the other system (rank)
  wifiStaNodes.Create (nWifi, systemWifi);
  NodeContainer wifiApNode = p2pNodes.Get (0);

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");

  WifiHelper wifi;

  NetDeviceContainer staDevices;
  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));
  staDevices = wifi.Install (phy, mac, wifiStaNodes);

  NetDeviceContainer apDevices;
  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid));
  apDevices = wifi.Install (phy, mac, wifiApNode);

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
  mobility.Install (wifiStaNodes);

  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);

  InternetStackHelper stack;
  stack.Install (csmaNodes);
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);

  Ipv4AddressHelper address;

  address.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer p2pInterfaces;
  p2pInterfaces = address.Assign (p2pDevices);

  address.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer csmaInterfaces;
  csmaInterfaces = address.Assign (csmaDevices);

  address.SetBase ("10.1.3.0", "255.255.255.0");
  address.Assign (staDevices);
  address.Assign (apDevices);

  // If this rank is systemCsma, 
  // it should contain the server application, 
  // since it is on one of the csma nodes
  if (systemId == systemCsma)
    {
      UdpEchoServerHelper echoServer (9);

      ApplicationContainer serverApps = echoServer.Install (csmaNodes.Get (nCsma));
      serverApps.Start (Seconds (1.0));
      serverApps.Stop (Seconds (10.0));

      if (testing)
        {
          serverApps.Get (0)->TraceConnectWithoutContext ("RxWithAddresses", MakeCallback (&SinkTracer::SinkTrace));
        }
    }

  // If this rank is systemWifi
  // it should contain the client application, 
  // since it is on one of the wifi nodes
  if (systemId == systemWifi)
    {
      UdpEchoClientHelper echoClient (csmaInterfaces.GetAddress (nCsma), 9);
      echoClient.SetAttribute ("MaxPackets", UintegerValue (1));
      echoClient.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
      echoClient.SetAttribute ("PacketSize", UintegerValue (1024));

      ApplicationContainer clientApps = 
        echoClient.Install (wifiStaNodes.Get (nWifi - 1));
      clientApps.Start (Seconds (2.0));
      clientApps.Stop (Seconds (10.0));

      if (testing)
        {
          clientApps.Get (0)->TraceConnectWithoutContext ("RxWithAddresses", MakeCallback (&SinkTracer::SinkTrace));
        }
    }
 
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  Simulator::Stop (Seconds (10.0));

  if (tracing == true)
    {
      // Depending on the system Id (rank), the pcap information 
      // traced will be different.  For example, the ethernet pcap
      // will be empty for rank0, since these nodes are placed on 
      // on rank 1.  All ethernet traffic will take place on rank 1.
      // Similar differences are seen in the p2p and wireless pcaps.
      if (systemId == systemCsma)
        {
          pointToPoint.EnablePcapAll ("third-distributed-csma");
          phy.EnablePcap ("third-distributed-csma", apDevices.Get (0));
          csma.EnablePcap ("third-distributed-csma", csmaDevices.Get (0), true);
        }
      else // systemWifi
        {
          pointToPoint.EnablePcapAll ("third-distributed-wifi");
          phy.EnablePcap ("third-distributed-wifi", apDevices.Get (0));
          csma.EnablePcap ("third-distributed-wifi", csmaDevices.Get (0), true);
        }
    }

  Simulator::Run ();
  Simulator::Destroy ();

  if (testing)
    {
      SinkTracer::Verify (2);
    }

  // Exit the MPI execution environment
  MpiInterface::Disable ();
  
  return 0;
}
