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
 */

#include <string>
#include "ns3/core-module.h"
#include "ns3/mpi-interface.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"
#include "ns3/brite-module.h"
#include "ns3/nix-vector-helper.h"



#include <iostream>
#include <fstream>

#ifdef NS3_MPI
#include <mpi.h>
#endif

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("BriteMPITest");

int
main (int argc, char *argv[])
{
#ifdef NS3_MPI
  // Distributed simulation setup
  MpiInterface::Enable (&argc, &argv);
  GlobalValue::Bind ("SimulatorImplementationType",
                     StringValue ("ns3::DistributedSimulatorImpl"));

  LogComponentEnable ("BriteMPITest", LOG_LEVEL_ALL);
  LogComponentEnable ("TcpSocketBase", LOG_LEVEL_INFO);

  uint32_t systemId = MpiInterface::GetSystemId ();
  uint32_t systemCount = MpiInterface::GetSize ();

  // Check for valid distributed parameters.
  // For just this particular example, must have 2 and only 2 Logical Processors (LPs)
  NS_ASSERT_MSG (systemCount == 2, "This demonstration requires 2 and only 2 logical processors.");

  // BRITE needs a configuration file to build its graph. By default, this
  // example will use the TD_ASBarabasi_RTWaxman.conf file. There are many others
  // which can be found in the BRITE/conf_files directory
  std::string confFile = "src/brite/examples/conf_files/TD_ASBarabasi_RTWaxman.conf";
  bool tracing = false;
  bool nix = false;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("confFile", "BRITE conf file", confFile);
  cmd.AddValue ("tracing", "Enable or disable ascii tracing", tracing);
  cmd.AddValue ("nix", "Enable or disable nix-vector routing", nix);

  cmd.Parse (argc,argv);

  // Invoke the BriteTopologyHelper and pass in a BRITE
  // configuration file and a seed file. This will use
  // BRITE to build a graph from which we can build the ns-3 topology
  BriteTopologyHelper bth (confFile);

  PointToPointHelper p2p;


  InternetStackHelper stack;

  if (nix)
    {
      Ipv4NixVectorHelper nixRouting;
      stack.SetRoutingHelper (nixRouting);
    }

  Ipv4AddressHelper address;
  address.SetBase ("10.0.0.0", "255.255.255.252");

  //build topology as normal but also pass systemCount
  bth.BuildBriteTopology (stack, systemCount);
  bth.AssignIpv4Addresses (address);

  NS_LOG_LOGIC ("Number of AS created " << bth.GetNAs ());

  uint16_t port = 5001;

  NodeContainer client;
  NodeContainer server;

  //For this example will use AS 0 and AS 1 which will be on separate systems
  //due to the mod divide used to assign AS to system.

  //GetSystemNumberForAs (uint32_t) can be used to determine which system an
  //AS is assigned to
  NS_LOG_LOGIC ("AS 0 has been assigned to system " << bth.GetSystemNumberForAs (0));
  NS_LOG_LOGIC ("As 1 has been assigned to system " << bth.GetSystemNumberForAs (1));

  //install client node on last leaf node of AS 0
  client.Add (CreateObject<Node> (0));
  stack.Install (client);
  int numLeafNodesInAsZero = bth.GetNLeafNodesForAs (0);
  client.Add (bth.GetLeafNodeForAs (0, numLeafNodesInAsZero - 1));

  //install server node on last leaf node on AS 1
  server.Add (CreateObject<Node> (1));
  stack.Install (server);
  int numLeafNodesInAsOne = bth.GetNLeafNodesForAs (1);
  server.Add (bth.GetLeafNodeForAs (1, numLeafNodesInAsOne - 1));

  p2p.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("2ms"));

  NetDeviceContainer p2pClientDevices;
  NetDeviceContainer p2pServerDevices;

  p2pClientDevices = p2p.Install (client);
  p2pServerDevices = p2p.Install (server);

  address.SetBase ("10.1.0.0", "255.255.0.0");
  Ipv4InterfaceContainer clientInterfaces;
  clientInterfaces = address.Assign (p2pClientDevices);

  address.SetBase ("10.2.0.0", "255.255.0.0");
  Ipv4InterfaceContainer serverInterfaces;
  serverInterfaces = address.Assign (p2pServerDevices);

  if (!nix)
    {
      Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
    }

  //only has two systems in this example.  Install applications only on nodes in my system


  //Moved here to get totalRX at end
  ApplicationContainer sinkApps;

  if (systemId == 1)
    {

      Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
      PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);
      sinkApps.Add (packetSinkHelper.Install (server.Get (0)));
      sinkApps.Start (Seconds (0.0));
      sinkApps.Stop (Seconds (10.0));
    }

  if (systemId == 0)
    {
      OnOffHelper clientHelper ("ns3::TcpSocketFactory", Address ());
      clientHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
      clientHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

      ApplicationContainer clientApps;
      AddressValue remoteAddress (InetSocketAddress (serverInterfaces.GetAddress (0), port));
      clientHelper.SetAttribute ("Remote", remoteAddress);
      clientApps.Add (clientHelper.Install (client.Get (0)));
      clientApps.Start (Seconds (1.0)); // Start 1 second after sink
      clientApps.Stop (Seconds (9.0)); // Stop before the sink
    }

  if (!nix)
    {
      Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
    }

  if (tracing)
    {
      AsciiTraceHelper ascii;
      p2p.EnableAsciiAll (ascii.CreateFileStream ("briteLeaves.tr"));
    }

  // Run the simulator
  Simulator::Stop (Seconds (200.0));
  Simulator::Run ();
  Simulator::Destroy ();

  if (systemId == 1)
    {
      Ptr<PacketSink> sink1 = DynamicCast<PacketSink> (sinkApps.Get (0));
      NS_LOG_DEBUG ("Total Bytes Received: " << sink1->GetTotalRx ());
    }

  MpiInterface::Disable ();

  return 0;

#else
  NS_FATAL_ERROR ("Can't use distributed simulator without MPI compiled in");
#endif
}
