/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright 2018. Lawrence Livermore National Security, LLC.
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
 * Author: Steven Smith <smith84@llnl.gov>
 */

/**
 * \file
 * \ingroup mpi
 *
 * This test is equivalent to simple-distributed with the addition of 
 * initialization of MPI by user code (this script) and providing
 * a communicator to ns-3.  The ns-3 communicator is smaller than
 * MPI Comm World as might be the case if ns-3 is run in parallel
 * with another simulator.
 *
 * TestDistributed creates a dumbbell topology and logically splits it in
 * half.  The left half is placed on logical processor 0 and the right half
 * is placed on logical processor 1.
 *
 *                 -------   -------
 *                  RANK 0    RANK 1
 *                 ------- | -------
 *                         |
 * n0 ---------|           |           |---------- n6
 *             |           |           |
 * n1 -------\ |           |           | /------- n7
 *            n4 ----------|---------- n5
 * n2 -------/ |           |           | \------- n8
 *             |           |           |
 * n3 ---------|           |           |---------- n9
 *
 *
 * OnOff clients are placed on each left leaf node. Each right leaf node
 * is a packet sink for a left leaf node.  As a packet travels from one
 * logical processor to another (the link between n4 and n5), MPI messages
 * are passed containing the serialized packet. The message is then
 * deserialized into a new packet and sent on as normal.
 *
 * One packet is sent from each left leaf node.  The packet sinks on the
 * right leaf nodes output logging information when they receive the packet.
 */

#include "mpi-test-fixtures.h"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mpi-interface.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/nix-vector-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/packet-sink-helper.h"

#include "mpi.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SimpleDistributedMpiComm");

/**
 * Tag for whether this rank should go into a new communicator
 * ns-3 ranks will have color == 1.
 * @{
 */
const int NS_COLOR = 1;
const int NOT_NS_COLOR = NS_COLOR + 1;
/** @} */

/**
 * Report my rank, in both MPI_COMM_WORLD and the split communicator.
 *
 * \param [in] color My role, either ns-3 rank or other rank.
 * \param [in] splitComm The split communicator.
 */
void
ReportRank (int color, MPI_Comm splitComm)
{
  int otherId=0;
  int otherSize=1;
  
  MPI_Comm_rank (splitComm, &otherId);
  MPI_Comm_size (splitComm, &otherSize);
  
  if (color == NS_COLOR)
    {
      RANK0COUT ( "ns-3 rank:  ");
    }
  else
    {
      RANK0COUT ( "Other rank: ");
    }
  
  RANK0COUTAPPEND ( "in MPI_COMM_WORLD: " << SinkTracer::GetWorldRank () << ":" << SinkTracer::GetWorldSize ()
                    << ", in splitComm: "    << otherId   << ":" << otherSize
                    << std::endl);
  
}  // ReportRank()

int
main (int argc, char *argv[])
{
  bool nix = true;
  bool nullmsg = false;
  bool tracing = false;
  bool init = false;
  bool verbose = false;
  bool testing = false;

  // Parse command line
  CommandLine cmd(__FILE__);
  cmd.AddValue ("nix", "Enable the use of nix-vector or global routing", nix);
  cmd.AddValue ("nullmsg", "Enable the use of null-message synchronization (instead of granted time window)", nullmsg);
  cmd.AddValue ("tracing", "Enable pcap tracing", tracing);
  cmd.AddValue ("init", "ns-3 should initialize MPI by calling MPI_Init", init);
  cmd.AddValue ("verbose", "verbose output", verbose);
  cmd.AddValue ("test", "Enable regression test output", testing);
  cmd.Parse (argc, argv);

  // Defer reporting the configuration until we know the communicator
  
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

  // MPI_Init

  if (init)
    {
      // Initialize MPI directly
      MPI_Init(&argc, &argv);
    }
  else
    {
      // Let ns-3 call MPI_Init and MPI_Finalize
      MpiInterface::Enable (&argc, &argv);
    }

  SinkTracer::Init ();

  auto worldSize = SinkTracer::GetWorldSize ();
  auto worldRank = SinkTracer::GetWorldRank ();

  if ( (!init) && (worldSize != 2))
    {
      RANK0COUT ("This simulation requires exactly 2 logical processors if --init is not set." << std::endl);
      return 1;
    }

  if (worldSize < 2)
    {
      RANK0COUT ("This simulation requires 2  or more logical processors." << std::endl);
      return 1;
    }

  // Set up the MPI communicator for ns-3
  //  Condition                         ns-3 Communicator
  //  a.  worldSize = 2    copy of MPI_COMM_WORLD
  //  b.  worldSize > 2    communicator of ranks 1-2

  // Flag to record that we created a communicator so we can free it at the end.
  bool freeComm = false;  
  // The new communicator, if we create one
  MPI_Comm splitComm = MPI_COMM_WORLD;
  // The list of ranks assigned to ns-3
  std::string ns3Ranks;
  // Tag for whether this rank should go into a new communicator
  int color = MPI_UNDEFINED;


  if (worldSize == 2)
    {
      std::stringstream ss;
      color = NS_COLOR;
      ss << "MPI_COMM_WORLD (" << worldSize << " ranks)";
      ns3Ranks = ss.str ();
      splitComm = MPI_COMM_WORLD;
      freeComm = false;
    }
  else
    {
      //  worldSize > 2    communicator of ranks 1-2
      
      // Put ranks 1-2 in the new communicator
      if (worldRank == 1 || worldRank == 2)
        {
          color = NS_COLOR;
        }
      else
        {
          color = NOT_NS_COLOR;
        }
      std::stringstream ss;
      ss << "Split [1-2] (out of " << worldSize << " ranks) from MPI_COMM_WORLD";
      ns3Ranks = ss.str ();
      
      // Now create the new communicator
      MPI_Comm_split (MPI_COMM_WORLD, color, worldRank, &splitComm);
      freeComm = true;
    }


  if(init)
    {
      MpiInterface::Enable (splitComm);
    }

  // Report the configuration from rank 0 only
  RANK0COUT (cmd.GetName () << "\n");
  RANK0COUT ("\n" );
  RANK0COUT ("Configuration:\n" );
  RANK0COUT ("Routing:           " << (nix ? "nix-vector" : "global") << "\n");
  RANK0COUT ("Synchronization:   " << (nullmsg ? "null-message" : "granted time window (YAWNS)") << "\n");
  RANK0COUT ("MPI_Init called:   " << (init ? "explicitly by this program" : "implicitly by ns3::MpiInterface::Enable()") << "\n" );
  RANK0COUT ("ns-3 Communicator: " << ns3Ranks << "\n");
  RANK0COUT ("PCAP tracing:      " << (tracing ? "" : "not") << " enabled\n");
  RANK0COUT ("\n");
  RANK0COUT ("Rank assignments:" << std::endl);

  if (worldRank == 0)
    {
      ReportRank (color, splitComm);
    }

  if(verbose)
    {
      // Circulate a token to have each rank report in turn
      int token;
    
      if (worldRank == 0)
        {
          token = 1;
        }
      else
        {
          MPI_Recv (&token, 1, MPI_INT, worldRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          ReportRank (color, splitComm);
        }

      MPI_Send (&token, 1, MPI_INT, (worldRank + 1) % worldSize, 0, MPI_COMM_WORLD);

      if (worldRank == 0)
        {
          MPI_Recv (&token, 1, MPI_INT, worldSize - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }  // circulate token to report rank

  RANK0COUT (std::endl);

  if (color != NS_COLOR)
    {
      // Do other work outside the ns-3 communicator
      
      // In real use of a separate communicator from ns-3
      // the other tasks would be running another simulator
      // or other desired work here..
      
      // Our work is done, just wait for everyone else to finish.

      MpiInterface::Disable ();

      if(init)
        {
          MPI_Finalize ();
        }
      
      return 0;
    }

  // The code below here is essentially the same as simple-distributed.cc
  // --------------------------------------------------------------------
  
  // We use a trace instead of relying on NS_LOG
  
  if (verbose)
    {
      LogComponentEnable ("PacketSink", LOG_LEVEL_INFO);
    }

  uint32_t systemId = MpiInterface::GetSystemId ();
  uint32_t systemCount = MpiInterface::GetSize ();

  // Check for valid distributed parameters.
  // Both this script and simple-distributed.cc will work
  // with arbitrary numbers of ranks, as long as there are at least 2.
  if (systemCount < 2)
    {
      RANK0COUT ("This simulation requires at least 2 logical processors." << std::endl);
      return 1;
    }

  // Some default values
  Config::SetDefault ("ns3::OnOffApplication::PacketSize", UintegerValue (512));
  Config::SetDefault ("ns3::OnOffApplication::DataRate", StringValue ("1Mbps"));
  Config::SetDefault ("ns3::OnOffApplication::MaxBytes", UintegerValue (512));

  // Create leaf nodes on left with system id 0
  NodeContainer leftLeafNodes;
  leftLeafNodes.Create (4, 0);

  // Create router nodes.  Left router
  // with system id 0, right router with
  // system id 1
  NodeContainer routerNodes;
  Ptr<Node> routerNode1 = CreateObject<Node> (0);
  Ptr<Node> routerNode2 = CreateObject<Node> (1);
  routerNodes.Add (routerNode1);
  routerNodes.Add (routerNode2);

  // Create leaf nodes on left with system id 1
  NodeContainer rightLeafNodes;
  rightLeafNodes.Create (4, 1);

  PointToPointHelper routerLink;
  routerLink.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  routerLink.SetChannelAttribute ("Delay", StringValue ("5ms"));

  PointToPointHelper leafLink;
  leafLink.SetDeviceAttribute ("DataRate", StringValue ("1Mbps"));
  leafLink.SetChannelAttribute ("Delay", StringValue ("2ms"));

  // Add link connecting routers
  NetDeviceContainer routerDevices;
  routerDevices = routerLink.Install (routerNodes);

  // Add links for left side leaf nodes to left router
  NetDeviceContainer leftRouterDevices;
  NetDeviceContainer leftLeafDevices;
  for (uint32_t i = 0; i < 4; ++i)
    {
      NetDeviceContainer temp = leafLink.Install (leftLeafNodes.Get (i), routerNodes.Get (0));
      leftLeafDevices.Add (temp.Get (0));
      leftRouterDevices.Add (temp.Get (1));
    }

  // Add links for right side leaf nodes to right router
  NetDeviceContainer rightRouterDevices;
  NetDeviceContainer rightLeafDevices;
  for (uint32_t i = 0; i < 4; ++i)
    {
      NetDeviceContainer temp = leafLink.Install (rightLeafNodes.Get (i), routerNodes.Get (1));
      rightLeafDevices.Add (temp.Get (0));
      rightRouterDevices.Add (temp.Get (1));
    }

  InternetStackHelper stack;
  Ipv4NixVectorHelper nixRouting;
  Ipv4StaticRoutingHelper staticRouting;

  Ipv4ListRoutingHelper list;
  list.Add (staticRouting, 0);
  list.Add (nixRouting, 10);

  if (nix)
    {
      stack.SetRoutingHelper (list); // has effect on the next Install ()
    }

  stack.InstallAll ();

  Ipv4InterfaceContainer routerInterfaces;
  Ipv4InterfaceContainer leftLeafInterfaces;
  Ipv4InterfaceContainer leftRouterInterfaces;
  Ipv4InterfaceContainer rightLeafInterfaces;
  Ipv4InterfaceContainer rightRouterInterfaces;

  Ipv4AddressHelper leftAddress;
  leftAddress.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4AddressHelper routerAddress;
  routerAddress.SetBase ("10.2.1.0", "255.255.255.0");

  Ipv4AddressHelper rightAddress;
  rightAddress.SetBase ("10.3.1.0", "255.255.255.0");

  // Router-to-Router interfaces
  routerInterfaces = routerAddress.Assign (routerDevices);

  // Left interfaces
  for (uint32_t i = 0; i < 4; ++i)
    {
      NetDeviceContainer ndc;
      ndc.Add (leftLeafDevices.Get (i));
      ndc.Add (leftRouterDevices.Get (i));
      Ipv4InterfaceContainer ifc = leftAddress.Assign (ndc);
      leftLeafInterfaces.Add (ifc.Get (0));
      leftRouterInterfaces.Add (ifc.Get (1));
      leftAddress.NewNetwork ();
    }

  // Right interfaces
  for (uint32_t i = 0; i < 4; ++i)
    {
      NetDeviceContainer ndc;
      ndc.Add (rightLeafDevices.Get (i));
      ndc.Add (rightRouterDevices.Get (i));
      Ipv4InterfaceContainer ifc = rightAddress.Assign (ndc);
      rightLeafInterfaces.Add (ifc.Get (0));
      rightRouterInterfaces.Add (ifc.Get (1));
      rightAddress.NewNetwork ();
    }

  if (!nix)
    {
      Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
    }

  if (tracing == true)
    {
      if (systemId == 0)
        {
          routerLink.EnablePcap("router-left", routerDevices, true);
          leafLink.EnablePcap("leaf-left", leftLeafDevices, true);
        }
      
      if (systemId == 1)
        {
          routerLink.EnablePcap("router-right", routerDevices, true);
          leafLink.EnablePcap("leaf-right", rightLeafDevices, true);
        }
    }

  // Create a packet sink on the right leafs to receive packets from left leafs
  uint16_t port = 50000;
  if (systemId == 1)
    {
      Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
      PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", sinkLocalAddress);
      ApplicationContainer sinkApp;
      for (uint32_t i = 0; i < 4; ++i)
        {
          auto apps = sinkHelper.Install (rightLeafNodes.Get (i));
          auto sink = DynamicCast<PacketSink> (apps.Get (0));
          NS_ASSERT_MSG (sink, "Couldn't get PacketSink application.");
          if (testing)
            {
              sink->TraceConnectWithoutContext ("RxWithAddresses",  MakeCallback(&SinkTracer::SinkTrace));
            }
          sinkApp.Add (apps);
        }
      sinkApp.Start (Seconds (1.0));
      sinkApp.Stop (Seconds (5));
    }

  // Create the OnOff applications to send
  if (systemId == 0)
    {
      OnOffHelper clientHelper ("ns3::UdpSocketFactory", Address ());
      clientHelper.SetAttribute
        ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
      clientHelper.SetAttribute
        ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

      ApplicationContainer clientApps;
      for (uint32_t i = 0; i < 4; ++i)
        {
          AddressValue remoteAddress
            (InetSocketAddress (rightLeafInterfaces.GetAddress (i), port));
          clientHelper.SetAttribute ("Remote", remoteAddress);
          clientApps.Add (clientHelper.Install (leftLeafNodes.Get (i)));
        }
      clientApps.Start (Seconds (1.0));
      clientApps.Stop (Seconds (5));
    }

  RANK0COUT (std::endl);

  Simulator::Stop (Seconds (5));
  Simulator::Run ();
  Simulator::Destroy ();

  // --------------------------------------------------------------------
  // Conditional cleanup based on whether we built a communicator
  // and called MPI_Init
  
  if (freeComm)
    {
      MPI_Comm_free (&splitComm);
    }

  if (testing)
    {
      SinkTracer::Verify (4);
    }
  
  // Clean up the ns-3 MPI execution environment
  // This will call MPI_Finalize if MpiInterface::Initialize was called
  MpiInterface::Disable ();

  if (init)
    {
      // We called MPI_Init, so we have to call MPI_Finalize
      MPI_Finalize ();
    }
  
  return 0;
}
