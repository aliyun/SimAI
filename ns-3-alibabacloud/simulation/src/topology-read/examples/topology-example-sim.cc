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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 * Author: Valerio Sartini <valesar@gmail.com>
 *
 * This program conducts a simple experiment: It builds up a topology based on
 * either Inet or Orbis trace files. A random node is then chosen, and all the
 * other nodes will send a packet to it. The TTL is measured and reported as an histogram.
 *
 */

#include <ctime>

#include <sstream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/nix-vector-helper.h"

#include "ns3/topology-read-module.h"
#include <list>

/**
 * \file
 * \ingroup topology
 * Example of TopologyReader: .read in a topology in a specificed format.
 */

//  Document the available input files
/**
 * \file RocketFuel_toposample_1239_weights.txt
 * Example TopologyReader input file in RocketFuel format;
 * to read this with topology-example-sim.cc use \c --format=Rocket
 */
/**
 * \file Inet_toposample.txt
 * Example TopologyReader input file in Inet format;
 * to read this with topology-example-sim.cc use \c --format=Inet
 */
/**
 * \file Inet_small_toposample.txt
 * Example TopologyReader input file in Inet format;
 * to read this with topology-example-sim.cc use \c --format=Inet
 */
/**
 * \file Orbis_toposample.txt
 * Example TopologyReader input file in Orbis format;
 * to read this with topology-example-sim.cc use \c --format=Orbis
 */

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TopologyCreationExperiment");

/**
 * Print the TTL of received packet
 * \param p received packet
 * \param ad sender address
 */
static void SinkRx (Ptr<const Packet> p, const Address &ad)
{
  Ipv4Header ipv4;
  p->PeekHeader (ipv4);
  std::cout << "TTL: " << (unsigned)ipv4.GetTtl () << std::endl;
}


// ----------------------------------------------------------------------
// -- main
// ----------------------------------------------
int main (int argc, char *argv[])
{

  std::string format ("Inet");
  std::string input ("src/topology-read/examples/Inet_small_toposample.txt");

  // Set up command line parameters used to control the experiment.
  CommandLine cmd (__FILE__);
  cmd.AddValue ("format", "Format to use for data input [Orbis|Inet|Rocketfuel].",
                format);
  cmd.AddValue ("input", "Name of the input file.",
                input);
  cmd.Parse (argc, argv);


  // ------------------------------------------------------------
  // -- Read topology data.
  // --------------------------------------------

  // Pick a topology reader based in the requested format.
  TopologyReaderHelper topoHelp;
  topoHelp.SetFileName (input);
  topoHelp.SetFileType (format);
  Ptr<TopologyReader> inFile = topoHelp.GetTopologyReader ();

  NodeContainer nodes;

  if (inFile != 0)
    {
      nodes = inFile->Read ();
    }

  if (inFile->LinksSize () == 0)
    {
      NS_LOG_ERROR ("Problems reading the topology file. Failing.");
      return -1;
    }

  // ------------------------------------------------------------
  // -- Create nodes and network stacks
  // --------------------------------------------
  NS_LOG_INFO ("creating internet stack");
  InternetStackHelper stack;

  // Setup NixVector Routing
  Ipv4NixVectorHelper nixRouting;
  stack.SetRoutingHelper (nixRouting);  // has effect on the next Install ()
  stack.Install (nodes);

  NS_LOG_INFO ("creating IPv4 addresses");
  Ipv4AddressHelper address;
  address.SetBase ("10.0.0.0", "255.255.255.252");

  int totlinks = inFile->LinksSize ();

  NS_LOG_INFO ("creating node containers");
  NodeContainer* nc = new NodeContainer[totlinks];
  TopologyReader::ConstLinksIterator iter;
  int i = 0;
  for ( iter = inFile->LinksBegin (); iter != inFile->LinksEnd (); iter++, i++ )
    {
      nc[i] = NodeContainer (iter->GetFromNode (), iter->GetToNode ());
    }

  NS_LOG_INFO ("creating net device containers");
  NetDeviceContainer* ndc = new NetDeviceContainer[totlinks];
  PointToPointHelper p2p;
  for (int i = 0; i < totlinks; i++)
    {
      // p2p.SetChannelAttribute ("Delay", TimeValue(MilliSeconds(weight[i])));
      p2p.SetChannelAttribute ("Delay", StringValue ("2ms"));
      p2p.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
      ndc[i] = p2p.Install (nc[i]);
    }

  // it crates little subnets, one for each couple of nodes.
  NS_LOG_INFO ("creating IPv4 interfaces");
  Ipv4InterfaceContainer* ipic = new Ipv4InterfaceContainer[totlinks];
  for (int i = 0; i < totlinks; i++)
    {
      ipic[i] = address.Assign (ndc[i]);
      address.NewNetwork ();
    }


  uint32_t totalNodes = nodes.GetN ();
  Ptr<UniformRandomVariable> unifRandom = CreateObject<UniformRandomVariable> ();
  unifRandom->SetAttribute ("Min", DoubleValue (0));
  unifRandom->SetAttribute ("Max", DoubleValue (totalNodes - 1));

  unsigned int randomServerNumber = unifRandom->GetInteger (0, totalNodes - 1);

  Ptr<Node> randomServerNode = nodes.Get (randomServerNumber);
  Ptr<Ipv4> ipv4Server = randomServerNode->GetObject<Ipv4> ();
  Ipv4InterfaceAddress iaddrServer = ipv4Server->GetAddress (1,0);
  Ipv4Address ipv4AddrServer = iaddrServer.GetLocal ();

  // ------------------------------------------------------------
  // -- Send around packets to check the ttl
  // --------------------------------------------
  Config::SetDefault ("ns3::Ipv4RawSocketImpl::Protocol", StringValue ("2"));
  InetSocketAddress dst = InetSocketAddress ( ipv4AddrServer );

  OnOffHelper onoff = OnOffHelper ("ns3::Ipv4RawSocketFactory", dst);
  onoff.SetConstantRate (DataRate (15000));
  onoff.SetAttribute ("PacketSize", UintegerValue (1200));

  NodeContainer clientNodes;
  for ( unsigned int i = 0; i < nodes.GetN (); i++ )
    {
      if (i != randomServerNumber )
        {
          Ptr<Node> clientNode = nodes.Get (i);
          clientNodes.Add (clientNode);
        }
    }

  ApplicationContainer apps = onoff.Install (clientNodes);
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (2.0));

  PacketSinkHelper sink = PacketSinkHelper ("ns3::Ipv4RawSocketFactory", dst);
  apps = sink.Install (randomServerNode);
  apps.Start (Seconds (0.0));
  apps.Stop (Seconds (3.0));

  // we trap the packet sink receiver to extract the TTL.
  Config::ConnectWithoutContext ("/NodeList/*/ApplicationList/*/$ns3::PacketSink/Rx",
                                 MakeCallback (&SinkRx));

  // ------------------------------------------------------------
  // -- Run the simulation
  // --------------------------------------------
  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();

  delete[] ipic;
  delete[] ndc;
  delete[] nc;

  NS_LOG_INFO ("Done.");

  return 0;

  // end main
}
