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

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/csma-star-helper.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv6-address-generator.h"

// Network topology (default)
//
//            n2     +          +     n3          .
//             | ... |\        /| ... |           .
//             ======= \      / =======           .
//              CSMA    \    /   CSMA             .
//                       \  /                     .
//            n1     +--- n0 ---+     n4          .
//             | ... |   /  \   | ... |           .
//             =======  /    \  =======           .
//              CSMA   /      \  CSMA             .
//                    /        \                  .
//            n6     +          +     n5          .
//             | ... |          | ... |           .
//             =======          =======           .
//              CSMA             CSMA             .
//

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("CsmaStar");

int 
main (int argc, char *argv[])
{

  //
  // Set up some default values for the simulation.
  //
  Config::SetDefault ("ns3::OnOffApplication::PacketSize", UintegerValue (137));

  // ??? try and stick 15kb/s into the data rate
  Config::SetDefault ("ns3::OnOffApplication::DataRate", StringValue ("14kb/s"));

  //
  // Default number of nodes in the star.  Overridable by command line argument.
  //
  uint32_t nSpokes = 7;
  uint32_t useIpv6 = 0;
  Ipv6Address ipv6AddressBase = Ipv6Address("2001::");
  Ipv6Prefix ipv6AddressPrefix = Ipv6Prefix(64);

  CommandLine cmd (__FILE__);
  cmd.AddValue ("nSpokes", "Number of spoke nodes to place in the star", nSpokes);
  cmd.AddValue ("useIpv6", "Use Ipv6", useIpv6);
  cmd.Parse (argc, argv);

  NS_LOG_INFO ("Build star topology.");
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
  csma.SetChannelAttribute ("Delay", StringValue ("1ms"));
  CsmaStarHelper star (nSpokes, csma);

  NodeContainer fillNodes;

  //
  // Just to be nasy, hang some more nodes off of the CSMA channel for each
  // spoke, so that there are a total of 16 nodes on each channel.  Stash
  // all of these new devices into a container.
  //
  NetDeviceContainer fillDevices;

  uint32_t nFill = 14;
  for (uint32_t i = 0; i < star.GetSpokeDevices ().GetN (); ++i)
    {
      Ptr<Channel> channel = star.GetSpokeDevices ().Get (i)->GetChannel ();
      Ptr<CsmaChannel> csmaChannel = channel->GetObject<CsmaChannel> ();
      NodeContainer newNodes;
      newNodes.Create (nFill);
      fillNodes.Add (newNodes);
      fillDevices.Add (csma.Install (newNodes, csmaChannel));
    }

  NS_LOG_INFO ("Install internet stack on all nodes.");
  InternetStackHelper internet;
  star.InstallStack (internet);
  internet.Install (fillNodes);

  NS_LOG_INFO ("Assign IP Addresses.");
  if (useIpv6 == 0)
    {
      star.AssignIpv4Addresses (Ipv4AddressHelper ("10.1.0.0", "255.255.255.0"));
    }
  else
    {
      star.AssignIpv6Addresses (ipv6AddressBase, ipv6AddressPrefix);
    }

  //
  // We assigned addresses to the logical hub and the first "drop" of the 
  // CSMA network that acts as the spoke, but we also have a number of fill
  // devices (nFill) also hanging off the CSMA network.  We have got to 
  // assign addresses to them as well.  We put all of the fill devices into
  // a single device container, so the first nFill devices are associated
  // with the channel connected to spokeDevices.Get (0), the second nFill
  // devices afe associated with the channel connected to spokeDevices.Get (1)
  // etc.
  //
  Ipv4AddressHelper address;
  Ipv6AddressHelper address6;
  for(uint32_t i = 0; i < star.SpokeCount (); ++i)
    {
      if (useIpv6 == 0)
        {
          std::ostringstream subnet;
          subnet << "10.1." << i << ".0";
          NS_LOG_INFO ("Assign IP Addresses for CSMA subnet " << subnet.str ());
          address.SetBase (subnet.str ().c_str (), "255.255.255.0", "0.0.0.3");

          for (uint32_t j = 0; j < nFill; ++j)
            {
              address.Assign (fillDevices.Get (i * nFill + j));
            }
        }
      else
        {
          Ipv6AddressGenerator::Init (ipv6AddressBase, ipv6AddressPrefix);
          Ipv6Address v6network = Ipv6AddressGenerator::GetNetwork (ipv6AddressPrefix);
          address6.SetBase (v6network, ipv6AddressPrefix);

          for (uint32_t j = 0; j < nFill; ++j)
            {
              address6.Assign(fillDevices.Get (i * nFill + j));
            }
        }
    }

  NS_LOG_INFO ("Create applications.");
  //
  // Create a packet sink on the star "hub" to receive packets.
  // 
  uint16_t port = 50000;

  if (useIpv6 == 0)
    {
      Address hubLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
      PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", hubLocalAddress);
      ApplicationContainer hubApp = packetSinkHelper.Install (star.GetHub ());
      hubApp.Start (Seconds (1.0));
      hubApp.Stop (Seconds (10.0));
    }
  else 
    {
      Address hubLocalAddress6 (Inet6SocketAddress (Ipv6Address::GetAny (), port));
      PacketSinkHelper packetSinkHelper6 ("ns3::TcpSocketFactory", hubLocalAddress6);
      ApplicationContainer hubApp6 = packetSinkHelper6.Install (star.GetHub ());
      hubApp6.Start (Seconds (1.0));
      hubApp6.Stop (Seconds (10.0));
    }

  //
  // Create OnOff applications to send TCP to the hub, one on each spoke node.
  //
  OnOffHelper onOffHelper ("ns3::TcpSocketFactory", Address ());
  onOffHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  onOffHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

  ApplicationContainer spokeApps;

  for (uint32_t i = 0; i < star.SpokeCount (); ++i)
    {
      if (useIpv6 == 0)
        {
          AddressValue remoteAddress (InetSocketAddress (star.GetHubIpv4Address (i), port));
          onOffHelper.SetAttribute ("Remote", remoteAddress);
        }
      else
        {
          AddressValue remoteAddress (Inet6SocketAddress (star.GetHubIpv6Address (i), port));
          onOffHelper.SetAttribute ("Remote", remoteAddress);
        }
      spokeApps.Add (onOffHelper.Install (star.GetSpokeNode (i)));
    }

  spokeApps.Start (Seconds (1.0));
  spokeApps.Stop (Seconds (10.0));

  //
  // Because we are evil, we also add OnOff applications to send TCP to the hub 
  // from the fill devices on each CSMA link.  The first nFill nodes in the 
  // fillNodes container are on the CSMA network talking to the zeroth device
  // on the hub node.  The next nFill nodes are on the CSMA network talking to
  // the first device on the hub node, etc.  So the ith fillNode is associated
  // with the hub address found on the (i / nFill)th device on the hub node.
  //
  ApplicationContainer fillApps;

  for (uint32_t i = 0; i < fillNodes.GetN (); ++i)
    {
      AddressValue remoteAddress;
      if (useIpv6 == 0)
        {
          remoteAddress = AddressValue(InetSocketAddress (star.GetHubIpv4Address (i / nFill), port));
        }
      else
        {
          remoteAddress = AddressValue(Inet6SocketAddress (star.GetHubIpv6Address (i / nFill), port));
        }
      onOffHelper.SetAttribute ("Remote", remoteAddress);
      fillApps.Add (onOffHelper.Install (fillNodes.Get (i)));
    }

  fillApps.Start (Seconds (1.0));
  fillApps.Stop (Seconds (10.0));

  NS_LOG_INFO ("Enable static global routing.");
  //
  // Turn on global static routing so we can actually be routed across the star.
  //
  if (useIpv6 == 0)
    {
      Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
    }

  NS_LOG_INFO ("Enable pcap tracing.");
  //
  // Do pcap tracing on all devices on all nodes.
  //
  csma.EnablePcapAll ("csma-star", false);

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");

  return 0;
}
