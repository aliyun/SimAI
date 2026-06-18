/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze
 * Copyright (c) 2019 Caliola Engineering, LLC : RFC 6621 multicast packet de-duplication
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 * Modified (2019): Jared Dulmage <jared.dulmage@caliola.com>
 *   Demonstrates dissemination of multicast packets across a mesh
 *   network to all nodes over multiple hops.
 */

#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/simple-channel.h"
#include "ns3/simple-net-device.h"
#include "ns3/socket.h"
#include "ns3/boolean.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/config.h"
#include "ns3/data-rate.h"
#include "ns3/uinteger.h"

#include "ns3/names.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet-socket-address.h"
#include "ns3/random-variable-stream.h"

#include "ns3/ipv4-l3-protocol.h"
#include "ns3/ipv4-static-routing.h"
#include "ns3/udp-socket-factory.h"
#include "ns3/udp-socket.h"
#include "ns3/packet-sink.h"

#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/on-off-helper.h"
#include "ns3/trace-helper.h"

#include "ns3/traffic-control-layer.h"

#include <string>
#include <limits>
#include <functional>

using namespace ns3;

/**
 * Network topology:
 *
 *        /---- B ----\
 *  A ----      |       ---- D ---- E
 *        \---- C ----/
 *
 * This example demonstrates configuration of
 * static routing to realize broadcast-like
 * flooding of packets from node A 
 * across the illustrated topology.
 */
int
main (int argc, char *argv[])
{
  // multicast target
  const std::string targetAddr = "239.192.100.1";
  Config::SetDefault ("ns3::Ipv4L3Protocol::EnableDuplicatePacketDetection", BooleanValue (true));
  Config::SetDefault ("ns3::Ipv4L3Protocol::DuplicateExpire", TimeValue (Seconds (10)));

  // Create topology

  // Create nodes
  auto nodes = NodeContainer ();
  nodes.Create (5);

  // Name nodes
  Names::Add ("A", nodes.Get (0));
  Names::Add ("B", nodes.Get (1));
  Names::Add ("C", nodes.Get (2));
  Names::Add ("D", nodes.Get (3));
  Names::Add ("E", nodes.Get (4));

  SimpleNetDeviceHelper simplenet;
  auto devices = simplenet.Install (nodes);
  // name devices
  Names::Add ("A/dev", devices.Get (0));
  Names::Add ("B/dev", devices.Get (1));
  Names::Add ("C/dev", devices.Get (2));
  Names::Add ("D/dev", devices.Get (3));
  Names::Add ("E/dev", devices.Get (4));

  // setup static routes to facilitate multicast flood
  Ipv4ListRoutingHelper listRouting;
  Ipv4StaticRoutingHelper staticRouting;
  listRouting.Add (staticRouting, 0);

  InternetStackHelper internet;
  internet.SetIpv6StackInstall (false);
  internet.SetIpv4ArpJitter (true);
  internet.SetRoutingHelper (listRouting);
  internet.Install (nodes);

  Ipv4AddressHelper ipv4address;
  ipv4address.SetBase ("10.0.0.0", "255.255.255.0");
  ipv4address.Assign (devices);

  // add static routes for each node / device
  for (auto diter = devices.Begin (); diter != devices.End (); ++diter)
    {
      Ptr<Node> node = (*diter)->GetNode ();

      if (Names::FindName (node) == "A")
        {
          // route for host
          // Use host routing entry according to note in Ipv4StaticRouting::RouteOutput:
          //// Note:  Multicast routes for outbound packets are stored in the
          //// normal unicast table.  An implication of this is that it is not
          //// possible to source multicast datagrams on multiple interfaces.
          //// This is a well-known property of sockets implementation on
          //// many Unix variants.
          //// So, we just log it and fall through to LookupStatic ()
          auto ipv4 = node->GetObject <Ipv4> ();
          NS_ASSERT_MSG ((bool) ipv4, "Node " << Names::FindName (node) << " does not have Ipv4 aggregate");
          auto routing = staticRouting.GetStaticRouting (ipv4);
          routing->AddHostRouteTo (targetAddr.c_str (), ipv4->GetInterfaceForDevice (*diter), 0);
        }
      else
        {
          // route for forwarding
          staticRouting.AddMulticastRoute (node, Ipv4Address::GetAny (), targetAddr.c_str (), *diter, NetDeviceContainer (*diter));
        }
    }

  // set the topology, by default fully-connected
  auto channel = devices.Get (0)->GetChannel ();
  auto simplechannel = channel->GetObject <SimpleChannel> ();
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("A/dev"), Names::Find <SimpleNetDevice> ("D/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("D/dev"), Names::Find <SimpleNetDevice> ("A/dev"));

  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("A/dev"), Names::Find <SimpleNetDevice> ("E/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("E/dev"), Names::Find <SimpleNetDevice> ("A/dev"));

  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("B/dev"), Names::Find <SimpleNetDevice> ("E/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("E/dev"), Names::Find <SimpleNetDevice> ("B/dev"));

  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("C/dev"), Names::Find <SimpleNetDevice> ("E/dev"));
  simplechannel->BlackList (Names::Find <SimpleNetDevice> ("E/dev"), Names::Find <SimpleNetDevice> ("C/dev"));
  // ensure some time progress between re-transmissions
  simplechannel->SetAttribute ("Delay", TimeValue (MilliSeconds (1)));
  
  // sinks
  PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory", InetSocketAddress (Ipv4Address::GetAny (), 9));
  auto sinks = sinkHelper.Install ("B");
  sinks.Add (sinkHelper.Install ("C"));
  sinks.Add (sinkHelper.Install ("D"));
  sinks.Add (sinkHelper.Install ("E"));
  sinks.Start (Seconds (1));

  // source
  OnOffHelper onoffHelper ("ns3::UdpSocketFactory", InetSocketAddress (targetAddr.c_str (), 9));
  onoffHelper.SetAttribute ("DataRate", DataRateValue (DataRate ("8Mbps")));
  onoffHelper.SetAttribute ("MaxBytes", UintegerValue (10 * 1024));
  auto source = onoffHelper.Install ("A");
  source.Start (Seconds (1.1));

  // pcap traces
  for (auto end = nodes.End (),
           iter = nodes.Begin (); iter != end; ++iter)
    {
      internet.EnablePcapIpv4 ("smf-trace", (*iter)->GetId (), 1, false);
    }

  // run simulation
  Simulator::Run ();
  
  std::cout << "Node A sent " << 10 * 1024 << " bytes" << std::endl;
  for (auto end = sinks.End (),
           iter = sinks.Begin (); iter != end; ++iter)
    {
      auto node = (*iter)->GetNode ();
      auto sink = (*iter)->GetObject <PacketSink> ();
      std::cout << "Node " << Names::FindName (node) 
                << " received " << sink->GetTotalRx () << " bytes" << std::endl;
    }

  Simulator::Destroy ();

  Names::Clear ();
  return 0;
}

