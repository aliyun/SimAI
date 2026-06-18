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
 * Authors: Lalith Suresh <suresh.lalith@gmail.com>
 */

// Scenario:
//
// (Click)       CSMA    (non-Click)
//    A   ================   B
// (172.16.1.1)         (172.16.1.2)
//    (eth0)
//
//

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/click-internet-stack-helper.h"
#include "ns3/log.h"

using namespace ns3;

void ReceivePacket (Ptr<Socket> socket)
{
  NS_LOG_UNCOND ("Received one packet!");
}

int main (int argc, char *argv[])
{
#ifdef NS3_CLICK
  std::string clickConfigFolder = "src/click/examples";

  CommandLine cmd (__FILE__);
  cmd.AddValue ("clickConfigFolder", "Base folder for click configuration files", clickConfigFolder);
  cmd.Parse (argc, argv);

  NodeContainer csmaNodes;
  csmaNodes.Create (2);

  // Setup CSMA channel between the nodes
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  NetDeviceContainer csmaDevices = csma.Install (csmaNodes);

  // Install normal internet stack on node B
  InternetStackHelper internet;
  internet.Install (csmaNodes.Get (1));

  // Install Click on node A
  ClickInternetStackHelper clickinternet;
  clickinternet.SetClickFile (csmaNodes.Get (0), clickConfigFolder + "/nsclick-lan-single-interface.click");
  clickinternet.SetRoutingTableElement (csmaNodes.Get (0), "rt");
  clickinternet.Install (csmaNodes.Get (0));

  // Configure IP addresses for the nodes
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("172.16.1.0", "255.255.255.0");
  ipv4.Assign (csmaDevices);

  // Configure traffic application and sockets
  Address LocalAddress (InetSocketAddress (Ipv4Address::GetAny (), 50000));
  PacketSinkHelper packetSinkHelper ("ns3::TcpSocketFactory", LocalAddress);
  ApplicationContainer recvapp = packetSinkHelper.Install (csmaNodes.Get (1));
  recvapp.Start (Seconds (5.0));
  recvapp.Stop (Seconds (10.0));

  OnOffHelper onOffHelper ("ns3::TcpSocketFactory", Address ());
  onOffHelper.SetAttribute ("OnTime", StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
  onOffHelper.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));

  ApplicationContainer appcont;

  AddressValue remoteAddress (InetSocketAddress (Ipv4Address ("172.16.1.2"), 50000));
  onOffHelper.SetAttribute ("Remote", remoteAddress);
  appcont.Add (onOffHelper.Install (csmaNodes.Get (0)));

  appcont.Start (Seconds (5.0));
  appcont.Stop (Seconds (10.0));

  // For tracing
  csma.EnablePcap ("nsclick-simple-lan", csmaDevices, false);

  Simulator::Stop (Seconds (20.0));
  Simulator::Run ();

  Simulator::Destroy ();
  return 0;
#else
  NS_FATAL_ERROR ("Can't use ns-3-click without NSCLICK compiled in");
#endif
}
