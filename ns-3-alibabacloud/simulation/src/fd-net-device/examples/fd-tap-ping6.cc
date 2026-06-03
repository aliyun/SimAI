/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 University of Washington, 2012 INRIA
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
 */

// Allow ns-3 to ping a TAP device in the host machine.
//
//   -------------------------------------------------
//   | ns-3 simulation                               |
//   |                                               |
//   |  -------                        --------      |
//   | | node  |                      |  node  |     |
//   | | (r)   |                      |  (n)   |     |
//   | |       |                      |        |     |
//   |  ------- --------               --------      |
//   | | fd-   | csma-  |             | csma-  |     |
//   | | net-  | net-   |             | net-   |     |
//   | | device| device |             | device |     |
//   |  ------- --------               --------      |
//   |   |          |____csma channel_____|          |
//   |   |                                           | 
//   ----|------------------------------------------
//   |  ---            |
//   | |   |           |
//   | |TAP|           |
//   | |   |           |
//   |  ---            |
//   |                 |
//   |  host           |
//   ------------------
//                 
//                 

#include <sstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/fd-net-device-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TAPPing6Example");

int
main (int argc, char *argv[])
{
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);
  
  NS_LOG_INFO ("Ping6 Emulation Example with TAP");

  //
  // Since we are using a real piece of hardware we need to use the realtime
  // simulator.
  //
  GlobalValue::Bind ("SimulatorImplementationType", StringValue ("ns3::RealtimeSimulatorImpl"));

  //
  // Since we are going to be talking to real-world machines, we need to enable
  // calculation of checksums in our protocols.
  //
  GlobalValue::Bind ("ChecksumEnabled", BooleanValue (true));

  //
  // Create the two nodes.
  //
  Ptr<Node> n = CreateObject<Node> ();
  Ptr<Node> r = CreateObject<Node> ();
  NodeContainer net (n, r);

  //
  // Install IPv6 stack.
  //
  InternetStackHelper internetv6;
  internetv6.Install (net);

  //
  // Create CSMA channel.
  //
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (5000000));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  NetDeviceContainer devs = csma.Install (net);

  //
  // Assign IPv6 addresses.
  //
  Ipv6AddressHelper ipv6;
 
  ipv6.SetBase (Ipv6Address ("2001:0DB8:1::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer i1 = ipv6.Assign (devs);
  i1.SetForwarding (1, true);
  i1.SetDefaultRouteInAllNodes (1);
  
  ipv6.SetBase (Ipv6Address ("2001:0DB8:2::"), Ipv6Prefix (64));
  Ipv6Address tapAddr = ipv6.NewAddress ();
  std::stringstream ss;
  std::string tapIp; 
  tapAddr.Print (ss);
  ss >> tapIp;

  //
  // Create FdNetDevice.
  //
  TapFdNetDeviceHelper helper;
  helper.SetDeviceName ("tap0");
  helper.SetTapIpv6Address (tapIp.c_str ());
  helper.SetTapIpv6Prefix (64);

  NetDeviceContainer fdevs = helper.Install (r);
  Ptr<NetDevice> device = fdevs.Get (0);
  Ptr<FdNetDevice> fdevice = device->GetObject<FdNetDevice> ();
  fdevice-> SetIsMulticast (true);
  Ipv6InterfaceContainer i2 = ipv6.Assign (fdevs);
  i2.SetForwarding (0, true);
  i2.SetDefaultRouteInAllNodes (0);

  //
  // Create the Ping6 application.
  //
  uint32_t packetSize = 1024;
  uint32_t maxPacketCount = 1;
  Time interPacketInterval = Seconds (1.0);

  Ping6Helper ping6;

  ping6.SetRemote (tapIp.c_str ());
  
  ping6.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  ping6.SetAttribute ("Interval", TimeValue (interPacketInterval));
  ping6.SetAttribute ("PacketSize", UintegerValue (packetSize));
  ApplicationContainer apps = ping6.Install (n);
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (20.0));

  AsciiTraceHelper ascii;
  csma.EnableAsciiAll (ascii.CreateFileStream ("csma-ping6.tr"));
  csma.EnablePcapAll ("csma-ping6", true);

  //
  // Enable a promiscuous pcap trace to see what is coming and going on in the fd-net-device.
  //
  helper.EnablePcap ("fd-ping6", fdevice, true);

  //
  // Run the experiment.
  //
  NS_LOG_INFO ("Run Emulation.");
  Simulator::Stop (Seconds (200.0));
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}
