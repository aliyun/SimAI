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

// Network topology
//
//       n0    n1   n2   n3
//       |     |    |    |
//       =================
//              LAN
//
// This program demonstrates some basic use of the Object names capability
//

#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ObjectNamesExample");

uint32_t bytesReceived = 0;

void 
RxEvent (std::string context, Ptr<const Packet> packet)
{
  std::cout << Simulator::Now ().GetSeconds () << "s " << context << " packet size " << packet->GetSize () << std::endl;
  bytesReceived += packet->GetSize ();
}

int 
main (int argc, char *argv[])
{
  bool outputValidated = true;

  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  NodeContainer n;
  n.Create (4);

  //
  // We're going to use the zeroth node in the container as the client, and
  // the first node as the server.  Add some "human readable" names for these
  // nodes.  The names below will go into the name system as "/Names/clientZero"
  // and "/Names/server", but note that the Add function assumes that if you
  // omit the leading "/Names/" the remaining string is assumed to be rooted
  // in the "/Names" namespace. The following calls,
  //
  //  Names::Add ("clientZero", n.Get (0));
  //  Names::Add ("/Names/clientZero", n.Get (0));
  //
  // will produce identical results.
  //
  Names::Add ("clientZero", n.Get (0));
  Names::Add ("/Names/server", n.Get (1));

  //
  // It is possible to rename a node that has been previously named.  This is
  // useful in automatic name generation.  You can automatically generate node
  // names such as, "node-0", "node-1", etc., and then go back and change
  // the name of some distinguished node to another value --  "access-point" 
  // for example.  We illustrate this by just changing the client's name.
  // As is typical of the object name service, you can either provide or elide
  // the "/Names" prefix as you choose.
  //
  Names::Rename ("clientZero", "client");

  InternetStackHelper internet;
  internet.Install (n);

  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1400));
  NetDeviceContainer d = csma.Install (n);

  //
  // Add some human readable names for the devices we'll be interested in.
  // We add the names to the name space "under" the nodes we created above.
  // This has the effect of making "/Names/client/eth0" and "/Names/server/eth0".
  // In this case, we again omit the "/Names/" prefix on one call to illustrate
  // the shortcut.
  //
  Names::Add ("/Names/client/eth0", d.Get (0));
  Names::Add ("server/eth0", d.Get (1));

  //
  // You can use the object names that you've assigned in calls to the Config
  // system to set Object Attributes.  For example, you can set the Mtu 
  // Attribute of a Csma devices using the object naming service.  Note that 
  // in this case, the "/Names" prefix is always required since the _Config_ 
  // system always expects to see a fully qualified path name.
  //

  Ptr<CsmaNetDevice> csmaNetDevice = d.Get (0)->GetObject<CsmaNetDevice> ();
  UintegerValue val;
  csmaNetDevice->GetAttribute ("Mtu", val);
  std::cout << "MTU on device 0 before configuration is " << val.Get () << std::endl;
  
  Config::Set ("/Names/client/eth0/Mtu", UintegerValue (1234));

  // Check the attribute again
  csmaNetDevice->GetAttribute ("Mtu", val);
  std::cout << "MTU on device 0 after configuration is " << val.Get () << std::endl;

  if (val.Get () != 1234)
    {
      outputValidated = false;
    }

  //
  // You can mix and match names and Attributes in calls to the Config system.
  // For example, if "eth0" is a named object, you can get to its parent through
  // a different namespace.  For example, you could use the NodeList namespace
  // to get to the server node, and then continue seamlessly adding named objects
  // in the path. This is not nearly as readable as the previous version, but it
  // illustrates how you can mix and match object names and Attribute names.
  // Note that the config path now begins with a path in the "/NodeList" 
  // namespace.
  //
  Config::Set ("/NodeList/1/eth0/Mtu", UintegerValue (1234));

  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = ipv4.Assign (d);

  uint16_t port = 9;
  UdpEchoServerHelper server (port);
  //
  // Install the UdpEchoServer application on the server node using its name
  // directly.
  //
  ApplicationContainer apps = server.Install ("/Names/server");
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  uint32_t packetSize = 1024;
  uint32_t maxPacketCount = 1;
  Time interPacketInterval = Seconds (1.);
  UdpEchoClientHelper client (i.GetAddress (1), port);
  client.SetAttribute ("MaxPackets", UintegerValue (maxPacketCount));
  client.SetAttribute ("Interval", TimeValue (interPacketInterval));
  client.SetAttribute ("PacketSize", UintegerValue (packetSize));
  //
  // Install the UdpEchoClient application on the server node using its name
  // directly.
  //
  apps = client.Install ("/Names/client");
  apps.Start (Seconds (2.0));
  apps.Stop (Seconds (10.0));

  //
  // Use the Config system to connect a trace source using the object name
  // service to specify the path.  Note that in this case, the "/Names"
  // prefix is always required since the _Config_ system always expects to 
  // see a fully qualified path name 
  //
  Config::Connect ("/Names/client/eth0/MacRx", MakeCallback (&RxEvent));

  //
  // Set up some pcap tracing on the CSMA devices.  The names of the trace 
  // files will automatically correspond to the object names if present.
  // In this case, you will find trace files called:
  //
  //   object-names-client-eth0.pcap
  //   object-names-server-eth0.pcap
  //
  // since those nodes and devices have had names associated with them.  You
  // will also see:
  //
  //   object-names-2-1.pcap
  //   object-names-3-1.pcap
  //
  // since nodes two and three have no associated names.
  //
  csma.EnablePcapAll ("object-names");

  //
  // We can also create a trace file with a name we completely control by
  // overriding a couple of default parameters.
  //
  csma.EnablePcap ("client-device.pcap", d.Get (0), false, true);

  std::cout << "Running simulation..." << std::endl;
  Simulator::Run ();
  Simulator::Destroy ();

  // Expected to see ARP exchange and one packet
  // 64 bytes (minimum Ethernet frame size) x 2, plus (1024 + 8 + 20 + 18)
  if (bytesReceived != (64 + 64 + 1070))
    {
      outputValidated = false;
    }

  if (outputValidated == false)
    {
      std::cerr << "Program internal checking failed; returning with error" << std::endl; 
      return (1);
    }
}
