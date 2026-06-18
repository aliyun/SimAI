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
 * Based on simple-global-routing.cc
 * The Tunnel class adds two tunnels, n0<=>n3 and n1<=>n3
 */

// Network topology
//
//  n0
//     \ 5 Mb/s, 2ms
//      \          1.5Mb/s, 10ms
//       n2 -------------------------n3
//      /
//     / 5 Mb/s, 2ms
//   n1
//
// - all links are point-to-point links with indicated one-way BW/delay
// - Tracing of queues and packet receptions to file "virtual-net-device.tr"

// Tunneling changes (relative to the simple-global-routing example):
// - n0 will receive an extra virtual interface with hardcoded inner-tunnel address 11.0.0.1
// - n1 will also receive an extra virtual interface with the same inner-tunnel address 11.0.0.1
// - n3 will receive an extra virtual interface with inner-tunnel address 11.0.0.254
// - The flows will be between 11.0.0.x (inner-tunnel) addresses instead of 10.1.x.y ones
// - n3 will decide, on a per-packet basis, via random number, whether to
//   send the packet to n0 or to n1.
//
// Note: here we create a tunnel where IP packets are tunneled over
// UDP/IP, but tunneling directly IP-over-IP would also be possible;
// see src/node/ipv4-raw-socket-factory.h.

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/virtual-net-device.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("VirtualNetDeviceExample");

/**
 * \ingroup virtual-net-device
 * 
 * Tunnel class - its goal is to create and manage the tunnels between endpoints.
 */
class Tunnel
{
  Ptr<Socket> m_n3Socket; //!< Socket on the N3 node
  Ptr<Socket> m_n0Socket; //!< Socket on the N0 node
  Ptr<Socket> m_n1Socket; //!< Socket on the N1 node
  Ipv4Address m_n3Address;  //!< Address of the N3 node
  Ipv4Address m_n0Address;  //!< Address of the N0 node
  Ipv4Address m_n1Address;  //!< Address of the N1 node
  Ptr<UniformRandomVariable> m_rng; //!< Random number generator
  Ptr<VirtualNetDevice> m_n0Tap;  //!< VirtualNetDevice on the N0 node
  Ptr<VirtualNetDevice> m_n1Tap;  //!< VirtualNetDevice on the N1 node
  Ptr<VirtualNetDevice> m_n3Tap;  //!< VirtualNetDevice on the N3 node

  /**
   * Send a packet through the N0 VirtualNetDevice
   * \param packet Packet to send
   * \param source Source IPv4 address
   * \param dest Destination IPv4 address
   * \param protocolNumber Protocol number
   * \return true (always)
   */
  bool
  N0VirtualSend (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber)
  {
    NS_LOG_DEBUG ("Send to " << m_n3Address << ": " << *packet);
    m_n0Socket->SendTo (packet, 0, InetSocketAddress (m_n3Address, 667));
    return true;
  }

  /**
   * Send a packet through the N1 VirtualNetDevice
   * \param packet Packet to send
   * \param source Source IPv4 address
   * \param dest Destination IPv4 address
   * \param protocolNumber Protocol number
   * \return true (always)
   */
  bool
  N1VirtualSend (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber)
  {
    NS_LOG_DEBUG ("Send to " << m_n3Address << ": " << *packet);
    m_n1Socket->SendTo (packet, 0, InetSocketAddress (m_n3Address, 667));
    return true;
  }

  /**
   * Send a packet through the N3 VirtualNetDevice
   * \param packet Packet to send
   * \param source Source IPv4 address
   * \param dest Destination IPv4 address
   * \param protocolNumber Protocol number
   * \return true (always)
   */
  bool
  N3VirtualSend (Ptr<Packet> packet, const Address& source, const Address& dest, uint16_t protocolNumber)
  {
    if (m_rng->GetValue () < 0.25)
      {
        NS_LOG_DEBUG ("Send to " << m_n0Address << ": " << *packet);
        m_n3Socket->SendTo (packet, 0, InetSocketAddress (m_n0Address, 667));
      }
    else 
      {
        NS_LOG_DEBUG ("Send to " << m_n1Address << ": " << *packet);
        m_n3Socket->SendTo (packet, 0, InetSocketAddress (m_n1Address, 667));
      }
    return true;
  }

  /**
   * Receive a packet on the N3 VirtualNetDevice
   * \param socket Receiving socket
   */
  void N3SocketRecv (Ptr<Socket> socket)
  {
    Ptr<Packet> packet = socket->Recv (65535, 0);
    NS_LOG_DEBUG ("N3SocketRecv: " << *packet);
    m_n3Tap->Receive (packet, 0x0800, m_n3Tap->GetAddress (), m_n3Tap->GetAddress (), NetDevice::PACKET_HOST);
  }

  /**
   * Receive a packet on the N0 VirtualNetDevice
   * \param socket Receiving socket
   */
  void N0SocketRecv (Ptr<Socket> socket)
  {
    Ptr<Packet> packet = socket->Recv (65535, 0);
    NS_LOG_DEBUG ("N0SocketRecv: " << *packet);
    m_n0Tap->Receive (packet, 0x0800, m_n0Tap->GetAddress (), m_n0Tap->GetAddress (), NetDevice::PACKET_HOST);
  }

  /**
   * Receive a packet on the N1 VirtualNetDevice
   * \param socket Receiving socket
   */
  void N1SocketRecv (Ptr<Socket> socket)
  {
    Ptr<Packet> packet = socket->Recv (65535, 0);
    NS_LOG_DEBUG ("N1SocketRecv: " << *packet);
    m_n1Tap->Receive (packet, 0x0800, m_n1Tap->GetAddress (), m_n1Tap->GetAddress (), NetDevice::PACKET_HOST);
  }

public:

  /**
   * Constructor
   * \param n3 Pointer of Node 3
   * \param n0 Pointer of Node 0
   * \param n1 Pointer of Node 1
   * \param n3Addr IPv4 address of Node 3
   * \param n0Addr IPv4 address of Node 0
   * \param n1Addr IPv4 address of Node 1
   */
  Tunnel (Ptr<Node> n3, Ptr<Node> n0, Ptr<Node> n1,
          Ipv4Address n3Addr, Ipv4Address n0Addr, Ipv4Address n1Addr)
    : m_n3Address (n3Addr), m_n0Address (n0Addr), m_n1Address (n1Addr)
  {
    m_rng = CreateObject<UniformRandomVariable> ();
    m_n3Socket = Socket::CreateSocket (n3, TypeId::LookupByName ("ns3::UdpSocketFactory"));
    m_n3Socket->Bind (InetSocketAddress (Ipv4Address::GetAny (), 667));
    m_n3Socket->SetRecvCallback (MakeCallback (&Tunnel::N3SocketRecv, this));

    m_n0Socket = Socket::CreateSocket (n0, TypeId::LookupByName ("ns3::UdpSocketFactory"));
    m_n0Socket->Bind (InetSocketAddress (Ipv4Address::GetAny (), 667));
    m_n0Socket->SetRecvCallback (MakeCallback (&Tunnel::N0SocketRecv, this));

    m_n1Socket = Socket::CreateSocket (n1, TypeId::LookupByName ("ns3::UdpSocketFactory"));
    m_n1Socket->Bind (InetSocketAddress (Ipv4Address::GetAny (), 667));
    m_n1Socket->SetRecvCallback (MakeCallback (&Tunnel::N1SocketRecv, this));

    // n0 tap device
    m_n0Tap = CreateObject<VirtualNetDevice> ();
    m_n0Tap->SetAddress (Mac48Address ("11:00:01:02:03:01"));
    m_n0Tap->SetSendCallback (MakeCallback (&Tunnel::N0VirtualSend, this));
    n0->AddDevice (m_n0Tap);
    Ptr<Ipv4> ipv4 = n0->GetObject<Ipv4> ();
    uint32_t i = ipv4->AddInterface (m_n0Tap);
    ipv4->AddAddress (i, Ipv4InterfaceAddress (Ipv4Address ("11.0.0.1"), Ipv4Mask ("255.255.255.0")));
    ipv4->SetUp (i);

    // n1 tap device
    m_n1Tap = CreateObject<VirtualNetDevice> ();
    m_n1Tap->SetAddress (Mac48Address ("11:00:01:02:03:02"));
    m_n1Tap->SetSendCallback (MakeCallback (&Tunnel::N1VirtualSend, this));
    n1->AddDevice (m_n1Tap);
    ipv4 = n1->GetObject<Ipv4> ();
    i = ipv4->AddInterface (m_n1Tap);
    ipv4->AddAddress (i, Ipv4InterfaceAddress (Ipv4Address ("11.0.0.1"), Ipv4Mask ("255.255.255.0")));
    ipv4->SetUp (i);

    // n3 tap device
    m_n3Tap = CreateObject<VirtualNetDevice> ();
    m_n3Tap->SetAddress (Mac48Address ("11:00:01:02:03:04"));
    m_n3Tap->SetSendCallback (MakeCallback (&Tunnel::N3VirtualSend, this));
    n3->AddDevice (m_n3Tap);
    ipv4 = n3->GetObject<Ipv4> ();
    i = ipv4->AddInterface (m_n3Tap);
    ipv4->AddAddress (i, Ipv4InterfaceAddress (Ipv4Address ("11.0.0.254"), Ipv4Mask ("255.255.255.0")));
    ipv4->SetUp (i);

  }


};



int 
main (int argc, char *argv[])
{
  // Users may find it convenient to turn on explicit logging
  // for selected modules; the below lines suggest how to do this
#if 0 
  LogComponentEnable ("VirtualNetDeviceExample", LOG_LEVEL_INFO);
#endif
  Packet::EnablePrinting ();


  // Set up some default values for the simulation.  Use the 
  Config::SetDefault ("ns3::OnOffApplication::PacketSize", UintegerValue (210));
  Config::SetDefault ("ns3::OnOffApplication::DataRate", StringValue ("448kb/s"));

  // Allow the user to override any of the defaults and the above
  // Config::SetDefault ()s at run-time, via command-line arguments
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  NS_LOG_INFO ("Create nodes.");
  NodeContainer c;
  c.Create (4);
  NodeContainer n0n2 = NodeContainer (c.Get (0), c.Get (2));
  NodeContainer n1n2 = NodeContainer (c.Get (1), c.Get (2));
  NodeContainer n3n2 = NodeContainer (c.Get (3), c.Get (2));

  InternetStackHelper internet;
  internet.Install (c);

  // We create the channels first without any IP addressing information
  NS_LOG_INFO ("Create channels.");
  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("2ms"));
  NetDeviceContainer d0d2 = p2p.Install (n0n2);

  NetDeviceContainer d1d2 = p2p.Install (n1n2);

  p2p.SetDeviceAttribute ("DataRate", StringValue ("1500kbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("10ms"));
  NetDeviceContainer d3d2 = p2p.Install (n3n2);

  // Later, we add IP addresses.
  NS_LOG_INFO ("Assign IP Addresses.");
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i0i2 = ipv4.Assign (d0d2);

  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer i1i2 = ipv4.Assign (d1d2);

  ipv4.SetBase ("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer i3i2 = ipv4.Assign (d3d2);

  // Create router nodes, initialize routing database and set up the routing
  // tables in the nodes.
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Add the tunnels n0<=>n3 and n1<=>n3
  Tunnel tunnel (c.Get (3), c.Get (0), c.Get (1),
                 i3i2.GetAddress (0), i0i2.GetAddress (0), i1i2.GetAddress (0));

  // Create the OnOff application to send UDP datagrams of size
  // 210 bytes at a rate of 448 Kb/s
  NS_LOG_INFO ("Create Applications.");
  uint16_t port = 9;   // Discard port (RFC 863)
  OnOffHelper onoff ("ns3::UdpSocketFactory", 
                     Address (InetSocketAddress (Ipv4Address ("11.0.0.254"), port)));
  onoff.SetConstantRate (DataRate ("448kb/s"));
  ApplicationContainer apps = onoff.Install (c.Get (0));
  apps.Start (Seconds (1.0));
  apps.Stop (Seconds (10.0));

  // Create a packet sink to receive these packets
  PacketSinkHelper sink ("ns3::UdpSocketFactory",
                         Address (InetSocketAddress (Ipv4Address::GetAny (), port)));
  apps = sink.Install (c.Get (3));
  apps.Start (Seconds (1.0));
  //apps.Stop (Seconds (10.0));

  // Create a similar flow from n3 to n1, starting at time 1.1 seconds
  onoff.SetAttribute ("Remote", 
                      AddressValue (InetSocketAddress (Ipv4Address ("11.0.0.1"), port)));
  apps = onoff.Install (c.Get (3));
  apps.Start (Seconds (1.1));
  apps.Stop (Seconds (10.0));

  // Create a packet sink to receive these packets
  apps = sink.Install (c.Get (1));
  apps.Start (Seconds (1.1));
  //apps.Stop (Seconds (10.0));

  AsciiTraceHelper ascii;
  p2p.EnableAsciiAll (ascii.CreateFileStream ("virtual-net-device.tr"));
  p2p.EnablePcapAll ("virtual-net-device");

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");

  return 0;
}
