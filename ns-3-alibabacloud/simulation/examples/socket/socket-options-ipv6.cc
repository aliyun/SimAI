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
//       n0              n1
//       |               |
//       =================
//              LAN
//
// - UDP flows from n0 to n1

#include <fstream>
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/applications-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SocketOptionsIpv6");

void ReceivePacket (Ptr<Socket> socket)
{
  NS_LOG_INFO ("Received one packet!");
  Ptr<Packet> packet = socket->Recv ();
  SocketIpv6TclassTag tclassTag;
  if (packet->RemovePacketTag (tclassTag))
    {
      NS_LOG_INFO (" TCLASS = " << (uint32_t)tclassTag.GetTclass ());
    }
  SocketIpv6HopLimitTag hoplimitTag;
  if (packet->RemovePacketTag (hoplimitTag))
    {
      NS_LOG_INFO (" HOPLIMIT = " << (uint32_t)hoplimitTag.GetHopLimit ());
    }
}

static void SendPacket (Ptr<Socket> socket, uint32_t pktSize, 
                        uint32_t pktCount, Time pktInterval )
{
  if (pktCount > 0)
    {
      socket->Send (Create<Packet> (pktSize));
      Simulator::Schedule (pktInterval, &SendPacket, 
                           socket, pktSize,pktCount - 1, pktInterval);
    }
  else
    {
      socket->Close ();
    }
}

int 
main (int argc, char *argv[])
{
//
// Allow the user to override any of the defaults and the above Bind() at
// run-time, via command-line arguments
//
  uint32_t packetSize = 1024;
  uint32_t packetCount = 10;
  double packetInterval = 1.0;

  //Socket options for IPv6, currently TCLASS, HOPLIMIT, RECVTCLASS, and RECVHOPLIMIT
  uint32_t ipv6Tclass = 0;
  bool ipv6RecvTclass = true;
  uint32_t ipv6Hoplimit = 0;
  bool ipv6RecvHoplimit = true;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("PacketSize", "Packet size in bytes", packetSize);
  cmd.AddValue ("PacketCount", "Number of packets to send", packetCount);
  cmd.AddValue ("Interval", "Interval between packets", packetInterval);
  cmd.AddValue ("IPV6_TCLASS", "IPV6_TCLASS", ipv6Tclass);
  cmd.AddValue ("IPV6_RECVTCLASS", "IPV6_RECVTCLASS", ipv6RecvTclass);
  cmd.AddValue ("IPV6_HOPLIMIT", "IPV6_HOPLIMIT", ipv6Hoplimit);
  cmd.AddValue ("IPV6_RECVHOPLIMIT", "IPV6_RECVHOPLIMIT", ipv6RecvHoplimit);
  cmd.Parse (argc, argv);

  NS_LOG_INFO ("Create nodes.");
  NodeContainer n;
  n.Create (2);

  InternetStackHelper internet;
  internet.Install (n);

  Address serverAddress;

  NS_LOG_INFO ("Create channels.");
  CsmaHelper csma;
  csma.SetChannelAttribute ("DataRate", DataRateValue (DataRate (5000000)));
  csma.SetChannelAttribute ("Delay", TimeValue (MilliSeconds (2)));
  csma.SetDeviceAttribute ("Mtu", UintegerValue (1400));
  NetDeviceContainer d = csma.Install (n);


  NS_LOG_INFO ("Assign IP Addresses.");
  Ipv6AddressHelper ipv6;
  ipv6.SetBase ("2001:0000:f00d:cafe::", Ipv6Prefix (64));
  Ipv6InterfaceContainer i6 = ipv6.Assign (d);
  serverAddress = Address(i6.GetAddress (1,1));
   
  NS_LOG_INFO ("Create sockets.");
  //Receiver socket on n1
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> recvSink = Socket::CreateSocket (n.Get (1), tid);
  Inet6SocketAddress local = Inet6SocketAddress (Ipv6Address::GetAny (), 4477);
  recvSink->SetIpv6RecvTclass (ipv6RecvTclass);
  recvSink->SetIpv6RecvHopLimit (ipv6RecvHoplimit);
  recvSink->Bind (local);
  recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));

  //Sender socket on n0
  Ptr<Socket> source = Socket::CreateSocket (n.Get (0), tid);
  Inet6SocketAddress remote = Inet6SocketAddress (i6.GetAddress (1,1), 4477);
  //Set socket options, it is also possible to set the options after the socket has been created/connected.
  if (ipv6Tclass != 0)
    {
      source->SetIpv6Tclass (ipv6Tclass);
    }

  if (ipv6Hoplimit > 0)
    {
      source->SetIpv6HopLimit (ipv6Hoplimit);
    }
  source->Connect (remote);

  AsciiTraceHelper ascii;
  csma.EnableAsciiAll (ascii.CreateFileStream ("socket-options-ipv6.tr"));
  csma.EnablePcapAll ("socket-options-ipv6", false);

  //Schedule SendPacket
  Time interPacketInterval = Seconds (packetInterval);
  Simulator::ScheduleWithContext (source->GetNode ()->GetId (),
                                  Seconds (1.0), &SendPacket, 
                                  source, packetSize, packetCount, interPacketInterval);

  NS_LOG_INFO ("Run Simulation.");
  Simulator::Run ();
  Simulator::Destroy ();
  NS_LOG_INFO ("Done.");
}
