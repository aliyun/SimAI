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

NS_LOG_COMPONENT_DEFINE ("SocketOptionsIpv4");

void ReceivePacket (Ptr<Socket> socket)
{
  NS_LOG_INFO ("Received one packet!");
  Ptr<Packet> packet = socket->Recv ();
  SocketIpTosTag tosTag;
  if (packet->RemovePacketTag (tosTag))
    {
      NS_LOG_INFO (" TOS = " << (uint32_t)tosTag.GetTos ());
    }
  SocketIpTtlTag ttlTag;
  if (packet->RemovePacketTag (ttlTag))
    {
      NS_LOG_INFO (" TTL = " << (uint32_t)ttlTag.GetTtl ());
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

  //Socket options for IPv4, currently TOS, TTL, RECVTOS, and RECVTTL
  uint32_t ipTos = 0;
  bool ipRecvTos = true;
  uint32_t ipTtl = 0;
  bool ipRecvTtl = true;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("PacketSize", "Packet size in bytes", packetSize);
  cmd.AddValue ("PacketCount", "Number of packets to send", packetCount);
  cmd.AddValue ("Interval", "Interval between packets", packetInterval);
  cmd.AddValue ("IP_TOS", "IP_TOS", ipTos);
  cmd.AddValue ("IP_RECVTOS", "IP_RECVTOS", ipRecvTos);
  cmd.AddValue ("IP_TTL", "IP_TTL", ipTtl);
  cmd.AddValue ("IP_RECVTTL", "IP_RECVTTL", ipRecvTtl);
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
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer i = ipv4.Assign (d);
  serverAddress = Address(i.GetAddress (1));
   
  NS_LOG_INFO ("Create sockets.");
  //Receiver socket on n1
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> recvSink = Socket::CreateSocket (n.Get (1), tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 4477);
  recvSink->SetIpRecvTos (ipRecvTos);
  recvSink->SetIpRecvTtl (ipRecvTtl);
  recvSink->Bind (local);
  recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));

  //Sender socket on n0
  Ptr<Socket> source = Socket::CreateSocket (n.Get (0), tid);
  InetSocketAddress remote = InetSocketAddress (i.GetAddress (1), 4477);

  //Set socket options, it is also possible to set the options after the socket has been created/connected.
  if (ipTos > 0)
    {
      source->SetIpTos (ipTos);
    }

  if (ipTtl > 0)
    {
      source->SetIpTtl (ipTtl);
    }
  source->Connect (remote);

  AsciiTraceHelper ascii;
  csma.EnableAsciiAll (ascii.CreateFileStream ("socket-options-ipv4.tr"));
  csma.EnablePcapAll ("socket-options-ipv4", false);

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
