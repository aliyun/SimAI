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

/* Test program for multi-interface host, static routing

         Destination host (10.20.1.2)
                 |
                 | 10.20.1.0/24
              DSTRTR 
  10.10.1.0/24 /   \  10.10.2.0/24
              / \
           Rtr1    Rtr2
 10.1.1.0/24 |      | 10.1.2.0/24
             |      /
              \    /
             Source
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SocketBoundRoutingExample");

void SendStuff (Ptr<Socket> sock, Ipv4Address dstaddr, uint16_t port);
void BindSock (Ptr<Socket> sock, Ptr<NetDevice> netdev);
void srcSocketRecv (Ptr<Socket> socket);
void dstSocketRecv (Ptr<Socket> socket);

int 
main (int argc, char *argv[])
{

  // Allow the user to override any of the defaults and the above
  // DefaultValue::Bind ()s at run-time, via command-line arguments
  CommandLine cmd (__FILE__);
  cmd.Parse (argc, argv);

  Ptr<Node> nSrc = CreateObject<Node> ();
  Ptr<Node> nDst = CreateObject<Node> ();
  Ptr<Node> nRtr1 = CreateObject<Node> ();
  Ptr<Node> nRtr2 = CreateObject<Node> ();
  Ptr<Node> nDstRtr = CreateObject<Node> ();

  NodeContainer c = NodeContainer (nSrc, nDst, nRtr1, nRtr2, nDstRtr);

  InternetStackHelper internet;
  internet.Install (c);

  // Point-to-point links
  NodeContainer nSrcnRtr1 = NodeContainer (nSrc, nRtr1);
  NodeContainer nSrcnRtr2 = NodeContainer (nSrc, nRtr2);
  NodeContainer nRtr1nDstRtr = NodeContainer (nRtr1, nDstRtr);
  NodeContainer nRtr2nDstRtr = NodeContainer (nRtr2, nDstRtr);
  NodeContainer nDstRtrnDst = NodeContainer (nDstRtr, nDst);

  // We create the channels first without any IP addressing information
  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("2ms"));
  NetDeviceContainer dSrcdRtr1 = p2p.Install (nSrcnRtr1);
  NetDeviceContainer dSrcdRtr2 = p2p.Install (nSrcnRtr2);
  NetDeviceContainer dRtr1dDstRtr = p2p.Install (nRtr1nDstRtr);
  NetDeviceContainer dRtr2dDstRtr = p2p.Install (nRtr2nDstRtr);
  NetDeviceContainer dDstRtrdDst = p2p.Install (nDstRtrnDst);

  Ptr<NetDevice> SrcToRtr1=dSrcdRtr1.Get (0);
  Ptr<NetDevice> SrcToRtr2=dSrcdRtr2.Get (0);

  // Later, we add IP addresses.
  Ipv4AddressHelper ipv4;
  ipv4.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer iSrciRtr1 = ipv4.Assign (dSrcdRtr1);
  ipv4.SetBase ("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer iSrciRtr2 = ipv4.Assign (dSrcdRtr2);
  ipv4.SetBase ("10.10.1.0", "255.255.255.0");
  Ipv4InterfaceContainer iRtr1iDstRtr = ipv4.Assign (dRtr1dDstRtr);
  ipv4.SetBase ("10.10.2.0", "255.255.255.0");
  Ipv4InterfaceContainer iRtr2iDstRtr = ipv4.Assign (dRtr2dDstRtr);
  ipv4.SetBase ("10.20.1.0", "255.255.255.0");
  Ipv4InterfaceContainer iDstRtrDst = ipv4.Assign (dDstRtrdDst);

  Ptr<Ipv4> ipv4Src = nSrc->GetObject<Ipv4> ();
  Ptr<Ipv4> ipv4Rtr1 = nRtr1->GetObject<Ipv4> ();
  Ptr<Ipv4> ipv4Rtr2 = nRtr2->GetObject<Ipv4> ();
  Ptr<Ipv4> ipv4DstRtr = nDstRtr->GetObject<Ipv4> ();
  Ptr<Ipv4> ipv4Dst = nDst->GetObject<Ipv4> ();

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> staticRoutingSrc = ipv4RoutingHelper.GetStaticRouting (ipv4Src);
  Ptr<Ipv4StaticRouting> staticRoutingRtr1 = ipv4RoutingHelper.GetStaticRouting (ipv4Rtr1);
  Ptr<Ipv4StaticRouting> staticRoutingRtr2 = ipv4RoutingHelper.GetStaticRouting (ipv4Rtr2);
  Ptr<Ipv4StaticRouting> staticRoutingDstRtr = ipv4RoutingHelper.GetStaticRouting (ipv4DstRtr);
  Ptr<Ipv4StaticRouting> staticRoutingDst = ipv4RoutingHelper.GetStaticRouting (ipv4Dst);

  // Create static routes from Src to Dst
  staticRoutingRtr1->AddHostRouteTo (Ipv4Address ("10.20.1.2"), Ipv4Address ("10.10.1.2"), 2);
  staticRoutingRtr2->AddHostRouteTo (Ipv4Address ("10.20.1.2"), Ipv4Address ("10.10.2.2"), 2);

  // Two routes to same destination - setting separate metrics. 
  // You can switch these to see how traffic gets diverted via different routes
  staticRoutingSrc->AddHostRouteTo (Ipv4Address ("10.20.1.2"), Ipv4Address ("10.1.1.2"), 1,5);
  staticRoutingSrc->AddHostRouteTo (Ipv4Address ("10.20.1.2"), Ipv4Address ("10.1.2.2"), 2,10);

  // Creating static routes from DST to Source pointing to Rtr1 VIA Rtr2(!)
  staticRoutingDst->AddHostRouteTo (Ipv4Address ("10.1.1.1"), Ipv4Address ("10.20.1.1"), 1);
  staticRoutingDstRtr->AddHostRouteTo (Ipv4Address ("10.1.1.1"), Ipv4Address ("10.10.2.1"), 2);
  staticRoutingRtr2->AddHostRouteTo (Ipv4Address ("10.1.1.1"), Ipv4Address ("10.1.2.1"), 1);

  // There are no apps that can utilize the Socket Option so doing the work directly..
  // Taken from tcp-large-transfer example

  Ptr<Socket> srcSocket = Socket::CreateSocket (nSrc, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  srcSocket->Bind ();
  srcSocket->SetRecvCallback (MakeCallback (&srcSocketRecv));

  Ptr<Socket> dstSocket = Socket::CreateSocket (nDst, TypeId::LookupByName ("ns3::UdpSocketFactory"));
  uint16_t dstport = 12345;
  Ipv4Address dstaddr ("10.20.1.2");
  InetSocketAddress dst = InetSocketAddress (dstaddr, dstport);
  dstSocket->Bind (dst);
  dstSocket->SetRecvCallback (MakeCallback (&dstSocketRecv));

  AsciiTraceHelper ascii;
  p2p.EnableAsciiAll (ascii.CreateFileStream ("socket-bound-static-routing.tr"));
  p2p.EnablePcapAll ("socket-bound-static-routing");

  LogComponentEnableAll (LOG_PREFIX_TIME);
  LogComponentEnable ("SocketBoundRoutingExample", LOG_LEVEL_INFO);

  // First packet as normal (goes via Rtr1)
  Simulator::Schedule (Seconds (0.1),&SendStuff, srcSocket, dstaddr, dstport);
  // Second via Rtr1 explicitly
  Simulator::Schedule (Seconds (1.0),&BindSock, srcSocket, SrcToRtr1);
  Simulator::Schedule (Seconds ( 1.1),&SendStuff, srcSocket, dstaddr, dstport);
  // Third via Rtr2 explicitly
  Simulator::Schedule (Seconds (2.0),&BindSock, srcSocket, SrcToRtr2);
  Simulator::Schedule (Seconds (2.1),&SendStuff, srcSocket, dstaddr, dstport);
  // Fourth again as normal (goes via Rtr1)
  Simulator::Schedule (Seconds (3.0),&BindSock, srcSocket, Ptr<NetDevice>(0));
  Simulator::Schedule (Seconds (3.1),&SendStuff, srcSocket, dstaddr, dstport);
  // If you uncomment what's below, it results in ASSERT failing since you can't 
  // bind to a socket not existing on a node
  // Simulator::Schedule(Seconds(4.0),&BindSock, srcSocket, dDstRtrdDst.Get(0)); 
  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}

void SendStuff (Ptr<Socket> sock, Ipv4Address dstaddr, uint16_t port)
{
  Ptr<Packet> p = Create<Packet> ();
  p->AddPaddingAtEnd (100);
  sock->SendTo (p, 0, InetSocketAddress (dstaddr,port));
  return;
}

void BindSock (Ptr<Socket> sock, Ptr<NetDevice> netdev)
{
  sock->BindToNetDevice (netdev);
  return;
}

void
srcSocketRecv (Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> packet = socket->RecvFrom (from);
  packet->RemoveAllPacketTags ();
  packet->RemoveAllByteTags ();
  NS_LOG_INFO ("Source Received " << packet->GetSize () << " bytes from " << InetSocketAddress::ConvertFrom (from).GetIpv4 ());
  if (socket->GetBoundNetDevice ())
    {
      NS_LOG_INFO ("Socket was bound");
    } 
  else
    {
      NS_LOG_INFO ("Socket was not bound");
    }
}

void
dstSocketRecv (Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> packet = socket->RecvFrom (from);
  packet->RemoveAllPacketTags ();
  packet->RemoveAllByteTags ();
  InetSocketAddress address = InetSocketAddress::ConvertFrom (from);
  NS_LOG_INFO ("Destination Received " << packet->GetSize () << " bytes from " << address.GetIpv4 ());
  NS_LOG_INFO ("Triggering packet back to source node's interface 1");
  SendStuff (socket, Ipv4Address ("10.1.1.1"), address.GetPort ());
}
