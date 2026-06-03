/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/* This program is free software; you can redistribute it and/or modify
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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 * USA
 *
 */

// OLSR was observed to not converge in simple 3-nodes varying topology.
// https://www.nsnam.org/bugzilla/show_bug.cgi?id=780
// tcpdump -r bug780-0-0.pcap -nn -tt icmp | wc
// should show about 395 packets; there is a ping outage from time
// 123-127 due to the mobility.

#include "ns3/test.h"
#include "ns3/olsr-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/olsr-routing-protocol.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/boolean.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/simple-net-device.h"
#include "ns3/icmpv4.h"
#include "bug780-test.h"

namespace ns3 {
namespace olsr {

Bug780Test::Bug780Test () :
  TestCase ("Test OLSR bug 780"),
  m_time (Seconds (200.0)), m_seq (0), m_recvCount (0)
{
}

Bug780Test::~Bug780Test ()
{
}

void
Bug780Test::DoRun ()
{
  RngSeedManager::SetSeed (12345);
  RngSeedManager::SetRun (12345);
  CreateNodes ();

  Simulator::Stop (m_time);
  Simulator::Run ();

  NS_TEST_EXPECT_MSG_EQ (m_recvCount, 192, "192 out of 200 ping received.");

  Simulator::Destroy ();
}

void
Bug780Test::CreateNodes (void)
{
  NodeContainer c;
  c.Create (3);

  // install TCP/IP & OLSR
  OlsrHelper olsr;
  InternetStackHelper internet;
  internet.SetRoutingHelper (olsr);
  internet.Install (c);
  int64_t streamsUsed = olsr.AssignStreams (c, 0);
  NS_TEST_EXPECT_MSG_EQ (streamsUsed, 3, "Should have assigned 3 streams");

  // create channel & devices
  SimpleNetDeviceHelper simpleNetHelper;
  simpleNetHelper.SetDeviceAttribute ("DataRate", StringValue ("10Mbps"));
  simpleNetHelper.SetChannelAttribute ("Delay", StringValue ("2ms"));
  NetDeviceContainer nd = simpleNetHelper.Install (c);

  Ipv4AddressHelper addressAdhoc;
  addressAdhoc.SetBase ("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer adhocInterfaces;
  adhocInterfaces = addressAdhoc.Assign (nd);

  // Blacklist some devices (equivalent to Wireless out of range)
  Ptr<SimpleNetDevice> nd0 = DynamicCast<SimpleNetDevice> (nd.Get (0));
  Ptr<SimpleNetDevice> nd2 = DynamicCast<SimpleNetDevice> (nd.Get (2));
  Ptr<SimpleChannel> ch = DynamicCast<SimpleChannel> (nd.Get (0)->GetChannel ());

  Simulator::Schedule (Seconds (100.0), &SimpleChannel::BlackList, ch, nd0, nd2);
  Simulator::Schedule (Seconds (100.0), &SimpleChannel::BlackList, ch, nd2, nd0);

  // 3. Setup ping
  m_socket = Socket::CreateSocket (c.Get (0), TypeId::LookupByName ("ns3::Ipv4RawSocketFactory"));
  m_socket->SetAttribute ("Protocol", UintegerValue (1)); // icmp
  m_socket->SetRecvCallback (MakeCallback (&Bug780Test::Receive, this));
  InetSocketAddress src = InetSocketAddress (Ipv4Address::GetAny (), 0);
  m_socket->Bind (src);
  InetSocketAddress dst = InetSocketAddress (adhocInterfaces.GetAddress (2), 0);
  m_socket->Connect (dst);

  SendPing ();
}

void
Bug780Test::SendPing ()
{
  if (Simulator::Now () >= m_time)
    {
      return;
    }

  Ptr<Packet> p = Create<Packet> ();
  Icmpv4Echo echo;
  echo.SetSequenceNumber (m_seq);
  m_seq++;
  echo.SetIdentifier (0);

  Ptr<Packet> dataPacket = Create<Packet> (56);
  echo.SetData (dataPacket);
  p->AddHeader (echo);
  Icmpv4Header header;
  header.SetType (Icmpv4Header::ICMPV4_ECHO);
  header.SetCode (0);
  if (Node::ChecksumEnabled ())
    {
      header.EnableChecksum ();
    }
  p->AddHeader (header);
  m_socket->Send (p, 0);
  Simulator::Schedule (Seconds (1), &Bug780Test::SendPing, this);
}

void
Bug780Test::Receive (Ptr<Socket> socket)
{
  while (m_socket->GetRxAvailable () > 0)
    {
      Address from;
      Ptr<Packet> p = m_socket->RecvFrom (0xffffffff, 0, from);

      NS_ASSERT (InetSocketAddress::IsMatchingType (from));
      InetSocketAddress realFrom = InetSocketAddress::ConvertFrom (from);
      NS_ASSERT (realFrom.GetPort () == 1); // protocol should be icmp.
      Ipv4Header ipv4;
      p->RemoveHeader (ipv4);
      NS_ASSERT (ipv4.GetProtocol () == 1); // protocol should be icmp.
      Icmpv4Header icmp;
      p->RemoveHeader (icmp);
      if (icmp.GetType () == Icmpv4Header::ICMPV4_ECHO_REPLY)
        {
          m_recvCount++;
        }
    }
}

} // namespace olsr
} // namespace ns3

