/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' di Firenze, Italy
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
 */

#include "ns3/test.h"
#include "ns3/socket-factory.h"
#include "ns3/simulator.h"
#include "ns3/socket.h"
#include "ns3/boolean.h"

#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv6-address-helper.h"
#include "ns3/icmpv6-l4-protocol.h"

#include "ns3/sixlowpan-net-device.h"
#include "ns3/sixlowpan-header.h"
#include "ns3/sixlowpan-helper.h"
#include "mock-net-device.h"

#include <string>
#include <limits>
#include <vector>

using namespace ns3;

/**
 * \ingroup sixlowpan-tests
 *
 * \brief 6LoWPAN IPHC stateful compression Test
 */
class SixlowpanIphcStatefulImplTest : public TestCase
{
  /**
   * \brief Structure to hold the Rx/Tx packets.
   */
  typedef struct
  {
    Ptr<Packet> packet;   /**< Packet data */
    Address src;   /**< Source address */
    Address dst;   /**< Destination address */
  } Data;


  std::vector<Data> m_txPackets; //!< Transmitted packets
  std::vector<Data> m_rxPackets; //!< Received packets

  /**
   * Receive from a MockDevice.
   * \param device a pointer to the net device which is calling this function
   * \param packet the packet received
   * \param protocol the 16 bit protocol number associated with this packet.
   * \param source the address of the sender
   * \param destination the address of the receiver
   * \param packetType type of packet received (broadcast/multicast/unicast/otherhost)
   * \returns true.
   */
  bool ReceiveFromMockDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                              Address const &source, Address const &destination, NetDevice::PacketType packetType);

  /**
   * Promiscuous receive from a SixLowPanNetDevice.
   * \param device a pointer to the net device which is calling this function
   * \param packet the packet received
   * \param protocol the 16 bit protocol number associated with this packet.
   * \param source the address of the sender
   * \param destination the address of the receiver
   * \param packetType type of packet received (broadcast/multicast/unicast/otherhost)
   * \returns true.
   */
  bool PromiscReceiveFromSixLowPanDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                                          Address const &source, Address const &destination, NetDevice::PacketType packetType);

  /**
   * Send one packet.
   * \param device the device to send from
   * \param from sender address
   * \param to destination address
   */
  void SendOnePacket (Ptr<NetDevice> device, Ipv6Address from, Ipv6Address to);

  NetDeviceContainer m_mockDevices; //!< MockNetDevice container
  NetDeviceContainer m_sixDevices; //!< SixLowPanNetDevice container

public:
  virtual void DoRun (void);
  SixlowpanIphcStatefulImplTest ();
};

SixlowpanIphcStatefulImplTest::SixlowpanIphcStatefulImplTest ()
  : TestCase ("Sixlowpan IPHC stateful implementation")
{
}

bool
SixlowpanIphcStatefulImplTest::ReceiveFromMockDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                                                      Address const &source, Address const &destination, NetDevice::PacketType packetType)
{
  Data incomingPkt;
  incomingPkt.packet = packet->Copy ();
  incomingPkt.src = source;
  incomingPkt.dst = destination;
  m_txPackets.push_back (incomingPkt);

  Ptr<MockNetDevice> mockDev = DynamicCast<MockNetDevice> (m_mockDevices.Get(1));
  if (mockDev)
    {
      uint32_t id = mockDev->GetNode ()->GetId ();
      Simulator::ScheduleWithContext (id, Time(1), &MockNetDevice::Receive, mockDev, incomingPkt.packet, protocol, destination, source, packetType);
    }
  return true;
}

bool
SixlowpanIphcStatefulImplTest::PromiscReceiveFromSixLowPanDevice (Ptr<NetDevice> device, Ptr<const Packet> packet, uint16_t protocol,
                                                                  Address const &source, Address const &destination, NetDevice::PacketType packetType)
{
  Data incomingPkt;
  incomingPkt.packet = packet->Copy ();
  incomingPkt.src = source;
  incomingPkt.dst = destination;
  m_rxPackets.push_back (incomingPkt);

  return true;
}

void
SixlowpanIphcStatefulImplTest::SendOnePacket (Ptr<NetDevice> device, Ipv6Address from, Ipv6Address to)
{
  Ptr<Packet> pkt = Create<Packet> (10);
  Ipv6Header ipHdr;
  ipHdr.SetSource (from);
  ipHdr.SetDestination (to);
  ipHdr.SetHopLimit (64);
  ipHdr.SetPayloadLength (10);
  ipHdr.SetNextHeader (0xff);
  pkt->AddHeader (ipHdr);

  device->Send (pkt, Mac48Address ("00:00:00:00:00:02"), 0);
}

void
SixlowpanIphcStatefulImplTest::DoRun (void)
{
  NodeContainer nodes;
  nodes.Create(2);

  // First node, setup NetDevices.
  Ptr<MockNetDevice> mockNetDevice0 = CreateObject<MockNetDevice> ();
  nodes.Get (0)->AddDevice (mockNetDevice0);
  mockNetDevice0->SetNode (nodes.Get (0));
  mockNetDevice0->SetAddress (Mac48Address ("00:00:00:00:00:01"));
  mockNetDevice0->SetMtu (150);
  mockNetDevice0->SetSendCallback ( MakeCallback (&SixlowpanIphcStatefulImplTest::ReceiveFromMockDevice, this) );
  m_mockDevices.Add (mockNetDevice0);

  // Second node, setup NetDevices.
  Ptr<MockNetDevice> mockNetDevice1 = CreateObject<MockNetDevice> ();
  nodes.Get (1)->AddDevice (mockNetDevice1);
  mockNetDevice1->SetNode (nodes.Get (1));
  mockNetDevice1->SetAddress (Mac48Address ("00:00:00:00:00:02"));
  mockNetDevice1->SetMtu (150);
  m_mockDevices.Add (mockNetDevice1);

  InternetStackHelper internetv6;
  internetv6.Install (nodes);

  SixLowPanHelper sixlowpan;
  m_sixDevices = sixlowpan.Install (m_mockDevices);
  m_sixDevices.Get (1)->SetPromiscReceiveCallback ( MakeCallback (&SixlowpanIphcStatefulImplTest::PromiscReceiveFromSixLowPanDevice, this) );

  Ipv6AddressHelper ipv6;
  ipv6.SetBase (Ipv6Address ("2001:2::"), Ipv6Prefix (64));
  Ipv6InterfaceContainer deviceInterfaces;
  deviceInterfaces = ipv6.Assign (m_sixDevices);

  // This is a hack to prevent Router Solicitations and Duplicate Address Detection being sent.
  for (auto i = nodes.Begin (); i != nodes.End (); i++)
    {
      Ptr<Node> node = *i;
      Ptr<Ipv6L3Protocol> ipv6L3 = (*i)->GetObject<Ipv6L3Protocol> ();
      if (ipv6L3)
        {
          ipv6L3->SetAttribute ("IpForward", BooleanValue (true));
          ipv6L3->SetAttribute ("SendIcmpv6Redirect", BooleanValue (false));
        }
      Ptr<Icmpv6L4Protocol> icmpv6 = (*i)->GetObject<Icmpv6L4Protocol> ();
      if (icmpv6)
        {
          icmpv6->SetAttribute ("DAD", BooleanValue (false));
        }
    }

  sixlowpan.AddContext (m_sixDevices, 0, Ipv6Prefix ("2001:2::", 64), Time (Minutes (30)));
  sixlowpan.AddContext (m_sixDevices, 1, Ipv6Prefix ("2001:1::", 64), Time (Minutes (30)));

  Ipv6Address srcElided = deviceInterfaces.GetAddress (0, 1);
  Ipv6Address dstElided = Ipv6Address::MakeAutoconfiguredAddress (Mac48Address ("00:00:00:00:00:02"), Ipv6Prefix ("2001:2::", 64));

  Simulator::Schedule (Seconds (1), &SixlowpanIphcStatefulImplTest::SendOnePacket, this, m_sixDevices.Get (0),
                       Ipv6Address::GetAny (),
                       dstElided);

  Simulator::Schedule (Seconds (2), &SixlowpanIphcStatefulImplTest::SendOnePacket, this, m_sixDevices.Get (0),
                       Ipv6Address ("2001:2::f00d:f00d:cafe:cafe"),
                       Ipv6Address ("2001:1::0000:00ff:fe00:cafe"));

  Simulator::Schedule (Seconds (3), &SixlowpanIphcStatefulImplTest::SendOnePacket, this, m_sixDevices.Get (0),
                       Ipv6Address ("2001:1::0000:00ff:fe00:cafe"),
                       Ipv6Address ("2001:1::f00d:f00d:cafe:cafe"));

  Simulator::Schedule (Seconds (4), &SixlowpanIphcStatefulImplTest::SendOnePacket, this, m_sixDevices.Get (0),
                       srcElided,
                       Ipv6Address ("2001:1::f00d:f00d:cafe:cafe"));

  Simulator::Stop (Seconds (10));

  Simulator::Run ();
  Simulator::Destroy ();

  // ------ Now the tests ------------

  SixLowPanIphc iphcHdr;
  Ipv6Header ipv6Hdr;

  // first packet sent, expected CID(0) SAC(1) SAM (0) M(0) DAC(1) DAM (3)
  m_txPackets[0].packet->RemoveHeader (iphcHdr);
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetCid (), false, "CID should be false, is true");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSac (), true, "SAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSam (), SixLowPanIphc::HC_INLINE, "SAM should be HC_INLINE, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetM (), false, "M should be false, is true");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDac (), true, "DAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSrcContextId (), 0, "Src context should be 0, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDstContextId (), 0, "Dst context should be 0, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDam (), SixLowPanIphc::HC_COMPR_0, "DAM should be HC_COMPR_0, it is not");

  // first packet received, expected :: -> dstElided
  m_rxPackets[0].packet->RemoveHeader (ipv6Hdr);
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetSource (), Ipv6Address::GetAny (), "Src address wrongly rebuilt");
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetDestination (), dstElided, "Dst address wrongly rebuilt");

  // second packet sent, expected CID(1) SAC(1) SAM (1) M(0) DAC(1) DAM (2)
  m_txPackets[1].packet->RemoveHeader (iphcHdr);
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetCid (), true, "CID should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSac (), true, "SAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSam (), SixLowPanIphc::HC_COMPR_64, "SAM should be HC_COMPR_64, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetM (), false, "M should be false, is true");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDac (), true, "DAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSrcContextId (), 0, "Src context should be 0, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDstContextId (), 1, "Dst context should be 1, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDam (), SixLowPanIphc::HC_COMPR_16, "DAM should be HC_COMPR_16, it is not");

  // second packet received, expected 2001:2::f00d:f00d:cafe:cafe -> 2001:1::0000:00ff:fe00:cafe
  m_rxPackets[1].packet->RemoveHeader (ipv6Hdr);
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetSource (), Ipv6Address ("2001:2::f00d:f00d:cafe:cafe"), "Src address wrongly rebuilt");
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetDestination (), Ipv6Address ("2001:1::0000:00ff:fe00:cafe"), "Dst address wrongly rebuilt");

  // third packet sent, expected CID(17) SAC(1) SAM (2) M(0) DAC(1) DAM (1)
  m_txPackets[2].packet->RemoveHeader (iphcHdr);
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetCid (), true, "CID should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSac (), true, "SAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSam (), SixLowPanIphc::HC_COMPR_16, "SAM should be HC_COMPR_16, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetM (), false, "M should be false, is true");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDac (), true, "DAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSrcContextId (), 1, "Src context should be 1, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDstContextId (), 1, "Dst context should be 1, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDam (), SixLowPanIphc::HC_COMPR_64, "DAM should be HC_COMPR_64, it is not");

  // third packet received, expected 2001:1::0000:00ff:fe00:cafe -> 2001:1::f00d:f00d:cafe:cafe
  m_rxPackets[2].packet->RemoveHeader (ipv6Hdr);
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetSource (), Ipv6Address ("2001:1::0000:00ff:fe00:cafe"), "Src address wrongly rebuilt");
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetDestination (), Ipv6Address ("2001:1::f00d:f00d:cafe:cafe"), "Dst address wrongly rebuilt");

  // fourth packet sent, expected CID(1) SAC(1) SAM (3) M(0) DAC(1) DAM (1)
  m_txPackets[3].packet->RemoveHeader (iphcHdr);
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetCid (), true, "CID should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSac (), true, "SAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSam (), SixLowPanIphc::HC_COMPR_0, "SAM should be HC_COMPR_0, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetM (), false, "M should be false, is true");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDac (), true, "DAC should be true, is false");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetSrcContextId (), 0, "Src context should be 0, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDstContextId (), 1, "Dst context should be 1, it is not");
  NS_TEST_EXPECT_MSG_EQ (iphcHdr.GetDam (), SixLowPanIphc::HC_COMPR_64, "DAM should be HC_COMPR_64, it is not");

  // fourth packet received, expected srcElided -> 2001:1::f00d:f00d:cafe:cafe
  m_rxPackets[3].packet->RemoveHeader (ipv6Hdr);
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetSource (), srcElided, "Src address wrongly rebuilt");
  NS_TEST_EXPECT_MSG_EQ (ipv6Hdr.GetDestination (), Ipv6Address ("2001:1::f00d:f00d:cafe:cafe"), "Dst address wrongly rebuilt");

  m_rxPackets.clear ();
  m_txPackets.clear ();
}


/**
 * \ingroup sixlowpan-tests
 *
 * \brief 6LoWPAN IPHC TestSuite
 */
class SixlowpanIphcStatefulTestSuite : public TestSuite
{
public:
  SixlowpanIphcStatefulTestSuite ();
private:
};

SixlowpanIphcStatefulTestSuite::SixlowpanIphcStatefulTestSuite ()
  : TestSuite ("sixlowpan-iphc-stateful", UNIT)
{
  AddTestCase (new SixlowpanIphcStatefulImplTest (), TestCase::QUICK);
}

static SixlowpanIphcStatefulTestSuite g_sixlowpanIphcStatefulTestSuite; //!< Static variable for test initialization
