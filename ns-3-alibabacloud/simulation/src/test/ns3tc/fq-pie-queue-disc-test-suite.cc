/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
 * Copyright (c) 2020 NITK Surathkal (modified for FQ-PIE)
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
 * Authors: Pasquale Imputato <p.imputato@gmail.com>
 *          Stefano Avallone <stefano.avallone@unina.it>
 * Modified for FQ-PIE by: Bhaskar Kataria <bhaskar.k7920@gmail.com>
 *                         Tom Henderson <tomhend@u.washington.edu>
 *                         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *                         Vivek Jain <jain.vivek.anand@gmail.com>
 *                         Ankit Deepak <adadeepak8@gmail.com>
 *
 */

#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/fq-pie-queue-disc.h"
#include "ns3/pie-queue-disc.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-packet-filter.h"
#include "ns3/ipv4-queue-disc-item.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv6-header.h"
#include "ns3/ipv6-packet-filter.h"
#include "ns3/ipv6-queue-disc-item.h"
#include "ns3/tcp-header.h"
#include "ns3/udp-header.h"
#include "ns3/string.h"
#include "ns3/pointer.h"

using namespace ns3;

/// Variable to assign g_hash to a new packet's flow
static int32_t g_hash;

/**
 * \ingroup system-tests-tc
 * 
 * Simple test packet filter able to classify IPv4 packets.
 */
class Ipv4FqPieTestPacketFilter : public Ipv4PacketFilter
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  Ipv4FqPieTestPacketFilter ();
  virtual ~Ipv4FqPieTestPacketFilter ();

private:
  /**
   * Classify a QueueDiscItem
   * \param item The item to classify (unused).
   * \return a pre-set hash value.
   */
  virtual int32_t DoClassify (Ptr<QueueDiscItem> item) const;

  /**
   * Check the protocol.
   * \param item The item to check (unused).
   * \return true.
   */
  virtual bool CheckProtocol (Ptr<QueueDiscItem> item) const;
};

TypeId
Ipv4FqPieTestPacketFilter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv4FqPieTestPacketFilter")
    .SetParent<Ipv4PacketFilter> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv4FqPieTestPacketFilter> ()
  ;
  return tid;
}

Ipv4FqPieTestPacketFilter::Ipv4FqPieTestPacketFilter ()
{}

Ipv4FqPieTestPacketFilter::~Ipv4FqPieTestPacketFilter ()
{}

int32_t
Ipv4FqPieTestPacketFilter::DoClassify (Ptr<QueueDiscItem> item) const
{
  return g_hash;
}

bool
Ipv4FqPieTestPacketFilter::CheckProtocol (Ptr<QueueDiscItem> item) const
{
  return true;
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests packets for which there is no suitable filter.
 */
class FqPieQueueDiscNoSuitableFilter : public TestCase
{
public:
  FqPieQueueDiscNoSuitableFilter ();
  virtual ~FqPieQueueDiscNoSuitableFilter ();

private:
  virtual void DoRun (void);
};

FqPieQueueDiscNoSuitableFilter::FqPieQueueDiscNoSuitableFilter ()
  : TestCase ("Test packets that are not classified by any filter")
{}

FqPieQueueDiscNoSuitableFilter::~FqPieQueueDiscNoSuitableFilter ()
{}

void
FqPieQueueDiscNoSuitableFilter::DoRun (void)
{
  // Packets that cannot be classified by the available filters should be dropped
  Ptr<FqPieQueueDisc> queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("MaxSize", StringValue ("4p"));
  Ptr<Ipv4FqPieTestPacketFilter> filter = CreateObject<Ipv4FqPieTestPacketFilter> ();
  queueDisc->AddPacketFilter (filter);

  g_hash = -1;
  queueDisc->SetQuantum (1500);
  queueDisc->Initialize ();

  Ptr<Packet> p;
  p = Create<Packet> ();
  Ptr<Ipv6QueueDiscItem> item;
  Ipv6Header ipv6Header;
  Address dest;
  item = Create<Ipv6QueueDiscItem> (p, dest, 0, ipv6Header);
  queueDisc->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetNQueueDiscClasses (), 0, "no flow queue should have been created");

  p = Create<Packet> (reinterpret_cast<const uint8_t*> ("hello, world"), 12);
  item = Create<Ipv6QueueDiscItem> (p, dest, 0, ipv6Header);
  queueDisc->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetNQueueDiscClasses (), 0, "no flow queue should have been created");

  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests the IP flows separation and the packet limit.
 */
class FqPieQueueDiscIPFlowsSeparationAndPacketLimit : public TestCase
{
public:
  FqPieQueueDiscIPFlowsSeparationAndPacketLimit ();
  virtual ~FqPieQueueDiscIPFlowsSeparationAndPacketLimit ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet.
   * \param queue the queue disc
   * \param hdr the IPv4 header
   */
  void AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr);
};

FqPieQueueDiscIPFlowsSeparationAndPacketLimit::FqPieQueueDiscIPFlowsSeparationAndPacketLimit ()
  : TestCase ("Test IP flows separation and packet limit")
{}

FqPieQueueDiscIPFlowsSeparationAndPacketLimit::~FqPieQueueDiscIPFlowsSeparationAndPacketLimit ()
{}

void
FqPieQueueDiscIPFlowsSeparationAndPacketLimit::AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr)
{
  Ptr<Packet> p = Create<Packet> (100);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, hdr);
  queue->Enqueue (item);
}

void
FqPieQueueDiscIPFlowsSeparationAndPacketLimit::DoRun (void)
{
  Ptr<FqPieQueueDisc> queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("MaxSize", StringValue ("4p"));

  queueDisc->SetQuantum (1500);
  queueDisc->Initialize ();

  Ipv4Header hdr;
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (7);

  // Add three packets from the first flow
  AddPacket (queueDisc, hdr);
  AddPacket (queueDisc, hdr);
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 3, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the flow queue");

  // Add two packets from the second flow
  hdr.SetDestination (Ipv4Address ("10.10.1.7"));
  // Add the first packet
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 4, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the flow queue");
  // Add the second packet that causes two packets to be dropped from the fat flow (max backlog = 300, threshold = 150)
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 3, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the flow queue");

  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests the deficit per flow.
 */
class FqPieQueueDiscDeficit : public TestCase
{
public:
  FqPieQueueDiscDeficit ();
  virtual ~FqPieQueueDiscDeficit ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet.
   * \param queue The queue disc.
   * \param hdr The IPv4 header.
   */
  void AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr);
};

FqPieQueueDiscDeficit::FqPieQueueDiscDeficit ()
  : TestCase ("Test credits and flows status")
{}

FqPieQueueDiscDeficit::~FqPieQueueDiscDeficit ()
{}

void
FqPieQueueDiscDeficit::AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr)
{
  Ptr<Packet> p = Create<Packet> (100);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, hdr);
  queue->Enqueue (item);
}

void
FqPieQueueDiscDeficit::DoRun (void)
{
  Ptr<FqPieQueueDisc> queueDisc = CreateObject<FqPieQueueDisc> ();

  queueDisc->SetQuantum (90);
  queueDisc->Initialize ();

  Ipv4Header hdr;
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (7);

  // Add a packet from the first flow
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 1, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the first flow queue");
  Ptr<FqPieFlow> flow1 = StaticCast<FqPieFlow> (queueDisc->GetQueueDiscClass (0));
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), static_cast<int32_t> (queueDisc->GetQuantum ()), "the deficit of the first flow must equal the quantum");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::NEW_FLOW, "the first flow must be in the list of new queues");
  // Dequeue a packet
  queueDisc->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 0, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 0, "unexpected number of packets in the first flow queue");
  // the deficit for the first flow becomes 90 - (100+20) = -30
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), -30, "unexpected deficit for the first flow");

  // Add two packets from the first flow
  AddPacket (queueDisc, hdr);
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 2, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::NEW_FLOW, "the first flow must still be in the list of new queues");

  // Add two packets from the second flow
  hdr.SetDestination (Ipv4Address ("10.10.1.10"));
  AddPacket (queueDisc, hdr);
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 4, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the second flow queue");
  Ptr<FqPieFlow> flow2 = StaticCast<FqPieFlow> (queueDisc->GetQueueDiscClass (1));
  NS_TEST_ASSERT_MSG_EQ (flow2->GetDeficit (), static_cast<int32_t> (queueDisc->GetQuantum ()), "the deficit of the second flow must equal the quantum");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetStatus (), FqPieFlow::NEW_FLOW, "the second flow must be in the list of new queues");

  // Dequeue a packet (from the second flow, as the first flow has a negative deficit)
  queueDisc->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 3, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");
  // the first flow got a quantum of deficit (-30+90=60) and has been moved to the end of the list of old queues
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), 60, "unexpected deficit for the first flow");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::OLD_FLOW, "the first flow must be in the list of old queues");
  // the second flow has a negative deficit (-30) and is still in the list of new queues
  NS_TEST_ASSERT_MSG_EQ (flow2->GetDeficit (), -30, "unexpected deficit for the second flow");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetStatus (), FqPieFlow::NEW_FLOW, "the second flow must be in the list of new queues");

  // Dequeue a packet (from the first flow, as the second flow has a negative deficit)
  queueDisc->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 2, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");
  // the first flow has a negative deficit (60-(100+20)= -60) and stays in the list of old queues
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), -60, "unexpected deficit for the first flow");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::OLD_FLOW, "the first flow must be in the list of old queues");
  // the second flow got a quantum of deficit (-30+90=60) and has been moved to the end of the list of old queues
  NS_TEST_ASSERT_MSG_EQ (flow2->GetDeficit (), 60, "unexpected deficit for the second flow");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetStatus (), FqPieFlow::OLD_FLOW, "the second flow must be in the list of new queues");

  // Dequeue a packet (from the second flow, as the first flow has a negative deficit)
  queueDisc->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 1, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 0, "unexpected number of packets in the second flow queue");
  // the first flow got a quantum of deficit (-60+90=30) and has been moved to the end of the list of old queues
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), 30, "unexpected deficit for the first flow");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::OLD_FLOW, "the first flow must be in the list of old queues");
  // the second flow has a negative deficit (60-(100+20)= -60)
  NS_TEST_ASSERT_MSG_EQ (flow2->GetDeficit (), -60, "unexpected deficit for the second flow");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetStatus (), FqPieFlow::OLD_FLOW, "the second flow must be in the list of new queues");

  // Dequeue a packet (from the first flow, as the second flow has a negative deficit)
  queueDisc->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 0, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 0, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 0, "unexpected number of packets in the second flow queue");
  // the first flow has a negative deficit (30-(100+20)= -90)
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), -90, "unexpected deficit for the first flow");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::OLD_FLOW, "the first flow must be in the list of old queues");
  // the second flow got a quantum of deficit (-60+90=30) and has been moved to the end of the list of old queues
  NS_TEST_ASSERT_MSG_EQ (flow2->GetDeficit (), 30, "unexpected deficit for the second flow");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetStatus (), FqPieFlow::OLD_FLOW, "the second flow must be in the list of new queues");

  // Dequeue a packet
  queueDisc->Dequeue ();
  // the first flow is at the head of the list of old queues but has a negative deficit, thus it gets a quantun
  // of deficit (-90+90=0) and is moved to the end of the list of old queues. Then, the second flow (which has a
  // positive deficit) is selected, but the second flow is empty and thus it is set to inactive. The first flow is
  // reconsidered, but it has a null deficit, hence it gets another quantum of deficit (0+90=90). Then, the first
  // flow is reconsidered again, now it has a positive deficit and hence it is selected. But, it is empty and
  // therefore is set to inactive, too.
  NS_TEST_ASSERT_MSG_EQ (flow1->GetDeficit (), 90, "unexpected deficit for the first flow");
  NS_TEST_ASSERT_MSG_EQ (flow1->GetStatus (), FqPieFlow::INACTIVE, "the first flow must be inactive");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetDeficit (), 30, "unexpected deficit for the second flow");
  NS_TEST_ASSERT_MSG_EQ (flow2->GetStatus (), FqPieFlow::INACTIVE, "the second flow must be inactive");

  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests the TCP flows separation.
 */
class FqPieQueueDiscTCPFlowsSeparation : public TestCase
{
public:
  FqPieQueueDiscTCPFlowsSeparation ();
  virtual ~FqPieQueueDiscTCPFlowsSeparation ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet.
   * \param queue The queue disc.
   * \param ipHdr The IPv4 header.
   * \param tcpHdr The TCP header.
   */
  void AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header ipHdr, TcpHeader tcpHdr);
};

FqPieQueueDiscTCPFlowsSeparation::FqPieQueueDiscTCPFlowsSeparation ()
  : TestCase ("Test TCP flows separation")
{}

FqPieQueueDiscTCPFlowsSeparation::~FqPieQueueDiscTCPFlowsSeparation ()
{}

void
FqPieQueueDiscTCPFlowsSeparation::AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header ipHdr, TcpHeader tcpHdr)
{
  Ptr<Packet> p = Create<Packet> (100);
  p->AddHeader (tcpHdr);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, ipHdr);
  queue->Enqueue (item);
}

void
FqPieQueueDiscTCPFlowsSeparation::DoRun (void)
{
  Ptr<FqPieQueueDisc> queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("MaxSize", StringValue ("10p"));

  queueDisc->SetQuantum (1500);
  queueDisc->Initialize ();

  Ipv4Header hdr;
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (6);

  TcpHeader tcpHdr;
  tcpHdr.SetSourcePort (7);
  tcpHdr.SetDestinationPort (27);

  // Add three packets from the first flow
  AddPacket (queueDisc, hdr, tcpHdr);
  AddPacket (queueDisc, hdr, tcpHdr);
  AddPacket (queueDisc, hdr, tcpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 3, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");

  // Add a packet from the second flow
  tcpHdr.SetSourcePort (8);
  AddPacket (queueDisc, hdr, tcpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 4, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");

  // Add a packet from the third flow
  tcpHdr.SetDestinationPort (28);
  AddPacket (queueDisc, hdr, tcpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 5, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (2)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the third flow queue");

  // Add two packets from the fourth flow
  tcpHdr.SetSourcePort (7);
  AddPacket (queueDisc, hdr, tcpHdr);
  AddPacket (queueDisc, hdr, tcpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 7, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (2)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the third flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (3)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the third flow queue");

  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests the UDP flows separation
 */
class FqPieQueueDiscUDPFlowsSeparation : public TestCase
{
public:
  FqPieQueueDiscUDPFlowsSeparation ();
  virtual ~FqPieQueueDiscUDPFlowsSeparation ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet.
   * \param queue The queue disc.
   * \param ipHdr The IPv4 header.
   * \param udpHdr The UDP header.
   */
  void AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header ipHdr, UdpHeader udpHdr);
};

FqPieQueueDiscUDPFlowsSeparation::FqPieQueueDiscUDPFlowsSeparation ()
  : TestCase ("Test UDP flows separation")
{}

FqPieQueueDiscUDPFlowsSeparation::~FqPieQueueDiscUDPFlowsSeparation ()
{}

void
FqPieQueueDiscUDPFlowsSeparation::AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header ipHdr, UdpHeader udpHdr)
{
  Ptr<Packet> p = Create<Packet> (100);
  p->AddHeader (udpHdr);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, ipHdr);
  queue->Enqueue (item);
}

void
FqPieQueueDiscUDPFlowsSeparation::DoRun (void)
{
  Ptr<FqPieQueueDisc> queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("MaxSize", StringValue ("10p"));

  queueDisc->SetQuantum (1500);
  queueDisc->Initialize ();

  Ipv4Header hdr;
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (17);

  UdpHeader udpHdr;
  udpHdr.SetSourcePort (7);
  udpHdr.SetDestinationPort (27);

  // Add three packets from the first flow
  AddPacket (queueDisc, hdr, udpHdr);
  AddPacket (queueDisc, hdr, udpHdr);
  AddPacket (queueDisc, hdr, udpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 3, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");

  // Add a packet from the second flow
  udpHdr.SetSourcePort (8);
  AddPacket (queueDisc, hdr, udpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 4, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");

  // Add a packet from the third flow
  udpHdr.SetDestinationPort (28);
  AddPacket (queueDisc, hdr, udpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 5, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (2)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the third flow queue");

  // Add two packets from the fourth flow
  udpHdr.SetSourcePort (7);
  AddPacket (queueDisc, hdr, udpHdr);
  AddPacket (queueDisc, hdr, udpHdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 7, "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3, "unexpected number of packets in the first flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the second flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (2)->GetQueueDisc ()->GetNPackets (), 1, "unexpected number of packets in the third flow queue");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (3)->GetQueueDisc ()->GetNPackets (), 2, "unexpected number of packets in the third flow queue");

  Simulator::Destroy ();
}


/**
 * \ingroup system-tests-tc
 * 
 * \brief This class tests linear probing, collision response, and set
 * creation capability of set associative hashing in FqPIE.
 * 
 * This class tests linear probing, collision response, and set
 * creation capability of set associative hashing in FqPIE.
 * We modified DoClassify () and CheckProtocol () so that we could control
 * the hash returned for each packet. In the beginning, we use flow hashes
 * ranging from 0 to 7. These must go into different queues in the same set.
 * The set number for these is obtained using outerhash, which is 0.
 * When a new packet arrives with flow hash 1024, outerhash = 0 is obtained
 * and the first set is iteratively searched.
 * The packet is eventually added to queue 0 since the tags of queues
 * in the set do not match with the hash of the flow. The tag of queue 0 is
 * updated as 1024. When a packet with hash 1025 arrives, outerhash = 0
 * is obtained and the first set is iteratively searched.
 * Since there is no match, it is added to queue 0 and the tag of queue 0 is
 * updated to 1025.
 *
 * The variable outerhash stores the nearest multiple of 8 that is lesser than
 * the hash. When a flow hash of 20 arrives, the value of outerhash
 * is 16. Since m_flowIndices[16] wasn't previously allotted, a new flow
 * is created, and the tag corresponding to this queue is set to 20.
*/
class FqPieQueueDiscSetLinearProbing : public TestCase
{
public:
  FqPieQueueDiscSetLinearProbing ();
  virtual ~FqPieQueueDiscSetLinearProbing ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet.
   * \param queue The queue disc.
   * \param hdr The IPv4 header.
   */
   void AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr);
};

FqPieQueueDiscSetLinearProbing::FqPieQueueDiscSetLinearProbing ()
  : TestCase ("Test credits and flows status")
{}

FqPieQueueDiscSetLinearProbing::~FqPieQueueDiscSetLinearProbing ()
{}

void
FqPieQueueDiscSetLinearProbing::AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr)
{
  Ptr<Packet> p = Create<Packet> (100);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, hdr);
  queue->Enqueue (item);
}

void
FqPieQueueDiscSetLinearProbing::DoRun (void)
{
  Ptr<FqPieQueueDisc> queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("EnableSetAssociativeHash", BooleanValue (true));
  queueDisc->SetQuantum (90);
  queueDisc->Initialize ();

  Ptr<Ipv4FqPieTestPacketFilter> filter = CreateObject<Ipv4FqPieTestPacketFilter> ();
  queueDisc->AddPacketFilter (filter);

  Ipv4Header hdr;
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (7);

  g_hash = 0;
  AddPacket (queueDisc, hdr);
  g_hash = 1;
  AddPacket (queueDisc, hdr);
  AddPacket (queueDisc, hdr);
  g_hash = 2;
  AddPacket (queueDisc, hdr);
  g_hash = 3;
  AddPacket (queueDisc, hdr);
  g_hash = 4;
  AddPacket (queueDisc, hdr);
  AddPacket (queueDisc, hdr);
  g_hash = 5;
  AddPacket (queueDisc, hdr);
  g_hash = 6;
  AddPacket (queueDisc, hdr);
  g_hash = 7;
  AddPacket (queueDisc, hdr);
  g_hash = 1024;
  AddPacket (queueDisc, hdr);

  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 11,
                         "unexpected number of packets in the queue disc");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 2,
                         "unexpected number of packets in the first flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetNPackets (), 2,
                         "unexpected number of packets in the second flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (2)->GetQueueDisc ()->GetNPackets (), 1,
                         "unexpected number of packets in the third flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (3)->GetQueueDisc ()->GetNPackets (), 1,
                         "unexpected number of packets in the fourth flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (4)->GetQueueDisc ()->GetNPackets (), 2,
                         "unexpected number of packets in the fifth flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (5)->GetQueueDisc ()->GetNPackets (), 1,
                         "unexpected number of packets in the sixth flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (6)->GetQueueDisc ()->GetNPackets (), 1,
                         "unexpected number of packets in the seventh flow queue of set one");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (7)->GetQueueDisc ()->GetNPackets (), 1,
                         "unexpected number of packets in the eighth flow queue of set one");
  g_hash = 1025;
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetNPackets (), 3,
                         "unexpected number of packets in the first flow of set one");
  g_hash = 10;
  AddPacket (queueDisc, hdr);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetQueueDiscClass (8)->GetQueueDisc ()->GetNPackets (), 1,
                         "unexpected number of packets in the first flow of set two");
  Simulator::Destroy ();
}


/**
 * \ingroup system-tests-tc
 * 
 * \brief This class tests L4S mode. 
 * 
 * This test is divided to sub test one without hash collisions and so ECT0 and ECT1 flows are
 * classified into different flows.
 * Sub Test 1
 * 70 packets are enqueued into both the flows with the delay of 0.5ms between two enqueues, and dequeued with the delay of
 * 1ms between two dequeues.
 * Sub Test 2
 * 140(70 ECT0 + 70 ECT1) packets are enqueued such that ECT1 packets are enqueued at 0.5ms, 1.5ms, 2.5ms and so on, and ECT0 packets are
 * enqueued are enqueued at 1ms, 2ms, 3ms and so on
 * Any future classifier options (e.g. SetAssociativehash) should be disabled to prevent a hash collision on this test case.
 */
class FqPieQueueDiscL4sMode : public TestCase
{
public:
  FqPieQueueDiscL4sMode ();
  virtual ~FqPieQueueDiscL4sMode ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue the given number of packets.
   * \param queue The queue disc.
   * \param hdr The IPv4 header.
   * \param nPkt The number of packets.
   */
  void AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr, uint32_t nPkt);
  /**
   * Enqueue the given number of packets at different times.
   * \param queue The queue disc.
   * \param hdr The IPv4 header.
   * \param delay The time between two consecutive enqueue operations.
   * \param nPkt The number of packets.
   */
  void AddPacketWithDelay (Ptr<FqPieQueueDisc> queue,Ipv4Header hdr, double delay, uint32_t nPkt);
  /**
   * Dequeue the given number of packets.
   * \param queue The queue disc.
   * \param nPkt The number of packets.
   */
  void Dequeue (Ptr<FqPieQueueDisc> queue, uint32_t nPkt);
  /**
   * Dequeue the given number of packets at different times.
   * \param queue The queue disc.
   * \param delay The time between two consecutive dequeue operations.
   * \param nPkt The number of packets.
   */
  void DequeueWithDelay (Ptr<FqPieQueueDisc> queue, double delay, uint32_t nPkt);
};

FqPieQueueDiscL4sMode::FqPieQueueDiscL4sMode ()
  : TestCase ("Test L4S mode")
{}

FqPieQueueDiscL4sMode::~FqPieQueueDiscL4sMode ()
{}

void
FqPieQueueDiscL4sMode::AddPacket (Ptr<FqPieQueueDisc> queue, Ipv4Header hdr, uint32_t nPkt)
{
  Address dest;
  Ptr<Packet> p = Create<Packet> (100);
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, hdr);
      queue->Enqueue (item);
    }
}

void
FqPieQueueDiscL4sMode::AddPacketWithDelay (Ptr<FqPieQueueDisc> queue,Ipv4Header hdr, double delay, uint32_t nPkt)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &FqPieQueueDiscL4sMode::AddPacket, this, queue, hdr, 1);
    }
}

void
FqPieQueueDiscL4sMode::Dequeue (Ptr<FqPieQueueDisc> queue, uint32_t nPkt)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Ptr<QueueDiscItem> item = queue->Dequeue ();
    }
}

void
FqPieQueueDiscL4sMode::DequeueWithDelay (Ptr<FqPieQueueDisc> queue, double delay, uint32_t nPkt)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &FqPieQueueDiscL4sMode::Dequeue, this, queue, 1);
    }
}

void
FqPieQueueDiscL4sMode::DoRun (void)
{
  // Test is divided into 2 sub test cases:
  // 1) Without hash collisions
  // 2) With hash collisions

  // Test case 1, Without hash collisions
  Ptr<FqPieQueueDisc> queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("MaxSize", StringValue ("10240p"), 
    "UseEcn", BooleanValue (true), "Perturbation", UintegerValue (0),
    "UseL4s", BooleanValue (true), "CeThreshold", TimeValue (MilliSeconds (2)));

  queueDisc->SetQuantum (1514);
  queueDisc->Initialize ();
  Ipv4Header hdr;
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (7);
  hdr.SetEcn (Ipv4Header::ECN_ECT1);

  // Add 70 ECT1 (ECN capable) packets from the first flow
  // Set delay = 0.5ms
  double delay = 0.0005;
  Simulator::Schedule (Time (Seconds (0)), &FqPieQueueDiscL4sMode::AddPacketWithDelay, this, queueDisc, hdr, delay, 70);

  // Add 70 ECT0 (ECN capable) packets from second flow
  hdr.SetEcn (Ipv4Header::ECN_ECT0);
  hdr.SetDestination (Ipv4Address ("10.10.1.10"));
  Simulator::Schedule (Time (Seconds (0)), &FqPieQueueDiscL4sMode::AddPacketWithDelay, this, queueDisc, hdr, delay, 70);

  //Dequeue 140 packets with delay 1ms
  delay = 0.001;
  DequeueWithDelay (queueDisc, delay, 140);
  Simulator::Stop (Seconds (10.0));
  Simulator::Run ();

  Ptr<PieQueueDisc> q0 = queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetObject <PieQueueDisc> ();
  Ptr<PieQueueDisc> q1 = queueDisc->GetQueueDiscClass (1)->GetQueueDisc ()->GetObject <PieQueueDisc> ();

  NS_TEST_EXPECT_MSG_EQ (q0->GetStats ().GetNMarkedPackets (PieQueueDisc::CE_THRESHOLD_EXCEEDED_MARK), 66, "There should be 66 marked packets"
                         "4th packet is enqueued at 2ms and dequeued at 4ms hence the delay of 2ms which not greater than CE threshold"
                         "5th packet is enqueued at 2.5ms and dequeued at 5ms hence the delay of 2.5ms and subsequent packet also do have delay"
                         "greater than CE threshold so all the packets after 4th packet are marked");
  NS_TEST_EXPECT_MSG_EQ (q0->GetStats ().GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP), 0, "Queue delay is less than max burst allowance so"
                         "There should not be any dropped packets");
  NS_TEST_EXPECT_MSG_EQ (q0->GetStats ().GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK), 0, "There should not be any marked packets");
  NS_TEST_EXPECT_MSG_EQ (q1->GetStats ().GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK), 0, "There should not be marked packets.");
  NS_TEST_EXPECT_MSG_EQ (q1->GetStats ().GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP), 0, "There should not be any dropped packets");

  Simulator::Destroy ();

  // Test case 2, With hash collisions
  queueDisc = CreateObjectWithAttributes<FqPieQueueDisc> ("MaxSize", StringValue ("10240p"), 
    "UseEcn", BooleanValue (true), "Perturbation", UintegerValue (0),
    "UseL4s", BooleanValue (true), "CeThreshold", TimeValue (MilliSeconds (2)));

  queueDisc->SetQuantum (1514);
  queueDisc->Initialize ();
  hdr.SetPayloadSize (100);
  hdr.SetSource (Ipv4Address ("10.10.1.1"));
  hdr.SetDestination (Ipv4Address ("10.10.1.2"));
  hdr.SetProtocol (7);
  hdr.SetEcn (Ipv4Header::ECN_ECT1);

  // Add 70 ECT1 (ECN capable) packets from the first flow
  // Set delay = 1ms
  delay = 0.001;
  Simulator::Schedule (Time (Seconds (0.0005)), &FqPieQueueDiscL4sMode::AddPacket, this, queueDisc, hdr, 1);
  Simulator::Schedule (Time (Seconds (0.0005)), &FqPieQueueDiscL4sMode::AddPacketWithDelay, this, queueDisc, hdr, delay, 69);

  // Add 70 ECT0 (ECN capable) packets from first flow
  hdr.SetEcn (Ipv4Header::ECN_ECT0);
  Simulator::Schedule (Time (Seconds (0)), &FqPieQueueDiscL4sMode::AddPacketWithDelay, this, queueDisc, hdr, delay, 70);

  //Dequeue 140 packets with delay 1ms
  DequeueWithDelay (queueDisc, delay, 140);
  Simulator::Stop (Seconds (1.0));
  Simulator::Run ();
  q0 = queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetObject <PieQueueDisc> ();
  q0 = queueDisc->GetQueueDiscClass (0)->GetQueueDisc ()->GetObject <PieQueueDisc> ();

  NS_TEST_EXPECT_MSG_EQ (q0->GetStats ().GetNMarkedPackets (PieQueueDisc::CE_THRESHOLD_EXCEEDED_MARK), 68, "There should be 68 marked packets"
                         "2nd ECT1 packet is enqueued at 1.5ms and dequeued at 3ms hence the delay of 1.5ms which not greater than CE threshold"
                         "3rd packet is enqueued at 2.5ms and dequeued at 5ms hence the delay of 2.5ms and subsequent packet also do have delay"
                         "greater than CE threshold so all the packets after 2nd packet are marked");
  NS_TEST_EXPECT_MSG_EQ (q0->GetStats ().GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP), 0, "Queue delay is less than max burst allowance so"
                         "There should not be any dropped packets");
  NS_TEST_EXPECT_MSG_EQ (q0->GetStats ().GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK), 0, "There should not be any marked packets");

  Simulator::Destroy ();
}


/**
 * \ingroup system-tests-tc
 * 
 * FQ-PIE queue disc test suite.
 */
class FqPieQueueDiscTestSuite : public TestSuite
{
public:
  FqPieQueueDiscTestSuite ();
};

FqPieQueueDiscTestSuite::FqPieQueueDiscTestSuite ()
  : TestSuite ("fq-pie-queue-disc", UNIT)
{
  AddTestCase (new FqPieQueueDiscNoSuitableFilter, TestCase::QUICK);
  AddTestCase (new FqPieQueueDiscIPFlowsSeparationAndPacketLimit, TestCase::QUICK);
  AddTestCase (new FqPieQueueDiscDeficit, TestCase::QUICK);
  AddTestCase (new FqPieQueueDiscTCPFlowsSeparation, TestCase::QUICK);
  AddTestCase (new FqPieQueueDiscUDPFlowsSeparation, TestCase::QUICK);
  AddTestCase (new FqPieQueueDiscSetLinearProbing, TestCase::QUICK);
  AddTestCase (new FqPieQueueDiscL4sMode, TestCase::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite.
static FqPieQueueDiscTestSuite g_fqPieQueueDiscTestSuite;
