/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 */

#include "ns3/test.h"
#include "ns3/simulator.h"
#include "ns3/pfifo-fast-queue-disc.h"
#include "ns3/drop-tail-queue.h"
#include "ns3/ipv4-queue-disc-item.h"
#include "ns3/ipv6-queue-disc-item.h"
#include "ns3/enum.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include "ns3/object-factory.h"
#include "ns3/socket.h"

using namespace ns3;

/**
 * \ingroup system-tests-tc
 * 
 * This class tests that each possible TOS is enqueued in the right band
 */
class PfifoFastQueueDiscTosPrioritization : public TestCase
{
public:
  PfifoFastQueueDiscTosPrioritization ();
  virtual ~PfifoFastQueueDiscTosPrioritization ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet and checks that it's added to the proper band.
   * 
   * \param queue The queue disc.
   * \param tos The TOS of the packet.
   * \param band Expected band.
   */
  void TestTosValue (Ptr<PfifoFastQueueDisc> queue, uint8_t tos, uint32_t band);
};

PfifoFastQueueDiscTosPrioritization::PfifoFastQueueDiscTosPrioritization ()
  : TestCase ("Test TOS-based prioritization")
{
}

PfifoFastQueueDiscTosPrioritization::~PfifoFastQueueDiscTosPrioritization ()
{
}

void
PfifoFastQueueDiscTosPrioritization::TestTosValue (Ptr<PfifoFastQueueDisc> queue, uint8_t tos, uint32_t band)
{
  Ptr<Packet> p = Create<Packet> (100);
  Ipv4Header ipHeader;
  ipHeader.SetPayloadSize (100);
  ipHeader.SetTos (tos);
  ipHeader.SetProtocol (6);
  SocketPriorityTag priorityTag;
  priorityTag.SetPriority (Socket::IpTos2Priority (tos));
  p->AddPacketTag (priorityTag);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, ipHeader);
  queue->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queue->GetInternalQueue (band)->GetNPackets (), 1, "enqueued to unexpected band");
  queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queue->GetInternalQueue (band)->GetNPackets (), 0, "unable to dequeue");
}

void
PfifoFastQueueDiscTosPrioritization::DoRun (void)
{
  Ptr<PfifoFastQueueDisc> queueDisc = CreateObject<PfifoFastQueueDisc> ();
  for (uint16_t i = 0; i < 3; i++)
    {
      Ptr<DropTailQueue<QueueDiscItem> > queue = CreateObject<DropTailQueue<QueueDiscItem> > ();
      bool ok = queue->SetAttributeFailSafe ("MaxSize", StringValue ("1000p"));
      NS_TEST_ASSERT_MSG_EQ (ok, true, "unable to set attribute");
      queueDisc->AddInternalQueue (queue);
    }
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (0)->GetNPackets (), 0, "initialized non-zero");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 0, "initialized non-zero");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (2)->GetNPackets (), 0, "initialized non-zero");

                                      // Service name       priority         band
  TestTosValue (queueDisc, 0x00, 1);  // Normal service  -> Best Effort (0) -> 1
  TestTosValue (queueDisc, 0x02, 1);  // MMC             -> Best Effort (0) -> 1
  TestTosValue (queueDisc, 0x04, 1);  // MR              -> Best Effort (0) -> 1
  TestTosValue (queueDisc, 0x06, 1);  // MMC+MR          -> Best Effort (0) -> 1
  TestTosValue (queueDisc, 0x08, 2);  // Max. Throughput -> Bulk (2)        -> 2
  TestTosValue (queueDisc, 0x0a, 2);  // MMC+MT          -> Bulk (2)        -> 2
  TestTosValue (queueDisc, 0x0c, 2);  // MR+MT           -> Bulk (2)        -> 2
  TestTosValue (queueDisc, 0x0e, 2);  // MMC+MR+MT       -> Bulk (2)        -> 2
  TestTosValue (queueDisc, 0x10, 0);  // Minimize Delay  -> Interactive (6) -> 0
  TestTosValue (queueDisc, 0x12, 0);  // MMC+MD          -> Interactive (6) -> 0
  TestTosValue (queueDisc, 0x14, 0);  // MR+MD           -> Interactive (6) -> 0
  TestTosValue (queueDisc, 0x16, 0);  // MMC+MR+MD       -> Interactive (6) -> 0
  TestTosValue (queueDisc, 0x18, 1);  // MT+MD           -> Int. Bulk (4)   -> 1
  TestTosValue (queueDisc, 0x1a, 1);  // MMC+MT+MD       -> Int. Bulk (4)   -> 1
  TestTosValue (queueDisc, 0x1c, 1);  // MR+MT+MD        -> Int. Bulk (4)   -> 1
  TestTosValue (queueDisc, 0x1e, 1);  // MMC+MR+MT+MD    -> Int. Bulk (4)   -> 1
  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests that each possible DSCP is enqueued in the right band.
 */
class PfifoFastQueueDiscDscpPrioritization : public TestCase
{
public:
  PfifoFastQueueDiscDscpPrioritization ();
  virtual ~PfifoFastQueueDiscDscpPrioritization ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet and checks that it's added to the proper band.
   * 
   * \param queue The queue disc.
   * \param dscp The DSCP of the packet.
   * \param band Expected band.
   */
  void TestDscpValue (Ptr<PfifoFastQueueDisc> queue, Ipv4Header::DscpType dscp, uint32_t band);
};

PfifoFastQueueDiscDscpPrioritization::PfifoFastQueueDiscDscpPrioritization ()
  : TestCase ("Test DSCP-based prioritization")
{
}

PfifoFastQueueDiscDscpPrioritization::~PfifoFastQueueDiscDscpPrioritization ()
{
}

void
PfifoFastQueueDiscDscpPrioritization::TestDscpValue (Ptr<PfifoFastQueueDisc> queue, Ipv4Header::DscpType dscp, uint32_t band)
{
  Ptr<Packet> p = Create<Packet> (100);
  Ipv4Header ipHeader;
  ipHeader.SetPayloadSize (100);
  ipHeader.SetProtocol (6);
  ipHeader.SetDscp (dscp);
  SocketPriorityTag priorityTag;
  priorityTag.SetPriority (Socket::IpTos2Priority (ipHeader.GetTos ()));
  p->AddPacketTag (priorityTag);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, ipHeader);
  queue->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queue->GetInternalQueue (band)->GetNPackets (), 1, "enqueued to unexpected band");
  queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ (queue->GetInternalQueue (band)->GetNPackets (), 0, "unable to dequeue");
}

void
PfifoFastQueueDiscDscpPrioritization::DoRun (void)
{
  Ptr<PfifoFastQueueDisc> queueDisc = CreateObject<PfifoFastQueueDisc> ();
  for (uint16_t i = 0; i < 3; i++)
    {
      Ptr<DropTailQueue<QueueDiscItem> > queue = CreateObject<DropTailQueue<QueueDiscItem> > ();
      bool ok = queue->SetAttributeFailSafe ("MaxSize", StringValue ("1000p"));
      NS_TEST_ASSERT_MSG_EQ (ok, true, "unable to set attribute");
      queueDisc->AddInternalQueue (queue);
    }
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (0)->GetNPackets (), 0, "initialized non-zero");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 0, "initialized non-zero");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (2)->GetNPackets (), 0, "initialized non-zero");

                                                          // priority         band
  TestDscpValue (queueDisc, Ipv4Header::DscpDefault, 1);  // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_EF, 1);      // Int. Bulk (4)   -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF11, 2);    // Bulk (2)        -> 2
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF21, 2);    // Bulk (2)        -> 2
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF31, 2);    // Bulk (2)        -> 2
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF41, 2);    // Bulk (2)        -> 2
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF12, 0);    // Interactive (6) -> 0
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF22, 0);    // Interactive (6) -> 0
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF32, 0);    // Interactive (6) -> 0
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF42, 0);    // Interactive (6) -> 0
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF13, 1);    // Int. Bulk (4)   -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF23, 1);    // Int. Bulk (4)   -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF33, 1);    // Int. Bulk (4)   -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_AF43, 1);    // Int. Bulk (4)   -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS1, 1);     // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS2, 1);     // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS3, 1);     // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS4, 1);     // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS5, 1);     // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS6, 1);     // Best Effort (0) -> 1
  TestDscpValue (queueDisc, Ipv4Header::DSCP_CS7, 1);     // Best Effort (0) -> 1
  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests that each band is txqueuelen deep.
 */
class PfifoFastQueueDiscOverflow : public TestCase
{
public:
  PfifoFastQueueDiscOverflow ();
  virtual ~PfifoFastQueueDiscOverflow ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue a packet.
   * 
   * \param queue The queue disc.
   * \param dscp The DSCP of the packet.
   */
  void AddPacket (Ptr<PfifoFastQueueDisc> queue, Ipv4Header::DscpType dscp);
};

PfifoFastQueueDiscOverflow::PfifoFastQueueDiscOverflow ()
  : TestCase ("Test queue overflow")
{
}

PfifoFastQueueDiscOverflow::~PfifoFastQueueDiscOverflow ()
{
}

void
PfifoFastQueueDiscOverflow::AddPacket (Ptr<PfifoFastQueueDisc> queue, Ipv4Header::DscpType dscp)
{
  Ptr<Packet> p = Create<Packet> (100);
  Ipv4Header ipHeader;
  ipHeader.SetPayloadSize (100);
  ipHeader.SetProtocol (6);
  ipHeader.SetDscp (dscp);
  SocketPriorityTag priorityTag;
  priorityTag.SetPriority (Socket::IpTos2Priority (ipHeader.GetTos ()));
  p->AddPacketTag (priorityTag);
  Address dest;
  Ptr<Ipv4QueueDiscItem> item = Create<Ipv4QueueDiscItem> (p, dest, 0, ipHeader);
  queue->Enqueue (item);
}

void
PfifoFastQueueDiscOverflow::DoRun (void)
{
  Ptr<PfifoFastQueueDisc> queueDisc = CreateObjectWithAttributes<PfifoFastQueueDisc> ("MaxSize", StringValue ("6p"));
  Ptr<DropTailQueue<QueueDiscItem> > band0 = CreateObjectWithAttributes<DropTailQueue<QueueDiscItem> > ("MaxSize", StringValue ("6p"));
  Ptr<DropTailQueue<QueueDiscItem> > band1 = CreateObjectWithAttributes<DropTailQueue<QueueDiscItem> > ("MaxSize", StringValue ("6p"));
  Ptr<DropTailQueue<QueueDiscItem> > band2 = CreateObjectWithAttributes<DropTailQueue<QueueDiscItem> > ("MaxSize", StringValue ("6p"));
  queueDisc->AddInternalQueue (band0);
  queueDisc->AddInternalQueue (band1);
  queueDisc->AddInternalQueue (band2);

  // Add two packets per each band
  AddPacket (queueDisc, Ipv4Header::DSCP_AF42); // 0
  AddPacket (queueDisc, Ipv4Header::DSCP_AF42); // 0
  AddPacket (queueDisc, Ipv4Header::DSCP_AF13); // 1
  AddPacket (queueDisc, Ipv4Header::DSCP_AF13); // 1
  AddPacket (queueDisc, Ipv4Header::DSCP_AF11); // 2
  AddPacket (queueDisc, Ipv4Header::DSCP_AF11); // 2
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (0)->GetNPackets (), 2, "unexpected queue depth");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 2, "unexpected queue depth");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (2)->GetNPackets (), 2, "unexpected queue depth");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 6, "unexpected queue depth");
  // Add a third packet to each band
  AddPacket (queueDisc, Ipv4Header::DSCP_AF42); // 0
  AddPacket (queueDisc, Ipv4Header::DSCP_AF13); // 1
  AddPacket (queueDisc, Ipv4Header::DSCP_AF11); // 2
  // Bands should still have two packets each
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (0)->GetNPackets (), 2, "unexpected queue depth");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 2, "unexpected queue depth");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (2)->GetNPackets (), 2, "unexpected queue depth");
  NS_TEST_ASSERT_MSG_EQ (queueDisc->QueueDisc::GetNPackets (), 6, "unexpected queue depth");
  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * This class tests that packets without a priority tag are handled by placing
 * them into band 1.
 */
class PfifoFastQueueDiscNoPriority : public TestCase
{
public:
  PfifoFastQueueDiscNoPriority ();
  virtual ~PfifoFastQueueDiscNoPriority ();

private:
  virtual void DoRun (void);
};

PfifoFastQueueDiscNoPriority::PfifoFastQueueDiscNoPriority ()
  : TestCase ("Test queue with no priority tag")
{
}

PfifoFastQueueDiscNoPriority::~PfifoFastQueueDiscNoPriority ()
{
}

void
PfifoFastQueueDiscNoPriority::DoRun (void)
{
  // all packets with non-IP headers should enqueue in band 1
  Ptr<PfifoFastQueueDisc> queueDisc = CreateObject<PfifoFastQueueDisc> ();
  for (uint16_t i = 0; i < 3; i++)
    {
      Ptr<DropTailQueue<QueueDiscItem> > queue = CreateObject<DropTailQueue<QueueDiscItem> > ();
      bool ok = queue->SetAttributeFailSafe ("MaxSize", StringValue ("1000p"));
      NS_TEST_ASSERT_MSG_EQ (ok, true, "unable to set attribute");
      queueDisc->AddInternalQueue (queue);
    }
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 0, "unexpected queue depth");
  Ptr<Packet> p;
  p = Create<Packet> ();
  Ptr<Ipv6QueueDiscItem> item;
  Ipv6Header ipv6Header;
  Address dest;
  item = Create<Ipv6QueueDiscItem> (p, dest, 0, ipv6Header);
  queueDisc->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 1, "unexpected queue depth");
  p = Create<Packet> (reinterpret_cast<const uint8_t*> ("hello, world"), 12);
  item = Create<Ipv6QueueDiscItem> (p, dest, 0, ipv6Header);
  queueDisc->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 2, "unexpected queue depth");
  p = Create<Packet> (100);
  uint8_t *buf = new uint8_t[100];
  uint8_t counter = 0;
  for (uint32_t i = 0; i < 100; i++)
    {
      buf[i] = counter++;
    }
  p->CopyData (buf, 100);
  item = Create<Ipv6QueueDiscItem> (p, dest, 0, ipv6Header);
  queueDisc->Enqueue (item);
  NS_TEST_ASSERT_MSG_EQ (queueDisc->GetInternalQueue (1)->GetNPackets (), 3, "unexpected queue depth");
  delete[] buf;
  Simulator::Destroy ();
}

/**
 * \ingroup system-tests-tc
 * 
 * PfifoFast queue disc test suite.
 */
class PfifoFastQueueDiscTestSuite : public TestSuite
{
public:
  PfifoFastQueueDiscTestSuite ();
};

PfifoFastQueueDiscTestSuite::PfifoFastQueueDiscTestSuite ()
  : TestSuite ("pfifo-fast-queue-disc", UNIT)
{
  AddTestCase (new PfifoFastQueueDiscTosPrioritization, TestCase::QUICK);
  AddTestCase (new PfifoFastQueueDiscDscpPrioritization, TestCase::QUICK);
  AddTestCase (new PfifoFastQueueDiscOverflow, TestCase::QUICK);
  AddTestCase (new PfifoFastQueueDiscNoPriority, TestCase::QUICK);
}

/// Do not forget to allocate an instance of this TestSuite.
static PfifoFastQueueDiscTestSuite g_pfifoFastQueueTestSuite;
