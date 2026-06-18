/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Universita' degli Studi di Napoli Federico II
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
 * Authors: Stefano Avallone <stavallo@unina.it>
 *
 */

#include "ns3/test.h"
#include "ns3/queue-disc.h"
#include "ns3/drop-tail-queue.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include <map>

using namespace ns3;

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Queue Disc Test Item
 */
class qdTestItem : public QueueDiscItem
{
public:
  /**
   * Constructor
   *
   * \param p the packet
   * \param addr the address
   */
  qdTestItem (Ptr<Packet> p, const Address & addr);
  virtual ~qdTestItem ();
  virtual void AddHeader (void);
  virtual bool Mark (void);
};

qdTestItem::qdTestItem (Ptr<Packet> p, const Address & addr)
  : QueueDiscItem (p, addr, 0)
{
}

qdTestItem::~qdTestItem ()
{
}

void
qdTestItem::AddHeader (void)
{
}

bool
qdTestItem::Mark (void)
{
  return false;
}


/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test Child Queue Disc that may drop packets before enqueue or after dequeue
 */
class TestChildQueueDisc : public QueueDisc
{
public:
  /**
   * Constructor
   */
  TestChildQueueDisc ();
  virtual ~TestChildQueueDisc ();
  virtual bool DoEnqueue (Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue (void);
  virtual bool CheckConfig (void);
  virtual void InitializeParams (void);

  // Reasons for dropping packets
  static constexpr const char* BEFORE_ENQUEUE = "Before enqueue";  //!< Drop before enqueue
  static constexpr const char* AFTER_DEQUEUE = "After dequeue";  //!< Drop after dequeue
};

TestChildQueueDisc::TestChildQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::SINGLE_INTERNAL_QUEUE)
{
}

TestChildQueueDisc::~TestChildQueueDisc ()
{
}

bool
TestChildQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  // Drop the packet if there are already 4 packets queued
  if (GetNPackets () >= 4)
    {
      DropBeforeEnqueue (item, BEFORE_ENQUEUE);
      return false;
    }
  return GetInternalQueue (0)->Enqueue (item);
}

Ptr<QueueDiscItem>
TestChildQueueDisc::DoDequeue (void) 
{
  Ptr<QueueDiscItem> item = GetInternalQueue (0)->Dequeue ();

  // Drop the packet if at least 2 packets remain in the queue
  while (GetNPackets () >= 2)
    {
      DropAfterDequeue (item, AFTER_DEQUEUE);
      item = GetInternalQueue (0)->Dequeue ();
    }
  return item;
}

bool
TestChildQueueDisc::CheckConfig (void)
{
  AddInternalQueue (CreateObject<DropTailQueue<QueueDiscItem> > ());
  return true;
}

void
TestChildQueueDisc::InitializeParams (void)
{
}


/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test Parent Queue Disc having a child of type TestChildQueueDisc
 */
class TestParentQueueDisc : public QueueDisc
{
public:
  /**
   * Constructor
   */
  TestParentQueueDisc ();
  virtual ~TestParentQueueDisc ();
  virtual bool DoEnqueue (Ptr<QueueDiscItem> item);
  virtual Ptr<QueueDiscItem> DoDequeue (void);
  virtual bool CheckConfig (void);
  virtual void InitializeParams (void);
};

TestParentQueueDisc::TestParentQueueDisc ()
  : QueueDisc (QueueDiscSizePolicy::SINGLE_CHILD_QUEUE_DISC)
{
}

TestParentQueueDisc::~TestParentQueueDisc ()
{
}

bool
TestParentQueueDisc::DoEnqueue (Ptr<QueueDiscItem> item)
{
  return GetQueueDiscClass (0)->GetQueueDisc ()->Enqueue (item);
}

Ptr<QueueDiscItem>
TestParentQueueDisc::DoDequeue (void) 
{
  return GetQueueDiscClass (0)->GetQueueDisc ()->Dequeue ();
}

bool
TestParentQueueDisc::CheckConfig (void)
{
  Ptr<QueueDiscClass> c = CreateObject<QueueDiscClass> ();
  c->SetQueueDisc (CreateObject<TestChildQueueDisc> ());
  AddQueueDiscClass (c);
  return true;
}

void
TestParentQueueDisc::InitializeParams (void)
{
}


/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Keep statistics based on traces
 */
class TestCounter
{
public:
  /**
   * Constructor
   */
  TestCounter ();
  virtual ~TestCounter ();

  /**
   * Connect private methods to the queue disc traces
   * \param qd the queue disc
   */
  void ConnectTraces (Ptr<QueueDisc> qd);

private:
  /**
   * Update statistics after a packet has been enqueued
   * \param item the enqueued packet
   */
  void PacketEnqueued (Ptr<const QueueDiscItem> item);
  /**
   * Update statistics after a packet has been dequeued
   * \param item the dequeued packet
   */
  void PacketDequeued (Ptr<const QueueDiscItem> item);
  /**
   * Update statistics after a packet has been dropped before enqueue
   * \param item the dropped packet
   * \param reason the reason why the packet was dropped
   */
  void PacketDbe (Ptr<const QueueDiscItem> item, const char* reason);
  /**
   * Update statistics after a packet has been dropped after dequeue
   * \param item the dropped packet
   * \param reason the reason why the packet was dropped
   */
  void PacketDad (Ptr<const QueueDiscItem> item, const char* reason);

  uint32_t m_nPackets;     //!< Number of queued packets
  uint32_t m_nBytes;       //!< Number of queued bytes
  uint32_t m_nDbePackets;  //!< Number of packets dropped before enqueue
  uint32_t m_nDbeBytes;    //!< Number of packets dropped before enqueue
  uint32_t m_nDadPackets;  //!< Number of packets dropped after dequeue
  uint32_t m_nDadBytes;    //!< Number of packets dropped after dequeue

  friend class QueueDiscTracesTestCase;   //!< Needs to access private members
};

TestCounter::TestCounter ()
  : m_nPackets (0),
    m_nBytes (0),
    m_nDbePackets (0),
    m_nDbeBytes (0),
    m_nDadPackets (0),
    m_nDadBytes (0)
{
}

TestCounter::~TestCounter ()
{
}

void
TestCounter::PacketEnqueued (Ptr<const QueueDiscItem> item)
{
  m_nPackets++;
  m_nBytes += item->GetSize ();
}

void
TestCounter::PacketDequeued (Ptr<const QueueDiscItem> item)
{
  m_nPackets--;
  m_nBytes -= item->GetSize ();
}

void
TestCounter::PacketDbe (Ptr<const QueueDiscItem> item, const char* reason)
{
  m_nDbePackets++;
  m_nDbeBytes += item->GetSize ();
}


void
TestCounter::PacketDad (Ptr<const QueueDiscItem> item, const char* reason)
{
  m_nDadPackets++;
  m_nDadBytes += item->GetSize ();
}

void
TestCounter::ConnectTraces (Ptr<QueueDisc> qd)
{
  qd->TraceConnectWithoutContext ("Enqueue", MakeCallback (&TestCounter::PacketEnqueued, this));
  qd->TraceConnectWithoutContext ("Dequeue", MakeCallback (&TestCounter::PacketDequeued, this));
  qd->TraceConnectWithoutContext ("DropBeforeEnqueue", MakeCallback (&TestCounter::PacketDbe, this));
  qd->TraceConnectWithoutContext ("DropAfterDequeue", MakeCallback (&TestCounter::PacketDad, this));
}


/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Queue Disc Traces Test Case
 *
 * This test case makes use of a test queue disc acting as root queue disc and
 * having a single child queue disc, which has a single DropTail queue. The
 * Enqueue, Dequeue, DropBeforeEnqueue and DropAfterDequeue traces of both queue
 * discs are connected to the methods of TestCounter objects that keep track of
 * the amount of packets/bytes queued inside each queue disc and the cumulative
 * amount of packets/bytes dropped before enqueue and after dequeue. A series of
 * enqueue, dequeue and peek operations are performed on the root queue disc and
 * both the statistics kept by the QueueDisc class and the values computed by the
 * TestCounter objects are compared with the expected values.
 */
class QueueDiscTracesTestCase : public TestCase
{
public:
  QueueDiscTracesTestCase ();
  virtual void DoRun (void);

  /**
   * Check that queued packets/bytes are consistent with what is expected
   * \param qd the queue disc
   * \param nPackets the expected number of packets
   * \param nBytes the expected number of bytes
   */
  void CheckQueued (Ptr<QueueDisc> qd, uint32_t nPackets, uint32_t nBytes);
  /**
   * Check that packets/bytes dropped before enqueue are consistent with what is expected
   * \param qd the queue disc
   * \param nDbePackets the expected number of packets
   * \param nDbeBytes the expected number of bytes
   */
  void CheckDroppedBeforeEnqueue (Ptr<QueueDisc> qd, uint32_t nDbePackets, uint32_t nDbeBytes);
  /**
   * Check that packets/bytes dropped after dequeue are consistent with what is expected
   * \param qd the queue disc
   * \param nDadPackets the expected number of packets
   * \param nDadBytes the expected number of bytes
   */
  void CheckDroppedAfterDequeue (Ptr<QueueDisc> qd, uint32_t nDadPackets, uint32_t nDadBytes);

private:
  std::map<Ptr<QueueDisc>,TestCounter> m_counter;  //!< counters for the queue discs
};

QueueDiscTracesTestCase::QueueDiscTracesTestCase ()
  : TestCase ("Sanity check on the queue disc traces and statistics")
{
}

void
QueueDiscTracesTestCase::CheckQueued (Ptr<QueueDisc> qd, uint32_t nPackets, uint32_t nBytes)
{
  NS_TEST_ASSERT_MSG_EQ (qd->GetNPackets (), nPackets,
                         "Verify that the number of queued packets is computed correctly");
  NS_TEST_ASSERT_MSG_EQ (m_counter[qd].m_nPackets, nPackets,
                         "Verify that the number of queued packets is computed correctly");

  NS_TEST_ASSERT_MSG_EQ (qd->GetNBytes (), nBytes,
                         "Verify that the number of queued bytes is computed correctly");
  NS_TEST_ASSERT_MSG_EQ (m_counter[qd].m_nBytes, nBytes,
                         "Verify that the number of queued bytes is computed correctly");
}

void
QueueDiscTracesTestCase::CheckDroppedBeforeEnqueue (Ptr<QueueDisc> qd, uint32_t nDbePackets, uint32_t nDbeBytes)
{
  QueueDisc::Stats stats = qd->GetStats ();

  NS_TEST_ASSERT_MSG_EQ (stats.nTotalDroppedPacketsBeforeEnqueue, nDbePackets,
                         "Verify that the number of packets dropped before enqueue is computed correctly");
  NS_TEST_ASSERT_MSG_EQ (m_counter[qd].m_nDbePackets, nDbePackets,
                         "Verify that the number of packets dropped before enqueue is computed correctly");

  NS_TEST_ASSERT_MSG_EQ (stats.nTotalDroppedBytesBeforeEnqueue, nDbeBytes,
                         "Verify that the number of bytes dropped before enqueue is computed correctly");
  NS_TEST_ASSERT_MSG_EQ (m_counter[qd].m_nDbeBytes, nDbeBytes,
                         "Verify that the number of bytes dropped before enqueue is computed correctly");
}

void
QueueDiscTracesTestCase::CheckDroppedAfterDequeue (Ptr<QueueDisc> qd, uint32_t nDadPackets, uint32_t nDadBytes)
{
  QueueDisc::Stats stats = qd->GetStats ();

  NS_TEST_ASSERT_MSG_EQ (stats.nTotalDroppedPacketsAfterDequeue, nDadPackets,
                         "Verify that the number of packets dropped after dequeue is computed correctly");
  NS_TEST_ASSERT_MSG_EQ (m_counter[qd].m_nDadPackets, nDadPackets,
                         "Verify that the number of packets dropped after dequeue is computed correctly");

  NS_TEST_ASSERT_MSG_EQ (stats.nTotalDroppedBytesAfterDequeue, nDadBytes,
                         "Verify that the number of bytes dropped after dequeue is computed correctly");
  NS_TEST_ASSERT_MSG_EQ (m_counter[qd].m_nDadBytes, nDadBytes,
                         "Verify that the number of bytes dropped after dequeue is computed correctly");
}

void
QueueDiscTracesTestCase::DoRun (void)
{
  Address dest;
  uint32_t pktSizeUnit = 100;
  Ptr<const QueueDiscItem> item;

  // Create queue discs
  Ptr<QueueDisc> root = CreateObject<TestParentQueueDisc> ();
  root->Initialize ();

  Ptr<QueueDisc> child = root->GetQueueDiscClass (0)->GetQueueDisc ();

  NS_TEST_ASSERT_MSG_NE (child, 0, "The child queue disc has not been created");

  // Create counters and connect traces to the counters
  m_counter.emplace (root, TestCounter ());
  m_counter.emplace (child, TestCounter ());

  m_counter[root].ConnectTraces (root);
  m_counter[child].ConnectTraces (child);

  // Enqueue 4 packets. They must all be enqueued
  for (uint16_t i = 1; i <= 4; i++)
    {
      root->Enqueue (Create<qdTestItem> (Create<Packet>(pktSizeUnit * i), dest));

      CheckQueued (root, i, pktSizeUnit * i * (i+1) / 2);
      CheckDroppedBeforeEnqueue (root, 0, 0);
      CheckDroppedAfterDequeue (root, 0, 0);

      CheckQueued (child, i, pktSizeUnit * i * (i+1) / 2);
      CheckDroppedBeforeEnqueue (child, 0, 0);
      CheckDroppedAfterDequeue (child, 0, 0);
    }

  // The fifth packet is dropped before enqueue by the child queue disc.
  // The packet drop is notified to the root queue disc.
  root->Enqueue (Create<qdTestItem> (Create<Packet>(pktSizeUnit * 5), dest));

  CheckQueued (root, 4, pktSizeUnit * 10);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 0, 0);

  CheckQueued (child, 4, pktSizeUnit * 10);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 0, 0);

  // Peek one packet. The default DoPeek method asks the root queue disc to dequeue
  // a packet, even though the statistics are not updated and the dequeue trace is
  // not fired. The root queue disc asks the child queue disc to dequeue a packet.
  // In this case, two packets (those having size of 100 and 200 bytes) are dequeued
  // and dropped by the child queue disc. Therefore, the dequeue trace of the root
  // queue disc is fired twice and the packet drops are notified to the root queue
  // disc to reflect the fact that two packets are no longer in the queue disc.
  // The peeked packet is still part of the root queue disc, but no longer part
  // of the child queue disc.
  item = root->Peek ();

  NS_TEST_ASSERT_MSG_NE (item, 0, "A packet must have been returned");
  NS_TEST_ASSERT_MSG_EQ (item->GetSize (), pktSizeUnit * 3, "The peeked packet has not the expected size");

  CheckQueued (root, 2, pktSizeUnit * 7);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 1, pktSizeUnit * 4);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  // Peek again. Nothing changes.
  item = root->Peek ();

  NS_TEST_ASSERT_MSG_NE (item, 0, "A packet must have been returned");
  NS_TEST_ASSERT_MSG_EQ (item->GetSize (), pktSizeUnit * 3, "The peeked packet has not the expected size");

  CheckQueued (root, 2, pktSizeUnit * 7);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 1, pktSizeUnit * 4);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  // Dequeue one packet. The root queue disc returns the previously peeked packet.
  item = root->Dequeue ();

  NS_TEST_ASSERT_MSG_NE (item, 0, "A packet must have been returned");
  NS_TEST_ASSERT_MSG_EQ (item->GetSize (), pktSizeUnit * 3, "The dequeued packet has not the expected size");

  CheckQueued (root, 1, pktSizeUnit * 4);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 1, pktSizeUnit * 4);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  // Dequeue the last packet.
  item = root->Dequeue ();

  NS_TEST_ASSERT_MSG_NE (item, 0, "A packet must have been returned");
  NS_TEST_ASSERT_MSG_EQ (item->GetSize (), pktSizeUnit * 4, "The dequeued packet has not the expected size");

  CheckQueued (root, 0, 0);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 0, 0);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  // Peek a packet. No packet is left.
  item = root->Peek ();

  NS_TEST_ASSERT_MSG_EQ (item, 0, "No packet must have been returned");

  CheckQueued (root, 0, 0);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 0, 0);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  // Enqueue one packet.
  root->Enqueue (Create<qdTestItem> (Create<Packet>(pktSizeUnit), dest));

  CheckQueued (root, 1, pktSizeUnit);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 1, pktSizeUnit);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  // Dequeue one packet.
  item = root->Dequeue ();

  NS_TEST_ASSERT_MSG_NE (item, 0, "A packet must have been returned");
  NS_TEST_ASSERT_MSG_EQ (item->GetSize (), pktSizeUnit, "The dequeued packet has not the expected size");

  CheckQueued (root, 0, 0);
  CheckDroppedBeforeEnqueue (root, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (root, 2, pktSizeUnit * 3);

  CheckQueued (child, 0, 0);
  CheckDroppedBeforeEnqueue (child, 1, pktSizeUnit * 5);
  CheckDroppedAfterDequeue (child, 2, pktSizeUnit * 3);

  Simulator::Destroy ();
}


/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Queue Disc Traces Test Suite
 */
static class QueueDiscTracesTestSuite : public TestSuite
{
public:
  QueueDiscTracesTestSuite ()
    : TestSuite ("queue-disc-traces", UNIT)
  {
    AddTestCase (new QueueDiscTracesTestCase (), TestCase::QUICK);
  }
} g_queueDiscTracesTestSuite; ///< the test suite
