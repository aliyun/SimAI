/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
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
#include "ns3/fifo-queue-disc.h"
#include "ns3/queue.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/object-factory.h"
#include <vector>

using namespace ns3;

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Fifo Queue Disc Test Item
 */
class FifoQueueDiscTestItem : public QueueDiscItem
{
public:
  /**
   * Constructor
   *
   * \param p the packet
   * \param addr the address
   */
  FifoQueueDiscTestItem (Ptr<Packet> p, const Address & addr);
  virtual ~FifoQueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  FifoQueueDiscTestItem (const FifoQueueDiscTestItem &) = delete;
  FifoQueueDiscTestItem & operator = (const FifoQueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark (void);

private:
  FifoQueueDiscTestItem ();
};

FifoQueueDiscTestItem::FifoQueueDiscTestItem (Ptr<Packet> p, const Address & addr)
  : QueueDiscItem (p, addr, 0)
{
}

FifoQueueDiscTestItem::~FifoQueueDiscTestItem ()
{
}

void
FifoQueueDiscTestItem::AddHeader (void)
{
}

bool
FifoQueueDiscTestItem::Mark (void)
{
  return false;
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Fifo Queue Disc Test Case
 */
class FifoQueueDiscTestCase : public TestCase
{
public:
  FifoQueueDiscTestCase ();
  virtual void DoRun (void);
private:
  /**
   * Run test function
   * \param mode the test mode
   */
  void RunFifoTest (QueueSizeUnit mode);
  /**
   * Run test function
   * \param q the queue disc
   * \param qSize the expected size of the queue disc
   * \param pktSize the packet size
   */
  void DoRunFifoTest (Ptr<FifoQueueDisc> q, uint32_t qSize, uint32_t pktSize);
};

FifoQueueDiscTestCase::FifoQueueDiscTestCase ()
  : TestCase ("Sanity check on the fifo queue disc implementation")
{
}

void
FifoQueueDiscTestCase::DoRunFifoTest (Ptr<FifoQueueDisc> q, uint32_t qSize, uint32_t pktSize)
{
  std::vector<uint64_t> uids;
  Ptr<Packet> p;
  Ptr<QueueDiscItem> item;
  Address dest;
  uint32_t modeSize = (q->GetMaxSize ().GetUnit () == QueueSizeUnit::PACKETS ? 1 : pktSize);
  uint32_t numPackets = qSize / modeSize;

  NS_TEST_ASSERT_MSG_EQ (q->GetCurrentSize ().GetValue (), 0, "The queue disc should be empty");

  // create and enqueue numPackets packets and store their UIDs; check they are all enqueued
  for (uint32_t i = 1; i <= numPackets; i++)
    {
      p = Create<Packet> (pktSize);
      uids.push_back (p->GetUid ());
      q->Enqueue (Create<FifoQueueDiscTestItem> (p, dest));
      NS_TEST_ASSERT_MSG_EQ (q->GetCurrentSize ().GetValue (), i * modeSize, "There should be " << i << " packet(s) in there");
    }

  // no room for another packet
  NS_TEST_ASSERT_MSG_EQ (q->Enqueue (Create<FifoQueueDiscTestItem> (p, dest)), 
                         false, "There should be no room for another packet");

  // dequeue and check packet order
  for (uint32_t i = 1; i <= numPackets; i++)
    {
      item = q->Dequeue ();
      NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "A packet should have been dequeued");
      NS_TEST_ASSERT_MSG_EQ (q->GetCurrentSize ().GetValue (), (numPackets-i) * modeSize, "There should be " << numPackets-i << " packet(s) in there");
      NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), uids[i-1], "was this the right packet?");
    }

  item = q->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item == 0), true, "There are really no packets in there");
}

void
FifoQueueDiscTestCase::RunFifoTest (QueueSizeUnit mode)
{
  Ptr<FifoQueueDisc> queue;
  uint32_t numPackets = 10;
  uint32_t pktSize = 1000;
  uint32_t modeSize = (mode == QueueSizeUnit::PACKETS ? 1 : pktSize);

  // test 1: set the limit on the queue disc before initialization
  queue = CreateObject<FifoQueueDisc> ();

  NS_TEST_ASSERT_MSG_EQ (queue->GetNInternalQueues (), 0, "Verify that the queue disc has no internal queue");

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize",
                                                      QueueSizeValue (QueueSize (mode, numPackets*modeSize))),
                         true, "Verify that we can actually set the attribute MaxSize");

  queue->Initialize ();

  DoRunFifoTest (queue, numPackets*modeSize, pktSize);


  // test 2: set the limit on the queue disc after initialization
  queue = CreateObject<FifoQueueDisc> ();

  NS_TEST_ASSERT_MSG_EQ (queue->GetNInternalQueues (), 0, "Verify that the queue disc has no internal queue");

  queue->Initialize ();

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize",
                                                      QueueSizeValue (QueueSize (mode, numPackets*modeSize))),
                         true, "Verify that we can actually set the attribute MaxSize");

  DoRunFifoTest (queue, numPackets*modeSize, pktSize);


  // test 3: set the limit on the internal queue before initialization
  queue = CreateObject<FifoQueueDisc> ();

  NS_TEST_ASSERT_MSG_EQ (queue->GetNInternalQueues (), 0, "Verify that the queue disc has no internal queue");

  ObjectFactory factory;
  factory.SetTypeId ("ns3::DropTailQueue<QueueDiscItem>");
  if (mode == QueueSizeUnit::PACKETS)
    {
      factory.Set ("MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, numPackets)));
    }
  else
    {
      factory.Set ("MaxSize", QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, numPackets*pktSize)));
    }
  queue->AddInternalQueue (factory.Create<QueueDisc::InternalQueue> ());

  queue->Initialize ();

  DoRunFifoTest (queue, numPackets*modeSize, pktSize);


  // test 4: set the limit on the internal queue after initialization
  queue = CreateObject<FifoQueueDisc> ();

  NS_TEST_ASSERT_MSG_EQ (queue->GetNInternalQueues (), 0, "Verify that the queue disc has no internal queue");

  queue->Initialize ();

  NS_TEST_ASSERT_MSG_EQ (queue->GetNInternalQueues (), 1, "Verify that the queue disc got an internal queue");

  Ptr<QueueDisc::InternalQueue> iq = queue->GetInternalQueue (0);

  if (mode == QueueSizeUnit::PACKETS)
    {
      NS_TEST_ASSERT_MSG_EQ (iq->SetAttributeFailSafe ("MaxSize",
                                                       QueueSizeValue (QueueSize  (QueueSizeUnit::PACKETS, numPackets))),
                             true, "Verify that we can actually set the attribute MaxSize on the internal queue");
    }
  else
    {
      NS_TEST_ASSERT_MSG_EQ (iq->SetAttributeFailSafe ("MaxSize",
                                                       QueueSizeValue (QueueSize  (QueueSizeUnit::BYTES, numPackets*pktSize))),
                             true, "Verify that we can actually set the attribute MaxSize on the internal queue");
    }

  DoRunFifoTest (queue, numPackets*modeSize, pktSize);
}

void
FifoQueueDiscTestCase::DoRun (void)
{
  RunFifoTest (QueueSizeUnit::PACKETS);
  RunFifoTest (QueueSizeUnit::BYTES);
  Simulator::Destroy ();
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Fifo Queue Disc Test Suite
 */
static class FifoQueueDiscTestSuite : public TestSuite
{
public:
  FifoQueueDiscTestSuite ()
    : TestSuite ("fifo-queue-disc", UNIT)
  {
    AddTestCase (new FifoQueueDiscTestCase (), TestCase::QUICK);
  }
} g_fifoQueueTestSuite; ///< the test suite
