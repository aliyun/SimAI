/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 NITK Surathkal
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
 * Ported to ns-3 by: Vignesh Kannan <vignesh2496@gmail.com>
 *                    Harsh Lara <harshapplefan@gmail.com>
 *                    Jendaipou Palmei <jendaipoupalmei@gmail.com>
 *                    Shefali Gupta <shefaligups11@gmail.com>
 *                    Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

#include "ns3/test.h"
#include "ns3/cobalt-queue-disc.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

using namespace ns3;
/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Cobalt Queue Disc Test Item
 */
class CobaltQueueDiscTestItem : public QueueDiscItem
{
public:
  /**
   * Constructor
   *
   * \param p packet
   * \param addr address
   * \param protocol protocol
   * \param ecnCapable ECN capable
   */
  CobaltQueueDiscTestItem (Ptr<Packet> p, const Address & addr,uint16_t protocol, bool ecnCapable);
  virtual ~CobaltQueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  CobaltQueueDiscTestItem (const CobaltQueueDiscTestItem &) = delete;
  CobaltQueueDiscTestItem & operator = (const CobaltQueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark (void);

private:
  CobaltQueueDiscTestItem ();

  bool m_ecnCapablePacket; ///< ECN capable packet?
};

CobaltQueueDiscTestItem::CobaltQueueDiscTestItem (Ptr<Packet> p, const Address & addr,uint16_t protocol, bool ecnCapable)
  : QueueDiscItem (p, addr, ecnCapable),
    m_ecnCapablePacket (ecnCapable)
{
}

CobaltQueueDiscTestItem::~CobaltQueueDiscTestItem ()
{
}

void
CobaltQueueDiscTestItem::AddHeader (void)
{
}

bool
CobaltQueueDiscTestItem::Mark (void)
{
  if (m_ecnCapablePacket)
    {
      return true;
    }
  return false;
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 1: simple enqueue/dequeue with no drops
 */
class CobaltQueueDiscBasicEnqueueDequeue : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the mode
   */
  CobaltQueueDiscBasicEnqueueDequeue (QueueSizeUnit mode);
  virtual void DoRun (void);

  /**
   * Queue test size function
   * \param queue the queue disc
   * \param size the size
   * \param error the error string
   *
   */

private:
  QueueSizeUnit m_mode; ///< mode
};

CobaltQueueDiscBasicEnqueueDequeue::CobaltQueueDiscBasicEnqueueDequeue (QueueSizeUnit mode)
  : TestCase ("Basic enqueue and dequeue operations, and attribute setting" + std::to_string (mode))
{
  m_mode = mode;
}

void
CobaltQueueDiscBasicEnqueueDequeue::DoRun (void)
{
  Ptr<CobaltQueueDisc> queue = CreateObject<CobaltQueueDisc> ();

  uint32_t pktSize = 1000;
  uint32_t modeSize = 0;

  Address dest;

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Interval", StringValue ("50ms")), true,
                         "Verify that we can actually set the attribute Interval");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Target", StringValue ("4ms")), true,
                         "Verify that we can actually set the attribute Target");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");

  if (m_mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
    }
  else if (m_mode == QueueSizeUnit::PACKETS)
    {
      modeSize = 1;
    }
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 1500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  queue->Initialize ();

  Ptr<Packet> p1, p2, p3, p4, p5, p6;
  p1 = Create<Packet> (pktSize);
  p2 = Create<Packet> (pktSize);
  p3 = Create<Packet> (pktSize);
  p4 = Create<Packet> (pktSize);
  p5 = Create<Packet> (pktSize);
  p6 = Create<Packet> (pktSize);

  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 0 * modeSize, "There should be no packets in queue");
  queue->Enqueue (Create<CobaltQueueDiscTestItem> (p1, dest,0, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 1 * modeSize, "There should be one packet in queue");
  queue->Enqueue (Create<CobaltQueueDiscTestItem> (p2, dest,0, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 2 * modeSize, "There should be two packets in queue");
  queue->Enqueue (Create<CobaltQueueDiscTestItem> (p3, dest,0, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 3 * modeSize, "There should be three packets in queue");
  queue->Enqueue (Create<CobaltQueueDiscTestItem> (p4, dest,0, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 4 * modeSize, "There should be four packets in queue");
  queue->Enqueue (Create<CobaltQueueDiscTestItem> (p5, dest,0, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 5 * modeSize, "There should be five packets in queue");
  queue->Enqueue (Create<CobaltQueueDiscTestItem> (p6, dest,0, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 6 * modeSize, "There should be six packets in queue");

  NS_TEST_ASSERT_MSG_EQ (queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::OVERLIMIT_DROP), 0, "There should be no packets being dropped due to full queue");

  Ptr<QueueDiscItem> item;

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the first packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 5 * modeSize, "There should be five packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p1->GetUid (), "was this the first packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the second packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 4 * modeSize, "There should be four packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p2->GetUid (), "Was this the second packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the third packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 3 * modeSize, "There should be three packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p3->GetUid (), "Was this the third packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the forth packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 2 * modeSize, "There should be two packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p4->GetUid (), "Was this the fourth packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the fifth packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 1 * modeSize, "There should be one packet in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p5->GetUid (), "Was this the fifth packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the last packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 0 * modeSize, "There should be zero packet in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p6->GetUid (), "Was this the sixth packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item == 0), true, "There are really no packets in queue");

  NS_TEST_ASSERT_MSG_EQ (queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP), 0, "There should be no packet drops according to Cobalt algorithm");
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 2: Cobalt Queue Disc Drop Test Item
 */
class CobaltQueueDiscDropTest : public TestCase
{
public:
  CobaltQueueDiscDropTest ();
  virtual void DoRun (void);
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   */
  void Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt);
  /**
   * Run Cobalt test function
   * \param mode the mode
   */
  void RunDropTest (QueueSizeUnit mode);
  /**
   * Enqueue the given number of packets, each of the given size,
   * at different times.
   *
   * \param queue the queue disc
   * \param size the packet size in bytes
   * \param nPkt the number of packets
   */
  void EnqueueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt);

};

CobaltQueueDiscDropTest::CobaltQueueDiscDropTest ()
  : TestCase ("Drop tests verification for both packets and bytes mode")
{
}

void
CobaltQueueDiscDropTest::RunDropTest (QueueSizeUnit mode)

{
  uint32_t pktSize = 1500;
  uint32_t modeSize = 0;
  Ptr<CobaltQueueDisc> queue = CreateObject<CobaltQueueDisc> ();

  if (mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
    }
  else if (mode == QueueSizeUnit::PACKETS)
    {
      modeSize = 1;
    }

  queue = CreateObject<CobaltQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, modeSize * 100))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");
  queue->Initialize ();

  if (mode == QueueSizeUnit::BYTES)
    {
      EnqueueWithDelay (queue, pktSize, 200);
    }
  else
    {
      EnqueueWithDelay (queue, 1, 200);
    }

  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();

  QueueDisc::Stats st = queue->GetStats ();

// The Pdrop value should increase, from it's default value of zero
  NS_TEST_ASSERT_MSG_NE (queue->GetPdrop (), 0, "Pdrop should be non-zero");
  NS_TEST_ASSERT_MSG_NE (st.GetNDroppedPackets (CobaltQueueDisc::OVERLIMIT_DROP), 0, "Drops due to queue overflow should be non-zero");
}

void
CobaltQueueDiscDropTest::EnqueueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
  Address dest;
  double delay = 0.01;  // enqueue packets with delay
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &CobaltQueueDiscDropTest::Enqueue, this, queue, size, 1);
    }
}

void
CobaltQueueDiscDropTest::Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CobaltQueueDiscTestItem> (Create<Packet> (size), dest, 0, true));
    }
}

void
CobaltQueueDiscDropTest::DoRun (void)
{
  RunDropTest (QueueSizeUnit::PACKETS);
  RunDropTest (QueueSizeUnit::BYTES);
  Simulator::Destroy ();
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 3: Cobalt Queue Disc ECN marking Test Item
 */
class CobaltQueueDiscMarkTest : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the mode
   */
  CobaltQueueDiscMarkTest (QueueSizeUnit mode);
  virtual void DoRun (void);

private:
  /**
   * \brief Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   * \param ecnCapable ECN capable traffic
   */
  void Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt, bool ecnCapable);
  /**
   * \brief Dequeue function
   * \param queue the queue disc
   * \param modeSize the mode size
   * \param testCase the test case number
   */
  void Dequeue (Ptr<CobaltQueueDisc> queue, uint32_t modeSize, uint32_t testCase);
  /**
   * \brief Drop next tracer function
   * \param oldVal the old value of m_dropNext
   * \param newVal the new value of m_dropNext
   */
  void DropNextTracer (int64_t oldVal, int64_t newVal);
  QueueSizeUnit m_mode; ///< mode
  uint32_t m_dropNextCount;    ///< count the number of times m_dropNext is recalculated
  uint32_t nPacketsBeforeFirstDrop; ///< Number of packets in the queue before first drop
  uint32_t nPacketsBeforeFirstMark; ///< Number of packets in the queue before first mark
};

CobaltQueueDiscMarkTest::CobaltQueueDiscMarkTest (QueueSizeUnit mode)
  : TestCase ("Basic mark operations")
{
  m_mode = mode;
  m_dropNextCount = 0;
}

void
CobaltQueueDiscMarkTest::DropNextTracer ([[maybe_unused]] int64_t oldVal, [[maybe_unused]] int64_t newVal)
{
  m_dropNextCount++;
}

void
CobaltQueueDiscMarkTest::DoRun (void)
{
  // Test is divided into 3 sub test cases:
  // 1) Packets are not ECN capable.
  // 2) Packets are ECN capable.
  // 3) Some packets are ECN capable.

  // Test case 1
  Ptr<CobaltQueueDisc> queue = CreateObject<CobaltQueueDisc> ();
  uint32_t pktSize = 1000;
  uint32_t modeSize = 0;
  nPacketsBeforeFirstDrop = 0;
  nPacketsBeforeFirstMark = 0;

  if (m_mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
    }
  else if (m_mode == QueueSizeUnit::PACKETS)
    {
      modeSize = 1;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (false)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");
  queue->Initialize ();

  // Not-ECT traffic to induce packet drop
  Enqueue (queue, pktSize, 20, false);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // there should not be any dropped packets in this interval
  Time waitUntilFirstDequeue =  2 * queue->GetTarget ();
  Simulator::Schedule (waitUntilFirstDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 1);

  // This dequeue should cause a packet to be dropped
  Time waitUntilSecondDequeue = waitUntilFirstDequeue + 2 * queue->GetInterval ();
  Simulator::Schedule (waitUntilSecondDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 1);

  Simulator::Run ();
  Simulator::Destroy ();

  // Test case 2, queue with ECN capable traffic for marking of packets instead of dropping
  queue = CreateObject<CobaltQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");
  queue->Initialize ();

  // ECN capable traffic
  Enqueue (queue, pktSize, 20, true);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // there should not be any marked packets in this interval
  Simulator::Schedule (waitUntilFirstDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 2);

  // This dequeue should cause a packet to be marked
  Simulator::Schedule (waitUntilSecondDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 2);

  // This dequeue should cause a packet to be marked as dropnext is equal to current time
  Simulator::Schedule (waitUntilSecondDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 2);

  // In dropping phase and it's time for next packet to be marked
  // the dequeue should cause additional packet to be marked
  Simulator::Schedule (waitUntilSecondDequeue * 2, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 2);

  Simulator::Run ();
  Simulator::Destroy ();

  // Test case 3, some packets are ECN capable
  queue = CreateObject<CobaltQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");
  queue->Initialize ();

  // First 3 packets in the queue are ecnCapable
  Enqueue (queue, pktSize, 3, true);
  // Rest of the packet are not ecnCapable
  Enqueue (queue, pktSize, 17, false);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // there should not be any marked packets in this interval
  Simulator::Schedule (waitUntilFirstDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 3);

  // This dequeue should cause a packet to be marked
  Simulator::Schedule (waitUntilSecondDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 3);

  // This dequeue should cause a packet to be marked as dropnext is equal to current time
  Simulator::Schedule (waitUntilSecondDequeue, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 3);

  // In dropping phase and it's time for next packet to be dropped as packets are not ECN capable
  // the dequeue should cause packet to be dropped
  Simulator::Schedule (waitUntilSecondDequeue * 2, &CobaltQueueDiscMarkTest::Dequeue, this, queue, modeSize, 3);

  Simulator::Run ();
  Simulator::Destroy ();
}

void
CobaltQueueDiscMarkTest::Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt, bool ecnCapable)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CobaltQueueDiscTestItem> (Create<Packet> (size), dest, 0, ecnCapable));
    }
}

void
CobaltQueueDiscMarkTest::Dequeue (Ptr<CobaltQueueDisc> queue, uint32_t modeSize, uint32_t testCase)
{
  uint32_t initialMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
  uint32_t initialQSize = queue->GetCurrentSize ().GetValue ();
  uint32_t initialDropNext = queue->GetDropNext ();
  Time currentTime = Simulator::Now ();
  uint32_t currentDropCount = 0;
  uint32_t currentMarkCount = 0;

  if (initialMarkCount > 0 && currentTime.GetNanoSeconds () > initialDropNext && testCase == 3)
    {
      queue->TraceConnectWithoutContext ("DropNext", MakeCallback (&CobaltQueueDiscMarkTest::DropNextTracer, this));
    }

  if (initialQSize != 0)
    {
      Ptr<QueueDiscItem> item = queue->Dequeue ();
      if (testCase == 1)
        {
          currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
          if (currentDropCount != 0)
            {
              nPacketsBeforeFirstDrop = initialQSize;
            }
        }
       if (testCase == 2)
        {
          if (initialMarkCount == 0 && currentTime > queue->GetTarget ())
            {
              if (currentTime < queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be 1 packet dequeued.");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 0, "We are not in dropping state."
                                        "Sojourn time has just gone above target from below."
                                        "Hence, there should be no marked packets");
                }
              else if (currentTime >= queue->GetInterval ())
                {
                  nPacketsBeforeFirstMark = initialQSize;
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize,
                                        "Sojourn time has been above target for at least interval."
                                        "We enter the dropping state and perform initial packet marking"
                                        "So there should be only 1 more packet dequeued.");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 1, "There should be 1 marked packet");
                }
            }
          else if (initialMarkCount > 0)
            {
              if (currentTime.GetNanoSeconds () <= initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                        "Sojourn is still above target."
                                        "There should be only 1 more packet dequeued");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 2, "There should be 2 marked packet as."
                                                              "current dropnext is equal to current time.");
                }
              else if (currentTime.GetNanoSeconds () > initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                        "It's time for packet to be marked"
                                        "So there should be only 1 more packet dequeued");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 3, "There should be 3 marked packet");
                  NS_TEST_EXPECT_MSG_EQ (nPacketsBeforeFirstDrop, nPacketsBeforeFirstMark, "Number of packets in the queue before drop should be equal"
                                        "to number of packets in the queue before first mark as the behavior until packet N should be the same.");
                }
            }
        }
      else if (testCase == 3)
        {
          if (initialMarkCount == 0 && currentTime > queue->GetTarget ())
            {
              if (currentTime < queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be 1 packet dequeued.");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 0, "We are not in dropping state."
                                        "Sojourn time has just gone above target from below."
                                        "Hence, there should be no marked packets");
                }
              else if (currentTime >= queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize,
                                        "Sojourn time has been above target for at least interval."
                                        "We enter the dropping state and perform initial packet marking"
                                        "So there should be only 1 more packet dequeued.");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 1, "There should be 1 marked packet");
                }
            }
          else if (initialMarkCount > 0)
            {
              if (currentTime.GetNanoSeconds () <= initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                        "Sojourn is still above target."
                                        "So there should be only 1 more packet dequeued");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 2, "There should be 2 marked packet"
                                                              "as dropnext is equal to current time");
                }
              else if (currentTime.GetNanoSeconds () > initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP);
                  currentMarkCount = queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::FORCED_MARK);
                  NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - (m_dropNextCount + 1) * modeSize, "We are in dropping state."
                                        "It's time for packet to be dropped as packets are not ecnCapable"
                                        "The number of packets dequeued equals to the number of times m_dropNext is updated plus initial dequeue");
                  NS_TEST_EXPECT_MSG_EQ (currentDropCount, m_dropNextCount, "The number of drops equals to the number of times m_dropNext is updated");
                  NS_TEST_EXPECT_MSG_EQ (currentMarkCount, 2, "There should still be only 2 marked packet");
                }
            }
        }
    }
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 4: Cobalt Queue Disc CE Threshold marking Test Item
 */
class CobaltQueueDiscCeThresholdTest : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the queue size unit mode
   */
  CobaltQueueDiscCeThresholdTest (QueueSizeUnit mode);
  virtual ~CobaltQueueDiscCeThresholdTest ();

private:
  virtual void DoRun (void);
  /**
   * \brief Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   */
  void Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt);
  /**
   * \brief Enqueue with delay function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   * \param delay the delay between the enqueueing of the packets
   */
  void EnqueueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt, Time delay);
  /**
   * \brief Dequeue function
   * \param queue the queue disc
   * \param modeSize the mode size
   */
  void Dequeue (Ptr<CobaltQueueDisc> queue, uint32_t modeSize);
  /**
   * \brief Dequeue with delay function
   * \param queue the queue disc
   * \param modeSize the mode size
   * \param nPkt the number of packets
   * \param delay the delay between the enqueueing of the packets
   */
  void DequeueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t modeSize, uint32_t nPkt, Time delay);
  QueueSizeUnit m_mode; ///< mode
};

CobaltQueueDiscCeThresholdTest::CobaltQueueDiscCeThresholdTest (QueueSizeUnit mode)
  : TestCase ("Test CE Threshold marking")
{
  m_mode = mode;
}

CobaltQueueDiscCeThresholdTest::~CobaltQueueDiscCeThresholdTest ()
{
}

void
CobaltQueueDiscCeThresholdTest::Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CobaltQueueDiscTestItem> (Create<Packet> (size), dest, 0, true));
    }
}

void
CobaltQueueDiscCeThresholdTest::EnqueueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt, Time delay)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay.GetSeconds ())), &CobaltQueueDiscCeThresholdTest::Enqueue, this, queue, size, 1);
    }
}

void
CobaltQueueDiscCeThresholdTest::Dequeue (Ptr<CobaltQueueDisc> queue,  uint32_t modeSize)
{
  Ptr<QueueDiscItem> item = queue->Dequeue ();

  if (Simulator::Now () > MilliSeconds (11) && Simulator::Now () < MilliSeconds (28))
    {
      NS_TEST_EXPECT_MSG_EQ (queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::CE_THRESHOLD_EXCEEDED_MARK), 1, "There should be only 1 packet"
                                                                 "mark, the delay between the enqueueing of the packets decreased after the"
                                                                 "1st mark (packet enqueued at 11ms) and increased for the packet enqueued after 20.6ms."
                                                                 "Queue delay remains below or equal to 1ms for the packet enqueued before 28ms");
    }
  if (Simulator::Now () > MilliSeconds (31) )
    {
      NS_TEST_EXPECT_MSG_EQ (queue->GetStats ().GetNMarkedPackets (CobaltQueueDisc::CE_THRESHOLD_EXCEEDED_MARK), 3, "There should be 3 packet"
                                                                 "marks, the delay between the enqueueing of the packets decreased after 1st mark"
                                                                 "(packet enqueued at 11ms) and increased for the packet enqueued after 20.6ms."
                                                                 "Queue delay remains below 1ms for the packets enqueued before 28ms and increases"
                                                                 "for the packets enqueued after 28ms.");
    }
}

void
CobaltQueueDiscCeThresholdTest::DequeueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t modeSize, uint32_t nPkt, Time delay)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay.GetSeconds ())), &CobaltQueueDiscCeThresholdTest::Dequeue, this, queue, modeSize);
    }
}

void
CobaltQueueDiscCeThresholdTest::DoRun (void)
{
  Ptr<CobaltQueueDisc> queue = CreateObject<CobaltQueueDisc> ();
  uint32_t pktSize = 1000;
  uint32_t modeSize = 0;

  if (m_mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
    }
  else if (m_mode == QueueSizeUnit::PACKETS)
    {
      modeSize = 1;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("CeThreshold", TimeValue (MilliSeconds (1))),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");
  queue->Initialize ();

  // Enqueue 11 packets every 1ms
  EnqueueWithDelay (queue, pktSize, 11, MilliSeconds (1));

  // With every dequeue, queue delay increases by 0.1ms as packet enqueues every 1ms while dequeues at 1.1ms
  // so at 11th dequeue, the dequeued packet should be marked.
  Time dequeueInterval =  MicroSeconds (1100);
  DequeueWithDelay (queue, modeSize, 11, dequeueInterval);

  // First mark occured for the packet enqueued at 11ms, ideally TCP would decrease sending rate
  // which can be simulated by increasing interval between subsequent enqueues, so packets are now enqueued with a delay 1.2ms.
  Time waitUntilFirstMark = MilliSeconds(11);
  Simulator::Schedule (waitUntilFirstMark, &CobaltQueueDiscCeThresholdTest::EnqueueWithDelay, this, queue, pktSize, 9, MicroSeconds (1200));

  // Keep dequeueing with the same delay
  Simulator::Schedule (waitUntilFirstMark, &CobaltQueueDiscCeThresholdTest::DequeueWithDelay, this, queue, modeSize, 9, dequeueInterval);

  // Queue delay becomes 0.2ms for the packet enqueued at 20.6ms, time to decrease interval between subsequent enqueues,
  // as ideally TCP would again start increasing sending rate
  Time waitUntilDecreasingEnqueueDelay = waitUntilFirstMark + MilliSeconds(9);
  Simulator::Schedule (waitUntilDecreasingEnqueueDelay, &CobaltQueueDiscCeThresholdTest::EnqueueWithDelay, this, queue, pktSize, 10, MilliSeconds (1));

  // Keep dequeueing with the same delay
  Simulator::Schedule (waitUntilFirstMark, &CobaltQueueDiscCeThresholdTest::DequeueWithDelay, this, queue, modeSize, 10, dequeueInterval);

  Simulator::Run ();
  Simulator::Destroy ();
}


/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 5: Cobalt Queue Disc Enhanced Blue Test Item
 * This test checks that the Blue Enhancement is working correctly. This test checks that Pdrop should increase as expected with
 * a certain number of dropped packets which is fixed by assigning a stream to the uniform random variable.
 * This test case is divided into 2 sub test cases
 * 1) With Blue Enhancement enabled, the test checks that Pdrop should increase as expected with
 *    a certain number of dropped packets which is fixed by assigning a stream to the uniform random variable.
 * 2) Without Blue Enhancement, Pdrop should remain zero and there should not be any dropped packets as ECN is enabled.
 */
class CobaltQueueDiscEnhancedBlueTest : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the queue size unit mode
   */
  CobaltQueueDiscEnhancedBlueTest (QueueSizeUnit mode);
  virtual ~CobaltQueueDiscEnhancedBlueTest ();

private:
  virtual void DoRun (void);
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   */
  void Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt);
  /**
   * \brief Dequeue function
   * \param queue the queue disc
   */
  void Dequeue (Ptr<CobaltQueueDisc> queue);
  /**
   * \brief Dequeue with delay function
   * \param queue the queue disc
   * \param nPkt the number of packets
   * \param delay the delay between the enqueueing of the packets
   */
  void DequeueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t nPkt, Time delay);
  QueueSizeUnit m_mode; ///< mode
};

CobaltQueueDiscEnhancedBlueTest::CobaltQueueDiscEnhancedBlueTest (QueueSizeUnit mode)
  : TestCase ("Enhanced Blue tests verification for both packets and bytes mode")
{
  m_mode = mode;
}

CobaltQueueDiscEnhancedBlueTest::~CobaltQueueDiscEnhancedBlueTest ()
{
}

void
CobaltQueueDiscEnhancedBlueTest::DoRun (void)

{
  uint32_t pktSize = 1500;
  uint32_t modeSize = 0;
  Ptr<CobaltQueueDisc> queue = CreateObject<CobaltQueueDisc> ();

  if (m_mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
    }
  else if (m_mode == QueueSizeUnit::PACKETS)
    {
      modeSize = 1;
    }
  queue->Initialize ();
  queue->AssignStreams (1);
  Enqueue (queue, modeSize, 200);
  DequeueWithDelay (queue, 100, MilliSeconds (10));

  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();

  QueueDisc::Stats st = queue->GetStats ();

  // The Pdrop value should increase, from it's default value of zero
  NS_TEST_ASSERT_MSG_EQ (queue->GetPdrop (), 0.234375, "Pdrop should be increased by 1/256 for every packet whose sojourn time is above 400ms."
                        " From the 41st dequeue until the last one, sojourn time is above 400ms, so 60 packets have sojourn time above 400ms"
                        "hence Pdrop should be increased 60*(1/256) which is 0.234375");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP), 49, "There should a fixed number of drops (49 here)");
  Simulator::Destroy ();


  queue = CreateObject<CobaltQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("BlueThreshold", TimeValue (Time::Max ())), true,
                         "Disable Blue enhancement");
  queue->Initialize ();
  Enqueue (queue, modeSize, 200);
  DequeueWithDelay (queue, 100, MilliSeconds (10));

  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();

  st = queue->GetStats ();

  NS_TEST_ASSERT_MSG_EQ (queue->GetPdrop (), 0, "Pdrop should be zero");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (CobaltQueueDisc::TARGET_EXCEEDED_DROP), 0, "There should not any dropped packets");
  Simulator::Destroy ();
}

void
CobaltQueueDiscEnhancedBlueTest::Enqueue (Ptr<CobaltQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CobaltQueueDiscTestItem> (Create<Packet> (size), dest, 0, true));
    }
}

void
CobaltQueueDiscEnhancedBlueTest::Dequeue (Ptr<CobaltQueueDisc> queue)
{
  Ptr<QueueDiscItem> item = queue->Dequeue ();
}

void
CobaltQueueDiscEnhancedBlueTest::DequeueWithDelay (Ptr<CobaltQueueDisc> queue, uint32_t nPkt, Time delay)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay.GetSeconds ())), &CobaltQueueDiscEnhancedBlueTest::Dequeue, this, queue);
    }
}


/**
 * The COBALT queue disc test suite.
 */
static class CobaltQueueDiscTestSuite : public TestSuite
{
public:
  CobaltQueueDiscTestSuite ()
    : TestSuite ("cobalt-queue-disc", UNIT)
  {
    // Test 1: simple enqueue/dequeue with no drops
    AddTestCase (new CobaltQueueDiscBasicEnqueueDequeue (PACKETS), TestCase::QUICK);
    AddTestCase (new CobaltQueueDiscBasicEnqueueDequeue (BYTES), TestCase::QUICK);
    // Test 2: Drop test
    AddTestCase (new CobaltQueueDiscDropTest (), TestCase::QUICK);
    // Test 3: Mark test
    AddTestCase (new CobaltQueueDiscMarkTest (PACKETS), TestCase::QUICK);
    AddTestCase (new CobaltQueueDiscMarkTest (BYTES), TestCase::QUICK);
    // Test 4: CE threshold marking test
    AddTestCase (new CobaltQueueDiscCeThresholdTest (PACKETS), TestCase::QUICK);
    AddTestCase (new CobaltQueueDiscCeThresholdTest (BYTES), TestCase::QUICK);
    // Test 4: Blue enhancement test
    AddTestCase (new CobaltQueueDiscEnhancedBlueTest (PACKETS), TestCase::QUICK);
    AddTestCase (new CobaltQueueDiscEnhancedBlueTest (BYTES), TestCase::QUICK);
  }
} g_cobaltQueueTestSuite; ///< the test suite
