/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 ResiliNets, ITTC, University of Kansas
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
 * Author: Truc Anh N Nguyen <trucanh524@gmail.com>
 * Modified by:   Pasquale Imputato <p.imputato@gmail.com>
 *
 */

#include "ns3/test.h"
#include "ns3/codel-queue-disc.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/simulator.h"

using namespace ns3;

// The following code borrowed from Linux codel.h, for unit testing
#define REC_INV_SQRT_BITS_ns3 (8 * sizeof(uint16_t))
/* or sizeof_in_bits(rec_inv_sqrt) */
/* needed shift to get a Q0.32 number from rec_inv_sqrt */
#define REC_INV_SQRT_SHIFT_ns3 (32 - REC_INV_SQRT_BITS_ns3)

static uint16_t _codel_Newton_step (uint16_t rec_inv_sqrt, uint32_t count)
{
  uint32_t invsqrt = ((uint32_t)rec_inv_sqrt) << REC_INV_SQRT_SHIFT_ns3;
  uint32_t invsqrt2 = ((uint64_t)invsqrt * invsqrt) >> 32;
  uint64_t val = (3LL << 32) - ((uint64_t)count * invsqrt2);

  val >>= 2; /* avoid overflow in following multiply */
  val = (val * invsqrt) >> (32 - 2 + 1);
  return static_cast<uint16_t>(val >> REC_INV_SQRT_SHIFT_ns3);
}

static uint32_t _reciprocal_scale (uint32_t val, uint32_t ep_ro)
{
  return (uint32_t)(((uint64_t)val * ep_ro) >> 32);
}
// End Linux borrow

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Codel Queue Disc Test Item
 */
class CodelQueueDiscTestItem : public QueueDiscItem {
public:
  /**
   * Constructor
   *
   * \param p packet
   * \param addr address
   * \param ecnCapable ECN capable
   */
  CodelQueueDiscTestItem (Ptr<Packet> p, const Address & addr, bool ecnCapable);
  virtual ~CodelQueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  CodelQueueDiscTestItem (const CodelQueueDiscTestItem &) = delete;
  CodelQueueDiscTestItem & operator = (const CodelQueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark(void);

private:
  CodelQueueDiscTestItem ();
  
  bool m_ecnCapablePacket; ///< ECN capable packet?
};

CodelQueueDiscTestItem::CodelQueueDiscTestItem (Ptr<Packet> p, const Address & addr, bool ecnCapable)
  : QueueDiscItem (p, addr, ecnCapable),
    m_ecnCapablePacket (ecnCapable)
{
}

CodelQueueDiscTestItem::~CodelQueueDiscTestItem ()
{
}

void
CodelQueueDiscTestItem::AddHeader (void)
{
}

bool
CodelQueueDiscTestItem::Mark (void)
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
class CoDelQueueDiscBasicEnqueueDequeue : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the mode
   */
  CoDelQueueDiscBasicEnqueueDequeue (QueueSizeUnit mode);
  virtual void DoRun (void);

private:
  QueueSizeUnit m_mode; ///< mode
};

CoDelQueueDiscBasicEnqueueDequeue::CoDelQueueDiscBasicEnqueueDequeue (QueueSizeUnit mode)
  : TestCase ("Basic enqueue and dequeue operations, and attribute setting")
{
  m_mode = mode;
}

void
CoDelQueueDiscBasicEnqueueDequeue::DoRun (void)
{
  Ptr<CoDelQueueDisc> queue = CreateObject<CoDelQueueDisc> ();

  uint32_t pktSize = 1000;
  uint32_t modeSize = 0;
  
  Address dest;

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinBytes", UintegerValue (pktSize)), true,
                         "Verify that we can actually set the attribute MinBytes");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Interval", StringValue ("50ms")), true,
                         "Verify that we can actually set the attribute Interval");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Target", StringValue ("4ms")), true,
                         "Verify that we can actually set the attribute Target");

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

  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 0 * modeSize, "There should be no packets in queue");
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p1, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 1 * modeSize, "There should be one packet in queue");
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p2, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 2 * modeSize, "There should be two packets in queue");
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p3, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 3 * modeSize, "There should be three packets in queue");
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p4, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 4 * modeSize, "There should be four packets in queue");
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p5, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 5 * modeSize, "There should be five packets in queue");
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p6, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 6 * modeSize, "There should be six packets in queue");

  NS_TEST_ASSERT_MSG_EQ (queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::OVERLIMIT_DROP),
                         0, "There should be no packets being dropped due to full queue");

  Ptr<QueueDiscItem> item;

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the first packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 5 * modeSize, "There should be five packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p1->GetUid (), "was this the first packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the second packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 4 * modeSize, "There should be four packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p2->GetUid (), "Was this the second packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the third packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 3 * modeSize, "There should be three packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p3->GetUid (), "Was this the third packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the forth packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 2 * modeSize, "There should be two packets in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p4->GetUid (), "Was this the fourth packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the fifth packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 1 * modeSize, "There should be one packet in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p5->GetUid (), "Was this the fifth packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the last packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 0 * modeSize, "There should be zero packet in queue");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p6->GetUid (), "Was this the sixth packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item == 0), true, "There are really no packets in queue");

  NS_TEST_ASSERT_MSG_EQ (queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP), 0,
                         "There should be no packet drops according to CoDel algorithm");
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 2: enqueue with drops due to queue overflow
 */
class CoDelQueueDiscBasicOverflow : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the mode
   */
  CoDelQueueDiscBasicOverflow (QueueSizeUnit mode);
  virtual void DoRun (void);

private:
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   */
  void Enqueue (Ptr<CoDelQueueDisc> queue, uint32_t size, uint32_t nPkt);
  QueueSizeUnit m_mode; ///< mode
};

CoDelQueueDiscBasicOverflow::CoDelQueueDiscBasicOverflow (QueueSizeUnit mode)
  : TestCase ("Basic overflow behavior")
{
  m_mode = mode;
}

void
CoDelQueueDiscBasicOverflow::DoRun (void)
{
  Ptr<CoDelQueueDisc> queue = CreateObject<CoDelQueueDisc> ();
  uint32_t pktSize = 1000;
  uint32_t modeSize = 0;

  Address dest;

  if (m_mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
    }
  else if (m_mode == QueueSizeUnit::PACKETS)
    {
      modeSize = 1;
    }

  Ptr<Packet> p1, p2, p3;
  p1 = Create<Packet> (pktSize);
  p2 = Create<Packet> (pktSize);
  p3 = Create<Packet> (pktSize);

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinBytes", UintegerValue (pktSize)), true,
                         "Verify that we can actually set the attribute MinBytes");

  queue->Initialize ();

  Enqueue (queue, pktSize, 500);
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p1, dest, false));
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p2, dest, false));
  queue->Enqueue (Create<CodelQueueDiscTestItem> (p3, dest, false));

  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize().GetValue (), 500 * modeSize, "There should be 500 packets in queue");
  NS_TEST_ASSERT_MSG_EQ (queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::OVERLIMIT_DROP),
                         3, "There should be three packets being dropped due to full queue");
}

void
CoDelQueueDiscBasicOverflow::Enqueue (Ptr<CoDelQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CodelQueueDiscTestItem> (Create<Packet> (size), dest, false));
    }
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 3: NewtonStep unit test - test against explicit port of Linux implementation
 */
class CoDelQueueDiscNewtonStepTest : public TestCase
{
public:
  CoDelQueueDiscNewtonStepTest ();
  virtual void DoRun (void);
};

CoDelQueueDiscNewtonStepTest::CoDelQueueDiscNewtonStepTest ()
  : TestCase ("NewtonStep arithmetic unit test")
{
}

void
CoDelQueueDiscNewtonStepTest::DoRun (void)
{
  Ptr<CoDelQueueDisc> queue = CreateObject<CoDelQueueDisc> ();

  // Spot check a few points in the expected operational range of
  // CoDelQueueDisc's m_count and m_recInvSqrt variables
  uint16_t result;
  for (uint16_t recInvSqrt = 0xff; recInvSqrt > 0; recInvSqrt /= 2)
    {
      for (uint32_t count = 1; count < 0xff; count *= 2)
        {
           result = queue->NewtonStep (recInvSqrt, count);
           // Test that ns-3 value is exactly the same as the Linux value
           NS_TEST_ASSERT_MSG_EQ (_codel_Newton_step (recInvSqrt, count), result,
                         "ns-3 NewtonStep() fails to match Linux equivalent");
        }
    }
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 4: ControlLaw unit test - test against explicit port of Linux implementation
 */
class CoDelQueueDiscControlLawTest : public TestCase
{
public:
  CoDelQueueDiscControlLawTest ();
  virtual void DoRun (void);
  /**
   * Codel control law function
   * \param t
   * \param interval
   * \param recInvSqrt
   * \returns the codel control law
   */
  uint32_t _codel_control_law (uint32_t t, uint32_t interval, uint32_t recInvSqrt);
};

CoDelQueueDiscControlLawTest::CoDelQueueDiscControlLawTest ()
  : TestCase ("ControlLaw arithmetic unit test")
{
}

// The following code borrowed from Linux codel.h,
// except the addition of queue parameter
uint32_t
CoDelQueueDiscControlLawTest::_codel_control_law (uint32_t t, uint32_t interval, uint32_t recInvSqrt)
{
  return t + _reciprocal_scale (interval, recInvSqrt << REC_INV_SQRT_SHIFT_ns3);
}
// End Linux borrrow

void
CoDelQueueDiscControlLawTest::DoRun (void)
{
  Ptr<CoDelQueueDisc> queue = CreateObject<CoDelQueueDisc> ();

  // Check a few points within the operational range of ControlLaw
  uint32_t interval = queue->Time2CoDel (MilliSeconds (100));

  uint32_t codelTimeVal;
  for (Time timeVal = Seconds (0); timeVal <= Seconds (20); timeVal += MilliSeconds (100))
    {
      for (uint16_t recInvSqrt = 0xff; recInvSqrt > 0; recInvSqrt /= 2)
        {
          codelTimeVal = queue->Time2CoDel (timeVal);
          uint32_t ns3Result = queue->ControlLaw (codelTimeVal, interval, recInvSqrt); 
          uint32_t linuxResult = _codel_control_law (codelTimeVal, interval, recInvSqrt); 
          NS_TEST_ASSERT_MSG_EQ (ns3Result, linuxResult, "Linux result for ControlLaw should equal ns-3 result");
        }
    }
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 5: enqueue/dequeue with drops according to CoDel algorithm
 */
class CoDelQueueDiscBasicDrop : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the mode
   */
  CoDelQueueDiscBasicDrop (QueueSizeUnit mode);
  virtual void DoRun (void);

private:
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   */
  void Enqueue (Ptr<CoDelQueueDisc> queue, uint32_t size, uint32_t nPkt);
  /** Dequeue function
   * \param queue the queue disc
   * \param modeSize the mode size
   */
  void Dequeue (Ptr<CoDelQueueDisc> queue, uint32_t modeSize);
  /**
   * Drop next tracer function
   * \param oldVal the old value
   * \param newVal the new value
   */
  void DropNextTracer (uint32_t oldVal, uint32_t newVal);
  QueueSizeUnit m_mode; ///< mode
  uint32_t m_dropNextCount;    ///< count the number of times m_dropNext is recalculated
};

CoDelQueueDiscBasicDrop::CoDelQueueDiscBasicDrop (QueueSizeUnit mode)
  : TestCase ("Basic drop operations")
{
  m_mode = mode;
  m_dropNextCount = 0;
}

void
CoDelQueueDiscBasicDrop::DropNextTracer ([[maybe_unused]] uint32_t oldVal, [[maybe_unused]] uint32_t newVal)
{
  m_dropNextCount++;
}

void
CoDelQueueDiscBasicDrop::DoRun (void)
{
  Ptr<CoDelQueueDisc> queue = CreateObject<CoDelQueueDisc> ();
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

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");

  queue->Initialize ();

  Enqueue (queue, pktSize, 20);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // the dequeue should be successful in this interval
  Time waitUntilFirstDequeue =  2 * queue->GetTarget ();
  Simulator::Schedule (waitUntilFirstDequeue, &CoDelQueueDiscBasicDrop::Dequeue, this, queue, modeSize);

  // This dequeue should cause a drop
  Time waitUntilSecondDequeue = waitUntilFirstDequeue + 2 * queue->GetInterval ();
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicDrop::Dequeue, this, queue, modeSize);

  // Although we are in dropping state, it's not time for next drop
  // the dequeue should not cause a drop
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicDrop::Dequeue, this, queue, modeSize);

  // In dropping time and it's time for next drop
  // the dequeue should cause additional packet drops
  Simulator::Schedule (waitUntilSecondDequeue * 2, &CoDelQueueDiscBasicDrop::Dequeue, this, queue, modeSize);

  Simulator::Run ();
  Simulator::Destroy ();
}

void
CoDelQueueDiscBasicDrop::Enqueue (Ptr<CoDelQueueDisc> queue, uint32_t size, uint32_t nPkt)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CodelQueueDiscTestItem> (Create<Packet> (size), dest, false));
    }
}

void
CoDelQueueDiscBasicDrop::Dequeue (Ptr<CoDelQueueDisc> queue, uint32_t modeSize)
{
  uint32_t initialDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
  uint32_t initialQSize = queue->GetCurrentSize ().GetValue ();
  uint32_t initialDropNext = queue->GetDropNext ();
  Time currentTime = Simulator::Now ();
  uint32_t currentDropCount = 0;

  if (initialDropCount > 0 && currentTime.GetMicroSeconds () >= initialDropNext)
    {
      queue->TraceConnectWithoutContext ("DropNext", MakeCallback (&CoDelQueueDiscBasicDrop::DropNextTracer, this));
    }

  if (initialQSize != 0)
    {
      Ptr<QueueDiscItem> item = queue->Dequeue ();
      if (initialDropCount == 0 && currentTime > queue->GetTarget ())
        {
          if (currentTime < queue->GetInterval ())
            {
              currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
              NS_TEST_EXPECT_MSG_EQ (currentDropCount, 0, "We are not in dropping state."
                                     "Sojourn time has just gone above target from below."
                                     "Hence, there should be no packet drops");
              NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be 1 packet dequeued.");

            }
          else if (currentTime >= queue->GetInterval ())
            {
              currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
              NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - 2 * modeSize, "Sojourn time has been above target for at least interval."
                                     "We enter the dropping state, perform initial packet drop, and dequeue the next."
                                     "So there should be 2 more packets dequeued.");
              NS_TEST_EXPECT_MSG_EQ (currentDropCount, 1, "There should be 1 packet drop");
            }
        }
      else if (initialDropCount > 0)
        { // In dropping state
          if (currentTime.GetMicroSeconds () < initialDropNext)
            {
              currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
              NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                     "Sojourn is still above target."
                                     "However, it's not time for next drop."
                                     "So there should be only 1 more packet dequeued");

              NS_TEST_EXPECT_MSG_EQ (currentDropCount, 1, "There should still be only 1 packet drop from the last dequeue");
            }
          else if (currentTime.GetMicroSeconds () >= initialDropNext)
            {
              currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
              NS_TEST_EXPECT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - (m_dropNextCount + 1) * modeSize, "We are in dropping state."
                                     "It's time for next drop."
                                     "The number of packets dequeued equals to the number of times m_dropNext is updated plus initial dequeue");
              NS_TEST_EXPECT_MSG_EQ (currentDropCount, 1 + m_dropNextCount, "The number of drops equals to the number of times m_dropNext is updated plus 1 from last dequeue");
            }
        }
    }
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Test 6: enqueue/dequeue with marks according to CoDel algorithm
 */
class CoDelQueueDiscBasicMark : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param mode the mode
   */
  CoDelQueueDiscBasicMark (QueueSizeUnit mode);
  virtual void DoRun (void);

private:
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   * \param ecnCapable ECN capable traffic
   */
  void Enqueue (Ptr<CoDelQueueDisc> queue, uint32_t size, uint32_t nPkt, bool ecnCapable);
  /** Dequeue function
   * \param queue the queue disc
   * \param modeSize the mode size
   * \param testCase the test case number
   */
  void Dequeue (Ptr<CoDelQueueDisc> queue, uint32_t modeSize, uint32_t testCase);
  /**
   * Drop next tracer function
   * \param oldVal the old value
   * \param newVal the new value
   */
  void DropNextTracer (uint32_t oldVal, uint32_t newVal);
  QueueSizeUnit m_mode; ///< mode
  uint32_t m_dropNextCount;    ///< count the number of times m_dropNext is recalculated
  uint32_t nPacketsBeforeFirstDrop; ///< Number of packets in the queue before first drop
  uint32_t nPacketsBeforeFirstMark; ///< Number of packets in the queue before first mark
};

CoDelQueueDiscBasicMark::CoDelQueueDiscBasicMark (QueueSizeUnit mode)
  : TestCase ("Basic mark operations")
{
  m_mode = mode;
  m_dropNextCount = 0;
}

void
CoDelQueueDiscBasicMark::DropNextTracer ([[maybe_unused]] uint32_t oldVal, [[maybe_unused]] uint32_t newVal)
{
  m_dropNextCount++;
}

void
CoDelQueueDiscBasicMark::DoRun (void)
{
  // Test is divided into 4 sub test cases:
  // 1) Packets are not ECN capable.
  // 2) Packets are ECN capable but marking due to exceeding CE threshold disabled
  // 3) Some packets are ECN capable, with CE threshold set to 2ms.
  // 4) Packets are ECN capable and CE threshold set to 2ms

  // Test case 1
  Ptr<CoDelQueueDisc> queue = CreateObject<CoDelQueueDisc> ();
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
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  
  queue->Initialize ();

  // Not-ECT traffic to induce packet drop
  Enqueue (queue, pktSize, 20, false);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // there should not be any dropped packets in this interval
  Time waitUntilFirstDequeue =  2 * queue->GetTarget ();
  Simulator::Schedule (waitUntilFirstDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 1);

  // This dequeue should cause a packet to be dropped
  Time waitUntilSecondDequeue = waitUntilFirstDequeue + 2 * queue->GetInterval ();
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 1);

  Simulator::Run ();
  Simulator::Destroy ();

  // Test case 2, queue with ECN capable traffic for marking of packets instead of dropping
  queue = CreateObject<CoDelQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");

  queue->Initialize ();

  // ECN capable traffic to induce packets to be marked
  Enqueue (queue, pktSize, 20, true);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // there should not be any target exceeded marked packets in this interval
  Simulator::Schedule (waitUntilFirstDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 2);

  // This dequeue should cause a packet to be marked
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 2);

  // Although we are in dropping state, it's not time for next packet to be target exceeded marked
  // the dequeue should not cause a packet to be target exceeded marked
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 2);

  // In dropping time and it's time for next packet to be target exceeded marked
  // the dequeue should cause additional packet to be target exceeded marked
  Simulator::Schedule (waitUntilSecondDequeue * 2, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 2);

  Simulator::Run ();
  Simulator::Destroy ();

  // Test case 3, some packets are ECN capable, with CE threshold set to 2ms
  queue = CreateObject<CoDelQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("CeThreshold", TimeValue (MilliSeconds (2))),
                         true, "Verify that we can actually set the attribute CeThreshold");
  
  queue->Initialize ();

  // First 3 packets in the queue are ecnCapable
  Enqueue (queue, pktSize, 3, true);
  // Rest of the packet are not ecnCapable
  Enqueue (queue, pktSize, 17, false);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // Although the first dequeue occurs with a sojourn time above target
  // there should not be any target exceeded marked packets in this interval
  Simulator::Schedule (waitUntilFirstDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 3);

  // This dequeue should cause a packet to be marked
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 3);

  // Although we are in dropping state, it's not time for next packet to be target exceeded marked
  // the dequeue should not cause a packet to be target exceeded marked
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 3);

  // In dropping time and it's time for next packet to be dropped as packets are not ECN capable
  // the dequeue should cause packet to be dropped
  Simulator::Schedule (waitUntilSecondDequeue * 2, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 3);

  Simulator::Run ();
  Simulator::Destroy ();

  // Test case 4, queue with ECN capable traffic and CeThreshold set for marking of packets instead of dropping
  queue = CreateObject<CoDelQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (m_mode, modeSize * 500))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)),
                         true, "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("CeThreshold", TimeValue (MilliSeconds (2))),
                         true, "Verify that we can actually set the attribute CeThreshold");

  queue->Initialize ();

  // ECN capable traffic to induce packets to be marked
  Enqueue (queue, pktSize, 20, true);
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 20 * modeSize, "There should be 20 packets in queue.");

  // The first dequeue occurs with a sojourn time below CE threshold
  // there should not any be CE threshold exceeded marked packets
  Simulator::Schedule (MilliSeconds (1), &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 4);

  // Sojourn time above CE threshold so this dequeue should cause a packet to be CE thershold exceeded marked
  Simulator::Schedule (MilliSeconds (3), &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 4);

  // the dequeue should cause a packet to be CE threshold exceeded marked
  Simulator::Schedule (waitUntilFirstDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 4);

  // In dropping time and it's time for next packet to be dropped but because of using ECN, packet should be marked
  Simulator::Schedule (waitUntilSecondDequeue, &CoDelQueueDiscBasicMark::Dequeue, this, queue, modeSize, 4);

  Simulator::Run ();
  Simulator::Destroy ();
}

void
CoDelQueueDiscBasicMark::Enqueue (Ptr<CoDelQueueDisc> queue, uint32_t size, uint32_t nPkt, bool ecnCapable)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<CodelQueueDiscTestItem> (Create<Packet> (size), dest, ecnCapable));
    }
}

void
CoDelQueueDiscBasicMark::Dequeue (Ptr<CoDelQueueDisc> queue, uint32_t modeSize, uint32_t testCase)
{
  uint32_t initialTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
  uint32_t initialCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
  uint32_t initialQSize = queue->GetCurrentSize ().GetValue ();
  uint32_t initialDropNext = queue->GetDropNext ();
  Time currentTime = Simulator::Now ();
  uint32_t currentDropCount = 0;
  uint32_t currentTargetMarkCount = 0;
  uint32_t currentCeThreshMarkCount = 0;

  if (initialTargetMarkCount > 0 && currentTime.GetMicroSeconds () >= initialDropNext && testCase == 3)
    {
      queue->TraceConnectWithoutContext ("DropNext", MakeCallback (&CoDelQueueDiscBasicMark::DropNextTracer, this));
    }

  if (initialQSize != 0)
    {
      Ptr<QueueDiscItem> item = queue->Dequeue ();
      if (testCase == 1)
        {
          currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
          if (currentDropCount == 1)
            {
              nPacketsBeforeFirstDrop = initialQSize;
            }
        }
      else if (testCase == 2)
        {
          if (initialTargetMarkCount == 0 && currentTime > queue->GetTarget ())
            {
              if (currentTime < queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be 1 packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 0, "We are not in dropping state."
                                        "Sojourn time has just gone above target from below."
                                        "Hence, there should be no target exceeded marked packets");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 0, "Marking due to CE threshold is disabled"
                                        "Hence, there should not be any CE threshold exceeded marked packet");
                }
              else if (currentTime >= queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  nPacketsBeforeFirstMark = initialQSize;
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "Sojourn time has been above target for at least interval."
                                        "We enter the dropping state and perform initial packet marking"
                                        "So there should be only 1 more packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 1, "There should be 1 target exceeded marked packet");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 0, "There should not be any CE threshold exceeded marked packet");
                }
            }
          else if (initialTargetMarkCount > 0)
            { // In dropping state
              if (currentTime.GetMicroSeconds () < initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                        "Sojourn is still above target."
                                        "However, it's not time for next target exceeded mark."
                                        "So there should be only 1 more packet dequeued");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 1, "There should still be only 1 target exceeded marked packet from the last dequeue");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 0, "There should not be any CE threshold exceeded marked packet");
                }
              else if (currentTime.GetMicroSeconds () >= initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                        "It's time for packet to be marked"
                                        "So there should be only 1 more packet dequeued");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 2, "There should 2 target exceeded marked packet");
                  NS_TEST_ASSERT_MSG_EQ (nPacketsBeforeFirstDrop, nPacketsBeforeFirstMark, "Number of packets in the queue before drop should be equal"
                                        "to number of packets in the queue before first mark as the behavior untill packet N should be the same.");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 0, "There should not be any CE threshold exceeded marked packet");
                }
            }
        }
      else if (testCase == 3)
        {
          if (initialTargetMarkCount == 0 && currentTime > queue->GetTarget ())
            {
              if (currentTime < queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be 1 packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 0, "We are not in dropping state."
                                        "Sojourn time has just gone above target from below."
                                        "Hence, there should be no target exceeded marked packets");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 1, "Sojourn time has gone above CE threshold."
                                        "Hence, there should be 1 CE threshold exceeded marked packet");
                }
              else if (currentTime >= queue->GetInterval ())
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "Sojourn time has been above target for at least interval."
                                        "We enter the dropping state and perform initial packet marking"
                                        "So there should be only 1 more packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 1, "There should be 1 target exceeded marked packet");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 1, "There should be 1 CE threshold exceeded marked packets");
                }
            }
          else if (initialTargetMarkCount > 0)
            { // In dropping state
              if (currentTime.GetMicroSeconds () < initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "We are in dropping state."
                                        "Sojourn is still above target."
                                        "However, it's not time for next target exceeded mark."
                                        "So there should be only 1 more packet dequeued");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 1, "There should still be only 1 target exceeded marked packet from the last dequeue");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 2, "There should be 2 CE threshold exceeded marked packets");
                }
              else if (currentTime.GetMicroSeconds () >= initialDropNext)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentTargetMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::TARGET_EXCEEDED_MARK);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - (m_dropNextCount + 1) * modeSize, "We are in dropping state."
                                        "It's time for packet to be dropped as packets are not ecnCapable"
                                        "The number of packets dequeued equals to the number of times m_dropNext is updated plus initial dequeue");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, m_dropNextCount, "The number of drops equals to the number of times m_dropNext is updated");
                  NS_TEST_ASSERT_MSG_EQ (currentTargetMarkCount, 1, "There should still be only 1 target exceeded marked packet");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 2, "There should still be 2 CE threshold exceeded marked packet as packets are not ecnCapable");
                }
            }
        }
      else if (testCase == 4)
        {
          if (currentTime < queue->GetTarget ())
            {
              if (initialCeThreshMarkCount == 0 && currentTime < MilliSeconds (2))
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be 1 packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 0, "Sojourn time has not gone above CE threshold."
                                        "Hence, there should not be any CE threshold exceeded marked packet");
                }
              else
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be only 1 more packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 1, "Sojourn time has gone above CE threshold."
                                        "There should be 1 CE threshold exceeded marked packet");
                }
            }
          else if (initialCeThreshMarkCount > 0 && currentTime < queue->GetInterval ())
            { 
              if (initialCeThreshMarkCount < 2)
                {
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be only 1 more packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 2, "There should be 2 CE threshold exceeded marked packets");
                }
              else
                { // In dropping state
                  currentDropCount = queue->GetStats ().GetNDroppedPackets (CoDelQueueDisc::TARGET_EXCEEDED_DROP);
                  currentCeThreshMarkCount = queue->GetStats ().GetNMarkedPackets (CoDelQueueDisc::CE_THRESHOLD_EXCEEDED_MARK);
                  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), initialQSize - modeSize, "There should be only 1 more packet dequeued.");
                  NS_TEST_ASSERT_MSG_EQ (currentDropCount, 0, "There should not be any packet drops");
                  NS_TEST_ASSERT_MSG_EQ (currentCeThreshMarkCount, 3, "There should be 3 CE threshold exceeded marked packet");
                }
            }
        }
    }
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief CoDel Queue Disc Test Suite
 */
static class CoDelQueueDiscTestSuite : public TestSuite
{
public:
  CoDelQueueDiscTestSuite ()
    : TestSuite ("codel-queue-disc", UNIT)
  {
    // Test 1: simple enqueue/dequeue with no drops
    AddTestCase (new CoDelQueueDiscBasicEnqueueDequeue (QueueSizeUnit::PACKETS), TestCase::QUICK);
    AddTestCase (new CoDelQueueDiscBasicEnqueueDequeue (QueueSizeUnit::BYTES), TestCase::QUICK);
    // Test 2: enqueue with drops due to queue overflow
    AddTestCase (new CoDelQueueDiscBasicOverflow (QueueSizeUnit::PACKETS), TestCase::QUICK);
    AddTestCase (new CoDelQueueDiscBasicOverflow (QueueSizeUnit::BYTES), TestCase::QUICK);
    // Test 3: test NewtonStep() against explicit port of Linux implementation
    AddTestCase (new CoDelQueueDiscNewtonStepTest (), TestCase::QUICK);
    // Test 4: test ControlLaw() against explicit port of Linux implementation
    AddTestCase (new CoDelQueueDiscControlLawTest (), TestCase::QUICK);
    // Test 5: enqueue/dequeue with drops according to CoDel algorithm
    AddTestCase (new CoDelQueueDiscBasicDrop (QueueSizeUnit::PACKETS), TestCase::QUICK);
    AddTestCase (new CoDelQueueDiscBasicDrop (QueueSizeUnit::BYTES), TestCase::QUICK);
    // Test 6: enqueue/dequeue with marks according to CoDel algorithm
    AddTestCase (new CoDelQueueDiscBasicMark (QueueSizeUnit::PACKETS), TestCase::QUICK);
    AddTestCase (new CoDelQueueDiscBasicMark (QueueSizeUnit::BYTES), TestCase::QUICK);
  }
} g_coDelQueueTestSuite; ///< the test suite
