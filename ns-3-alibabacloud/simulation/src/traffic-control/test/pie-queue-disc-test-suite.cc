/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 NITK Surathkal
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
 * Authors: Shravya Ks <shravya.ks0@gmail.com>
 *          Smriti Murali <m.smriti.95@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/test.h"
#include "ns3/pie-queue-disc.h"
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
 * \brief Pie Queue Disc Test Item
 */
class PieQueueDiscTestItem : public QueueDiscItem
{
public:
  /**
   * Constructor
   *
   * \param p the packet
   * \param addr the address
   * \param ecnCapable ECN capable flag
   */
  PieQueueDiscTestItem (Ptr<Packet> p, const Address & addr, bool ecnCapable);
  virtual ~PieQueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  PieQueueDiscTestItem (const PieQueueDiscTestItem &) = delete;
  PieQueueDiscTestItem & operator = (const PieQueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark (void);

  // ** Variables for testing m_isCapDropAdjustment
  double m_maxDropProbDiff = 0.0;   //!< Maximum difference between two consecutive drop probability values
  double m_prevDropProb = 0.0;      //!< Previous drop probability
  bool m_checkProb = false;         //!< Enable/Disable drop probability checks

  // ** Variable for testing ECN
  double m_maxDropProb = 0.0;       //!< Maximum value of drop probability
  bool m_ecnCapable = false;        //!< Enable/Disable ECN capability

  // ** Variables for testing Derandomization
  bool m_checkAccuProb = false;     //!< Enable/Disable accumulated drop probability checks
  bool m_constAccuProb = false;     //!< Enable/Disable fixed accumulated drop probability
  bool m_checkMaxAccuProb = false;  //!< Enable/Disable Maximum accumulated drop probability checks
  double m_accuProbError = 0.0;     //!< Error in accumulated drop probability
  double m_prevAccuProb = 0.0;      //!< Previous accumulated drop probability
  double m_setAccuProb = 0.0;       //!< Value to be set for accumulated drop probability
  uint32_t m_expectedDrops = 0;     //!< Number of expected unforced drops

private:
  PieQueueDiscTestItem ();

  bool m_ecnCapablePacket; //!< ECN capable packet?
};

PieQueueDiscTestItem::PieQueueDiscTestItem (Ptr<Packet> p, const Address & addr, bool ecnCapable)
  : QueueDiscItem (p, addr, 0), m_ecnCapablePacket (ecnCapable)
{
}

PieQueueDiscTestItem::~PieQueueDiscTestItem ()
{
}

void
PieQueueDiscTestItem::AddHeader (void)
{
}

bool
PieQueueDiscTestItem::Mark (void)
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
 * \brief Pie Queue Disc Test Case
 */
class PieQueueDiscTestCase : public TestCase
{
public:
  PieQueueDiscTestCase ();
  virtual void DoRun (void);
private:
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   * \param testAttributes attributes for testing
   */
  void Enqueue (Ptr<PieQueueDisc> queue, uint32_t size, uint32_t nPkt, Ptr<PieQueueDiscTestItem> testAttributes);
  /**
   * Enqueue with delay function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   * \param testAttributes attributes for testing
   */
  void EnqueueWithDelay (Ptr<PieQueueDisc> queue, uint32_t size, uint32_t nPkt, Ptr<PieQueueDiscTestItem> testAttributes);
  /**
   * Dequeue function
   * \param queue the queue disc
   * \param nPkt the number of packets
   */
  void Dequeue (Ptr<PieQueueDisc> queue, uint32_t nPkt);
  /**
   * Dequeue with delay function
   * \param queue the queue disc
   * \param delay the delay
   * \param nPkt the number of packets
   */
  void DequeueWithDelay (Ptr<PieQueueDisc> queue, double delay, uint32_t nPkt);
  /**
   * Run test function
   * \param mode the test mode
   */
  void RunPieTest (QueueSizeUnit mode);
  /**
   * \brief Check Drop Probability
   * \param queue the queue disc
   * \param testAttributes attributes for testing
   */
  void CheckDropProb (Ptr<PieQueueDisc> queue, Ptr<PieQueueDiscTestItem> testAttributes);
  /**
   * \brief Check Accumulated Drop Probability
   * \param queue the queue disc
   * \param testAttributes attributes for testing
   */
  void CheckAccuProb (Ptr<PieQueueDisc> queue, Ptr<PieQueueDiscTestItem> testAttributes);
  /**
   * \brief Check Maximum Accumulated Drop Probability
   * \param queue the queue disc
   * \param testAttributes attributes for testing
   */
  void CheckMaxAccuProb (Ptr<PieQueueDisc> queue, Ptr<PieQueueDiscTestItem> testAttributes);
};

PieQueueDiscTestCase::PieQueueDiscTestCase ()
  : TestCase ("Sanity check on the pie queue disc implementation")
{
}

void
PieQueueDiscTestCase::RunPieTest (QueueSizeUnit mode)
{
  uint32_t pktSize = 0;

  // 1 for packets; pktSize for bytes
  uint32_t modeSize = 1;

  uint32_t qSize = 300;
  Ptr<PieQueueDisc> queue = CreateObject<PieQueueDisc> ();


  // test 1: simple enqueue/dequeue with defaults, no drops
  Address dest;
  // PieQueueDiscItem pointer for attributes
  Ptr<PieQueueDiscTestItem> testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);

  if (mode == QueueSizeUnit::BYTES)
    {
      // pktSize should be same as MeanPktSize to avoid performance gap between byte and packet mode
      pktSize = 1000;
      modeSize = pktSize;
      qSize = qSize * modeSize;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");

  Ptr<Packet> p1, p2, p3, p4, p5, p6, p7, p8;
  p1 = Create<Packet> (pktSize);
  p2 = Create<Packet> (pktSize);
  p3 = Create<Packet> (pktSize);
  p4 = Create<Packet> (pktSize);
  p5 = Create<Packet> (pktSize);
  p6 = Create<Packet> (pktSize);
  p7 = Create<Packet> (pktSize);
  p8 = Create<Packet> (pktSize);

  queue->Initialize ();
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 0 * modeSize, "There should be no packets in there");
  queue->Enqueue (Create<PieQueueDiscTestItem> (p1, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 1 * modeSize, "There should be one packet in there");
  queue->Enqueue (Create<PieQueueDiscTestItem> (p2, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 2 * modeSize, "There should be two packets in there");
  queue->Enqueue (Create<PieQueueDiscTestItem> (p3, dest, false));
  queue->Enqueue (Create<PieQueueDiscTestItem> (p4, dest, false));
  queue->Enqueue (Create<PieQueueDiscTestItem> (p5, dest, false));
  queue->Enqueue (Create<PieQueueDiscTestItem> (p6, dest, false));
  queue->Enqueue (Create<PieQueueDiscTestItem> (p7, dest, false));
  queue->Enqueue (Create<PieQueueDiscTestItem> (p8, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 8 * modeSize, "There should be eight packets in there");

  Ptr<QueueDiscItem> item;

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the first packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 7 * modeSize, "There should be seven packets in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p1->GetUid (), "was this the first packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the second packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 6 * modeSize, "There should be six packet in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p2->GetUid (), "Was this the second packet ?");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the third packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 5 * modeSize, "There should be five packets in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p3->GetUid (), "Was this the third packet ?");

  item = queue->Dequeue ();
  item = queue->Dequeue ();
  item = queue->Dequeue ();
  item = queue->Dequeue ();
  item = queue->Dequeue ();

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item == 0), true, "There are really no packets in there");


  // test 2: more data with defaults, unforced drops but no forced drops
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  pktSize = 1000;  // pktSize != 0 because DequeueThreshold always works in bytes
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Tupdate", TimeValue (Seconds (0.03))), true,
                         "Verify that we can actually set the attribute Tupdate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("DequeueThreshold", UintegerValue (10000)), true,
                         "Verify that we can actually set the attribute DequeueThreshold");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QueueDelayReference", TimeValue (Seconds (0.02))), true,
                         "Verify that we can actually set the attribute QueueDelayReference");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxBurstAllowance", TimeValue (Seconds (0.1))), true,
                         "Verify that we can actually set the attribute MaxBurstAllowance");
  queue->Initialize ();
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.012, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  QueueDisc::Stats st = queue->GetStats ();
  uint32_t test2 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (test2, 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 3: same as test 2, but with higher QueueDelayReference
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Tupdate", TimeValue (Seconds (0.03))), true,
                         "Verify that we can actually set the attribute Tupdate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("DequeueThreshold", UintegerValue (10000)), true,
                         "Verify that we can actually set the attribute DequeueThreshold");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QueueDelayReference", TimeValue (Seconds (0.08))), true,
                         "Verify that we can actually set the attribute QueueDelayReference");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxBurstAllowance", TimeValue (Seconds (0.1))), true,
                         "Verify that we can actually set the attribute MaxBurstAllowance");
  queue->Initialize ();
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.012, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test3 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_LT (test3, test2, "Test 3 should have less unforced drops than test 2");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 4: same as test 2, but with reduced dequeue rate
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Tupdate", TimeValue (Seconds (0.03))), true,
                         "Verify that we can actually set the attribute Tupdate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("DequeueThreshold", UintegerValue (10000)), true,
                         "Verify that we can actually set the attribute DequeueThreshold");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QueueDelayReference", TimeValue (Seconds (0.02))), true,
                         "Verify that we can actually set the attribute QueueDelayReference");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxBurstAllowance", TimeValue (Seconds (0.1))), true,
                         "Verify that we can actually set the attribute MaxBurstAllowance");
  queue->Initialize ();
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.015, 400); // delay between two successive dequeue events is increased
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test4 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_GT (test4, test2, "Test 4 should have more unforced drops than test 2");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 5: same dequeue rate as test 4, but with higher Tupdate
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Tupdate", TimeValue (Seconds (0.09))), true,
                         "Verify that we can actually set the attribute Tupdate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("DequeueThreshold", UintegerValue (10000)), true,
                         "Verify that we can actually set the attribute DequeueThreshold");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QueueDelayReference", TimeValue (Seconds (0.02))), true,
                         "Verify that we can actually set the attribute QueueDelayReference");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxBurstAllowance", TimeValue (Seconds (0.1))), true,
                         "Verify that we can actually set the attribute MaxBurstAllowance");
  queue->Initialize ();
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.015, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test5 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_LT (test5, test4, "Test 5 should have less unforced drops than test 4");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 6: same as test 2, but with UseDequeueRateEstimator enabled
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseDequeueRateEstimator", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseTimestamp");
  queue->Initialize ();
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test6 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (test6, 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 7: test with CapDropAdjustment disabled
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseCapDropAdjustment", BooleanValue (false)), true,
                         "Verify that we can actually set the attribute UseCapDropAdjustment");
  queue->Initialize ();
  testAttributes->m_checkProb = true;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test7 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (test7, 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");
  NS_TEST_ASSERT_MSG_GT (testAttributes->m_maxDropProbDiff, 0.02,
                              "Maximum increase in drop probability should be greater than 0.02");


  // test 8: test with CapDropAdjustment enabled
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseCapDropAdjustment", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseCapDropAdjustment");
  queue->Initialize ();
  testAttributes->m_checkProb = true;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test8 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (test8, 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");
  NS_TEST_ASSERT_MSG_LT (testAttributes->m_maxDropProbDiff, 0.0200000000000001,
                              "Maximum increase in drop probability should be less than or equal to 0.02");

  
  // test 9: PIE queue disc is ECN enabled, but packets are not ECN capable
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MarkEcnThreshold", DoubleValue (0.3)), true,
                        "Verify that we can actually set the attribute MarkEcnThreshold");
  queue->Initialize ();
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test9 = st.GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK);
  NS_TEST_ASSERT_MSG_EQ (test9, 0, "There should be zero unforced marks");
  NS_TEST_ASSERT_MSG_NE (st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP), 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 10: Packets are ECN capable, but PIE queue disc is not ECN enabled
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (false)), true,
                         "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MarkEcnThreshold", DoubleValue (0.3)), true,
                        "Verify that we can actually set the attribute MarkEcnThreshold");
  queue->Initialize ();
  testAttributes->m_ecnCapable = true;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test10 = st.GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK);
  NS_TEST_ASSERT_MSG_EQ (test10, 0, "There should be zero unforced marks");
  NS_TEST_ASSERT_MSG_NE (st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP), 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 11: Packets and PIE queue disc both are ECN capable
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseEcn");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MarkEcnThreshold", DoubleValue (0.3)), true,
                        "Verify that we can actually set the attribute MarkEcnThreshold");
  queue->Initialize ();
  testAttributes->m_ecnCapable = true;
  testAttributes->m_checkProb = true;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test11 = st.GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK);
  NS_TEST_ASSERT_MSG_NE (test11, 0, "There should be some unforced marks");
  // There are unforced drops because the value of m_maxDropProb goes beyond 0.3 in this test.
  // PIE drops the packets even when they are ECN capable if drop probability is more than 30%.
  NS_TEST_ASSERT_MSG_NE (st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP), 0, "There should be some unforced drops");
  // Confirm that m_maxDropProb goes above 0.3 in this test
  NS_TEST_ASSERT_MSG_GT (testAttributes->m_maxDropProb, 0.3, "Maximum Drop probability should be greater than 0.3");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 12: test with derandomization enabled
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseDerandomization", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseDerandomization");
  queue->Initialize ();
  testAttributes->m_checkAccuProb = true;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test12 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (test12, 0, "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");
  NS_TEST_ASSERT_MSG_EQ (testAttributes->m_accuProbError, 0.0, "There should not be any error in setting accuProb");


  // test 13: same as test 11 but with accumulated drop probability set below the low threshold
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseDerandomization", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseDerandomization");
  queue->Initialize ();
  testAttributes->m_constAccuProb = true;
  // Final value of accumulated drop probability to drop packet will be maximum 0.84 while threshold to drop packet is 0.85
  testAttributes->m_setAccuProb = -0.16;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test13 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_EQ (test13, 0, "There should be zero unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 14: same as test 12 but with accumulated drop probability set above the high threshold
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxBurstAllowance", TimeValue (Seconds (0.0))), true,
                         "Verify that we can actually set the attribute MaxBurstAllowance");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseDerandomization", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseDerandomization");
  queue->Initialize ();
  testAttributes->m_constAccuProb = true;
  testAttributes->m_checkMaxAccuProb = true;
  // Final value of accumulated drop probability to drop packet will be minimum 8.6 while threshold to drop packet is 8.5
  testAttributes->m_setAccuProb = 8.6;
  EnqueueWithDelay (queue, pktSize, 400, testAttributes);
  DequeueWithDelay (queue, 0.014, 400);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test14 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_EQ (test14, testAttributes->m_expectedDrops,
                        "The number of unforced drops should be equal to number of expected unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (PieQueueDisc::FORCED_DROP), 0, "There should be zero forced drops");


  // test 15: tests Active/Inactive feature, ActiveThreshold set to a high value so PIE never starts and there should
  // not be any drops
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize)));
  queue->SetAttributeFailSafe ("ActiveThreshold", TimeValue (Seconds (1)));
  queue->Initialize ();
  
  EnqueueWithDelay (queue, pktSize, 100, testAttributes);
  DequeueWithDelay (queue, 0.02, 100);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test15 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_EQ (test15, 0, "There should not be any drops.");
  NS_TEST_ASSERT_MSG_EQ (st.GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK), 0, "There should be zero marks");


  // test 16: tests Active/Inactive feature, ActiveThreshold set to a low value so PIE starts early
  // and some packets should be dropped.
  queue = CreateObject<PieQueueDisc> ();
  // PieQueueDiscItem pointer for attributes
  testAttributes = Create<PieQueueDiscTestItem> (Create<Packet> (pktSize), dest, false);
  queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize)));
  queue->SetAttributeFailSafe ("ActiveThreshold", TimeValue (Seconds (0.001)));
  queue->Initialize ();
  
  EnqueueWithDelay (queue, pktSize, 100, testAttributes);
  DequeueWithDelay (queue, 0.02, 100);
  Simulator::Stop (Seconds (8.0));
  Simulator::Run ();
  st = queue->GetStats ();
  uint32_t test16 = st.GetNDroppedPackets (PieQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (test16, 0, "There should be some drops.");
  NS_TEST_ASSERT_MSG_EQ (st.GetNMarkedPackets (PieQueueDisc::UNFORCED_MARK), 0, "There should be zero marks");
}

void
PieQueueDiscTestCase::Enqueue (Ptr<PieQueueDisc> queue, uint32_t size, uint32_t nPkt, Ptr<PieQueueDiscTestItem> testAttributes)
{
  Address dest;
   for (uint32_t i = 0; i < nPkt; i++)
    {
      if (testAttributes->m_constAccuProb)
        {
          queue->m_accuProb = testAttributes->m_setAccuProb;
          if (testAttributes->m_checkMaxAccuProb)
            {
              CheckMaxAccuProb (queue, testAttributes);
            }
        }
      queue->Enqueue (Create<PieQueueDiscTestItem> (Create<Packet> (size), dest, testAttributes->m_ecnCapable));
      if (testAttributes->m_checkProb)
        {
          CheckDropProb (queue, testAttributes);
        }
      if (testAttributes->m_checkAccuProb)
        {
          CheckAccuProb (queue, testAttributes);
        }
    }
}

void
PieQueueDiscTestCase::CheckDropProb (Ptr<PieQueueDisc> queue, Ptr<PieQueueDiscTestItem> testAttributes)
{
  double dropProb = queue->m_dropProb;
  if (testAttributes->m_maxDropProb < dropProb)
    {
      testAttributes->m_maxDropProb = dropProb;
    }
  if (testAttributes->m_prevDropProb > 0.1)
    {
      double currentDiff = dropProb - testAttributes->m_prevDropProb;
      if (testAttributes->m_maxDropProbDiff < currentDiff)
        {
          testAttributes->m_maxDropProbDiff = currentDiff;
        }
    }
  testAttributes->m_prevDropProb = dropProb;
}

void
PieQueueDiscTestCase::CheckAccuProb (Ptr<PieQueueDisc> queue, Ptr<PieQueueDiscTestItem> testAttributes)
{
  double dropProb = queue->m_dropProb;
  double accuProb = queue->m_accuProb;
  if (accuProb != 0)
    {
      double expectedAccuProb = testAttributes->m_prevAccuProb + dropProb;
      testAttributes->m_accuProbError = accuProb - expectedAccuProb;
    }
  testAttributes->m_prevAccuProb = accuProb;
}

void
PieQueueDiscTestCase::CheckMaxAccuProb (Ptr<PieQueueDisc> queue, Ptr<PieQueueDiscTestItem> testAttributes)
{
  queue->m_dropProb = 0.001;
  QueueSize queueSize = queue->GetCurrentSize ();
  if ((queueSize.GetUnit () == QueueSizeUnit::PACKETS && queueSize.GetValue () > 2) || (queueSize.GetUnit () == QueueSizeUnit::BYTES && queueSize.GetValue () > 2000))
    {
      testAttributes->m_expectedDrops = testAttributes->m_expectedDrops + 1;
    }
}

void
PieQueueDiscTestCase::EnqueueWithDelay (Ptr<PieQueueDisc> queue, uint32_t size, uint32_t nPkt, Ptr<PieQueueDiscTestItem> testAttributes)
{
  Address dest;
  double delay = 0.01;  // enqueue packets with delay
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &PieQueueDiscTestCase::Enqueue, this, queue, size, 1, testAttributes);
    }
}

void
PieQueueDiscTestCase::Dequeue (Ptr<PieQueueDisc> queue, uint32_t nPkt)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Ptr<QueueDiscItem> item = queue->Dequeue ();
    }
}

void
PieQueueDiscTestCase::DequeueWithDelay (Ptr<PieQueueDisc> queue, double delay, uint32_t nPkt)
{
  for (uint32_t i = 0; i < nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &PieQueueDiscTestCase::Dequeue, this, queue, 1);
    }
}

void
PieQueueDiscTestCase::DoRun (void)
{
  RunPieTest (QueueSizeUnit::PACKETS);
  RunPieTest (QueueSizeUnit::BYTES);
  Simulator::Destroy ();
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Pie Queue Disc Test Suite
 */
static class PieQueueDiscTestSuite : public TestSuite
{
public:
  PieQueueDiscTestSuite ()
    : TestSuite ("pie-queue-disc", UNIT)
  {
    AddTestCase (new PieQueueDiscTestCase (), TestCase::QUICK);
  }
} g_pieQueueTestSuite; ///< the test suite
