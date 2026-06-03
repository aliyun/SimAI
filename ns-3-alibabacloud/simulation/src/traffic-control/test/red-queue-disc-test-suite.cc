/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright © 2011 Marcos Talau
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
 * Author: Marcos Talau (talau@users.sourceforge.net)
 * Modified by:   Pasquale Imputato <p.imputato@gmail.com>
 *
 */

#include "ns3/test.h"
#include "ns3/red-queue-disc.h"
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
 * \brief Red Queue Disc Test Item
 */
class RedQueueDiscTestItem : public QueueDiscItem {
public:
  /**
   * Constructor
   *
   * \param p packet
   * \param addr address
   * \param ecnCapable ECN capable flag
   */
  RedQueueDiscTestItem (Ptr<Packet> p, const Address & addr, bool ecnCapable);
  virtual ~RedQueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  RedQueueDiscTestItem (const RedQueueDiscTestItem &) = delete;
  RedQueueDiscTestItem & operator = (const RedQueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark(void);

private:
  RedQueueDiscTestItem ();

  bool m_ecnCapablePacket; ///< ECN capable packet?
};

RedQueueDiscTestItem::RedQueueDiscTestItem (Ptr<Packet> p, const Address & addr, bool ecnCapable)
  : QueueDiscItem (p, addr, 0),
    m_ecnCapablePacket (ecnCapable)
{
}

RedQueueDiscTestItem::~RedQueueDiscTestItem ()
{
}

void
RedQueueDiscTestItem::AddHeader (void)
{
}

bool
RedQueueDiscTestItem::Mark (void)
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
 * \brief Red Queue Disc Test Case
 */
class RedQueueDiscTestCase : public TestCase
{
public:
  RedQueueDiscTestCase ();
  virtual void DoRun (void);
private:
  /**
   * Enqueue function
   * \param queue the queue disc
   * \param size the size
   * \param nPkt the number of packets
   * \param ecnCapable ECN capable flag
   */
  void Enqueue (Ptr<RedQueueDisc> queue, uint32_t size, uint32_t nPkt, bool ecnCapable);
  /**
   * Run RED test function
   * \param mode the mode
   */
  void RunRedTest (QueueSizeUnit mode);
};

RedQueueDiscTestCase::RedQueueDiscTestCase ()
  : TestCase ("Sanity check on the red queue implementation")
{
}

void
RedQueueDiscTestCase::RunRedTest (QueueSizeUnit mode)
{
  uint32_t pktSize = 0;
  // 1 for packets; pktSize for bytes
  uint32_t modeSize = 1;
  double minTh = 2;
  double maxTh = 5;
  uint32_t qSize = 8;
  Ptr<RedQueueDisc> queue = CreateObject<RedQueueDisc> ();

  // test 1: simple enqueue/dequeue with no drops
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");

  Address dest;
  
  if (mode == QueueSizeUnit::BYTES)
    {
      // pktSize should be same as MeanPktSize to avoid performance gap between byte and packet mode
      pktSize = 500;
      modeSize = pktSize;
      queue->SetTh (minTh * modeSize, maxTh * modeSize);
      queue->SetMaxSize (QueueSize (mode, qSize * modeSize));
    }

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
  queue->Enqueue (Create<RedQueueDiscTestItem> (p1, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 1 * modeSize, "There should be one packet in there");
  queue->Enqueue (Create<RedQueueDiscTestItem> (p2, dest, false));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 2 * modeSize, "There should be two packets in there");
  queue->Enqueue (Create<RedQueueDiscTestItem> (p3, dest, false));
  queue->Enqueue (Create<RedQueueDiscTestItem> (p4, dest, false));
  queue->Enqueue (Create<RedQueueDiscTestItem> (p5, dest, false));
  queue->Enqueue (Create<RedQueueDiscTestItem> (p6, dest, false));
  queue->Enqueue (Create<RedQueueDiscTestItem> (p7, dest, false));
  queue->Enqueue (Create<RedQueueDiscTestItem> (p8, dest, false));
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


  // test 2: more data, but with no drops
  queue = CreateObject<RedQueueDisc> ();
  minTh = 70 * modeSize;
  maxTh = 150 * modeSize;
  qSize = 300 * modeSize;
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  QueueDisc::Stats st = queue->GetStats ();
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP), 0,
                         "There should be zero unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (RedQueueDisc::FORCED_DROP), 0,
                         "There should be zero forced dropps");
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (QueueDisc::INTERNAL_QUEUE_DROP), 0,
                         "There should be zero drops due to queue limit");

  // save number of drops from tests
  struct d {
    uint32_t test3;
    uint32_t test4;
    uint32_t test5;
    uint32_t test6;
    uint32_t test7;
    uint32_t test11;
    uint32_t test12;
    uint32_t test13;
  } drop;


  // test 3: more data, now drops due QW change
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.020)), true,
                         "Verify that we can actually set the attribute QW");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test3 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP)
               + st.GetNDroppedPackets (RedQueueDisc::FORCED_DROP)
               + st.GetNDroppedPackets (QueueDisc::INTERNAL_QUEUE_DROP);
  NS_TEST_ASSERT_MSG_NE (drop.test3, 0, "There should be some dropped packets");


  // test 4: reduced maxTh, this causes more drops
  maxTh = 100 * modeSize;
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.020)), true,
                         "Verify that we can actually set the attribute QW");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test4 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP)
               + st.GetNDroppedPackets (RedQueueDisc::FORCED_DROP)
               + st.GetNDroppedPackets (QueueDisc::INTERNAL_QUEUE_DROP);
  NS_TEST_ASSERT_MSG_GT (drop.test4, drop.test3, "Test 4 should have more drops than test 3");


  // test 5: change drop probability to a high value (LInterm)
  maxTh = 150 * modeSize;
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.020)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (5)), true,
                         "Verify that we can actually set the attribute LInterm");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test5 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP)
               + st.GetNDroppedPackets (RedQueueDisc::FORCED_DROP)
               + st.GetNDroppedPackets (QueueDisc::INTERNAL_QUEUE_DROP);
  NS_TEST_ASSERT_MSG_GT (drop.test5, drop.test3, "Test 5 should have more drops than test 3");


  // test 6: disable Gentle param
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.020)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (false)), true,
                         "Verify that we can actually set the attribute Gentle");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test6 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP)
               + st.GetNDroppedPackets (RedQueueDisc::FORCED_DROP)
               + st.GetNDroppedPackets (QueueDisc::INTERNAL_QUEUE_DROP);
  NS_TEST_ASSERT_MSG_GT (drop.test6, drop.test3, "Test 6 should have more drops than test 3");


  // test 7: disable Wait param
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.020)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Wait", BooleanValue (false)), true,
                         "Verify that we can actually set the attribute Wait");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test7 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP)
               + st.GetNDroppedPackets (RedQueueDisc::FORCED_DROP)
               + st.GetNDroppedPackets (QueueDisc::INTERNAL_QUEUE_DROP);
  NS_TEST_ASSERT_MSG_GT (drop.test7, drop.test3, "Test 7 should have more drops than test 3");


  // test 8: RED queue disc is ECN enabled, but packets are not ECN capable
  queue = CreateObject<RedQueueDisc> ();
  minTh = 30 * modeSize;
  maxTh = 90 * modeSize;
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (2)), true,
                         "Verify that we can actually set the attribute LInterm");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute Gentle");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseECN");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  // Packets are not ECN capable, so there should be only unforced drops, no unforced marks
  NS_TEST_ASSERT_MSG_NE (st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP), 0,
                         "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNMarkedPackets (RedQueueDisc::UNFORCED_MARK), 0,
                         "There should be no unforced marks");


  // test 9: Packets are ECN capable, but RED queue disc is not ECN enabled
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (2)), true,
                         "Verify that we can actually set the attribute LInterm");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute Gentle");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (false)), true,
                         "Verify that we can actually set the attribute UseECN");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, true);
  st = queue->GetStats ();
  // RED queue disc is not ECN enabled, so there should be only unforced drops, no unforced marks
  NS_TEST_ASSERT_MSG_NE (st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP), 0,
                         "There should be some unforced drops");
  NS_TEST_ASSERT_MSG_EQ (st.GetNMarkedPackets (RedQueueDisc::UNFORCED_MARK), 0,
                         "There should be no unforced marks");


  // test 10: Packets are ECN capable and RED queue disc is ECN enabled
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (2)), true,
                         "Verify that we can actually set the attribute LInterm");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute Gentle");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("UseEcn", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute UseECN");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, true);
  st = queue->GetStats ();
  // Packets are ECN capable, RED queue disc is ECN enabled; there should be only unforced marks, no unforced drops
  NS_TEST_ASSERT_MSG_EQ (st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP), 0,
                         "There should be no unforced drops");
  NS_TEST_ASSERT_MSG_NE (st.GetNMarkedPackets (RedQueueDisc::UNFORCED_MARK), 0,
                         "There should be some unforced marks");


  // test 11: RED with default parameter settings, linear drop probability and fixed m_curMaxP
  queue = CreateObject<RedQueueDisc> ();
  minTh = 30 * modeSize;
  maxTh = 90 * modeSize;
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (2)), true,
                         "Verify that we can actually set the attribute LInterm");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute Gentle");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test11 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_NE (drop.test11, 0, "There should some dropped packets due to probability mark");


  // test 12: Feng's Adaptive RED with default parameter settings and varying m_curMaxP
  queue = CreateObject<RedQueueDisc> ();
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (2)), true,
                         "Verify that we can actually set the attribute LInterm");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute Gentle");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("FengAdaptive", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute FengAdaptive");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test12 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_LT (drop.test12, drop.test11, "Test 12 should have less drops due to probability mark than test 11");


  // test 13: RED with Nonlinear drop probability
  queue = CreateObject<RedQueueDisc> ();
  minTh = 30 * modeSize;
  maxTh = 90 * modeSize;
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MinTh", DoubleValue (minTh)), true,
                         "Verify that we can actually set the attribute MinTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxTh", DoubleValue (maxTh)), true,
                         "Verify that we can actually set the attribute MaxTh");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("QW", DoubleValue (0.002)), true,
                         "Verify that we can actually set the attribute QW");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("LInterm", DoubleValue (2)), true,
                         "Verify that we can actually set the attribute LInterm");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Gentle", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute Gentle");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("NLRED", BooleanValue (true)), true,
                         "Verify that we can actually set the attribute NLRED");
  queue->Initialize ();
  Enqueue (queue, pktSize, 300, false);
  st = queue->GetStats ();
  drop.test13 = st.GetNDroppedPackets (RedQueueDisc::UNFORCED_DROP);
  NS_TEST_ASSERT_MSG_LT (drop.test13, drop.test11, "Test 13 should have less drops due to probability mark than test 11");

}

void 
RedQueueDiscTestCase::Enqueue (Ptr<RedQueueDisc> queue, uint32_t size, uint32_t nPkt, bool ecnCapable)
{
  Address dest;
  for (uint32_t i = 0; i < nPkt; i++)
    {
      queue->Enqueue (Create<RedQueueDiscTestItem> (Create<Packet> (size), dest, ecnCapable));
    }
}

void
RedQueueDiscTestCase::DoRun (void)
{
  RunRedTest (QueueSizeUnit::PACKETS);
  RunRedTest (QueueSizeUnit::BYTES);
  Simulator::Destroy ();

}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Red Queue Disc Test Suite
 */
static class RedQueueDiscTestSuite : public TestSuite
{
public:
  RedQueueDiscTestSuite ()
    : TestSuite ("red-queue-disc", UNIT)
  {
    AddTestCase (new RedQueueDiscTestCase (), TestCase::QUICK);
  }
} g_redQueueTestSuite; ///< the test suite
