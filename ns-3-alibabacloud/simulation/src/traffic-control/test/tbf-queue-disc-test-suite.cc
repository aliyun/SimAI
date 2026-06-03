/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Kungliga Tekniska Högskolan
 *               2017 Universita' degli Studi di Napoli Federico II
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
 * Authors: Surya Seetharaman <suryaseetharaman.9@gmail.com>
 *          Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/test.h"
#include "ns3/tbf-queue-disc.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/node-container.h"
#include "ns3/simple-net-device.h"
#include "ns3/simple-channel.h"
#include "ns3/traffic-control-layer.h"
#include "ns3/config.h"

using namespace ns3;

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Tbf Queue Disc Test Item
 */
class TbfQueueDiscTestItem : public QueueDiscItem {
public:
  /**
   * Constructor
   *
   * \param p the packet
   * \param addr the address
   */
  TbfQueueDiscTestItem (Ptr<Packet> p, const Address & addr);
  virtual ~TbfQueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  TbfQueueDiscTestItem (const TbfQueueDiscTestItem &) = delete;
  TbfQueueDiscTestItem & operator = (const TbfQueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark (void);

private:
  TbfQueueDiscTestItem ();
};

TbfQueueDiscTestItem::TbfQueueDiscTestItem (Ptr<Packet> p, const Address & addr)
  : QueueDiscItem (p, addr, 0)
{
}

TbfQueueDiscTestItem::~TbfQueueDiscTestItem ()
{
}

void
TbfQueueDiscTestItem::AddHeader (void)
{
}

bool
TbfQueueDiscTestItem::Mark (void)
{
  return false;
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Tbf Queue Disc Test Case
 */
class TbfQueueDiscTestCase : public TestCase
{
public:
  TbfQueueDiscTestCase ();
  virtual void DoRun (void);
private:
  /**
   * Enqueue function
   * \param queue the queue disc into which enqueue needs to be done
   * \param dest the destination address
   * \param size the size of the packet in bytes to be enqueued
   */
  void Enqueue (Ptr<TbfQueueDisc> queue, Address dest, uint32_t size);
  /**
   * DequeueAndCheck function to check if a packet is blocked or not after dequeuing and verify against expected result
   * \param queue the queue disc on which DequeueAndCheck needs to be done
   * \param flag the boolean value against which the return value of dequeue () has to be compared with
   * \param printStatement the string to be printed in the NS_TEST_EXPECT_MSG_EQ
   */
  void DequeueAndCheck (Ptr<TbfQueueDisc> queue, bool flag, std::string printStatement);
  /**
   * Run TBF test function
   * \param mode the mode
   */
  void RunTbfTest (QueueSizeUnit mode);
};

TbfQueueDiscTestCase::TbfQueueDiscTestCase ()
  : TestCase ("Sanity check on the TBF queue implementation")
{
}

void
TbfQueueDiscTestCase::RunTbfTest (QueueSizeUnit mode)
{
  uint32_t pktSize = 1500;
  // 1 for packets; pktSize for bytes
  uint32_t modeSize = 1;
  uint32_t qSize = 4;
  uint32_t burst = 6000;
  uint32_t mtu = 0;
  DataRate rate = DataRate ("6KB/s");
  DataRate peakRate = DataRate ("0KB/s");

  Ptr<TbfQueueDisc> queue = CreateObject<TbfQueueDisc> ();

  // test 1: Simple Enqueue/Dequeue with verification of attribute setting
  /* 1. There is no second bucket since "peakRate" is set to 0.
     2. A simple enqueue of five packets, each containing 1500B is followed by
        the dequeue those five packets.
     3. The subtraction of tokens from the first bucket to send out each of the
        five packets is monitored and verified.
     Note : The number of tokens in the first bucket is full at the beginning.
            With the dequeuing of each packet, the number of tokens keeps decreasing.
            So packets are dequeued as long as there are enough tokens in the bucket. */

  if (mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
      qSize = qSize * modeSize;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Burst", UintegerValue (burst)), true,
                         "Verify that we can actually set the attribute Burst");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Mtu", UintegerValue (mtu)), true,
                         "Verify that we can actually set the attribute Mtu");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Rate", DataRateValue (rate)), true,
                         "Verify that we can actually set the attribute Rate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("PeakRate", DataRateValue (peakRate)), true,
                         "Verify that we can actually set the attribute PeakRate");

  Address dest;

  Ptr<Packet> p1, p2, p3, p4, p5;
  p1 = Create<Packet> (pktSize);
  p2 = Create<Packet> (pktSize);
  p3 = Create<Packet> (pktSize);
  p4 = Create<Packet> (pktSize);
  p5 = Create<Packet> (pktSize);

  queue->Initialize ();
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 0 * modeSize, "There should be no packets in there");
  queue->Enqueue (Create<TbfQueueDiscTestItem> (p1, dest));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 1 * modeSize, "There should be one packet in there");
  queue->Enqueue (Create<TbfQueueDiscTestItem> (p2, dest));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 2 * modeSize, "There should be two packets in there");
  queue->Enqueue (Create<TbfQueueDiscTestItem> (p3, dest));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 3 * modeSize, "There should be three packets in there");
  queue->Enqueue (Create<TbfQueueDiscTestItem> (p4, dest));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 4 * modeSize, "There should be four packets in there");
  queue->Enqueue (Create<TbfQueueDiscTestItem> (p5, dest));
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 4 * modeSize,
                         "There should still be four packets in there as this enqueue cannot happen since QueueLimit will be exceeded");

  Ptr<QueueDiscItem> item;
  NS_TEST_ASSERT_MSG_EQ (queue->GetFirstBucketTokens (), burst, "The first token bucket should be full");
  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the first packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 3 * modeSize, "There should be three packets in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p1->GetUid (), "was this the first packet ?");
  NS_TEST_ASSERT_MSG_EQ (queue->GetFirstBucketTokens (), burst - (1 * pktSize),
                         "The number of tokens in the first bucket should be one pktSize lesser");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the second packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 2 * modeSize, "There should be two packets in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p2->GetUid (), "Was this the second packet ?");
  NS_TEST_ASSERT_MSG_EQ (queue->GetFirstBucketTokens (), burst - (2 * pktSize),
                         "The number of tokens in the first bucket should be two pktSizes lesser");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the third packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 1 * modeSize, "There should be one packet in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p3->GetUid (), "Was this the third packet ?");
  NS_TEST_ASSERT_MSG_EQ (queue->GetFirstBucketTokens (), burst - (3 * pktSize),
                         "The number of tokens in the first bucket should be three pktSizes lesser");

  item = queue->Dequeue ();
  NS_TEST_ASSERT_MSG_EQ ((item != 0), true, "I want to remove the fourth packet");
  NS_TEST_ASSERT_MSG_EQ (queue->GetCurrentSize ().GetValue (), 0 * modeSize, "There should be zero packet in there");
  NS_TEST_ASSERT_MSG_EQ (item->GetPacket ()->GetUid (), p4->GetUid (), "Was this the fourth packet ?");
  NS_TEST_ASSERT_MSG_EQ (queue->GetFirstBucketTokens (), burst - (4 * pktSize),
                         "The number of tokens in the first bucket should be four pktSizes lesser");

  // test 2 : When DataRate == FirstBucketTokenRate; packets should pass smoothly.
  queue = CreateObject<TbfQueueDisc> ();
  qSize = 10;
  pktSize = 1000;
  burst = 10000;
  mtu = 1000;
  rate = DataRate ("10KB/s");
  peakRate = DataRate ("100KB/s");
  uint32_t nPkt = qSize;

  if (mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
      qSize = qSize * modeSize;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Burst", UintegerValue (burst)), true,
                         "Verify that we can actually set the attribute Burst");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Mtu", UintegerValue (mtu)), true,
                         "Verify that we can actually set the attribute Mtu");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Rate", DataRateValue (rate)), true,
                         "Verify that we can actually set the attribute Rate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("PeakRate", DataRateValue (peakRate)), true,
                         "Verify that we can actually set the attribute PeakRate");

  queue->Initialize ();
  double delay = 0.09;
  for (uint32_t i = 1; i <= nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &TbfQueueDiscTestCase::Enqueue, this, queue, dest, pktSize);
    }
  delay = 0.1;
  for (uint32_t i = 1; i <= nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &TbfQueueDiscTestCase::DequeueAndCheck, this,
                           queue, true, "No packet should be blocked");
    }
  Simulator::Stop (Seconds (1));
  Simulator::Run ();

  // test 3 : When DataRate >>> FirstBucketTokenRate; some packets should get blocked and waking of queue should get scheduled.
  /* 10 packets are enqueued and then dequeued. Since the token rate is less than the data rate, the last packet i.e the 10th
     packet gets blocked and waking of queue is scheduled after a time when enough tokens will be available. At that time the
     10th packet passes through. */
  queue = CreateObject<TbfQueueDisc> ();

  Config::SetDefault ("ns3::QueueDisc::Quota", UintegerValue (1));
  NodeContainer nodesA;
  nodesA.Create (2);
  Ptr<SimpleNetDevice> txDevA = CreateObject<SimpleNetDevice> ();
  nodesA.Get (0)->AddDevice (txDevA);
  Ptr<SimpleNetDevice> rxDevA = CreateObject<SimpleNetDevice> ();
  nodesA.Get (1)->AddDevice (rxDevA);
  Ptr<SimpleChannel> channelA = CreateObject<SimpleChannel> ();
  txDevA->SetChannel (channelA);
  rxDevA->SetChannel (channelA);
  txDevA->SetNode (nodesA.Get (0));
  rxDevA->SetNode (nodesA.Get (1));

  dest = txDevA->GetAddress ();

  Ptr<TrafficControlLayer> tcA = CreateObject<TrafficControlLayer> ();
  nodesA.Get (0)->AggregateObject (tcA);
  tcA->SetRootQueueDiscOnDevice (txDevA, queue);
  tcA->Initialize ();

  burst = 5000;
  mtu = 1000;
  rate = DataRate ("5KB/s");
  peakRate = DataRate ("100KB/s");

  if (mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
      qSize = qSize * modeSize;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Burst", UintegerValue (burst)), true,
                         "Verify that we can actually set the attribute Burst");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Mtu", UintegerValue (mtu)), true,
                         "Verify that we can actually set the attribute Mtu");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Rate", DataRateValue (rate)), true,
                         "Verify that we can actually set the attribute Rate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("PeakRate", DataRateValue (peakRate)), true,
                         "Verify that we can actually set the attribute PeakRate");

  delay = 0.09;
  for (uint32_t i = 1; i <= nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &TbfQueueDiscTestCase::Enqueue, this, queue, dest, pktSize);
    }
  delay = 0.1;
  for (uint32_t i = 1; i <= nPkt; i++)
    {
      if (i == 10)
        {
          Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &TbfQueueDiscTestCase::DequeueAndCheck, this,
                               queue, false, "10th packet should be blocked");
        }
      else
        {
          Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &TbfQueueDiscTestCase::DequeueAndCheck, this,
                               queue, true, "This packet should not be blocked");
        }
    }
  Simulator::Stop (Seconds (1.3));
  Simulator::Run ();

  // test 4 : This test checks the peakRate control of packet dequeue, when DataRate < FirstBucketTokenRate.
  /* 10 packets each of size 1000 bytes are enqueued followed by
     their dequeue. The data rate (25 KB/s) is not sufficiently higher than the btokens rate (15 KB/s), so that in
     the startup phase the first bucket is not empty. Hence when adequate tokens are present in the second (peak) bucket,
     the packets get transmitted, otherwise they are blocked. So basically the transmission of packets falls under the
     regulation of the second bucket since first bucket will always have excess tokens. TBF does not let all
     the packets go smoothly without any control just because there are excess tokens in the first bucket. */
  queue = CreateObject<TbfQueueDisc> ();

  Config::SetDefault ("ns3::QueueDisc::Quota", UintegerValue (1));
  NodeContainer nodesB;
  nodesB.Create (2);
  Ptr<SimpleNetDevice> txDevB = CreateObject<SimpleNetDevice> ();
  nodesB.Get (0)->AddDevice (txDevB);
  Ptr<SimpleNetDevice> rxDevB = CreateObject<SimpleNetDevice> ();
  nodesB.Get (1)->AddDevice (rxDevB);
  Ptr<SimpleChannel> channelB = CreateObject<SimpleChannel> ();
  txDevB->SetChannel (channelB);
  rxDevB->SetChannel (channelB);
  txDevB->SetNode (nodesB.Get (0));
  rxDevB->SetNode (nodesB.Get (1));

  dest = txDevB->GetAddress ();

  Ptr<TrafficControlLayer> tcB = CreateObject<TrafficControlLayer> ();
  nodesB.Get (0)->AggregateObject (tcB);
  tcB->SetRootQueueDiscOnDevice (txDevB, queue);
  tcB->Initialize ();

  burst = 15000;
  mtu = 1000;
  pktSize = 1000;
  rate = DataRate ("15KB/s");
  peakRate = DataRate ("20KB/s");

  if (mode == QueueSizeUnit::BYTES)
    {
      modeSize = pktSize;
      qSize = qSize * modeSize;
    }

  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("MaxSize", QueueSizeValue (QueueSize (mode, qSize))),
                         true, "Verify that we can actually set the attribute MaxSize");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Burst", UintegerValue (burst)), true,
                         "Verify that we can actually set the attribute Burst");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Mtu", UintegerValue (mtu)), true,
                         "Verify that we can actually set the attribute Mtu");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("Rate", DataRateValue (rate)), true,
                         "Verify that we can actually set the attribute Rate");
  NS_TEST_ASSERT_MSG_EQ (queue->SetAttributeFailSafe ("PeakRate", DataRateValue (peakRate)), true,
                         "Verify that we can actually set the attribute PeakRate");

  queue->Initialize ();
  delay = 0.04;
  for (uint32_t i = 1; i <= nPkt; i++)
    {
      Simulator::Schedule (Time (Seconds ((i + 1) * delay)), &TbfQueueDiscTestCase::Enqueue, this, queue, dest, pktSize);
    }

  // The pattern being checked is a pattern of dequeue followed by blocked.  The delay between enqueues is not sufficient
  // to allow ptokens to refill befor the next dequeue.  The first enqueue is at 1.08s in the future, and the attempted
  // dequeue is at 1.10s in the future.  The first dequeue will always succeed.  The second enqueue is 1.12s and attempted
  // dequeue is at 1.14s in the future, but the last dequeue was 0.04s prior; only 800 tokens can be refilled in 0.04s
  // at a peak rate of 20Kbps.  The actual dequeue occurs at 0.01s further into the future when ptokens refills to 1000.
  // To repeat the pattern, odd-numbered dequeue events should be spaced at intervals of at least 100ms, and the
  // even-numbered dequeue events (that block) should be 0.04s (delay) following the last odd-numbered dequeue event.
  double nextDelay = (2 * delay) + 0.02; // 20ms after first enqueue to attempt the first dequeue;
  for (uint32_t i = 1; i <= nPkt; i++)
    {
      if (i % 2 == 1)
        {
          Simulator::Schedule (Seconds (nextDelay), &TbfQueueDiscTestCase::DequeueAndCheck, this,
                               queue, true, "1st packet should not be blocked");
          nextDelay += 0.04;
        }
      else
        {
          Simulator::Schedule (Seconds (nextDelay), &TbfQueueDiscTestCase::DequeueAndCheck, this,
                               queue, false, "This packet should be blocked");
          nextDelay += 0.06;  // Need 0.04 + 0.06 seconds to allow the next packet to be dequeued without block
        }
    }
  Simulator::Stop (Seconds (0.55));
  Simulator::Run ();

}

void
TbfQueueDiscTestCase::Enqueue (Ptr<TbfQueueDisc> queue, Address dest, uint32_t size)
{
  queue->Enqueue (Create<TbfQueueDiscTestItem> (Create<Packet> (size), dest));
}

void
TbfQueueDiscTestCase::DequeueAndCheck (Ptr<TbfQueueDisc> queue, bool flag, std::string printStatement)
{
  Ptr<QueueDiscItem> item = queue->Dequeue ();
  NS_TEST_EXPECT_MSG_EQ ((item != 0), flag, printStatement);
}

void
TbfQueueDiscTestCase::DoRun (void)
{
  RunTbfTest (QueueSizeUnit::PACKETS);
  RunTbfTest (QueueSizeUnit::BYTES);
  Simulator::Destroy ();

}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Tbf Queue Disc Test Suite
 */
static class TbfQueueDiscTestSuite : public TestSuite
{
public:
  TbfQueueDiscTestSuite ()
    : TestSuite ("tbf-queue-disc", UNIT)
  {
    AddTestCase (new TbfQueueDiscTestCase (), TestCase::QUICK);
  }
} g_tbfQueueTestSuite; ///< the test suite
