/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2021 IITP RAS
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
 * Author: Alexander Krotov <krotov@iitp.ru>
 */

#include "ns3/test.h"
#include "ns3/wifi-mac-queue.h"
#include "ns3/simulator.h"

using namespace ns3;

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Test DROP_OLDEST setting.
 *
 * This test verifies the correctness of DROP_OLDEST policy when packets
 * are pushed into the front of the queue. This case is not handled
 * by the underlying ns3::Queue<WifiMacQueueItem>.
 */
class WifiMacQueueDropOldestTest : public TestCase
{
public:
  /**
   * \brief Constructor
   */
  WifiMacQueueDropOldestTest ();

  void DoRun () override;
};

WifiMacQueueDropOldestTest::WifiMacQueueDropOldestTest()
  : TestCase ("Test DROP_OLDEST setting")
{
}

void
WifiMacQueueDropOldestTest::DoRun ()
{
  auto wifiMacQueue = CreateObject<WifiMacQueue> (AC_BE);
  wifiMacQueue->SetMaxSize (QueueSize ("5p"));
  wifiMacQueue->SetAttribute ("DropPolicy", EnumValue (WifiMacQueue::DROP_OLDEST));

  // Initialize the queue with 5 packets.
  std::vector<uint64_t> packetUids;
  for (uint32_t i = 0; i < 5; i++)
    {
      WifiMacHeader header;
      header.SetType (WIFI_MAC_QOSDATA);
      header.SetQosTid (0);
      auto packet = Create<Packet> ();
      auto item = Create<WifiMacQueueItem> (packet, header);
      wifiMacQueue->PushFront (item);

      packetUids.push_back (packet->GetUid ());
    }

  // Check that all elements are inserted successfully.
  auto it = wifiMacQueue->begin ();
  NS_TEST_EXPECT_MSG_EQ (wifiMacQueue->GetNPackets (), 5, "Queue has unexpected number of elements");
  for (uint32_t i = 5; i > 0; i--)
    {
      NS_TEST_EXPECT_MSG_EQ ((*it)->GetPacket ()->GetUid (),
                             packetUids.at (i - 1),
                             "Stored packet is not the expected one");
      it++;
    }

  // Push another element in front of the queue.
  WifiMacHeader header;
  header.SetType (WIFI_MAC_QOSDATA);
  header.SetQosTid (0);
  auto packet = Create<Packet> ();
  auto item = Create<WifiMacQueueItem> (packet, header);
  wifiMacQueue->PushFront (item);

  // Update the vector of expected packet UIDs.
  packetUids.at (4) = packet->GetUid ();

  // Check that front packet was replaced correctly.
  it = wifiMacQueue->begin ();
  NS_TEST_EXPECT_MSG_EQ (wifiMacQueue->GetNPackets (), 5, "Queue has unexpected number of elements");
  for (uint32_t i = 5; i > 0; i--)
    {
      NS_TEST_EXPECT_MSG_EQ ((*it)->GetPacket ()->GetUid (),
                             packetUids.at (i - 1),
                             "Stored packet is not the expected one");
      it++;
    }

  Simulator::Destroy ();
}

/**
 * \ingroup wifi-test
 * \ingroup tests
 *
 * \brief Wifi MAC Queue Test Suite
 */
class WifiMacQueueTestSuite : public TestSuite
{
  public:
    WifiMacQueueTestSuite ();
};

WifiMacQueueTestSuite::WifiMacQueueTestSuite ()
  : TestSuite ("wifi-mac-queue", UNIT)
{
  AddTestCase (new WifiMacQueueDropOldestTest, TestCase::QUICK);
}

static WifiMacQueueTestSuite g_wifiMacQueueTestSuite; ///< the test suite
