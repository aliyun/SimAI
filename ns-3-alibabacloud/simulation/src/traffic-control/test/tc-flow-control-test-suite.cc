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
 * Author: Stefano Avallone <stavallo@unina.it>
 *
 */

#include <algorithm>
#include <string>

#include "ns3/test.h"
#include "ns3/uinteger.h"
#include "ns3/pointer.h"
#include "ns3/string.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/node-container.h"
#include "ns3/traffic-control-layer.h"
#include "ns3/traffic-control-helper.h"
#include "ns3/simple-net-device-helper.h"
#include "ns3/data-rate.h"
#include "ns3/net-device-queue-interface.h"
#include "ns3/queue.h"
#include "ns3/config.h"

using namespace ns3;

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Queue Disc Test Item
 */
class QueueDiscTestItem : public QueueDiscItem {
public:
  /**
   * Constructor
   *
   * \param p the packet stored in this item
   */
  QueueDiscTestItem (Ptr<Packet> p);
  virtual ~QueueDiscTestItem ();

  // Delete copy constructor and assignment operator to avoid misuse
  QueueDiscTestItem (const QueueDiscTestItem &) = delete;
  QueueDiscTestItem & operator = (const QueueDiscTestItem &) = delete;

  virtual void AddHeader (void);
  virtual bool Mark(void);

private:
  QueueDiscTestItem ();
};

QueueDiscTestItem::QueueDiscTestItem (Ptr<Packet> p)
  : QueueDiscItem (p, Mac48Address (), 0)
{
}

QueueDiscTestItem::~QueueDiscTestItem ()
{
}

void
QueueDiscTestItem::AddHeader (void)
{
}

bool
QueueDiscTestItem::Mark (void)
{
  return false;
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Traffic Control Flow Control Test Case
 */
class TcFlowControlTestCase : public TestCase
{
public:
  /**
   * Constructor
   *
   * \param tt the test type
   * \param deviceQueueLength the queue length of the device
   * \param totalTxPackets the toal number of packets to transmit
   */
  TcFlowControlTestCase (QueueSizeUnit tt, uint32_t deviceQueueLength, uint32_t totalTxPackets);
  virtual ~TcFlowControlTestCase ();
private:
  virtual void DoRun (void);
  /**
   * Instruct a node to send a specified number of packets
   * \param n the node
   * \param nPackets the number of packets to send
   */
  void SendPackets (Ptr<Node> n, uint16_t nPackets);
  /**
   * Check if the device queue stores the expected number of packets
   * \param dev the device
   * \param nPackets the expected number of packets stored in the device queue
   * \param msg the message to print if a different number of packets are stored
   */
  void CheckPacketsInDeviceQueue (Ptr<NetDevice> dev, uint16_t nPackets, const std::string msg);
  /**
   * Check if the device queue is in the expected status (stopped or not)
   * \param dev the device
   * \param value the expected status of the queue (true means stopped)
   * \param msg the message to print if the status of the device queue is different
   */
  void CheckDeviceQueueStopped (Ptr<NetDevice> dev, bool value, const std::string msg);
  /**
   * Check if the queue disc stores the expected number of packets
   * \param dev the device the queue disc is installed on
   * \param nPackets the expected number of packets stored in the queue disc
   * \param msg the message to print if a different number of packets are stored
   */
  void CheckPacketsInQueueDisc (Ptr<NetDevice> dev, uint16_t nPackets, const std::string msg);
  QueueSizeUnit m_type;         //!< the test type
  uint32_t m_deviceQueueLength; //!< the queue length of the device
  uint32_t m_totalTxPackets;    //!< the toal number of packets to transmit
};

TcFlowControlTestCase::TcFlowControlTestCase (QueueSizeUnit tt, uint32_t deviceQueueLength, uint32_t totalTxPackets)
  : TestCase ("Test the operation of the flow control mechanism"),
    m_type (tt), m_deviceQueueLength(deviceQueueLength), m_totalTxPackets(totalTxPackets)
{
}

TcFlowControlTestCase::~TcFlowControlTestCase ()
{
}

void
TcFlowControlTestCase::SendPackets (Ptr<Node> n, uint16_t nPackets)
{
  Ptr<TrafficControlLayer> tc = n->GetObject<TrafficControlLayer> ();
  for (uint16_t i = 0; i < nPackets; i++)
    {
      tc->Send (n->GetDevice (0), Create<QueueDiscTestItem> (Create<Packet> (1000)));
    }
}

void
TcFlowControlTestCase::CheckPacketsInDeviceQueue (Ptr<NetDevice> dev, uint16_t nPackets, const std::string msg)
{
  PointerValue ptr;
  dev->GetAttributeFailSafe ("TxQueue", ptr);
  Ptr<Queue<Packet> > queue = ptr.Get<Queue<Packet> > ();
  NS_TEST_EXPECT_MSG_EQ (queue->GetNPackets (), nPackets, msg);
}

void
TcFlowControlTestCase::CheckDeviceQueueStopped (Ptr<NetDevice> dev, bool value, const std::string msg)
{
  Ptr<NetDeviceQueueInterface> ndqi = dev->GetObject<NetDeviceQueueInterface> ();
  NS_ASSERT_MSG (ndqi, "A device queue interface has not been aggregated to the device");
  NS_TEST_EXPECT_MSG_EQ (ndqi->GetTxQueue (0)->IsStopped (), value, msg);
}

void
TcFlowControlTestCase::CheckPacketsInQueueDisc (Ptr<NetDevice> dev, uint16_t nPackets, const std::string msg)
{
  Ptr<TrafficControlLayer> tc = dev->GetNode ()->GetObject<TrafficControlLayer> ();
  Ptr<QueueDisc> qdisc = tc->GetRootQueueDiscOnDevice (dev);
  NS_TEST_EXPECT_MSG_EQ (qdisc->GetNPackets (), nPackets, msg);
}


void
TcFlowControlTestCase::DoRun (void)
{
  NodeContainer n;
  n.Create (2);

  n.Get (0)->AggregateObject (CreateObject<TrafficControlLayer> ());
  n.Get (1)->AggregateObject (CreateObject<TrafficControlLayer> ());

  SimpleNetDeviceHelper simple;

  NetDeviceContainer rxDevC = simple.Install (n.Get (1));

  simple.SetDeviceAttribute ("DataRate", DataRateValue (DataRate ("1Mb/s")));
  simple.SetQueue ("ns3::DropTailQueue", "MaxSize",
                   StringValue (m_type == QueueSizeUnit::PACKETS
                                    ? std::to_string (m_deviceQueueLength) + "p"
                                    : std::to_string (m_deviceQueueLength) + "B"));

  Ptr<NetDevice> txDev;
  txDev = simple.Install (n.Get (0), DynamicCast<SimpleChannel> (rxDevC.Get (0)->GetChannel ())).Get (0);
  txDev->SetMtu (2500);

  TrafficControlHelper tch = TrafficControlHelper::Default ();
  tch.Install (txDev);

  // transmit 10 packets at time 0
  Simulator::Schedule (Time (Seconds (0)), &TcFlowControlTestCase::SendPackets,
                      this, n.Get (0), m_totalTxPackets);

  if (m_type == QueueSizeUnit::PACKETS)
    {
      /*
       * When the device queue is in packet mode, all the packets enqueued in the
       * queue disc are correctly transmitted, even if the device queue is stopped
       * when the last packet is received from the upper layers
       * 
       * We have the following invariants:
       *  - totalPackets = txPackets + deviceQueuePackets + qdiscPackets
       *  - deviceQueuePackets = MIN(totalPackets - txPackets, deviceQueueLen)
       *  - qdiscPackets = MAX(totalPackets - txPackets - deviceQueuePackets, 0)
       * 
       * The transmission of each packet takes 1000B/1Mbps = 8ms
       * 
       * We check the values of deviceQueuePackets and qdiscPackets 1ms after each
       * packet is transmitted (i.e. at 1ms, 9ms, 17ms, ...), as well as verifying
       * that the device queue is stopped or not, as appropriate.
       */

      uint32_t checkTimeMs = 0;
      uint32_t deviceQueuePackets = 0;
      uint32_t qdiscPackets = 0;

      uint32_t txPackets = 0;
      for (txPackets = 1; txPackets <= m_totalTxPackets; txPackets++)
        {
          checkTimeMs = 8 * (txPackets - 1) + 1; // Check 1ms after each packet is sent
          deviceQueuePackets = std::min (m_totalTxPackets - txPackets, m_deviceQueueLength);
          qdiscPackets = std::max (m_totalTxPackets - txPackets - deviceQueuePackets, (uint32_t)0);
          if (deviceQueuePackets == m_deviceQueueLength)
            {
              Simulator::Schedule (
                  Time (MilliSeconds (checkTimeMs)),
                  &TcFlowControlTestCase::CheckDeviceQueueStopped, this, txDev, true,
                  "The device queue must be stopped after " + std::to_string (checkTimeMs) + "ms");
            }
          else
            {
              Simulator::Schedule (Time (MilliSeconds (checkTimeMs)),
                                   &TcFlowControlTestCase::CheckDeviceQueueStopped, this, txDev,
                                   false,
                                   "The device queue must not be stopped after " +
                                       std::to_string (checkTimeMs) + "ms");
            }

          Simulator::Schedule (
              Time (MilliSeconds (checkTimeMs)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
              this, txDev, deviceQueuePackets,
              "There must be " + std::to_string (m_deviceQueueLength) + " packets in the device after " +
                  std::to_string (checkTimeMs) + "ms");

          Simulator::Schedule (
              Time (MilliSeconds (checkTimeMs)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
              this, txDev, qdiscPackets,
              "There must be " + std::to_string (qdiscPackets) +
                  " packets in the queue disc after " + std::to_string (checkTimeMs) + "ms");
        }
    }
  else
    {
      // TODO: Make this test parametric as well, and add new test cases
      /*
       * When the device queue is in byte mode, all the packets enqueued in the
       * queue disc are correctly transmitted, even if the device queue is stopped
       * when the last packet is received from the upper layers
       */

      // The transmission of each packet takes 1000B/1Mbps = 8ms
      // After 1ms, we have 3 packets in the device queue (stopped) and 6 in the queue disc
      Simulator::Schedule (Time (MilliSeconds (1)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 1ms");
      Simulator::Schedule (Time (MilliSeconds (1)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 1ms");
      Simulator::Schedule (Time (MilliSeconds (1)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 6, "There must be 6 packets in the queue disc after 1ms");

      // After 9ms, we have 3 packets in the device queue (stopped) and 5 in the queue disc
      Simulator::Schedule (Time (MilliSeconds (9)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 9ms");
      Simulator::Schedule (Time (MilliSeconds (9)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 9ms");
      Simulator::Schedule (Time (MilliSeconds (9)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 5, "There must be 5 packets in the queue disc after 9ms");

      // After 17ms, we have 3 packets in the device queue (stopped) and 4 in the queue disc
      Simulator::Schedule (Time (MilliSeconds (17)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 17ms");
      Simulator::Schedule (Time (MilliSeconds (17)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 17ms");
      Simulator::Schedule (Time (MilliSeconds (17)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 4, "There must be 4 packets in the queue disc after 17ms");

      // After 25ms, we have 3 packets in the device queue (stopped) and 3 in the queue disc
      Simulator::Schedule (Time (MilliSeconds (25)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 25ms");
      Simulator::Schedule (Time (MilliSeconds (25)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 25ms");
      Simulator::Schedule (Time (MilliSeconds (25)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 3, "There must be 3 packets in the queue disc after 25ms");

      // After 33ms, we have 3 packets in the device queue (stopped) and 2 in the queue disc
      Simulator::Schedule (Time (MilliSeconds (33)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 33ms");
      Simulator::Schedule (Time (MilliSeconds (33)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 33ms");
      Simulator::Schedule (Time (MilliSeconds (33)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 2, "There must be 2 packets in the queue disc after 33ms");

      // After 41ms, we have 3 packets in the device queue (stopped) and 1 in the queue disc
      Simulator::Schedule (Time (MilliSeconds (41)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 41ms");
      Simulator::Schedule (Time (MilliSeconds (41)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 41ms");
      Simulator::Schedule (Time (MilliSeconds (41)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 1, "There must be 1 packet in the queue disc after 41ms");

      // After 49ms, we have 3 packets in the device queue (stopped) and the queue disc is empty
      Simulator::Schedule (Time (MilliSeconds (49)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 3, "There must be 3 packets in the device queue after 49ms");
      Simulator::Schedule (Time (MilliSeconds (49)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, true, "The device queue must be stopped after 49ms");
      Simulator::Schedule (Time (MilliSeconds (49)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 0, "The queue disc must be empty after 49ms");

      // After 57ms, we have 2 packets in the device queue (not stopped) and the queue disc is empty
      Simulator::Schedule (Time (MilliSeconds (57)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 2, "There must be 2 packets in the device queue after 57ms");
      Simulator::Schedule (Time (MilliSeconds (57)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, false, "The device queue must not be stopped after 57ms");
      Simulator::Schedule (Time (MilliSeconds (57)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 0, "The queue disc must be empty after 57ms");

      // After 81ms, all packets must have been transmitted (the device queue and the queue disc are empty)
      Simulator::Schedule (Time (MilliSeconds (81)), &TcFlowControlTestCase::CheckPacketsInDeviceQueue,
                          this, txDev, 0, "The device queue must be empty after 81ms");
      Simulator::Schedule (Time (MilliSeconds (81)), &TcFlowControlTestCase::CheckDeviceQueueStopped,
                          this, txDev, false, "The device queue must not be stopped after 81ms");
      Simulator::Schedule (Time (MilliSeconds (81)), &TcFlowControlTestCase::CheckPacketsInQueueDisc,
                          this, txDev, 0, "The queue disc must be empty after 81ms");
    }

  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup traffic-control-test
 * \ingroup tests
 *
 * \brief Traffic Control Flow Control Test Suite
 */
static class TcFlowControlTestSuite : public TestSuite
{
public:
  TcFlowControlTestSuite ()
    : TestSuite ("tc-flow-control", UNIT)
  {
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 1, 10), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 5, 10), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 9, 10), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 10, 10), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 11, 10), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 15, 10), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 1, 1), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 2, 1), TestCase::QUICK);
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::PACKETS, 5, 1), TestCase::QUICK);

    // TODO: Right now, this test only works for 5000B and 10 packets (it's hard coded). Should
    // also be made parametric.
    AddTestCase (new TcFlowControlTestCase (QueueSizeUnit::BYTES, 5000, 10), TestCase::QUICK);
  }
} g_tcFlowControlTestSuite; ///< the test suite
