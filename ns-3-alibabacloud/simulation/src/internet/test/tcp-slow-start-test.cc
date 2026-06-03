/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
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
#include "ns3/log.h"
#include "ns3/test.h"
#include "ns3/simple-channel.h"
#include "ns3/node.h"
#include "ns3/config.h"
#include "ns3/tcp-header.h"
#include "tcp-general-test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpSlowStartTest");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Test the normal behavior for slow start
 *
 * As method for checking the slow start, a callback is attached to the
 * congestion window. With the knowledge of the number of segments, we can calculate
 * if the value of the cWnd is right. Also, with a fixed delay for each packet,
 * we can know if the timing is correct.
 *
 * Check what is done inside CWndTrace.
 *
 * \see CWndTrace
 */
class
TcpSlowStartNormalTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param segmentSize Segment size.
   * \param packetSize Packet size.
   * \param initSsTh Initial SlowStart threshold.
   * \param packets Packet counter.
   * \param congControl Congestion control.
   * \param desc Test description.
   */
  TcpSlowStartNormalTest (uint32_t segmentSize, uint32_t packetSize,
                          uint32_t initSsTh, uint32_t packets, const TypeId& congControl,
                          const std::string &desc);

protected:
  virtual void CWndTrace (uint32_t oldValue, uint32_t newValue);
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who);
  void QueueDrop (SocketWho who);
  void PhyDrop (SocketWho who);

  virtual void ConfigureEnvironment ();
  virtual void ConfigureProperties ();

  uint32_t m_ackedBytes;        //!< ACKed bytes.
  uint32_t m_sentBytes;         //!< Sent bytes.
  uint32_t m_totalAckedBytes;   //!< Total ACKed bytes.
  uint32_t m_allowedIncrease;   //!< Allowed increase.

  bool   m_initial; //!< First cycle flag.

private:
  uint32_t m_segmentSize;   //!< Segment size.
  uint32_t m_packetSize;    //!< Packet size.
  uint32_t m_packets;       //!< Packet counter.
};

TcpSlowStartNormalTest::TcpSlowStartNormalTest (uint32_t segmentSize,
                                                uint32_t packetSize,
                                                uint32_t initSsTh,
                                                uint32_t packets,
                                                const TypeId &typeId,
                                                const std::string &desc)
  : TcpGeneralTest (desc),
    m_ackedBytes (0),
    m_sentBytes (0),
    m_totalAckedBytes (0),
    m_allowedIncrease (0),
    m_initial (true),
    m_segmentSize (segmentSize),
    m_packetSize (packetSize),
    m_packets (packets)
{
  m_congControlTypeId = typeId;
}

void
TcpSlowStartNormalTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (m_packets);
  SetAppPktSize (m_packetSize);
}

void
TcpSlowStartNormalTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetInitialSsThresh (SENDER, 400000);
  SetSegmentSize (SENDER, m_segmentSize);
  SetSegmentSize (RECEIVER, m_segmentSize);
}

void
TcpSlowStartNormalTest::QueueDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the queue; cannot validate slow start");
}

void
TcpSlowStartNormalTest::PhyDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the phy: cannot validate slow start");
}

/**
 * \brief Trace the cWnd over the slow start
 *
 * This method is called each time the cWnd changes. It should be updated only
 * by MSS bytes at time. Since the size doubles each RTT, a timing test is also
 * performed: the doubling should be made in 0.5s from the first (0.5s is
 * the delay of the SimpleChannel which connect the two socket).
 *
 * \param oldValue old value of cWnd
 * \param newValue new value of cWnd
 */
void
TcpSlowStartNormalTest::CWndTrace (uint32_t oldValue, uint32_t newValue)
{
  uint32_t segSize = GetSegSize (TcpGeneralTest::SENDER);
  uint32_t increase = newValue - oldValue;

  if (m_initial)
    {
      m_initial = false;
      NS_LOG_INFO ("Ignored update to " << newValue <<
                   " with a segsize of " << segSize);
      return;
    }

  // The increase in RFC should be <= of segSize. In ns-3 we force = segSize
  NS_TEST_ASSERT_MSG_EQ (increase, segSize, "Increase different than segsize");
  NS_TEST_ASSERT_MSG_LT_OR_EQ (newValue, GetInitialSsThresh (SENDER), "cWnd increased over ssth");

  NS_LOG_INFO ("Incremented cWnd by " << segSize << " bytes in Slow Start " <<
               "achieving a value of " << newValue);

  NS_TEST_ASSERT_MSG_GT_OR_EQ (m_allowedIncrease, 1, "Increase not allowed");
  m_allowedIncrease--;
}

void
TcpSlowStartNormalTest::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);

  if (who == SENDER && Simulator::Now ().GetSeconds () > 5.0)
    {
      m_sentBytes += GetSegSize (TcpGeneralTest::SENDER);
    }
}

void
TcpSlowStartNormalTest::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);

  if (who == SENDER && Simulator::Now ().GetSeconds () > 5.0)
    {
      uint32_t acked = h.GetAckNumber ().GetValue () - m_totalAckedBytes - 1;
      m_totalAckedBytes += acked;
      m_ackedBytes += acked;

      NS_LOG_INFO ("Ack of " << acked << " bytes, acked this round=" << m_ackedBytes);

      if (m_ackedBytes >= GetSegSize (SENDER))
        {
          NS_LOG_INFO ("FULL ACK achieved, bytes=" << m_ackedBytes);
          m_allowedIncrease += 1;
          m_ackedBytes -= GetSegSize (SENDER);
        }

      while (m_ackedBytes >= GetSegSize (SENDER))
        {
          m_ackedBytes -= GetSegSize (SENDER);
        }
    }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief A slow start test using a socket which sends smaller ACKs
 *
 * The same test are performed over a connection where, on one side, there is
 * a malicious socket which sends smaller ACKs than the segment received.
 *
 * Slow start behavior should not change.
 */
class
TcpSlowStartAttackerTest : public TcpSlowStartNormalTest
{
public:
  /**
   * \brief Constructor.
   * \param segmentSize Segment size.
   * \param packetSize Packet size.
   * \param initSsTh Initial SlowStart threshold.
   * \param packets Packet counter.
   * \param congControl Congestion control.
   * \param desc Test description.
   */
  TcpSlowStartAttackerTest (uint32_t segmentSize, uint32_t packetSize,
                            uint32_t initSsTh, uint32_t packets, const TypeId& congControl,
                            const std::string &desc);

protected:
  virtual Ptr<TcpSocketMsgBase> CreateReceiverSocket (Ptr<Node> node);
};

TcpSlowStartAttackerTest::TcpSlowStartAttackerTest (uint32_t segmentSize,
                                                    uint32_t packetSize,
                                                    uint32_t initSsTh,
                                                    uint32_t packets,
                                                    const TypeId &typeId,
                                                    const std::string &msg)
  : TcpSlowStartNormalTest (segmentSize, packetSize, initSsTh, packets, typeId, msg)
{

}

Ptr<TcpSocketMsgBase>
TcpSlowStartAttackerTest::CreateReceiverSocket (Ptr<Node> node)
{
  Ptr<TcpSocketSmallAcks> socket = DynamicCast<TcpSocketSmallAcks> (
      CreateSocket (node,
                    TcpSocketSmallAcks::GetTypeId (),
                    m_congControlTypeId));
  socket->SetBytesToAck (125);

  return socket;
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP Slow Start TestSuite.
 */
class TcpSlowStartTestSuite : public TestSuite
{
public:
  TcpSlowStartTestSuite () : TestSuite ("tcp-slow-start-test", UNIT)
  {
    // This test have less packets to transmit than SsTh
    std::list<TypeId> types = {
      TcpNewReno::GetTypeId (),
    };

    for (const auto &t : types)
      {
        std::string typeName = t.GetName ();

        AddTestCase (new TcpSlowStartNormalTest (500, 500, 10000, 10, t,
                                                 "slow start 500 byte, " + typeName),
                     TestCase::QUICK);
        AddTestCase (new TcpSlowStartNormalTest (1000, 1000, 10000, 9, t,
                                                 "slow start 1000 byte, " + typeName),
                     TestCase::QUICK);
        AddTestCase (new TcpSlowStartNormalTest (500, 250, 10000, 10, t,
                                                 "slow start small packets, " + typeName),
                     TestCase::QUICK);
        AddTestCase (new TcpSlowStartAttackerTest (500, 500, 10000, 10, t,
                                                   "slow start ack attacker, 500 byte, " + typeName),
                     TestCase::QUICK);
        AddTestCase (new TcpSlowStartAttackerTest (1000, 1000, 10000, 9, t,
                                                   "slow start ack attacker, 1000 byte, " + typeName),
                     TestCase::QUICK);
      }
  }
};

static TcpSlowStartTestSuite g_tcpSlowStartTestSuite; //!< Static variable for test initialization
