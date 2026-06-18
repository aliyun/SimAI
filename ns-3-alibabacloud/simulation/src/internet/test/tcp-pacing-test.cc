/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 NITK Surathkal
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
 * Authors: Deepak Kumaraswamy <deepakkavoor99@gmail.com>
 *
 */
#include "ns3/log.h"
#include "ns3/simple-channel.h"
#include "ns3/config.h"
#include "ns3/test.h"
#include "tcp-general-test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpPacingTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Test the behavior of TCP pacing
 *
 * This test checks that packets are paced at correct intervals.  The test
 * uses a shadow pacing rate calculation assumed to match the internal
 * pacing calculation.  Therefore, if you modify the values of
 * pacingSsRatio and pacingCaRatio herein, ensure that you also change
 * values used in the TCP implementation to match.
 *
 * This test environment uses an RTT of 100ms
 * The time interval between two consecutive packet transmissions is measured
 * Pacing rate should be at least cwnd / rtt to transmit cwnd segments per
 * rtt.  Linux multiples this basic ratio by an additional factor, to yield
 * pacingRate = (factor * cwnd) / rtt
 * Since pacingRate can also be written as segmentSize / interval
 * we can solve for interval = (segmentSize * rtt) / (factor * cwnd)
 *
 * The test checks whether the measured interval lies within a tolerance
 * value of the expected interval.  The tolerance or error margin
 * was chosen to be 10 Nanoseconds, that could be due to delay introduced
 * by the application's send process.
 *
 * This check should not be performed for a packet transmission after the
 * sender has sent all bytes corresponding to the window and is awaiting an
 * ACK from the receiver (corresponding to m_isFullCwndSent).
 * Pacing check should be performed when the sender is actively sending packets from cwnd.
 *
 * The same argument applies when the sender has finished sending packets and is awaiting a
 * FIN/ACK from the receiver, to send the final ACK
 *
 * As can be seen in TcpSocketBase::UpdatePacingRate (), different pacing
 * ratios are used when cwnd < ssThresh / 2 and when cwnd > ssThresh / 2
 *
 * A few key points to note:
 * - In TcpSocketBase, pacing rate is updated whenever an ACK is received.
 *
 * - The factors that could contribute to a different value of pacing rate include
 * congestion window and RTT.
 *
 * - However, the cWnd trace is called after Rx () trace and we therefore
 * update the expected interval in cWnd trace.
 *
 * - An RTT trace is also necessary, since using delayed ACKs may lead to a
 * higher RTT measurement at the sender's end. The expected interval should
 * be updated here as well.
 *
 * - When sending the current packet, TcpSocketBase automatically decides the time at
 * which next packet should be sent. So, if say packet 3 is sent and an ACK is
 * received before packet 4 is sent (thus updating the pacing rate), this does not change
 * the time at which packet 4 is to be sent. When packet 4 is indeed sent later, the new
 * pacing rate is used to decide when packet 5 will be sent. This behavior is captured in
 * m_nextPacketInterval, which is not affected by change of pacing rate before the next
 * packet is sent. The important observation here is to realize the contrast between
 * m_expectedInterval and m_nextPacketInterval.
 *
 */
class
TcpPacingTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param segmentSize Segment size at the TCP layer (bytes).
   * \param packetSize Size of packets sent at the application layer (bytes).
   * \param packets Number of packets.
   * \param pacingSsRatio Pacing Ratio during Slow Start (multiplied by 100)
   * \param pacingCaRatio Pacing Ratio during Congestion Avoidance (multiplied by 100)
   * \param ssThresh slow start threshold (bytes)
   * \param paceInitialWindow whether to pace the initial window
   * \param delAckMaxCount Delayed ACK max count parameter
   * \param congControl Type of congestion control.
   * \param desc The test description.
   */
  TcpPacingTest (uint32_t segmentSize, uint32_t packetSize,
                 uint32_t packets, uint16_t pacingSsRatio, uint16_t pacingCaRatio,
                 uint32_t ssThresh, bool paceInitialWindow, uint32_t delAckMaxCount,
                 const TypeId& congControl, const std::string &desc);

protected:
  virtual void CWndTrace (uint32_t oldValue, uint32_t newValue);
  virtual void RttTrace (Time oldTime, Time newTime);
  virtual void BytesInFlightTrace (uint32_t oldValue, uint32_t newValue);
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who);
  virtual void QueueDrop (SocketWho who);
  virtual void PhyDrop (SocketWho who);
  virtual void NormalClose (SocketWho who);

  /**
   * \brief Update the expected interval at which next packet will be sent
   */
  virtual void UpdateExpectedInterval (void);

  virtual void ConfigureEnvironment ();
  virtual void ConfigureProperties ();

private:
  uint32_t m_segmentSize;             //!< Segment size
  uint32_t m_packetSize;              //!< Size of the packets
  uint32_t m_packets;                 //!< Number of packets
  EventId  m_event;                   //!< Check event
  bool     m_initial;                 //!< True on first run
  uint32_t m_initialCwnd;             //!< Initial value of cWnd
  uint32_t m_curCwnd;                 //!< Current sender cWnd
  bool     m_isFullCwndSent;          //!< True if all bytes for that cWnd is sent and sender is waiting for an ACK
  uint32_t m_bytesInFlight;           //!< Current bytes in flight
  Time     m_prevTxTime;              //!< Time when Tx was previously called
  uint16_t m_pacingSsRatio;           //!< Pacing factor during Slow Start
  uint16_t m_pacingCaRatio;           //!< Pacing factor during Congestion Avoidance
  uint32_t m_ssThresh;                //!< Slow start threshold
  bool     m_paceInitialWindow;       //!< True if initial window should be paced
  uint32_t m_delAckMaxCount;          //!< Delayed ack count for receiver
  bool     m_isConnAboutToEnd;        //!< True when sender receives a FIN/ACK from receiver
  Time     m_transmissionStartTime;   //!< Time at which sender starts data transmission
  Time     m_expectedInterval;        //!< Theoretical estimate of the time at which next packet is scheduled for transmission
  uint32_t m_packetsSent;             //!< Number of packets sent by sender so far
  Time     m_nextPacketInterval;      //!< Time maintained by Tx () trace about interval at which next packet will be sent
  Time     m_tracedRtt;               //!< Traced value of RTT, which may be different from the environment RTT in case of delayed ACKs
};


TcpPacingTest::TcpPacingTest (uint32_t segmentSize,
                              uint32_t packetSize,
                              uint32_t packets,
                              uint16_t pacingSsRatio,
                              uint16_t pacingCaRatio,
                              uint32_t ssThresh,
                              bool paceInitialWindow,
                              uint32_t delAckMaxCount,
                              const TypeId &typeId,
                              const std::string &desc)
  : TcpGeneralTest (desc),
    m_segmentSize (segmentSize),
    m_packetSize (packetSize),
    m_packets (packets),
    m_initial (true),
    m_initialCwnd (10),
    m_curCwnd (0),
    m_isFullCwndSent (true),
    m_bytesInFlight (0),
    m_prevTxTime (0),
    m_pacingSsRatio (pacingSsRatio),
    m_pacingCaRatio (pacingCaRatio),
    m_ssThresh (ssThresh),
    m_paceInitialWindow (paceInitialWindow),
    m_delAckMaxCount (delAckMaxCount),
    m_isConnAboutToEnd (false),
    m_transmissionStartTime (Seconds (0)),
    m_expectedInterval (Seconds (0)),
    m_packetsSent (0),
    m_nextPacketInterval (Seconds (0)),
    m_tracedRtt (Seconds (0))
{
  m_congControlTypeId = typeId;
}

void
TcpPacingTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktSize (m_packetSize);
  SetAppPktCount (m_packets);
  SetAppPktInterval (NanoSeconds (10));
  SetMTU (1500);
  SetTransmitStart (Seconds (0));
  SetPropagationDelay (MilliSeconds (50));
}

void TcpPacingTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetSegmentSize (SENDER, m_segmentSize);
  SetInitialSsThresh (SENDER, m_ssThresh);
  SetInitialCwnd (SENDER, m_initialCwnd);
  SetPacingStatus (SENDER, true);
  SetPaceInitialWindow (SENDER, m_paceInitialWindow);
  SetDelAckMaxCount (RECEIVER, m_delAckMaxCount);
  NS_LOG_DEBUG ("segSize: " << m_segmentSize << " ssthresh: " << m_ssThresh <<
                " paceInitialWindow: " << m_paceInitialWindow << " delAckMaxCount " << m_delAckMaxCount);
}

void
TcpPacingTest::RttTrace (Time oldTime, Time newTime)
{
  NS_LOG_FUNCTION (this << oldTime << newTime);
  m_tracedRtt = newTime;
  UpdateExpectedInterval ();
}

void
TcpPacingTest::CWndTrace (uint32_t oldValue, uint32_t newValue)
{
  NS_LOG_FUNCTION (this << oldValue << newValue);
  m_curCwnd = newValue;
  if (m_initial)
    {
      m_initial = false;
    }
  // CWndTrace () is called after Rx ()
  // Therefore, call UpdateExpectedInterval () here instead of in Rx ()
  UpdateExpectedInterval ();
}

void
TcpPacingTest::BytesInFlightTrace (uint32_t oldValue, uint32_t newValue)
{
  m_bytesInFlight = newValue;
}

void
TcpPacingTest::UpdateExpectedInterval (void)
{
  double_t factor;
  Time rtt = 2 * GetPropagationDelay ();
  if (m_curCwnd < m_ssThresh / 2)
    {
      factor = static_cast<double> (m_pacingSsRatio) / 100;
    }
  else
    {
      factor = static_cast<double> (m_pacingCaRatio) / 100;
    }

  if (!m_paceInitialWindow && (m_curCwnd == m_initialCwnd * m_segmentSize))
    {
      // If initial cwnd is not paced, we expect packet pacing interval to be zero
      m_expectedInterval = Seconds (0);
    }
  else
    {
      // Use the estimate according to update equation
      m_expectedInterval = Seconds ((m_segmentSize * m_tracedRtt.GetSeconds ()) / (factor * m_curCwnd));
    }
}

void
TcpPacingTest::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      uint8_t flags = h.GetFlags ();
      uint8_t hasFin = flags & TcpHeader::FIN;
      uint8_t hasAck = flags & TcpHeader::ACK;
      if (hasFin && hasAck)
        {
          m_isConnAboutToEnd = true;
          NS_LOG_DEBUG ("Sender received a FIN/ACK packet");
        }
      else
        {
          m_isConnAboutToEnd = false;
          NS_LOG_DEBUG ("Sender received an ACK packet");
        }
    }
}

void
TcpPacingTest::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);

  if (who == SENDER)
    {
      m_packetsSent++;
      // Start pacing checks from the second data packet onwards because
      // an interval to check does not exist for the first data packet.
      // The first two (non-data) packets correspond to SYN and an
      // empty ACK, respectively, so start checking after three packets are sent
      bool beyondInitialDataSegment = (m_packetsSent > 3);
      Time actualInterval = Simulator::Now () - m_prevTxTime;
      NS_LOG_DEBUG ("TX sent: packetsSent: " << m_packetsSent << " fullCwnd: " << m_isFullCwndSent << " nearEnd: " <<
                    m_isConnAboutToEnd << " beyondInitialDataSegment " << beyondInitialDataSegment);
      if (!m_isFullCwndSent && !m_isConnAboutToEnd && beyondInitialDataSegment)
        {
          // Consider a small error margin, and ensure that the actual and expected intervals lie within this error
          Time errorMargin = NanoSeconds (10);
          NS_TEST_ASSERT_MSG_LT_OR_EQ (std::abs ((actualInterval - m_nextPacketInterval).GetSeconds ()), errorMargin.GetSeconds (), "Packet delivery in slow start didn't match pacing rate");
          NS_LOG_DEBUG ("Pacing Check: interval (s): " << actualInterval.GetSeconds () <<
                        " expected interval (s): " << m_nextPacketInterval.GetSeconds () <<
                        " difference (s): " << std::abs ((actualInterval - m_nextPacketInterval).GetSeconds ()) <<
                        " errorMargin (s): " << errorMargin.GetSeconds ());
        }

      m_prevTxTime = Simulator::Now ();
      // bytesInFlight isn't updated yet. Its trace is called after Tx
      // so add an additional m_segmentSize to bytesInFlight
      uint32_t soonBytesInFlight = m_bytesInFlight + m_segmentSize;
      bool canPacketBeSent = ((m_curCwnd - soonBytesInFlight) >= m_segmentSize);
      if (!canPacketBeSent || (m_curCwnd == 0))
        {
          m_isFullCwndSent = true;
        }
      else
        {
          m_isFullCwndSent = false;
        }
      m_nextPacketInterval = m_expectedInterval;
      NS_LOG_DEBUG ("Next expected interval (s): " << m_nextPacketInterval.GetSeconds ());
    }
}

void
TcpPacingTest::QueueDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the queue; cannot validate congestion avoidance");
}

void
TcpPacingTest::PhyDrop (SocketWho who)
{
  NS_FATAL_ERROR ("Drop on the phy: cannot validate congestion avoidance");
}

void
TcpPacingTest::NormalClose (SocketWho who)
{
  if (who == SENDER)
    {
      m_event.Cancel ();
    }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TestSuite for the behavior of TCP pacing
 */
class TcpPacingTestSuite : public TestSuite
{
public:
  TcpPacingTestSuite () : TestSuite ("tcp-pacing-test", UNIT)
  {
    uint16_t pacingSsRatio = 200;
    uint16_t pacingCaRatio = 120;
    uint32_t segmentSize = 1000;
    uint32_t packetSize = 1000;
    uint32_t numPackets = 40;
    uint32_t delAckMaxCount = 1;
    TypeId tid = TcpNewReno::GetTypeId ();
    uint32_t ssThresh = 1e9; // default large value
    bool paceInitialWindow = false;
    std::string description;

    description = std::string ("Pacing case 1: Slow start only, no initial pacing");
    AddTestCase (new TcpPacingTest (segmentSize, packetSize, numPackets, pacingSsRatio, pacingCaRatio, ssThresh, paceInitialWindow, delAckMaxCount, tid, description), TestCase::QUICK);

    paceInitialWindow = true;
    description = std::string ("Pacing case 2: Slow start only, initial pacing");
    AddTestCase (new TcpPacingTest (segmentSize, packetSize, numPackets, pacingSsRatio, pacingCaRatio, ssThresh, paceInitialWindow, delAckMaxCount, tid, description), TestCase::QUICK);

    // set ssThresh to some smaller value to check that pacing
    // slows down in second half of slow start, then transitions to CA
    description = std::string ("Pacing case 3: Slow start, followed by transition to Congestion avoidance, no initial pacing");
    paceInitialWindow = false;
    ssThresh = 40;
    numPackets = 60;
    AddTestCase (new TcpPacingTest (segmentSize, packetSize, numPackets, pacingSsRatio, pacingCaRatio, ssThresh, paceInitialWindow, delAckMaxCount, tid, description), TestCase::QUICK);

    // Repeat tests, but with more typical delAckMaxCount == 2
    delAckMaxCount = 2;
    paceInitialWindow = false;
    ssThresh = 1e9;
    numPackets = 40;
    description = std::string ("Pacing case 4: Slow start only, no initial pacing, delayed ACKs");
    AddTestCase (new TcpPacingTest (segmentSize, packetSize, numPackets, pacingSsRatio, pacingCaRatio, ssThresh, paceInitialWindow, delAckMaxCount, tid, description), TestCase::QUICK);

    paceInitialWindow = true;
    description = std::string ("Pacing case 5: Slow start only, initial pacing, delayed ACKs");
    AddTestCase (new TcpPacingTest (segmentSize, packetSize, numPackets, pacingSsRatio, pacingCaRatio, ssThresh, paceInitialWindow, delAckMaxCount, tid, description), TestCase::QUICK);

    description = std::string ("Pacing case 6: Slow start, followed by transition to Congestion avoidance, no initial pacing, delayed ACKs");
    paceInitialWindow = false;
    ssThresh = 40;
    numPackets = 60;
    AddTestCase (new TcpPacingTest (segmentSize, packetSize, numPackets, pacingSsRatio, pacingCaRatio, ssThresh, paceInitialWindow, delAckMaxCount, tid, description), TestCase::QUICK);
  }
};

static TcpPacingTestSuite g_tcpPacingTest; //!< Static variable for test initialization
