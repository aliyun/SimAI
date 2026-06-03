/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 ResiliNets, ITTC, University of Kansas
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
 * Author: Truc Anh N. Nguyen <annguyen@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 */

#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-yeah.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpYeahTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing TcpYeah additive increase algorithm.
 */
class TcpYeahIncrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param ssThresh Slow Start Threshold.
   * \param segmentSize Segment size.
   * \param nextTxSeq Next Tx sequence number.
   * \param lastAckedSeq Last ACKed sequence number.
   * \param segmentsAcked Number of segments ACKed.
   * \param minRtt Minimum RTT.
   * \param name Test description.
   */
  TcpYeahIncrementTest (uint32_t cWnd,
                        uint32_t ssThresh,
                        uint32_t segmentSize,
                        SequenceNumber32 nextTxSeq,
                        SequenceNumber32 lastAckedSeq,
                        uint32_t segmentsAcked,
                        Time minRtt,
                        const std::string &name);

private:
  virtual void DoRun ();
  /**
   * \brief Increases the TCP window.
   * \param cong The congestion control.
   */
  void IncreaseWindow (Ptr<TcpYeah> cong);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_ssThresh;    //!< Slow Start Threshold.
  uint32_t m_segmentSize; //!< Segment size.
  SequenceNumber32 m_nextTxSeq; //!< Next Tx sequence number.
  SequenceNumber32 m_lastAckedSeq;  //!< Last ACKed sequence number.
  uint32_t m_segmentsAcked; //!< Number of segments ACKed.
  Time m_baseRtt;         //!< Base RTT.
  Time m_minRtt;          //!< Min RTT.
  uint32_t m_doingRenoNow;  //!< TCP Reno fallback.
  uint32_t m_cntRtt;      //!< RTT counter.
  uint32_t m_renoCount;   //!< Reno counter.
};

TcpYeahIncrementTest::TcpYeahIncrementTest (uint32_t cWnd,
                                            uint32_t ssThresh,
                                            uint32_t segmentSize,
                                            SequenceNumber32 nextTxSeq,
                                            SequenceNumber32 lastAckedSeq,
                                            uint32_t segmentsAcked,
                                            Time minRtt,
                                            const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_ssThresh (ssThresh),
    m_segmentSize (segmentSize),
    m_nextTxSeq (nextTxSeq),
    m_lastAckedSeq (lastAckedSeq),
    m_segmentsAcked (segmentsAcked),
    m_baseRtt (MilliSeconds (100)),
    m_minRtt (minRtt),
    m_doingRenoNow (0),
    m_cntRtt (4),
    m_renoCount (2)
{
}

void
TcpYeahIncrementTest::DoRun ()
{
  Ptr<TcpSocketState> state = CreateObject <TcpSocketState> ();
  state->m_cWnd = m_cWnd;
  state->m_ssThresh = m_ssThresh;
  state->m_segmentSize = m_segmentSize;
  state->m_nextTxSequence = m_nextTxSeq;
  state->m_lastAckedSeq = m_lastAckedSeq;
  state->m_minRtt = m_minRtt;

  Ptr<TcpYeah> cong = CreateObject <TcpYeah> ();

  // Set baseRtt to 100 ms
  cong->PktsAcked (state, m_segmentsAcked, m_baseRtt);

  // Reset YeAH to assign a new value of minRtt
  cong->CongestionStateSet (state, TcpSocketState::CA_OPEN);
  cong->PktsAcked (state, m_segmentsAcked, m_minRtt);

  // 2 more calls to PktsAcked to increment cntRtt beyond 2
  cong->PktsAcked (state, m_segmentsAcked, m_minRtt);
  cong->PktsAcked (state, m_segmentsAcked, m_minRtt);

  cong->IncreaseWindow (state, m_segmentsAcked);
  IncreaseWindow (cong);

  NS_TEST_ASSERT_MSG_EQ (state->m_cWnd.Get (), m_cWnd,
                         "CWnd has not updated correctly");
}

void
TcpYeahIncrementTest::IncreaseWindow (Ptr<TcpYeah> cong)
{
  uint32_t segCwnd = m_cWnd / m_segmentSize;

  if (m_cWnd < m_ssThresh)
    { // NewReno slow start
      if (m_segmentsAcked >= 1)
        {
          m_cWnd += m_segmentSize;
          m_segmentsAcked -= 1;
          NS_LOG_INFO ("In SlowStart, updated to cwnd " << m_cWnd <<
                       " ssthresh " << m_ssThresh);
        }
    }
  else if (!m_doingRenoNow)
    { // Fast mode, follow STCP increment rule
      UintegerValue aiFactor;
      cong->GetAttribute ("StcpAiFactor", aiFactor);
      uint32_t w = std::min (segCwnd, (uint32_t) aiFactor.Get ());
      uint32_t delta = m_segmentsAcked / w;
      m_cWnd += delta * m_segmentSize;
      NS_LOG_INFO ("In Fast mode, updated to cwnd " << m_cWnd <<
                   " ssthresh " << m_ssThresh);
    }
  else
    { // NewReno congestion avoidance
      if (m_segmentsAcked > 0)
        {
          double adder = static_cast<double> (m_segmentSize * m_segmentSize) / m_cWnd;
          adder = std::max (1.0, adder);
          m_cWnd += static_cast<uint32_t> (adder);
          NS_LOG_INFO ("In CongAvoid, updated to cwnd " << m_cWnd <<
                       " ssthresh " << m_ssThresh);
        }

    }
  if (m_lastAckedSeq >= m_nextTxSeq)
    {
      if (m_cntRtt > 2)
        {
          /*
           * Calculate the extra number of packets in queue
           */
          Time rttQueue = m_minRtt - m_baseRtt;
          double bw = segCwnd / m_minRtt.GetSeconds ();
          uint32_t queue = bw * rttQueue.GetSeconds ();

          // Calculate network congestion level
          double L = rttQueue.GetSeconds () / m_baseRtt.GetSeconds ();

          UintegerValue alpha;
          cong->GetAttribute ("Alpha", alpha);
          UintegerValue phy;
          cong->GetAttribute ("Phy", phy);
          UintegerValue gamma;
          cong->GetAttribute ("Gamma", gamma);
          UintegerValue epsilon;
          cong->GetAttribute ("Epsilon", epsilon);
          UintegerValue zeta;
          cong->GetAttribute ("Zeta", zeta);

          if (queue > alpha.Get () || L > (1 / phy.Get ()) )
            { // Slow mode
              NS_LOG_INFO ("Enter Slow mode");
              if (queue > alpha.Get () && segCwnd > m_renoCount)
                { // Precautionary decongestion
                  uint32_t reduction = std::min (queue / (uint32_t) gamma.Get (),
                                        segCwnd >> (uint32_t) epsilon.Get ());
                  segCwnd -= reduction;
                  segCwnd = std::max (segCwnd, m_renoCount);
                  m_cWnd = segCwnd * m_segmentSize;
                  m_ssThresh = m_cWnd;
                  NS_LOG_INFO ("In Slow mode, after precautionary decongestion, "
                               "updated to cwnd " << m_cWnd << " ssthresh " <<
                               m_ssThresh);
                }
            }
        }
    }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing TcpYeah multiplicative decrease algorithm
 */
class TcpYeahDecrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param ssThresh Slow Start Threshold.
   * \param segmentSize Segment size.
   * \param nextTxSeq Next Tx sequence number.
   * \param lastAckedSeq Last ACKed sequence number.
   * \param minRtt Minimum RTT.
   * \param rho TCP Yeah rho param.
   * \param name Test description.
   */
  TcpYeahDecrementTest (uint32_t cWnd,
                        uint32_t ssThresh,
                        uint32_t segmentSize,
                        SequenceNumber32 nextTxSeq,
                        SequenceNumber32 lastAckedSeq,
                        Time minRtt,
                        UintegerValue rho,
                        const std::string &name);

private:
  virtual void DoRun (void);
  /**
   * \brief Calculate the Slow Start threshold.
   * \param cong The TCP state.
   * \returns The SshThreshold.
   */
  uint32_t CalculateSsThresh (Ptr<TcpYeah> cong);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_ssThresh;    //!< Slow Start Threshold.
  uint32_t m_segmentSize; //!< Segment size.
  uint32_t m_doingRenoNow;  //!< TCP Reno fallback.
  SequenceNumber32 m_nextTxSeq; //!< Next Tx sequence number.
  SequenceNumber32 m_lastAckedSeq;  //!< Last ACKed sequence number.
  Time m_minRtt;          //!< Min RTT.
  Time m_baseRtt;         //!< Base RTT.
  uint32_t m_segmentsAcked; //!< Number of segments ACKed.
  UintegerValue m_rho;      //!< TCP Yeah rho param.
};

TcpYeahDecrementTest::TcpYeahDecrementTest (uint32_t cWnd,
                                            uint32_t ssThresh,
                                            uint32_t segmentSize,
                                            SequenceNumber32 nextTxSeq,
                                            SequenceNumber32 lastAckedSeq,
                                            Time minRtt,
                                            UintegerValue rho,
                                            const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_ssThresh (ssThresh),
    m_segmentSize (segmentSize),
    m_doingRenoNow (0),
    m_nextTxSeq (nextTxSeq),
    m_lastAckedSeq (lastAckedSeq),
    m_minRtt (minRtt),
    m_baseRtt (MilliSeconds (100)),
    m_segmentsAcked (2),
    m_rho (rho)
{
}

void
TcpYeahDecrementTest::DoRun ()
{
  Ptr<TcpSocketState> state = CreateObject <TcpSocketState> ();
  state->m_cWnd = m_cWnd;
  state->m_nextTxSequence = m_nextTxSeq;
  state->m_lastAckedSeq = m_lastAckedSeq;
  state->m_segmentSize = m_segmentSize;
  state->m_ssThresh = m_ssThresh;

  Ptr<TcpYeah> cong = CreateObject <TcpYeah> ();

  // Re-set rho to 1 for this unit test
  cong->SetAttribute ("Rho", UintegerValue (m_rho));

  // Set baseRtt to 100 ms
  cong->PktsAcked (state, m_segmentsAcked, m_baseRtt);

  // Set minRtt to a different value
  cong->CongestionStateSet (state, TcpSocketState::CA_OPEN);
  cong->PktsAcked (state, m_segmentsAcked, m_minRtt);

  // 2 more calls to PktsAcked to increment cntRtt beyond 2
  cong->PktsAcked (state, m_segmentsAcked, m_minRtt);
  cong->PktsAcked (state, m_segmentsAcked, m_minRtt);

  // Calculate queue backlog
  cong->IncreaseWindow (state, m_segmentsAcked);

  // Now get the value of ssThresh
  uint32_t ssThresh = cong->GetSsThresh (state, m_cWnd);

  // Our calculation of ssThresh
  uint32_t ssThreshVal = CalculateSsThresh (cong);

  NS_TEST_ASSERT_MSG_EQ (ssThresh, ssThreshVal,
                         "SsThresh has not updated correctly");
}

uint32_t
TcpYeahDecrementTest::CalculateSsThresh (Ptr<TcpYeah> cong)
{
  // Calculate queue backlog
  uint32_t segCwnd = m_cWnd / m_segmentSize;
  uint32_t reduction;

  UintegerValue delta;
  cong->GetAttribute ("Delta", delta);

  Time rttQueue =  m_minRtt - m_baseRtt;

  // queue = rttQueue * bw = rttQueue * (cwnd/RTTmin)
  double bw = segCwnd / m_minRtt.GetSeconds ();
  uint32_t queue = bw * rttQueue.GetSeconds ();

  NS_LOG_LOGIC ("queue backlog" << queue);

  if (m_doingRenoNow < m_rho.Get ())
    {
      reduction = std::max (queue, segCwnd >> delta.Get ());
      reduction = std::min (reduction, std::max (segCwnd >> 1, (uint32_t) 2));
      NS_LOG_INFO ("Reduction amount for yeah upon loss = " << reduction);
    }
  else
    {
      reduction = std::max (segCwnd >> 1, (uint32_t) 2);
      NS_LOG_INFO ("Reduction amount for reno upon loss = " << reduction);
    }


  return (m_cWnd - (reduction * m_segmentSize));

}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP Yeah TestSuite
 */
class TcpYeahTestSuite : public TestSuite
{
public:
  TcpYeahTestSuite () : TestSuite ("tcp-yeah-test", UNIT)
  {
    AddTestCase (new TcpYeahIncrementTest (20 * 1446, 25 * 1446, 1446, SequenceNumber32 (2893),
                                           SequenceNumber32 (1447), 1, MilliSeconds (105),
                                           "YeAH test on cWnd when in slow start"),
                 TestCase::QUICK);
    AddTestCase (new TcpYeahIncrementTest (30 * 1446, 25 * 1446, 1446, SequenceNumber32 (2893),
                                           SequenceNumber32 (1447), 30, MilliSeconds (105),
                                           "YeAH test on cWnd when in Fast mode"),
                 TestCase::QUICK);
    AddTestCase (new TcpYeahIncrementTest (40 * 356, 30 * 356, 356, SequenceNumber32 (20761),
                                           SequenceNumber32 (21117), 1, MilliSeconds (120),
                                           "YeAH test on cWnd when in slow mode without precautionary decongestion"),
                 TestCase::QUICK);
    AddTestCase (new TcpYeahIncrementTest (100 * 356, 70 * 356, 356, SequenceNumber32 (20761),
                                           SequenceNumber32 (21117), 1, MilliSeconds (600),
                                           "YeAH test on cWnd when in slow mode with precautionary decongestion"),
                 TestCase::QUICK);
    AddTestCase (new TcpYeahDecrementTest (40 * 1446, 30 * 1446, 1446, SequenceNumber32 (2893),
                                           SequenceNumber32 (7230), MilliSeconds (120), UintegerValue (0),
                                           "YeAH test on ssThresh upon loss while competing with Reno flows"),
                 TestCase::QUICK);
    AddTestCase (new TcpYeahDecrementTest (57 * 1446, 42 * 1446, 1446, SequenceNumber32 (2893),
                                           SequenceNumber32 (7230), MilliSeconds (200), UintegerValue (2),
                                           "YeAH test on ssThresh upon loss while not competing with Reno flows"),
                 TestCase::QUICK);
  }
};

static TcpYeahTestSuite g_tcpYeahTest; //!< Static variable for test initialization

