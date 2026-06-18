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
#include "ns3/tcp-vegas.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpVegasTestSuite");

/**
 * \brief TcpVegas congestion control algorithm test
 */
class TcpVegasTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param ssThresh Slow Start Threshold.
   * \param rtt The RTT.
   * \param segmentsAcked Number of segments ACKed.
   * \param nextTxSeq Next Tx sequence number.
   * \param lastAckedSeq Last ACKed sequence number.
   * \param name Test description.
   */
  TcpVegasTest (uint32_t cWnd,
                uint32_t segmentSize,
                uint32_t ssThresh,
                Time rtt,
                uint32_t segmentsAcked,
                SequenceNumber32 nextTxSeq,
                SequenceNumber32 lastAckedSeq,
                const std::string &name);

private:
  virtual void DoRun (void);
  /**
   * \brief Increases the TCP window.
   * \param cong The congestion control.
   */
  void IncreaseWindow (Ptr<TcpVegas> cong);
  /**
   * brief Get and check the SSH threshold.
   * \param cong The congestion control.
   */
  void GetSsThresh (Ptr<TcpVegas> cong);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_segmentSize; //!< Segment size.
  uint32_t m_ssThresh;    //!< Slow Start Threshold.
  Time m_rtt;             //!< RTT.
  uint32_t m_segmentsAcked; //!< Number of segments ACKed.
  SequenceNumber32 m_nextTxSeq; //!< Next Tx sequence number.
  SequenceNumber32 m_lastAckedSeq;  //!< Last ACKed sequence number.

  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

TcpVegasTest::TcpVegasTest (uint32_t cWnd,
                            uint32_t segmentSize,
                            uint32_t ssThresh,
                            Time rtt,
                            uint32_t segmentsAcked,
                            SequenceNumber32 nextTxSeq,
                            SequenceNumber32 lastAckedSeq,
                            const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_ssThresh (ssThresh),
    m_rtt (rtt),
    m_segmentsAcked (segmentsAcked),
    m_nextTxSeq (nextTxSeq),
    m_lastAckedSeq (lastAckedSeq)
{
}

void
TcpVegasTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;
  m_state->m_ssThresh = m_ssThresh;
  m_state->m_nextTxSequence = m_nextTxSeq;
  m_state->m_lastAckedSeq = m_lastAckedSeq;
  m_state->m_minRtt = m_rtt;

  Ptr<TcpVegas> cong = CreateObject <TcpVegas> ();

  // Set baseRtt to 100 ms
  cong->PktsAcked (m_state, m_segmentsAcked, MilliSeconds (100));

  // Re-set Vegas to assign a new value of minRtt
  cong->CongestionStateSet (m_state, TcpSocketState::CA_OPEN);
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  // 2 more calls to PktsAcked to increment cntRtt beyond 2
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  // Update cwnd using Vegas algorithm
  cong->IncreaseWindow (m_state, m_segmentsAcked);

  // Our calculation of cwnd
  IncreaseWindow (cong);

  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), m_cWnd,
                         "CWnd has not updated correctly");
  NS_TEST_ASSERT_MSG_EQ (m_state->m_ssThresh.Get (), m_ssThresh,
                         "SsThresh has not updated correctly");
}

void
TcpVegasTest::IncreaseWindow (Ptr<TcpVegas> cong)
{
  Time baseRtt = MilliSeconds (100);
  uint32_t segCwnd = m_cWnd / m_segmentSize;

  // Calculate expected throughput
  uint64_t expectedCwnd;
  expectedCwnd = (uint64_t) segCwnd * (double) baseRtt.GetMilliSeconds () / (double) m_rtt.GetMilliSeconds ();

  // Calculate the difference between actual and expected throughput
  uint32_t diff;
  diff = segCwnd - expectedCwnd;

  // Get the alpha,beta, and gamma attributes
  UintegerValue alpha, beta, gamma;
  cong->GetAttribute ("Alpha", alpha);
  cong->GetAttribute ("Beta", beta);
  cong->GetAttribute ("Gamma", gamma);

  if (diff > gamma.Get () && (m_cWnd < m_ssThresh))
    { // Change from slow-start to linear increase/decrease mode
      segCwnd = std::min (segCwnd, (uint32_t) expectedCwnd + 1);
      m_cWnd = segCwnd * m_segmentSize;
      GetSsThresh (cong);
    }
  else if (m_cWnd < m_ssThresh)
    { // Execute Reno slow start
      if (m_segmentsAcked >= 1)
        {
          m_cWnd += m_segmentSize;
          m_segmentsAcked--;
        }
    }
  else
    { // Linear increase/decrease mode
      if (diff > beta.Get ())
        {
          m_cWnd = (segCwnd - 1) * m_segmentSize;
          GetSsThresh (cong);
        }
      else if (diff < alpha.Get ())
        {
          m_cWnd = (segCwnd + 1) * m_segmentSize;
        }
      else
        {
        }
    }
  m_ssThresh = std::max (m_ssThresh, 3 * m_cWnd / 4);
}

void
TcpVegasTest::GetSsThresh (Ptr<TcpVegas> cong)
{
  m_ssThresh = std::max (std::min (m_ssThresh, m_cWnd - m_segmentSize), 2 * m_segmentSize);
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP Vegas TestSuite
 */
class TcpVegasTestSuite : public TestSuite
{
public:
  TcpVegasTestSuite () : TestSuite ("tcp-vegas-test", UNIT)
  {
    AddTestCase (new TcpVegasTest (38 * 1446, 1446, 40 * 1446, MilliSeconds (106), 1, SequenceNumber32 (2893), SequenceNumber32 (5785),
                                   "Vegas test on cWnd and ssThresh when in slow start and diff > gamma"),
                 TestCase::QUICK);
    AddTestCase (new TcpVegasTest (5 * 536, 536, 10 * 536, MilliSeconds (118), 1, SequenceNumber32 (3216), SequenceNumber32 (3753),
                                   "Vegas test on cWnd and ssThresh when in slow start and diff < gamma"),
                 TestCase::QUICK);
    AddTestCase (new TcpVegasTest (60 * 346, 346, 40 * 346, MilliSeconds (206), 1, SequenceNumber32 (20761), SequenceNumber32 (21107),
                                   "Vegas test on cWnd and ssThresh when diff > beta"),
                 TestCase::QUICK);
    AddTestCase (new TcpVegasTest (15 * 1446, 1446, 10 * 1446, MilliSeconds (106), 1, SequenceNumber32 (21691), SequenceNumber32 (24583),
                                   "Vegas test on cWnd and ssThresh when diff < alpha"),
                 TestCase::QUICK);
    AddTestCase (new TcpVegasTest (20 * 746, 746, 10 * 746, MilliSeconds (109), 1, SequenceNumber32 (14921), SequenceNumber32 (15667),
                                   "Vegas test on cWnd and ssThresh when alpha <= diff <= beta"),
                 TestCase::QUICK);
  }
};

static TcpVegasTestSuite g_tcpVegasTest; //!< Static variable for test initialization
