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
 * Authors: Charitha Sangaraju <charitha29193@gmail.com>
 *          Nandita G <gm.nandita@gmail.com>
 *          Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-lp.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpLpTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the behaviour common to New Reno
 */
class TcpLpToNewReno : public TestCase
{
public:
  /**
   * Constructor
   * \param cWnd Congestion window size
   * \param segmentSize Segment size
   * \param segmentsAcked Segments acked
   * \param ssThresh Slow start threshold
   * \param rtt RTT
   * \param name Test case name
   */
  TcpLpToNewReno (uint32_t cWnd, uint32_t segmentSize,
                  uint32_t segmentsAcked, uint32_t ssThresh, Time rtt, const std::string &name);

private:
  virtual void DoRun (void);
  uint32_t m_cWnd;          //!< Congestion window size
  uint32_t m_segmentSize;   //!< Segment size
  uint32_t m_ssThresh;      //!< Slow start threshold
  uint32_t m_segmentsAcked; //!< Segments acked
  Time m_rtt;               //!< RTT
  Ptr<TcpSocketState> m_state;  //!< TCP socket state
};

TcpLpToNewReno::TcpLpToNewReno (uint32_t cWnd, uint32_t segmentSize,
                                uint32_t segmentsAcked, uint32_t ssThresh, Time rtt, const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_ssThresh (ssThresh),
    m_segmentsAcked (segmentsAcked),
    m_rtt (rtt)
{
}

void
TcpLpToNewReno::DoRun ()
{
  m_state = CreateObject <TcpSocketState> ();
  m_state->m_cWnd = m_cWnd;
  m_state->m_ssThresh = m_ssThresh;
  m_state->m_segmentSize = m_segmentSize;

  Ptr<TcpSocketState> state = CreateObject <TcpSocketState> ();
  state->m_cWnd = m_cWnd;
  state->m_ssThresh = m_ssThresh;
  state->m_segmentSize = m_segmentSize;

  Ptr<TcpLp> cong = CreateObject <TcpLp> ();

  m_state->m_rcvTimestampValue = 2;
  m_state->m_rcvTimestampEchoReply = 1;

  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);
  cong->IncreaseWindow (m_state, m_segmentsAcked);

  Ptr<TcpNewReno> NewRenoCong = CreateObject <TcpNewReno> ();
  NewRenoCong->IncreaseWindow (state, m_segmentsAcked);

  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), state->m_cWnd.Get (),
                         "cWnd has not updated correctly");
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing TcpLp when cwd exceeds threshold
 */
class TcpLpInferenceTest1 : public TestCase
{
public:
  /**
   * Constructor
   * \param cWnd Congestion window size
   * \param segmentSize Segment size
   * \param segmentsAcked Segments acked
   * \param rtt RTT
   * \param name Test case name
   */
  TcpLpInferenceTest1 (uint32_t cWnd, uint32_t segmentSize,
                       uint32_t segmentsAcked, Time rtt, const std::string &name);

private:
  virtual void DoRun (void);

  uint32_t m_cWnd;          //!< Congestion window size
  uint32_t m_segmentSize;   //!< Segment size
  uint32_t m_segmentsAcked; //!< Segments acked
  Time m_rtt;               //!< RTT
  Ptr<TcpSocketState> m_state;  //!< TCP socket state
};

TcpLpInferenceTest1::TcpLpInferenceTest1 (uint32_t cWnd, uint32_t segmentSize,
                                          uint32_t segmentsAcked, Time rtt, const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_segmentsAcked (segmentsAcked),
    m_rtt (rtt)
{
}

void
TcpLpInferenceTest1::DoRun ()
{
  m_state = CreateObject <TcpSocketState> ();
  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;

  Ptr<TcpLp> cong = CreateObject <TcpLp> ();

  m_state->m_rcvTimestampValue = 2;
  m_state->m_rcvTimestampEchoReply = 1;

  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  m_state->m_rcvTimestampValue = 14;
  m_state->m_rcvTimestampEchoReply = 4;
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  m_cWnd =  m_cWnd / 2;
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), m_cWnd,
                         "cWnd has not updated correctly");
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing TcpLp when it is inference phase
 */
class TcpLpInferenceTest2 : public TestCase
{
public:
  /**
   * Constructor
   * \param cWnd Congestion window size
   * \param segmentSize Segment size
   * \param segmentsAcked Segments acked
   * \param rtt RTT
   * \param name Test case name
   */
  TcpLpInferenceTest2 (uint32_t cWnd, uint32_t segmentSize,
                       uint32_t segmentsAcked, Time rtt, const std::string &name);

private:
  virtual void DoRun (void);

  uint32_t m_cWnd;          //!< Congestion window size
  uint32_t m_segmentSize;   //!< Segment size
  uint32_t m_segmentsAcked; //!< Segments acked
  Time m_rtt;               //!< RTT
  Ptr<TcpSocketState> m_state;  //!< TCP socket state
};

TcpLpInferenceTest2::TcpLpInferenceTest2 (uint32_t cWnd, uint32_t segmentSize,uint32_t segmentsAcked, Time rtt, const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_segmentsAcked (segmentsAcked),
    m_rtt (rtt)
{
}

void
TcpLpInferenceTest2::DoRun ()
{
  m_state = CreateObject <TcpSocketState> ();
  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;

  Ptr<TcpLp> cong = CreateObject <TcpLp> ();

  m_state->m_rcvTimestampValue = 2;
  m_state->m_rcvTimestampEchoReply = 1;
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  m_state->m_rcvTimestampValue = 14;
  m_state->m_rcvTimestampEchoReply = 4;
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  m_state->m_rcvTimestampValue = 25;
  m_state->m_rcvTimestampEchoReply = 15;
  cong->PktsAcked (m_state, m_segmentsAcked, m_rtt);

  m_cWnd = 1U * m_segmentSize;

  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), m_cWnd,
                         "cWnd has not updated correctly");
  Simulator::Run ();
  Simulator::Destroy ();
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * Test the behaviour common to New Reno
 */
class TcpLpTestSuite : public TestSuite
{
public:
  TcpLpTestSuite () : TestSuite ("tcp-lp-test", UNIT)
  {
    AddTestCase (new TcpLpToNewReno (4 * 1446, 1446, 2, 2 * 1446, MilliSeconds (100), "LP falls to New Reno if the cwd is within threshold"), TestCase::QUICK);

    AddTestCase (new TcpLpInferenceTest1 (2 * 1446, 1446, 2, MilliSeconds (100), "LP enters Inference phase when cwd exceeds threshold for the first time"), TestCase::QUICK);

    AddTestCase (new TcpLpInferenceTest2 (2 * 1446, 1446, 2, MilliSeconds (100), "LP reduces cWnd to 1 if cwd exceeds threshold in inference phase"), TestCase::QUICK);
  }
};

static TcpLpTestSuite g_tcplpTest; //!< static var for test initialization

}
