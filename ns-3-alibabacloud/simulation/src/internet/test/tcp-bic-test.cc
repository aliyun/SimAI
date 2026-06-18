/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello, <natale.patriciello@gmail.com>
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

#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-bic.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpBicTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the congestion avoidance increment on TcpBic
 */
class TcpBicIncrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param ssThresh Slow Start Threshold.
   * \param segmentsAcked Number of segments acked.
   * \param lastMaxCwnd Last max Cwnd.
   * \param name Test description.
   */
  TcpBicIncrementTest (uint32_t cWnd,
                       uint32_t segmentSize,
                       uint32_t ssThresh,
                       uint32_t segmentsAcked,
                       uint32_t lastMaxCwnd,
                       const std::string &name);

private:
  virtual void DoRun (void);

  /**
   * \brief Update the TCP socket state.
   * \param tcb The TCP socket state.
   * \returns The ack counter.
   */
  uint32_t Update (Ptr<TcpSocketState> tcb);

  /**
   * \brief Execute the test.
   */
  void ExecuteTest (void);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_segmentSize; //!< Segment size.
  uint32_t m_ssThresh;    //!< Slow Start Threshold.
  uint32_t m_segmentsAcked; //!< Number of segments acked.
  uint32_t m_lastMaxCwnd;   //!< Last max Cwnd.
  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

TcpBicIncrementTest::TcpBicIncrementTest (uint32_t cWnd,
                                          uint32_t segmentSize,
                                          uint32_t ssThresh,
                                          uint32_t segmentsAcked,
                                          uint32_t lastMaxCwnd,
                                          const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_ssThresh (ssThresh),
    m_segmentsAcked (segmentsAcked),
    m_lastMaxCwnd (lastMaxCwnd)
{
}

void
TcpBicIncrementTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;
  m_state->m_ssThresh = m_ssThresh;

  Simulator::Schedule (Seconds (0.0), &TcpBicIncrementTest::ExecuteTest, this);
  Simulator::Run ();
  Simulator::Destroy ();
}

void
TcpBicIncrementTest::ExecuteTest ()
{
  uint32_t segCwnd = m_cWnd / m_segmentSize;

  uint32_t ackCnt = Update (m_state);

  if (m_segmentsAcked > ackCnt)
    {
      NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), segCwnd * m_segmentSize + m_segmentSize,
                             "Bic has not increment cWnd");
      /*  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), 27000,
                               "Bic has not increment cWnd");*/
    }
  else
    {
      NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), segCwnd * m_segmentSize,
                             "Bic has modified cWnd");
    }
}

uint32_t
TcpBicIncrementTest::Update (Ptr<TcpSocketState> tcb)
{
  uint32_t segCwnd = tcb->m_cWnd / tcb->m_segmentSize;
  Ptr<TcpBic> cong = CreateObject <TcpBic> ();
  cong->m_lastMaxCwnd = m_lastMaxCwnd;
  UintegerValue lowWindow, bsCoeff, wMax;
  UintegerValue smoothPart;
  cong->GetAttribute ("LowWnd", lowWindow);
  cong->GetAttribute ("BinarySearchCoefficient", bsCoeff);
  cong->GetAttribute ("MaxIncr", wMax);
  cong->GetAttribute ("SmoothPart", smoothPart);

  cong->IncreaseWindow (m_state, m_segmentsAcked);

  uint32_t ackCnt = 0;

  if (segCwnd < lowWindow.Get ())
    {
      ackCnt = segCwnd;
      return ackCnt;
    }
  if (segCwnd < m_lastMaxCwnd)
    {
      double midPt = (m_lastMaxCwnd - segCwnd) / bsCoeff.Get ();
      if (midPt > wMax.Get ())
        {
          //Linear increase
          ackCnt = segCwnd / wMax.Get ();
        }
      else if (midPt <= 1)
        {
          ackCnt = (segCwnd * smoothPart.Get ()) / bsCoeff.Get ();
        }
      else
        {
          //Binary search increase
          ackCnt = segCwnd / midPt;
        }
    }
  else
    {
      if (segCwnd < m_lastMaxCwnd + bsCoeff.Get ())
        {
          /* slow start AMD linear increase */
          ackCnt = (segCwnd * smoothPart.Get ()) / bsCoeff.Get ();
        }
      else if (segCwnd < m_lastMaxCwnd + wMax.Get () * (bsCoeff.Get () - 1))
        {
          /* slow start */
          ackCnt = (segCwnd * (bsCoeff.Get () - 1)) / (segCwnd - m_lastMaxCwnd);
        }
      else
        {
          /* linear increase */
          ackCnt = segCwnd / wMax.Get ();

        }
    }
  return ackCnt;

}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the congestion avoidance decrement on TcpBic
 */
class TcpBicDecrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param fastConvergence Fast convergence.
   * \param lastMaxCwnd Last max Cwnd.
   * \param name Test description.
   */
  TcpBicDecrementTest (uint32_t cWnd,
                       uint32_t segmentSize,
                       BooleanValue fastConvergence,
                       uint32_t lastMaxCwnd,
                       const std::string &name);

private:
  virtual void DoRun (void);

  /**
   * \brief Execute the test.
   */
  void ExecuteTest (void);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_segmentSize; //!< Segment size.
  BooleanValue m_fastConvergence;   //!< Fast convergence.
  uint32_t m_lastMaxCwnd;   //!< Last max Cwnd.
  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

TcpBicDecrementTest::TcpBicDecrementTest (uint32_t cWnd,
                                          uint32_t segmentSize,
                                          BooleanValue fastConvergence,
                                          uint32_t lastMaxCwnd,
                                          const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_fastConvergence (fastConvergence),
    m_lastMaxCwnd (lastMaxCwnd)
{
}

void
TcpBicDecrementTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;

  Simulator::Schedule (Seconds (0.0), &TcpBicDecrementTest::ExecuteTest, this);
  Simulator::Run ();
  Simulator::Destroy ();
}

void
TcpBicDecrementTest::ExecuteTest ()
{
  Ptr<TcpBic> cong = CreateObject <TcpBic> ();
  cong->m_lastMaxCwnd = m_lastMaxCwnd;
  cong->SetAttribute ("FastConvergence", m_fastConvergence);

  uint32_t segCwnd = m_cWnd / m_segmentSize;
  uint32_t retSsThresh = cong->GetSsThresh (m_state, m_state->m_cWnd);
  uint32_t retLastMaxCwnd = cong->m_lastMaxCwnd;

  DoubleValue beta;
  UintegerValue lowWindow;
  cong->GetAttribute ("Beta", beta);
  cong->GetAttribute ("LowWnd", lowWindow);

  uint32_t lastMaxCwnd, ssThresh;

  if (segCwnd < m_lastMaxCwnd && m_fastConvergence.Get ())
    {
      lastMaxCwnd = beta.Get () * segCwnd;
      NS_TEST_ASSERT_MSG_EQ (retLastMaxCwnd, lastMaxCwnd,
                             "Bic has not updated lastMaxCwnd during fast convergence");
    }
  else
    {
      lastMaxCwnd = segCwnd;
      NS_TEST_ASSERT_MSG_EQ (retLastMaxCwnd, lastMaxCwnd,
                             "Bic has not reset lastMaxCwnd to current cwnd (in segments)");
    }


  if (segCwnd < lowWindow.Get ())
    {
      ssThresh = std::max (2 * m_segmentSize,  m_cWnd / 2);
      NS_TEST_ASSERT_MSG_EQ (retSsThresh, ssThresh,
                             "Bic has not updated ssThresh when cWnd less than lowWindow");
    }
  else
    {
      ssThresh = std::max (segCwnd * beta.Get (), 2.0) * m_segmentSize;
      NS_TEST_ASSERT_MSG_EQ (retSsThresh, ssThresh,
                             "Bic has not updated ssThresh when cWnd greater than lowWindow");
    }
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP Bic TestSuite
 */
class TcpBicTestSuite : public TestSuite
{
public:
  TcpBicTestSuite () : TestSuite ("tcp-bic-test", UNIT)
  {
    AddTestCase (new TcpBicIncrementTest (10 * 536, 536, 9 * 536, 11, 0,
                                          "Bic increment test: under lowCwnd & enough ACKs received"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (10 * 536, 536, 9 * 536, 8, 0,
                                          "Bic increment test: under lowCwnd but not enough ACKs received"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (18 * 1446, 1446, 15 * 1446, 5, 90,
                                          "Bic increment test: linear increase when distance exceeds S_max"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (18 * 1446, 1446, 15 * 1446, 24, 20,
                                          "Bic increment test: binary search increase with smooth part"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (19 * 1, 1, 17 * 1, 2, 83,
                                          "Bic increment test: binary search increase"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (15 * 536, 536, 9 * 536, 19, 13,
                                          "Bic increment test: slow start AMD linear increase"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (22 * 1000, 1000, 9 * 1000, 9, 16,
                                          "Bic increment test: slow start but not enough ACKs received"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicIncrementTest (65 * 1000, 1000, 9 * 1000, 2, 16,
                                          "Bic increment test: linear incrase but not enough ACKs received"),
                 TestCase::QUICK);

    AddTestCase (new TcpBicDecrementTest (5 * 1446, 1446, true, 10,
                                          "Bic decrement test: fast convergence & cwnd less than lowWindow"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicDecrementTest (5 * 1446, 1446, false, 10,
                                          "Bic decrement test: not in fast convergence & cwnd less than lowWindow"),
                 TestCase::QUICK);
    AddTestCase (new TcpBicDecrementTest (15 * 1446, 1446, false, 10,
                                          "Bic decrement test: not in fast convergence & cwnd greater than lowWindow"),
                 TestCase::QUICK);
  }
};

static TcpBicTestSuite g_tcpBicTest; //!< Static variable for test initialization
