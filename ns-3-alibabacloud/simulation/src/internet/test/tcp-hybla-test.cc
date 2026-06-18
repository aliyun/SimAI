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
#include "ns3/tcp-hybla.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpHyblaTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the congestion avoidance increment on TcpHybla
 */
class TcpHyblaIncrementTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param ssThresh Slow Start Threshold.
   * \param segmentSize Segment size.
   * \param rtt Round trip time.
   * \param name Test description.
   */
  TcpHyblaIncrementTest (uint32_t cWnd, uint32_t ssThresh,
                         uint32_t segmentSize, const Time& rtt,
                         const std::string &name);

private:
  virtual void DoRun (void);

  /**
   * \brief Tracks TCP Hybla rho parameter changes.
   * \param oldVal Previous value.
   * \param newVal Actual value.
   */
  void RhoUpdated (double oldVal, double newVal);

  uint32_t m_cWnd;        //!< Congestion window.
  uint32_t m_ssThresh;    //!< Slow Start Threshold.
  uint32_t m_segmentSize; //!< Segment size.
  Time m_rtt;             //!< Round trip time.
  double m_rho;           //!< TCP Hybla rho parameter.
  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

TcpHyblaIncrementTest::TcpHyblaIncrementTest (uint32_t cWnd, uint32_t ssThresh,
                                              uint32_t segmentSize, const Time &rtt,
                                              const std::string &name)
  : TestCase (name),
  m_cWnd (cWnd),
  m_ssThresh (ssThresh),
  m_segmentSize (segmentSize),
  m_rtt (rtt),
  m_rho (0)
{
}

void
TcpHyblaIncrementTest::RhoUpdated ([[maybe_unused]] double oldVal, double newVal)
{
  m_rho = newVal;
}

void
TcpHyblaIncrementTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_ssThresh = m_ssThresh;
  m_state->m_segmentSize = m_segmentSize;
  m_state->m_minRtt = m_rtt;

  Ptr<TcpHybla> cong = CreateObject <TcpHybla> ();

  cong->TraceConnectWithoutContext ("Rho",
                                    MakeCallback (&TcpHyblaIncrementTest::RhoUpdated, this));

  // Each received ACK weight is "coeffA". To see an increase of 1 MSS, we need
  // to ACK at least segCwnd/coeffA ACK.

  TimeValue rRtt;
  cong->GetAttribute ("RRTT", rRtt);
  cong->PktsAcked (m_state, 1, m_rtt);

  double calcRho = std::max (m_rtt.GetSeconds () / rRtt.Get ().GetSeconds (), 1.0);

  NS_TEST_ASSERT_MSG_NE (m_rho, 0.0,
                         "Rho never updated by implementation");
  NS_TEST_ASSERT_MSG_EQ_TOL (calcRho, m_rho, 0.01,
                         "Different rho values between implementation and test");

  cong->IncreaseWindow (m_state, 1);

  if (m_cWnd <= m_ssThresh)
    {
      // We expect an increment of 2^rho - 1, which does not go beyond ssThresh
      double inc = std::pow (2, calcRho) - 1.0;
      uint32_t cWndExpected = m_cWnd + (inc * m_segmentSize);
      NS_TEST_ASSERT_MSG_LT_OR_EQ (m_state->m_cWnd.Get (), m_state->m_ssThresh.Get (),
                                   "Congestion window has gone too far");
      NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), cWndExpected,
                             "Congestion window different than expected");
    }
  else
    {
      // We expect an increment of rho^2 / cWnd
      uint32_t segCwnd = m_cWnd / m_segmentSize;
      double inc = std::pow (m_rho, 2) / ((double) segCwnd);
      uint32_t cWndExpected = m_cWnd + (inc * m_segmentSize);

      if (inc >= 1.0)
        {
          // LT because implementation does not add value less than MSS.
          NS_TEST_ASSERT_MSG_LT_OR_EQ (m_state->m_cWnd.Get (), cWndExpected,
                                       "Congestion window different than expected");
        }
    }

}



/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP Hybla TestSuite
 */
class TcpHyblaTestSuite : public TestSuite
{
public:
  TcpHyblaTestSuite () : TestSuite ("tcp-hybla-test", UNIT)
  {
    AddTestCase (new TcpHyblaIncrementTest (1000, 0xFFFFFFFF, 500, MilliSeconds (55), "Rho=1.1, slow start"),
                 TestCase::QUICK);
    AddTestCase (new TcpHyblaIncrementTest (1000, 0xFFFFFFFF, 500, MilliSeconds (100), "Rho=2, slow start"),
                 TestCase::QUICK);
    AddTestCase (new TcpHyblaIncrementTest (1000, 0xFFFFFFFF, 500, MilliSeconds (750), "Rho=30, slow start"),
                 TestCase::QUICK);
    AddTestCase (new TcpHyblaIncrementTest (1000, 500, 500, Seconds (0.55), "Rho=1.1, cong avoid"),
                 TestCase::QUICK);
    AddTestCase (new TcpHyblaIncrementTest (1000, 500, 500, Seconds (0.1), "Rho=2, cong avoid"),
                 TestCase::QUICK);
    AddTestCase (new TcpHyblaIncrementTest (1000, 500, 500, Seconds (0.75), "Rho=30, cong avoid"),
                 TestCase::QUICK);
  }
};

static TcpHyblaTestSuite g_tcpHyblaTest; //!< Static variable for test initialization
