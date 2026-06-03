/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 NITK Surathkal
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
 * Author: Viyom Mittal <viyommittal@gmail.com>
 *         Vivek Jain <jain.vivek.anand@gmail.com>
 *         Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 *
 */

#include "ns3/test.h"
#include "ns3/log.h"
#include "ns3/tcp-congestion-ops.h"
#include "ns3/tcp-recovery-ops.h"
#include "ns3/tcp-socket-base.h"
#include "ns3/tcp-prr-recovery.h"
#include "ns3/string.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpPrrRecoveryTestSuite");

/**
 * \brief PRR Recovery algorithm test
 */
class PrrRecoveryTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param ssThresh Slow Start Threshold.
   * \param unAckDataCount Unacknowledged data at the start of recovery.
   * \param bytesInFlight Current bytes in flight.
   * \param m_deliveredBytes Bytes SACKed on last acknowledgment.
   * \param bytesSent Bytes sent while in recovery phase.
   * \param reductionBound Type of reduction bound to be used.
   * \param name Test description.
   */
  PrrRecoveryTest (uint32_t cWnd,
                   uint32_t segmentSize,
                   uint32_t ssThresh,
                   uint32_t unAckDataCount,
                   uint32_t bytesInFlight,
                   uint32_t m_deliveredBytes,
                   uint32_t bytesSent,
                   const std::string &reductionBound,
                   const std::string &name);

private:
  virtual void DoRun (void);

  uint32_t m_cWnd;                    //!< Congestion window.
  uint32_t m_segmentSize;             //!< Segment size.
  uint32_t m_ssThresh;                //!< Slow Start Threshold.
  uint32_t m_unAckDataCount;          //!< Unacknowledged data at the start of recovery.
  uint32_t m_bytesInFlight;           //!< Current bytes in flight.
  uint32_t m_deliveredBytes;         //!< Bytes SACKed on last acknowledgment.
  uint32_t m_bytesSent;               //!< Bytes sent while in recovery phase.
  const std::string m_reductionBound; //!< Type of reduction bound to be used.

  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

PrrRecoveryTest::PrrRecoveryTest (uint32_t cWnd,
                                  uint32_t segmentSize,
                                  uint32_t ssThresh,
                                  uint32_t unAckDataCount,
                                  uint32_t bytesInFlight,
                                  uint32_t deliveredBytes,
                                  uint32_t bytesSent,
                                  const std::string &reductionBound,
                                  const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_ssThresh (ssThresh),
    m_unAckDataCount (unAckDataCount),
    m_bytesInFlight (bytesInFlight),
    m_deliveredBytes (deliveredBytes),
    m_bytesSent (bytesSent),
    m_reductionBound (reductionBound)
{
}

void
PrrRecoveryTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_cWndInfl = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;
  m_state->m_ssThresh = m_ssThresh;
  m_state->m_bytesInFlight = m_bytesInFlight;

  Ptr<TcpPrrRecovery> recovery = CreateObject <TcpPrrRecovery> ();
  recovery->SetAttribute ("ReductionBound", StringValue (m_reductionBound));

  recovery->EnterRecovery (m_state, 3, m_unAckDataCount, 0);

  NS_TEST_ASSERT_MSG_GT_OR_EQ (m_state->m_cWnd.Get (), m_cWnd + m_segmentSize,
                               "There should be at least one transmission on entering recovery");

  for (uint32_t iterator = 0; iterator < m_bytesSent; )
    {
      recovery->UpdateBytesSent (m_segmentSize);
      iterator += m_segmentSize;
    }

  m_bytesInFlight += m_state->m_cWnd.Get () - m_cWnd;
  m_state->m_bytesInFlight = m_bytesInFlight;
  m_cWnd = m_state->m_cWnd.Get ();
  recovery->DoRecovery (m_state, m_deliveredBytes);

  if (m_bytesInFlight > m_state->m_ssThresh)
    {
      NS_TEST_ASSERT_MSG_LT_OR_EQ (m_state->m_cWnd.Get (), m_cWnd,
                                   "Updated cwnd should be less than or equal to the existing cwnd");
    }
  else
    {
      NS_TEST_ASSERT_MSG_GT_OR_EQ (m_state->m_cWnd.Get (), m_cWnd,
                                   "Updated cwnd should be greater than or equal to the existing cwnd");
    }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief PRR Recovery TestSuite
 */
class PrrRecoveryTestSuite : public TestSuite
{
public:
  PrrRecoveryTestSuite () : TestSuite ("tcp-prr-recovery-test", UNIT)
  {
    AddTestCase (new PrrRecoveryTest (3000, 500, 2500, 3000, 3000, 500, 1000, "SSRB",
                                      "Prr test on cWnd when bytesInFlight is greater than ssThresh with SSRB"),
                 TestCase::QUICK);
    AddTestCase (new PrrRecoveryTest (1000, 500, 2500, 3000, 1000, 500, 1000, "SSRB",
                                      "Prr test on cWnd when bytesInFlight is lower than ssThresh with SSRB"),
                 TestCase::QUICK);
    AddTestCase (new PrrRecoveryTest (3000, 500, 2500, 3000, 3000, 500, 1000, "CRB",
                                      "Prr test on cWnd when bytesInFlight is greater than ssThresh with CRB"),
                 TestCase::QUICK);
    AddTestCase (new PrrRecoveryTest (1000, 500, 2500, 3000, 1000, 500, 1000, "CRB",
                                      "Prr test on cWnd when bytesInFlight is lower than ssThresh with CRB"),
                 TestCase::QUICK);
  }
};

static PrrRecoveryTestSuite g_TcpPrrRecoveryTest; //!< Static variable for test initialization
