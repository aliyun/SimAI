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
#include "ns3/string.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpClassicRecoveryTestSuite");

/**
 * \brief Classic Recovery algorithm test
 */
class ClassicRecoveryTest : public TestCase
{
public:
  /**
   * \brief Constructor.
   * \param cWnd Congestion window.
   * \param segmentSize Segment size.
   * \param ssThresh Slow Start Threshold.
   * \param dupAckCount Duplicate acknowledgement Threshold.
   * \param name Test description.
   */
  ClassicRecoveryTest (uint32_t cWnd,
                       uint32_t segmentSize,
                       uint32_t ssThresh,
                       uint32_t dupAckCount,
                       const std::string &name);

private:
  virtual void DoRun (void);

  uint32_t m_cWnd;                    //!< Congestion window.
  uint32_t m_segmentSize;             //!< Segment size.
  uint32_t m_ssThresh;                //!< Slow Start Threshold.
  uint32_t m_dupAckCount;             //!< Duplicate acknowledgement Threshold.

  Ptr<TcpSocketState> m_state;  //!< TCP socket state.
};

ClassicRecoveryTest::ClassicRecoveryTest (uint32_t cWnd,
                                          uint32_t segmentSize,
                                          uint32_t ssThresh,
                                          uint32_t dupAckCount,
                                          const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_ssThresh (ssThresh),
    m_dupAckCount (dupAckCount)
{
}

void
ClassicRecoveryTest::DoRun ()
{
  m_state = CreateObject<TcpSocketState> ();

  m_state->m_cWnd = m_cWnd;
  m_state->m_segmentSize = m_segmentSize;
  m_state->m_ssThresh = m_ssThresh;

  Ptr<TcpClassicRecovery> recovery = CreateObject <TcpClassicRecovery> ();

  NS_TEST_ASSERT_MSG_EQ (recovery->GetName (), "TcpClassicRecovery",
                         "The name of recovery used should be TcpClassicRecovery");

  recovery->EnterRecovery (m_state, m_dupAckCount, 1000, 0);
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd, m_state->m_ssThresh,
                         "cWnd should be set to ssThresh on entering recovery");
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWndInfl, (m_state->m_ssThresh + (m_dupAckCount * m_state->m_segmentSize)),
                         "cWndInfl should be set to (ssThresh + dupAckCount * segmentSize) on entering recovery");

  uint32_t cWndInflPrevious = m_state->m_cWndInfl;
  uint32_t cWndPrevious = m_state->m_cWnd;
  recovery->DoRecovery (m_state, 500);
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWndInfl, (cWndInflPrevious + m_state->m_segmentSize),
                         "m_cWndInfl should be incresed by one segmentSize on calling DoRecovery");
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd, cWndPrevious,
                         "cWnd should not change in recovery");

  recovery->ExitRecovery (m_state);
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWndInfl, m_state->m_ssThresh,
                         "cWndInfl should be set to ssThresh on exiting recovery");
  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd, m_state->m_ssThresh,
                         "cWnd should be set to ssThresh on exiting recovery");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Classic Recovery TestSuite
 */
class ClassicRecoveryTestSuite : public TestSuite
{
public:
  ClassicRecoveryTestSuite () : TestSuite ("tcp-classic-recovery-test", UNIT)
  {
    AddTestCase (new ClassicRecoveryTest (3000, 500, 2500, 3, "Classic recovery test with 500 bytes segmentSize"),
                 TestCase::QUICK);
    AddTestCase (new ClassicRecoveryTest (3000, 1000, 2500, 3, "Classic recovery test with 1000 bytes segmentSize"),
                 TestCase::QUICK);
    AddTestCase (new ClassicRecoveryTest (3000, 500, 2500, 4, "Classic recovery test with 4 DupAck threshold"),
                 TestCase::QUICK);
    AddTestCase (new ClassicRecoveryTest (3000, 500, 1000, 3, "Classic recovery test with 1000 bytes ssThresh"),
                 TestCase::QUICK);
    AddTestCase (new ClassicRecoveryTest (2500, 500, 2500, 3, "Classic recovery test with same cWnd and ssThresh"),
                 TestCase::QUICK);
    AddTestCase (new ClassicRecoveryTest (1000, 500, 2500, 3, "Classic recovery test with cWnd lesser than ssThresh"),
		 TestCase::QUICK);
  }
};

static ClassicRecoveryTestSuite g_TcpClassicRecoveryTest; //!< Static variable for test initialization
