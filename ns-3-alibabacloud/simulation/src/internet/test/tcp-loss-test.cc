/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Tom Henderson
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
 */

#include <list>
#include "ns3/log.h"
#include "ns3/error-model.h"
#include "tcp-general-test.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpLossTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Check rollover of sequence number and how that affects loss recovery
 *
 * This test checks that fast recovery is entered correctly even if it has
 * been a long time since the last recovery event.  Merge request !156 
 * reported the error and fixed the issue with large transfers.
 *
 * The issue reported is that fast recovery detection relies on comparing
 * the sequence number against the last recovery point, maintained by
 * an 'm_recover' variable.  The sequence number is a 32-bit value that
 * wraps, so the comparison of 'is current sequence number > m_recover'
 * will work for only about half of the 32-bit sequence space.  This
 * test confirms that the above inequality will correctly evaluate even if
 * the current sequence number is greater than half of the sequence space
 * above m_recover.
 *
 * The test configures a large bandwidth, long flow (to wrap the sequence
 * space) and inserts two forced losses.  In the first test case, the two
 * sequence numbers are close in sequence; we expect to see the TCP
 * implementation cycle between CA_OPEN->CA_DISORDER->CA_RECOVERY->CA_OPEN.
 * We also check in the second case when the two sequence numbers are far
 * apart (more than half) in the sequence space.
 */
class TcpLargeTransferLossTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   *
   * \param firstLoss First packet number to force loss
   * \param secondLoss Second packet number to force loss
   * \param lastSegment Last packet number to transmit
   * \param desc Description about the test
   */
  TcpLargeTransferLossTest (uint32_t firstLoss, uint32_t secondLoss, uint32_t lastSegment, const std::string& desc);

protected:
  void ConfigureProperties (void);
  void ConfigureEnvironment (void);
  void FinalChecks (void);
  void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  void CongStateTrace (const TcpSocketState::TcpCongState_t oldValue,
                       const TcpSocketState::TcpCongState_t newValue);
  Ptr<ErrorModel> CreateReceiverErrorModel ();
private:
  uint32_t m_firstLoss;             //!< First segment loss
  uint32_t m_secondLoss;            //!< Second segment loss
  uint32_t m_sent {0};              //!< Number of segments sent
  uint32_t m_received {0};          //!< Number of segments received
  uint32_t m_lastSegment {0};       //!< Last received segment
  std::list<int> m_expectedStates;  //!< Expected TCP states
};

TcpLargeTransferLossTest::TcpLargeTransferLossTest (uint32_t firstLoss, uint32_t secondLoss, uint32_t lastSegment, const std::string &desc)
  : TcpGeneralTest (desc),
    m_firstLoss (firstLoss),
    m_secondLoss (secondLoss),
    m_lastSegment (lastSegment)
{
  NS_TEST_ASSERT_MSG_NE (m_lastSegment, 0, "Last segment should be > 0");
  NS_TEST_ASSERT_MSG_GT (m_secondLoss, m_firstLoss, "Second segment number should be greater than first");
  m_expectedStates.push_back (TcpSocketState::CA_OPEN);
  m_expectedStates.push_back (TcpSocketState::CA_DISORDER);
  m_expectedStates.push_back (TcpSocketState::CA_RECOVERY);
  m_expectedStates.push_back (TcpSocketState::CA_OPEN);
  m_expectedStates.push_back (TcpSocketState::CA_DISORDER);
  m_expectedStates.push_back (TcpSocketState::CA_RECOVERY);
  m_expectedStates.push_back (TcpSocketState::CA_OPEN);
}

void
TcpLargeTransferLossTest::CongStateTrace (const TcpSocketState::TcpCongState_t oldValue,
                                          const TcpSocketState::TcpCongState_t newValue)
{
  int expectedOldState = m_expectedStates.front ();
  m_expectedStates.pop_front ();
  NS_TEST_ASSERT_MSG_EQ (oldValue, expectedOldState, "State transition wrong");
  NS_TEST_ASSERT_MSG_EQ (newValue, m_expectedStates.front (), "State transition wrong");
}

void
TcpLargeTransferLossTest::ConfigureEnvironment ()
{
  NS_LOG_FUNCTION (this);
  TcpGeneralTest::ConfigureEnvironment ();
  SetPropagationDelay (MicroSeconds (1));  // Keep low to avoid window limit
  SetTransmitStart (Seconds (1));
  SetAppPktSize (1000);
  // Note:  4294967295 is the maximum TCP sequence number, so rollover will be
  // on the 4294967th packet with a packet (segment) size of 1000 bytes
  SetAppPktCount (m_lastSegment);
  SetAppPktInterval (MicroSeconds (8)); // 1 Gb/s
}

void
TcpLargeTransferLossTest::ConfigureProperties ()
{
  NS_LOG_FUNCTION (this);
  TcpGeneralTest::ConfigureProperties ();
  SetSegmentSize (SENDER, 1000);
  SetSegmentSize (RECEIVER, 1000);
}

Ptr<ErrorModel>
TcpLargeTransferLossTest::CreateReceiverErrorModel ()
{
  Ptr<ReceiveListErrorModel> rem = CreateObject<ReceiveListErrorModel> ();
  std::list<uint32_t> errorList;
  errorList.push_back (m_firstLoss);
  errorList.push_back (m_secondLoss);
  rem->SetList (errorList);
  return rem;
}

void 
TcpLargeTransferLossTest::Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  m_sent++;
}

void 
TcpLargeTransferLossTest::Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  m_received++;
}

void
TcpLargeTransferLossTest::FinalChecks (void)
{
  // The addition of 2 accounts for the two forcibly lost packets
  NS_TEST_ASSERT_MSG_EQ (m_sent, (m_received + 2), "Did not observe expected number of sent packets");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * Test various packet losses
 */
class TcpLossTestSuite : public TestSuite
{
public:
  TcpLossTestSuite () : TestSuite ("tcp-loss-test", UNIT)
  {
    // For large transfer tests, the three sequence numbers passed in
    // are the segment (i.e. not byte) number that should be dropped first,
    // then the second drop, and then the last segment number to send
    // 
    // If we force a loss at packet 1000 and then shortly after at 2000,
    // the TCP logic should correctly pass this case (no sequence wrapping).
    AddTestCase (new TcpLargeTransferLossTest (1000, 2000, 2500, "large-transfer-loss-without-wrap"), TestCase::EXTENSIVE);
    // If we force a loss at packet 1000 and then much later at 3294967, the
    // second sequence number will evaluate to less than 1000 considering
    // the sequence space wrap, so check this case also.
    AddTestCase (new TcpLargeTransferLossTest (1000, 3294967, 3295100, "large-transfer-loss-with-wrap"), TestCase::EXTENSIVE);
  }
};

static TcpLossTestSuite g_tcpLossTest; //!< static var for test initialization
