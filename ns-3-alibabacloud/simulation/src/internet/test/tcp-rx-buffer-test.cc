/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
#include "ns3/packet.h"
#include "ns3/log.h"

#include "ns3/tcp-rx-buffer.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpRxBufferTestSuite");

/**
 * \ingroup internet-tests
 * \ingroup tests
 *
 * \brief The TcpRxBuffer Test
 */
class TcpRxBufferTestCase : public TestCase
{
public:
  TcpRxBufferTestCase ();

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  /**
   * \brief Test the SACK list update.
   */
  void TestUpdateSACKList ();
};

TcpRxBufferTestCase::TcpRxBufferTestCase ()
  : TestCase ("TcpRxBuffer Test")
{
}

void
TcpRxBufferTestCase::DoRun ()
{
  TestUpdateSACKList ();
}

void
TcpRxBufferTestCase::TestUpdateSACKList ()
{
  TcpRxBuffer rxBuf;
  TcpOptionSack::SackList sackList;
  TcpOptionSack::SackList::iterator it;
  Ptr<Packet> p = Create<Packet> (100);
  TcpHeader h;

  // In order sequence
  h.SetSequenceNumber (SequenceNumber32 (1));
  rxBuf.SetNextRxSequence (SequenceNumber32 (1));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (101),
                         "Sequence number differs from expected");
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 0,
                         "SACK list with an element, while should be empty");


  // Out-of-order sequence (SACK generated)
  h.SetSequenceNumber (SequenceNumber32 (501));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (101),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 1,
                         "SACK list should contain one element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (501),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (601),
                         "SACK block different than expected");

  // In order sequence, not greater than the previous (the old SACK still in place)
  h.SetSequenceNumber (SequenceNumber32 (101));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (201),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 1,
                         "SACK list should contain one element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (501),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (601),
                         "SACK block different than expected");

  // Out of order sequence, merge on the right
  h.SetSequenceNumber (SequenceNumber32 (401));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (201),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 1,
                         "SACK list should contain one element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (601),
                         "SACK block different than expected");

  // Out of order sequence, merge on the left
  h.SetSequenceNumber (SequenceNumber32 (601));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (201),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 1,
                         "SACK list should contain one element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (701),
                         "SACK block different than expected");

  // out of order sequence, different block, check also the order (newer first)
  h.SetSequenceNumber (SequenceNumber32 (901));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (201),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 2,
                         "SACK list should contain two element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (901),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1001),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (701),
                         "SACK block different than expected");

  // another out of order seq, different block, check the order (newer first)
  h.SetSequenceNumber (SequenceNumber32 (1201));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (201),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 3,
                         "SACK list should contain three element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1301),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (901),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1001),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (701),
                         "SACK block different than expected");

  // another out of order seq, different block, check the order (newer first)
  h.SetSequenceNumber (SequenceNumber32 (1401));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (201),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 4,
                         "SACK list should contain four element");
  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1501),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1301),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (901),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1001),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (701),
                         "SACK block different than expected");

  // in order block! See if something get stripped off..
  h.SetSequenceNumber (SequenceNumber32 (201));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (301),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 4,
                         "SACK list should contain four element");

  // in order block! See if something get stripped off..
  h.SetSequenceNumber (SequenceNumber32 (301));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (701),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 3,
                         "SACK list should contain three element");

  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1501),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1301),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (901),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1001),
                         "SACK block different than expected");

  // out of order block, I'm expecting a left-merge with a move on the top
  h.SetSequenceNumber (SequenceNumber32 (801));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (701),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 3,
                         "SACK list should contain three element");

  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (801),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1001),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1501),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1301),
                         "SACK block different than expected");

  // In order block! Strip things away..
  h.SetSequenceNumber (SequenceNumber32 (701));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (1001),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 2,
                         "SACK list should contain two element");

  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1401),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1501),
                         "SACK block different than expected");
  ++it;
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1301),
                         "SACK block different than expected");

  // out of order... I'm expecting a right-merge with a move on top
  h.SetSequenceNumber (SequenceNumber32 (1301));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (1001),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 1,
                         "SACK list should contain one element");

  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1501),
                         "SACK block different than expected");

  // In order
  h.SetSequenceNumber (SequenceNumber32 (1001));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (1101),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 1,
                         "SACK list should contain one element");

  it = sackList.begin ();
  NS_TEST_ASSERT_MSG_EQ (it->first, SequenceNumber32 (1201),
                         "SACK block different than expected");
  NS_TEST_ASSERT_MSG_EQ (it->second, SequenceNumber32 (1501),
                         "SACK block different than expected");

  // In order, empty the list
  h.SetSequenceNumber (SequenceNumber32 (1101));
  rxBuf.Add (p, h);

  NS_TEST_ASSERT_MSG_EQ (rxBuf.NextRxSequence (), SequenceNumber32 (1501),
                         "Sequence number differs from expected");
  sackList = rxBuf.GetSackList ();
  NS_TEST_ASSERT_MSG_EQ (sackList.size (), 0,
                         "SACK list should contain no element");
}

void
TcpRxBufferTestCase::DoTeardown ()
{
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief the TestSuite for the TcpRxBuffer test case
 */
class TcpRxBufferTestSuite : public TestSuite
{
public:
  TcpRxBufferTestSuite ()
    : TestSuite ("tcp-rx-buffer", UNIT)
  {
    AddTestCase (new TcpRxBufferTestCase, TestCase::QUICK);
  }
};
static TcpRxBufferTestSuite  g_tcpRxBufferTestSuite;
