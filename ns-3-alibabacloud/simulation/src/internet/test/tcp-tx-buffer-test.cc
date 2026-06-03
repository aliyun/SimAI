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

#include <limits>
#include "ns3/test.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/log.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpTxBufferTestSuite");

/**
 * \ingroup internet-tests
 * \ingroup tests
 *
 * \brief The TcpTxBuffer Test
 */
class TcpTxBufferTestCase : public TestCase
{
public:
  /** \brief Constructor */
  TcpTxBufferTestCase ();

private:
  virtual void DoRun (void);
  virtual void DoTeardown (void);

  /** \brief Test if a segment is really set as lost */
  void TestIsLost ();
  /** \brief Test the generation of an unsent block */
  void TestNewBlock ();
  /** \brief Test the generation of a previously sent block */
  void TestTransmittedBlock ();
  /** \brief Test the generation of the "next" block */
  void TestNextSeg ();
  /** \brief Test the logic of merging items in GetTransmittedSegment()
   * which is triggered by CopyFromSequence()*/
  void TestMergeItemsWhenGetTransmittedSegment ();
  /**
   * \brief Callback to provide a value of receiver window
   * \returns the receiver window size
   */
  uint32_t GetRWnd (void) const;
};

TcpTxBufferTestCase::TcpTxBufferTestCase ()
  : TestCase ("TcpTxBuffer Test")
{
}

void
TcpTxBufferTestCase::DoRun ()
{
  Simulator::Schedule (Seconds (0.0), &TcpTxBufferTestCase::TestIsLost, this);
  /*
   * Cases for new block:
   * -> is exactly the same as stored
   * -> starts over the boundary, but ends earlier
   * -> starts over the boundary, but ends after
   */
  Simulator::Schedule (Seconds (0.0), &TcpTxBufferTestCase::TestNewBlock, this);

  /*
   * Cases for transmitted block:
   * -> is exactly the same as previous
   * -> starts over the boundary, but ends earlier
   * -> starts over the boundary, but ends after
   * -> starts inside a packet, ends right
   * -> starts inside a packet, ends earlier in the same packet
   * -> starts inside a packet, ends in another packet
   */
  Simulator::Schedule (Seconds (0.0),
                       &TcpTxBufferTestCase::TestTransmittedBlock, this);
  Simulator::Schedule (Seconds (0.0),
                       &TcpTxBufferTestCase::TestNextSeg, this);

  /*
   * Case for transmitted block:
   *  -> transmitted packets are marked differently for m_lost under some scenarios
   *  -> packets could be small than MSS when socket buffer is not a multiple of MSS.
   *  -> during retransmission, the sender tries to send a full segment but it
   *     should stop to merge items when they have different values for m_lost.
   */
  Simulator::Schedule (Seconds (0.0),
                         &TcpTxBufferTestCase::TestMergeItemsWhenGetTransmittedSegment, this);

  Simulator::Run ();
  Simulator::Destroy ();
}

void
TcpTxBufferTestCase::TestIsLost ()
{
  Ptr<TcpTxBuffer> txBuf = CreateObject<TcpTxBuffer> ();
  txBuf->SetRWndCallback (MakeCallback (&TcpTxBufferTestCase::GetRWnd, this));
  SequenceNumber32 head (1);
  txBuf->SetHeadSequence (head);
  SequenceNumber32 ret;
  Ptr<TcpOptionSack> sack = CreateObject<TcpOptionSack> ();
  txBuf->SetSegmentSize (1000);
  txBuf->SetDupAckThresh (3);

  txBuf->Add(Create<Packet> (10000));

  for (uint8_t i = 0; i <10 ; ++i)
    txBuf->CopyFromSequence (1000, SequenceNumber32((i*1000)+1));

  for (uint8_t i = 0; i < 10 ; ++i)
    NS_TEST_ASSERT_MSG_EQ (txBuf->IsLost(SequenceNumber32((i*1000)+1)), false,
                           "Lost is true, but it's not");

  sack->AddSackBlock (TcpOptionSack::SackBlock (SequenceNumber32 (1001), SequenceNumber32 (2001)));
  txBuf->Update(sack->GetSackList());

  for (uint8_t i = 0; i < 10 ; ++i)
    NS_TEST_ASSERT_MSG_EQ (txBuf->IsLost(SequenceNumber32((i*1000)+1)), false,
                           "Lost is true, but it's not");

  sack->AddSackBlock (TcpOptionSack::SackBlock (SequenceNumber32 (2001), SequenceNumber32 (3001)));
  txBuf->Update(sack->GetSackList());

  for (uint8_t i = 0; i < 10 ; ++i)
    NS_TEST_ASSERT_MSG_EQ (txBuf->IsLost(SequenceNumber32((i*1000)+1)), false,
                           "Lost is true, but it's not");

  sack->AddSackBlock (TcpOptionSack::SackBlock (SequenceNumber32 (3001), SequenceNumber32 (4001)));
  txBuf->Update(sack->GetSackList());

  NS_TEST_ASSERT_MSG_EQ (txBuf->IsLost(SequenceNumber32(1)), true,
                         "Lost is true, but it's not");

  for (uint8_t i = 1; i < 10 ; ++i)
    NS_TEST_ASSERT_MSG_EQ (txBuf->IsLost(SequenceNumber32((i*1000)+1)), false,
                           "Lost is true, but it's not");


}

uint32_t
TcpTxBufferTestCase::GetRWnd (void) const
{
  // Assume unlimited receiver window
  return std::numeric_limits<uint32_t>::max ();
}

void
TcpTxBufferTestCase::TestNextSeg ()
{
  Ptr<TcpTxBuffer> txBuf = CreateObject<TcpTxBuffer> ();;
  txBuf->SetRWndCallback (MakeCallback (&TcpTxBufferTestCase::GetRWnd, this));
  SequenceNumber32 head (1);
  SequenceNumber32 ret;
  SequenceNumber32 retHigh;
  txBuf->SetSegmentSize (150);
  txBuf->SetDupAckThresh (3);
  uint32_t dupThresh = 3;
  uint32_t segmentSize = 150;
  Ptr<TcpOptionSack> sack = CreateObject<TcpOptionSack> ();

  // At the beginning the values of dupThresh and segmentSize don't matter
  NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), false,
                         "NextSeq should not be returned at the beginning");

  txBuf->SetHeadSequence (head);
  NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), false,
                         "NextSeq should not be returned with no data");

  // Add a single, 30000-bytes long, packet
  txBuf->Add (Create<Packet> (30000));
  NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                         "No NextSeq with data at beginning");
  NS_TEST_ASSERT_MSG_EQ (ret.GetValue (), head.GetValue (),
                         "Different NextSeq than expected at the beginning");

  // Simulate sending 100 packets, 150 bytes long each, from seq 1
  for (uint32_t i=0; i<100; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                             "No NextSeq with data while \"transmitting\"");
      NS_TEST_ASSERT_MSG_EQ (ret, head + (segmentSize * i),
                             "Different NextSeq than expected while \"transmitting\"");
      txBuf->CopyFromSequence (segmentSize, ret);
    }

  // Ok, now simulate we lost the first segment [1;151], and that we have
  // limited transmit. NextSeg should return (up to dupThresh-1) new pieces of data
  SequenceNumber32 lastRet = ret; // This is like m_highTx
  for (uint32_t i=1; i<dupThresh; ++i) // iterate dupThresh-1 times (limited transmit)
    {
      SequenceNumber32 begin = head + (segmentSize * i);
      SequenceNumber32 end = begin + segmentSize;
      sack->AddSackBlock (TcpOptionSack::SackBlock (begin, end));
      txBuf->Update (sack->GetSackList ());

      // new data expected and sent
      NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                             "No NextSeq with SACK block while \"transmitting\"");
      NS_TEST_ASSERT_MSG_EQ (ret, lastRet + segmentSize,
                             "Different NextSeq than expected in limited transmit");
      txBuf->CopyFromSequence (segmentSize, ret);
      sack->ClearSackList ();
      lastRet = ret;
    }

  // Limited transmit was ok; now there is the dupThresh-th dupack.
  // Now we need to retransmit the first block..
  sack->AddSackBlock (TcpOptionSack::SackBlock (head + (segmentSize * (dupThresh)),
                                                head + (segmentSize * (dupThresh)) + segmentSize));
  txBuf->Update (sack->GetSackList ());
  NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                         "No NextSeq with SACK block for Fast Recovery");
  NS_TEST_ASSERT_MSG_EQ (ret, head,
                         "Different NextSeq than expected for Fast Recovery");
  txBuf->CopyFromSequence (segmentSize, ret);
  sack->ClearSackList ();

  // Fast Retransmission was ok; now check some additional dupacks.
  for (uint32_t i=1; i<=4; ++i)
    {
      sack->AddSackBlock (TcpOptionSack::SackBlock (head + (segmentSize * (dupThresh+i)),
                                                    head + (segmentSize * (dupThresh+i)) + segmentSize));
      txBuf->Update (sack->GetSackList ());
      NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                             "No NextSeq with SACK block after recv dupacks in FR");
      NS_TEST_ASSERT_MSG_EQ (ret, lastRet + segmentSize,
                             "Different NextSeq than expected after recv dupacks in FR");
      txBuf->CopyFromSequence (segmentSize, ret);
      sack->ClearSackList ();
      lastRet = ret;
    }

  // Well now we receive a partial ACK, corresponding to the segment we retransmitted.
  // Unfortunately, the next one is lost as well; but NextSeg should be smart enough
  // to give us the next segment (head + segmentSize) to retransmit.
  /* In this particular case, we are checking the fact that we have badly crafted
   * the SACK blocks. Talking in segment, we transmitted 1,2,3,4,5 ... and then
   * received dupack for 1. While receiving these, we crafted SACK block in the
   * way that 2,3,4,... were correctly received. Now, if we receive an ACK for 2,
   * we clearly crafted the corresponding ACK wrongly. TcpTxBuffer should be able
   * to "backoff" that flag on its HEAD (segment 2). We still don't know for segment
   * 3,4 .. so keep them.
   */
  head = head + segmentSize;
  txBuf->DiscardUpTo (head);

  NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                         "No NextSeq with SACK block after receiving partial ACK");
  NS_TEST_ASSERT_MSG_EQ (ret, head,
                         "Different NextSeq than expected after receiving partial ACK ");
  txBuf->CopyFromSequence (segmentSize, ret);

  // Now, check for one more dupack...
  sack->AddSackBlock (TcpOptionSack::SackBlock (head + (segmentSize * (dupThresh+6)),
                                                head + (segmentSize * (dupThresh+6)) + segmentSize));
  txBuf->Update (sack->GetSackList ());
  NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                         "No NextSeq with SACK block after recv dupacks after partial ack");
  NS_TEST_ASSERT_MSG_EQ (ret, lastRet + segmentSize,
                         "Different NextSeq than expected after recv dupacks after partial ack");
  txBuf->CopyFromSequence (segmentSize, ret);
  sack->ClearSackList ();
  head = lastRet = ret + segmentSize;

  // And now ack everything we sent to date!
  txBuf->DiscardUpTo (head);

  // And continue normally until the end
  for (uint32_t i=0; i<93; ++i)
    {
      NS_TEST_ASSERT_MSG_EQ (txBuf->NextSeg (&ret, &retHigh, false), true,
                             "No NextSeq with data while \"transmitting\"");
      NS_TEST_ASSERT_MSG_EQ (ret, head + (segmentSize * i),
                             "Different NextSeq than expected while \"transmitting\"");
      txBuf->CopyFromSequence (segmentSize, ret);
    }

  txBuf->DiscardUpTo (ret+segmentSize);
  NS_TEST_ASSERT_MSG_EQ (txBuf->Size (), 0,
                         "Data inside the buffer");
}

void
TcpTxBufferTestCase::TestNewBlock ()
{
  // Manually recreating all the conditions
  Ptr<TcpTxBuffer> txBuf = CreateObject<TcpTxBuffer> ();
  txBuf->SetRWndCallback (MakeCallback (&TcpTxBufferTestCase::GetRWnd, this));
  txBuf->SetHeadSequence (SequenceNumber32 (1));
  txBuf->SetSegmentSize (100);

  // get a packet which is exactly the same stored
  Ptr<Packet> p1 = Create<Packet> (100);
  txBuf->Add (p1);

  NS_TEST_ASSERT_MSG_EQ (txBuf->SizeFromSequence (SequenceNumber32 (1)), 100,
                         "TxBuf miscalculates size");
  NS_TEST_ASSERT_MSG_EQ (txBuf->BytesInFlight (), 0,
                         "TxBuf miscalculates size of in flight segments");

  Ptr<Packet> ret = txBuf->CopyFromSequence (100, SequenceNumber32 (1))->GetPacketCopy ();
  NS_TEST_ASSERT_MSG_EQ (ret->GetSize (), 100,
                         "Returned packet has different size than requested");
  NS_TEST_ASSERT_MSG_EQ (txBuf->SizeFromSequence (SequenceNumber32 (1)), 100,
                         "TxBuf miscalculates size");
  NS_TEST_ASSERT_MSG_EQ (txBuf->BytesInFlight (), 100,
                         "TxBuf miscalculates size of in flight segments");

  txBuf->DiscardUpTo (SequenceNumber32 (101));
  NS_TEST_ASSERT_MSG_EQ (txBuf->SizeFromSequence (SequenceNumber32 (101)), 0,
                         "TxBuf miscalculates size");
  NS_TEST_ASSERT_MSG_EQ (txBuf->BytesInFlight (), 0,
                         "TxBuf miscalculates size of in flight segments");

  // starts over the boundary, but ends earlier

  Ptr<Packet> p2 = Create<Packet> (100);
  txBuf->Add (p2);

  ret = txBuf->CopyFromSequence (50, SequenceNumber32 (101))->GetPacketCopy ();
  NS_TEST_ASSERT_MSG_EQ (ret->GetSize (), 50,
                         "Returned packet has different size than requested");
  NS_TEST_ASSERT_MSG_EQ (txBuf->SizeFromSequence (SequenceNumber32 (151)), 50,
                         "TxBuf miscalculates size");
  NS_TEST_ASSERT_MSG_EQ (txBuf->BytesInFlight (), 50,
                         "TxBuf miscalculates size of in flight segments");

  // starts over the boundary, but ends after
  Ptr<Packet> p3 = Create<Packet> (100);
  txBuf->Add (p3);

  ret = txBuf->CopyFromSequence (70, SequenceNumber32 (151))->GetPacketCopy ();
  NS_TEST_ASSERT_MSG_EQ (ret->GetSize (), 70,
                         "Returned packet has different size than requested");
  NS_TEST_ASSERT_MSG_EQ (txBuf->SizeFromSequence (SequenceNumber32 (221)), 80,
                         "TxBuf miscalculates size");
  NS_TEST_ASSERT_MSG_EQ (txBuf->BytesInFlight (), 120,
                         "TxBuf miscalculates size of in flight segments");

  ret = txBuf->CopyFromSequence (3000, SequenceNumber32 (221))->GetPacketCopy ();
  NS_TEST_ASSERT_MSG_EQ (ret->GetSize (), 80,
                         "Returned packet has different size than requested");
  NS_TEST_ASSERT_MSG_EQ (txBuf->SizeFromSequence (SequenceNumber32 (301)), 0,
                         "TxBuf miscalculates size");
  NS_TEST_ASSERT_MSG_EQ (txBuf->BytesInFlight (), 200,
                         "TxBuf miscalculates size of in flight segments");

  // Clear everything
  txBuf->DiscardUpTo (SequenceNumber32 (381));
  NS_TEST_ASSERT_MSG_EQ (txBuf->Size (), 0,
                         "Size is different than expected");
}

void
TcpTxBufferTestCase::TestMergeItemsWhenGetTransmittedSegment ()
{
  TcpTxBuffer txBuf;
  SequenceNumber32 head (1);
  txBuf.SetHeadSequence (head);
  txBuf.SetSegmentSize (2000);

  txBuf.Add(Create<Packet> (2000));
  txBuf.CopyFromSequence (1000, SequenceNumber32(1));
  txBuf.CopyFromSequence (1000, SequenceNumber32(1001));
  txBuf.MarkHeadAsLost();

  // GetTransmittedSegment() will be called and handle the case that two items
  // have different m_lost value.
  txBuf.CopyFromSequence (2000, SequenceNumber32(1));
}

void
TcpTxBufferTestCase::TestTransmittedBlock ()
{
}

void
TcpTxBufferTestCase::DoTeardown ()
{
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief the TestSuite for the TcpTxBuffer test case
 */
class TcpTxBufferTestSuite : public TestSuite
{
public:
  TcpTxBufferTestSuite ()
    : TestSuite ("tcp-tx-buffer", UNIT)
  {
    AddTestCase (new TcpTxBufferTestCase, TestCase::QUICK);
  }
};

static TcpTxBufferTestSuite  g_tcpTxBufferTestSuite; //!< Static variable for test initialization
