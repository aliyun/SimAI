/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Natale Patriciello <natale.patriciello@gmail.com>
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
#include "ns3/log.h"
#include "ns3/tcp-westwood.h"
#include "tcp-general-test.h"
#include "ns3/simple-channel.h"
#include "ns3/node.h"
#include "tcp-error-model.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpFastRetrTest");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Test the fast retransmission
 *
 * Checking what is happening is not so easy, so there are a lot of variables
 * which helps to keep track on what is happening.
 * The idea is following sequence and ack numbers which are exchanged,
 * testing if they are the same as the implementation transmits.
 */
class TcpFastRetrTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   * \param congControl Type of congestion control.
   * \param seqToKill Sequence number of the packet to drop.
   * \param msg Test message.
   */
  TcpFastRetrTest (TypeId congControl, uint32_t seqToKill, const std::string &msg);

  virtual Ptr<ErrorModel> CreateSenderErrorModel ();
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();

  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);

protected:
  virtual void RcvAck      (const Ptr<const TcpSocketState> tcb,
                            const TcpHeader& h, SocketWho who);
  virtual void ProcessedAck (const Ptr<const TcpSocketState> tcb,
                             const TcpHeader& h, SocketWho who);

  virtual void CongStateTrace (const TcpSocketState::TcpCongState_t oldValue,
                               const TcpSocketState::TcpCongState_t newValue);

  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);

  virtual void AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who);

  /**
   * \brief Check if the packet being dropped is the right one.
   * \param ipH IPv4 header.
   * \param tcpH TCP header.
   * \param p The packet.
   */
  void PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> p);
  virtual void FinalChecks ();

  virtual void ConfigureProperties ();
  virtual void ConfigureEnvironment ();

  bool m_pktDropped;      //!< The packet has been dropped.
  bool m_pktWasDropped;   //!< The packet was dropped (according to the receiver).
  uint32_t m_seqToKill;   //!< Sequence number to drop.
  uint32_t m_dupAckReceived;  //!< DipACk received.

  SequenceNumber32 m_previousAck;     //!< Previous ACK received.
  SequenceNumber32 m_sndNextExpSeq;   //!< Sender next expected sequence number.
  SequenceNumber32 m_rcvNextExpAck;   //!< Receiver next expected sequence number.

  uint32_t m_countRetr; //!< Retry counter.

  uint32_t m_bytesRcvButNotAcked; //!< Number of bytes received but not acked.

  Ptr<TcpSeqErrorModel> m_errorModel; //!< Error model.
};

TcpFastRetrTest::TcpFastRetrTest (TypeId typeId, uint32_t seqToKill,
                                  const std::string &msg)
  : TcpGeneralTest (msg),
    m_pktDropped (false),
    m_pktWasDropped (false),
    m_seqToKill (seqToKill),
    m_dupAckReceived (0),
    m_sndNextExpSeq (0),
    m_rcvNextExpAck (1),
    m_countRetr (0),
    m_bytesRcvButNotAcked (0)
{
  m_congControlTypeId = typeId;
}

void
TcpFastRetrTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetInitialSsThresh (SENDER, 0);
}

void
TcpFastRetrTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (100);
}

Ptr<ErrorModel>
TcpFastRetrTest::CreateSenderErrorModel ()
{
  return 0;
}

Ptr<ErrorModel>
TcpFastRetrTest::CreateReceiverErrorModel ()
{
  m_errorModel = CreateObject<TcpSeqErrorModel> ();
  m_errorModel->AddSeqToKill (SequenceNumber32 (m_seqToKill));
  m_errorModel->SetDropCallback (MakeCallback (&TcpFastRetrTest::PktDropped, this));

  return m_errorModel;
}


Ptr<TcpSocketMsgBase>
TcpFastRetrTest::CreateSenderSocket (Ptr<Node> node)
{
  Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket (node);
  socket->SetAttribute ("MinRto", TimeValue (Seconds (10.0)));

  return socket;
}

void
TcpFastRetrTest::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      // Nothing to check
      NS_LOG_INFO ("\tSENDER Rx " << h);
    }
  else if (who == RECEIVER)
    {
      NS_LOG_INFO ("\tRECEIVER Rx " << h);

      // Receiver has received the missing segment
      if (h.GetSequenceNumber ().GetValue () == m_seqToKill)
        {
          m_pktDropped = false;
          if (m_bytesRcvButNotAcked > 0)
            {
              m_rcvNextExpAck += m_bytesRcvButNotAcked + GetSegSize (SENDER);
              m_bytesRcvButNotAcked = 0;
            }
        }

      // Count all the received bytes not acked
      if (m_pktDropped)
        {
          m_bytesRcvButNotAcked += GetSegSize (SENDER);
        }
    }
}

void
TcpFastRetrTest::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      NS_LOG_INFO ("\tSENDER Tx " << h << " size=" << p->GetSize ());

      if (h.GetSequenceNumber ().GetValue () == m_seqToKill && m_pktDropped)
        {
          // Spotted the retransmission!
          m_countRetr++;
          NS_TEST_ASSERT_MSG_EQ (m_countRetr, 1,
                                 "Segment retransmitted too many times");
        }
      else
        {
          // No delayed ACK involved here.
          while (h.GetSequenceNumber () < m_sndNextExpSeq)
            {
              m_sndNextExpSeq -= GetSegSize (SENDER);
            }

          if (h.GetSequenceNumber ().GetValue () != 50002)
            {
              NS_TEST_ASSERT_MSG_EQ (m_sndNextExpSeq, h.GetSequenceNumber (),
                                     "Sequence number expected differs");
            }
        }

      if (m_sndNextExpSeq.GetValue () == 0)
        {
          // SYN
          m_sndNextExpSeq = SequenceNumber32 (1);
        }
      else if (m_sndNextExpSeq.GetValue () == 1 && p->GetSize () == 32)
        {
          // Pure ACK in three-way handshake, then we expect data
          m_sndNextExpSeq = SequenceNumber32 (1);
        }
      else
        {
          // Data segments
          m_sndNextExpSeq += GetSegSize (SENDER);
        }
    }
  else if (who == RECEIVER)
    {
      NS_LOG_INFO ("\tRECEIVER Tx, " << h << " size=" << p->GetSize ());

      if (h.GetFlags () == (TcpHeader::SYN | TcpHeader::ACK))
        {
          NS_TEST_ASSERT_MSG_EQ (h.GetSequenceNumber ().GetValue (), 0,
                                 "SYN pkt has not 0 as initial sequence number."
                                 "Probably, random sqn number has been implemented."
                                 "Check this test");
        }
      else
        {
          NS_TEST_ASSERT_MSG_EQ (h.GetSequenceNumber ().GetValue (), 1,
                                 "ACK pkt has not 1 as sequence number."
                                 "Probably, random sqn number has been implemented."
                                 "Check this test");
        }

      // Accounted for delayed ACK, but not received.
      while (h.GetAckNumber () < m_rcvNextExpAck)
        {
          m_rcvNextExpAck -= GetSegSize (SENDER);
        }

      if (m_rcvNextExpAck.GetValue () >= 50001)
        {
          m_rcvNextExpAck = 50002;
        }

      NS_TEST_ASSERT_MSG_EQ (h.GetAckNumber (), m_rcvNextExpAck,
                             "ACKing something not considered");

      if (m_pktDropped)
        {
          m_rcvNextExpAck = SequenceNumber32 (m_seqToKill);
        }
      else
        {
          switch (m_rcvNextExpAck.GetValue ())
            {
            case 0:
              m_rcvNextExpAck = SequenceNumber32 (1);
              break;
            case 1:
              m_rcvNextExpAck += GetSegSize (SENDER);
              break;
            case 50002:
              break;
            default:
              m_rcvNextExpAck += GetSegSize (SENDER) * GetDelAckCount (SENDER);
            }
        }
    }
}

void
TcpFastRetrTest::RcvAck (const Ptr<const TcpSocketState> tcb, const TcpHeader &h,
                         SocketWho who)
{
  NS_LOG_FUNCTION (this << tcb << h << who);

  if (who == SENDER)
    {
      if (h.GetAckNumber ().GetValue () < m_seqToKill)
        {
          NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_OPEN,
                                 "Not in OPEN state to respond to a loss");
          NS_TEST_ASSERT_MSG_EQ (GetDupAckCount (SENDER), 0,
                                 "Dupack different than 0 but no loss detected");
        }
      else if (h.GetAckNumber ().GetValue () == m_seqToKill)
        {
          NS_TEST_ASSERT_MSG_EQ (GetDupAckCount (SENDER), m_dupAckReceived,
                                 "Dupack count differs");

          if (GetDupAckCount (SENDER) == 0 &&
              GetDupAckCount (SENDER) < GetReTxThreshold (SENDER))
            {
              NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_OPEN,
                                     "Not in OPEN state for processing dupack");
            }
          else if (GetDupAckCount (SENDER) > 0 &&
                   GetDupAckCount (SENDER) < GetReTxThreshold (SENDER))
            {
              NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_DISORDER,
                                     "Not in DISORDER state after receiving dupacks");
            }
          else if (GetDupAckCount (SENDER) >= GetReTxThreshold (SENDER))
            {
              NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_RECOVERY,
                                     "Not in RECOVERY state after reaching retxthresh");
            }
        }
    }
  else if (who == RECEIVER)
    {
      NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_OPEN,
                             "Receiver not in OPEN state");
    }
}

void
TcpFastRetrTest::ProcessedAck (const Ptr<const TcpSocketState> tcb, const TcpHeader &h,
                               SocketWho who)
{
  NS_LOG_FUNCTION (this << tcb << h << who);

  if (who == SENDER)
    {
      if (m_previousAck == h.GetAckNumber () && h.GetAckNumber ().GetValue () < 50002)
        {
          m_dupAckReceived++;

          NS_TEST_ASSERT_MSG_GT_OR_EQ (m_dupAckReceived, GetDupAckCount (SENDER),
                                       "Count of dupAck differs");

          if (GetDupAckCount (SENDER) < GetReTxThreshold (SENDER))
            {
              NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_DISORDER,
                                     "DupAck less than ReTxThreshold but not "
                                     "in DISORDER state");
            }
          else
            {
              NS_TEST_ASSERT_MSG_GT_OR_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_RECOVERY,
                                           "DupAck greater than ReTxThreshold but not "
                                           "in RECOVERY or LOSS state");
              m_pktWasDropped = true;
            }
        }
      else if (m_previousAck < h.GetAckNumber ())
        {
          m_dupAckReceived = 0;
        }

      m_previousAck = h.GetAckNumber ();
    }
  else if (who == RECEIVER)
    {
      NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_OPEN,
                             "Different state than OPEN in the receiver");
    }
}

void
TcpFastRetrTest::AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who)
{
  NS_TEST_ASSERT_MSG_EQ (true, false, "RTO isn't expected here");
}

void
TcpFastRetrTest::CongStateTrace (const TcpSocketState::TcpCongState_t oldValue,
                                 const TcpSocketState::TcpCongState_t newValue)
{
  NS_LOG_FUNCTION (this << oldValue << newValue);

  if (oldValue == TcpSocketState::CA_OPEN && newValue == TcpSocketState::CA_DISORDER)
    {
    }
  else if (oldValue == TcpSocketState::CA_OPEN
           && newValue == TcpSocketState::CA_RECOVERY
           && GetReTxThreshold (SENDER) > 1)
    {
      NS_TEST_ASSERT_MSG_EQ (true, false,
                             "Invalid OPEN to RECOVERY state change");
    }
  else if (oldValue == TcpSocketState::CA_DISORDER
           && newValue == TcpSocketState::CA_RECOVERY)
    {
      NS_TEST_ASSERT_MSG_EQ (GetReTxThreshold (SENDER), GetDupAckCount (SENDER),
                             "DISORDER to RECOVERY state change but not reached "
                             "the ReTxThreshold");
    }
}


void
TcpFastRetrTest::PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> p)
{
  NS_LOG_FUNCTION (this << ipH << tcpH);

  m_pktDropped = true;
  m_rcvNextExpAck = tcpH.GetSequenceNumber ();

  NS_TEST_ASSERT_MSG_EQ (tcpH.GetSequenceNumber (), SequenceNumber32 (m_seqToKill),
                         "Packet dropped but sequence number differs");
}

void
TcpFastRetrTest::FinalChecks ()
{
  NS_TEST_ASSERT_MSG_EQ (m_pktWasDropped, true,
                         "Packet was not dropped at all");
  NS_TEST_ASSERT_MSG_EQ (m_countRetr, 1,
                         "Segment was not retransmitted at all");
  NS_TEST_ASSERT_MSG_EQ (m_rcvNextExpAck.GetValue (), 50002,
                         "Not all data have been transmitted");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testsuite for the fast retransmission
 */
class TcpFastRetrTestSuite : public TestSuite
{
public:
  TcpFastRetrTestSuite () : TestSuite ("tcp-fast-retr-test", UNIT)
  {
    std::list<TypeId> types;
    types.insert (types.begin (), TcpWestwood::GetTypeId ());
    types.insert (types.begin (), TcpNewReno::GetTypeId ());

    for (std::list<TypeId>::iterator it = types.begin (); it != types.end (); ++it)
      {
        AddTestCase (new TcpFastRetrTest ((*it), 5001, "Fast Retransmit testing"), TestCase::QUICK);
      }
  }
};

static TcpFastRetrTestSuite g_TcpFastRetrTestSuite; //!< Static variable for test initialization

