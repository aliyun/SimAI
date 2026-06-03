/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 NITK Surathkal
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
 * Author: Shravya K.S. <shravya.ks0@gmail.com>
 *
 */

#include "ns3/ipv4.h"
#include "ns3/ipv6.h"
#include "ns3/ipv4-end-point.h"
#include "ns3/ipv6-end-point.h"
#include "tcp-general-test.h"
#include "ns3/node.h"
#include "ns3/log.h"
#include "tcp-error-model.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-dctcp.h"
#include "ns3/tcp-linux-reno.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/config.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpDctcpTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Validates the setting of ECT and ECE codepoints for DCTCP enabled traffic
 */
class TcpDctcpCodePointsTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   *
   * \param testCase Test case number
   * \param desc Description about the test
   */
  TcpDctcpCodePointsTest (uint8_t testCase, const std::string &desc);

protected:
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);
  virtual Ptr<TcpSocketMsgBase> CreateReceiverSocket (Ptr<Node> node);
  void ConfigureProperties ();
  void ConfigureEnvironment ();

private:
  uint32_t m_senderSent;      //!< Number of packets sent by the sender
  uint32_t m_receiverSent;    //!< Number of packets sent by the receiver
  uint32_t m_senderReceived;  //!< Number of packets received by the sender
  uint8_t m_testCase;         //!< Test type
};

TcpDctcpCodePointsTest::TcpDctcpCodePointsTest (uint8_t testCase, const std::string &desc)
  : TcpGeneralTest (desc),
    m_senderSent (0),
    m_receiverSent (0),
    m_senderReceived (0),
    m_testCase (testCase)
{
}

void
TcpDctcpCodePointsTest::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  bool foundTag = false;  // IpTosTag will only be found if ECN bits are set
  if (who == SENDER && (m_testCase == 1 || m_testCase == 2))
    {
      m_senderSent++;
      SocketIpTosTag ipTosTag;
      foundTag = p->PeekPacketTag (ipTosTag);
      if (m_testCase == 1)
        {
          if (m_senderSent == 1)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, true, "Tag not found");
              NS_TEST_ASSERT_MSG_EQ (unsigned (ipTosTag.GetTos ()), 0x1, "IP TOS should have ECT1 for SYN packet for DCTCP traffic");
            }
          if (m_senderSent == 3)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, true, "Tag not found");
              NS_TEST_ASSERT_MSG_EQ (unsigned (ipTosTag.GetTos ()), 0x1, "IP TOS should have ECT1 for data packets for DCTCP traffic");
            }
        }
      else
        {
          if (m_senderSent == 1)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, false, "IP TOS should not have ECT1 for SYN packet for DCTCP traffic");
            }
          if (m_senderSent == 3)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, true, "Tag not found");
              NS_TEST_ASSERT_MSG_EQ (unsigned (ipTosTag.GetTos ()), 0x2, "IP TOS should have ECT0 for data packets for non-DCTCP but ECN enabled traffic");
            }
        }
    }
  else if (who == RECEIVER && (m_testCase == 1 || m_testCase == 2))
    {
      m_receiverSent++;
      SocketIpTosTag ipTosTag;
      foundTag = p->PeekPacketTag (ipTosTag);
      if (m_testCase == 1)
        {
          if (m_receiverSent == 1)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, true, "Tag not found");
              NS_TEST_ASSERT_MSG_EQ (unsigned (ipTosTag.GetTos ()), 0x1, "IP TOS should have ECT1 for SYN+ACK packet for DCTCP traffic");
            }
          if (m_receiverSent == 2)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, true, "Tag not found");
              NS_TEST_ASSERT_MSG_EQ (unsigned (ipTosTag.GetTos ()), 0x1, "IP TOS should have ECT1 for pure ACK packets for DCTCP traffic");
            }
        }
      else
        {
          if (m_receiverSent == 1)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, false, "IP TOS should have neither ECT0 nor ECT1 for SYN+ACK packet for non-DCTCP traffic");
            }
          if (m_receiverSent == 2)
            {
              NS_TEST_ASSERT_MSG_EQ (foundTag, false, "IP TOS should not have ECT1 for pure ACK packets for non-DCTCP traffic but ECN enabled traffic");
            }
        }
    }
}

void
TcpDctcpCodePointsTest::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER && m_testCase == 3)
    {
      m_senderReceived++;
      if (m_senderReceived == 2 && m_testCase == 3)
        {
          NS_TEST_ASSERT_MSG_NE (((h.GetFlags ()) & TcpHeader::ECE), 0, "The flag ECE should be set in TCP header of the packet sent by the receiver when it receives a packet with CE bit set in IP header");
        }
      if (m_senderReceived > 2 && m_testCase == 3)
        {
          NS_TEST_ASSERT_MSG_EQ (((h.GetFlags ()) & TcpHeader::ECE), 0, "The flag ECE should be not be set in TCP header of the packet sent by the receiver if it receives a packet without CE bit set in IP header inspite of Sender not sending CWR flags to it");
        }
    }
}

void
TcpDctcpCodePointsTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetUseEcn (SENDER, TcpSocketState::On);
  SetUseEcn (RECEIVER, TcpSocketState::On);
}

void
TcpDctcpCodePointsTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  Config::SetDefault ("ns3::TcpDctcp::UseEct0", BooleanValue (false));
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief A TCP socket which sends a data packet with CE flags set for test 3.
 *
 * The SendDataPacket function of this class sends data packet numbered 1  with CE flags set and also doesn't set
 * CWR flags on receipt of ECE flags for test 3. This is done to verify that DCTCP receiver sends ECE only if it
 * receives CE inspite of sender not sending CWR flags for ECE
 *
 */
class TcpDctcpCongestedRouter : public TcpSocketMsgBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  uint32_t m_dataPacketSent;  //!< Number of packets sent
  uint8_t m_testCase;         //!< Test type

  TcpDctcpCongestedRouter ()
    : TcpSocketMsgBase ()
  {
    m_dataPacketSent = 0;
  }

  /**
   * \brief Constructor.
   * \param other The object to copy from.
   */
  TcpDctcpCongestedRouter (const TcpDctcpCongestedRouter &other)
    : TcpSocketMsgBase (other)
  {
  }

  /**
   * Set the test case type
   * \param testCase test case type
   */
  void SetTestCase (uint8_t testCase);
protected:
  virtual uint32_t SendDataPacket (SequenceNumber32 seq, uint32_t maxSize, bool withAck);
  virtual void ReTxTimeout ();
  Ptr<TcpSocketBase> Fork (void);
};

NS_OBJECT_ENSURE_REGISTERED (TcpDctcpCongestedRouter);

TypeId
TcpDctcpCongestedRouter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpDctcpCongestedRouter")
    .SetParent<TcpSocketMsgBase> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpDctcpCongestedRouter> ()
  ;
  return tid;
}

void
TcpDctcpCongestedRouter::ReTxTimeout ()
{
  TcpSocketBase::ReTxTimeout ();
}

void
TcpDctcpCongestedRouter::SetTestCase (uint8_t testCase)
{
  m_testCase = testCase;
}

uint32_t
TcpDctcpCongestedRouter::SendDataPacket (SequenceNumber32 seq, uint32_t maxSize, bool withAck)
{
  NS_LOG_FUNCTION (this << seq << maxSize << withAck);
  m_dataPacketSent++;

  bool isRetransmission = false;
  if (seq != m_tcb->m_highTxMark)
    {
      isRetransmission = true;
    }

  Ptr<Packet> p = m_txBuffer->CopyFromSequence (maxSize, seq)->GetPacketCopy ();
  uint32_t sz = p->GetSize (); // Size of packet
  uint8_t flags = withAck ? TcpHeader::ACK : 0;
  uint32_t remainingData = m_txBuffer->SizeFromSequence (seq + SequenceNumber32 (sz));

  if (withAck)
    {
      m_delAckEvent.Cancel ();
      m_delAckCount = 0;
    }

  // For test 3, we don't send CWR flags on receipt of ECE to check if Receiver sends ECE only when there is CE flags
  if (m_tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD && m_ecnEchoSeq.Get () > m_ecnCWRSeq.Get () && !isRetransmission && m_testCase != 3)
    {
      NS_LOG_INFO ("Backoff mechanism by reducing CWND  by half because we've received ECN Echo");
      m_tcb->m_ssThresh = m_tcb->m_cWnd;
      flags |= TcpHeader::CWR;
      m_ecnCWRSeq = seq;
      m_tcb->m_ecnState = TcpSocketState::ECN_CWR_SENT;
      NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CWR_SENT");
      NS_LOG_INFO ("CWR flags set");
      NS_LOG_DEBUG (TcpSocketState::TcpCongStateName[m_tcb->m_congState] << " -> CA_CWR");
      if (m_tcb->m_congState == TcpSocketState::CA_OPEN)
        {
          m_congestionControl->CongestionStateSet (m_tcb, TcpSocketState::CA_CWR);
          m_tcb->m_congState = TcpSocketState::CA_CWR;
        }
    }
  /*
   * Add tags for each socket option.
   * Note that currently the socket adds both IPv4 tag and IPv6 tag
   * if both options are set. Once the packet got to layer three, only
   * the corresponding tags will be read.
   */
  if (GetIpTos ())
    {
      SocketIpTosTag ipTosTag;

      NS_LOG_LOGIC (" ECT bits should not be set on retransmitted packets ");
      if (m_testCase == 3 && m_dataPacketSent == 1  && !isRetransmission)
        {
          ipTosTag.SetTos (GetIpTos () | 0x3);
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && (GetIpTos () & 0x3) == 0 && !isRetransmission)
            {
              ipTosTag.SetTos (GetIpTos () | 0x1);
            }
          else
            {
              ipTosTag.SetTos (GetIpTos ());
            }
        }
      p->AddPacketTag (ipTosTag);
    }
  else
    {
      SocketIpTosTag ipTosTag;
      if (m_testCase == 3 && m_dataPacketSent == 1 && !isRetransmission)
        {
          ipTosTag.SetTos (0x3);
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && !isRetransmission)
            {
              ipTosTag.SetTos (0x1);
            }
        }
      p->AddPacketTag (ipTosTag);
    }

  if (IsManualIpv6Tclass ())
    {
      SocketIpv6TclassTag ipTclassTag;
      if (m_testCase == 3 && m_dataPacketSent == 1 && !isRetransmission )
        {
          ipTclassTag.SetTclass (GetIpv6Tclass () | 0x3);
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && (GetIpv6Tclass () & 0x3) == 0 && !isRetransmission)
            {
              ipTclassTag.SetTclass (GetIpv6Tclass () | 0x1);
            }
          else
            {
              ipTclassTag.SetTclass (GetIpv6Tclass ());
            }
        }
      p->AddPacketTag (ipTclassTag);
    }
  else
    {
      SocketIpv6TclassTag ipTclassTag;
      if (m_testCase == 3 && m_dataPacketSent == 1 && !isRetransmission)
        {
          ipTclassTag.SetTclass (0x3);
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && !isRetransmission)
            {
              ipTclassTag.SetTclass (0x1);
            }
        }
      p->AddPacketTag (ipTclassTag);
    }

  if (IsManualIpTtl ())
    {
      SocketIpTtlTag ipTtlTag;
      ipTtlTag.SetTtl (GetIpTtl ());
      p->AddPacketTag (ipTtlTag);
    }

  if (IsManualIpv6HopLimit ())
    {
      SocketIpv6HopLimitTag ipHopLimitTag;
      ipHopLimitTag.SetHopLimit (GetIpv6HopLimit ());
      p->AddPacketTag (ipHopLimitTag);
    }

  uint8_t priority = GetPriority ();
  if (priority)
    {
      SocketPriorityTag priorityTag;
      priorityTag.SetPriority (priority);
      p->ReplacePacketTag (priorityTag);
    }

  if (m_closeOnEmpty && (remainingData == 0))
    {
      flags |= TcpHeader::FIN;
      if (m_state == ESTABLISHED)
        { // On active close: I am the first one to send FIN
          NS_LOG_DEBUG ("ESTABLISHED -> FIN_WAIT_1");
          m_state = FIN_WAIT_1;
        }
      else if (m_state == CLOSE_WAIT)
        { // On passive close: Peer sent me FIN already
          NS_LOG_DEBUG ("CLOSE_WAIT -> LAST_ACK");
          m_state = LAST_ACK;
        }
    }
  TcpHeader header;
  header.SetFlags (flags);
  header.SetSequenceNumber (seq);
  header.SetAckNumber (m_tcb->m_rxBuffer->NextRxSequence ());
  if (m_endPoint)
    {
      header.SetSourcePort (m_endPoint->GetLocalPort ());
      header.SetDestinationPort (m_endPoint->GetPeerPort ());
    }
  else
    {
      header.SetSourcePort (m_endPoint6->GetLocalPort ());
      header.SetDestinationPort (m_endPoint6->GetPeerPort ());
    }
  header.SetWindowSize (AdvertisedWindowSize ());
  AddOptions (header);

  if (m_retxEvent.IsExpired ())
    {
      // Schedules retransmit timeout. m_rto should be already doubled.

      NS_LOG_LOGIC (this << " SendDataPacket Schedule ReTxTimeout at time " <<
                    Simulator::Now ().GetSeconds () << " to expire at time " <<
                    (Simulator::Now () + m_rto.Get ()).GetSeconds () );
      m_retxEvent = Simulator::Schedule (m_rto, &TcpDctcpCongestedRouter::ReTxTimeout, this);
    }

  m_txTrace (p, header, this);

  if (m_endPoint)
    {
      m_tcp->SendPacket (p, header, m_endPoint->GetLocalAddress (),
                         m_endPoint->GetPeerAddress (), m_boundnetdevice);
      NS_LOG_DEBUG ("Send segment of size " << sz << " with remaining data " <<
                    remainingData << " via TcpL4Protocol to " <<  m_endPoint->GetPeerAddress () <<
                    ". Header " << header);
    }
  else
    {
      m_tcp->SendPacket (p, header, m_endPoint6->GetLocalAddress (),
                         m_endPoint6->GetPeerAddress (), m_boundnetdevice);
      NS_LOG_DEBUG ("Send segment of size " << sz << " with remaining data " <<
                    remainingData << " via TcpL4Protocol to " <<  m_endPoint6->GetPeerAddress () <<
                    ". Header " << header);
    }

  UpdateRttHistory (seq, sz, isRetransmission);

  // Notify the application of the data being sent unless this is a retransmit
  if (seq + sz > m_tcb->m_highTxMark)
    {
      Simulator::ScheduleNow (&TcpDctcpCongestedRouter::NotifyDataSent, this,
                              (seq + sz - m_tcb->m_highTxMark.Get ()));
    }
  // Update highTxMark
  m_tcb->m_highTxMark = std::max (seq + sz, m_tcb->m_highTxMark.Get ());
  return sz;
}

Ptr<TcpSocketBase>
TcpDctcpCongestedRouter::Fork (void)
{
  return CopyObject<TcpDctcpCongestedRouter> (this);
}

Ptr<TcpSocketMsgBase>
TcpDctcpCodePointsTest::CreateSenderSocket (Ptr<Node> node)
{
  if (m_testCase == 2)
    {
      return TcpGeneralTest::CreateSenderSocket (node);
    }
  else if (m_testCase == 3)
    {
      Ptr<TcpDctcpCongestedRouter> socket = DynamicCast<TcpDctcpCongestedRouter> (
          CreateSocket (node,
                        TcpDctcpCongestedRouter::GetTypeId (),
                        TcpDctcp::GetTypeId ()));
      socket->SetTestCase (m_testCase);
      return socket;
    }
  else
    {
      return TcpGeneralTest::CreateSocket (node, TcpSocketMsgBase::GetTypeId (), TcpDctcp::GetTypeId ());
    }
}

Ptr<TcpSocketMsgBase>
TcpDctcpCodePointsTest::CreateReceiverSocket (Ptr<Node> node)
{
  if (m_testCase == 2)
    {
      return TcpGeneralTest::CreateReceiverSocket (node);
    }
  else
    {
      return TcpGeneralTest::CreateSocket (node, TcpSocketMsgBase::GetTypeId (), TcpDctcp::GetTypeId ());
    }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief DCTCP should be same as Linux during slow start
 */
class TcpDctcpToLinuxReno : public TestCase
{
public:
  /**
   * \brief Constructor
   *
   * \param cWnd congestion window
   * \param segmentSize segment size
   * \param ssThresh slow start threshold
   * \param segmentsAcked segments acked
   * \param highTxMark high tx mark
   * \param lastAckedSeq last acked seq
   * \param rtt RTT
   * \param name Name of the test
   */
  TcpDctcpToLinuxReno (uint32_t cWnd, uint32_t segmentSize, uint32_t ssThresh,
                     uint32_t segmentsAcked, SequenceNumber32 highTxMark,
                     SequenceNumber32 lastAckedSeq, Time rtt, const std::string &name);

private:
  virtual void DoRun (void);
  /** \brief Execute the test
   */
  void ExecuteTest (void);

  uint32_t m_cWnd;                        //!< cWnd
  uint32_t m_segmentSize;                 //!< segment size
  uint32_t m_segmentsAcked;               //!< segments acked
  uint32_t m_ssThresh;                    //!< ss thresh
  Time m_rtt;                             //!< rtt
  SequenceNumber32 m_highTxMark;          //!< high tx mark
  SequenceNumber32 m_lastAckedSeq;        //!< last acked seq
  Ptr<TcpSocketState> m_state;            //!< state
};

TcpDctcpToLinuxReno::TcpDctcpToLinuxReno (uint32_t cWnd, uint32_t segmentSize, uint32_t ssThresh,
                                      uint32_t segmentsAcked, SequenceNumber32 highTxMark,
                                      SequenceNumber32 lastAckedSeq, Time rtt, const std::string &name)
  : TestCase (name),
    m_cWnd (cWnd),
    m_segmentSize (segmentSize),
    m_segmentsAcked (segmentsAcked),
    m_ssThresh (ssThresh),
    m_rtt (rtt),
    m_highTxMark (highTxMark),
    m_lastAckedSeq (lastAckedSeq)
{
}

void
TcpDctcpToLinuxReno::DoRun ()
{
  Simulator::Schedule (Seconds (0.0), &TcpDctcpToLinuxReno::ExecuteTest, this);
  Simulator::Run ();
  Simulator::Destroy ();
}

void
TcpDctcpToLinuxReno::ExecuteTest ()
{
  m_state = CreateObject <TcpSocketState> ();
  m_state->m_cWnd = m_cWnd;
  m_state->m_ssThresh = m_ssThresh;
  m_state->m_segmentSize = m_segmentSize;
  m_state->m_highTxMark = m_highTxMark;
  m_state->m_lastAckedSeq = m_lastAckedSeq;

  Ptr<TcpSocketState> state = CreateObject <TcpSocketState> ();
  state->m_cWnd = m_cWnd;
  state->m_ssThresh = m_ssThresh;
  state->m_segmentSize = m_segmentSize;
  state->m_highTxMark = m_highTxMark;
  state->m_lastAckedSeq = m_lastAckedSeq;

  Ptr<TcpDctcp> cong = CreateObject <TcpDctcp> ();
  cong->IncreaseWindow (m_state, m_segmentsAcked);

  Ptr<TcpLinuxReno> LinuxRenoCong = CreateObject <TcpLinuxReno> ();
  LinuxRenoCong->IncreaseWindow (state, m_segmentsAcked);

  NS_TEST_ASSERT_MSG_EQ (m_state->m_cWnd.Get (), state->m_cWnd.Get (),
                         "cWnd has not updated correctly");
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP DCTCP TestSuite
 */
class TcpDctcpTestSuite : public TestSuite
{
public:
  TcpDctcpTestSuite () : TestSuite ("tcp-dctcp-test", UNIT)
  {
    AddTestCase (new TcpDctcpToLinuxReno (2 * 1446, 1446, 4 * 1446, 2, SequenceNumber32 (4753), SequenceNumber32 (3216), MilliSeconds (100), "DCTCP falls to New Reno for slowstart"), TestCase::QUICK);
    AddTestCase (new TcpDctcpCodePointsTest (1, "ECT Test : Check if ECT is set on Syn, Syn+Ack, Ack and Data packets for DCTCP packets"),
                 TestCase::QUICK);
    AddTestCase (new TcpDctcpCodePointsTest (2, "ECT Test : Check if ECT is not set on Syn, Syn+Ack and Ack but set on Data packets for non-DCTCP but ECN enabled traffic"),TestCase::QUICK);
    AddTestCase (new TcpDctcpCodePointsTest (3, "ECE Functionality Test: ECE should only be sent by receiver when it receives CE flags"),
                 TestCase::QUICK);
  }
};

static TcpDctcpTestSuite g_tcpdctcpTest; //!< static var for test initialization
