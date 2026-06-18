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
 * Authors: Shravya Ks <shravya.ks0@gmail.com>
 *
 */
#include "ns3/ipv4.h"
#include "ns3/ipv6.h"
#include "ns3/ipv4-interface-address.h"
#include "ns3/ipv4-route.h"
#include "ns3/ipv6-route.h"
#include "ns3/ipv4-routing-protocol.h"
#include "ns3/ipv6-routing-protocol.h"
#include "ns3/ipv4-end-point.h"
#include "ns3/ipv6-end-point.h"
#include "tcp-general-test.h"
#include "ns3/node.h"
#include "ns3/log.h"
#include "tcp-error-model.h"
#include "ns3/tcp-l4-protocol.h"
#include "ns3/tcp-tx-buffer.h"
#include "ns3/tcp-rx-buffer.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpEcnTestSuite");
/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief checks if ECT, CWR and ECE bits are set correctly in different scenarios
 *
 * This test suite will run four combinations of enabling ECN (sender off and receiver off; sender on and receiver off;
 * sender off and receiver on; sender on and receiver on;) and checks that the TOS byte of eventual packets transmitted
 * or received have ECT, CWR, and ECE set correctly (or not). It also checks if congestion window is being reduced by half
 * only once per every window on receipt of ECE flags
 *
 */
class TcpEcnTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   *
   * \param testcase test case number
   * \param desc Description about the ECN capabilities of sender and receiver
   */
  TcpEcnTest (uint32_t testcase, const std::string &desc);

protected:
  virtual void CWndTrace (uint32_t oldValue, uint32_t newValue);
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);
  void ConfigureProperties ();

private:
  uint32_t m_cwndChangeCount;     //!< Number of times the congestion window did change
  uint32_t m_senderSent;          //!< Number of segments sent by the sender
  uint32_t m_senderReceived;      //!< Number of segments received by the sender
  uint32_t m_receiverReceived;    //!< Number of segments received by the receiver
  uint32_t m_testcase;            //!< Test case type
};


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief A TCP socket which sends certain data packets with CE flags set for tests 5 and 6.
 *
 * The SendDataPacket function of this class sends data packets numbered 1 and 3 with CE flags set
 * for test 5 to verify if ECE and CWR bits are correctly set by receiver and sender respectively. It
 * also sets CE flags on data packets 4 and 5 in test case 6 to check if sender reduces congestion window
 * appropriately and also only once per every window.
 *
 */
class TcpSocketCongestedRouter : public TcpSocketMsgBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  uint32_t m_dataPacketSent;  //!< Number of packets sent
  uint8_t m_testcase;         //!< Test case type

  TcpSocketCongestedRouter ()
    : TcpSocketMsgBase ()
  {
    m_dataPacketSent = 0;
  }

  /**
   * \brief Constructor.
   * \param other The object to copy from.
   */
  TcpSocketCongestedRouter (const TcpSocketCongestedRouter &other)
    : TcpSocketMsgBase (other)
  {
  }

  /**
   * Set the test case type
   * \param testCase Test case type
   */
  void SetTestCase (uint8_t testCase);

protected:
  virtual uint32_t SendDataPacket (SequenceNumber32 seq, uint32_t maxSize, bool withAck);
  virtual void ReTxTimeout ();
  Ptr<TcpSocketBase> Fork (void);
};

NS_OBJECT_ENSURE_REGISTERED (TcpSocketCongestedRouter);

TypeId
TcpSocketCongestedRouter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpSocketCongestedRouter")
    .SetParent<TcpSocketMsgBase> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpSocketCongestedRouter> ()
  ;
  return tid;
}

void
TcpSocketCongestedRouter::ReTxTimeout ()
{
  TcpSocketBase::ReTxTimeout ();
}

void
TcpSocketCongestedRouter::SetTestCase (uint8_t testCase)
{
  m_testcase = testCase;
}

uint32_t
TcpSocketCongestedRouter::SendDataPacket (SequenceNumber32 seq, uint32_t maxSize, bool withAck)
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

  // Sender should reduce the Congestion Window as a response to receiver's ECN Echo notification only once per window
  if (m_tcb->m_ecnState == TcpSocketState::ECN_ECE_RCVD && m_ecnEchoSeq.Get () > m_ecnCWRSeq.Get () && !isRetransmission)
    {
      NS_LOG_DEBUG (TcpSocketState::EcnStateName[m_tcb->m_ecnState] << " -> ECN_CWR_SENT");
      m_tcb->m_ecnState = TcpSocketState::ECN_CWR_SENT;
      m_ecnCWRSeq = seq;
      flags |= TcpHeader::CWR;
      NS_LOG_INFO ("CWR flags set");
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
      if ( m_testcase == 5 && (m_dataPacketSent == 1 || m_dataPacketSent == 3))
        {
          ipTosTag.SetTos (MarkEcnCe (GetIpTos ()));
        }
      else if ( m_testcase == 6 &&  ( m_dataPacketSent == 4 || m_dataPacketSent == 5))
        {
          ipTosTag.SetTos (MarkEcnCe (GetIpTos ()));
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && (GetIpTos () & 0x3) == 0)
            {
              ipTosTag.SetTos (MarkEcnEct0 (GetIpTos ()));
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
      if ( m_testcase == 5 && (m_dataPacketSent == 1 || m_dataPacketSent == 3))
        {
          ipTosTag.SetTos (MarkEcnCe (GetIpTos ()));
        }
      else if ( m_testcase == 6 && ( m_dataPacketSent == 4 || m_dataPacketSent == 5))
        {
          ipTosTag.SetTos (MarkEcnCe (GetIpTos ()));
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
            {
              ipTosTag.SetTos (MarkEcnEct0 (GetIpTos ()));
            }
        }
      p->AddPacketTag (ipTosTag);
    }

  if (IsManualIpv6Tclass ())
    {
      SocketIpv6TclassTag ipTclassTag;
      if ( m_testcase == 5 && (m_dataPacketSent == 1  || m_dataPacketSent == 3))
        {
          ipTclassTag.SetTclass (MarkEcnCe (GetIpv6Tclass ()));
        }
      else if ( m_testcase == 6 && ( m_dataPacketSent == 4 || m_dataPacketSent == 5))
        {
          ipTclassTag.SetTclass (MarkEcnCe (GetIpv6Tclass ()));
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED && (GetIpv6Tclass () & 0x3) == 0)
            {
              ipTclassTag.SetTclass (MarkEcnEct0 (GetIpv6Tclass ()));
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
      if ( m_testcase == 5 && (m_dataPacketSent == 1  || m_dataPacketSent == 3))
        {
          ipTclassTag.SetTclass (MarkEcnCe (GetIpv6Tclass ()));
        }
      else if ( m_testcase == 6 &&( m_dataPacketSent == 4 || m_dataPacketSent == 5 ))
        {
          ipTclassTag.SetTclass (MarkEcnCe (GetIpv6Tclass ()));
        }
      else
        {
          if (m_tcb->m_ecnState != TcpSocketState::ECN_DISABLED)
            {
              ipTclassTag.SetTclass (MarkEcnEct0 (GetIpv6Tclass ()));
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
      m_retxEvent = Simulator::Schedule (m_rto, &TcpSocketCongestedRouter::ReTxTimeout, this);
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
      Simulator::ScheduleNow (&TcpSocketCongestedRouter::NotifyDataSent, this,
                              (seq + sz - m_tcb->m_highTxMark.Get ()));
    }
  // Update highTxMark
  m_tcb->m_highTxMark = std::max (seq + sz, m_tcb->m_highTxMark.Get ());
  return sz;
}

Ptr<TcpSocketBase>
TcpSocketCongestedRouter::Fork (void)
{
  return CopyObject<TcpSocketCongestedRouter> (this);
}


TcpEcnTest::TcpEcnTest (uint32_t testcase, const std::string &desc)
  : TcpGeneralTest (desc),
  m_cwndChangeCount (0),
  m_senderSent (0),
  m_senderReceived (0),
  m_receiverReceived (0),
  m_testcase (testcase)
{
}

void
TcpEcnTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  if (m_testcase == 2 || m_testcase == 4 || m_testcase == 5 || m_testcase == 6)
    {
      SetUseEcn (SENDER, TcpSocketState::On);
    }
  if (m_testcase == 3 || m_testcase == 4 ||m_testcase == 5 || m_testcase == 6)
    {
      SetUseEcn (RECEIVER, TcpSocketState::On);
    }
}

void
TcpEcnTest::CWndTrace (uint32_t oldValue, uint32_t newValue)
{
  if (m_testcase == 6)
    {
      if (newValue < oldValue)
        {
          m_cwndChangeCount++;
          NS_TEST_ASSERT_MSG_EQ (m_cwndChangeCount, 1, "Congestion window should be reduced once per every window");
          NS_TEST_ASSERT_MSG_EQ (newValue, 1000, "Congestion window should not drop below 2 segments");
        }
    }
}

void
TcpEcnTest::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == RECEIVER)
    {
      if (m_receiverReceived == 0)
        {
          NS_TEST_ASSERT_MSG_NE (((h.GetFlags ()) & TcpHeader::SYN), 0, "SYN should be received as first message at the receiver");
          if (m_testcase == 2 || m_testcase == 4 || m_testcase == 5 ||m_testcase == 6)
            {
              NS_TEST_ASSERT_MSG_NE (((h.GetFlags ()) & TcpHeader::ECE) && ((h.GetFlags ()) & TcpHeader::CWR), 0, "The flags ECE + CWR should be set in the TCP header of first message received at receiver when sender is ECN Capable");
            }
          else
            {
              NS_TEST_ASSERT_MSG_EQ (((h.GetFlags ()) & TcpHeader::ECE) && ((h.GetFlags ()) & TcpHeader::CWR), 0, "The flags ECE + CWR should not be set in the TCP header of first message received at receiver when sender is not ECN Capable");
            }
        }
      else if (m_receiverReceived == 1)
        {
          NS_TEST_ASSERT_MSG_NE (((h.GetFlags ()) & TcpHeader::ACK), 0, "ACK should be received as second message at receiver");
        }
      else if (m_receiverReceived == 3 && m_testcase == 5)
        {
          NS_TEST_ASSERT_MSG_NE (((h.GetFlags ()) & TcpHeader::CWR), 0, "Sender should send CWR on receipt of ECE");
        }
      m_receiverReceived++;
    }
  else if (who == SENDER)
    {
      if (m_senderReceived == 0)
        {
          NS_TEST_ASSERT_MSG_NE (((h.GetFlags ()) & TcpHeader::SYN) && ((h.GetFlags ()) & TcpHeader::ACK), 0, "SYN+ACK received as first message at sender");
          if (m_testcase == 4 || m_testcase == 5 || m_testcase == 6)
            {
              NS_TEST_ASSERT_MSG_NE ((h.GetFlags () & TcpHeader::ECE), 0, "The flag ECE should be set in the TCP header of first message received at sender when both receiver and sender are ECN Capable");
            }
          else
            {
              NS_TEST_ASSERT_MSG_EQ (((h.GetFlags ()) & TcpHeader::ECE), 0, "The flag ECE should not be set in the TCP header of first message received at sender when  either receiver or sender are not ECN Capable");
            }
        }
      if ( m_testcase == 5 && m_receiverReceived > 12)
        {
          NS_TEST_ASSERT_MSG_EQ (((h.GetFlags ()) & TcpHeader::ECE), 0, "The flag ECE should not be set in TCP header of the packet sent by the receiver after sender sends CWR flags to receiver and receiver receives a packet without CE bit set in IP header");
        }
      m_senderReceived++;
    }
}

void
TcpEcnTest::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      m_senderSent++;
      if (m_senderSent == 3)
        {
          SocketIpTosTag ipTosTag;
          bool found = p->PeekPacketTag (ipTosTag);
          uint16_t ipTos = 0;
          if (found)
            {
              ipTos = static_cast<uint16_t> (ipTosTag.GetTos ());
            }
          if (m_testcase == 4 || m_testcase == 6)
            {
              NS_TEST_ASSERT_MSG_EQ (ipTos, 0x2, "IP TOS should have ECT set if ECN negotiation between endpoints is successful");
            }
          else if (m_testcase == 5)
            {
              if (m_senderSent == 3 || m_senderSent == 5)
                {
                  NS_TEST_ASSERT_MSG_EQ (ipTos, 0x3, "IP TOS should have CE bit set for 3rd and 5th packet sent in test case 5");
                }
              else
                {
                  NS_TEST_ASSERT_MSG_EQ (ipTos, 0x2, "IP TOS should have ECT set if ECN negotiation between endpoints is successful");
                }
            }
          else
            {
              NS_TEST_ASSERT_MSG_NE (ipTos, 0x2, "IP TOS should not have ECT set if ECN negotiation between endpoints is unsuccessful");
            }
        }
    }
}

Ptr<TcpSocketMsgBase>
TcpEcnTest::CreateSenderSocket (Ptr<Node> node)
{
  if (m_testcase == 5 || m_testcase == 6)
    {
      Ptr<TcpSocketCongestedRouter> socket = DynamicCast<TcpSocketCongestedRouter> (
          CreateSocket (node,
                        TcpSocketCongestedRouter::GetTypeId (),
                        m_congControlTypeId));
      socket->SetTestCase (m_testcase);
      return socket;
    }
  else
    {
      return TcpGeneralTest::CreateSenderSocket (node);
    }
}

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP ECN TestSuite
 */
class TcpEcnTestSuite : public TestSuite
{
public:
  TcpEcnTestSuite () : TestSuite ("tcp-ecn-test", UNIT)
  {
    AddTestCase (new TcpEcnTest (1, "ECN Negotiation Test : ECN incapable sender and ECN incapable receiver"),
                 TestCase::QUICK);
    AddTestCase (new TcpEcnTest (2, "ECN Negotiation Test : ECN capable sender and ECN incapable receiver"),
                 TestCase::QUICK);
    AddTestCase (new TcpEcnTest (3, "ECN Negotiation Test : ECN incapable sender and ECN capable receiver"),
                 TestCase::QUICK);
    AddTestCase (new TcpEcnTest (4, "ECN Negotiation Test : ECN capable sender and ECN capable receiver"),
                 TestCase::QUICK);
    AddTestCase (new TcpEcnTest (5, "ECE and CWR Functionality Test: ECN capable sender and ECN capable receiver"),
                 TestCase::QUICK);
    AddTestCase (new TcpEcnTest (6, "Congestion Window Reduction Test :ECN capable sender and ECN capable receiver"),
                 TestCase::QUICK);
  }
};

static TcpEcnTestSuite g_tcpECNTestSuite; //!< static var for test initialization

} // namespace ns3
