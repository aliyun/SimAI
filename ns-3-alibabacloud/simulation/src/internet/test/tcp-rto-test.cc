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

#include "ns3/node.h"
#include "ns3/log.h"
#include "ns3/simple-channel.h"
#include "ns3/rtt-estimator.h"
#include "tcp-general-test.h"
#include "tcp-error-model.h"

NS_LOG_COMPONENT_DEFINE ("TcpRtoTest");

using namespace ns3;

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the moments after an RTO expiration
 *
 * The scope of this test is to be sure that, after an RTO expiration,
 * the TCP implementation set the correct state in the ACK state machine,
 * and marks the lost segment as lost; then, after the retransmission, the
 * state is fully recovered. This is the base check, where only one segment
 * (the first) is lost and retransmitted.
 *
 */
class TcpRtoTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param congControl Congestion control type.
   * \param msg Test description.
   */
  TcpRtoTest (const TypeId &congControl, const std::string &msg);

protected:

  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);
  virtual void AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who);
  virtual void RcvAck      (const Ptr<const TcpSocketState> tcb,
                            const TcpHeader& h, SocketWho who);
  virtual void ProcessedAck (const Ptr<const TcpSocketState> tcb,
                             const TcpHeader& h, SocketWho who);
  virtual void FinalChecks ();
  virtual void ConfigureProperties ();
  virtual void ConfigureEnvironment ();

private:
  bool m_afterRTOExpired;      //!< True if RTO is expired.
  bool m_segmentReceived; //!< True if segments have been received.
};

TcpRtoTest::TcpRtoTest (const TypeId &congControl, const std::string &desc)
  : TcpGeneralTest (desc),
    m_afterRTOExpired (false),
    m_segmentReceived (false)
{
  m_congControlTypeId = congControl;
}

void
TcpRtoTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (100);
}

void
TcpRtoTest::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetInitialSsThresh (SENDER, 0);
}

Ptr<TcpSocketMsgBase>
TcpRtoTest::CreateSenderSocket (Ptr<Node> node)
{
  // Get a really low RTO, and let them fire as soon as possible since
  // we are interested only in what happen after it expires
  Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket (node);
  socket->SetAttribute ("MinRto", TimeValue (Seconds (0.5)));

  return socket;
}

void
TcpRtoTest::AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who)
{
  // In this test, the RTO fires for the first segment (and no more).
  // This function is called after the management of the RTO expiration,
  // and because of this we must check all the involved variables.
  NS_TEST_ASSERT_MSG_EQ (m_afterRTOExpired, false,
                         "Second RTO expired");
  NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_LOSS,
                         "Ack state machine not in LOSS state after a loss");

  m_afterRTOExpired = true;
}

void
TcpRtoTest::RcvAck (const Ptr<const TcpSocketState> tcb, const TcpHeader& h,
                    SocketWho who)
{
  // Called after the first ack is received (the lost segment has been
  // successfully retransmitted. We must check on the sender that variables
  // are in the same state as they where after AfterRTOExpired if it is the first
  // ACK after the loss; in every other case, all must be OPEN and the counter
  // set to 0.

  if (m_afterRTOExpired && who == SENDER)
    {
      NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_LOSS,
                             "Ack state machine not in LOSS state after a loss");
    }
  else
    {
      NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_OPEN,
                             "Ack state machine not in OPEN state after recovering "
                             "from loss");
    }
}

void
TcpRtoTest::ProcessedAck (const Ptr<const TcpSocketState> tcb, const TcpHeader &h,
                          SocketWho who)
{
  // Called after the ACK processing. Every time we should be in OPEN state,
  // without any packet lost or marked as retransmitted, in both the sockets

  NS_TEST_ASSERT_MSG_EQ (GetCongStateFrom (tcb), TcpSocketState::CA_OPEN,
                         "Ack state machine not in OPEN state after recovering "
                         "from loss");

  if (who == SENDER)
    {
      m_afterRTOExpired = false;
      m_segmentReceived = true;
    }
}

void
TcpRtoTest::FinalChecks ()
{
  // At least one time we should process an ACK; otherwise, the segment
  // has not been retransmitted, and this is bad

  NS_TEST_ASSERT_MSG_EQ (m_segmentReceived, true,
                         "Retransmission has not been done");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the ssthresh behavior after the RTO expires
 *
 * The scope of this test is to be sure that, after an RTO expiration,
 * the TCP implementation sets the correct ssthresh value
 *
 */
class TcpSsThreshRtoTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param congControl congestion control type
   * \param seqToDrop sequence number to drop
   * \param minRto minimum RTO
   * \param msg test description
   */
  TcpSsThreshRtoTest (const TypeId &congControl, uint32_t seqToDrop, Time minRto, const std::string &msg);

protected:

  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();
  virtual void BytesInFlightTrace (uint32_t oldValue, uint32_t newValue);
  virtual void SsThreshTrace (uint32_t oldValue, uint32_t newValue);
  virtual void BeforeRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who);
  virtual void AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who);

  virtual void ConfigureEnvironment ();

  /**
   * \brief Called when a packet has been dropped.
   * \param ipH IPv4 header.
   * \param tcpH TCP header.
   * \param p The packet.
   */
  void PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> p);

private:
  uint32_t m_bytesInFlight; //!< Store the number of bytes in flight
  uint32_t m_bytesInFlightBeforeRto; //!< Store the number of bytes in flight before the RTO expiration
  uint32_t m_ssThreshSocket; //!< the ssThresh as computed by the socket
  uint32_t m_seqToDrop; //!< the sequence number to drop
  Time m_minRtoTime; //!< the minimum RTO time
};

TcpSsThreshRtoTest::TcpSsThreshRtoTest (const TypeId &congControl, uint32_t seqToDrop, Time minRto, const std::string &desc)
  : TcpGeneralTest (desc),
    m_seqToDrop (seqToDrop),
    m_minRtoTime (minRto)
{
  m_congControlTypeId = congControl;
}

void
TcpSsThreshRtoTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (100);
  SetAppPktInterval (MicroSeconds(100));
  SetPropagationDelay (MilliSeconds (1));
}

Ptr<TcpSocketMsgBase>
TcpSsThreshRtoTest::CreateSenderSocket (Ptr<Node> node)
{
  Ptr<TcpSocketMsgBase> socket = TcpGeneralTest::CreateSenderSocket (node);
  socket->SetAttribute ("MinRto", TimeValue (m_minRtoTime));
  NS_LOG_DEBUG("TcpSsThreshRtoTest create sender socket");

  return socket;
}

Ptr<ErrorModel>
TcpSsThreshRtoTest::CreateReceiverErrorModel ()
{
  NS_LOG_DEBUG("TcpSsThreshRtoTest create errorModel");

  Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel> ();

  for (uint32_t i = 0; i < 3; ++i)
    {
      errorModel->AddSeqToKill (SequenceNumber32 (m_seqToDrop));
    }

  errorModel->SetDropCallback (MakeCallback (&TcpSsThreshRtoTest::PktDropped, this));

  return errorModel;
}

void
TcpSsThreshRtoTest::PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH,
                                Ptr<const Packet> p)
{
  NS_LOG_DEBUG ("DROPPED! " << tcpH);
}

void
TcpSsThreshRtoTest::BytesInFlightTrace (uint32_t oldValue, uint32_t newValue)
{
  NS_LOG_DEBUG ("Socket BytesInFlight=" << newValue);
  m_bytesInFlight = newValue;
}

void
TcpSsThreshRtoTest::SsThreshTrace (uint32_t oldValue, uint32_t newValue)
{
  NS_LOG_DEBUG ("Socket ssThresh=" << newValue);
  m_ssThreshSocket = newValue;
}

void
TcpSsThreshRtoTest::BeforeRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who)
{
  NS_LOG_DEBUG ("Before RTO for connection " << who);

  // Get the bytesInFlight value before the expiration of the RTO

  if (who == SENDER)
    {
      m_bytesInFlightBeforeRto = m_bytesInFlight;
      NS_LOG_DEBUG("BytesInFlight before RTO Expired " << m_bytesInFlight);
    }
}

void
TcpSsThreshRtoTest::AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who)
{
  NS_LOG_DEBUG ("After RTO for " << who);
  Ptr<TcpSocketMsgBase> senderSocket = GetSenderSocket();

  // compute the ssThresh according to RFC 5681, using the
  uint32_t ssThresh = std::max (m_bytesInFlightBeforeRto / 2, 2 * tcb->m_segmentSize);

  NS_LOG_DEBUG ("ssThresh " << ssThresh << " m_ssThreshSocket " << m_ssThreshSocket);

  NS_TEST_ASSERT_MSG_EQ (ssThresh, m_ssThreshSocket,
                         "Slow Start Threshold is incorrect");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Testing the timing of RTO
 *
 * Checking if RTO is doubled ONLY after a retransmission.
 */
class TcpTimeRtoTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param congControl Congestion control type.
   * \param msg Test description.
   */
  TcpTimeRtoTest (const TypeId &congControl, const std::string &msg);

protected:
  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();
  virtual void ErrorClose  (SocketWho who);
  virtual void AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who);
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  virtual void FinalChecks ();

  virtual void ConfigureEnvironment ();

  /**
   * \brief Called when a packet has been dropped.
   * \param ipH IPv4 header.
   * \param tcpH TCP header.
   * \param p The packet.
   */
  void PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> p);

private:
  uint32_t m_senderSentSegments;  //!< Number of segments sent.
  Time m_previousRTO;             //!< Previous RTO.
  bool m_closed;                  //!< True if the connection is closed.
};


TcpTimeRtoTest::TcpTimeRtoTest (const TypeId &congControl, const std::string &desc)
  : TcpGeneralTest (desc),
    m_senderSentSegments (0),
    m_closed (false)
{
  m_congControlTypeId = congControl;
}

void
TcpTimeRtoTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (100);
}


Ptr<TcpSocketMsgBase>
TcpTimeRtoTest::CreateSenderSocket (Ptr<Node> node)
{
  Ptr<TcpSocketMsgBase> s = TcpGeneralTest::CreateSenderSocket (node);
  s->SetAttribute ("DataRetries", UintegerValue (6));

  return s;
}

Ptr<ErrorModel>
TcpTimeRtoTest::CreateReceiverErrorModel ()
{
  Ptr<TcpSeqErrorModel> errorModel = CreateObject<TcpSeqErrorModel> ();

  // Drop packet for 7 times. At the 7th, the connection should be dropped.
  for (uint32_t i = 0; i < 7; ++i)
    {
      errorModel->AddSeqToKill (SequenceNumber32 (1));
    }

  errorModel->SetDropCallback (MakeCallback (&TcpTimeRtoTest::PktDropped, this));

  return errorModel;
}

void
TcpTimeRtoTest::Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who)
{
  NS_LOG_FUNCTION (this << p << h << who);

  if (who == SENDER)
    {
      ++m_senderSentSegments;
      NS_LOG_INFO ("Measured RTO:" << GetRto (SENDER).GetSeconds ());

      if (h.GetFlags () & TcpHeader::SYN)
        {
          NS_TEST_ASSERT_MSG_EQ (m_senderSentSegments, 1, "Number of segments sent is different than 1");

          Time s_rto = GetRto (SENDER);
          NS_TEST_ASSERT_MSG_EQ (s_rto, GetConnTimeout (SENDER),
                                 "SYN packet sent without respecting "
                                 "ConnTimeout attribute");
        }
      else
        {
          NS_LOG_INFO ("TX: " << h << m_senderSentSegments);

          NS_TEST_ASSERT_MSG_EQ (h.GetSequenceNumber ().GetValue (), 1,
                                 "First packet was not correctly sent");

          // Remember, from RFC:
          // m_rto = Max (m_rtt->GetEstimate () +
          //              Max (m_clockGranularity, m_rtt->GetVariation ()*4), m_minRto);

          if (m_senderSentSegments == 2)
            { // ACK of SYN-ACK, rto set for the first time, since now we have
              // an estimation of RTT

              Ptr<RttEstimator> rttEstimator = GetRttEstimator (SENDER);
              Time clockGranularity = GetClockGranularity (SENDER);
              m_previousRTO = rttEstimator->GetEstimate ();

              if (clockGranularity > rttEstimator->GetVariation () * 4)
                {
                  m_previousRTO += clockGranularity;
                }
              else
                {
                  m_previousRTO += rttEstimator->GetVariation () * 4;
                }

              m_previousRTO = Max (m_previousRTO, GetMinRto (SENDER));

              NS_TEST_ASSERT_MSG_EQ_TOL (GetRto (SENDER), m_previousRTO, Seconds (0.01),
                                         "RTO value differs from calculation");
            }
          else if (m_senderSentSegments == 3)
            { // First data packet. RTO should be the same as before

              NS_TEST_ASSERT_MSG_EQ_TOL (GetRto (SENDER), m_previousRTO, Seconds (0.01),
                                         "RTO value has changed unexpectedly");

            }
        }
    }
  else if (who == RECEIVER)
    {
    }
}

void
TcpTimeRtoTest::ErrorClose (SocketWho who)
{
  m_closed = true;
}

void
TcpTimeRtoTest::AfterRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who)
{
  NS_TEST_ASSERT_MSG_EQ (who, SENDER, "RTO in Receiver. That's unexpected");

  Time actualRto = GetRto (SENDER);

  if (actualRto < Seconds (60))
    {
      NS_TEST_ASSERT_MSG_EQ_TOL (actualRto, m_previousRTO + m_previousRTO, Seconds (0.01),
                                 "RTO has not doubled after an expiration");
      m_previousRTO += m_previousRTO;
    }
  else
    {
      NS_TEST_ASSERT_MSG_EQ (actualRto, Seconds (60),
                             "RTO goes beyond 60 second limit");
    }
}

void
TcpTimeRtoTest::PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH,
                            Ptr<const Packet> p)
{
  NS_LOG_INFO ("DROPPED! " << tcpH);
}

void
TcpTimeRtoTest::FinalChecks ()
{
  NS_TEST_ASSERT_MSG_EQ (m_closed, true,
                         "Socket has not been closed after retrying data retransmissions");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TCP RTO TestSuite
 */
class TcpRtoTestSuite : public TestSuite
{
public:
  TcpRtoTestSuite () : TestSuite ("tcp-rto-test", UNIT)
  {
    std::list<TypeId> types = {
      TcpNewReno::GetTypeId (),
    };

    for (const auto &t : types)
      {
        AddTestCase (new TcpRtoTest (t, t.GetName () + " RTO retransmit testing"), TestCase::QUICK);

        constexpr uint32_t seqToDrop = 25001;

        // With RTO of 0.5 seconds, BytesInFlight winds down to zero before RTO
        AddTestCase (new TcpSsThreshRtoTest (
                        t, seqToDrop, Seconds (0.5),
                        t.GetName () + " RTO ssthresh testing, set to 2*MSL"),
                     TestCase::QUICK);

        // With RTO of 0.005 seconds, FlightSize/2 > 2*SMSS
        AddTestCase (new TcpSsThreshRtoTest (
                        t, seqToDrop, Seconds (0.005),
                        t.GetName () + " RTO ssthresh testing, set to half of BytesInFlight"),
                     TestCase::QUICK);

        AddTestCase (new TcpTimeRtoTest (t, t.GetName () + " RTO timing testing"), TestCase::QUICK);
      }
  }
};

static TcpRtoTestSuite g_TcpRtoTestSuite; //!< Static variable for test initialization
