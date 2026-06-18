/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Natale Patriciello <natale.patriciello@gmail.com>
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

#include "tcp-general-test.h"
#include "ns3/node.h"
#include "ns3/log.h"
#include "ns3/config.h"
#include "tcp-error-model.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpBytesInFlightTestSuite");

/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief Check the value of BytesInFlight against a home-made guess
 *
 * The guess is made wrt to segments that travel the network; we have,
 * in theory, the possibility to know the real amount of bytes in flight. However
 * this value is useless, since the sender bases its guess on the received ACK.
 *
 * \see Tx
 * \see BytesInFlightTrace
 */
class TcpBytesInFlightTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor.
   * \param desc Description.
   * \param toDrop Packets to drop.
   */
  TcpBytesInFlightTest (const std::string &desc, std::vector<uint32_t> &toDrop);

protected:
  /**
   * \brief Create a receiver error model.
   * \returns The receiver error model.
   */
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();
  /**
   * \brief Receive a packet.
   * \param p The packet.
   * \param h The TCP header.
   * \param who Who the socket belongs to (sender or receiver).
   */
  virtual void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  /**
   * \brief Transmit a packet.
   * \param p The packet.
   * \param h The TCP header.
   * \param who Who the socket belongs to (sender or receiver).
   */
  virtual void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  /**
   * \brief Track the bytes in flight.
   * \param oldValue previous value.
   * \param newValue actual value.
   */
  virtual void BytesInFlightTrace (uint32_t oldValue, uint32_t newValue);

  /**
   * \brief Called when a packet is dropped.
   * \param ipH The IP header.
   * \param tcpH The TCP header.
   * \param p The packet.
   */
  void PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> p);

  /**
   * \brief Configure the test.
   */
  void ConfigureEnvironment ();

  /**
   * \brief Do the checks before the RTO expires.
   * \param tcb The TcpSocketState.
   * \param who The socket.
   */
  void BeforeRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who);

  /**
   * \brief Update when RTO expires
   * \param oldVal old time value
   * \param newVal new time value
   */
  void RTOExpired (Time oldVal, Time newVal);

  /**
   * \brief Do the final checks.
   */
  void FinalChecks ();

private:
  uint32_t m_guessedBytesInFlight;  //!< Guessed bytes in flight.
  uint32_t m_dupAckRecv;            //!< Number of DupACKs received.
  SequenceNumber32 m_lastAckRecv;   //!< Last ACK received.
  SequenceNumber32 m_greatestSeqSent; //!< greatest sequence number sent.
  std::vector<uint32_t> m_toDrop;     //!< List of SequenceNumber to drop
};

TcpBytesInFlightTest::TcpBytesInFlightTest (const std::string &desc,
                                            std::vector<uint32_t> &toDrop)
  : TcpGeneralTest (desc),
    m_guessedBytesInFlight (0),
    m_dupAckRecv (0),
    m_lastAckRecv (1),
    m_greatestSeqSent (0),
    m_toDrop (toDrop)
{
}

void
TcpBytesInFlightTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (30);
  SetPropagationDelay (MilliSeconds (50));
  SetTransmitStart (Seconds (2.0));

  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (false));
}

Ptr<ErrorModel>
TcpBytesInFlightTest::CreateReceiverErrorModel ()
{
  Ptr<TcpSeqErrorModel> m_errorModel = CreateObject<TcpSeqErrorModel> ();
  for (std::vector<uint32_t>::iterator it = m_toDrop.begin (); it != m_toDrop.end (); ++it)
    {
      m_errorModel->AddSeqToKill (SequenceNumber32 (*it));
    }

  m_errorModel->SetDropCallback (MakeCallback (&TcpBytesInFlightTest::PktDropped, this));

  return m_errorModel;
}

void
TcpBytesInFlightTest::BeforeRTOExpired (const Ptr<const TcpSocketState> tcb, SocketWho who)
{
  NS_LOG_DEBUG ("Before RTO for " << who);
  GetSenderSocket ()->TraceConnectWithoutContext ("RTO", MakeCallback (&TcpBytesInFlightTest::RTOExpired, this));
}

void
TcpBytesInFlightTest::RTOExpired (Time oldVal, Time newVal)
{
  NS_LOG_DEBUG ("RTO expired at " << newVal.GetSeconds ());
  m_guessedBytesInFlight = 0;
}

void
TcpBytesInFlightTest::PktDropped (const Ipv4Header &ipH, const TcpHeader &tcpH,
                                  Ptr<const Packet> p)
{
  NS_LOG_DEBUG ("Drop seq= " << tcpH.GetSequenceNumber () << " size " << p->GetSize ());
}

void
TcpBytesInFlightTest::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == RECEIVER)
    {
    }
  else if (who == SENDER)
    {
      if (h.GetAckNumber () > m_lastAckRecv)
        { // New ack
          uint32_t diff = h.GetAckNumber () - m_lastAckRecv;
          NS_LOG_DEBUG ("Recv ACK=" << h.GetAckNumber ());

          if (m_dupAckRecv > 0)
            { // Previously we got some ACKs
              if (h.GetAckNumber () >= m_greatestSeqSent)
                { // This an ACK which acknowledge all the window
                  m_guessedBytesInFlight = 0; // All outstanding data acked
                  diff = 0;
                  m_dupAckRecv = 0;
                }
              else
                {
                  // Partial ACK: Update the dupAck received count
                  m_dupAckRecv -= diff / GetSegSize (SENDER);
                  // During fast recovery the TCP data sender respond to a partial acknowledgment
                  // by inferring that the next in-sequence packet has been lost (RFC5681)
                  m_guessedBytesInFlight -= GetSegSize (SENDER);
                }
            }

          if ((h.GetFlags () & TcpHeader::FIN) != 0
              || m_guessedBytesInFlight + 1 == diff)
            { // received the ACK for the FIN (which includes 1 spurious byte)
              diff -= 1;
            }
          m_guessedBytesInFlight -= diff;
          m_lastAckRecv = h.GetAckNumber ();
          NS_LOG_DEBUG ("Update m_guessedBytesInFlight to " <<
                        m_guessedBytesInFlight);
        }
      else if (h.GetAckNumber () == m_lastAckRecv
               && m_lastAckRecv != SequenceNumber32 (1)
               && (h.GetFlags () & TcpHeader::FIN) == 0)
        {
          // For each dupack I should guess that a segment has been received
          // Please do not count FIN and SYN/ACK as dupacks
          m_guessedBytesInFlight -= GetSegSize (SENDER);
          m_dupAckRecv++;
          // RFC 6675 says after two dupacks, the segment is considered lost
          if (m_dupAckRecv == 3)
            {
              NS_LOG_DEBUG ("Loss of a segment detected");
              m_guessedBytesInFlight -= GetSegSize (SENDER);
            }
          NS_LOG_DEBUG ("Dupack received, Update m_guessedBytesInFlight to " <<
                        m_guessedBytesInFlight);
        }

    }
}

void
TcpBytesInFlightTest::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      static SequenceNumber32 retr = SequenceNumber32 (0);
      static uint32_t times = 0;

      if (m_greatestSeqSent <= h.GetSequenceNumber ())
        { // This is not a retransmission
          m_greatestSeqSent = h.GetSequenceNumber ();
          times = 0;
        }

      if (retr == h.GetSequenceNumber ())
        {
          ++times;
        }

      if (times < 2)
        {
          // count retransmission only one time
          m_guessedBytesInFlight += p->GetSize ();
        }
      retr = h.GetSequenceNumber ();

      NS_LOG_DEBUG ("TX size=" << p->GetSize () << " seq=" << h.GetSequenceNumber () <<
                    " m_guessedBytesInFlight=" << m_guessedBytesInFlight);
    }
}

void
TcpBytesInFlightTest::BytesInFlightTrace (uint32_t oldValue, uint32_t newValue)
{
  NS_LOG_DEBUG ("Socket BytesInFlight=" << newValue <<
                " mine is=" << m_guessedBytesInFlight);
  NS_TEST_ASSERT_MSG_EQ (m_guessedBytesInFlight, newValue,
                         "At time " << Simulator::Now ().GetSeconds () << "; guessed and measured bytes in flight differs");
}

void
TcpBytesInFlightTest::FinalChecks ()
{
  NS_TEST_ASSERT_MSG_EQ (m_guessedBytesInFlight, 0,
                         "Still present bytes in flight at the end of the transmission");
}


/**
 * \ingroup internet-test
 * \ingroup tests
 *
 * \brief TestSuite: Check the value of BytesInFlight against a home-made guess
 */
class TcpBytesInFlightTestSuite : public TestSuite
{
public:
  TcpBytesInFlightTestSuite () : TestSuite ("tcp-bytes-in-flight-test", UNIT)
  {
    std::vector<uint32_t> toDrop;
    AddTestCase (new TcpBytesInFlightTest ("BytesInFlight value, no drop", toDrop),
                 TestCase::QUICK);
    toDrop.push_back (4001);
    AddTestCase (new TcpBytesInFlightTest ("BytesInFlight value, one drop", toDrop),
                 TestCase::QUICK);
    toDrop.push_back (4001);
    AddTestCase (new TcpBytesInFlightTest ("BytesInFlight value, two drop of same segment", toDrop),
                 TestCase::QUICK);
    toDrop.pop_back ();
    toDrop.push_back (4501);
    AddTestCase (new TcpBytesInFlightTest ("BytesInFlight value, two drop of consecutive segments", toDrop),
                 TestCase::QUICK);
  }
};

static TcpBytesInFlightTestSuite g_tcpBytesInFlightTestSuite; //!< Static variable for test initialization

