/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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
#include "tcp-error-model.h"
#include "ns3/node.h"
#include "ns3/log.h"
#include "ns3/config.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpCloseTestSuite");

/**
 * \brief Check if the TCP correctly close the connection after receiving
 * previously lost data
 */
class TcpCloseWithLossTestCase : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   * \param sackEnabled Enable or disable SACK
   */
  TcpCloseWithLossTestCase (bool sackEnabled);

protected:
  Ptr<ErrorModel> CreateReceiverErrorModel ();
  virtual void ConfigureProperties ();
  void Tx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  void Rx (const Ptr<const Packet> p, const TcpHeader&h, SocketWho who);
  void FinalChecks ();

  virtual void NormalClose (SocketWho who)
  {
    if (who == SENDER)
      m_sendClose = true;
    else
      m_recvClose = true;
  }

  /**
   * Called when a packet is dropped.
   * \param ipH IP header
   * \param tcpH TCP header
   * \param pkt packet
   */
  void PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> pkt);
  
private:
  Ptr<TcpSeqErrorModel> m_errorModel; //!< The error model
  bool m_sendClose;                   //!< true when the sender has closed
  bool m_recvClose;                   //!< true when the receiver has closed
  bool m_synReceived;                 //!< true when the receiver gets SYN
  bool m_sackEnabled;                 //!< true if sack should be enabled
};

TcpCloseWithLossTestCase::TcpCloseWithLossTestCase (bool sackEnabled)
  : TcpGeneralTest ("Testing connection closing with retransmissions")
{
  m_sendClose = false;
  m_recvClose = false;
  m_synReceived = false;
  m_sackEnabled = sackEnabled;
}

void
TcpCloseWithLossTestCase::ConfigureProperties ()
{
  TcpGeneralTest::ConfigureProperties ();
  SetAppPktSize (1400);
  SetAppPktCount (1);
  SetAppPktInterval (MilliSeconds (0));
  SetInitialCwnd (SENDER, 4);
  SetMTU (1452);
  SetSegmentSize (SENDER, 700);
  GetSenderSocket ()->SetAttribute ("Timestamp", BooleanValue (false));
  GetReceiverSocket ()->SetAttribute ("Timestamp", BooleanValue (false));
  GetSenderSocket ()->SetAttribute("Sack", BooleanValue (m_sackEnabled));
}

void
TcpCloseWithLossTestCase::FinalChecks()
{
  NS_TEST_ASSERT_MSG_EQ (m_sendClose, true, "Sender has not closed successfully the connection");
  NS_TEST_ASSERT_MSG_EQ (m_recvClose, true, "Recv has not closed successfully the connection");
}

Ptr<ErrorModel>
TcpCloseWithLossTestCase::CreateReceiverErrorModel ()
{
  m_errorModel = CreateObject<TcpSeqErrorModel> ();
  m_errorModel->AddSeqToKill (SequenceNumber32 (1));

  m_errorModel->SetDropCallback (MakeCallback (&TcpCloseWithLossTestCase::PktDropped, this));

  return m_errorModel;
}

void
TcpCloseWithLossTestCase::PktDropped (const Ipv4Header &ipH, const TcpHeader& tcpH, Ptr<const Packet> pkt)
{
  NS_LOG_INFO ("Dropped "<< tcpH);
}

void
TcpCloseWithLossTestCase::Tx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      NS_LOG_INFO ("Sender TX: " << h << " size " << p->GetSize ());
    }
  else
    {
      NS_LOG_INFO ("Receiver TX: " << h << " size " << p->GetSize ());
    }
}

void
TcpCloseWithLossTestCase::Rx (const Ptr<const Packet> p, const TcpHeader &h, SocketWho who)
{
  if (who == SENDER)
    {
      NS_LOG_INFO ("Sender RX: " << h << " size " << p->GetSize ());
    }
  else
    {
      NS_LOG_INFO ("Receiver RX: " << h << " size " << p->GetSize ());
    }
}

/**
 * Check if the TCP is correctly closing its state
 */
class TcpTcpCloseTestSuite : public TestSuite
{
public:
  TcpTcpCloseTestSuite ()
    : TestSuite ("tcp-close", UNIT)
  {
    AddTestCase (new TcpCloseWithLossTestCase (true), TestCase::QUICK);
    AddTestCase (new TcpCloseWithLossTestCase (false), TestCase::QUICK);
  }

};

static TcpTcpCloseTestSuite g_tcpTcpCloseTestSuite; //!< Static variable for test initialization

} // namespace ns3
