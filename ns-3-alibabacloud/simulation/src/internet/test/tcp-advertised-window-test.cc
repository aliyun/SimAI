/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Christoph Doepmann <doepmanc@informatik.hu-berlin.de>
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
#include "tcp-error-model.h"
#include "ns3/random-variable-stream.h"
#include "ns3/tcp-rx-buffer.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpAdvertisedWindowTestSuite");

/**
 * \ingroup internet-tests
 * \ingroup tests
 * \brief Socket that wraps every call to AdvertisedWindowSize ().
 */

class TcpSocketAdvertisedWindowProxy : public TcpSocketMsgBase
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /** \brief typedef for a cb */
  typedef Callback<void, uint16_t, uint16_t> InvalidAwndCallback;

  /**
   * \brief Constructor
   */
  TcpSocketAdvertisedWindowProxy () : TcpSocketMsgBase (), m_segmentSize(0)
  {
  }

  /**
   * \brief Copy-constructor
   *
   * \param other Other obj
   */
  TcpSocketAdvertisedWindowProxy (const TcpSocketAdvertisedWindowProxy &other)
    : TcpSocketMsgBase (other)
  {
    m_segmentSize = other.m_segmentSize;
    m_inwalidAwndCb = other.m_inwalidAwndCb;
  }

  /**
   * \brief Set the invalid AdvWnd callback
   *
   * \param cb callback to set
   */
  void SetInvalidAwndCb (InvalidAwndCallback cb);

  /**
   * \brief Set the expected segment size
   *
   * \param seg Segment size
   */
  void SetExpectedSegmentSize (uint16_t seg) { m_segmentSize = seg; };

protected:
  virtual Ptr<TcpSocketBase> Fork ();
  virtual uint16_t AdvertisedWindowSize (bool scale = true) const;

private:
  uint16_t OldAdvertisedWindowSize (bool scale = true) const;
  InvalidAwndCallback m_inwalidAwndCb; //!< Callback

  /**
   * \brief Test meta-information: size of the segments that are received.
   * 
   * This is necessary for making sure the calculated awnd only differs by
   * exactly that one segment that was not yet read by the application.
   */
  uint16_t m_segmentSize;
};

void
TcpSocketAdvertisedWindowProxy::SetInvalidAwndCb (InvalidAwndCallback cb)
{
  NS_ASSERT (!cb.IsNull ());
  m_inwalidAwndCb = cb;
}

TypeId
TcpSocketAdvertisedWindowProxy::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpSocketAdvertisedWindowProxy")
    .SetParent<TcpSocketMsgBase> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpSocketAdvertisedWindowProxy> ()
  ;
  return tid;
}

Ptr<TcpSocketBase>
TcpSocketAdvertisedWindowProxy::Fork (void)
{
  return CopyObject<TcpSocketAdvertisedWindowProxy> (this);
}

uint16_t
TcpSocketAdvertisedWindowProxy::AdvertisedWindowSize (bool scale) const
{
  NS_LOG_FUNCTION (this << scale);

  uint16_t newAwnd = TcpSocketMsgBase::AdvertisedWindowSize (scale);
  uint16_t oldAwnd = OldAdvertisedWindowSize (scale);

  if (!m_tcb->m_rxBuffer->Finished ())
    {
      // The calculated windows will only be exactly equal if there is no data
      // in the receive buffer yet.
      if (newAwnd != oldAwnd)
        {
          uint32_t available = m_tcb->m_rxBuffer->Available ();
          // If the values differ, make sure this is only due to the single segment
          // the socket just got, which has not yet been read by the application.
          // Therefore, the difference should be exactly the size of one segment
          // (but taking scale and m_maxWinSize into account).
          uint32_t newAwndKnownDifference = newAwnd;
          if (scale)
            {
              newAwndKnownDifference += (available >> m_rcvWindShift);
            }
          else
            {
              newAwndKnownDifference += available;
            }

          if (newAwndKnownDifference > m_maxWinSize)
            {
              newAwndKnownDifference = m_maxWinSize;
            }

          if (static_cast<uint16_t> (newAwndKnownDifference) != oldAwnd)
            {
              if (!m_inwalidAwndCb.IsNull ())
                {
                  m_inwalidAwndCb(oldAwnd, newAwnd);
                }

            }
        }
    }


  return newAwnd;
}

/**
 * The legacy code used for calculating the advertised window.
 *
 * This was copied from tcp-socket-base.cc before changing the formula.
 * \param scale true if should scale the window
 * \return the old adv wnd value
 */
uint16_t
TcpSocketAdvertisedWindowProxy::OldAdvertisedWindowSize (bool scale) const
{
  NS_LOG_FUNCTION (this << scale);
  //NS_LOG_DEBUG ("MaxRxSequence () = " << m_tcb->m_rxBuffer->MaxRxSequence ());
  //NS_LOG_DEBUG ("NextRxSequence () = " << m_tcb->m_rxBuffer->NextRxSequence ());
  //NS_LOG_DEBUG ("MaxBufferSize () = " << m_tcb->m_rxBuffer->MaxBufferSize ());
  //NS_LOG_DEBUG ("m_rcvWindShift = " << static_cast<uint16_t> (m_rcvWindShift));
  //NS_LOG_DEBUG ("m_maxWinSize = " << m_maxWinSize);
  //NS_LOG_DEBUG ("Available () = " << m_tcb->m_rxBuffer->Available ());
  uint32_t w = m_tcb->m_rxBuffer->MaxBufferSize ();

  if (scale)
    {
      w >>= m_rcvWindShift;
    }
  if (w > m_maxWinSize)
    {
      w = m_maxWinSize;
      NS_LOG_WARN ("Adv window size truncated to " << m_maxWinSize << "; possibly to avoid overflow of the 16-bit integer");
    }
  NS_LOG_DEBUG ("Returning AdvertisedWindowSize of " << static_cast<uint16_t> (w));
  return static_cast<uint16_t> (w);
}

NS_OBJECT_ENSURE_REGISTERED (TcpSocketAdvertisedWindowProxy);

/**
 * \ingroup internet-tests
 * \ingroup test
 *
 * \brief An error model that randomly drops a given rátio of TCP segments.
 */
class TcpDropRatioErrorModel : public TcpGeneralErrorModel
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  /**
   * \brief Constructor
   * \param dropRatio the drop ratio
   */
  TcpDropRatioErrorModel (double dropRatio)
    : TcpGeneralErrorModel (), m_dropRatio(dropRatio)
  {
    m_prng = CreateObject<UniformRandomVariable> ();
  }

protected:
  virtual bool ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                           uint32_t packetSize);

private:
  virtual void DoReset (void) { };
  double m_dropRatio; //!< Drop ratio
  Ptr<UniformRandomVariable> m_prng; //!< Random variable
};

NS_OBJECT_ENSURE_REGISTERED (TcpDropRatioErrorModel);

TypeId
TcpDropRatioErrorModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::TcpDropRatioErrorModel")
    .SetParent<TcpGeneralErrorModel> ()
  ;
  return tid;
}

bool
TcpDropRatioErrorModel::ShouldDrop (const Ipv4Header &ipHeader, const TcpHeader &tcpHeader,
                              uint32_t packetSize)
{
  return m_prng->GetValue () < m_dropRatio; 
}

/**
 * \ingroup internet-tests
 * \ingroup test
 * \brief Test the new formula for calculating TCP's advertised window size.
 *
 * In TcpSocketBase, the advertised window is now calculated as
 *
 *   m_tcb->m_rxBuffer->MaxRxSequence () - m_tcb->m_rxBuffer->NextRxSequence ()
 * 
 * instead of the previous
 *
 *   m_tcb->m_rxBuffer->MaxBufferSize ()
 * 
 * This change was introduced with regard to situations in which the receiviing
 * application does not read from the socket as fast as possible (see bug 2559
 * for details). This test ensures that no regression is introduced for other,
 * "normal" cases.
 *
 * TcpGeneralTest ensures via ReceivePacket () that incoming packets are
 * quickly consumed by the application layer, simulating a fast-reading
 * application. We can only reasonably compare the old and the new AWND
 * computation in this case.
 */
class TcpAdvertisedWindowTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   * \param desc description
   * \param size segment size
   * \param packets number of packets to send
   * \param lossRatio error ratio
   */
  TcpAdvertisedWindowTest (const std::string &desc, uint32_t size, uint32_t packets, double lossRatio);

protected:
  virtual void ConfigureEnvironment ();
  virtual Ptr<TcpSocketMsgBase> CreateReceiverSocket (Ptr<Node> node);
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();

private:
  /** \brief Callback called for the update of the awnd
   * \param oldAwnd Old advertised window
   * \param newAwnd new value
   */
  void InvalidAwndCb (uint16_t oldAwnd, uint16_t newAwnd);
  uint32_t m_pktSize; //!< Packet size
  uint32_t m_pktCount; //!< Pkt count
  double m_lossRatio; //!< Loss ratio
};

TcpAdvertisedWindowTest::TcpAdvertisedWindowTest (const std::string &desc,
                                              uint32_t size, uint32_t packets, double lossRatio)
  : TcpGeneralTest (desc),
    m_pktSize (size),
    m_pktCount (packets),
    m_lossRatio (lossRatio)
{
}

void
TcpAdvertisedWindowTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (m_pktCount);
  SetPropagationDelay (MilliSeconds (50));
  SetTransmitStart (Seconds (2.0));
  SetAppPktSize (m_pktSize);

}

Ptr<TcpSocketMsgBase>
TcpAdvertisedWindowTest::CreateReceiverSocket (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);

  Ptr<TcpSocketMsgBase> sock = CreateSocket (node, TcpSocketAdvertisedWindowProxy::GetTypeId (), m_congControlTypeId);
  DynamicCast<TcpSocketAdvertisedWindowProxy> (sock)->SetExpectedSegmentSize (500);
  DynamicCast<TcpSocketAdvertisedWindowProxy> (sock)->SetInvalidAwndCb (
        MakeCallback (&TcpAdvertisedWindowTest::InvalidAwndCb, this));

  return sock;
}

Ptr<ErrorModel>
TcpAdvertisedWindowTest::CreateReceiverErrorModel ()
{
  return CreateObject<TcpDropRatioErrorModel> (m_lossRatio);
}

void
TcpAdvertisedWindowTest::InvalidAwndCb (uint16_t oldAwnd, uint16_t newAwnd)
{
  NS_TEST_ASSERT_MSG_EQ (oldAwnd, newAwnd,
                         "Old and new AWND calculations do not match.");
}
//-----------------------------------------------------------------------------

/**
 * \ingroup internet-tests
 * \ingroup test
 * \brief Test the TCP's advertised window size when there is a loss of specific packets.
 */
class TcpAdvWindowOnLossTest : public TcpGeneralTest
{
public:
  /**
   * \brief Constructor
   * \param desc description
   * \param size segment size
   * \param packets number of packets to send
   * \param toDrop packets to be dropped
   */
  TcpAdvWindowOnLossTest (const std::string &desc, uint32_t size, uint32_t packets,
                          std::vector<uint32_t> &toDrop);

protected:
  virtual void ConfigureEnvironment ();
  virtual Ptr<TcpSocketMsgBase> CreateReceiverSocket (Ptr<Node> node);
  virtual Ptr<TcpSocketMsgBase> CreateSenderSocket (Ptr<Node> node);
  virtual Ptr<ErrorModel> CreateReceiverErrorModel ();

private:
  /** \brief Callback called for the update of the awnd
   * \param oldAwnd Old advertised window
   * \param newAwnd new value
   */
  void InvalidAwndCb (uint16_t oldAwnd, uint16_t newAwnd);
  uint32_t m_pktSize; //!< Packet size
  uint32_t m_pktCount; //!< Pkt count
  std::vector<uint32_t> m_toDrop; //!< Sequences to drop
};

TcpAdvWindowOnLossTest::TcpAdvWindowOnLossTest (const std::string &desc,
                                                uint32_t size, uint32_t packets,
                                                std::vector<uint32_t> &toDrop)
  : TcpGeneralTest (desc),
    m_pktSize (size),
    m_pktCount (packets),
    m_toDrop (toDrop)
{
}

void
TcpAdvWindowOnLossTest::ConfigureEnvironment ()
{
  TcpGeneralTest::ConfigureEnvironment ();
  SetAppPktCount (m_pktCount);
  SetPropagationDelay (MilliSeconds (50));
  SetTransmitStart (Seconds (2.0));
  SetAppPktSize (m_pktSize);
}

Ptr<TcpSocketMsgBase>
TcpAdvWindowOnLossTest::CreateReceiverSocket (Ptr<Node> node)
{
  NS_LOG_FUNCTION (this);

  Ptr<TcpSocketMsgBase> sock = CreateSocket (node, TcpSocketAdvertisedWindowProxy::GetTypeId (), m_congControlTypeId);
  DynamicCast<TcpSocketAdvertisedWindowProxy> (sock)->SetExpectedSegmentSize (500);
  DynamicCast<TcpSocketAdvertisedWindowProxy> (sock)->SetInvalidAwndCb (
        MakeCallback (&TcpAdvWindowOnLossTest::InvalidAwndCb, this));

  return sock;
}

Ptr<TcpSocketMsgBase>
TcpAdvWindowOnLossTest::CreateSenderSocket (Ptr<Node> node)
{
  auto socket = TcpGeneralTest::CreateSenderSocket (node);
  socket->SetAttribute("InitialCwnd", UintegerValue (10*m_pktSize));

  return socket;
}

Ptr<ErrorModel>
TcpAdvWindowOnLossTest::CreateReceiverErrorModel ()
{
  Ptr<TcpSeqErrorModel> m_errorModel = CreateObject<TcpSeqErrorModel> ();
  for (std::vector<uint32_t>::iterator it = m_toDrop.begin (); it != m_toDrop.end (); ++it)
    {
      m_errorModel->AddSeqToKill (SequenceNumber32 (*it));
    }

  return m_errorModel;
}

void
TcpAdvWindowOnLossTest::InvalidAwndCb (uint16_t oldAwnd, uint16_t newAwnd)
{
  NS_TEST_ASSERT_MSG_EQ (oldAwnd, newAwnd,
                         "Old and new AWND calculations do not match.");
}

//-----------------------------------------------------------------------------

/**
 * \ingroup internet-tests
 * \ingroup test
 *
 * \brief Test Suite for TCP adv window
 */
class TcpAdvertisedWindowTestSuite : public TestSuite
{
public:
  TcpAdvertisedWindowTestSuite () : TestSuite ("tcp-advertised-window-test", UNIT)
  {
    AddTestCase (new TcpAdvertisedWindowTest ("TCP advertised window size, small seg + no loss", 500, 100, 0.0),
                 TestCase::QUICK);
    AddTestCase (new TcpAdvertisedWindowTest ("TCP advertised window size, small seg + loss", 500, 100, 0.1),
                 TestCase::QUICK);
    AddTestCase (new TcpAdvertisedWindowTest ("TCP advertised window size, large seg + no loss", 1000, 100, 0.0),
                 TestCase::QUICK);
    AddTestCase (new TcpAdvertisedWindowTest ("TCP advertised window size, large seg + small loss", 1000, 100, 0.1),
                 TestCase::QUICK);
    AddTestCase (new TcpAdvertisedWindowTest ("TCP advertised window size, large seg + big loss", 1000, 100, 0.3),
                 TestCase::QUICK);
    AddTestCase (new TcpAdvertisedWindowTest ("TCP advertised window size, complete loss", 1000, 100, 1.0),
                 TestCase::QUICK);

    std::vector<uint32_t> toDrop;
    toDrop.push_back(8001);
    toDrop.push_back(9001);
    AddTestCase (new TcpAdvWindowOnLossTest ("TCP advertised window size, after FIN loss", 1000, 10, toDrop));
  }
};

static TcpAdvertisedWindowTestSuite g_tcpAdvertisedWindowTestSuite; //<! static obj for test initialization

