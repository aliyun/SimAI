/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#include "uan-mac-rc.h"
#include "uan-header-rc.h"
#include "uan-tx-mode.h"
#include "uan-phy.h"
#include "uan-header-common.h"
#include "uan-phy-dual.h"

#include "ns3/log.h"
#include "ns3/nstime.h"
#include "ns3/simulator.h"
#include "ns3/assert.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"

#include <list>
#include <utility>



namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("UanMacRc");

NS_OBJECT_ENSURE_REGISTERED (UanMacRc);

Reservation::Reservation ()
  : m_length (0),
    m_frameNo (0),
    m_retryNo (0),
    m_transmitted (false)
{

}

Reservation::Reservation (std::list<std::pair <Ptr<Packet>, Mac8Address > > &list, uint8_t frameNo, uint32_t maxPkts)
  : m_frameNo (frameNo),
    m_retryNo (0),
    m_transmitted (false)
{
  uint32_t numPkts = (maxPkts) ? maxPkts : static_cast<uint32_t> (list.size ());
  uint32_t length = 0;
  UanHeaderRcData dh;
  UanHeaderCommon ch;

  for (uint32_t i = 0; i < numPkts; i++)
    {
      length += list.front ().first->GetSize () +
        ch.GetSerializedSize () +
        dh.GetSerializedSize ();
      m_pktList.push_back (list.front ());
      list.pop_front ();
    }
  m_length = length;
}

Reservation::~Reservation ()
{
  std::list<std::pair <Ptr<Packet>, Mac8Address > >::iterator it;
  for (it = m_pktList.begin (); it != m_pktList.end (); it++)
    {
      it->first = Ptr<Packet> ((Packet *) 0);
    }
  m_pktList.clear ();
  m_timestamp.clear ();
}
uint32_t
Reservation::GetNoFrames () const
{
  return static_cast<uint32_t> (m_pktList.size ());
}

uint32_t
Reservation::GetLength () const
{
  return m_length;
}

const std::list<std::pair <Ptr<Packet>, Mac8Address > > &
Reservation::GetPktList (void) const
{
  return m_pktList;
}

uint8_t
Reservation::GetFrameNo () const
{
  return m_frameNo;
}

uint8_t
Reservation::GetRetryNo () const
{
  return m_retryNo;
}

Time
Reservation::GetTimestamp (uint8_t n) const
{
  return m_timestamp[n];
}

bool
Reservation::IsTransmitted () const
{
  return m_transmitted;
}

void
Reservation::SetFrameNo (uint8_t fn)
{
  m_frameNo = fn;
}

void
Reservation::AddTimestamp (Time t)
{
  m_timestamp.push_back (t);
}

void
Reservation::IncrementRetry ()
{
  m_retryNo++;
}

void
Reservation::SetTransmitted ([[maybe_unused]] bool t)
{
  m_transmitted = true;
}

uint32_t UanMacRc::m_cntrlSends = 0;

UanMacRc::UanMacRc ()
  : UanMac (),
    m_state (UNASSOCIATED),
    m_rtsBlocked (false),
    m_currentRate (10),
    m_frameNo (0),
    m_cleared (false)
{
  m_ev = CreateObject<ExponentialRandomVariable> ();

  UanHeaderCommon ch;
  UanHeaderRcCts ctsh;
  UanHeaderRcCtsGlobal ctsg;

  m_ctsSizeN = ctsh.GetSerializedSize ();
  m_ctsSizeG = ch.GetSerializedSize () + ctsg.GetSerializedSize ();
}

UanMacRc::~UanMacRc ()
{
}

void
UanMacRc::Clear ()
{
  if (m_cleared)
    {
      return;
    }
  m_cleared = true;
  if (m_phy)
    {
      m_phy->Clear ();
      m_phy = 0;
    }
  std::list<std::pair <Ptr<Packet>, Mac8Address > >::iterator it;
  for (it = m_pktQueue.begin (); it != m_pktQueue.end (); it++)
    {
      it->first = 0;
    }
  m_pktQueue.clear ();
  m_resList.clear ();
  m_startAgain.Cancel ();
  m_rtsEvent.Cancel ();
}

void
UanMacRc::DoDispose ()
{
  Clear ();
  UanMac::DoDispose ();
}

TypeId
UanMacRc::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::UanMacRc")
    .SetParent<UanMac> ()
    .SetGroupName ("Uan")
    .AddConstructor<UanMacRc> ()
    .AddAttribute ("RetryRate",
                   "Number of retry attempts per second (of RTS/GWPING).",
                   DoubleValue (1 / 5.0),
                   MakeDoubleAccessor (&UanMacRc::m_retryRate),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("MaxFrames",
                   "Maximum number of frames to include in a single RTS.",
                   UintegerValue (1),
                   MakeUintegerAccessor (&UanMacRc::m_maxFrames),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("QueueLimit",
                   "Maximum packets to queue at MAC.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&UanMacRc::m_queueLimit),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("SIFS",
                   "Spacing to give between frames (this should match gateway).",
                   TimeValue (Seconds (0.2)),
                   MakeTimeAccessor (&UanMacRc::m_sifs),
                   MakeTimeChecker ())
    .AddAttribute ("NumberOfRates",
                   "Number of rate divisions supported by each PHY.",
                   UintegerValue (0),
                   MakeUintegerAccessor (&UanMacRc::m_numRates),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MinRetryRate",
                   "Smallest allowed RTS retry rate.",
                   DoubleValue (0.01),
                   MakeDoubleAccessor (&UanMacRc::m_minRetryRate),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("RetryStep",
                   "Retry rate increment.",
                   DoubleValue (0.01),
                   MakeDoubleAccessor (&UanMacRc::m_retryStep),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("MaxPropDelay",
                   "Maximum possible propagation delay to gateway.",
                   TimeValue (Seconds (2)),
                   MakeTimeAccessor (&UanMacRc::m_learnedProp),
                   MakeTimeChecker ())
    .AddTraceSource ("Enqueue",
                     "A  (data) packet arrived at MAC for transmission.",
                     MakeTraceSourceAccessor (&UanMacRc::m_enqueueLogger),
                     "ns3::UanMacRc::QueueTracedCallback")
    .AddTraceSource ("Dequeue",
                     "A  (data) packet was passed down to PHY from MAC.",
                     MakeTraceSourceAccessor (&UanMacRc::m_dequeueLogger),
                     "ns3::UanMacRc::QueueTracedCallback")
    .AddTraceSource ("RX",
                     "A packet was destined for and received at this MAC layer.",
                     MakeTraceSourceAccessor (&UanMacRc::m_rxLogger),
                     "ns3::UanMac::PacketModeTracedCallback")
  ;
  return tid;
}

int64_t
UanMacRc::AssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this << stream);
  m_ev->SetStream (stream);
  return 1;
}

bool
UanMacRc::Enqueue (Ptr<Packet> packet, uint16_t protocolNumber, const Address &dest)
{
  if (protocolNumber > 0)
    {
      NS_LOG_WARN ("Warning: UanMacRc does not support multiple protocols.  protocolNumber argument to Enqueue is being ignored");
    }


  if (m_pktQueue.size () >= m_queueLimit)
    {
      return false;
    }

  m_pktQueue.push_back (std::make_pair (packet, Mac8Address::ConvertFrom (dest)));

  switch (m_state)
    {
    case UNASSOCIATED:
      Associate ();
      return true;
    case IDLE:
      if (!m_rtsEvent.IsRunning ())
        {
          SendRts ();
        }
      return true;
    case GWPSENT:
    case RTSSENT:
    case DATATX:
      return true;
    }

  return true;
}

void
UanMacRc::SetForwardUpCb (Callback<void, Ptr<Packet>, uint16_t, const Mac8Address&> cb)
{
  m_forwardUpCb = cb;
}

void
UanMacRc::AttachPhy (Ptr<UanPhy> phy)
{
  m_phy = phy;
  m_phy->SetReceiveOkCallback (MakeCallback (&UanMacRc::ReceiveOkFromPhy, this));
}

void
UanMacRc::ReceiveOkFromPhy (Ptr<Packet> pkt, [[maybe_unused]] double sinr, UanTxMode mode)
{
  UanHeaderCommon ch;
  pkt->RemoveHeader (ch);
  if (ch.GetDest () == Mac8Address::ConvertFrom (GetAddress ()) || ch.GetDest () == Mac8Address::GetBroadcast ())
    {
      m_rxLogger (pkt, mode);
    }

  switch (ch.GetType ())
    {
    case TYPE_DATA:

      if (ch.GetDest () == Mac8Address::ConvertFrom (GetAddress ()))
        {
          NS_LOG_DEBUG (Now ().As (Time::S) << " Node " << Mac8Address::ConvertFrom (GetAddress ()) << 
                        " UanMacRc Receiving DATA packet from PHY");
          UanHeaderRcData dh;
          pkt->RemoveHeader (dh);
          m_forwardUpCb (pkt, ch.GetProtocolNumber (), ch.GetSrc ());
        }
      break;
    case TYPE_RTS:
      // Currently don't respond to RTS packets at non-gateway nodes
      // (Code assumes single network neighberhood)
      break;
    case TYPE_CTS:
      {
        uint32_t ctsBytes = ch.GetSerializedSize () + pkt->GetSize ();
        m_assocAddr = ch.GetSrc ();
        UanHeaderRcCtsGlobal ctsg;
        pkt->RemoveHeader (ctsg);
        m_currentRate = ctsg.GetRateNum ();
        m_retryRate = m_minRetryRate + m_retryStep*ctsg.GetRetryRate ();

        UanHeaderRcRts rhtmp;

        Time winDelay = ctsg.GetWindowTime ();

        if (winDelay > Time (0))
          {
            m_rtsBlocked = false;
            Simulator::Schedule (winDelay, &UanMacRc::BlockRtsing, this);
          }
        else
          {
            NS_FATAL_ERROR (Now ().As (Time::S) << " Node " << 
                            Mac8Address::ConvertFrom (GetAddress ()) << " Received window period < 0");
          }

        UanHeaderRcCts ctsh;
        ctsh.SetAddress (Mac8Address::GetBroadcast ());
        while (pkt->GetSize () > 0)
          {
            pkt->RemoveHeader (ctsh);
            if (ctsh.GetAddress () == Mac8Address::ConvertFrom (GetAddress ()))
              {
                if (m_state == GWPSENT)
                  {
                    m_assocAddr = ch.GetSrc ();
                    ScheduleData (ctsh, ctsg, ctsBytes);
                  }
                else if (m_state == RTSSENT)
                  {
                    ScheduleData (ctsh, ctsg, ctsBytes);
                  }
                else
                  {
                    NS_LOG_DEBUG (Now ().As (Time::S) << " Node " << 
                                  Mac8Address::ConvertFrom (GetAddress ()) << 
                                  " received CTS while state != RTSSENT or GWPING");
                  }
              }
          }
      }
      break;
    case TYPE_GWPING:
      // Do not respond to GWPINGS at non-gateway nodes
      break;
    case TYPE_ACK:
      m_rtsBlocked = true;
      if (ch.GetDest () != Mac8Address::ConvertFrom (GetAddress ()))
        {
          return;
        }
      ProcessAck (pkt);
      break;
    default:
      NS_FATAL_ERROR ("Unknown packet type " << ch.GetType () << " received at node " << GetAddress ());
    }

}

void
UanMacRc::ScheduleData (const UanHeaderRcCts &ctsh, const UanHeaderRcCtsGlobal &ctsg, uint32_t ctsBytes)
{
  NS_ASSERT (m_state == RTSSENT || m_state == GWPSENT);



  std::list<Reservation>::iterator it = m_resList.begin ();
  for (; it != m_resList.end (); it++)
    {
      if (it->GetFrameNo () == ctsh.GetFrameNo ())
        {
          break;
        }
    }
  if (it == m_resList.end ())
    {
      NS_LOG_DEBUG (Now ().As (Time::S) << " Node " << 
                    Mac8Address::ConvertFrom (GetAddress ()) << 
                    " received CTS packet with no corresponding reservation!");
      return;
    }
  NS_LOG_DEBUG (Now ().As (Time::S) << 
                " Node " << Mac8Address::ConvertFrom (GetAddress ()) << 
                " received CTS packet.  Scheduling data");
  it->SetTransmitted ();

  double currentBps = m_phy->GetMode (m_currentRate).GetDataRateBps ();

  m_learnedProp = Simulator::Now () - ctsg.GetTxTimeStamp () - Seconds (ctsBytes * 8.0 / currentBps);


  Time arrTime = ctsg.GetTxTimeStamp () + ctsh.GetDelayToTx ();
  Time txTime = arrTime - m_learnedProp;

  Time startDelay = txTime - Simulator::Now ();

  Time frameDelay = Seconds (0);

  const std::list<std::pair <Ptr<Packet>, Mac8Address > > l = it->GetPktList ();
  std::list<std::pair <Ptr<Packet>, Mac8Address > >::const_iterator pit;
  pit = l.begin ();



  for (uint8_t i = 0; i < it->GetNoFrames (); i++, pit++)
    {
      Ptr<Packet> pkt = (*pit).first->Copy ();

      UanHeaderRcData dh;
      dh.SetFrameNo (i);
      dh.SetPropDelay (m_learnedProp);
      pkt->AddHeader (dh);

      UanHeaderCommon ch;
      ch.SetType (TYPE_DATA);
      ch.SetDest (m_assocAddr);
      ch.SetSrc (Mac8Address::ConvertFrom (GetAddress ()));

      pkt->AddHeader (ch);
      Time eventTime = startDelay + frameDelay;
      if (eventTime < Time (0))
        {
          NS_FATAL_ERROR ("Scheduling error resulted in very negative data transmission time! eventTime = " << eventTime.As (Time::S));
        }
      NS_LOG_DEBUG (Now ().As (Time::S) << 
                    " Node " << Mac8Address::ConvertFrom (GetAddress ()) << 
                    " scheduling with delay " << eventTime.As (Time::S) << 
                    " propDelay " << m_learnedProp.As (Time::S) << 
                    " start delay " << startDelay.As (Time::S) << 
                    " arrival time " << arrTime.As (Time::S));
      Simulator::Schedule (eventTime, &UanMacRc::SendPacket, this, pkt, m_currentRate);
      frameDelay = frameDelay + m_sifs + Seconds (pkt->GetSize () / currentBps);
    }

  m_state = IDLE;
  if (!m_pktQueue.empty ())
    {

      if (m_rtsEvent.IsRunning ())
        {
          m_rtsEvent.Cancel ();
        }

      m_ev->SetAttribute ("Mean", DoubleValue (1 / m_retryRate));
      double timeout = m_ev->GetValue ();
      m_rtsEvent = Simulator::Schedule (Seconds (timeout), &UanMacRc::SendRts, this);
    }

}

void
UanMacRc::SendPacket (Ptr<Packet> pkt, uint32_t rate)
{
  UanHeaderCommon ch;
  pkt->PeekHeader (ch);
  std::string type;
  switch (ch.GetType ())
    {
    case TYPE_DATA:
      type = "DATA";
      break;
    case TYPE_RTS:
      type = "RTS";
      break;
    case TYPE_CTS:
      type = "CTS";
      break;
    case TYPE_ACK:
      type = "ACK";
      break;
    case TYPE_GWPING:
      type = "GWPING";
      break;
    default:
      type = "UNKNOWN";
      break;
    }
  NS_LOG_DEBUG (Now ().As (Time::S) << 
                " Node " << Mac8Address::ConvertFrom (GetAddress ()) << 
                " transmitting " << pkt->GetSize () << 
                " byte packet of type " << type << " with rate " << rate << 
                "(" << m_phy->GetMode (rate).GetDataRateBps () << ") to " << ch.GetDest ());
  m_dequeueLogger (pkt, rate);
  m_phy->SendPacket (pkt, rate);
}

void
UanMacRc::ProcessAck (Ptr<Packet> ack)
{
  UanHeaderRcAck ah;
  ack->RemoveHeader (ah);

  std::list<Reservation>::iterator it = m_resList.begin ();
  for (; it != m_resList.end (); it++)
    {
      if (it->GetFrameNo () == ah.GetFrameNo ())
        {
          break;
        }
    }
  if (it == m_resList.end ())
    {
      NS_LOG_DEBUG ("In " << __func__ << " could not find reservation corresponding to received ACK");
      return;
    }
  if (!it->IsTransmitted ())
    {
      return;
    }
  if (ah.GetNoNacks () > 0)
    {
      const std::list<std::pair <Ptr<Packet>, Mac8Address > > l = it->GetPktList ();
      std::list<std::pair <Ptr<Packet>, Mac8Address > >::const_iterator pit;
      pit = l.begin ();

      const std::set<uint8_t> &nacks = ah.GetNackedFrames ();
      std::set<uint8_t>::iterator nit = nacks.begin ();
      uint8_t pnum = 0;
      for (; nit != nacks.end (); nit++)
        {
          NS_LOG_DEBUG (Now ().As (Time::S) << " Node " << 
                        Mac8Address::ConvertFrom (GetAddress ()) << 
                        " Received NACK for " << (uint32_t) *nit);
          while (pnum < *nit)
            {
              pit++;
              pnum++;
            }
          UanHeaderRcData dh;
          UanHeaderCommon ch;
          m_pktQueue.push_front (*pit);
        }
    }
  else
    {
      NS_LOG_DEBUG (Now ().As (Time::S) << " Node " << 
                    Mac8Address::ConvertFrom (GetAddress ()) << 
                    " received ACK for all frames");
    }
  m_resList.erase (it);
}

UanHeaderRcRts
UanMacRc::CreateRtsHeader (const Reservation &res)
{
  UanHeaderRcRts rh = UanHeaderRcRts ();

  rh.SetLength (static_cast<uint16_t> (res.GetLength ()));
  rh.SetNoFrames (static_cast<uint8_t> (res.GetNoFrames ()));
  rh.SetTimeStamp (res.GetTimestamp (res.GetRetryNo ()));
  rh.SetFrameNo (res.GetFrameNo ());
  rh.SetRetryNo (res.GetRetryNo ());
  return rh;
}

void
UanMacRc::Associate (void)
{
  m_cntrlSends++;

  Reservation res (m_pktQueue, m_frameNo, m_maxFrames);
  res.AddTimestamp (Simulator::Now ());
  m_frameNo++;
  m_resList.push_back (res);
  Ptr<UanPhyDual> phyDual = m_phy->GetObject<UanPhyDual> ();
  bool phy1ok = IsPhy1Ok ();
  if (phy1ok && !phyDual->IsPhy2Tx () & !m_rtsBlocked)
    {
      Ptr<Packet> pkt = Create<Packet> (0);
      pkt->AddHeader (CreateRtsHeader (res));
      pkt->AddHeader (UanHeaderCommon (Mac8Address::ConvertFrom (GetAddress ()), Mac8Address::GetBroadcast (), static_cast<uint8_t>(TYPE_GWPING), 0));
      NS_LOG_DEBUG (Now ().As (Time::S) << " Sending first GWPING " << *pkt);
      SendPacket (pkt,m_currentRate + m_numRates);
    }
  m_state = GWPSENT;
  NS_ASSERT (!m_rtsEvent.IsRunning ());
  m_ev->SetAttribute ("Mean", DoubleValue (1 / m_retryRate));
  double timeout = m_ev->GetValue ();
  m_rtsEvent = Simulator::Schedule (Seconds (timeout), &UanMacRc::AssociateTimeout, this);
}

void
UanMacRc::AssociateTimeout ()
{
  m_cntrlSends++;
  if (m_state != GWPSENT)
    {
      return;
    }
  Ptr<UanPhyDual> phyDual = m_phy->GetObject<UanPhyDual> ();
  bool phy1ok = IsPhy1Ok ();
  if (phy1ok && !phyDual->IsPhy2Tx () && !m_rtsBlocked)
    {
      Ptr<Packet> pkt = Create<Packet> ();

      Reservation res = m_resList.back ();
      m_resList.pop_back ();
      res.AddTimestamp (Simulator::Now ());
      res.IncrementRetry ();

      pkt->AddHeader (CreateRtsHeader (res));
      pkt->AddHeader (UanHeaderCommon (Mac8Address::ConvertFrom (GetAddress ()), Mac8Address::GetBroadcast (), static_cast<uint8_t> (TYPE_GWPING), 0));

      SendPacket (pkt,m_currentRate + m_numRates);
      m_resList.push_back (res);
    }
  NS_ASSERT (!m_rtsEvent.IsRunning ());
  m_ev->SetAttribute ("Mean", DoubleValue (1 / m_retryRate));
  double timeout = m_ev->GetValue ();
  m_rtsEvent = Simulator::Schedule (Seconds (timeout), &UanMacRc::AssociateTimeout, this);
}


void
UanMacRc::SendRts (void)
{
  m_cntrlSends++;
  if (m_state == RTSSENT)
    {
      return;
    }

  NS_ASSERT (!m_pktQueue.empty ());

  Reservation res (m_pktQueue, m_frameNo, m_maxFrames);
  res.AddTimestamp (Simulator::Now ());
  m_frameNo++;
  m_resList.push_back (res);
  Ptr<UanPhyDual> phyDual = m_phy->GetObject<UanPhyDual> ();
  bool phy1ok = IsPhy1Ok ();
  if (phy1ok && !phyDual->IsPhy2Tx () && !m_rtsBlocked )
    {
      Ptr<Packet> pkt = Create<Packet> (0);
      pkt->AddHeader (CreateRtsHeader (res));
      pkt->AddHeader (UanHeaderCommon (Mac8Address::ConvertFrom (GetAddress ()), Mac8Address::GetBroadcast (), static_cast<uint8_t> (TYPE_RTS), 0));
      SendPacket (pkt,m_currentRate + m_numRates);
    }
  m_state = RTSSENT;
  NS_ASSERT (!m_rtsEvent.IsRunning ());
  m_ev->SetAttribute ("Mean", DoubleValue (1 / m_retryRate));
  double timeout = m_ev->GetValue ();
  m_rtsEvent = Simulator::Schedule (Seconds (timeout), &UanMacRc::RtsTimeout, this);

}

// We assume here that packet types are known at detection.
bool
UanMacRc::IsPhy1Ok ()
{
  Ptr<UanPhyDual> phyDual = m_phy->GetObject<UanPhyDual> ();

  bool phy1ok = true;
  if (phyDual->IsPhy1Rx ())
    {
      Ptr<Packet> pkt = phyDual->GetPhy1PacketRx ();
      UanHeaderCommon ch;
      pkt->PeekHeader (ch);
      if (ch.GetType () == TYPE_CTS || ch.GetType () == TYPE_ACK)
        {
          phy1ok = false;
        }
      else if (ch.GetDest () == Mac8Address::ConvertFrom (GetAddress ()))
        {
          phy1ok = false;
        }
    }
  return phy1ok;
}

void
UanMacRc::RtsTimeout (void)
{
  m_cntrlSends++;

  if (m_state != RTSSENT)
    {
      return;
    }
  Ptr<UanPhyDual> phyDual = m_phy->GetObject<UanPhyDual> ();

  bool phy1ok = IsPhy1Ok ();
  if (phy1ok && !phyDual->IsPhy2Tx () && !m_rtsBlocked)
    {

      if (m_resList.empty ())
        {
          NS_FATAL_ERROR (Now ().As (Time::S) << " Node " << 
                          Mac8Address::ConvertFrom (GetAddress ()) << 
                          " tried to retry RTS with empty reservation list");
        }
      Ptr<Packet> pkt = Create<Packet> (0);

      Reservation res = m_resList.back ();
      NS_ASSERT (!res.IsTransmitted ());
      m_resList.pop_back ();
      res.AddTimestamp (Simulator::Now ());
      res.IncrementRetry ();
      m_resList.push_back (res);
      pkt->AddHeader (CreateRtsHeader (res));
      pkt->AddHeader (UanHeaderCommon (Mac8Address::ConvertFrom (GetAddress ()), Mac8Address::GetBroadcast (), static_cast<uint8_t> (TYPE_RTS), 0));
      SendPacket (pkt,m_currentRate + m_numRates);

    }
  m_state = RTSSENT;
  NS_ASSERT (!m_rtsEvent.IsRunning ());
  m_ev->SetAttribute ("Mean", DoubleValue (1 / m_retryRate));
  double timeout = m_ev->GetValue ();
  m_rtsEvent = Simulator::Schedule (Seconds (timeout), &UanMacRc::RtsTimeout, this);
}

void 
UanMacRc::BlockRtsing (void)
{
  m_rtsBlocked = true;
}

} // namespace ns3
