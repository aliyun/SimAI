/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Mirko Banchi <mk.banchi@gmail.com>
 *          Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/log.h"
#include "wifi-mac-queue-item.h"
#include "wifi-mac-trailer.h"
#include "wifi-utils.h"
#include "msdu-aggregator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiMacQueueItem");

WifiMacQueueItem::WifiMacQueueItem (Ptr<const Packet> p, const WifiMacHeader & header)
  : WifiMacQueueItem (p, header, Simulator::Now ())
{
}

WifiMacQueueItem::WifiMacQueueItem (Ptr<const Packet> p, const WifiMacHeader & header, Time tstamp)
  : m_packet (p),
    m_header (header),
    m_tstamp (tstamp),
    m_queueAc (AC_UNDEF)
{
  if (header.IsQosData () && header.IsQosAmsdu ())
    {
      m_msduList = MsduAggregator::Deaggregate (p->Copy ());
    }
  m_inFlight = false;
}

WifiMacQueueItem::~WifiMacQueueItem ()
{
}

Ptr<WifiMacQueueItem>
WifiMacQueueItem::GetItem (void) const
{
  NS_ASSERT (IsQueued ());
  return *m_queueIt;
}

Ptr<const Packet>
WifiMacQueueItem::GetPacket (void) const
{
  return m_packet;
}

const WifiMacHeader&
WifiMacQueueItem::GetHeader (void) const
{
  return m_header;
}

WifiMacHeader&
WifiMacQueueItem::GetHeader (void)
{
  return m_header;
}

Mac48Address
WifiMacQueueItem::GetDestinationAddress (void) const
{
  return m_header.GetAddr1 ();
}

Time
WifiMacQueueItem::GetTimeStamp (void) const
{
  return m_tstamp;
}

uint32_t
WifiMacQueueItem::GetPacketSize (void) const
{
  return m_packet->GetSize ();
}

uint32_t
WifiMacQueueItem::GetSize (void) const
{
  return GetPacketSize () + m_header.GetSerializedSize () + WIFI_MAC_FCS_LENGTH;
}

bool
WifiMacQueueItem::IsFragment (void) const
{
  return m_header.IsMoreFragments () || m_header.GetFragmentNumber () > 0;
}

Ptr<Packet>
WifiMacQueueItem::GetProtocolDataUnit (void) const
{
  Ptr<Packet> mpdu = m_packet->Copy ();
  mpdu->AddHeader (m_header);
  AddWifiMacTrailer (mpdu);
  return mpdu;
}

void
WifiMacQueueItem::Aggregate (Ptr<const WifiMacQueueItem> msdu)
{
  NS_ASSERT (msdu != 0);
  NS_LOG_FUNCTION (this << *msdu);
  NS_ABORT_MSG_IF (!msdu->GetHeader ().IsQosData () || msdu->GetHeader ().IsQosAmsdu (),
                   "Only QoS data frames that do not contain an A-MSDU can be aggregated");

  if (m_msduList.empty ())
    {
      // An MSDU is going to be aggregated to this MPDU, hence this has to be an A-MSDU now
      Ptr<const WifiMacQueueItem> firstMsdu = Create<const WifiMacQueueItem> (*this);
      m_packet = Create<Packet> ();
      DoAggregate (firstMsdu);

      m_header.SetQosAmsdu ();
      // Set Address3 according to Table 9-26 of 802.11-2016
      if (m_header.IsToDs () && !m_header.IsFromDs ())
        {
          // from STA to AP: BSSID is in Address1
          m_header.SetAddr3 (m_header.GetAddr1 ());
        }
      else if (!m_header.IsToDs () && m_header.IsFromDs ())
        {
          // from AP to STA: BSSID is in Address2
          m_header.SetAddr3 (m_header.GetAddr2 ());
        }
      // in the WDS case (ToDS = FromDS = 1), both Address 3 and Address 4 need
      // to be set to the BSSID, but neither Address 1 nor Address 2 contain the
      // BSSID. Hence, it is left up to the caller to set these Address fields.
    }
  DoAggregate (msdu);
}

void
WifiMacQueueItem::DoAggregate (Ptr<const WifiMacQueueItem> msdu)
{
  NS_LOG_FUNCTION (this << *msdu);

  // build the A-MSDU Subframe header
  AmsduSubframeHeader hdr;
  /*
   * (See Table 9-26 of 802.11-2016)
   *
   * ToDS | FromDS |  DA   |  SA
   *   0  |   0    | Addr1 | Addr2
   *   0  |   1    | Addr1 | Addr3
   *   1  |   0    | Addr3 | Addr2
   *   1  |   1    | Addr3 | Addr4
   */
  hdr.SetDestinationAddr (msdu->GetHeader ().IsToDs () ? msdu->GetHeader ().GetAddr3 ()
                                                       : msdu->GetHeader ().GetAddr1 ());
  hdr.SetSourceAddr (!msdu->GetHeader ().IsFromDs () ? msdu->GetHeader ().GetAddr2 ()
                                                     : (!msdu->GetHeader ().IsToDs ()
                                                        ? msdu->GetHeader ().GetAddr3 ()
                                                        : msdu->GetHeader ().GetAddr4 ()));
  hdr.SetLength (static_cast<uint16_t> (msdu->GetPacket ()->GetSize ()));

  m_msduList.push_back ({msdu->GetPacket (), hdr});

  // build the A-MSDU
  NS_ASSERT (m_packet);
  Ptr<Packet> amsdu = m_packet->Copy ();

  // pad the previous A-MSDU subframe if the A-MSDU is not empty
  if (m_packet->GetSize () > 0)
    {
      uint8_t padding = MsduAggregator::CalculatePadding (m_packet->GetSize ());

      if (padding)
        {
          amsdu->AddAtEnd (Create<Packet> (padding));
        }
    }

  // add A-MSDU subframe header and MSDU
  Ptr<Packet> amsduSubframe = msdu->GetPacket ()->Copy ();
  amsduSubframe->AddHeader (hdr);
  amsdu->AddAtEnd (amsduSubframe);
  m_packet = amsdu;

  /* "The expiration of the A-MSDU lifetime timer occurs only when the lifetime
    * timer of all of the constituent MSDUs of the A-MSDU have expired" (Section
    * 10.12 of 802.11-2016)
    */
  // The timestamp of the A-MSDU is the most recent among those of the MSDUs
  m_tstamp = Max (m_tstamp, msdu->GetTimeStamp ());
}

bool
WifiMacQueueItem::IsQueued (void) const
{
  return m_queueAc != AC_UNDEF;
}

AcIndex
WifiMacQueueItem::GetQueueAc (void) const
{
  NS_ASSERT (IsQueued ());
  return m_queueAc;
}

void
WifiMacQueueItem::SetInFlight (void)
{
  m_inFlight = true;
}

void
WifiMacQueueItem::ResetInFlight (void)
{
  m_inFlight = false;
}

bool
WifiMacQueueItem::IsInFlight (void) const
{
  return m_inFlight;
}

WifiMacQueueItem::DeaggregatedMsdusCI
WifiMacQueueItem::begin (void)
{
  return m_msduList.begin ();
}

WifiMacQueueItem::DeaggregatedMsdusCI
WifiMacQueueItem::end (void)
{
  return m_msduList.end ();
}

void
WifiMacQueueItem::Print (std::ostream& os) const
{
  os << m_header.GetTypeString ()
     << ", payloadSize=" << GetPacketSize ()
     << ", to=" << m_header.GetAddr1 ()
     << ", seqN=" << m_header.GetSequenceNumber ()
     << ", duration/ID=" << m_header.GetDuration ()
     << ", lifetime=" << (Simulator::Now () - m_tstamp).As (Time::US);
  if (m_header.IsQosData ())
    {
      os << ", tid=" << +m_header.GetQosTid ();
      if (m_header.IsQosNoAck ())
        {
          os << ", ack=NoAck";
        }
      else if (m_header.IsQosAck ())
        {
          os << ", ack=NormalAck";
        }
      else if (m_header.IsQosBlockAck ())
        {
          os << ", ack=BlockAck";
        }
    }
  os << ", packet=" << m_packet
     << ", queued=" << IsQueued ()
     << ", inflight=" << IsInFlight ();
}

std::ostream & operator << (std::ostream &os, const WifiMacQueueItem &item)
{
  item.Print (os);
  return os;
}

} //namespace ns3
