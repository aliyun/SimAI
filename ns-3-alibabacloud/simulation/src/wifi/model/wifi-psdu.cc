/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/packet.h"
#include "ns3/log.h"
#include "wifi-psdu.h"
#include "wifi-mac-trailer.h"
#include "mpdu-aggregator.h"
#include "ampdu-subframe-header.h"
#include "wifi-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiPsdu");

WifiPsdu::WifiPsdu (Ptr<const Packet> p, const WifiMacHeader & header)
  : m_isSingle (false)
{
  m_mpduList.push_back (Create<WifiMacQueueItem> (p, header));
  m_size = header.GetSerializedSize () + p->GetSize () + WIFI_MAC_FCS_LENGTH;
}

WifiPsdu::WifiPsdu (Ptr<WifiMacQueueItem> mpdu, bool isSingle)
  : m_isSingle (isSingle)
{
  m_mpduList.push_back (mpdu);
  m_size = mpdu->GetSize ();

  if (isSingle)
    {
      m_size += 4;    // A-MPDU Subframe header size
    }
}

WifiPsdu::WifiPsdu (Ptr<const WifiMacQueueItem> mpdu, bool isSingle)
  : WifiPsdu (Create<WifiMacQueueItem> (*mpdu), isSingle)
{
}

WifiPsdu::WifiPsdu (std::vector<Ptr<WifiMacQueueItem>> mpduList)
  : m_isSingle (mpduList.size () == 1),
    m_mpduList (mpduList)
{
  NS_ABORT_MSG_IF (mpduList.empty (), "Cannot initialize a WifiPsdu with an empty MPDU list");

  m_size = 0;
  for (auto& mpdu : m_mpduList)
    {
      m_size = MpduAggregator::GetSizeIfAggregated (mpdu->GetSize (), m_size);
    }
}

WifiPsdu::~WifiPsdu ()
{
}

bool
WifiPsdu::IsSingle (void) const
{
  return m_isSingle;
}

bool
WifiPsdu::IsAggregate (void) const
{
  return (m_mpduList.size () > 1 || m_isSingle);
}

Ptr<const Packet>
WifiPsdu::GetPacket (void) const
{
  Ptr<Packet> packet = Create<Packet> ();
  if (m_mpduList.size () == 1 && !m_isSingle)
    {
      packet = m_mpduList.at (0)->GetPacket ()->Copy ();
      packet->AddHeader (m_mpduList.at (0)->GetHeader ());
      AddWifiMacTrailer (packet);
    }
  else if (m_isSingle)
    {
      MpduAggregator::Aggregate (m_mpduList.at (0), packet, true);
    }
  else
    {
      for (auto& mpdu : m_mpduList)
        {
          MpduAggregator::Aggregate (mpdu, packet, false);
        }
    }
  return packet;
}

Mac48Address
WifiPsdu::GetAddr1 (void) const
{
  Mac48Address ra = m_mpduList.at (0)->GetHeader ().GetAddr1 ();
  // check that the other MPDUs have the same RA
  for (std::size_t i = 1; i < m_mpduList.size (); i++)
    {
      if (m_mpduList.at (i)->GetHeader ().GetAddr1 () != ra)
        {
          NS_ABORT_MSG ("MPDUs in an A-AMPDU must have the same receiver address");
        }
    }
  return ra;
}

Mac48Address
WifiPsdu::GetAddr2 (void) const
{
  Mac48Address ta = m_mpduList.at (0)->GetHeader ().GetAddr2 ();
  // check that the other MPDUs have the same TA
  for (std::size_t i = 1; i < m_mpduList.size (); i++)
    {
      if (m_mpduList.at (i)->GetHeader ().GetAddr2 () != ta)
        {
          NS_ABORT_MSG ("MPDUs in an A-AMPDU must have the same transmitter address");
        }
    }
  return ta;
}

Time
WifiPsdu::GetDuration (void) const
{
  Time duration = m_mpduList.at (0)->GetHeader ().GetDuration ();
  // check that the other MPDUs have the same Duration/ID
  for (std::size_t i = 1; i < m_mpduList.size (); i++)
    {
      if (m_mpduList.at (i)->GetHeader ().GetDuration () != duration)
        {
          NS_ABORT_MSG ("MPDUs in an A-AMPDU must have the same Duration/ID");
        }
    }
  return duration;
}

void
WifiPsdu::SetDuration (Time duration)
{
  NS_LOG_FUNCTION (this << duration);
  for (auto& mpdu : m_mpduList)
    {
      mpdu->GetHeader ().SetDuration (duration);
    }
}

std::set<uint8_t>
WifiPsdu::GetTids (void) const
{
  std::set<uint8_t> s;
  for (auto& mpdu : m_mpduList)
    {
      if (mpdu->GetHeader ().IsQosData ())
        {
          s.insert (mpdu->GetHeader ().GetQosTid ());
        }
    }
  return s;
}

WifiMacHeader::QosAckPolicy
WifiPsdu::GetAckPolicyForTid (uint8_t tid) const
{
  NS_LOG_FUNCTION (this << +tid);
  WifiMacHeader::QosAckPolicy policy;
  auto it = m_mpduList.begin ();
  bool found = false;

  // find the first QoS Data frame with the given TID
  do
    {
      if ((*it)->GetHeader ().IsQosData () && (*it)->GetHeader ().GetQosTid () == tid)
        {
          policy = (*it)->GetHeader ().GetQosAckPolicy ();
          found = true;
        }
      it++;
    } while (!found && it != m_mpduList.end ());

  NS_ABORT_MSG_IF (!found, "No QoS Data frame in the PSDU");

  // check that the other QoS Data frames with the given TID have the same ack policy
  while (it != m_mpduList.end ())
    {
      if ((*it)->GetHeader ().IsQosData () && (*it)->GetHeader ().GetQosTid () == tid
          && (*it)->GetHeader ().GetQosAckPolicy () != policy)
        {
          NS_ABORT_MSG ("QoS Data frames with the same TID must have the same QoS Ack Policy");
        }
      it++;
    }
  return policy;
}

void
WifiPsdu::SetAckPolicyForTid (uint8_t tid, WifiMacHeader::QosAckPolicy policy)
{
  NS_LOG_FUNCTION (this << +tid << policy);
  for (auto& mpdu : m_mpduList)
    {
      if (mpdu->GetHeader ().IsQosData () && mpdu->GetHeader ().GetQosTid () == tid)
        {
          mpdu->GetHeader ().SetQosAckPolicy (policy);
        }
    }
}

uint16_t
WifiPsdu::GetMaxDistFromStartingSeq (uint16_t startingSeq) const
{
  NS_LOG_FUNCTION (this << startingSeq);

  uint16_t maxDistFromStartingSeq = 0;
  bool foundFirst = false;

  for (auto& mpdu : m_mpduList)
    {
      uint16_t currSeqNum = mpdu->GetHeader ().GetSequenceNumber ();

      if (mpdu->GetHeader ().IsQosData () && !QosUtilsIsOldPacket (startingSeq, currSeqNum))
        {
          uint16_t currDistToStartingSeq = (currSeqNum - startingSeq + SEQNO_SPACE_SIZE) % SEQNO_SPACE_SIZE;

          if (!foundFirst || currDistToStartingSeq > maxDistFromStartingSeq)
            {
              foundFirst = true;
              maxDistFromStartingSeq = currDistToStartingSeq;
            }
        }
    }

  if (!foundFirst)
    {
      NS_LOG_DEBUG ("All QoS Data frames in this PSDU are old frames");
      return SEQNO_SPACE_SIZE;
    }
  NS_LOG_DEBUG ("Returning " << maxDistFromStartingSeq);
  return maxDistFromStartingSeq;
}

uint32_t
WifiPsdu::GetSize (void) const
{
  return m_size;
}

const WifiMacHeader &
WifiPsdu::GetHeader (std::size_t i) const
{
  return m_mpduList.at (i)->GetHeader ();
}

WifiMacHeader &
WifiPsdu::GetHeader (std::size_t i)
{
  return m_mpduList.at (i)->GetHeader ();
}

Ptr<const Packet>
WifiPsdu::GetPayload (std::size_t i) const
{
  return m_mpduList.at (i)->GetPacket ();
}

Time
WifiPsdu::GetTimeStamp (std::size_t i) const
{
  return m_mpduList.at (i)->GetTimeStamp ();
}

Ptr<Packet>
WifiPsdu::GetAmpduSubframe (std::size_t i) const
{
  NS_ASSERT (i < m_mpduList.size ());
  Ptr<Packet> subframe = m_mpduList.at (i)->GetProtocolDataUnit ();
  subframe->AddHeader (MpduAggregator::GetAmpduSubframeHeader (static_cast<uint16_t> (subframe->GetSize ()),
                                                               m_isSingle));
  size_t padding = GetAmpduSubframeSize (i) - subframe->GetSize ();
  if (padding > 0)
    {
      Ptr<Packet> pad = Create<Packet> (padding);
      subframe->AddAtEnd (pad);
    }
  return subframe;
}

std::size_t
WifiPsdu::GetAmpduSubframeSize (std::size_t i) const
{
  NS_ASSERT (i < m_mpduList.size ());
  size_t subframeSize = 4; //A-MPDU Subframe header size
  subframeSize += m_mpduList.at (i)->GetSize ();
  if (i != m_mpduList.size () - 1) //add padding if not last
    {
      subframeSize += MpduAggregator::CalculatePadding (subframeSize);
    }
  return subframeSize;
}

std::size_t
WifiPsdu::GetNMpdus (void) const
{
  return m_mpduList.size ();
}

std::vector<Ptr<WifiMacQueueItem>>::const_iterator
WifiPsdu::begin (void) const
{
  return m_mpduList.begin ();
}

std::vector<Ptr<WifiMacQueueItem>>::iterator
WifiPsdu::begin (void)
{
  return m_mpduList.begin ();
}

std::vector<Ptr<WifiMacQueueItem>>::const_iterator
WifiPsdu::end (void) const
{
  return m_mpduList.end ();
}

std::vector<Ptr<WifiMacQueueItem>>::iterator
WifiPsdu::end (void)
{
  return m_mpduList.end ();
}

void
WifiPsdu::Print (std::ostream& os) const
{
  os << "size=" << m_size;
  if (IsAggregate ())
    {
      os << ", A-MPDU of " << GetNMpdus () << " MPDUs";
      for (const auto& mpdu : m_mpduList)
        {
          os << " (" << *mpdu << ")";
        }
      }
  else
    {
      os << ", " << ((m_isSingle) ? "S-MPDU" : "normal MPDU")
         << " (" << *(m_mpduList.at (0)) << ")";
    }
}

std::ostream & operator << (std::ostream &os, const WifiPsdu &psdu)
{
  psdu.Print (os);
  return os;
}

} //namespace ns3
