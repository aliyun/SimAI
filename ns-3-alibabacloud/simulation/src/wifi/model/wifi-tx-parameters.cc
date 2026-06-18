/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Universita' degli Studi di Napoli Federico II
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

#include "wifi-tx-parameters.h"
#include "wifi-mac-queue-item.h"
#include "wifi-mac-trailer.h"
#include "msdu-aggregator.h"
#include "mpdu-aggregator.h"
#include "wifi-protection.h"
#include "wifi-acknowledgment.h"
#include "ns3/packet.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiTxParameters");

WifiTxParameters::WifiTxParameters ()
{
}

WifiTxParameters::WifiTxParameters (const WifiTxParameters& txParams)
{
  m_txVector = txParams.m_txVector;
  m_protection = (txParams.m_protection ? txParams.m_protection->Copy () : nullptr);
  m_acknowledgment = (txParams.m_acknowledgment ? txParams.m_acknowledgment->Copy () : nullptr);
  m_txDuration = txParams.m_txDuration;
  m_info = txParams.m_info;
}

WifiTxParameters&
WifiTxParameters::operator= (const WifiTxParameters& txParams)
{
  // check for self-assignment
  if (&txParams == this)
    {
      return *this;
    }

  m_txVector = txParams.m_txVector;
  m_protection = (txParams.m_protection ? txParams.m_protection->Copy () : nullptr);
  m_acknowledgment = (txParams.m_acknowledgment ? txParams.m_acknowledgment->Copy () : nullptr);
  m_txDuration = txParams.m_txDuration;
  m_info = txParams.m_info;

  return *this;
}

void
WifiTxParameters::Clear (void)
{
  NS_LOG_FUNCTION (this);

  // Reset the current info
  m_info.clear ();
  m_txVector = WifiTxVector ();
  m_protection.reset (nullptr);
  m_acknowledgment.reset (nullptr);
  m_txDuration = Time::Min ();
}

const WifiTxParameters::PsduInfo*
WifiTxParameters::GetPsduInfo (Mac48Address receiver) const
{
  auto infoIt = m_info.find (receiver);

  if (infoIt == m_info.end ())
    {
      return nullptr;
    }
  return &infoIt->second;
}

const WifiTxParameters::PsduInfoMap&
WifiTxParameters::GetPsduInfoMap (void) const
{
  return m_info;
}

void
WifiTxParameters::AddMpdu (Ptr<const WifiMacQueueItem> mpdu)
{
  NS_LOG_FUNCTION (this << *mpdu);

  const WifiMacHeader& hdr = mpdu->GetHeader ();

  auto infoIt = m_info.find (hdr.GetAddr1 ());

  if (infoIt == m_info.end ())
    {
      // this is an MPDU starting a new PSDU
      std::map<uint8_t, std::set<uint16_t>> seqNumbers;
      if (hdr.IsQosData ())
        {
          seqNumbers[hdr.GetQosTid ()] = {hdr.GetSequenceNumber ()};
        }

      // Insert the info about the given frame
      m_info.emplace (hdr.GetAddr1 (),
                      PsduInfo {hdr, mpdu->GetPacketSize (), 0, seqNumbers});
      return;
    }

  // a PSDU for the receiver of the given MPDU is already being built
  NS_ASSERT_MSG ((hdr.IsQosData () && !hdr.HasData ()) || infoIt->second.amsduSize > 0,
                 "An MPDU can only be aggregated to an existing (A-)MPDU");

  // The (A-)MSDU being built is included in an A-MPDU subframe
  infoIt->second.ampduSize = MpduAggregator::GetSizeIfAggregated (infoIt->second.header.GetSize ()
                                                                  + infoIt->second.amsduSize
                                                                  + WIFI_MAC_FCS_LENGTH,
                                                                  infoIt->second.ampduSize);
  infoIt->second.header = hdr;
  infoIt->second.amsduSize = mpdu->GetPacketSize ();

  if (hdr.IsQosData ())
    {
      auto ret = infoIt->second.seqNumbers.emplace (hdr.GetQosTid (),
                                                    std::set<uint16_t> {hdr.GetSequenceNumber ()});
      
      if (!ret.second)
        {
          // insertion did not happen because an entry with the same TID already exists
          ret.first->second.insert (hdr.GetSequenceNumber ());
        }
    }
}

uint32_t
WifiTxParameters::GetSizeIfAddMpdu (Ptr<const WifiMacQueueItem> mpdu) const
{
  NS_LOG_FUNCTION (this << *mpdu);

  auto infoIt = m_info.find (mpdu->GetHeader ().GetAddr1 ());

  if (infoIt == m_info.end ())
    {
      // this is an MPDU starting a new PSDU
      if (m_txVector.GetModulationClass () >= WIFI_MOD_CLASS_VHT)
        {
          // All MPDUs are sent with the A-MPDU structure
          return MpduAggregator::GetSizeIfAggregated (mpdu->GetSize (), 0);
        }
      return mpdu->GetSize ();
    }

  // aggregate the (A-)MSDU being built to the existing A-MPDU (if any)
  uint32_t ampduSize = MpduAggregator::GetSizeIfAggregated (infoIt->second.header.GetSize ()
                                                            + infoIt->second.amsduSize
                                                            + WIFI_MAC_FCS_LENGTH,
                                                            infoIt->second.ampduSize);
  // aggregate the new MPDU to the A-MPDU
  return MpduAggregator::GetSizeIfAggregated (mpdu->GetSize (), ampduSize);
}

void
WifiTxParameters::AggregateMsdu (Ptr<const WifiMacQueueItem> msdu)
{
  NS_LOG_FUNCTION (this << *msdu);

  auto infoIt = m_info.find (msdu->GetHeader ().GetAddr1 ());
  NS_ASSERT_MSG (infoIt != m_info.end (),
                 "There must be already an MPDU addressed to the same receiver");

  infoIt->second.amsduSize = GetSizeIfAggregateMsdu (msdu).first;
  infoIt->second.header.SetQosAmsdu ();
}

std::pair<uint32_t, uint32_t>
WifiTxParameters::GetSizeIfAggregateMsdu (Ptr<const WifiMacQueueItem> msdu) const
{
  NS_LOG_FUNCTION (this << *msdu);

  NS_ASSERT_MSG (msdu->GetHeader ().IsQosData (), "Can only aggregate a QoS data frame to an A-MSDU");

  auto infoIt = m_info.find (msdu->GetHeader ().GetAddr1 ());
  NS_ASSERT_MSG (infoIt != m_info.end (),
                 "There must be already an MPDU addressed to the same receiver");

  NS_ASSERT_MSG (infoIt->second.amsduSize > 0,
                 "The amsduSize should be set to the size of the previous MSDU(s)");
  NS_ASSERT_MSG (infoIt->second.header.IsQosData (),
                 "The MPDU being built for this receiver must be a QoS data frame");
  NS_ASSERT_MSG (infoIt->second.header.GetQosTid () == msdu->GetHeader ().GetQosTid (),
                 "The MPDU being built must belong to the same TID as the MSDU to aggregate");
  NS_ASSERT_MSG (infoIt->second.seqNumbers.find (msdu->GetHeader ().GetQosTid ()) != infoIt->second.seqNumbers.end (),
                 "At least one MPDU with the same TID must have been added previously");

  // all checks passed
  uint32_t currAmsduSize = infoIt->second.amsduSize;

  if (!infoIt->second.header.IsQosAmsdu ())
    {
      // consider the A-MSDU subframe for the first MSDU
      currAmsduSize = MsduAggregator::GetSizeIfAggregated (currAmsduSize, 0);
    }

  uint32_t newAmsduSize = MsduAggregator::GetSizeIfAggregated (msdu->GetPacket ()->GetSize (), currAmsduSize);
  uint32_t newMpduSize = infoIt->second.header.GetSize () + newAmsduSize + WIFI_MAC_FCS_LENGTH;

  if (infoIt->second.ampduSize > 0 || m_txVector.GetModulationClass () >= WIFI_MOD_CLASS_VHT)
    {
      return {newAmsduSize, MpduAggregator::GetSizeIfAggregated (newMpduSize, infoIt->second.ampduSize)};
    }
  
  return {newAmsduSize, newMpduSize};
}

uint32_t
WifiTxParameters::GetSize (Mac48Address receiver) const
{
  NS_LOG_FUNCTION (this << receiver);

  auto infoIt = m_info.find (receiver);

  if (infoIt == m_info.end ())
    {
      return 0;
    }

  uint32_t newMpduSize = infoIt->second.header.GetSize () + infoIt->second.amsduSize + WIFI_MAC_FCS_LENGTH;

  if (infoIt->second.ampduSize > 0 || m_txVector.GetModulationClass () >= WIFI_MOD_CLASS_VHT)
    {
      return MpduAggregator::GetSizeIfAggregated (newMpduSize, infoIt->second.ampduSize);
    }
  
  return newMpduSize;  
}

void
WifiTxParameters::Print (std::ostream& os) const
{
  os << "TXVECTOR=" << m_txVector;
  if (m_protection)
    {
      os << ", Protection=" << m_protection.get ();
    }
  if (m_acknowledgment)
    {
      os << ", Acknowledgment=" << m_acknowledgment.get ();
    }
  os << ", PSDUs:";
  for (const auto& info : m_info)
    {
      os << " [To=" << info.second.header.GetAddr1 () << ", A-MSDU size="
         << info.second.amsduSize << ", A-MPDU size=" << info.second.ampduSize << "]";
    }
}

std::ostream & operator << (std::ostream &os, const WifiTxParameters* txParams)
{
  txParams->Print (os);
  return os;
}

} //namespace ns3
