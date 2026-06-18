/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ht-operation.h"

namespace ns3 {

HtOperation::HtOperation ()
  : m_primaryChannel (0),
    m_secondaryChannelOffset (0),
    m_staChannelWidth (0),
    m_rifsMode (0),
    m_reservedInformationSubset1 (0),
    m_htProtection (0),
    m_nonGfHtStasPresent (1),
    m_reservedInformationSubset2_1 (0),
    m_obssNonHtStasPresent (0),
    m_reservedInformationSubset2_2 (0),
    m_reservedInformationSubset3_1 (0),
    m_dualBeacon (0),
    m_dualCtsProtection (0),
    m_stbcBeacon (0),
    m_lSigTxopProtectionFullSupport (0),
    m_pcoActive (0),
    m_pcoPhase (0),
    m_reservedInformationSubset3_2 (0),
    m_reservedMcsSet1 (0),
    m_rxHighestSupportedDataRate (0),
    m_reservedMcsSet2 (0),
    m_txMcsSetDefined (0),
    m_txRxMcsSetUnequal (0),
    m_txMaxNSpatialStreams (0),
    m_txUnequalModulation (0),
    m_reservedMcsSet3 (0),
    m_htSupported (0)
{
  for (uint8_t k = 0; k < MAX_SUPPORTED_MCS; k++)
    {
      m_rxMcsBitmask[k] = 0;
    }
}

WifiInformationElementId
HtOperation::ElementId () const
{
  return IE_HT_OPERATION;
}

void
HtOperation::SetHtSupported (uint8_t htSupported)
{
  m_htSupported = htSupported;
}

uint8_t
HtOperation::GetInformationFieldSize () const
{
  //we should not be here if HT is not supported
  NS_ASSERT (m_htSupported > 0);
  return 22;
}

void
HtOperation::SetPrimaryChannel (uint8_t ctrl)
{
  m_primaryChannel = ctrl;
}

void
HtOperation::SetSecondaryChannelOffset (uint8_t secondaryChannelOffset)
{
  m_secondaryChannelOffset = secondaryChannelOffset;
}

void
HtOperation::SetStaChannelWidth (uint8_t staChannelWidth)
{
  m_staChannelWidth = staChannelWidth;
}

void
HtOperation::SetRifsMode (uint8_t rifsMode)
{
  m_rifsMode = rifsMode;
}

void
HtOperation::SetHtProtection (uint8_t htProtection)
{
  m_htProtection = htProtection;
}

void
HtOperation::SetNonGfHtStasPresent (uint8_t nonGfHtStasPresent)
{
  m_nonGfHtStasPresent = nonGfHtStasPresent;
}

void
HtOperation::SetObssNonHtStasPresent (uint8_t obssNonHtStasPresent)
{
  m_obssNonHtStasPresent = obssNonHtStasPresent;
}

void
HtOperation::SetDualBeacon (uint8_t dualBeacon)
{
  m_dualBeacon = dualBeacon;
}

void
HtOperation::SetDualCtsProtection (uint8_t dualCtsProtection)
{
  m_dualCtsProtection = dualCtsProtection;
}

void
HtOperation::SetStbcBeacon (uint8_t stbcBeacon)
{
  m_stbcBeacon = stbcBeacon;
}

void
HtOperation::SetLSigTxopProtectionFullSupport (uint8_t lSigTxopProtectionFullSupport)
{
  m_lSigTxopProtectionFullSupport = lSigTxopProtectionFullSupport;
}

void
HtOperation::SetPcoActive (uint8_t pcoActive)
{
  m_pcoActive = pcoActive;
}

void
HtOperation::SetPhase (uint8_t pcoPhase)
{
  m_pcoPhase = pcoPhase;
}

void
HtOperation::SetRxMcsBitmask (uint8_t index)
{
  m_rxMcsBitmask[index] = 1;
}

void
HtOperation::SetRxHighestSupportedDataRate (uint16_t maxSupportedRate)
{
  m_rxHighestSupportedDataRate = maxSupportedRate;
}

void
HtOperation::SetTxMcsSetDefined (uint8_t txMcsSetDefined)
{
  m_txMcsSetDefined = txMcsSetDefined;
}

void
HtOperation::SetTxRxMcsSetUnequal (uint8_t txRxMcsSetUnequal)
{
  m_txRxMcsSetUnequal = txRxMcsSetUnequal;
}

void
HtOperation::SetTxMaxNSpatialStreams (uint8_t maxTxSpatialStreams)
{
  m_txMaxNSpatialStreams = maxTxSpatialStreams - 1; //0 for 1 SS, 1 for 2 SSs, etc
}

void
HtOperation::SetTxUnequalModulation (uint8_t txUnequalModulation)
{
  m_txUnequalModulation = txUnequalModulation;
}

uint8_t
HtOperation::GetPrimaryChannel (void) const
{
  return m_primaryChannel;
}

uint8_t
HtOperation::GetSecondaryChannelOffset (void) const
{
  return m_secondaryChannelOffset;
}

uint8_t
HtOperation::GetStaChannelWidth (void) const
{
  return m_staChannelWidth;
}

uint8_t
HtOperation::GetRifsMode (void) const
{
  return m_rifsMode;
}

uint8_t
HtOperation::GetHtProtection (void) const
{
  return m_htProtection;
}

uint8_t
HtOperation::GetNonGfHtStasPresent (void) const
{
  return m_nonGfHtStasPresent;
}

uint8_t
HtOperation::GetObssNonHtStasPresent (void) const
{
  return m_obssNonHtStasPresent;
}

uint8_t
HtOperation::GetDualBeacon (void) const
{
  return m_dualBeacon;
}

uint8_t
HtOperation::GetDualCtsProtection (void) const
{
  return m_dualCtsProtection;
}

uint8_t
HtOperation::GetStbcBeacon (void) const
{
  return m_stbcBeacon;
}

uint8_t
HtOperation::GetLSigTxopProtectionFullSupport (void) const
{
  return m_lSigTxopProtectionFullSupport;
}

uint8_t
HtOperation::GetPcoActive (void) const
{
  return m_pcoActive;
}

uint8_t
HtOperation::GetPhase (void) const
{
  return m_pcoPhase;
}

bool
HtOperation::IsSupportedMcs (uint8_t mcs) const
{
  if (m_rxMcsBitmask[mcs] == 1)
    {
      return true;
    }
  return false;
}

uint16_t
HtOperation::GetRxHighestSupportedDataRate (void) const
{
  return m_rxHighestSupportedDataRate;
}

uint8_t
HtOperation::GetTxMcsSetDefined (void) const
{
  return m_txMcsSetDefined;
}

uint8_t
HtOperation::GetTxRxMcsSetUnequal (void) const
{
  return m_txRxMcsSetUnequal;
}

uint8_t
HtOperation::GetTxMaxNSpatialStreams (void) const
{
  return m_txMaxNSpatialStreams;
}

uint8_t
HtOperation::GetTxUnequalModulation (void) const
{
  return m_txUnequalModulation;
}

Buffer::Iterator
HtOperation::Serialize (Buffer::Iterator i) const
{
  if (m_htSupported < 1)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
HtOperation::GetSerializedSize () const
{
  if (m_htSupported < 1)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

uint8_t
HtOperation::GetInformationSubset1 (void) const
{
  uint8_t val = 0;
  val |= m_secondaryChannelOffset & 0x03;
  val |= (m_staChannelWidth & 0x01) << 2;
  val |= (m_rifsMode & 0x01) << 3;
  val |= (m_reservedInformationSubset1 & 0x0f) << 4;
  return val;
}

void
HtOperation::SetInformationSubset1 (uint8_t ctrl)
{
  m_secondaryChannelOffset = ctrl & 0x03;
  m_staChannelWidth = (ctrl >> 2) & 0x01;
  m_rifsMode = (ctrl >> 3) & 0x01;
  m_reservedInformationSubset1 = (ctrl >> 4) & 0x0f;
}

uint16_t
HtOperation::GetInformationSubset2 (void) const
{
  uint16_t val = 0;
  val |= m_htProtection & 0x03;
  val |= (m_nonGfHtStasPresent & 0x01) << 2;
  val |= (m_reservedInformationSubset2_1 & 0x01) << 3;
  val |= (m_obssNonHtStasPresent & 0x01) << 4;
  val |= (m_reservedInformationSubset2_2 & 0x07ff) << 5;
  return val;
}

void
HtOperation::SetInformationSubset2 (uint16_t ctrl)
{
  m_htProtection = ctrl & 0x03;
  m_nonGfHtStasPresent = (ctrl >> 2) & 0x01;
  m_reservedInformationSubset2_1 = (ctrl >> 3) & 0x01;
  m_obssNonHtStasPresent = (ctrl >> 4) & 0x01;
  m_reservedInformationSubset2_2 = static_cast<uint8_t> ((ctrl >> 5) & 0x07ff);
}

uint16_t
HtOperation::GetInformationSubset3 (void) const
{
  uint16_t val = 0;
  val |= m_reservedInformationSubset3_1 & 0x3f;
  val |= (m_dualBeacon & 0x01) << 6;
  val |= (m_dualCtsProtection & 0x01) << 7;
  val |= (m_stbcBeacon & 0x01) << 8;
  val |= (m_lSigTxopProtectionFullSupport & 0x01) << 9;
  val |= (m_pcoActive & 0x01) << 10;
  val |= (m_pcoPhase & 0x01) << 11;
  val |= (m_reservedInformationSubset3_2 & 0x0f) << 12;
  return val;
}

void
HtOperation::SetInformationSubset3 (uint16_t ctrl)
{
  m_reservedInformationSubset3_1 = ctrl & 0x3f;
  m_dualBeacon = (ctrl >> 6) & 0x01;
  m_dualCtsProtection = (ctrl >> 7) & 0x01;
  m_stbcBeacon = (ctrl >> 8) & 0x01;
  m_lSigTxopProtectionFullSupport = (ctrl >> 9) & 0x01;
  m_pcoActive = (ctrl >> 10) & 0x01;
  m_pcoPhase = (ctrl >> 11) & 0x01;
  m_reservedInformationSubset3_2 = (ctrl >> 12) & 0x0f;
}

void
HtOperation::SetBasicMcsSet (uint64_t ctrl1, uint64_t ctrl2)
{
  for (uint64_t i = 0; i < 77; i++)
    {
      if (i < 64)
        {
          m_rxMcsBitmask[i] = (ctrl1 >> i) & 0x01;
        }
      else
        {
          m_rxMcsBitmask[i] = (ctrl2 >> (i - 64)) & 0x01;
        }
    }
  m_reservedMcsSet1 = (ctrl2 >> 13) & 0x07;
  m_rxHighestSupportedDataRate = (ctrl2 >> 16) & 0x03ff;
  m_reservedMcsSet2 = (ctrl2 >> 26) & 0x3f;
  m_txMcsSetDefined = (ctrl2 >> 32) & 0x01;
  m_txRxMcsSetUnequal = (ctrl2 >> 33) & 0x01;
  m_txMaxNSpatialStreams = (ctrl2 >> 34) & 0x03;
  m_txUnequalModulation = (ctrl2 >> 36) & 0x01;
  m_reservedMcsSet3 = (ctrl2 >> 37) & 0x07ffffff;
}

uint64_t
HtOperation::GetBasicMcsSet1 (void) const
{
  uint64_t val = 0;
  for (uint64_t i = 63; i > 0; i--)
    {
      val = (val << 1) | (m_rxMcsBitmask[i] & 0x01);
    }
  val = (val << 1) | (m_rxMcsBitmask[0] & 0x01);
  return val;
}

uint64_t
HtOperation::GetBasicMcsSet2 (void) const
{
  uint64_t val = 0;
  val = val | (m_reservedMcsSet3 & 0x07ffffff);
  val = (val << 1) | (m_txUnequalModulation & 0x01);
  val = (val << 2) | (m_txMaxNSpatialStreams & 0x03);
  val = (val << 1) | (m_txRxMcsSetUnequal & 0x01);
  val = (val << 1) | (m_txMcsSetDefined & 0x01);
  val = (val << 6) | (m_reservedMcsSet2 & 0x3f);
  val = (val << 10) | (m_rxHighestSupportedDataRate & 0x3ff);
  val = (val << 3) | (m_reservedMcsSet1 & 0x07);

  for (uint64_t i = 13; i > 0; i--)
    {
      val = (val << 1) | ( m_rxMcsBitmask[i + 63] & 0x01);
    }
  return val;
}

void
HtOperation::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_htSupported == 1)
    {
      //write the corresponding value for each bit
      start.WriteU8 (GetPrimaryChannel ());
      start.WriteU8 (GetInformationSubset1 ());
      start.WriteU16 (GetInformationSubset2 ());
      start.WriteU16 (GetInformationSubset3 ());
      start.WriteHtolsbU64 (GetBasicMcsSet1 ());
      start.WriteHtolsbU64 (GetBasicMcsSet2 ());
    }
}

uint8_t
HtOperation::DeserializeInformationField (Buffer::Iterator start,
                                          uint8_t length)
{
  Buffer::Iterator i = start;
  uint8_t primarychannel = i.ReadU8 ();
  uint8_t informationsubset1 = i.ReadU8 ();
  uint16_t informationsubset2 = i.ReadU16 ();
  uint16_t informationsubset3 = i.ReadU16 ();
  uint64_t mcsset1 = i.ReadLsbtohU64 ();
  uint64_t mcsset2 = i.ReadLsbtohU64 ();
  SetPrimaryChannel (primarychannel);
  SetInformationSubset1 (informationsubset1);
  SetInformationSubset2 (informationsubset2);
  SetInformationSubset3 (informationsubset3);
  SetBasicMcsSet (mcsset1, mcsset2);
  return length;
}

std::ostream &
operator << (std::ostream &os, const HtOperation &htOperation)
{
  os << bool (htOperation.GetPrimaryChannel ())
     << "|" << +htOperation.GetSecondaryChannelOffset ()
     << "|" << bool (htOperation.GetStaChannelWidth ())
     << "|" << bool (htOperation.GetRifsMode ())
     << "|" << +htOperation.GetHtProtection ()
     << "|" << bool (htOperation.GetNonGfHtStasPresent ())
     << "|" << bool (htOperation.GetObssNonHtStasPresent ())
     << "|" << bool (htOperation.GetDualBeacon ())
     << "|" << bool (htOperation.GetDualCtsProtection ())
     << "|" << bool (htOperation.GetStbcBeacon ())
     << "|" << bool (htOperation.GetLSigTxopProtectionFullSupport ())
     << "|" << bool (htOperation.GetPcoActive ())
     << "|" << bool (htOperation.GetPhase ())
     << "|" << htOperation.GetRxHighestSupportedDataRate ()
     << "|" << bool (htOperation.GetTxMcsSetDefined ())
     << "|" << bool (htOperation.GetTxRxMcsSetUnequal ())
     << "|" << +htOperation.GetTxMaxNSpatialStreams ()
     << "|" << bool (htOperation.GetTxUnequalModulation ()) << "|";
  for (uint8_t i = 0; i < MAX_SUPPORTED_MCS; i++)
    {
      os << htOperation.IsSupportedMcs (i) << " ";
    }
  return os;
}

} //namespace ns3
