/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013
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
 * Authors: Ghada Badawy <gbadawy@rim.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ht-capabilities.h"

namespace ns3 {

HtCapabilities::HtCapabilities ()
  : m_ldpc (0),
    m_supportedChannelWidth (0),
    m_smPowerSave (0),
    m_greenField (0),
    m_shortGuardInterval20 (0),
    m_shortGuardInterval40 (0),
    m_txStbc (0),
    m_rxStbc (0),
    m_htDelayedBlockAck (0),
    m_maxAmsduLength (0),
    m_dssMode40 (0),
    m_psmpSupport (0),
    m_fortyMhzIntolerant (0),
    m_lsigProtectionSupport (0),
    m_maxAmpduLengthExponent (0),
    m_minMpduStartSpace (0),
    m_ampduReserved (0),
    m_reservedMcsSet1 (0),
    m_rxHighestSupportedDataRate (0),
    m_reservedMcsSet2 (0),
    m_txMcsSetDefined (0),
    m_txRxMcsSetUnequal (0),
    m_txMaxNSpatialStreams (0),
    m_txUnequalModulation (0),
    m_reservedMcsSet3 (0),
    m_pco (0),
    m_pcoTransitionTime (0),
    m_reservedExtendedCapabilities (0),
    m_mcsFeedback (0),
    m_htcSupport (0),
    m_reverseDirectionResponder (0),
    m_reservedExtendedCapabilities2 (0),
    m_implicitRxBfCapable (0),
    m_rxStaggeredSoundingCapable (0),
    m_txStaggeredSoundingCapable (0),
    m_rxNdpCapable (0),
    m_txNdpCapable (0),
    m_implicitTxBfCapable (0),
    m_calibration (0),
    m_explicitCsiTxBfCapable (0),
    m_explicitNoncompressedSteeringCapable (0),
    m_explicitCompressedSteeringCapable (0),
    m_explicitTxBfCsiFeedback (0),
    m_explicitNoncompressedBfFeedbackCapable (0),
    m_explicitCompressedBfFeedbackCapable (0),
    m_minimalGrouping (0),
    m_csiNBfAntennasSupported (0),
    m_noncompressedSteeringNBfAntennasSupported (0),
    m_compressedSteeringNBfAntennasSupported (0),
    m_csiMaxNRowsBfSupported (0),
    m_channelEstimationCapability (0),
    m_reservedTxBf (0),
    m_antennaSelectionCapability (0),
    m_explicitCsiFeedbackBasedTxASelCapable (0),
    m_antennaIndicesFeedbackBasedTxASelCapable (0),
    m_explicitCsiFeedbackCapable (0),
    m_antennaIndicesFeedbackCapable (0),
    m_rxASelCapable (0),
    m_txSoundingPpdusCapable (0),
    m_reservedASel (0),
    m_htSupported (0)
{
  for (uint8_t i = 0; i < MAX_SUPPORTED_MCS; i++)
    {
      m_rxMcsBitmask[i] = 0;
    }
}

WifiInformationElementId
HtCapabilities::ElementId () const
{
  return IE_HT_CAPABILITIES;
}

void
HtCapabilities::SetHtSupported (uint8_t htSupported)
{
  m_htSupported = htSupported;
}

void
HtCapabilities::SetLdpc (uint8_t ldpc)
{
  m_ldpc = ldpc;
}

void
HtCapabilities::SetSupportedChannelWidth (uint8_t supportedChannelWidth)
{
  m_supportedChannelWidth = supportedChannelWidth;
}

void
HtCapabilities::SetShortGuardInterval20 (uint8_t shortGuardInterval)
{
  m_shortGuardInterval20 = shortGuardInterval;
}

void
HtCapabilities::SetShortGuardInterval40 (uint8_t shortGuardInterval)
{
  m_shortGuardInterval40 = shortGuardInterval;
}

void
HtCapabilities::SetMaxAmsduLength (uint16_t maxAmsduLength)
{
  NS_ABORT_MSG_IF (maxAmsduLength != 3839 && maxAmsduLength != 7935,
                   "Invalid A-MSDU Max Length value");
  m_maxAmsduLength = (maxAmsduLength == 3839 ? 0 : 1);
}

void
HtCapabilities::SetLSigProtectionSupport (uint8_t lSigProtection)
{
  m_lsigProtectionSupport = lSigProtection;
}

void
HtCapabilities::SetMaxAmpduLength (uint32_t maxAmpduLength)
{
  for (uint8_t i = 0; i <= 3; i++)
    {
      if ((1ul << (13 + i)) - 1 == maxAmpduLength)
        {
          m_maxAmpduLengthExponent = i;
          return;
        }
    }
  NS_ABORT_MSG ("Invalid A-MPDU Max Length value");
}

void
HtCapabilities::SetRxMcsBitmask (uint8_t index)
{
  m_rxMcsBitmask[index] = 1;
}

void
HtCapabilities::SetRxHighestSupportedDataRate (uint16_t maxSupportedRate)
{
  m_rxHighestSupportedDataRate = maxSupportedRate;
}

void
HtCapabilities::SetTxMcsSetDefined (uint8_t txMcsSetDefined)
{
  m_txMcsSetDefined = txMcsSetDefined;
}

void
HtCapabilities::SetTxRxMcsSetUnequal (uint8_t txRxMcsSetUnequal)
{
  m_txRxMcsSetUnequal = txRxMcsSetUnequal;
}

void
HtCapabilities::SetTxMaxNSpatialStreams (uint8_t maxTxSpatialStreams)
{
  m_txMaxNSpatialStreams = maxTxSpatialStreams - 1; //0 for 1 SS, 1 for 2 SSs, etc
}

void
HtCapabilities::SetTxUnequalModulation (uint8_t txUnequalModulation)
{
  m_txUnequalModulation = txUnequalModulation;
}

uint8_t
HtCapabilities::GetLdpc (void) const
{
  return m_ldpc;
}

uint8_t
HtCapabilities::GetSupportedChannelWidth (void) const
{
  return m_supportedChannelWidth;
}

uint8_t
HtCapabilities::GetShortGuardInterval20 (void) const
{
  return m_shortGuardInterval20;
}

uint16_t
HtCapabilities::GetMaxAmsduLength (void) const
{
  if (m_maxAmsduLength == 0)
    {
      return 3839;
    }
  return 7935;
}

uint32_t
HtCapabilities::GetMaxAmpduLength (void) const
{
  return (1ul << (13 + m_maxAmpduLengthExponent)) - 1;
}

bool
HtCapabilities::IsSupportedMcs (uint8_t mcs) const
{
  if (m_rxMcsBitmask[mcs] == 1)
    {
      return true;
    }
  return false;
}

uint8_t
HtCapabilities::GetRxHighestSupportedAntennas (void) const
{
  for (uint8_t nRx = 2; nRx <= 4; nRx++)
    {
      for (uint8_t mcs = (nRx - 1) * 8; mcs <= ((7 * nRx) + (nRx - 1)); mcs++)
        {
          if (IsSupportedMcs (mcs) == false)
            {
              return (nRx - 1);
            }
        }
    }
  return 4;
}

uint8_t
HtCapabilities::GetInformationFieldSize () const
{
  //we should not be here if HT is not supported
  NS_ASSERT (m_htSupported > 0);
  return 26;
}

Buffer::Iterator
HtCapabilities::Serialize (Buffer::Iterator i) const
{
  if (m_htSupported < 1)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
HtCapabilities::GetSerializedSize () const
{
  if (m_htSupported < 1)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

uint16_t
HtCapabilities::GetHtCapabilitiesInfo (void) const
{
  uint16_t val = 0;
  val |= m_ldpc & 0x01;
  val |= (m_supportedChannelWidth & 0x01) << 1;
  val |= (m_smPowerSave & 0x03) << 2;
  val |= (m_greenField & 0x01) << 4;
  val |= (m_shortGuardInterval20 & 0x01) << 5;
  val |= (m_shortGuardInterval40 & 0x01) << 6;
  val |= (m_txStbc & 0x01) << 7;
  val |= (m_rxStbc & 0x03) << 8;
  val |= (m_htDelayedBlockAck & 0x01) << 10;
  val |= (m_maxAmsduLength & 0x01) << 11;
  val |= (m_dssMode40 & 0x01) << 12;
  val |= (m_psmpSupport & 0x01) << 13;
  val |= (m_fortyMhzIntolerant & 0x01) << 14;
  val |= (m_lsigProtectionSupport & 0x01) << 15;
  return val;
}

void
HtCapabilities::SetHtCapabilitiesInfo (uint16_t ctrl)
{
  m_ldpc = ctrl & 0x01;
  m_supportedChannelWidth = (ctrl >> 1) & 0x01;
  m_smPowerSave = (ctrl >> 2) & 0x03;
  m_greenField = (ctrl >> 4) & 0x01;
  m_shortGuardInterval20 = (ctrl >> 5) & 0x01;
  m_shortGuardInterval40 = (ctrl >> 6) & 0x01;
  m_txStbc = (ctrl >> 7) & 0x01;
  m_rxStbc = (ctrl >> 8) & 0x03;
  m_htDelayedBlockAck = (ctrl >> 10) & 0x01;
  m_maxAmsduLength = (ctrl >> 11) & 0x01;
  m_dssMode40 = (ctrl >> 12) & 0x01;
  m_psmpSupport = (ctrl >> 13) & 0x01;
  m_fortyMhzIntolerant = (ctrl >> 14) & 0x01;
  m_lsigProtectionSupport = (ctrl >> 15) & 0x01;
}

void
HtCapabilities::SetAmpduParameters (uint8_t ctrl)
{
  m_maxAmpduLengthExponent = ctrl & 0x03;
  m_minMpduStartSpace = (ctrl >> 2) & 0x1b;
  m_ampduReserved = (ctrl >> 5) & 0xe0;
}

uint8_t
HtCapabilities::GetAmpduParameters (void) const
{
  uint8_t val = 0;
  val |=  m_maxAmpduLengthExponent & 0x03;
  val |= (m_minMpduStartSpace & 0x1b) << 2;
  val |= (m_ampduReserved & 0xe0) << 5;
  return val;
}

void
HtCapabilities::SetSupportedMcsSet (uint64_t ctrl1, uint64_t ctrl2)
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
HtCapabilities::GetSupportedMcsSet1 (void) const
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
HtCapabilities::GetSupportedMcsSet2 (void) const
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

uint16_t
HtCapabilities::GetExtendedHtCapabilities (void) const
{
  uint16_t val = 0;
  val |= m_pco & 0x01;
  val |= (m_pcoTransitionTime & 0x03) << 1;
  val |= (m_reservedExtendedCapabilities & 0x1f) << 3;
  val |= (m_mcsFeedback & 0x03) << 8;
  val |= (m_htcSupport & 0x01) << 10;
  val |= (m_reverseDirectionResponder & 0x01) << 11;
  val |= (m_reservedExtendedCapabilities2 & 0x0f) << 12;
  return val;
}

void
HtCapabilities::SetExtendedHtCapabilities (uint16_t ctrl)
{
  m_pco = ctrl & 0x01;
  m_pcoTransitionTime = (ctrl >> 1) & 0x03;
  m_reservedExtendedCapabilities = (ctrl >> 3) & 0x1f;
  m_mcsFeedback = (ctrl >> 8) & 0x03;
  m_htcSupport = (ctrl >> 10) & 0x01;
  m_reverseDirectionResponder = (ctrl >> 11) & 0x01;
  m_reservedExtendedCapabilities2 = (ctrl >> 12) & 0x0f;
}

uint32_t
HtCapabilities::GetTxBfCapabilities (void) const
{
  uint32_t val = 0;
  val |= m_implicitRxBfCapable & 0x01;
  val |= (m_rxStaggeredSoundingCapable & 0x01) << 1;
  val |= (m_txStaggeredSoundingCapable & 0x01) << 2;
  val |= (m_rxNdpCapable & 0x01) << 3;
  val |= (m_txNdpCapable & 0x01) << 4;
  val |= (m_implicitTxBfCapable & 0x01) << 5;
  val |= (m_calibration & 0x03) << 6;
  val |= (m_explicitCsiTxBfCapable & 0x01) << 8;
  val |= (m_explicitNoncompressedSteeringCapable & 0x01) << 9;
  val |= (m_explicitCompressedSteeringCapable & 0x01) << 10;
  val |= (m_explicitTxBfCsiFeedback & 0x03) << 11;
  val |= (m_explicitNoncompressedBfFeedbackCapable & 0x03) << 13;
  val |= (m_explicitCompressedBfFeedbackCapable & 0x03) << 15;
  val |= (m_minimalGrouping & 0x03) << 17;
  val |= (m_csiNBfAntennasSupported & 0x03) << 19;
  val |= (m_noncompressedSteeringNBfAntennasSupported & 0x03) << 21;
  val |= (m_compressedSteeringNBfAntennasSupported & 0x03) << 23;
  val |= (m_csiMaxNRowsBfSupported & 0x03) << 25;
  val |= (m_channelEstimationCapability & 0x03) << 27;
  val |= (m_reservedTxBf & 0x07) << 29;
  return val;
}

void
HtCapabilities::SetTxBfCapabilities (uint32_t ctrl)
{
  m_implicitRxBfCapable = ctrl & 0x01;
  m_rxStaggeredSoundingCapable = (ctrl >> 1) & 0x01;
  m_txStaggeredSoundingCapable = (ctrl >> 2) & 0x01;
  m_rxNdpCapable = (ctrl >> 3) & 0x01;
  m_txNdpCapable = (ctrl >> 4) & 0x01;
  m_implicitTxBfCapable = (ctrl >> 5) & 0x01;
  m_calibration = (ctrl >> 6) & 0x03;
  m_explicitCsiTxBfCapable = (ctrl >> 8) & 0x01;
  m_explicitNoncompressedSteeringCapable = (ctrl >> 9) & 0x01;
  m_explicitCompressedSteeringCapable = (ctrl >> 10) & 0x01;
  m_explicitTxBfCsiFeedback = (ctrl >> 11) & 0x03;
  m_explicitNoncompressedBfFeedbackCapable = (ctrl >> 13) & 0x03;
  m_explicitCompressedBfFeedbackCapable = (ctrl >> 15) & 0x03;
  m_minimalGrouping = (ctrl >> 17) & 0x03;
  m_csiNBfAntennasSupported = (ctrl >> 19) & 0x03;
  m_noncompressedSteeringNBfAntennasSupported = (ctrl >> 21) & 0x03;
  m_compressedSteeringNBfAntennasSupported = (ctrl >> 23) & 0x03;
  m_csiMaxNRowsBfSupported = (ctrl >> 25) & 0x03;
  m_channelEstimationCapability = (ctrl >> 27) & 0x03;
  m_reservedTxBf = (ctrl >> 29) & 0x07;
}

uint8_t
HtCapabilities::GetAntennaSelectionCapabilities (void) const
{
  uint8_t val = 0;
  val |= m_antennaSelectionCapability & 0x01;
  val |= (m_explicitCsiFeedbackBasedTxASelCapable & 0x01) << 1;
  val |= (m_antennaIndicesFeedbackBasedTxASelCapable & 0x01) << 2;
  val |= (m_explicitCsiFeedbackCapable & 0x01) << 3;
  val |= (m_antennaIndicesFeedbackCapable & 0x01) << 4;
  val |= (m_rxASelCapable & 0x01) << 5;
  val |= (m_txSoundingPpdusCapable & 0x01) << 6;
  val |= (m_reservedASel & 0x01) << 7;
  return val;
}

void
HtCapabilities::SetAntennaSelectionCapabilities (uint8_t ctrl)
{
  m_antennaSelectionCapability = ctrl & 0x01;
  m_explicitCsiFeedbackBasedTxASelCapable = (ctrl >> 1) & 0x01;
  m_antennaIndicesFeedbackBasedTxASelCapable = (ctrl >> 2) & 0x01;
  m_explicitCsiFeedbackCapable = (ctrl >> 3) & 0x01;
  m_antennaIndicesFeedbackCapable = (ctrl >> 4) & 0x01;
  m_rxASelCapable = (ctrl >> 5) & 0x01;
  m_txSoundingPpdusCapable = (ctrl >> 6) & 0x01;
  m_reservedASel = (ctrl >> 7) & 0x01;
}

void
HtCapabilities::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_htSupported == 1)
    {
      //write the corresponding value for each bit
      start.WriteHtolsbU16 (GetHtCapabilitiesInfo ());
      start.WriteU8 (GetAmpduParameters ());
      start.WriteHtolsbU64 (GetSupportedMcsSet1 ());
      start.WriteHtolsbU64 (GetSupportedMcsSet2 ());
      start.WriteU16 (GetExtendedHtCapabilities ());
      start.WriteU32 (GetTxBfCapabilities ());
      start.WriteU8 (GetAntennaSelectionCapabilities ());
    }
}

uint8_t
HtCapabilities::DeserializeInformationField (Buffer::Iterator start,
                                             uint8_t length)
{
  Buffer::Iterator i = start;
  uint16_t htinfo = i.ReadLsbtohU16 ();
  uint8_t ampduparam = i.ReadU8 ();
  uint64_t mcsset1 = i.ReadLsbtohU64 ();
  uint64_t mcsset2 = i.ReadLsbtohU64 ();
  uint16_t extendedcapabilities = i.ReadU16 ();
  uint32_t txbfcapabilities = i.ReadU32 ();
  uint8_t aselcapabilities = i.ReadU8 ();
  SetHtCapabilitiesInfo (htinfo);
  SetAmpduParameters (ampduparam);
  SetSupportedMcsSet (mcsset1, mcsset2);
  SetExtendedHtCapabilities (extendedcapabilities);
  SetTxBfCapabilities (txbfcapabilities);
  SetAntennaSelectionCapabilities (aselcapabilities);
  return length;
}

std::ostream &
operator << (std::ostream &os, const HtCapabilities &htcapabilities)
{
  os << bool (htcapabilities.GetLdpc ())
     << "|" << bool (htcapabilities.GetSupportedChannelWidth ())
     << "|" << bool (htcapabilities.GetShortGuardInterval20 ()) << "|";
  for (uint8_t i = 0; i < MAX_SUPPORTED_MCS; i++)
    {
      os << htcapabilities.IsSupportedMcs (i) << " ";
    }
  return os;
}

} //namespace ns3
