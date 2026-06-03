/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016
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

#include "he-capabilities.h"
#include <algorithm>

namespace ns3 {

HeCapabilities::HeCapabilities ()
  : m_plusHtcHeSupport (0),
    m_twtRequesterSupport (0),
    m_twtResponderSupport (0),
    m_fragmentationSupport (0),
    m_maximumNumberOfFragmentedMsdus (0),
    m_minimumFragmentSize (0),
    m_triggerFrameMacPaddingDuration (0),
    m_multiTidAggregationRxSupport (0),
    m_heLinkAdaptation (0),
    m_allAckSupport (0),
    m_trsSupport (0),
    m_bsrSupport (0),
    m_broadcastTwtSupport (0),
    m_32bitBaBitmapSupport (0),
    m_muCascadeSupport (0),
    m_ackEnabledAggregationSupport (0),
    m_omControlSupport (0),
    m_ofdmaRaSupport (0),
    m_maxAmpduLengthExponent (0),
    m_amsduFragmentationSupport (0),
    m_flexibleTwtScheduleSupport (0),
    m_rxControlFrameToMultiBss (0),
    m_bsrpBqrpAmpduAggregation (0),
    m_qtpSupport (0),
    m_bqrSupport (0),
    m_psrResponder (0),
    m_ndpFeedbackReportSupport (0),
    m_opsSupport (0),
    m_amsduNotUnderBaInAmpduSupport (0),
    m_multiTidAggregationTxSupport (0),
    m_heSubchannelSelectiveTxSupport (0),
    m_ul2x996ToneRuSupport (0),
    m_omControlUlMuDataDisableRxSupport (0),
    m_heDynamicSmPowerSave (0),
    m_puncturedSoundingSupport (0),
    m_heVhtTriggerFrameRxSupport (0),
    m_channelWidthSet (0),
    m_puncturedPreambleRx (0),
    m_deviceClass (0),
    m_ldpcCodingInPayload (0),
    m_heSuPpdu1xHeLtf800nsGi (0),
    m_midambleRxMaxNsts (0),
    m_ndp4xHeLtfAnd32msGi (0),
    m_stbcTxLeq80MHz (0),
    m_stbcRxLeq80MHz (0),
    m_dopplerTx (0),
    m_dopplerRx (0),
    m_fullBwUlMuMimo (0),
    m_partialBwUlMuMimo (0),
    m_dcmMaxConstellationTx (0),
    m_dcmMaxNssTx (0),
    m_dcmMaxConstellationRx (0),
    m_dcmMaxNssRx (0),
    m_rxPartialBwSuInHeMu (0),
    m_suBeamformer (0),
    m_suBeamformee (0),
    m_muBeamformer (0),
    m_beamformeeStsForSmallerOrEqualThan80Mhz (0),
    m_beamformeeStsForLargerThan80Mhz (0),
    m_numberOfSoundingDimensionsForSmallerOrEqualThan80Mhz (0),
    m_numberOfSoundingDimensionsForLargerThan80Mhz (0),
    m_ngEqual16ForSuFeedbackSupport (0),
    m_ngEqual16ForMuFeedbackSupport (0),
    m_codebookSize42SuFeedback (0),
    m_codebookSize75MuFeedback (0),
    m_triggeredSuBfFeedback (0),
    m_triggeredMuBfFeedback (0),
    m_triggeredCqiFeedback (0),
    m_erPartialBandwidth (0),
    m_dlMuMimoOnPartialBandwidth (0),
    m_ppeThresholdPresent (0),
    m_psrBasedSrSupport (0),
    m_powerBoostFactorAlphaSupport (0),
    m_hePpdu4xHeLtf800nsGi (0),
    m_maxNc (0),
    m_stbcTxGt80MHz (0),
    m_stbcRxGt80MHz (0),
    m_heErSuPpdu4xHeLtf08sGi (0),
    m_hePpdu20MHzIn40MHz24GHz (0),
    m_hePpdu20MHzIn160MHz (0),
    m_hePpdu80MHzIn160MHz (0),
    m_heErSuPpdu1xHeLtf08Gi (0),
    m_midamble2xAnd1xHeLtf (0),
    m_dcmMaxRu (0),
    m_longerThan16HeSigbOfdm (0),
    m_nonTriggeredCqiFeedback (0),
    m_tx1024QamLt242Ru (0),
    m_rx1024QamLt242Ru (0),
    m_rxFullBwSuInHeMuCompressedSigB (0),
    m_rxFullBwSuInHeMuNonCompressedSigB (0),
    m_nominalPacketPadding (0),
    m_maxHeLtfRxInHeMuMoreThanOneRu (0),
    m_highestNssSupportedM1 (0),
    m_highestMcsSupported (0),
    m_heSupported (0)
{
  m_txBwMap.resize (8,0);
  m_rxBwMap.resize (8,0);
}

WifiInformationElementId
HeCapabilities::ElementId () const
{
  return IE_EXTENSION;
}

WifiInformationElementId
HeCapabilities::ElementIdExt () const
{
  return IE_EXT_HE_CAPABILITIES;
}

void
HeCapabilities::SetHeSupported (uint8_t heSupported)
{
  m_heSupported = heSupported;
}

uint8_t
HeCapabilities::GetInformationFieldSize () const
{
  //we should not be here if HE is not supported
  NS_ASSERT (m_heSupported > 0);
  // IEEE 802.11ax-2021 9.4.2.248 HE Capabilities element
  // Element ID Extension (1) + HE MAC Capabilities Information (6)
  // + HE PHY Capabilities Information (11) + Supported HE-MCS And NSS Set (4)
  // TODO: Supported HE-MCS And NSS Set field has variable length (4, 8 or 12)
  // TODO: PPE Thresholds field (optional) is not implemented
  return 22;
}

Buffer::Iterator
HeCapabilities::Serialize (Buffer::Iterator i) const
{
  if (m_heSupported < 1)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
HeCapabilities::GetSerializedSize () const
{
  if (m_heSupported < 1)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

void
HeCapabilities::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_heSupported == 1)
    {
      //write the corresponding value for each bit
      start.WriteHtolsbU32 (GetHeMacCapabilitiesInfo1 ());
      start.WriteU16 (GetHeMacCapabilitiesInfo2 ());
      start.WriteHtolsbU64 (GetHePhyCapabilitiesInfo1 ());
      start.WriteHtolsbU16 (GetHePhyCapabilitiesInfo2 ());
      start.WriteU8 (GetHePhyCapabilitiesInfo3 ());
      start.WriteHtolsbU32 (GetSupportedMcsAndNss ());
      // TODO: add another 32-bits field if 160 MHz channel is supported (variable length)
      // TODO: optional PPE Threshold field (variable length)
    }
}

uint8_t
HeCapabilities::DeserializeInformationField (Buffer::Iterator start, uint8_t length)
{
  Buffer::Iterator i = start;
  uint32_t macCapabilities1 = i.ReadLsbtohU32 ();
  uint16_t macCapabilities2 = i.ReadLsbtohU16 ();
  uint64_t phyCapabilities1 = i.ReadLsbtohU64 ();
  uint64_t phyCapabilities2 = i.ReadLsbtohU16 ();
  uint8_t phyCapabilities3 = i.ReadU8 ();
  uint32_t mcsset = i.ReadU32 ();
  SetHeMacCapabilitiesInfo (macCapabilities1, macCapabilities2);
  SetHePhyCapabilitiesInfo (phyCapabilities1, phyCapabilities2, phyCapabilities3);
  SetSupportedMcsAndNss (mcsset);
  // TODO: add another 32-bits field if 160 MHz channel is supported (variable length)
  // TODO: optional PPE Threshold field (variable length)
  return length;
}

void
HeCapabilities::SetHeMacCapabilitiesInfo (uint32_t ctrl1, uint16_t ctrl2)
{
  m_plusHtcHeSupport = ctrl1 & 0x01;
  m_twtRequesterSupport = (ctrl1 >> 1) & 0x01;
  m_twtResponderSupport = (ctrl1 >> 2) & 0x01;
  m_fragmentationSupport = (ctrl1 >> 3) & 0x03;
  m_maximumNumberOfFragmentedMsdus = (ctrl1 >> 5) & 0x07;
  m_minimumFragmentSize = (ctrl1 >> 8) & 0x03;
  m_triggerFrameMacPaddingDuration = (ctrl1 >> 10) & 0x03;
  m_multiTidAggregationRxSupport = (ctrl1 >> 12) & 0x07;
  m_heLinkAdaptation = (ctrl1 >> 15) & 0x03;
  m_allAckSupport = (ctrl1 >> 17) & 0x01;
  m_trsSupport = (ctrl1 >> 18) & 0x01;
  m_bsrSupport = (ctrl1 >> 19) & 0x01;
  m_broadcastTwtSupport = (ctrl1 >> 20) & 0x01;
  m_32bitBaBitmapSupport = (ctrl1 >> 21) & 0x01;
  m_muCascadeSupport = (ctrl1 >> 22) & 0x01;
  m_ackEnabledAggregationSupport = (ctrl1 >> 23) & 0x01;
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities
  // Bit 24 is reserved
  m_omControlSupport = (ctrl1 >> 25) & 0x01;
  m_ofdmaRaSupport = (ctrl1 >> 26) & 0x01;
  m_maxAmpduLengthExponent = (ctrl1 >> 27) & 0x03;
  m_amsduFragmentationSupport = (ctrl1 >> 29) & 0x01;
  m_flexibleTwtScheduleSupport = (ctrl1 >> 30) & 0x01;
  m_rxControlFrameToMultiBss = (ctrl1 >> 31) & 0x01;
  m_bsrpBqrpAmpduAggregation = ctrl2 & 0x01;
  m_qtpSupport = (ctrl2 >> 1) & 0x01;
  m_bqrSupport = (ctrl2 >> 2) & 0x01;
  m_psrResponder = (ctrl2 >> 3) & 0x01;
  m_ndpFeedbackReportSupport = (ctrl2 >> 4) & 0x01;
  m_opsSupport = (ctrl2 >> 5) & 0x01;
  m_amsduNotUnderBaInAmpduSupport = (ctrl2 >> 6) & 0x01;
  m_multiTidAggregationTxSupport = (ctrl2 >> 7) & 0x07;
  m_heSubchannelSelectiveTxSupport = (ctrl2 >> 10) & 0x01;
  m_ul2x996ToneRuSupport = (ctrl2 >> 11) & 0x01;
  m_omControlUlMuDataDisableRxSupport = (ctrl2 >> 12) & 0x01;
  m_heDynamicSmPowerSave = (ctrl2 >> 13) & 0x01;
  m_puncturedSoundingSupport = (ctrl2 >> 14) & 0x01;
  m_heVhtTriggerFrameRxSupport = (ctrl2 >> 15) & 0x01;
}

uint32_t
HeCapabilities::GetHeMacCapabilitiesInfo1 () const
{
  uint32_t val = 0;
  val |= m_plusHtcHeSupport & 0x01;
  val |= (m_twtRequesterSupport & 0x01) << 1;
  val |= (m_twtResponderSupport & 0x01) << 2;
  val |= (m_fragmentationSupport & 0x03) << 3;
  val |= (m_maximumNumberOfFragmentedMsdus & 0x07) << 5;
  val |= (m_minimumFragmentSize & 0x03) << 8;
  val |= (m_triggerFrameMacPaddingDuration & 0x03) << 10;
  val |= (m_multiTidAggregationRxSupport & 0x07) << 12;
  val |= (m_heLinkAdaptation & 0x03) << 15;
  val |= (m_allAckSupport & 0x01) << 17;
  val |= (m_trsSupport & 0x01) << 18;
  val |= (m_bsrSupport & 0x01) << 19;
  val |= (m_broadcastTwtSupport & 0x01) << 20;
  val |= (m_32bitBaBitmapSupport & 0x01) << 21;
  val |= (m_muCascadeSupport & 0x01) << 22;
  val |= (m_ackEnabledAggregationSupport & 0x01) << 23;
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities
  // Bit 24 is reserved
  val |= (m_omControlSupport & 0x01) << 25;
  val |= (m_ofdmaRaSupport & 0x01) << 26;
  val |= (m_maxAmpduLengthExponent & 0x03) << 27;
  val |= (m_amsduFragmentationSupport & 0x01) << 29;
  val |= (m_flexibleTwtScheduleSupport & 0x01) << 30;
  val |= (m_rxControlFrameToMultiBss & 0x01) << 31;
  return val;
}

uint16_t
HeCapabilities::GetHeMacCapabilitiesInfo2 () const
{
  uint16_t val = 0;
  val |= m_bsrpBqrpAmpduAggregation & 0x01;
  val |= (m_qtpSupport & 0x01) << 1;
  val |= (m_bqrSupport & 0x01) << 2;
  val |= (m_psrResponder & 0x01) << 3;
  val |= (m_ndpFeedbackReportSupport & 0x01) << 4;
  val |= (m_opsSupport & 0x01) << 5;
  val |= (m_amsduNotUnderBaInAmpduSupport & 0x01) << 6;
  val |= (m_multiTidAggregationTxSupport & 0x07) << 7;
  val |= (m_heSubchannelSelectiveTxSupport & 0x01) << 10;
  val |= (m_ul2x996ToneRuSupport & 0x01) << 11;
  val |= (m_omControlUlMuDataDisableRxSupport & 0x01) << 12;
  val |= (m_heDynamicSmPowerSave & 0x01) << 13;
  val |= (m_puncturedSoundingSupport & 0x01) << 14;
  val |= (m_heVhtTriggerFrameRxSupport & 0x01) << 15;
  return val;
}

void
HeCapabilities::SetHePhyCapabilitiesInfo (uint64_t ctrl1, uint16_t ctrl2, uint8_t ctrl3)
{
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities
  // Bit 0 is reserved
  m_channelWidthSet = (ctrl1 >> 1) & 0x7f;
  m_puncturedPreambleRx = (ctrl1 >> 8) & 0x0f;
  m_deviceClass = (ctrl1 >> 12) & 0x01;
  m_ldpcCodingInPayload = (ctrl1 >> 13) & 0x01;
  m_heSuPpdu1xHeLtf800nsGi = (ctrl1 >> 14) & 0x01;
  m_midambleRxMaxNsts = (ctrl1 >> 15) & 0x03;
  m_ndp4xHeLtfAnd32msGi = (ctrl1 >> 17) & 0x01;
  m_stbcTxLeq80MHz = (ctrl1 >> 18) & 0x01;
  m_stbcRxLeq80MHz = (ctrl1 >> 19) & 0x01;
  m_dopplerTx = (ctrl1 >> 20) & 0x01;
  m_dopplerRx = (ctrl1 >> 21) & 0x01;
  m_fullBwUlMuMimo = (ctrl1 >> 22) & 0x01;
  m_partialBwUlMuMimo = (ctrl1 >> 23) & 0x01;
  m_dcmMaxConstellationTx = (ctrl1 >> 24) & 0x03;
  m_dcmMaxNssTx = (ctrl1 >> 26) & 0x01;
  m_dcmMaxConstellationRx = (ctrl1 >> 27) & 0x03;
  m_dcmMaxNssRx = (ctrl1 >> 29) & 0x01;
  m_rxPartialBwSuInHeMu = (ctrl1 >> 30) & 0x01;
  m_suBeamformer = (ctrl1 >> 31) & 0x01;
  m_suBeamformee = (ctrl1 >> 32) & 0x01;
  m_muBeamformer = (ctrl1 >> 33) & 0x01;
  m_beamformeeStsForSmallerOrEqualThan80Mhz = (ctrl1 >> 34) & 0x07;
  m_beamformeeStsForLargerThan80Mhz = (ctrl1 >> 37) & 0x07;
  m_numberOfSoundingDimensionsForSmallerOrEqualThan80Mhz = (ctrl1 >> 40) & 0x07;
  m_numberOfSoundingDimensionsForLargerThan80Mhz = (ctrl1 >> 43) & 0x07;
  m_ngEqual16ForSuFeedbackSupport = (ctrl1 >> 46) & 0x01;
  m_ngEqual16ForMuFeedbackSupport = (ctrl1 >> 47) & 0x01;
  m_codebookSize42SuFeedback = (ctrl1 >> 48) & 0x01;
  m_codebookSize75MuFeedback = (ctrl1 >> 49) & 0x01;
  m_triggeredSuBfFeedback = (ctrl1 >> 50) & 0x01;
  m_triggeredMuBfFeedback = (ctrl1 >> 51) & 0x01;
  m_triggeredCqiFeedback = (ctrl1 >> 52) & 0x01;
  m_erPartialBandwidth = (ctrl1 >> 53) & 0x01;
  m_dlMuMimoOnPartialBandwidth = (ctrl1 >> 54) & 0x01;
  m_ppeThresholdPresent = (ctrl1 >> 55) & 0x01;
  m_psrBasedSrSupport = (ctrl1 >> 56) & 0x01;
  m_powerBoostFactorAlphaSupport = (ctrl1 >> 57) & 0x01;
  m_hePpdu4xHeLtf800nsGi = ctrl2 & 0x01;
  m_maxNc = (ctrl1 >> 59) & 0x07;
  m_stbcTxGt80MHz = (ctrl1 >> 62) & 0x01;
  m_stbcRxGt80MHz = (ctrl1 >> 63) & 0x01;
  m_heErSuPpdu4xHeLtf08sGi = ctrl2 & 0x01;
  m_hePpdu20MHzIn40MHz24GHz = (ctrl2 >> 1) & 0x01;
  m_hePpdu20MHzIn160MHz = (ctrl2 >> 2) & 0x01;
  m_hePpdu80MHzIn160MHz = (ctrl2 >> 3) & 0x01;
  m_heErSuPpdu1xHeLtf08Gi = (ctrl2 >> 4) & 0x01;
  m_midamble2xAnd1xHeLtf = (ctrl2 >> 5) & 0x01;
  m_dcmMaxRu = (ctrl2 >> 6) & 0x03;
  m_longerThan16HeSigbOfdm = (ctrl2 >> 8) & 0x01;
  m_nonTriggeredCqiFeedback = (ctrl2 >> 9) & 0x01;
  m_tx1024QamLt242Ru = (ctrl2 >> 10) & 0x01;
  m_rx1024QamLt242Ru = (ctrl2 >> 11) & 0x01;
  m_rxFullBwSuInHeMuCompressedSigB = (ctrl2 >> 12) & 0x01;
  m_rxFullBwSuInHeMuNonCompressedSigB = (ctrl2 >> 13) & 0x01;
  m_nominalPacketPadding = (ctrl2 >> 14) & 0x03;
  m_maxHeLtfRxInHeMuMoreThanOneRu = ctrl3 & 0x01;
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities
  // Bits 81-87 are reserved
}

uint64_t
HeCapabilities::GetHePhyCapabilitiesInfo1 () const
{
  uint64_t val = 0;
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities
  // Bit 0 is reserved
  val |= (m_channelWidthSet & 0x7f) << 1;
  val |= (m_puncturedPreambleRx & 0x0f) << 8;
  val |= (m_deviceClass & 0x01) << 12;
  val |= (m_ldpcCodingInPayload & 0x01) << 13;
  val |= (m_heSuPpdu1xHeLtf800nsGi & 0x01) << 14;
  val |= (m_midambleRxMaxNsts & 0x03) << 15;
  val |= (m_ndp4xHeLtfAnd32msGi & 0x01) << 17;
  val |= (m_stbcTxLeq80MHz & 0x01) << 18;
  val |= (m_stbcRxLeq80MHz & 0x01) << 19;
  val |= (m_dopplerTx & 0x01) << 20;
  val |= (m_dopplerRx & 0x01) << 21;
  val |= (m_fullBwUlMuMimo & 0x01) << 22;
  val |= (m_partialBwUlMuMimo & 0x01) << 23;
  val |= (m_dcmMaxConstellationTx & 0x03) << 24;
  val |= (m_dcmMaxNssTx & 0x01) << 26;
  val |= (m_dcmMaxConstellationRx & 0x03) << 27;
  val |= (m_dcmMaxNssRx & 0x01) << 29;
  val |= (m_rxPartialBwSuInHeMu & 0x01) << 30;
  val |= (m_suBeamformer & 0x01) << 31;
  val |= (static_cast<uint64_t> (m_suBeamformee) & 0x01) << 32;
  val |= (static_cast<uint64_t> (m_muBeamformer) & 0x01) << 33;
  val |= (static_cast<uint64_t> (m_beamformeeStsForSmallerOrEqualThan80Mhz) & 0x07) << 34;
  val |= (static_cast<uint64_t> (m_beamformeeStsForLargerThan80Mhz) & 0x07) << 37;
  val |= (static_cast<uint64_t> (m_numberOfSoundingDimensionsForSmallerOrEqualThan80Mhz) & 0x07) << 40;
  val |= (static_cast<uint64_t> (m_numberOfSoundingDimensionsForLargerThan80Mhz) & 0x07) << 43;
  val |= (static_cast<uint64_t> (m_ngEqual16ForSuFeedbackSupport) & 0x01) << 46;
  val |= (static_cast<uint64_t> (m_ngEqual16ForMuFeedbackSupport) & 0x01) << 47;
  val |= (static_cast<uint64_t> (m_codebookSize42SuFeedback) & 0x01) << 48;
  val |= (static_cast<uint64_t> (m_codebookSize75MuFeedback) & 0x01) << 49;
  val |= (static_cast<uint64_t> (m_triggeredSuBfFeedback) & 0x01) << 50;
  val |= (static_cast<uint64_t> (m_triggeredMuBfFeedback) & 0x01) << 51;
  val |= (static_cast<uint64_t> (m_triggeredCqiFeedback) & 0x01) << 52;
  val |= (static_cast<uint64_t> (m_erPartialBandwidth) & 0x01) << 53;
  val |= (static_cast<uint64_t> (m_dlMuMimoOnPartialBandwidth) & 0x01) << 54;
  val |= (static_cast<uint64_t> (m_ppeThresholdPresent) & 0x01) << 55;
  val |= (static_cast<uint64_t> (m_psrBasedSrSupport) & 0x01) << 56;
  val |= (static_cast<uint64_t> (m_powerBoostFactorAlphaSupport) & 0x01) << 57;
  val |= (static_cast<uint64_t> (m_hePpdu4xHeLtf800nsGi) & 0x01) << 58;
  val |= (static_cast<uint64_t> (m_maxNc) & 0x07) << 59;
  val |= (static_cast<uint64_t> (m_stbcTxGt80MHz) & 0x01) << 62;
  val |= (static_cast<uint64_t> (m_stbcRxGt80MHz) & 0x01) << 63;
  return val;
}

uint16_t
HeCapabilities::GetHePhyCapabilitiesInfo2 () const
{
  uint16_t val = 0;
  val |= m_heErSuPpdu4xHeLtf08sGi & 0x01;
  val |= (m_hePpdu20MHzIn40MHz24GHz & 0x01) << 1;
  val |= (m_hePpdu20MHzIn160MHz & 0x01) << 2;
  val |= (m_hePpdu80MHzIn160MHz & 0x01) << 3;
  val |= (m_heErSuPpdu1xHeLtf08Gi & 0x01) << 4;
  val |= (m_midamble2xAnd1xHeLtf & 0x01) << 5;
  val |= (m_dcmMaxRu & 0x03) << 6;
  val |= (m_longerThan16HeSigbOfdm & 0x01) << 8;
  val |= (m_nonTriggeredCqiFeedback & 0x01) << 9;
  val |= (m_tx1024QamLt242Ru & 0x01) << 10;
  val |= (m_rx1024QamLt242Ru & 0x01) << 11;
  val |= (m_rxFullBwSuInHeMuCompressedSigB & 0x01) << 12;
  val |= (m_rxFullBwSuInHeMuNonCompressedSigB & 0x01) << 13;
  val |= (m_nominalPacketPadding & 0x03) << 14;
  return val;
}

uint8_t
HeCapabilities::GetHePhyCapabilitiesInfo3 () const
{
  uint8_t val = 0;
  val |= m_maxHeLtfRxInHeMuMoreThanOneRu & 0x01;
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities
  // Bits 81-87 are reserved
  return val;
}

void
HeCapabilities::SetSupportedMcsAndNss (uint16_t ctrl)
{
  m_highestNssSupportedM1 = ctrl & 0x07;
  m_highestMcsSupported = (ctrl >> 3) & 0x07;
  NS_ASSERT (m_highestMcsSupported <= 4);
  uint8_t i;
  for (i = 0; i < 5; i++)
    {
      m_txBwMap[i] = (ctrl >> (6 + i)) & 0x01;
    }
  for (i = 0; i < 5; i++)
    {
      m_rxBwMap[i] = (ctrl >> (11 + i)) & 0x01;
    }
  // TODO: MCS NSS Descriptors
}

uint16_t
HeCapabilities::GetSupportedMcsAndNss () const
{
  uint16_t val = 0;
  val |= m_highestNssSupportedM1 & 0x07;
  val |= (m_highestMcsSupported & 0x07) << 3;
  uint8_t i;
  for (i = 0; i < 5; i++)
    {
      val |= (m_txBwMap[i] & 0x01) << (6 + 1);
    }
  for (i = 0; i < 5; i++)
    {
      val |= (m_rxBwMap[i] & 0x01) << (11 + 1);
    }
  // TODO: MCS NSS Descriptors
  return val;
}

// TODO: PPE threshold

bool
HeCapabilities::IsSupportedTxMcs (uint8_t mcs) const
{
  NS_ASSERT (mcs >= 0 && mcs <= 11);
  if (mcs <= 7)
    {
      return true;
    }
  if (mcs == 8 && m_highestMcsSupported >= 1)
    {
      return true;
    }
  if (mcs == 9 && m_highestMcsSupported >= 2)
    {
      return true;
    }
  if (mcs == 10 && m_highestMcsSupported >= 3)
    {
      return true;
    }
  if (mcs == 11 && m_highestMcsSupported == 4)
    {
      return true;
    }
  return false;
}

bool
HeCapabilities::IsSupportedRxMcs (uint8_t mcs) const
{
  NS_ASSERT (mcs >= 0 && mcs <= 11);
  if (mcs <= 7)
    {
      return true;
    }
  if (mcs == 8 && m_highestMcsSupported >= 1)
    {
      return true;
    }
  if (mcs == 9 && m_highestMcsSupported >= 2)
    {
      return true;
    }
  if (mcs == 10 && m_highestMcsSupported >= 3)
    {
      return true;
    }
  if (mcs == 11 && m_highestMcsSupported == 4)
    {
      return true;
    }
  return false;
}

void
HeCapabilities::SetChannelWidthSet (uint8_t channelWidthSet)
{
  NS_ASSERT (channelWidthSet <= 0x2f);
  m_channelWidthSet = channelWidthSet;
}

void
HeCapabilities::SetLdpcCodingInPayload (uint8_t ldpcCodingInPayload)
{
  m_ldpcCodingInPayload = ldpcCodingInPayload;
}

void
HeCapabilities::SetHeSuPpdu1xHeLtf800nsGi (bool heSuPpdu1xHeLtf800nsGi)
{
  m_heSuPpdu1xHeLtf800nsGi = (heSuPpdu1xHeLtf800nsGi ? 1 : 0);
}

void
HeCapabilities::SetHePpdu4xHeLtf800nsGi (bool hePpdu4xHeLtf800nsGi)
{
  m_hePpdu4xHeLtf800nsGi = (hePpdu4xHeLtf800nsGi ? 1 : 0);
}

void
HeCapabilities::SetMaxAmpduLength (uint32_t maxAmpduLength)
{
  for (uint8_t i = 0; i <= 3; i++)
    {
      if ((1ul << (20 + i)) - 1 == maxAmpduLength)
        {
          m_maxAmpduLengthExponent = i;
          return;
        }
    }
  NS_ABORT_MSG ("Invalid A-MPDU Max Length value");
}

void
HeCapabilities::SetHighestMcsSupported (uint8_t mcs)
{
  NS_ASSERT (mcs >= 7 && mcs <= 11);
  m_highestMcsSupported = mcs - 7;
}

void
HeCapabilities::SetHighestNssSupported (uint8_t nss)
{
  NS_ASSERT (nss >= 1 && nss <= 8);
  m_highestNssSupportedM1 = nss - 1;
}

uint8_t
HeCapabilities::GetChannelWidthSet (void) const
{
  return m_channelWidthSet;
}

uint8_t
HeCapabilities::GetLdpcCodingInPayload (void) const
{
  return m_ldpcCodingInPayload;
}

bool
HeCapabilities::GetHeSuPpdu1xHeLtf800nsGi (void) const
{
  return (m_heSuPpdu1xHeLtf800nsGi == 1);
}

bool
HeCapabilities::GetHePpdu4xHeLtf800nsGi (void) const
{
  return (m_hePpdu4xHeLtf800nsGi == 1);
}


uint8_t
HeCapabilities::GetHighestMcsSupported (void) const
{
  return m_highestMcsSupported + 7;
}

uint8_t
HeCapabilities::GetHighestNssSupported (void) const
{
  return m_highestNssSupportedM1 + 1;
}

uint32_t
HeCapabilities::GetMaxAmpduLength (void) const
{
  return std::min<uint32_t> ((1ul << (20 + m_maxAmpduLengthExponent)) - 1, 6500631);
}

std::ostream &
operator << (std::ostream &os, const HeCapabilities &HeCapabilities)
{
  os << HeCapabilities.GetHeMacCapabilitiesInfo1 () << "|"
     << +HeCapabilities.GetHeMacCapabilitiesInfo2 () << "|"
     << HeCapabilities.GetHePhyCapabilitiesInfo1 () << "|"
     << HeCapabilities.GetHePhyCapabilitiesInfo2 () << "|"
     << +HeCapabilities.GetHePhyCapabilitiesInfo3 () << "|"
     << HeCapabilities.GetSupportedMcsAndNss ();
  return os;
}

} //namespace ns3
