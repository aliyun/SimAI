/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
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

#ifndef HE_CAPABILITIES_H
#define HE_CAPABILITIES_H

#include "ns3/wifi-information-element.h"

namespace ns3 {

/**
 * \ingroup wifi
 *
 * The IEEE 802.11ax HE Capabilities
 */
class HeCapabilities : public WifiInformationElement
{
public:
  HeCapabilities ();

  // Implementations of pure virtual methods of WifiInformationElement
  WifiInformationElementId ElementId () const override;
  WifiInformationElementId ElementIdExt () const override;
  uint8_t GetInformationFieldSize () const override;
  void SerializeInformationField (Buffer::Iterator start) const override;
  uint8_t DeserializeInformationField (Buffer::Iterator start, uint8_t length) override;
  /* This information element is a bit special in that it is only
     included if the STA is an HE STA. To support this we
     override the Serialize and GetSerializedSize methods of
     WifiInformationElement. */
  Buffer::Iterator Serialize (Buffer::Iterator start) const override;
  uint16_t GetSerializedSize () const override;

  /**
   * Set HE supported
   * \param heSupported the HE supported indicator
   */
  void SetHeSupported (uint8_t heSupported);

  /**
   * Set the HE MAC Capabilities Info field in the HE Capabilities information element.
   *
   * \param ctrl1 the HE MAC Capabilities Info field 1 in the HE Capabilities information element
   * \param ctrl2 the HE MAC Capabilities Info field 2 in the HE Capabilities information element
   */
  void SetHeMacCapabilitiesInfo (uint32_t ctrl1, uint16_t ctrl2);
  /**
   * Set the HE PHY Capabilities Info field in the HE Capabilities information element.
   *
   * \param ctrl1 the HE PHY Capabilities Info field 1 in the HE Capabilities information element
   * \param ctrl2 the HE PHY Capabilities Info field 2 in the HE Capabilities information element
   * \param ctrl3 the HE PHY Capabilities Info field 3 in the HE Capabilities information element
   */
  void SetHePhyCapabilitiesInfo (uint64_t ctrl1, uint16_t ctrl2, uint8_t ctrl3);
  /**
   * Set the MCS and NSS field in the HE Capabilities information element.
   *
   * \param ctrl the MCS and NSS field in the HE Capabilities information element
   */
  void SetSupportedMcsAndNss (uint16_t ctrl);

  /**
   * Return the 4 first octets of the HE MAC Capabilities Info field in the HE Capabilities information element.
   *
   * \return the 4 first octets of the HE MAC Capabilities Info field in the HE Capabilities information element
   */
  uint32_t GetHeMacCapabilitiesInfo1 () const;
  /**
   * Return the last 2 octets of the HE MAC Capabilities Info field in the HE Capabilities information element.
   *
   * \return the last 2 octets of the HE MAC Capabilities Info field in the HE Capabilities information element
   */
  uint16_t GetHeMacCapabilitiesInfo2 () const;
  /**
   * Return the 8 first octets of the HE PHY Capabilities Info field in the HE Capabilities information element.
   *
   * \return the 8 first octets of the HE PHY Capabilities Info field in the HE Capabilities information element
   */
  uint64_t GetHePhyCapabilitiesInfo1 () const;
  /**
   * Return the octets 9-10 of the HE PHY Capabilities Info field in the HE Capabilities information element.
   *
   * \return the octets 9-10 of the HE PHY Capabilities Info field in the HE Capabilities information element
   */
  uint16_t GetHePhyCapabilitiesInfo2 () const;
  /**
   * Return the last octet of the HE PHY Capabilities Info field in the HE Capabilities information element.
   *
   * \return the last octet of the HE PHY Capabilities Info field in the HE Capabilities information element
   */
  uint8_t GetHePhyCapabilitiesInfo3 () const;
  /**
   * Return the MCS and NSS field in the HE Capabilities information element.
   *
   * \return the MCS and NSS field in the HE Capabilities information element
   */
  uint16_t GetSupportedMcsAndNss () const;

  // PHY Capabilities Info fields
  /**
   * Set channel width set.
   *
   * \param channelWidthSet the channel width set
   */
  void SetChannelWidthSet (uint8_t channelWidthSet);
  /**
   * Set indication whether the transmission and reception of LDPC encoded packets is supported.
   *
   * \param ldpcCodingInPayload indication whether the transmission and reception of LDPC encoded packets is supported
   */
  void SetLdpcCodingInPayload (uint8_t ldpcCodingInPayload);
  /**
   * Set 1xHE-LTF and 800ns GI in HE SU PPDU reception support
   *
   * \param heSuPpdu1xHeLtf800nsGi 1xHE-LTF and 800ns GI in HE SU PPDU reception support
   */
  void SetHeSuPpdu1xHeLtf800nsGi (bool heSuPpdu1xHeLtf800nsGi);
  /**
   * Set 4xHE-LTF and 800ns GI in HE SU PPDU and HE MU PPDU reception support
   *
   * \param heSuPpdu4xHeLtf800nsGi 4xHE-LTF and 800ns GI in HE SU PPDU and HE MU PPDU reception support
   */
  void SetHePpdu4xHeLtf800nsGi (bool heSuPpdu4xHeLtf800nsGi);
  /**
   * Get channel width set.
   *
   * \returns the channel width set
   */
  uint8_t GetChannelWidthSet (void) const;
  /**
   * Indicates support for the transmission and reception of LDPC encoded packets.
   *
   * \returns indication whether the transmission and reception of LDPC encoded packets is supported
   */
  uint8_t GetLdpcCodingInPayload (void) const;
  /**
   * Get 1xHE-LTF and 800ns GI in HE SU PPDU reception support
   *
   * \returns true if 1xHE-LTF and 800ns GI in HE SU PPDU reception is supported, false otherwise
   */
  bool GetHeSuPpdu1xHeLtf800nsGi (void) const;
  /**
   * Get 4xHE-LTF and 800ns GI in HE SU PPDU and HE MU PPDU reception support
   *
   * \returns true if 4xHE-LTF and 800ns GI in HE SU PPDU and HE MU PPDU reception is supported,
   *          false otherwise
   */
  bool GetHePpdu4xHeLtf800nsGi (void) const;
  /**
   * Get highest MCS supported.
   *
   * \returns the highest MCS is supported
   */
  uint8_t GetHighestMcsSupported (void) const;
  /**
   * Get highest NSS supported.
   *
   * \returns the highest supported NSS
   */
  uint8_t GetHighestNssSupported (void) const;

  // MAC Capabilities Info fields
  /**
   * Set the maximum AMPDU length.
   *
   * \param maxAmpduLength 2^(20 + x) - 1, x in the range 0 to 3
   */
  void SetMaxAmpduLength (uint32_t maxAmpduLength);

  // MCS and NSS field information
  /**
   * Set highest MCS supported.
   *
   * \param mcs the MCS
   */
  void SetHighestMcsSupported (uint8_t mcs);
  /**
   * Set highest NSS supported.
   *
   * \param nss the NSS
   */
  void SetHighestNssSupported (uint8_t nss);

  /**
   * Is RX MCS supported.
   *
   * \param mcs the MCS
   * \returns true if MCS transmit supported
   */
  bool IsSupportedTxMcs (uint8_t mcs) const;
  /**
   * Is RX MCS supported.
   *
   * \param mcs the MCS
   * \returns true if MCS receive supported
   */
  bool IsSupportedRxMcs (uint8_t mcs) const;

  /**
   * Return the maximum A-MPDU length.
   *
   * \return the maximum A-MPDU length
   */
  uint32_t GetMaxAmpduLength (void) const;


private:
  // MAC Capabilities Info fields
  // IEEE 802.11ax-2021 9.4.2.248.2 HE MAC Capabilities Information field
  uint8_t m_plusHtcHeSupport;                            //!< HTC HE support
  uint8_t m_twtRequesterSupport;                         //!< TWT requester support
  uint8_t m_twtResponderSupport;                         //!< TWT responder support
  uint8_t m_fragmentationSupport;                        //!< fragmentation support
  uint8_t m_maximumNumberOfFragmentedMsdus;              //!< maximum number of fragmentation MSDUs
  uint8_t m_minimumFragmentSize;                         //!< minimum fragment size
  uint8_t m_triggerFrameMacPaddingDuration;              //!< trigger frame MAC padding duration
  uint8_t m_multiTidAggregationRxSupport;                //!< multi-TID aggregation Rx support
  uint8_t m_heLinkAdaptation;                            //!< HE link adaptation
  uint8_t m_allAckSupport;                               //!< all Ack support
  uint8_t m_trsSupport;                                  //!< TRS support
  uint8_t m_bsrSupport;                                  //!< BSR support
  uint8_t m_broadcastTwtSupport;                         //!< broadcast TXT support
  uint8_t m_32bitBaBitmapSupport;                        //!< 32-bit BA bitmap support
  uint8_t m_muCascadeSupport;                            //!< MU cascade support
  uint8_t m_ackEnabledAggregationSupport;                //!< ack enabled aggregation support
  uint8_t m_omControlSupport;                            //!< operation mode control support
  uint8_t m_ofdmaRaSupport;                              //!< OFDMA RA support
  uint8_t m_maxAmpduLengthExponent;                      //!< maximum A-MPDU length exponent extension
  uint8_t m_amsduFragmentationSupport;                   //!< A-MSDU fragmentation support
  uint8_t m_flexibleTwtScheduleSupport;                  //!< flexible TWT schedule support
  uint8_t m_rxControlFrameToMultiBss;                    //!< receive control frame to multi-BSS
  uint8_t m_bsrpBqrpAmpduAggregation;                    //!< BSRP BQRP A-MPDU aggregation
  uint8_t m_qtpSupport;                                  //!< QTP support
  uint8_t m_bqrSupport;                                  //!< BQR support
  uint8_t m_psrResponder;                                //!< PSR responder
  uint8_t m_ndpFeedbackReportSupport;                    //!< NDP feedback report support
  uint8_t m_opsSupport;                                  //!< OPS support
  uint8_t m_amsduNotUnderBaInAmpduSupport;               //!< AMSDU not under BA in Ack enabled A-MPDU support
  uint8_t m_multiTidAggregationTxSupport;                //!< Multi-TID aggregation TX support
  uint8_t m_heSubchannelSelectiveTxSupport;              //!< HE subchannel selective transmission support
  uint8_t m_ul2x996ToneRuSupport;                        //!< UL 2x996 tone RU support
  uint8_t m_omControlUlMuDataDisableRxSupport;           //!< OM control UL MU data disable RX support
  uint8_t m_heDynamicSmPowerSave;                        //!< HE dynamic SM power save
  uint8_t m_puncturedSoundingSupport;                    //!< punctured sounding support
  uint8_t m_heVhtTriggerFrameRxSupport;                  //!< HE and VHT trigger frame RX support

  // PHY Capabilities Info fields
  // IEEE 802.11ax-2021 9.4.2.248.3 HE PHY Capabilities Information field
  uint8_t m_channelWidthSet;                         //!< channel width set
  uint8_t m_puncturedPreambleRx;                     //!< Punctured preamble Rx
  uint8_t m_deviceClass;                             //!< device class
  uint8_t m_ldpcCodingInPayload;                     //!< LDPC coding in payload
  uint8_t m_heSuPpdu1xHeLtf800nsGi;                  //!< HE SU PPDU with 1x HE LTF and 0.8us GI
  uint8_t m_midambleRxMaxNsts;                       //!< Midamble TX/RX max NSTS
  uint8_t m_ndp4xHeLtfAnd32msGi;                     //!< NDP with 4x HE-LTF and 3.2us GI
  uint8_t m_stbcTxLeq80MHz;                          //!< STBC TX <= 80MHz
  uint8_t m_stbcRxLeq80MHz;                          //!< STBC RX <= 80Mhz
  uint8_t m_dopplerTx;                               //!< Doppler Tx
  uint8_t m_dopplerRx;                               //!< Doppler Rx
  uint8_t m_fullBwUlMuMimo;                          //!< Full Bandwidth UL MU-MIMO
  uint8_t m_partialBwUlMuMimo;                       //!< Partial Bandwidth UL MU-MIMO
  uint8_t m_dcmMaxConstellationTx;                   //!< DCM Max Constellation Tx
  uint8_t m_dcmMaxNssTx;                             //!< DCM Max NSS Tx
  uint8_t m_dcmMaxConstellationRx;                   //!< DCM Max Constellation Rx
  uint8_t m_dcmMaxNssRx;                             //!< DCM Max NSS Rx
  uint8_t m_rxPartialBwSuInHeMu;                     //!< Rx Partial BW SU in 20 MHz HE MU PPDU
  uint8_t m_suBeamformer;                            //!< SU beamformer
  uint8_t m_suBeamformee;                            //!< SU beamformee
  uint8_t m_muBeamformer;                            //!< MU beamformer
  uint8_t m_beamformeeStsForSmallerOrEqualThan80Mhz; //!< beam formee STS for < 80 MHz
  uint8_t m_beamformeeStsForLargerThan80Mhz;         //!< beamformee STS for > 80MHz
  uint8_t m_numberOfSoundingDimensionsForSmallerOrEqualThan80Mhz; //!< # of sounding dimensions for < 80 MHz
  uint8_t m_numberOfSoundingDimensionsForLargerThan80Mhz;         //!< # of sounding dimensions for > 80 MHz
  uint8_t m_ngEqual16ForSuFeedbackSupport;          //!< equal 16 for SU feedback
  uint8_t m_ngEqual16ForMuFeedbackSupport;          //!< equal 16 for MU feedback
  uint8_t m_codebookSize42SuFeedback;               //!< Codebook Size = {4, 2} SU feedback
  uint8_t m_codebookSize75MuFeedback;               //!< Codebook Size = {7, 5} MU feedback
  uint8_t m_triggeredSuBfFeedback;                  //!< Triggered SU beamforming feedback
  uint8_t m_triggeredMuBfFeedback;                  //!< Triggered MU beamforming feedback
  uint8_t m_triggeredCqiFeedback;                   //!< Triggered CQI feedback
  uint8_t m_erPartialBandwidth;                     //!< Extended range partial bandwidth
  uint8_t m_dlMuMimoOnPartialBandwidth;             //!< DL MU-MIMO on partial bandwidth
  uint8_t m_ppeThresholdPresent;                    //!< PPE threshold present
  uint8_t m_psrBasedSrSupport;                      //!< PSR based SR support
  uint8_t m_powerBoostFactorAlphaSupport;           //!< power boost factor alpha support
  uint8_t m_hePpdu4xHeLtf800nsGi;                   //!< 4 times HE-LFT and 800ns GI support for HE-PPDUs
  uint8_t m_maxNc;                                  //!< Max Nc for HE compressed beamforming/CQI report
  uint8_t m_stbcTxGt80MHz;                          //!< STBC Tx > 80MHz
  uint8_t m_stbcRxGt80MHz;                          //!< STBC RX > 80MHz
  uint8_t m_heErSuPpdu4xHeLtf08sGi;                 //!< HE ER SU PPDU with 4x HE LTF and 0.8us GI
  uint8_t m_hePpdu20MHzIn40MHz24GHz;                //!< 20MHz in 40MHz HE PPDU in 2.4GHz band
  uint8_t m_hePpdu20MHzIn160MHz;                    //!< 20MHz in 160/80+80MHz HE PPDU
  uint8_t m_hePpdu80MHzIn160MHz;                    //!< 80MHz in 160/80+80MHz HE PPDU
  uint8_t m_heErSuPpdu1xHeLtf08Gi;                  //!< HE ER SU PPDU with 1x HE LTF and 0.8us GI
  uint8_t m_midamble2xAnd1xHeLtf;                   //!< Midamble TX/RX 2x and 1x HE-LTF
  uint8_t m_dcmMaxRu;                               //!< DCM Max RU
  uint8_t m_longerThan16HeSigbOfdm;                 //!< Longer than 16 HE SIG-=B OFDM symbols support
  uint8_t m_nonTriggeredCqiFeedback;                //!< Non-Triggered CQI feedback
  uint8_t m_tx1024QamLt242Ru;                       //!< TX 1024 QAM < 242 =-tone RU support
  uint8_t m_rx1024QamLt242Ru;                       //!< TX 1024 QAM < 242 =-tone RU support
  uint8_t m_rxFullBwSuInHeMuCompressedSigB;         //!< RX full BW SU using HE MU PPDU with compressed SIGB
  uint8_t m_rxFullBwSuInHeMuNonCompressedSigB;      //!< RX full BW SU using HE MU PPDU with non-compressed SIGB
  uint8_t m_nominalPacketPadding;                   //!< Nominal packet padding
  uint8_t m_maxHeLtfRxInHeMuMoreThanOneRu;          ///< max HE-LTF symbols STA can Rx in HE MU PPDU with more than one RU

  //MCS and NSS field information
  uint8_t m_highestNssSupportedM1; //!< highest NSS support M1
  uint8_t m_highestMcsSupported;   //!< highest MCS support
  std::vector<uint8_t> m_txBwMap;  //!< transmit BW map
  std::vector<uint8_t> m_rxBwMap;  //!< receive BW map

  /// This is used to decide if this element should be added to the frame or not
  uint8_t m_heSupported;
};

/**
 * output stream output operator
 * \param os the output stream
 * \param HeCapabilities the HE capabilities
 * \returns the output stream
 */
std::ostream &operator << (std::ostream &os, const HeCapabilities &HeCapabilities);

} //namespace ns3

#endif /* HE_CAPABILITY_H */
