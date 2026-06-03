/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Orange Labs
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
 * Authors: Rediet <getachew.redieteab@orange.com>
 *          Sébastien Deronne <sebastien.deronne@gmail.com> (for logic ported from wifi-phy and spectrum-wifi-phy)
 */

#ifndef HE_PHY_H
#define HE_PHY_H

#include "ns3/vht-phy.h"
#include "ns3/wifi-phy-band.h"
#include "ns3/callback.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::HePhy class
 * and ns3::HeSigAParameters struct.
 */

namespace ns3 {

/**
 * This defines the BSS membership value for HE PHY.
 */
#define HE_PHY 125

/**
 * Parameters for received HE-SIG-A for OBSS_PD based SR
 */
struct HeSigAParameters
{
  double rssiW;     ///< RSSI in W
  uint8_t bssColor; ///< BSS color
};

/**
 * \brief PHY entity for HE (11ax)
 * \ingroup wifi
 *
 * HE PHY is based on VHT PHY.
 *
 * Refer to P802.11ax/D4.0, clause 27.
 */
class HePhy : public VhtPhy
{
public:
  /**
   * Callback upon end of HE-SIG-A
   *
   * arg1: Parameters of HE-SIG-A
   */
  typedef Callback<void, HeSigAParameters> EndOfHeSigACallback;

  /**
   * Constructor for HE PHY
   *
   * \param buildModeList flag used to add HE modes to list (disabled
   *                      by child classes to only add child classes' modes)
   */
  HePhy (bool buildModeList = true);
  /**
   * Destructor for HE PHY
   */
  virtual ~HePhy ();

  WifiMode GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const override;
  WifiMode GetSigAMode (void) const override;
  WifiMode GetSigBMode (const WifiTxVector& txVector) const override;
  const PpduFormats & GetPpduFormats (void) const override;
  Time GetLSigDuration (WifiPreamble preamble) const override;
  Time GetTrainingDuration (const WifiTxVector& txVector,
                            uint8_t nDataLtf,
                            uint8_t nExtensionLtf = 0) const override;
  Time GetSigADuration (WifiPreamble preamble) const override;
  Time GetSigBDuration (const WifiTxVector& txVector) const override;
  Ptr<WifiPpdu> BuildPpdu (const WifiConstPsduMap & psdus,
                           const WifiTxVector& txVector,
                           Time ppduDuration) override;
  Ptr<const WifiPsdu> GetAddressedPsduInPpdu (Ptr<const WifiPpdu> ppdu) const override;
  void StartReceivePreamble (Ptr<WifiPpdu> ppdu,
                             RxPowerWattPerChannelBand& rxPowersW,
                             Time rxDuration) override;
  void CancelAllEvents (void) override;
  uint16_t GetStaId (const Ptr<const WifiPpdu> ppdu) const override;
  uint16_t GetMeasurementChannelWidth (const Ptr<const WifiPpdu> ppdu) const override;
  void StartTx (Ptr<WifiPpdu> ppdu) override;
  Time CalculateTxDuration (WifiConstPsduMap psduMap, const WifiTxVector& txVector, WifiPhyBand band) const override;

  /**
   * \return the BSS color of this PHY.
   */
  uint8_t GetBssColor (void) const;

  /**
   * Compute the L-SIG length value corresponding to the given HE TB PPDU duration.
   * If the latter is not a feasible duration (considering the selected guard interval),
   * a proper duration is computed and returned along with the L-SIG length value.
   *
   * \param ppduDuration the duration of the HE TB PPDU
   * \param txVector the TXVECTOR used for the transmission of this HE TB PPDU
   * \param band the frequency band being used
   *
   * \return the L-SIG length value and the adjusted HE TB PPDU duration.
   */
  static std::pair<uint16_t, Time> ConvertHeTbPpduDurationToLSigLength (Time ppduDuration,
                                                                        const WifiTxVector& txVector,
                                                                        WifiPhyBand band);
  /**
   * \param length the L-SIG length value
   * \param txVector the TXVECTOR used for the transmission of this HE TB PPDU
   * \param band the frequency band being used
   *
   * \return the duration of the HE TB PPDU corresponding to that L-SIG length value.
   */
  static Time ConvertLSigLengthToHeTbPpduDuration (uint16_t length, const WifiTxVector& txVector, WifiPhyBand band);
  /**
   * \param txVector the transmission parameters used for the HE TB PPDU
   *
   * \return the duration of the non-OFDMA portion of the HE TB PPDU.
   */
  Time CalculateNonOfdmaDurationForHeTb (const WifiTxVector& txVector) const;

  /**
   * Get the band in the TX spectrum associated with the RU used by the PSDU
   * transmitted to/by a given STA in a DL MU PPDU/HE TB PPDU
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the STA-ID of the station
   *
   * \return the RU band in the TX spectrum
   */
  WifiSpectrumBand GetRuBandForTx (const WifiTxVector& txVector, uint16_t staId) const;
  /**
   * Get the band in the RX spectrum associated with the RU used by the PSDU
   * transmitted to/by a given STA in a DL MU PPDU/HE TB PPDU
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the STA-ID of the station
   *
   * \return the RU band in the RX spectrum
   */
  WifiSpectrumBand GetRuBandForRx (const WifiTxVector& txVector, uint16_t staId) const;
  /**
   * Get the band used to transmit the non-OFDMA part of an HE TB PPDU.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the STA-ID of the station taking part of the UL MU
   *
   * \return the spectrum band used to transmit the non-OFDMA part of an HE TB PPDU
   */
  WifiSpectrumBand GetNonOfdmaBand (const WifiTxVector& txVector, uint16_t staId) const;
  /**
   * Get the width in MHz of the non-OFDMA portion of an HE TB PPDU
   *
   * \param ru the RU in which the HE TB PPDU is sent
   * \return the width in MHz of the non-OFDMA portion of an HE TB PPDU
   */
  uint16_t GetNonOfdmaWidth (HeRu::RuSpec ru) const;

  /**
   * \return the UID of the HE TB PPDU being received
   */
  uint64_t GetCurrentHeTbPpduUid (void) const;

  /**
   * Set the TRIGVECTOR and the associated expiration time. A TRIGVECTOR shall expire
   * when the TX timer associated with the transmission of the Trigger Frame expires.
   *
   * \param trigVector the TRIGVECTOR
   * \param validity the amount of time (from now) until expiration of the TRIGVECTOR
   */
  void SetTrigVector (const WifiTxVector& trigVector, Time validity);

  /**
   * Get the center frequency of the non-OFDMA part of the current TxVector for the
   * given STA-ID.
   * Note this method is only to be used for UL MU.
   *
   * \param txVector the TXVECTOR that has the RU allocation
   * \param staId the STA-ID of the station taking part of the UL MU
   * \return the center frequency in MHz corresponding to the non-OFDMA part of the HE TB PPDU
   */
  uint16_t GetCenterFrequencyForNonOfdmaPart (const WifiTxVector& txVector, uint16_t staId) const;

  /**
   * Set a callback for a end of HE-SIG-A.
   *
   * \param callback the EndOfHeSigACallback to set
   */
  void SetEndOfHeSigACallback (EndOfHeSigACallback callback);

  /**
   * Fire a EndOfHeSigA callback (if connected) once HE-SIG-A field has been received.
   * This method is scheduled immediatly after end of HE-SIG-A, once
   * field processing is finished.
   *
   * \param params the HE-SIG-A parameters
   */
  void NotifyEndOfHeSigA (HeSigAParameters params);

  /**
   * Initialize all HE modes.
   */
  static void InitializeModes (void);
  /**
   * Return the HE MCS corresponding to
   * the provided index.
   *
   * \param index the index of the MCS
   * \return an HE MCS
   */
  static WifiMode GetHeMcs (uint8_t index);

  /**
   * Return MCS 0 from HE MCS values.
   *
   * \return MCS 0 from HE MCS values
   */
  static WifiMode GetHeMcs0 (void);
  /**
   * Return MCS 1 from HE MCS values.
   *
   * \return MCS 1 from HE MCS values
   */
  static WifiMode GetHeMcs1 (void);
  /**
   * Return MCS 2 from HE MCS values.
   *
   * \return MCS 2 from HE MCS values
   */
  static WifiMode GetHeMcs2 (void);
  /**
   * Return MCS 3 from HE MCS values.
   *
   * \return MCS 3 from HE MCS values
   */
  static WifiMode GetHeMcs3 (void);
  /**
   * Return MCS 4 from HE MCS values.
   *
   * \return MCS 4 from HE MCS values
   */
  static WifiMode GetHeMcs4 (void);
  /**
   * Return MCS 5 from HE MCS values.
   *
   * \return MCS 5 from HE MCS values
   */
  static WifiMode GetHeMcs5 (void);
  /**
   * Return MCS 6 from HE MCS values.
   *
   * \return MCS 6 from HE MCS values
   */
  static WifiMode GetHeMcs6 (void);
  /**
   * Return MCS 7 from HE MCS values.
   *
   * \return MCS 7 from HE MCS values
   */
  static WifiMode GetHeMcs7 (void);
  /**
   * Return MCS 8 from HE MCS values.
   *
   * \return MCS 8 from HE MCS values
   */
  static WifiMode GetHeMcs8 (void);
  /**
   * Return MCS 9 from HE MCS values.
   *
   * \return MCS 9 from HE MCS values
   */
  static WifiMode GetHeMcs9 (void);
  /**
   * Return MCS 10 from HE MCS values.
   *
   * \return MCS 10 from HE MCS values
   */
  static WifiMode GetHeMcs10 (void);
  /**
   * Return MCS 11 from HE MCS values.
   *
   * \return MCS 11 from HE MCS values
   */
  static WifiMode GetHeMcs11 (void);

  /**
   * Return the coding rate corresponding to
   * the supplied HE MCS index. This function is used
   * as a callback for WifiMode operation.
   *
   * \param mcsValue the MCS index
   * \return the coding rate.
   */
  static WifiCodeRate GetCodeRate (uint8_t mcsValue);
  /**
   * Return the constellation size corresponding
   * to the supplied HE MCS index. This function is used
   * as a callback for WifiMode operation.
   *
   * \param mcsValue the MCS index
   * \return the size of modulation constellation.
   */
  static uint16_t GetConstellationSize (uint8_t mcsValue);
  /**
   * Return the PHY rate corresponding to the supplied HE MCS
   * index, channel width, guard interval, and number of
   * spatial stream. This function calls HtPhy::CalculatePhyRate
   * and is mainly used as a callback for WifiMode operation.
   *
   * \param mcsValue the HE MCS index
   * \param channelWidth the considered channel width in MHz
   * \param guardInterval the considered guard interval duration in nanoseconds
   * \param nss the considered number of stream
   *
   * \return the physical bit rate of this signal in bps.
   */
  static uint64_t GetPhyRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss);
  /**
   * Return the PHY rate corresponding to
   * the supplied TXVECTOR for the STA-ID.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the station ID for MU (unused if SU)
   * \return the physical bit rate of this signal in bps.
   */
  static uint64_t GetPhyRateFromTxVector (const WifiTxVector& txVector, uint16_t staId = SU_STA_ID);
  /**
   * Return the data rate corresponding to
   * the supplied TXVECTOR for the STA-ID.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the station ID for MU (unused if SU)
   * \return the data bit rate in bps.
   */
  static uint64_t GetDataRateFromTxVector (const WifiTxVector& txVector, uint16_t staId = SU_STA_ID);
  /**
   * Return the data rate corresponding to
   * the supplied HE MCS index, channel width,
   * guard interval, and number of spatial
   * streams.
   *
   * \param mcsValue the MCS index
   * \param channelWidth the channel width in MHz
   * \param guardInterval the guard interval duration in nanoseconds
   * \param nss the number of spatial streams
   * \return the data bit rate in bps.
   */
  static uint64_t GetDataRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss);
  /**
   * Calculate the rate in bps of the non-HT Reference Rate corresponding
   * to the supplied HE MCS index. This function calls CalculateNonHtReferenceRate
   * and is used as a callback for WifiMode operation.
   *
   * \param mcsValue the HE MCS index
   * \return the rate in bps of the non-HT Reference Rate.
   */
  static uint64_t GetNonHtReferenceRate (uint8_t mcsValue);
  /**
   * Check whether the combination in TXVECTOR is allowed.
   * This function is used as a callback for WifiMode operation.
   *
   * \param txVector the TXVECTOR
   * \returns true if this combination is allowed, false otherwise.
   */
  static bool IsAllowed (const WifiTxVector& txVector);

protected:
  PhyFieldRxStatus ProcessSigA (Ptr<Event> event, PhyFieldRxStatus status) override;
  PhyFieldRxStatus ProcessSigB (Ptr<Event> event, PhyFieldRxStatus status) override;
  Ptr<Event> DoGetEvent (Ptr<const WifiPpdu> ppdu, RxPowerWattPerChannelBand& rxPowersW) override;
  bool IsConfigSupported (Ptr<const WifiPpdu> ppdu) const override;
  void DoStartReceivePayload (Ptr<Event> event) override;
  std::pair<uint16_t, WifiSpectrumBand> GetChannelWidthAndBand (const WifiTxVector& txVector, uint16_t staId) const override;
  void DoEndReceivePayload (Ptr<const WifiPpdu> ppdu) override;
  void DoResetReceive (Ptr<Event> event) override;
  void DoAbortCurrentReception (WifiPhyRxfailureReason reason) override;
  uint64_t ObtainNextUid (const WifiTxVector& txVector) override;
  Ptr<SpectrumValue> GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const override;
  uint32_t GetMaxPsduSize (void) const override;
  WifiConstPsduMap GetWifiConstPsduMap (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector) const override;

  /**
   * Start receiving the PSDU (i.e. the first symbol of the PSDU has arrived) of an UL-OFDMA transmission.
   * This function is called upon the RX event corresponding to the OFDMA part of the UL MU PPDU.
   *
   * \param event the event holding incoming OFDMA part of the PPDU's information
   */
  void StartReceiveOfdmaPayload (Ptr<Event> event);

  /**
   * Return the rate (in bps) of the non-HT Reference Rate
   * which corresponds to the supplied code rate and
   * constellation size.
   *
   * \param codeRate the convolutional coding rate
   * \param constellationSize the size of modulation constellation
   * \returns the rate in bps.
   *
   * To convert an HE MCS to its corresponding non-HT Reference Rate
   * use the modulation and coding rate of the HT MCS
   * and lookup in Table 10-10 of IEEE P802.11ax/D6.0.
   */
  static uint64_t CalculateNonHtReferenceRate (WifiCodeRate codeRate, uint16_t constellationSize);
  /**
   * \param channelWidth the channel width in MHz
   * \return he number of usable subcarriers for data
   */
  static uint16_t GetUsableSubcarriers (uint16_t channelWidth);

  uint64_t m_previouslyTxPpduUid;  //!< UID of the previously sent PPDU, used by AP to recognize response HE TB PPDUs
  uint64_t m_currentHeTbPpduUid;   //!< UID of the HE TB PPDU being received

  std::map <uint16_t /* STA-ID */, EventId> m_beginOfdmaPayloadRxEvents; //!< the beginning of the OFDMA payload reception events (indexed by STA-ID)

  EndOfHeSigACallback m_endOfHeSigACallback; //!< end of HE-SIG-A callback
  WifiTxVector m_trigVector;                 //!< the TRIGVECTOR
  Time m_trigVectorExpirationTime;           //!< expiration time of the TRIGVECTOR

private:
  void BuildModeList (void) override;
  uint8_t GetNumberBccEncoders (const WifiTxVector& txVector) const override;
  Time GetSymbolDuration (const WifiTxVector& txVector) const override;

  /**
   * Create and return the HE MCS corresponding to
   * the provided index.
   * This method binds all the callbacks used by WifiMode.
   *
   * \param index the index of the MCS
   * \return an HE MCS
   */
  static WifiMode CreateHeMcs (uint8_t index);

  /**
   * Given a PPDU duration value, the TXVECTOR used to transmit the PPDU and
   * the PHY band, compute a valid PPDU duration considering the number and
   * duration of symbols, the preamble duration and the guard interval.
   *
   * \param ppduDuration the given PPDU duration
   * \param txVector the given TXVECTOR
   * \param band the PHY band
   * \return a valid PPDU duration
   */
  static Time GetValidPpduDuration (Time ppduDuration, const WifiTxVector& txVector,
                                    WifiPhyBand band);

  static const PpduFormats m_hePpduFormats; //!< HE PPDU formats
}; //class HePhy

} //namespace ns3

#endif /* HE_PHY_H */
