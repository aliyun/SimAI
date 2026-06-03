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
 *          Sébastien Deronne <sebastien.deronne@gmail.com> (for logic ported from wifi-phy)
 */

#ifndef HT_PHY_H
#define HT_PHY_H

#include "ns3/ofdm-phy.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::HtPhy class.
 */

namespace ns3 {

/**
 * This defines the BSS membership value for HT PHY.
 */
#define HT_PHY 127

/**
 * \brief PHY entity for HT (11n)
 * \ingroup wifi
 *
 * HT PHY is based on OFDM PHY.
 * Only HT-Mixed is supported (support for HT-Greenfield has been removed).
 * Only HT MCSs up to 31 are supported.
 *
 * Refer to IEEE 802.11-2016, clause 19.
 */
class HtPhy : public OfdmPhy
{
public:
  /**
   * Constructor for HT PHY
   *
   * \param maxNss the maximum number of spatial streams
   * \param buildModeList flag used to add HT modes to list (disabled
   *                      by child classes to only add child classes' modes)
   */
  HtPhy (uint8_t maxNss = 1, bool buildModeList = true);
  /**
   * Destructor for HT PHY
   */
  virtual ~HtPhy ();

  WifiMode GetMcs (uint8_t index) const override;
  bool IsMcsSupported (uint8_t index) const override;
  bool HandlesMcsModes (void) const override;
  WifiMode GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const override;
  const PpduFormats & GetPpduFormats (void) const override;
  Time GetDuration (WifiPpduField field, const WifiTxVector& txVector) const override;
  Time GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype,
                           bool incFlag, uint32_t &totalAmpduSize, double &totalAmpduNumSymbols,
                           uint16_t staId) const override;
  Ptr<WifiPpdu> BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration) override;

  /**
   * \return the WifiMode used for the L-SIG (non-HT header) field
   */
  static WifiMode GetLSigMode (void);
  /**
   * \return the WifiMode used for the HT-SIG field
   */
  virtual WifiMode GetHtSigMode (void) const;

  /**
   * \return the BSS membership selector for this PHY entity
   */
  uint8_t GetBssMembershipSelector (void) const;

  /**
   * Set the maximum supported MCS index __per spatial stream__.
   * For HT, this results in non-continuous indices for supported MCSs.
   *
   * \return the maximum MCS index per spatial stream supported by this entity
   */
  uint8_t GetMaxSupportedMcsIndexPerSs (void) const;
  /**
   * Set the maximum supported MCS index __per spatial stream__.
   * For HT, this results in non-continuous indices for supported MCSs.
   *
   * \param maxIndex the maximum MCS index per spatial stream supported by this entity
   *
   * The provided value should not be greater than maximum standard-defined value.
   */
  void SetMaxSupportedMcsIndexPerSs (uint8_t maxIndex);
  /**
   * Configure the maximum number of spatial streams supported
   * by this HT PHY.
   *
   * \param maxNss the maximum number of spatial streams
   */
  void SetMaxSupportedNss (uint8_t maxNss);

  /**
   * \param preamble the type of preamble
   * \return the duration of the L-SIG (non-HT header) field
   *
   * \see WIFI_PPDU_FIELD_NON_HT_HEADER
   */
  virtual Time GetLSigDuration (WifiPreamble preamble) const;
  /**
   * \param txVector the transmission parameters
   * \param nDataLtf the number of data LTF fields (excluding those in preamble)
   * \param nExtensionLtf the number of extension LTF fields
   * \return the duration of the training field
   *
   * \see WIFI_PPDU_FIELD_TRAINING
   */
  virtual Time GetTrainingDuration (const WifiTxVector& txVector,
                                    uint8_t nDataLtf, uint8_t nExtensionLtf = 0) const;
  /**
   * \return the duration of the HT-SIG field
   */
  virtual Time GetHtSigDuration (void) const;

  /**
   * Initialize all HT modes.
   */
  static void InitializeModes (void);
  /**
   * Return the HT MCS corresponding to
   * the provided index.
   *
   * \param index the index of the MCS
   * \return an HT MCS
   */
  static WifiMode GetHtMcs (uint8_t index);

  /**
   * Return MCS 0 from HT MCS values.
   *
   * \return MCS 0 from HT MCS values
   */
  static WifiMode GetHtMcs0 (void);
  /**
   * Return MCS 1 from HT MCS values.
   *
   * \return MCS 1 from HT MCS values
   */
  static WifiMode GetHtMcs1 (void);
  /**
   * Return MCS 2 from HT MCS values.
   *
   * \return MCS 2 from HT MCS values
   */
  static WifiMode GetHtMcs2 (void);
  /**
   * Return MCS 3 from HT MCS values.
   *
   * \return MCS 3 from HT MCS values
   */
  static WifiMode GetHtMcs3 (void);
  /**
   * Return MCS 4 from HT MCS values.
   *
   * \return MCS 4 from HT MCS values
   */
  static WifiMode GetHtMcs4 (void);
  /**
   * Return MCS 5 from HT MCS values.
   *
   * \return MCS 5 from HT MCS values
   */
  static WifiMode GetHtMcs5 (void);
  /**
   * Return MCS 6 from HT MCS values.
   *
   * \return MCS 6 from HT MCS values
   */
  static WifiMode GetHtMcs6 (void);
  /**
   * Return MCS 7 from HT MCS values.
   *
   * \return MCS 7 from HT MCS values
   */
  static WifiMode GetHtMcs7 (void);
  /**
   * Return MCS 8 from HT MCS values.
   *
   * \return MCS 8 from HT MCS values
   */
  static WifiMode GetHtMcs8 (void);
  /**
   * Return MCS 9 from HT MCS values.
   *
   * \return MCS 9 from HT MCS values
   */
  static WifiMode GetHtMcs9 (void);
  /**
   * Return MCS 10 from HT MCS values.
   *
   * \return MCS 10 from HT MCS values
   */
  static WifiMode GetHtMcs10 (void);
  /**
   * Return MCS 11 from HT MCS values.
   *
   * \return MCS 11 from HT MCS values
   */
  static WifiMode GetHtMcs11 (void);
  /**
   * Return MCS 12 from HT MCS values.
   *
   * \return MCS 12 from HT MCS values
   */
  static WifiMode GetHtMcs12 (void);
  /**
   * Return MCS 13 from HT MCS values.
   *
   * \return MCS 13 from HT MCS values
   */
  static WifiMode GetHtMcs13 (void);
  /**
   * Return MCS 14 from HT MCS values.
   *
   * \return MCS 14 from HT MCS values
   */
  static WifiMode GetHtMcs14 (void);
  /**
   * Return MCS 15 from HT MCS values.
   *
   * \return MCS 15 from HT MCS values
   */
  static WifiMode GetHtMcs15 (void);
  /**
   * Return MCS 16 from HT MCS values.
   *
   * \return MCS 16 from HT MCS values
   */
  static WifiMode GetHtMcs16 (void);
  /**
   * Return MCS 17 from HT MCS values.
   *
   * \return MCS 17 from HT MCS values
   */
  static WifiMode GetHtMcs17 (void);
  /**
   * Return MCS 18 from HT MCS values.
   *
   * \return MCS 18 from HT MCS values
   */
  static WifiMode GetHtMcs18 (void);
  /**
   * Return MCS 19 from HT MCS values.
   *
   * \return MCS 19 from HT MCS values
   */
  static WifiMode GetHtMcs19 (void);
  /**
   * Return MCS 20 from HT MCS values.
   *
   * \return MCS 20 from HT MCS values
   */
  static WifiMode GetHtMcs20 (void);
  /**
   * Return MCS 21 from HT MCS values.
   *
   * \return MCS 21 from HT MCS values
   */
  static WifiMode GetHtMcs21 (void);
  /**
   * Return MCS 22 from HT MCS values.
   *
   * \return MCS 22 from HT MCS values
   */
  static WifiMode GetHtMcs22 (void);
  /**
   * Return MCS 23 from HT MCS values.
   *
   * \return MCS 23 from HT MCS values
   */
  static WifiMode GetHtMcs23 (void);
  /**
   * Return MCS 24 from HT MCS values.
   *
   * \return MCS 24 from HT MCS values
   */
  static WifiMode GetHtMcs24 (void);
  /**
   * Return MCS 25 from HT MCS values.
   *
   * \return MCS 25 from HT MCS values
   */
  static WifiMode GetHtMcs25 (void);
  /**
   * Return MCS 26 from HT MCS values.
   *
   * \return MCS 26 from HT MCS values
   */
  static WifiMode GetHtMcs26 (void);
  /**
   * Return MCS 27 from HT MCS values.
   *
   * \return MCS 27 from HT MCS values
   */
  static WifiMode GetHtMcs27 (void);
  /**
   * Return MCS 28 from HT MCS values.
   *
   * \return MCS 28 from HT MCS values
   */
  static WifiMode GetHtMcs28 (void);
  /**
   * Return MCS 29 from HT MCS values.
   *
   * \return MCS 29 from HT MCS values
   */
  static WifiMode GetHtMcs29 (void);
  /**
   * Return MCS 30 from HT MCS values.
   *
   * \return MCS 30 from HT MCS values
   */
  static WifiMode GetHtMcs30 (void);
  /**
   * Return MCS 31 from HT MCS values.
   *
   * \return MCS 31 from HT MCS values
   */
  static WifiMode GetHtMcs31 (void);

  /**
   * Return the coding rate corresponding to
   * the supplied HT MCS index. This function calls
   * GetCodeRate and is used as a callback for
   * WifiMode operation.
   *
   * \param mcsValue the MCS index
   * \return the coding rate.
   */
  static WifiCodeRate GetHtCodeRate (uint8_t mcsValue);
  /**
   * Return the coding rate corresponding to
   * the supplied HT MCS index between 0 and 7,
   * since HT MCS index > 8 is used for higher NSS.
   * This function is reused by child classes.
   *
   * \param mcsValue the MCS index
   * \return the coding rate.
   */
  static WifiCodeRate GetCodeRate (uint8_t mcsValue);
  /**
   * Return the constellation size corresponding
   * to the supplied HT MCS index.
   *
   * \param mcsValue the MCS index
   * \return the size of modulation constellation.
   */
  static uint16_t GetHtConstellationSize (uint8_t mcsValue);
  /**
   * Return the constellation size corresponding
   * to the supplied HT MCS index between 0 and 7,
   * since HT MCS index > 8 is used for higher NSS.
   * This function is reused by child classes.
   *
   * \param mcsValue the MCS index
   * \return the size of modulation constellation.
   */
  static uint16_t GetConstellationSize (uint8_t mcsValue);
  /**
   * Return the PHY rate corresponding to the supplied HT MCS
   * index, channel width, guard interval, and number of
   * spatial stream. This function calls CalculatePhyRate
   * and is mainly used as a callback for WifiMode operation.
   *
   * \param mcsValue the HT MCS index
   * \param channelWidth the considered channel width in MHz
   * \param guardInterval the considered guard interval duration in nanoseconds
   * \param nss the considered number of stream
   *
   * \return the physical bit rate of this signal in bps.
   */
  static uint64_t GetPhyRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss);
  /**
   * Return the PHY rate corresponding to
   * the supplied TXVECTOR.
   * This function is mainly used as a callback
   * for WifiMode operation.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the station ID (only here to have a common signature for all callbacks)
   * \return the physical bit rate of this signal in bps.
   */
  static uint64_t GetPhyRateFromTxVector (const WifiTxVector& txVector, uint16_t staId);
  /**
   * Return the data rate corresponding to
   * the supplied TXVECTOR.
   * This function is mainly used as a callback
   * for WifiMode operation.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the station ID (only here to have a common signature for all callbacks)
   * \return the data bit rate in bps.
   */
  static uint64_t GetDataRateFromTxVector (const WifiTxVector& txVector, uint16_t staId);
  /**
   * Return the data rate corresponding to the supplied HT
   * MCS index, channel width, guard interval, and number
   * of spatial streams. This function is mainly used as a
   * callback for WifiMode operation.
   *
   * \param mcsValue the HT MCS index
   * \param channelWidth the channel width in MHz
   * \param guardInterval the guard interval duration in nanoseconds
   * \param nss the number of spatial streams
   * \return the data bit rate in bps.
   */
  static uint64_t GetDataRate (uint8_t mcsValue, uint16_t channelWidth, uint16_t guardInterval, uint8_t nss);
  /**
   * Calculate the rate in bps of the non-HT Reference Rate corresponding
   * to the supplied HT MCS index. This function calls CalculateNonHtReferenceRate
   * and is used as a callback for WifiMode operation.
   *
   * \param mcsValue the HT MCS index
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
  PhyFieldRxStatus DoEndReceiveField (WifiPpduField field, Ptr<Event> event) override;
  bool IsAllConfigSupported (WifiPpduField field, Ptr<const WifiPpdu> ppdu) const override;
  bool IsConfigSupported (Ptr<const WifiPpdu> ppdu) const override;
  Ptr<SpectrumValue> GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const override;
  uint32_t GetMaxPsduSize (void) const override;

  /**
   * Build mode list.
   * Should be redone whenever the maximum MCS index per spatial stream
   * ,or any other important parameter having an impact on the MCS index
   * (e.g. number of spatial streams for HT), changes.
   */
  virtual void BuildModeList (void);

  /**
   * \param txVector the transmission parameters
   * \return the number of BCC encoders used for data encoding
   */
  virtual uint8_t GetNumberBccEncoders (const WifiTxVector& txVector) const;
  /**
   * \param txVector the transmission parameters
   * \return the symbol duration (including GI)
   */
  virtual Time GetSymbolDuration (const WifiTxVector& txVector) const;

  /**
   * Return the PHY rate corresponding to
   * the supplied code rate and data rate.
   *
   * \param codeRate the code rate
   * \param dataRate the data rate in bps
   * \return the data bit rate in bps.
   */
  static uint64_t CalculatePhyRate (WifiCodeRate codeRate, uint64_t dataRate);
  /**
   * Return the rate (in bps) of the non-HT Reference Rate
   * which corresponds to the supplied code rate and
   * constellation size.
   *
   * \param codeRate the convolutional coding rate
   * \param constellationSize the size of modulation constellation
   * \returns the rate in bps.
   *
   * To convert an HT MCS to its corresponding non-HT Reference Rate
   * use the modulation and coding rate of the HT MCS
   * and lookup in Table 10-7 of IEEE 802.11-2016s.
   */
  static uint64_t CalculateNonHtReferenceRate (WifiCodeRate codeRate, uint16_t constellationSize);
  /**
   * Convert WifiCodeRate to a ratio, e.g., code ratio of WIFI_CODE_RATE_1_2 is 0.5.
   *
   * \param codeRate the code rate
   * \return the code rate in ratio.
   */
  static double GetCodeRatio (WifiCodeRate codeRate);
  /**
   * Calculates data rate from the supplied parameters.
   *
   * \param symbolDuration the symbol duration (in us) excluding guard interval
   * \param guardInterval the considered guard interval duration in nanoseconds
   * \param usableSubCarriers the number of usable subcarriers for data
   * \param numberOfBitsPerSubcarrier the number of data bits per subcarrier
   * \param codingRate the coding rate
   * \param nss the considered number of streams
   *
   * \return the data bit rate of this signal in bps.
   */
  static uint64_t CalculateDataRate (double symbolDuration, uint16_t guardInterval,
                                     uint16_t usableSubCarriers, uint16_t numberOfBitsPerSubcarrier,
                                     double codingRate, uint8_t nss);
  /**
   * \param channelWidth the channel width in MHz
   * \return he number of usable subcarriers for data
   */
  static uint16_t GetUsableSubcarriers (uint16_t channelWidth);

  uint8_t m_maxMcsIndexPerSs;          //!< the maximum MCS index per spatial stream as defined by the standard
  uint8_t m_maxSupportedMcsIndexPerSs; //!< the maximum supported MCS index per spatial stream
  uint8_t m_bssMembershipSelector;     //!< the BSS membership selector

private:
  /**
   * End receiving the HT-SIG, perform HT-specific actions, and
   * provide the status of the reception.
   *
   * \param event the event holding incoming PPDU's information
   * \return status of the reception of the HT-SIG
   */
  PhyFieldRxStatus EndReceiveHtSig (Ptr<Event> event);

  /**
   * Return the HT MCS corresponding to
   * the provided index.
   * This method binds all the callbacks used by WifiMode.
   *
   * \param index the index of the MCS
   * \return an HT MCS
   */
  static WifiMode CreateHtMcs (uint8_t index);

  uint8_t m_maxSupportedNss; //!< Maximum supported number of spatial streams (used to build HT MCS indices)

  static const PpduFormats m_htPpduFormats; //!< HT PPDU formats
}; //class HtPhy

} //namespace ns3

#endif /* HT_PHY_H */
