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
 *          Mathieu Lacage <mathieu.lacage@sophia.inria.fr> (for logic ported from wifi-phy)
 */

#ifndef OFDM_PHY_H
#define OFDM_PHY_H

#include "ns3/phy-entity.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::OfdmPhy class
 * and ns3::OfdmPhyVariant enum.
 */

namespace ns3 {

/**
 * \ingroup wifi
 * The OFDM (11a) PHY variants.
 *
 * \see OfdmPhy
 */
enum OfdmPhyVariant
{
  OFDM_PHY_DEFAULT,
  OFDM_PHY_10_MHZ,
  OFDM_PHY_5_MHZ
};

/**
 * \brief PHY entity for OFDM (11a)
 * \ingroup wifi
 *
 * This class is also used for the 10 MHz and 5 MHz bandwidth
 * variants addressing vehicular communications (default is 20 MHz
 * bandwidth).
 *
 * Refer to IEEE 802.11-2016, clause 17.
 */
class OfdmPhy : public PhyEntity
{
public:
  /**
   * Constructor for OFDM PHY
   *
   * \param variant the OFDM PHY variant
   * \param buildModeList flag used to add OFDM modes to list (disabled
   *                      by child classes to only add child classes' modes)
   */
  OfdmPhy (OfdmPhyVariant variant = OFDM_PHY_DEFAULT, bool buildModeList = true);
  /**
   * Destructor for OFDM PHY
   */
  virtual ~OfdmPhy ();

  WifiMode GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const override;
  const PpduFormats & GetPpduFormats (void) const override;
  Time GetDuration (WifiPpduField field, const WifiTxVector& txVector) const override;
  Time GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype,
                           bool incFlag, uint32_t &totalAmpduSize, double &totalAmpduNumSymbols,
                           uint16_t staId) const override;
  Ptr<WifiPpdu> BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration) override;

  /**
   * Initialize all OFDM modes (for all variants).
   */
  static void InitializeModes (void);
  /**
   * Return a WifiMode for OFDM
   * corresponding to the provided rate and
   * the channel bandwidth (20, 10, or 5 MHz).
   *
   * \param rate the rate in bps
   * \param bw the bandwidth in MHz
   * \return a WifiMode for OFDM
   */
  static WifiMode GetOfdmRate (uint64_t rate, uint16_t bw = 20);
  /**
   * Return a WifiMode for OFDM at 6 Mbps.
   *
   * \return a WifiMode for OFDM at 6 Mbps
   */
  static WifiMode GetOfdmRate6Mbps (void);
  /**
   * Return a WifiMode for OFDM at 9 Mbps.
   *
   * \return a WifiMode for OFDM at 9 Mbps
   */
  static WifiMode GetOfdmRate9Mbps (void);
  /**
   * Return a WifiMode for OFDM at 12Mbps.
   *
   * \return a WifiMode for OFDM at 12 Mbps
   */
  static WifiMode GetOfdmRate12Mbps (void);
  /**
   * Return a WifiMode for OFDM at 18 Mbps.
   *
   * \return a WifiMode for OFDM at 18 Mbps
   */
  static WifiMode GetOfdmRate18Mbps (void);
  /**
   * Return a WifiMode for OFDM at 24 Mbps.
   *
   * \return a WifiMode for OFDM at 24 Mbps
   */
  static WifiMode GetOfdmRate24Mbps (void);
  /**
   * Return a WifiMode for OFDM at 36 Mbps.
   *
   * \return a WifiMode for OFDM at 36 Mbps
   */
  static WifiMode GetOfdmRate36Mbps (void);
  /**
   * Return a WifiMode for OFDM at 48 Mbps.
   *
   * \return a WifiMode for OFDM at 48 Mbps
   */
  static WifiMode GetOfdmRate48Mbps (void);
  /**
   * Return a WifiMode for OFDM at 54 Mbps.
   *
   * \return a WifiMode for OFDM at 54 Mbps
   */
  static WifiMode GetOfdmRate54Mbps (void);
  /**
   * Return a WifiMode for OFDM at 3 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 3 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate3MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 4.5 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 4.5 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate4_5MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 6 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 6 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate6MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 9 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 9 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate9MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 12 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 12 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate12MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 18 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 18 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate18MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 24 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 24 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate24MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 27 Mbps with 10 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 27 Mbps with 10 MHz channel spacing
   */
  static WifiMode GetOfdmRate27MbpsBW10MHz (void);
  /**
   * Return a WifiMode for OFDM at 1.5 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 1.5 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate1_5MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 2.25 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 2.25 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate2_25MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 3 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 3 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate3MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 4.5 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 4.5 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate4_5MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 6 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 6 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate6MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 9 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 9 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate9MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 12 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 12 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate12MbpsBW5MHz (void);
  /**
   * Return a WifiMode for OFDM at 13.5 Mbps with 5 MHz channel spacing.
   *
   * \return a WifiMode for OFDM at 13.5 Mbps with 5 MHz channel spacing
   */
  static WifiMode GetOfdmRate13_5MbpsBW5MHz (void);

  /**
   * Return the WifiCodeRate from the OFDM mode's unique name using
   * ModulationLookupTable. This is mainly used as a callback for
   * WifiMode operation.
   *
   * \param name the unique name of the OFDM mode
   * \return WifiCodeRate corresponding to the unique name
   */
  static WifiCodeRate GetCodeRate (const std::string& name);
  /**
   * Return the constellation size from the OFDM mode's unique name using
   * ModulationLookupTable. This is mainly used as a callback for
   * WifiMode operation.
   *
   * \param name the unique name of the OFDM mode
   * \return constellation size corresponding to the unique name
   */
  static uint16_t GetConstellationSize (const std::string& name);
  /**
   * Return the PHY rate from the OFDM mode's unique name and
   * the supplied parameters. This function calls CalculatePhyRate
   * and is mainly used as a callback for WifiMode operation.
   *
   * \param name the unique name of the OFDM mode
   * \param channelWidth the considered channel width in MHz
   *
   * \return the physical bit rate of this signal in bps.
   */
  static uint64_t GetPhyRate (const std::string& name, uint16_t channelWidth);

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
   * Return the data rate from the OFDM mode's unique name and
   * the supplied parameters. This function calls CalculateDataRate and
   * is mainly used as a callback for WifiMode operation.
   *
   * \param name the unique name of the OFDM mode
   * \param channelWidth the considered channel width in MHz
   *
   * \return the data bit rate of this signal in bps.
   */
  static uint64_t GetDataRate (const std::string& name, uint16_t channelWidth);
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
  Ptr<SpectrumValue> GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const override;
  uint32_t GetMaxPsduSize (void) const override;

  /**
   * \param txVector the transmission parameters
   * \return the WifiMode used for the SIGNAL field
   */
  virtual WifiMode GetHeaderMode (const WifiTxVector& txVector) const;

  /**
   * \param txVector the transmission parameters
   * \return the duration of the preamble field
   *
   * \see WIFI_PPDU_FIELD_PREAMBLE
   */
  virtual Time GetPreambleDuration (const WifiTxVector& txVector) const;
  /**
   * \param txVector the transmission parameters
   * \return the duration of the SIGNAL field
   */
  virtual Time GetHeaderDuration (const WifiTxVector& txVector) const;

  /**
   * \return the number of service bits
   */
  uint8_t GetNumberServiceBits (void) const;
  /**
   * \param band the frequency band being used
   * \return the signal extension duration
   */
  Time GetSignalExtension (WifiPhyBand band) const;

  /**
   * End receiving the header, perform OFDM-specific actions, and
   * provide the status of the reception.
   *
   * \param event the event holding incoming PPDU's information
   * \return status of the reception of the header
   */
  PhyFieldRxStatus EndReceiveHeader (Ptr<Event> event);

  /**
   * Checks if the PPDU's bandwidth is supported by the PHY.
   *
   * \param ppdu the received PPDU
   * \return \c true if supported, \c false otherwise
   */
  virtual bool IsChannelWidthSupported (Ptr<const WifiPpdu> ppdu) const;
  /**
   * Checks if the signaled configuration (including bandwidth)
   * is supported by the PHY.
   *
   * \param field the current PPDU field (SIG used for checking config)
   * \param ppdu the received PPDU
   * \return \c true if supported, \c false otherwise
   */
  virtual bool IsAllConfigSupported (WifiPpduField field, Ptr<const WifiPpdu> ppdu) const;

  /**
   * Calculate the PHY rate in bps from code rate and data rate.
   *
   * \param codeRate the WifiCodeRate
   * \param dataRate the data rate in bps
   * \return the physical rate in bps from WifiCodeRate and data rate.
   */
  static uint64_t CalculatePhyRate (WifiCodeRate codeRate, uint64_t dataRate);
  /**
   * Convert WifiCodeRate to a ratio, e.g., code ratio of WIFI_CODE_RATE_1_2 is 0.5.
   *
   * \param codeRate the WifiCodeRate
   * \return the ratio form of WifiCodeRate.
   */
  static double GetCodeRatio (WifiCodeRate codeRate);
  /**
   * Calculates data rate from the supplied parameters.
   *
   * \param codeRate the code rate of the mode
   * \param constellationSize the size of modulation constellation
   * \param channelWidth the considered channel width in MHz
   *
   * \return the data bit rate of this signal in bps.
   */
  static uint64_t CalculateDataRate (WifiCodeRate codeRate, uint16_t constellationSize, uint16_t channelWidth);
  /**
   * Calculates data rate from the supplied parameters.
   *
   * \param symbolDuration the symbol duration (in us) excluding guard interval
   * \param guardInterval the considered guard interval duration in nanoseconds
   * \param usableSubCarriers the number of usable subcarriers for data
   * \param numberOfBitsPerSubcarrier the number of data bits per subcarrier
   * \param codingRate the coding rate
   *
   * \return the data bit rate of this signal in bps.
   */
  static uint64_t CalculateDataRate (double symbolDuration, uint16_t guardInterval,
                                     uint16_t usableSubCarriers, uint16_t numberOfBitsPerSubcarrier,
                                     double codingRate);

private:
  /**
   * Create an OFDM mode from a unique name, the unique name
   * must already be contained inside ModulationLookupTable.
   * This method binds all the callbacks used by WifiMode.
   *
   * \param uniqueName the unique name of the WifiMode
   * \param isMandatory whether the WifiMode is mandatory
   * \return the OFDM WifiMode
   */
  static WifiMode CreateOfdmMode (std::string uniqueName, bool isMandatory);

  static const PpduFormats m_ofdmPpduFormats; //!< OFDM PPDU formats

  static const ModulationLookupTable m_ofdmModulationLookupTable; //!< lookup table to retrieve code rate and constellation size corresponding to a unique name of modulation
}; //class OfdmPhy

} //namespace ns3

#endif /* OFDM_PHY_H */
