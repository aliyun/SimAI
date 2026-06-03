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

#ifndef VHT_PHY_H
#define VHT_PHY_H

#include "ns3/ht-phy.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::VhtPhy class.
 */

namespace ns3 {

/**
 * This defines the BSS membership value for VHT PHY.
 */
#define VHT_PHY 126

/**
 * \brief PHY entity for VHT (11ac)
 * \ingroup wifi
 *
 * VHT PHY is based on HT PHY.
 *
 * Refer to IEEE 802.11-2016, clause 21.
 */
class VhtPhy : public HtPhy
{
public:
  /**
   * Constructor for VHT PHY
   *
   * \param buildModeList flag used to add VHT modes to list (disabled
   *                      by child classes to only add child classes' modes)
   */
  VhtPhy (bool buildModeList = true);
  /**
   * Destructor for VHT PHY
   */
  virtual ~VhtPhy ();

  WifiMode GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const override;
  const PpduFormats & GetPpduFormats (void) const override;
  Time GetDuration (WifiPpduField field, const WifiTxVector& txVector) const override;
  Time GetLSigDuration (WifiPreamble preamble) const override;
  Time GetTrainingDuration (const WifiTxVector& txVector,
                            uint8_t nDataLtf,
                            uint8_t nExtensionLtf = 0) const override;
  Ptr<WifiPpdu> BuildPpdu (const WifiConstPsduMap & psdus,
                           const WifiTxVector& txVector,
                           Time ppduDuration) override;

  /**
   * \return the WifiMode used for the SIG-A field
   */
  virtual WifiMode GetSigAMode (void) const;
  /**
   * \param txVector the transmission parameters
   * \return the WifiMode used for the SIG-B field
   */
  virtual WifiMode GetSigBMode (const WifiTxVector& txVector) const;

  /**
   * \param preamble the type of preamble
   * \return the duration of the SIG-A field
   */
  virtual Time GetSigADuration (WifiPreamble preamble) const;
  /**
   * \param txVector the transmission parameters
   * \return the duration of the SIG-B field
   */
  virtual Time GetSigBDuration (const WifiTxVector& txVector) const;

  /**
   * Initialize all VHT modes.
   */
  static void InitializeModes (void);
  /**
   * Return the VHT MCS corresponding to
   * the provided index.
   *
   * \param index the index of the MCS
   * \return an VHT MCS
   */
  static WifiMode GetVhtMcs (uint8_t index);

  /**
   * Return MCS 0 from VHT MCS values.
   *
   * \return MCS 0 from VHT MCS values
   */
  static WifiMode GetVhtMcs0 (void);
  /**
   * Return MCS 1 from VHT MCS values.
   *
   * \return MCS 1 from VHT MCS values
   */
  static WifiMode GetVhtMcs1 (void);
  /**
   * Return MCS 2 from VHT MCS values.
   *
   * \return MCS 2 from VHT MCS values
   */
  static WifiMode GetVhtMcs2 (void);
  /**
   * Return MCS 3 from VHT MCS values.
   *
   * \return MCS 3 from VHT MCS values
   */
  static WifiMode GetVhtMcs3 (void);
  /**
   * Return MCS 4 from VHT MCS values.
   *
   * \return MCS 4 from VHT MCS values
   */
  static WifiMode GetVhtMcs4 (void);
  /**
   * Return MCS 5 from VHT MCS values.
   *
   * \return MCS 5 from VHT MCS values
   */
  static WifiMode GetVhtMcs5 (void);
  /**
   * Return MCS 6 from VHT MCS values.
   *
   * \return MCS 6 from VHT MCS values
   */
  static WifiMode GetVhtMcs6 (void);
  /**
   * Return MCS 7 from VHT MCS values.
   *
   * \return MCS 7 from VHT MCS values
   */
  static WifiMode GetVhtMcs7 (void);
  /**
   * Return MCS 8 from VHT MCS values.
   *
   * \return MCS 8 from VHT MCS values
   */
  static WifiMode GetVhtMcs8 (void);
  /**
   * Return MCS 9 from VHT MCS values.
   *
   * \return MCS 9 from VHT MCS values
   */
  static WifiMode GetVhtMcs9 (void);

  /**
   * Return the coding rate corresponding to
   * the supplied VHT MCS index. This function is
   * reused by child classes and is used as a callback
   * for WifiMode operation.
   *
   * \param mcsValue the MCS index
   * \return the coding rate.
   */
  static WifiCodeRate GetCodeRate (uint8_t mcsValue);
  /**
   * Return the constellation size corresponding
   * to the supplied VHT MCS index. This function is
   * reused by child classes and is used as a callback for
   * WifiMode operation.
   *
   * \param mcsValue the MCS index
   * \return the size of modulation constellation.
   */
  static uint16_t GetConstellationSize (uint8_t mcsValue);
  /**
   * Return the PHY rate corresponding to the supplied VHT MCS
   * index, channel width, guard interval, and number of
   * spatial stream. This function calls HtPhy::CalculatePhyRate
   * and is mainly used as a callback for WifiMode operation.
   *
   * \param mcsValue the VHT MCS index
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
   * Return the data rate corresponding to
   * the supplied VHT MCS index, channel width,
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
   * to the supplied VHT MCS index. This function calls CalculateNonHtReferenceRate
   * and is used as a callback for WifiMode operation.
   *
   * \param mcsValue the VHT MCS index
   * \return the rate in bps of the non-HT Reference Rate.
   */
  static uint64_t GetNonHtReferenceRate (uint8_t mcsValue);
  /**
   * Check whether the combination of <MCS, channel width, NSS> is allowed.
   * This function is used as a callback for WifiMode operation.
   *
   * \param mcsValue the considered MCS index
   * \param channelWidth the considered channel width in MHz
   * \param nss the considered number of streams
   * \returns true if this <MCS, channel width, NSS> combination is allowed, false otherwise.
   */
  static bool IsCombinationAllowed (uint8_t mcsValue, uint16_t channelWidth, uint8_t nss);
  /**
   * Check whether the combination in TXVECTOR is allowed.
   * This function is used as a callback for WifiMode operation.
   *
   * \param txVector the TXVECTOR
   * \returns true if this combination is allowed, false otherwise.
   */
  static bool IsAllowed (const WifiTxVector& txVector);

protected:
  WifiMode GetHtSigMode (void) const override;
  Time GetHtSigDuration (void) const override;
  uint8_t GetNumberBccEncoders (const WifiTxVector& txVector) const override;
  PhyFieldRxStatus DoEndReceiveField (WifiPpduField field, Ptr<Event> event) override;
  bool IsAllConfigSupported (WifiPpduField field, Ptr<const WifiPpdu> ppdu) const override;
  uint32_t GetMaxPsduSize (void) const override;

  /**
   * End receiving the SIG-A, perform VHT-specific actions, and
   * provide the status of the reception.
   *
   * Child classes can perform amendment-specific actions by specializing
   * \see ProcessSigA.
   *
   * \param event the event holding incoming PPDU's information
   * \return status of the reception of the SIG-A
   */
  PhyFieldRxStatus EndReceiveSigA (Ptr<Event> event);
  /**
   * End receiving the SIG-B, perform VHT-specific actions, and
   * provide the status of the reception.
   *
   * Child classes can perform amendment-specific actions by specializing
   * \see ProcessSigB.
   *
   * \param event the event holding incoming PPDU's information
   * \return status of the reception of the SIG-B
   */
  PhyFieldRxStatus EndReceiveSigB (Ptr<Event> event);

  /**
   * Process SIG-A, perform amendment-specific actions, and
   * provide an updated status of the reception.
   *
   * \param event the event holding incoming PPDU's information
   * \param status the status of the reception of the correctly received SIG-A after the configuration support check
   * \return the updated status of the reception of the SIG-A
   */
  virtual PhyFieldRxStatus ProcessSigA (Ptr<Event> event, PhyFieldRxStatus status);
  /**
   * Process SIG-B, perform amendment-specific actions, and
   * provide an updated status of the reception.
   *
   * \param event the event holding incoming PPDU's information
   * \param status the status of the reception of the correctly received SIG-B after the configuration support check
   * \return the updated status of the reception of the SIG-B
   */
  virtual PhyFieldRxStatus ProcessSigB (Ptr<Event> event, PhyFieldRxStatus status);

  /**
   * Return the rate (in bps) of the non-HT Reference Rate
   * which corresponds to the supplied code rate and
   * constellation size.
   *
   * \param codeRate the convolutional coding rate
   * \param constellationSize the size of modulation constellation
   * \returns the rate in bps.
   *
   * To convert an VHT MCS to its corresponding non-HT Reference Rate
   * use the modulation and coding rate of the HT MCS
   * and lookup in Table 10-7 of IEEE 802.11-2016.
   */
  static uint64_t CalculateNonHtReferenceRate (WifiCodeRate codeRate, uint16_t constellationSize);
  /**
   * \param channelWidth the channel width in MHz
   * \return he number of usable subcarriers for data
   */
  static uint16_t GetUsableSubcarriers (uint16_t channelWidth);

private:
  void BuildModeList (void) override;

  /**
   * Return the VHT MCS corresponding to
   * the provided index.
   * This method binds all the callbacks used by WifiMode.
   *
   * \param index the index of the MCS
   * \return a VHT MCS
   */
  static WifiMode CreateVhtMcs (uint8_t index);

  /**
   * Typedef for storing exceptions in the number of BCC encoders for VHT MCSs
   */
  typedef std::map< std::tuple<uint16_t /* channelWidth in MHz */,
                               uint8_t /* Nss */,
                               uint8_t /* MCS index */>, uint8_t /* Nes */ > NesExceptionMap;
  static const NesExceptionMap m_exceptionsMap; //!< exception map for number of BCC encoders (extracted from VHT-MCS tables)
  static const PpduFormats m_vhtPpduFormats;    //!< VHT PPDU formats
}; //class VhtPhy

} //namespace ns3

#endif /* VHT_PHY_H */
