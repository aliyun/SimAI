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

#ifndef DSSS_PHY_H
#define DSSS_PHY_H

#include "ns3/phy-entity.h"
#include <vector>

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::DsssPhy class.
 */

namespace ns3 {

/**
 * \brief PHY entity for HR/DSSS (11b)
 * \ingroup wifi
 *
 * Refer to IEEE 802.11-2016, clause 16 (HR/DSSS).
 * Note that DSSS rates (clause 15) are a subset
 * of HR/DSSS rates.
 */
class DsssPhy : public PhyEntity
{
public:
  /**
   * Constructor for HR/DSSS PHY
   */
  DsssPhy ();
  /**
   * Destructor for HR/DSSS PHY
   */
  virtual ~DsssPhy ();

  WifiMode GetSigMode (WifiPpduField field, const WifiTxVector& txVector) const override;
  const PpduFormats & GetPpduFormats (void) const override;
  Time GetDuration (WifiPpduField field, const WifiTxVector& txVector) const override;
  Time GetPayloadDuration (uint32_t size, const WifiTxVector& txVector, WifiPhyBand band, MpduType mpdutype,
                           bool incFlag, uint32_t &totalAmpduSize, double &totalAmpduNumSymbols,
                           uint16_t staId) const override;
  Ptr<WifiPpdu> BuildPpdu (const WifiConstPsduMap & psdus, const WifiTxVector& txVector, Time ppduDuration) override;
  uint32_t GetMaxPsduSize (void) const override;

  /**
   * Initialize all HR/DSSS modes.
   */
  static void InitializeModes (void);
  /**
   * Return a WifiMode for HR/DSSS
   * corresponding to the provided rate.
   *
   * \param rate the rate in bps
   * \return a WifiMode for HR/DSSS
   */
  static WifiMode GetDsssRate (uint64_t rate);
  /**
   * Return a WifiMode for DSSS at 1 Mbps.
   *
   * \return a WifiMode for DSSS at 1 Mbps
   */
  static WifiMode GetDsssRate1Mbps (void);
  /**
   * Return a WifiMode for DSSS at 2 Mbps.
   *
   * \return a WifiMode for DSSS at 2 Mbps
   */
  static WifiMode GetDsssRate2Mbps (void);
  /**
   * Return a WifiMode for HR/DSSS at 5.5 Mbps.
   *
   * \return a WifiMode for HR/DSSS at 5.5 Mbps
   */
  static WifiMode GetDsssRate5_5Mbps (void);
  /**
   * Return a WifiMode for HR/DSSS at 11 Mbps.
   *
   * \return a WifiMode for HR/DSSS at 11 Mbps
   */
  static WifiMode GetDsssRate11Mbps (void);

  /**
   * Return the WifiCodeRate from the DSSS or HR/DSSS mode's unique
   * name using ModulationLookupTable. This is mainly used as a
   * callback for WifiMode operation.
   *
   * \param name the unique name of the DSSS or HR/DSSS mode
   * \return WifiCodeRate corresponding to the unique name
   */
  static WifiCodeRate GetCodeRate (const std::string& name);
  /**
   * Return the constellation size from the DSSS or HR/DSSS mode's
   * unique name using ModulationLookupTable. This is mainly used
   * as a callback for WifiMode operation.
   *
   * \param name the unique name of the DSSS or HR/DSSS mode
   * \return constellation size corresponding to the unique name
   */
  static uint16_t GetConstellationSize (const std::string& name);
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
   * Return the data rate from the DSSS or HR/DSSS mode's unique name and
   * the supplied parameters. This function is mainly used as a callback
   * for WifiMode operation.
   *
   * \param name the unique name of the DSSS or HR/DSSS mode
   * \param modClass the modulation class, must be either WIFI_MOD_CLASS_DSSS or WIFI_MOD_CLASS_HR_DSSS
   *
   * \return the data bit rate of this signal in bps.
   */
  static uint64_t GetDataRate (const std::string& name, WifiModulationClass modClass);
  /**
   * Check whether the combination in TXVECTOR is allowed.
   * This function is used as a callback for WifiMode operation.
   *
   * \param txVector the TXVECTOR
   * \returns true if this combination is allowed, false otherwise.
   */
  static bool IsAllowed (const WifiTxVector& txVector);

private:
  PhyFieldRxStatus DoEndReceiveField (WifiPpduField field, Ptr<Event> event) override;
  Ptr<SpectrumValue> GetTxPowerSpectralDensity (double txPowerW, Ptr<const WifiPpdu> ppdu) const override;
  uint16_t GetRxChannelWidth (const WifiTxVector& txVector) const override;

  /**
   * \param txVector the transmission parameters
   * \return the WifiMode used for the PHY header field
   */
  WifiMode GetHeaderMode (const WifiTxVector& txVector) const;

  /**
   * \param txVector the transmission parameters
   * \return the duration of the PHY preamble field
   *
   * \see WIFI_PPDU_FIELD_PREAMBLE
   */
  Time GetPreambleDuration (const WifiTxVector& txVector) const;
  /**
   * \param txVector the transmission parameters
   * \return the duration of the PHY header field
   */
  Time GetHeaderDuration (const WifiTxVector& txVector) const;

  /**
   * End receiving the header, perform DSSS-specific actions, and
   * provide the status of the reception.
   *
   * \param event the event holding incoming PPDU's information
   * \return status of the reception of the header
   */
  PhyFieldRxStatus EndReceiveHeader (Ptr<Event> event);

  /**
   * Create a DSSS or HR/DSSS mode from a unique name, the unique name
   * must already be contained inside ModulationLookupTable.
   * This method binds all the callbacks used by WifiMode.
   *
   * \param uniqueName the unique name of the WifiMode
   * \param modClass the modulation class of the WifiMode, must be either WIFI_MOD_CLASS_DSSS or WIFI_MOD_CLASS_HR_DSSS
   * \return the DSSS or HR/DSSS WifiMode
   */
  static WifiMode CreateDsssMode (std::string uniqueName, WifiModulationClass modClass);

  static const PpduFormats m_dsssPpduFormats; //!< DSSS and HR/DSSS PPDU formats

  static const ModulationLookupTable m_dsssModulationLookupTable; //!< lookup table to retrieve code rate and constellation size corresponding to a unique name of modulation
}; //class DsssPhy

} //namespace ns3

#endif /* DSSS_PHY_H */
