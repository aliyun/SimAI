/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#ifndef WIFI_MODE_H
#define WIFI_MODE_H

#include "wifi-phy-common.h"
#include "ns3/attribute-helper.h"
#include "ns3/callback.h"
#include <vector>

namespace ns3 {

#define SU_STA_ID 65535

class WifiTxVector;

/**
 * \brief represent a single transmission mode
 * \ingroup wifi
 *
 * A WifiMode is implemented by a single integer which is used
 * to lookup in a global array the characteristics of the
 * associated transmission mode. It is thus extremely cheap to
 * keep a WifiMode variable around.
 *
 * \see attribute_WifiMode
 */
class WifiMode
{
public:
  /**
   * \returns true if this <MCS, channel width, NSS> combination is allowed, false otherwise.
   *
   * \param channelWidth the considered channel width in MHz
   * \param nss the considered number of streams
   */
  bool IsAllowed (uint16_t channelWidth, uint8_t nss) const;
  /**
   * \returns true if this TXVECTOR combination is allowed, false otherwise.
   *
   * \param txVector the const WifiTxVector& of the signal
   */
  bool IsAllowed (const WifiTxVector& txVector) const;
  /**
   *
   * \param channelWidth the considered channel width in MHz
   * \param guardInterval the considered guard interval duration in nanoseconds
   * \param nss the considered number of streams
   *
   * \returns the physical bit rate of this signal in bps.
   *
   * If a transmission mode uses 1/2 FEC, and if its
   * data rate is 3.25Mbps, the PHY rate is 6.5Mbps
   */
  uint64_t GetPhyRate (uint16_t channelWidth, uint16_t guardInterval, uint8_t nss) const;
  /**
   * \param txVector the const WifiTxVector& of the signal
   * \param staId the station ID for MU (unused if SU)
   *
   * \returns the physical bit rate of this signal in bps.
   *
   * If a transmission mode uses 1/2 FEC, and if its
   * data rate is 3.25Mbps, the PHY rate is 6.5Mbps
   */
  uint64_t GetPhyRate (const WifiTxVector& txVector, uint16_t staId = SU_STA_ID) const;
  /**
   * \param channelWidth the considered channel width in MHz
   *
   * \returns the physical bit rate of this non-HT signal.
  */
  uint64_t GetPhyRate (uint16_t channelWidth) const;
  /**
   *
   * \param channelWidth the considered channel width in MHz
   * \param guardInterval the considered guard interval duration in nanoseconds
   * \param nss the considered number of streams
   *
   * \returns the data bit rate of this signal in bps.
   */
  uint64_t GetDataRate (uint16_t channelWidth, uint16_t guardInterval, uint8_t nss) const;
  /**
   * \param txVector the const WifiTxVector& of the signal
   * \param staId the station ID for MU (unused if SU)
   *
   * \returns the data bit rate of this signal.
   */
  uint64_t GetDataRate (const WifiTxVector& txVector, uint16_t staId = SU_STA_ID) const;
  /**
   * \param channelWidth the considered channel width in MHz
   *
   * \returns the data bit rate of this non-HT.
  */
  uint64_t GetDataRate (uint16_t channelWidth) const;

  /**
   * \returns the coding rate of this transmission mode
   */
  WifiCodeRate GetCodeRate (void) const;
  /**
   * \returns the size of the modulation constellation.
   */
  uint16_t GetConstellationSize (void) const;
  /**
   * \returns the MCS value.
   */
  uint8_t GetMcsValue (void) const;
  /**
   * \returns a human-readable representation of this WifiMode
   * instance.
   */
  std::string GetUniqueName (void) const;
  /**
   * \returns true if this mode is a mandatory mode, false
   *          otherwise.
   */
  bool IsMandatory (void) const;
  /**
   * \returns the UID associated to this wireless mode.
   *
   * Each specific wireless mode should have a different UID.
   * For example, the 802.11b 1Mbps and the 802.11b 2Mbps modes
   * should have different UIDs.
   */
  uint32_t GetUid (void) const;
  /**
   *
   * \returns the Modulation Class (Section 9.7.8 "Modulation classes"; IEEE 802.11-2012)
   * to which this WifiMode belongs.
   */
  WifiModulationClass GetModulationClass () const;
  /**
   * \returns the rate (in bps) of the non-HT Reference Rate
   * which corresponds to the HT MCS of this WifiMode.
   *
   * To convert an HT MCS to is corresponding non-HT Reference Rate
   * use the modulation and coding rate of the HT MCS
   * and lookup in Table 9-5 of IEEE 802.11-2012.
   */
  uint64_t GetNonHtReferenceRate (void) const;
  /**
   * \param mode the WifiMode
   * \returns true if this WifiMode has a
   * a code rate strictly higher than mode.
   */
  bool IsHigherCodeRate (WifiMode mode) const;
  /**
   * \param mode the WifiMode
   * \returns true if this WifiMode has a
   * a rate strictly higher than mode.
   */
  bool IsHigherDataRate (WifiMode mode) const;

  /**
   * Create an invalid WifiMode. Calling any method on the
   * instance created will trigger an assert. This is useful
   * to separate the declaration of a WifiMode variable from
   * its initialization.
   */
  WifiMode ();
  /**
   * Create a WifiMode if the given string represents a valid
   * WifiMode name.
   *
   * \param name std::string of a valid WifiMode name
   */
  WifiMode (std::string name);


private:
  /// allow WifiModeFactory class access
  friend class WifiModeFactory;
  /**
   * Create a WifiMode from a given unique ID.
   *
   * \param uid unique ID
   */
  WifiMode (uint32_t uid);
  uint32_t m_uid; ///< UID
};

/**
 * Check if the two WifiModes are identical.
 *
 * \param a WifiMode
 * \param b WifiMode
 *
 * \return true if the two WifiModes are identical,
 *         false otherwise
 */
bool operator == (const WifiMode &a, const WifiMode &b);

/**
 * Check if the two WifiModes are different.
 *
 * \param a WifiMode
 * \param b WifiMode
 *
 * \return true if the two WifiModes are different,
 *         false otherwise
 */
bool operator != (const WifiMode &a, const WifiMode &b);

/**
 * Compare two WifiModes
 *
 * \param a WifiMode
 * \param b WifiMode
 *
 * \return true if a is less than b,
 *         false otherwise
 */
bool operator < (const WifiMode &a, const WifiMode &b);

/**
 * Serialize WifiMode to ostream (human-readable).
 *
 * \param os the output stream
 * \param mode the WifiMode
 *
 * \return std::ostream
 */
std::ostream & operator << (std::ostream & os, const WifiMode &mode);
/**
 * Serialize WifiMode from istream (human-readable).
 *
 * \param is the input stream
 * \param mode the WifiMode
 *
 * \return std::istream
 */
std::istream & operator >> (std::istream &is, WifiMode &mode);

ATTRIBUTE_HELPER_HEADER (WifiMode);

/**
 * In various parts of the code, folk are interested in maintaining a
 * list of transmission modes. The vector class provides a good basis
 * for this, but we here add some syntactic sugar by defining a
 * WifiModeList type, and a corresponding iterator.
 */
typedef std::vector<WifiMode> WifiModeList;
/**
 * An iterator for WifiModeList vector.
 */
typedef WifiModeList::const_iterator WifiModeListIterator;

/**
 * \brief create WifiMode class instances and keep track of them.
 *
 * This factory ensures that each WifiMode created has a unique name
 * and assigns to each of them a unique integer.
 */
class WifiModeFactory
{
public:
  // Typedefs for callbacks used by WifiModeItem
  /**
   * Typedef for callback used to retrieve code rate of a WifiMode
   * \return the code rate of the WifiMode.
   */
  typedef Callback<WifiCodeRate> CodeRateCallback;
  /**
   * Typedef for callback used to retrieve constellation size of a WifiMode
   * \return the size of modulation constellation of the WifiMode.
   */
  typedef Callback<uint16_t> ConstellationSizeCallback;
  /**
   * Typedef for callback used to calculate PHY rate of a WifiMode
   * from a TXVECTOR.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the station ID
   * \return the physical bit rate of the signal in bps.
   */
  typedef Callback<uint64_t, const WifiTxVector& /* txVector */, uint16_t /* staId */> PhyRateCallback;
  /**
   * Typedef for callback used to calculate data rate of a WifiMode
   * from a TXVECTOR.
   *
   * \param txVector the TXVECTOR used for the transmission
   * \param staId the station ID
   * \return the data rate of the signal in bps.
   */
  typedef Callback<uint64_t, const WifiTxVector& /* txVector */, uint16_t /* staId */> DataRateCallback;
  /**
   * Typedef for callback used to calculate Non-HT Reference Rate of
   * an MCS defined in HT or later amendment. For Non-HT modes (DSSS, OFDM,
   * etc) this should be defined as null.
   *
   * \return the rate (in bps) of the non-HT Reference Rate.
   */
  typedef Callback<uint64_t> NonHtReferenceRateCallback;
  /**
   * Typedef for callback used to check whether a given combination is allowed
   *
   * \param txVector the TXVECTOR containing the combination to check
   * \return true if combination of current WifiMode and TXVECTOR is allowed.
   */
  typedef Callback<bool, const WifiTxVector& /* txVector */> AllowedCallback;

  /**
   * \param uniqueName the name of the associated WifiMode. This name
   *        must be unique across _all_ instances.
   * \param modClass the class of modulation
   * \param isMandatory true if this WifiMode is mandatory, false otherwise.
   * \param codeRateCallback a callback function to retrieve coding rate of
   *        this WifiMode. If convolutional coding is used for this rate
   *        then the callback returns the convolutional coding rate used. If
   *        there is no explicit convolutional coding step (e.g., for DSSS
   *        rates) then the callback should returns WIFI_CODE_RATE_UNDEFINED.
   * \param constellationSizeCallback a callback function that returns the
   *        order of the constellation used.
   * \param phyRateCallback a callback function to calculate the PHY rate (in
   *        bps) of this WifiMode.
   * \param dataRateCallback a callback function to calculate the data rate
   *        (in bps) of this WifiMode.
   * \param isAllowedCallback a callback function to check whether a
   *        specific combination of this WifiMode is allowed.
   *
   * \return WifiMode
   *
   * Create a non-HT WifiMode.
   */
  static WifiMode CreateWifiMode (std::string uniqueName,
                                  WifiModulationClass modClass,
                                  bool isMandatory,
                                  CodeRateCallback codeRateCallback,
                                  ConstellationSizeCallback constellationSizeCallback,
                                  PhyRateCallback phyRateCallback,
                                  DataRateCallback dataRateCallback,
                                  AllowedCallback isAllowedCallback);

  /**
   * \param uniqueName the name of the associated WifiMode. This name
   *        must be unique across _all_ instances.
   * \param mcsValue the MCS value
   * \param modClass the class of modulation
   * \param isMandatory true if this WifiMode is mandatory, false otherwise.
   * \param codeRateCallback a callback function that returns the coding rate
   *        of this WifiMode.
   * \param constellationSizeCallback a callback function that returns the size
   *        of modulation constellation of this WifiMode.
   * \param phyRateCallback a callback function to calculate the PHY rate (in
   *        bps) of this WifiMode.
   * \param dataRateCallback a callback function to calculate the data rate (in
   *        bps) of this WifiMode.
   * \param nonHtReferenceRateCallback a callback function to calculate the rate
   *        (in bps) of the non-HT Reference Rate of this WifiMode.
   * \param isAllowedCallback a callback function to calculate whether a given
   *        combination of is allowed for this WifiMode.
   *
   * \return WifiMode
   *
   * Create a HT or later WifiMode.
   */
  static WifiMode CreateWifiMcs (std::string uniqueName,
                                 uint8_t mcsValue,
                                 WifiModulationClass modClass,
                                 bool isMandatory,
                                 CodeRateCallback codeRateCallback,
                                 ConstellationSizeCallback constellationSizeCallback,
                                 PhyRateCallback phyRateCallback,
                                 DataRateCallback dataRateCallback,
                                 NonHtReferenceRateCallback nonHtReferenceRateCallback,
                                 AllowedCallback isAllowedCallback);


private:
  /// allow WifiMode class access
  friend class WifiMode;
  friend std::istream & operator >> (std::istream &is, WifiMode &mode);

  /**
   * Return a WifiModeFactory
   *
   * \return a WifiModeFactory
   */
  static WifiModeFactory* GetFactory ();
  WifiModeFactory ();

  /**
   * This is the data associated to a unique WifiMode.
   * The integer stored in a WifiMode is in fact an index
   * in an array of WifiModeItem objects.
   */
  struct WifiModeItem
  {
    std::string uniqueUid;        ///< unique UID
    WifiModulationClass modClass; ///< modulation class
    bool isMandatory;             ///< flag to indicate whether this mode is mandatory
    uint8_t mcsValue;             ///< MCS value
    CodeRateCallback GetCodeRateCallback;                         ///< Callback to retrieve code rate of this WifiModeItem
    ConstellationSizeCallback GetConstellationSizeCallback;       ///< Callback to retrieve constellation size of this WifiModeItem
    PhyRateCallback GetPhyRateCallback;                           ///< Callback to calculate PHY rate in bps of this WifiModeItem
    DataRateCallback GetDataRateCallback;                         ///< Callback to calculate data rate in bps of this WifiModeItem
    NonHtReferenceRateCallback GetNonHtReferenceRateCallback;     ///< Callback to calculate non-HT reference rate of this WifiModeItem
    AllowedCallback IsAllowedCallback;                            ///< Callback to check whether a given combination of is allowed
  };

  /**
   * Search and return WifiMode from a given name.
   *
   * \param name human-readable WifiMode
   *
   * \return the WifiMode
   */
  WifiMode Search (std::string name) const;
  /**
   * Allocate a WifiModeItem from a given uniqueUid.
   *
   * \param uniqueUid the unique UID
   *
   * \return the allocated UID index
   */
  uint32_t AllocateUid (std::string uniqueUid);
  /**
   * Return a WifiModeItem at the given UID index.
   *
   * \param uid the UID index
   *
   * \return WifiModeItem at the given UID
   */
  WifiModeItem* Get (uint32_t uid);

  /**
   * typedef for a vector of WifiModeItem.
   */
  typedef std::vector<WifiModeItem> WifiModeItemList;
  WifiModeItemList m_itemList; ///< item list
};

} //namespace ns3

#endif /* WIFI_MODE_H */
