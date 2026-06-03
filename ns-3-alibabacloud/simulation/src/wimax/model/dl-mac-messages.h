/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008,2009 INRIA, UDcast
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
 * Authors: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *          Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 *                               <amine.ismail@UDcast.com>
 */

#ifndef DCD_CHANNEL_ENCODINGS_H
#define DCD_CHANNEL_ENCODINGS_H

#include <stdint.h>
#include "ns3/buffer.h"
#include <list>

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the DCD channel encodings as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 11.4.1 DCD channel encodings, page 659
 */
class DcdChannelEncodings
{
public:
  DcdChannelEncodings (void);
  virtual ~DcdChannelEncodings (void);

  /**
   * Set BS EIRP field
   * \param bs_eirp the BS EIRP
   */
  void SetBsEirp (uint16_t bs_eirp);
  /**
   * Set EIRX IR MAX field
   * \param rss_ir_max the EIRRX IR MAX
   */
  void SetEirxPIrMax (uint16_t rss_ir_max);
  /**
   * Set frequency field
   * \param frequency the frequency
   */
  void SetFrequency (uint32_t frequency);

  /**
   * Get BS EIRP field
   * \returns the BS EIRP
   */
  uint16_t GetBsEirp (void) const;
  /**
   * Get EIRX IR MAX field
   * \returns the EIRX IR MAX
   */
  uint16_t GetEirxPIrMax (void) const;
  /**
   * Get frequency function
   * \returns the frequency
   */
  uint32_t GetFrequency (void) const;

  /**
   * Get size field
   * \returns the size
   */
  uint16_t GetSize (void) const;

  /**
   * Write item
   * \param start the iterator
   * \returns the updated iterator
   */
  Buffer::Iterator Write (Buffer::Iterator start) const;
  /**
   * Read item
   * \param start the iterator
   * \returns the updated iterator
   */
  Buffer::Iterator Read (Buffer::Iterator start);
private:
  /**
   * Write item
   * \param start iterator
   * \returns the updated iterator
   */
  virtual Buffer::Iterator DoWrite (Buffer::Iterator start) const = 0;
  /**
   * Read item
   * \param start the iterator
   * \returns the updated iterator
   */
  virtual Buffer::Iterator DoRead (Buffer::Iterator start) = 0;

  uint16_t m_bsEirp; ///< BS EIRP
  uint16_t m_eirXPIrMax; ///< EIRX IR MAX
  uint32_t m_frequency; ///< frequency

};

} // namespace ns3

#endif /* DCD_CHANNEL_ENCODINGS_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef OFDM_DCD_CHANNEL_ENCODINGS_H
#define OFDM_DCD_CHANNEL_ENCODINGS_H

#include <stdint.h>
#include "ns3/mac48-address.h"

namespace ns3 {

/**
 * This class implements the OFDM DCD channel encodings as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 */
class OfdmDcdChannelEncodings : public DcdChannelEncodings
{
public:
  OfdmDcdChannelEncodings (void);
  ~OfdmDcdChannelEncodings (void);

  /**
   * Set channel number field
   * \param channelNr the channel number
   */
  void SetChannelNr (uint8_t channelNr);
  /**
   * Set TTG field
   * \param ttg the TTG
   */
  void SetTtg (uint8_t ttg);
  /**
   * Set RTG field
   * \param rtg the RTG
   */
  void SetRtg (uint8_t rtg);

  /**
   * Set base station ID field
   * \param baseStationId the base station ID
   */
  void SetBaseStationId (Mac48Address baseStationId);
  /**
   * Set frame duration code field
   * \param frameDurationCode the frame duration code
   */
  void SetFrameDurationCode (uint8_t frameDurationCode);
  /**
   * Set frame number field
   * \param frameNumber the frame number
   */
  void SetFrameNumber (uint32_t frameNumber);

  /**
   * Get channel number field
   * \returns the channel number
   */
  uint8_t GetChannelNr (void) const;
  /**
   * Get TTG field
   * \returns the TTG
   */
  uint8_t GetTtg (void) const;
  /**
   * Get RTG field
   * \returns the RTG
   */
  uint8_t GetRtg (void) const;

  /**
   * Get base station ID field
   * \returns the base station MAC address
   */
  Mac48Address GetBaseStationId (void) const;
  /**
   * Get frame duration code field
   * \returns the frame duration code
   */
  uint8_t GetFrameDurationCode (void) const;
  /**
   * Get frame number field
   * \returns the frame number
   */
  uint32_t GetFrameNumber (void) const;

  /**
   * Get size field
   * \returns the size
   */
  uint16_t GetSize (void) const;
private:
  /**
   * Write item
   * \param start the iterator
   * \returns the updated iterator
   */
  virtual Buffer::Iterator DoWrite (Buffer::Iterator start) const;
  /**
   * Read item
   * \param start the iterator
   * \returns the updated iterator
   */
  virtual Buffer::Iterator DoRead (Buffer::Iterator start);

  uint8_t m_channelNr; ///< channel number
  uint8_t m_ttg; ///< TTG
  uint8_t m_rtg; ///< RTG

  Mac48Address m_baseStationId; ///< base station ID
  uint8_t m_frameDurationCode; ///< frame duration code
  uint32_t m_frameNumber; ///< frame number
};

} // namespace ns3

#endif /* OFDM_DCD_CHANNEL_ENCODINGS_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef OFDM_DL_BURST_PROFILE_H
#define OFDM_DL_BURST_PROFILE_H

#include <stdint.h>
#include "ns3/buffer.h"

namespace ns3 {

/**
 * This class implements the OFDM Downlink burst profile descriptor as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 8.2.1.10 Burst profile formats page 416
 *
 */
class OfdmDlBurstProfile
{
public:
  /// DIUC enumeration
  enum Diuc
  {
    DIUC_STC_ZONE = 0,
    DIUC_BURST_PROFILE_1,
    DIUC_BURST_PROFILE_2,
    DIUC_BURST_PROFILE_3,
    DIUC_BURST_PROFILE_4,
    DIUC_BURST_PROFILE_5,
    DIUC_BURST_PROFILE_6,
    DIUC_BURST_PROFILE_7,
    DIUC_BURST_PROFILE_8,
    DIUC_BURST_PROFILE_9,
    DIUC_BURST_PROFILE_10,
    DIUC_BURST_PROFILE_11,
    // 12 is reserved
    DIUC_GAP = 13,
    DIUC_END_OF_MAP
  };

  OfdmDlBurstProfile (void);
  ~OfdmDlBurstProfile (void);

  /**
   * Set type field
   * \param type the type to set
   */
  void SetType (uint8_t type);
  /**
   * Set length field
   * \param length the length to set
   */
  void SetLength (uint8_t length);
  /**
   * Set DIUC field
   * \param diuc the DIUC
   */
  void SetDiuc (uint8_t diuc);

  /**
   * Set FEC code type
   * \param fecCodeType the FEC code type
   */
  void SetFecCodeType (uint8_t fecCodeType);

  /**
   * Get type function
   * \returns the type
   */
  uint8_t GetType (void) const;
  /** \returns the length field */
  uint8_t GetLength (void) const;
  /** \returns the DIUC field */
  uint8_t GetDiuc (void) const;

  /** \returns the FEC code type */
  uint8_t GetFecCodeType (void) const;

  /** \returns the size */
  uint16_t GetSize (void) const;

  /**
   * Write item
   * \param start the starting item iterator
   * \returns the iterator
   */
  Buffer::Iterator Write (Buffer::Iterator start) const;
  /**
   * Read item
   * \param start the starting item iterator
   * \returns the iterator
   */
  Buffer::Iterator Read (Buffer::Iterator start);
private:
  uint8_t m_type; ///< type
  uint8_t m_length; ///< length
  uint8_t m_diuc; ///< diuc

  // TLV Encoded information
  uint8_t m_fecCodeType; ///< FEC code type
};

} // namespace ns3

#endif /* OFDM_DL_BURST_PROFILE_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef DCD_H
#define DCD_H

#include <stdint.h>
#include "ns3/header.h"
#include <vector>

namespace ns3 {

/**
 * This class implements Downlink channel descriptor as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 6.3.2.3.1 Downlink Channel Descriptor (DCD) message, page 45
 */
class Dcd : public Header
{
public:
  Dcd (void);
  virtual ~Dcd (void);

  /**
   * Set configuration change count field
   * \param configurationChangeCount the configuration change count
   */
  void SetConfigurationChangeCount (uint8_t configurationChangeCount);
  /**
   * Set channel encodings field
   * \param channelEncodings the channel encodings
   */
  void SetChannelEncodings (OfdmDcdChannelEncodings channelEncodings);
  /**
   * Add DL burst profile field
   * \param dlBurstProfile the DL burst profile
   */
  void AddDlBurstProfile (OfdmDlBurstProfile dlBurstProfile);
  /**
   * Set number DL burst profile field
   * \param nrDlBurstProfiles the number of DL burst profiles
   */
  void SetNrDlBurstProfiles (uint8_t nrDlBurstProfiles);

  /**
   * Get configuration change count field
   * \returns the configuration change count
   */
  uint8_t GetConfigurationChangeCount (void) const;
  /**
   * Get channel encodings field
   * \returns the channel encodings
   */
  OfdmDcdChannelEncodings GetChannelEncodings (void) const;
  /**
   * Get DL burst profile field
   * \returns the DL burst profiles
   */
  std::vector<OfdmDlBurstProfile> GetDlBurstProfiles (void) const;
  /**
   * Get number DL burst profiles field
   * \returns the number of DL burst profiles
   */
  uint8_t GetNrDlBurstProfiles (void) const;

  /**
   * Get name field
   * \returns the name string
   */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint8_t m_reserved; ///< changed as per the amendment 802.16e-2005
  uint8_t m_configurationChangeCount; ///< configuration change count
  OfdmDcdChannelEncodings m_channelEncodings; ///< TLV Encoded information for the overall channel
  std::vector<OfdmDlBurstProfile> m_dlBurstProfiles; ///< vector of download burst profiles

  uint8_t m_nrDlBurstProfiles; ///< number DL purst profiles
};

} // namespace ns3

#endif /* DCD_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef OFDM_DL_MAP_IE_H
#define OFDM_DL_MAP_IE_H

#include <stdint.h>
#include "cid.h"

namespace ns3 {

/**
 * This class implements the OFDM DL-MAP information element as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 6.3.2.3.43.6 Compact DL-MAP IE page 109
 */
class OfdmDlMapIe
{
public:
  OfdmDlMapIe (void);
  ~OfdmDlMapIe (void);

  /**
   * Set CID function
   * \param cid the CID
   */
  void SetCid (Cid cid);
  /**
   * Set DIUC field
   * \param diuc the DIUC
   */
  void SetDiuc (uint8_t diuc);
  /**
   * Set preamble present field
   * \param preamblePresent the preamble present
   */
  void SetPreamblePresent (uint8_t preamblePresent);
  /**
   * Set start time field
   * \param startTime the start time value
   */
  void SetStartTime (uint16_t startTime);

  /**
   * Set CID field
   * \returns the CID
   */
  Cid GetCid (void) const;
  /**
   * Get DIUC field
   * \returns the DIUC
   */
  uint8_t GetDiuc (void) const;
  /**
   * Get preamble present field
   * \returns the preamble present indicator
   */
  uint8_t GetPreamblePresent (void) const;
  /**
   * Get start time field
   * \returns the start time
   */
  uint16_t GetStartTime (void) const;

  /**
   * Get size
   * \returns the size
   */
  uint16_t GetSize (void) const;

  /**
   * Wrtie item
   * \param start the iterator
   * \returns the updated iterator
   */
  Buffer::Iterator Write (Buffer::Iterator start) const;
  /**
   * Read item
   * \param start the iterator
   * \returns the updated iterator
   */
  Buffer::Iterator Read (Buffer::Iterator start);
private:
  Cid m_cid; ///< CID
  uint8_t m_diuc; ///< DIUC
  uint8_t m_preamblePresent; ///< preamble present
  uint16_t m_startTime; ///< start time

};

} // namespace ns3

#endif /* OFDM_DL_MAP_IE_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef DL_MAP_H
#define DL_MAP_H

#include <stdint.h>
#include <vector>
#include "ns3/header.h"
#include "ns3/mac48-address.h"

namespace ns3 {

/**
 * This class implements DL-MAP as described by "IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems"
 * 8.2.1.8.1 Compressed DL-MAP, page 402
 */
class DlMap : public Header
{
public:
  DlMap (void);
  virtual ~DlMap (void);

  /**
   * Set DCD count field
   * \param dcdCount the DCD count
   */
  void SetDcdCount (uint8_t dcdCount);
  /**
   * Set base station ID field
   * \param baseStationID the base station ID
   */
  void SetBaseStationId (Mac48Address baseStationID);
  /**
   * Add DL Map element field
   * \param dlMapElement the DL map element
   */
  void AddDlMapElement (OfdmDlMapIe dlMapElement);

  /**
   * Get DCD count field
   * \returns the DCD count
   */
  uint8_t GetDcdCount (void) const;
  /**
   * Get base station ID field
   * \returns the MAC address
   */
  Mac48Address GetBaseStationId (void) const;
  /**
   * Get DL Map elements field
   * \returns the DL map elements
   */
  std::list<OfdmDlMapIe> GetDlMapElements (void) const;

  /**
   * Get name field
   * \returns the name string
   */
  std::string GetName (void) const;
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  void Print (std::ostream &os) const;
  uint32_t GetSerializedSize (void) const;
  void Serialize (Buffer::Iterator start) const;
  uint32_t Deserialize (Buffer::Iterator start);
private:
  uint8_t m_dcdCount; ///< DCD count
  Mac48Address m_baseStationId; ///< base station ID
  std::list<OfdmDlMapIe> m_dlMapElements; ///< DL Map elements
  // m_paddingNibble; //fields to be implemented later on:
};

} // namespace ns3

#endif /* DL_MAP_H */
