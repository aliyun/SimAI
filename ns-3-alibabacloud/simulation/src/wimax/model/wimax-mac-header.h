/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008, 2009 INRIA, UDcast
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 *         Mohamed Amine Ismail <amine.ismail@sophia.inria.fr>
 */

#ifndef MAC_HEADER_TYPE_H
#define MAC_HEADER_TYPE_H

#include <stdint.h>
#include "ns3/header.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class Represents the HT (Header Type) field of generic MAC and bandwidth request headers
 * \see GenericMacHeader
 * \see BandwidthRequestHeader
*/
class MacHeaderType : public Header
{
public:
  /// Header type enumeration
  enum HeaderType
  {
    HEADER_TYPE_GENERIC,
    HEADER_TYPE_BANDWIDTH
  };

  /**
   * Constructor
   */
  MacHeaderType (void);
  /**
   * Constructor
   *
   * \param type MAC header type
   */
  MacHeaderType (uint8_t type);
  virtual ~MacHeaderType (void);
  /**
   * Set type field
   * \param type the type
   */
  void SetType (uint8_t type);
  /**
   * Get type field
   * \returns the type
   */
  uint8_t GetType (void) const;

  /**
   * Get name field
   * \returns the name
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
  uint8_t m_type; ///< MAC header type
};

} // namespace ns3

#endif /* MAC_HEADER_TYPE_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef GENERIC_MAC_HEADER_H
#define GENERIC_MAC_HEADER_H

#include <stdint.h>
#include "ns3/header.h"
#include "cid.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the Generic mac Header as described by IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems
 * 6.3.2.1.1 Generic MAC header, page 36
 */
class GenericMacHeader : public Header
{
public:
  GenericMacHeader (void);
  ~GenericMacHeader (void);

  /**
   * Set EC field
   * \param ec the EC
   */
  void SetEc (uint8_t ec);
  /**
   * Set type field
   * \param type the type
   */
  void SetType (uint8_t type);
  /**
   * Set CI field
   * \param ci the CI
   */
  void SetCi (uint8_t ci);
  /**
   * Set EKS field
   * \param eks the EKS
   */
  void SetEks (uint8_t eks);
  /**
   * Set length field
   * \param len the length
   */
  void SetLen (uint16_t len);
  /**
   * Set CID field
   * \param cid the CID
   */
  void SetCid (Cid cid);
  /**
   * Set HCS field
   * \param hcs the HCS
   */
  void SetHcs (uint8_t hcs);
  /**
   * Set HT field
   * \param ht the HT
   */
  void SetHt (uint8_t ht);

  /**
   * Get EC field
   * \returns the EC
   */
  uint8_t GetEc (void) const;
  /**
   * Get type field
   * \returns the type
   */
  uint8_t GetType (void) const;
  /**
   * Get CI field
   * \returns the CI
   */
  uint8_t GetCi (void) const;
  /**
   * Get EKS field
   * \returns the EKS
   */
  uint8_t GetEks (void) const;
  /**
   * Get length field
   * \returns the length
   */
  uint16_t GetLen (void) const;
  /**
   * Get CID field
   * \returns the CID
   */
  Cid GetCid (void) const;
  /**
   * Get HCS field
   * \returns the HCS
   */
  uint8_t GetHcs (void) const;
  /**
   * Get HT field
   * \returns the HT
   */
  uint8_t GetHt (void) const;
  /**
   * Get name field
   * \returns the name
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
  /**
   * Check HCS
   * \returns true if HCS is validated
   */
  bool check_hcs (void) const;
private:
  uint8_t m_ht; ///< Header type
  uint8_t m_ec; ///< Encryption Control
  uint8_t m_type; ///< type
  uint8_t m_esf; ///< ESF
  uint8_t m_ci; ///< CRC Indicator
  uint8_t m_eks; ///< Encryption Key Sequence
  uint8_t m_rsv1; ///< RSV
  uint16_t m_len; ///< length
  Cid m_cid; ///< CID
  uint8_t m_hcs; ///< Header Check Sequence
  uint8_t c_hcs; ///< calculated header check sequence; this is used to check if the received header is correct or not


};

} // namespace ns3

#endif /* GENERIC_MAC_HEADER */

// ----------------------------------------------------------------------------------------------------------

#ifndef BANDWIDTH_REQUEST_HEADER_H
#define BANDWIDTH_REQUEST_HEADER_H

#include <stdint.h>
#include "ns3/header.h"
#include "cid.h"

namespace ns3 {
/**
 * \ingroup wimax
 * This class implements the bandwidth-request mac Header as described by IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems
 * 6.3.2.1.2 Bandwidth request header, page 38
 */
class BandwidthRequestHeader : public Header
{
public:
  /// Header type enumeration
  enum HeaderType
  {
    HEADER_TYPE_INCREMENTAL, HEADER_TYPE_AGGREGATE
  };

  BandwidthRequestHeader (void);
  ~BandwidthRequestHeader (void);

  /**
   * Set HT field
   * \param ht the HT
   */
  void SetHt (uint8_t ht);
  /**
   * Set EC field
   * \param ec the EC
   */
  void SetEc (uint8_t ec);
  /**
   * Set type field
   * \param type the type
   */
  void SetType (uint8_t type);
  /**
   * Set BR field
   * \param br the BR
   */
  void SetBr (uint32_t br);
  /**
   * Set CID field
   * \param cid the CID
   */
  void SetCid (Cid cid);
  /**
   * Set HCS field
   * \param hcs the HCS
   */
  void SetHcs (uint8_t hcs);

  /**
   * Get HT field
   * \returns the HT
   */
  uint8_t GetHt (void) const;
  /**
   * Get EC field
   * \returns the EC
   */
  uint8_t GetEc (void) const;
  /**
   * Get type field
   * \returns the type
   */
  uint8_t GetType (void) const;
  /**
   * Get BR field
   * \returns the BR
   */
  uint32_t GetBr (void) const;
  /**
   * Get CID field
   * \returns the CID
   */
  Cid GetCid (void) const;
  /**
   * Get HCS field
   * \returns the HCS
   */
  uint8_t GetHcs (void) const;

  /**
   * Get name field
   * \returns the name
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
  /**
   * Check HCS
   * \returns true if HCS is validated
   */
  bool check_hcs (void) const;
private:
  uint8_t m_ht; ///< Header type
  uint8_t m_ec; ///< Encryption Control
  uint8_t m_type; ///< type
  uint32_t m_br; ///< Bandwidth Request
  Cid m_cid; ///< Connection identifier
  uint8_t m_hcs; ///< Header Check Sequence
  uint8_t c_hcs; ///< calculated header check sequence; this is used to check if the received header is correct or not

};

} // namespace ns3

#endif /* BANDWIDTH_REQUEST_HEADER_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef GRANT_MANAGEMENT_SUBHEADER_H
#define GRANT_MANAGEMENT_SUBHEADER_H

#include <stdint.h>
#include "ns3/header.h"

namespace ns3 {

/**
 * \ingroup wimax
 * This class implements the grant management sub-header as described by IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems
 * 6.3.2.2.2 Grant Management subheader, page 40
 */
class GrantManagementSubheader : public Header
{
public:
  GrantManagementSubheader (void);
  ~GrantManagementSubheader (void);

  /**
   * Set SI field
   * \param si the SI
   */
  void SetSi (uint8_t si);
  /**
   * Set PM field
   * \param pm the PM
   */
  void SetPm (uint8_t pm);
  /**
   * Set PRR field
   * \param pbr the PBR
   */
  void SetPbr (uint16_t pbr);

  /**
   * Get SI field
   * \returns the SI
   */
  uint8_t GetSi (void) const;
  /**
   * Get PM field
   * \returns the PM
   */
  uint8_t GetPm (void) const;
  /**
   * Get PBR field
   * \returns the PBR
   */
  uint16_t GetPbr (void) const;

  /**
   * Get name field
   * \returns the name
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
  // size of the Grant Management Subheader shall actually be 2 bytes

  uint8_t m_si; ///< Slip Indicator
  uint8_t m_pm; ///< Poll-Me bit (byte in this case)
  uint16_t m_pbr; ///< PiggyBack Request
};

} // namespace ns3

#endif /* GRANT_MANAGEMENT_SUBHEADER_H */

// ----------------------------------------------------------------------------------------------------------

#ifndef FRAGMENTATION_SUBHEADER_H
#define FRAGMENTATION_SUBHEADER_H

#include <stdint.h>
#include "ns3/header.h"

namespace ns3 {
/**
 * \ingroup wimax
 * This class implements the fragmentation sub-header as described by IEEE Standard for
 * Local and metropolitan area networks Part 16: Air Interface for Fixed Broadband Wireless Access Systems
 * 6.3.2.2.1 Fragmentation subheader, page 39
 */
class FragmentationSubheader : public Header
{
public:
  FragmentationSubheader (void);
  ~FragmentationSubheader (void);

  /**
   * Set FC field
   * \param fc the FC
   */
  void SetFc (uint8_t fc);
  /**
   * Set FSN field
   * \param fsn the FSN
   */
  void SetFsn (uint8_t fsn);

  /**
   * Get FC field
   * \returns the FC
   */
  uint8_t GetFc (void) const;
  /**
   * Get FSN field
   * \returns the FSN
   */
  uint8_t GetFsn (void) const;

  /**
   * Get name field
   * \returns the name
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
  uint8_t m_fc;  ///< Fragment Control
  uint8_t m_fsn; ///< Fragment Sequence Number
};
} // namespace ns3

#endif /* FRAGMENTATION_SUBHEADER_H */
