/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */


#ifndef MESH_WIFI_MAC_HEADER_H
#define MESH_WIFI_MAC_HEADER_H

#include "ns3/header.h"
#include "ns3/mac48-address.h"

namespace ns3 {
namespace dot11s {
/**
 * \ingroup dot11s
 *
 * \brief Mesh Control field, see Section 8.2.4.7.3 IEEE 802.11-2012
 *
 * Header format: | Mesh flags: 1 | TTL: 1 | Sequence number: 4 | Address ext.: 0, 6, or 12 |
 */
class MeshHeader : public Header
{
public:
  MeshHeader ();
  ~MeshHeader ();
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId ();
  virtual TypeId GetInstanceTypeId () const;
  virtual void Print (std::ostream &os) const;

  /**
   * Set extended address 4
   * \param address the MAC address to set
   */
  void SetAddr4 (Mac48Address address);
  /**
   * Set extended address 5
   * \param address the MAC address
   */
  void SetAddr5 (Mac48Address address);
  /**
   * Set extended address 6
   * \param address the MAC address
   */
  void SetAddr6 (Mac48Address address);
  /**
   * Get extended address 4
   * \returns the MAC address
   */
  Mac48Address GetAddr4 () const;
  /**
   * Get extended address 5
   * \returns the MAC address
   */
  Mac48Address GetAddr5 () const;
  /**
   * Get extended address 6
   * \returns the MAC address
   */
  Mac48Address GetAddr6 () const;

  /**
   * Set four-byte mesh sequence number
   * \param seqno the sequence number to set
   */
  void SetMeshSeqno (uint32_t seqno);
  /**
   * Get the four-byte mesh sequence number
   * \returns the sequence number
   */
  uint32_t GetMeshSeqno () const;

  /**
   * Set mesh TTL subfield corresponding to the remaining number of hops
   * the MSDU/MMPDU is forwarded. 
   * 
   * \param TTL the TTL value to set
   */
  void SetMeshTtl (uint8_t TTL);
  /**
   * Get mesh TTL function subfield value
   * \returns the TTL value
   */
  uint8_t GetMeshTtl () const;

  /**
   * Set Address Extension Mode
   * \param num_of_addresses value between 0 and 3 for the two-bit field
   */
  void SetAddressExt (uint8_t num_of_addresses);
  /**
   * Get Address Extension Mode
   * \returns the address extension mode value
   */
  uint8_t GetAddressExt () const;

  virtual uint32_t GetSerializedSize () const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
private:
  uint8_t m_meshFlags; ///< mesh flags
  uint8_t m_meshTtl; ///< mesh TTL
  uint32_t m_meshSeqno; ///< mesh sequence no
  Mac48Address m_addr4; ///< MAC address 4
  Mac48Address m_addr5; ///< MAC address 5
  Mac48Address m_addr6; ///< MAC address 6
  /**
   * equality operator
   *
   * \param a left hand side
   * \param b right hand side
   * \returns true if equal
   */
  friend bool operator== (const MeshHeader & a, const MeshHeader & b);
};
bool operator== (const MeshHeader & a, const MeshHeader & b);

} // namespace dot11s
} // namespace ns3
#endif /* MESH_WIFI_MAC_HEADER_H */
