/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008,2009 IITP RAS
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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 *          Aleksey Kovalenko <kovalenko@iitp.ru>
 */

#ifndef IE_DOT11S_PEER_MANAGEMENT_H
#define IE_DOT11S_PEER_MANAGEMENT_H

#include "ns3/mesh-information-element-vector.h"

namespace ns3 {
namespace dot11s {

/**
 * \ingroup dot11s
 * \brief Codes used by 802.11s Peer Management Protocol
 */
enum PmpReasonCode
{
  REASON11S_PEERING_CANCELLED = 52, // according to IEEE 802.11 - 2012
  REASON11S_MESH_MAX_PEERS = 53,
  REASON11S_MESH_CAPABILITY_POLICY_VIOLATION = 54,
  REASON11S_MESH_CLOSE_RCVD = 55,
  REASON11S_MESH_MAX_RETRIES = 56,
  REASON11S_MESH_CONFIRM_TIMEOUT = 57,
  REASON11S_MESH_INVALID_GTK = 58,
  REASON11S_MESH_INCONSISTENT_PARAMETERS = 59,
  REASON11S_MESH_INVALID_SECURITY_CAPABILITY =60,
  REASON11S_RESERVED = 67,
};


/// according to IEEE 802.11 - 2012
class IePeerManagement : public WifiInformationElement
{
public:
  IePeerManagement ();
  /// Subtype enumeration
  enum Subtype
  {
    PEER_OPEN = 1,
    PEER_CONFIRM = 2,
    PEER_CLOSE = 3,
  };
  /**
   * Set peer open function
   * \param localLinkId
   */
  void SetPeerOpen (uint16_t localLinkId);
  /**
   * Set peer close function
   * \param localLinkID the local link id
   * \param peerLinkId the peer link id
   * \param reasonCode the reason code 
   */   
  void SetPeerClose (uint16_t localLinkID, uint16_t peerLinkId, PmpReasonCode reasonCode);
  /**
   * Set peer confirm function
   * \param localLinkID the local link id
   * \param peerLinkId the peer link id
   */   
  void SetPeerConfirm (uint16_t localLinkID, uint16_t peerLinkId);

  /**
   * Get reason code function
   * \returns the reason code
   */
  PmpReasonCode GetReasonCode () const;
  /**
   * Get local link ID function
   * \returns the local link id
   */
  uint16_t GetLocalLinkId () const;
  /**
   * Get peer link ID function
   * \returns the peer link ID
   */ 
  uint16_t GetPeerLinkId () const;
  /**
   * Subtype is open function
   * \returns true if open
   */
  bool SubtypeIsOpen () const;
  /**
   * Subtype is close function
   * \returns true if closed
   */
  bool SubtypeIsClose () const;
  /**
   * Subtype is confirm function
   * \returns true if the subtype is confirm
   */
  bool SubtypeIsConfirm () const;
  /**
   * Get subtype function
   * \returns the subtype
   */
  uint8_t GetSubtype () const;

  // Inherited from WifiInformationElement
  virtual WifiInformationElementId ElementId () const;
  virtual uint8_t GetInformationFieldSize (void) const;
  virtual void SerializeInformationField (Buffer::Iterator i) const;
  virtual uint8_t DeserializeInformationField (Buffer::Iterator i, uint8_t length);
  virtual void Print (std::ostream& os) const;

private:
  uint8_t m_length; ///< length
  uint8_t m_subtype; ///< subtype
  uint16_t m_localLinkId; ///< local link ID
  /**
   * Present within confirm and may be present in close
   */
  uint16_t m_peerLinkId;
  /**
   * Present only within close frame
   */
  PmpReasonCode m_reasonCode;
  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const IePeerManagement & a, const IePeerManagement & b);
};
bool operator== (const IePeerManagement & a, const IePeerManagement & b);
std::ostream &operator << (std::ostream &os, const IePeerManagement &peerMan);
} // namespace dot11s
} // namespace ns3

#endif /* IE_DOT11S_PEER_MANAGEMENT_H */
