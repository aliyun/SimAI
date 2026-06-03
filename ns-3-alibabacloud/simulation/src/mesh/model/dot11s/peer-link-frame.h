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
 * Authors: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef PEER_LINK_FRAME_START_H
#define PEER_LINK_FRAME_START_H
#include "ns3/header.h"
#include "ns3/supported-rates.h"
#include "dot11s-mac-header.h"
#include "ie-dot11s-configuration.h"
#include "ie-dot11s-peering-protocol.h"
#include "ie-dot11s-id.h"
namespace ns3
{
namespace dot11s
{
/**
 * \ingroup dot11s
 *
 * \brief 802.11s Peer link open management frame
 * 
 * Peer link open start frame includes the following:
 * - Capability
 * - Supported rates
 * - Mesh ID of mesh
 * - Configuration
 */
class PeerLinkOpenStart : public Header
{
public:
  PeerLinkOpenStart ();
  ///\brief fields:
  struct PlinkOpenStartFields
  {
    IePeeringProtocol protocol; ///< Peering protocol version - 3 octets
    uint16_t capability;        ///< open and confirm
    SupportedRates rates;       ///< open and confirm
    IeMeshId meshId;            ///< open and close
    IeConfiguration config;     ///< open and confirm
  };
  /**
   * Set peer link open start fields
   * \param fields PlinkOpenStartFields to set
   */
  void SetPlinkOpenStart (PlinkOpenStartFields fields);
  /**
   * Get peer link open start fields
   * \return PlinkOpenStartFields
   */
  PlinkOpenStartFields GetFields () const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static  TypeId   GetTypeId ();
  // Inherited from header:
  virtual TypeId   GetInstanceTypeId () const;
  virtual void     Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize () const;
  virtual void     Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  uint16_t m_capability; ///< capability
  SupportedRates m_rates; ///< rates
  IeMeshId m_meshId; ///< mesh ID
  IeConfiguration m_config; ///< config

  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const PeerLinkOpenStart & a, const PeerLinkOpenStart & b);

  /**
   * assignment operator
   * \param peer the peer link open start
   * \returns the assigned value
   */
  PeerLinkOpenStart& operator= (const PeerLinkOpenStart & peer);
  /**
   * constructor
   * \param peer the peer link open start
   */
  PeerLinkOpenStart (const PeerLinkOpenStart & peer);

};

bool operator== (const PeerLinkOpenStart & a, const PeerLinkOpenStart & b);

/**
 * \ingroup dot11s
 *
 * \brief 802.11s Peer link close management frame
 * 
 * Peer link close frame includes the following:
 * - Mesh ID of mesh
 */
class PeerLinkCloseStart : public Header
{
public:
  PeerLinkCloseStart ();
  ///\brief fields:
  struct PlinkCloseStartFields
  {
    IePeeringProtocol protocol; ///< Peering protocol version - 3 octets
    IeMeshId meshId;            ///< open and close
  };
  /**
   * Set peer link close start fields
   * \param fields PlinkCloseStartFields to set
   */
  void SetPlinkCloseStart (PlinkCloseStartFields fields);
  /**
   * Get peer link close start fields 
   * \return PlinkOpenStartFields
   */
  PlinkCloseStartFields GetFields () const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static  TypeId   GetTypeId ();
  // Inherited from header:
  virtual TypeId   GetInstanceTypeId () const;
  virtual void     Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize () const;
  virtual void     Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  IeMeshId m_meshId; ///< mesh ID

  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const PeerLinkCloseStart & a, const PeerLinkCloseStart & b);

  /**
   * assignment operator
   *
   * \param peer the value to assign
   * \returns the assigned value
   */
  PeerLinkCloseStart& operator= (const PeerLinkCloseStart & peer);
  /// type conversion operator
  PeerLinkCloseStart (const PeerLinkCloseStart &);

};
bool operator== (const PeerLinkCloseStart & a, const PeerLinkCloseStart & b);

/**
 * \ingroup dot11s
 *
 * \brief 802.11s Peer link confirm management frame
 * 
 * Peer link confirm frame includes the following:
 * - Association ID field
 * - Supported rates
 * - Configuration
 */
class PeerLinkConfirmStart : public Header
{
public:
  PeerLinkConfirmStart ();
  ///\brief fields:
  struct PlinkConfirmStartFields
  {
    IePeeringProtocol protocol; ///< Peering protocol version - 3 octets
    uint16_t capability;        ///< open and confirm
    uint16_t aid;               ///< confirm only
    SupportedRates rates;       ///< open and confirm
    IeConfiguration config;     ///< open and confirm
  };
  /**
   * Set peer link confirm start fields
   * \param fields PlinkCloseStartFields to set
   */
  void SetPlinkConfirmStart (PlinkConfirmStartFields fields);
  /**
   * Get peer link confirm start fields
   * \return PlinkOpenStartFields
   */
  PlinkConfirmStartFields GetFields () const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static  TypeId   GetTypeId ();
  // Inherited from header:
  virtual TypeId   GetInstanceTypeId () const;
  virtual void     Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize () const;
  virtual void     Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  uint16_t m_capability; ///< capability
  uint16_t m_aid; ///< aid
  SupportedRates m_rates; ///< rates
  IeConfiguration m_config; ///< config

  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const PeerLinkConfirmStart & a, const PeerLinkConfirmStart & b);

  /**
   * assignment operator
   * \param peer the peer link confirm start
   * \returns the assigned value
   */
  PeerLinkConfirmStart& operator= (const PeerLinkConfirmStart & peer);
  /**
   * constructor
   * \param peer the peer link confirm start
   */
  PeerLinkConfirmStart (const PeerLinkConfirmStart & peer);

};
bool operator== (const PeerLinkConfirmStart & a, const PeerLinkConfirmStart & b);
} // namespace dot11s
} // namespace ns3
#endif
