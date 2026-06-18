/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Dalian University of Technology
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
 * Author: Junling Bu <linlinjavaer@gmail.com>
 */
#ifndef Vendor_Specific_Action_H
#define Vendor_Specific_Action_H

#include <ostream>
#include <map>
#include "ns3/header.h"
#include "ns3/packet.h"
#include "ns3/pointer.h"
#include "ns3/attribute.h"
#include "ns3/attribute-helper.h"
#include "ns3/address.h"

namespace ns3 {

class WifiMac;
class VendorSpecificContentManager;

/**
 * \ingroup wave
 *
 * \brief the organization identifier is a public organizationally
 * unique identifier assigned by the IEEE.
 *
 * Similar to protocol field of data packets that indicates which
 * entity of higher layer should handle received packets, Organization
 * Identifier field is used in Vendor Specific Action frames to
 * indicate which entity or higher layer should handle vendor specific
 * content.
 * Normally the value is assigned by IEEE and the length of field is
 * either 24 bits or 36 bits.
 * For more, see IEEE802.11p-2010 section 7.3.1.31 and 7.4.5
 *
 * \see attribute_OrganizationIdentifier
 */
class OrganizationIdentifier
{
public:
  OrganizationIdentifier (void);
  /**
   * Constructor
   *
   * \param str identifier name
   * \param length identifier length
   */
  OrganizationIdentifier (const uint8_t *str, uint32_t length);
  /**
   * Assignment operator
   * \param oi object to copy from
   * \return reference to the new object
   */
  OrganizationIdentifier& operator= (const OrganizationIdentifier& oi);
  virtual ~OrganizationIdentifier (void);

  /// OrganizationIdentifierType enumeration
  enum OrganizationIdentifierType
  {
    OUI24 = 3,  // 3 bytes
    OUI36 = 5,   // 5 bytes
    Unknown = 0,
  };

  /**
   * \returns last 4 bits when OrganizationIdentifier is OUI36
   */
  uint8_t GetManagementId (void) const;
  /**
   * \returns whether current OrganizationIdentifier is initial state
   */
  bool IsNull (void) const;
  /**
   * \param type set the type of current OrganizationIdentifier
   */
  void SetType (enum OrganizationIdentifierType type);
  /**
   * \returns whether this OrganizationIdentifier is OUI24 or OUI36.
   */
  enum OrganizationIdentifierType GetType (void) const;

  // below methods will be called by VendorSpecificActionHeader
  /**
   * Get serialized size
   * \returns the serialized size
   */
  uint32_t GetSerializedSize (void) const;
  /**
   * Serialize to buffer
   * \param start the iterator
   */
  void Serialize (Buffer::Iterator start) const;
  /**
   * Deserialize from buffer
   * \param start the iterator
   * \returns the deserialize size
   */
  uint32_t Deserialize (Buffer::Iterator start);

private:
  friend bool operator == (const OrganizationIdentifier& a, const OrganizationIdentifier& b);
  friend bool operator != (const OrganizationIdentifier& a, const OrganizationIdentifier& b);
  friend bool operator < (const OrganizationIdentifier& a, const OrganizationIdentifier& b);
  friend std::ostream& operator << (std::ostream& os, const OrganizationIdentifier& oi);
  friend std::istream& operator >> (std::istream& is, const OrganizationIdentifier& oi);

  enum OrganizationIdentifierType m_type; ///< OI type
  uint8_t m_oi[5]; ///< organization identifier
};

ATTRIBUTE_HELPER_HEADER (OrganizationIdentifier);

bool operator == (const OrganizationIdentifier& a, const OrganizationIdentifier& b);
bool operator != (const OrganizationIdentifier& a, const OrganizationIdentifier& b);
bool operator < (const OrganizationIdentifier& a, const OrganizationIdentifier& b);

std::ostream& operator << (std::ostream& os, const OrganizationIdentifier& oi);
std::istream& operator >> (std::istream& is, const OrganizationIdentifier& oi);

/**
 * see IEEE 802.11-2007 chapter 7.3.1.11
 * Table 7-24—Category values
 */
const static uint8_t CATEGORY_OF_VSA = 127;

/**
 * \ingroup wave
 * See IEEE 802.11-2007 chapter 7.3.1.11 and 7.4.5
 * also IEEE 802.11p-2010 chapter 7.4.5
 * Although WifiActionHeader has been defined in wifi mgt-header.h/.cc,
 * it is not a good way to inherit from it or add vendor specific action support.
 * The reason is Action field. Other Action frames contains Action field, and
 * the VSA frame contains OUI or variable OrganizationIdentifier instead of one byte Action field.
 * Header format: | Category | Organization IdentifierType | Vendor Specific content |
 *        Octets:       1                j                        Variable
 *
 * In ns-3, VSA frame will be divided into  VendorSpecificActionHeader vsaHeader
 * and Packet vsc that indicates vendor specific content.
 *
 * This frame is used by IEEE 1609.4 to send WSA packet of IEEE 1609.3,
 * and if users want to send management information without customized management
 * frames, he can use VSA frames.
 */
class VendorSpecificActionHeader : public Header
{
public:
  VendorSpecificActionHeader (void);
  virtual ~VendorSpecificActionHeader (void);

  /**
   * \param oi the OrganizationIdentifier of current VSA header
   */
  void SetOrganizationIdentifier (OrganizationIdentifier oi);
  /**
   * \returns current OrganizationIdentifier of the VSA header
   */
  OrganizationIdentifier GetOrganizationIdentifier (void) const;
  /**
   * Get the category field
   *
   * The category field shall be CATEGORY_OF_VSA
   *
   * \return The category field
   */
  uint8_t GetCategory (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  OrganizationIdentifier m_oi; ///< OI
  uint8_t m_category; ///< category
};

/**
 * \param mac a pointer to the mac object which is calling this callback
 * \param oi the organization identifier of vendor specific action frame
 * \param packet the vendor specific content packet received
 * \param sender the address of the sender
 * \returns true if the callback could handle the packet successfully;
 *       false otherwise.
 */
typedef Callback<bool, Ptr<WifiMac>, const OrganizationIdentifier &, Ptr<const Packet>,const Address &> VscCallback;

/**
 * \ingroup wave
 *
 * VendorSpecificContentManager class
 */
class VendorSpecificContentManager
{
public:
  VendorSpecificContentManager (void);
  virtual ~VendorSpecificContentManager (void);
  /**
   * \param oi the specific OrganizationIdentifier when receive management information
   * by VendorSpecificAction management frame.
   * \param cb the receive callback when oi related management packets are received
   */
  void RegisterVscCallback (OrganizationIdentifier oi, VscCallback cb);
  /**
   * \param oi the specific OrganizationIdentifier when receive management information
   * by VendorSpecificAction management frame.
   */
  void DeregisterVscCallback (OrganizationIdentifier &oi);
  /**
   * \param oi the specific OrganizationIdentifier when receive management information
   * by VendorSpecificAction management frame.
   * \return true if registered
   */
  bool IsVscCallbackRegistered (OrganizationIdentifier &oi);
  /**
   * \param oi the specific OrganizationIdentifier when receive management information
   * by VendorSpecificAction management frame.
   * \return VscCallback
   */
  VscCallback FindVscCallback (OrganizationIdentifier &oi);

private:
  /// VSC callback typedef
  typedef std::map<OrganizationIdentifier,VscCallback> VscCallbacks;
  /// VSC callback iterator typedef
  typedef std::map<OrganizationIdentifier,VscCallback>::iterator VscCallbacksI;

  VscCallbacks m_callbacks; ///< VSC callbacks
};

static std::vector<OrganizationIdentifier> OrganizationIdentifiers; ///< the OIs
}

#endif /* Vendor_Specific_Action_H */
