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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */

#ifndef WIFI_PREQ_INFORMATION_ELEMENT_H
#define WIFI_PREQ_INFORMATION_ELEMENT_H

#include <vector>

#include "ns3/mac48-address.h"
#include "ns3/mesh-information-element-vector.h"

namespace ns3 {
namespace dot11s {
/**
 * \ingroup dot11s
 * \brief Describes an address unit in PREQ information element
 * See 7.3.2.96 for more details
 */
class DestinationAddressUnit : public SimpleRefCount<DestinationAddressUnit>
{
public:
  DestinationAddressUnit ();
  /**
   * Set flags function
   * \param doFlag the DO flag
   * \param rfFlag the RF flag
   * \param usnFlag the USN flag
   */
  void SetFlags (bool doFlag, bool rfFlag, bool usnFlag);
  /**
   * Set destination address function
   * \param dest_address the destination MAC address
   */
  void SetDestinationAddress (Mac48Address dest_address);
  /**
   * Set destination sequence number
   * \param dest_seq_number the destination sequence number
   */
  void SetDestSeqNumber (uint32_t dest_seq_number);
  /**
   * Is do function
   * \returns true if DO flag is set
   */
  bool IsDo ();
  /**
   * is RF function
   * \returns true if RF flag is set
   */
  bool IsRf ();
  /**
   * Is USN function
   * \returns true if USN flag set
   */
  bool IsUsn ();
  /**
   * Get destination address function
   * \returns the MAC address
   */
  Mac48Address GetDestinationAddress () const;
  /**
   * Get destination sequence number
   * \returns the destination sequence number
   */
  uint32_t GetDestSeqNumber () const;
private:
  bool m_do; ///< DO
  bool m_rf; ///< RF
  bool m_usn; ///< USN
  Mac48Address m_destinationAddress; ///< destination address
  uint32_t m_destSeqNumber; ///< destination sequence number

  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const DestinationAddressUnit & a, const DestinationAddressUnit & b);
};
/**
 * \ingroup dot11s
 * \brief See 7.3.2.96 of 802.11s draft 2.07
 */
class IePreq : public WifiInformationElement
{
public:
  IePreq ();
  ~IePreq ();
  /**
   * Add a destination address unit: flags, destination and sequence
   * number
   *
   * \param doFlag
   * \param rfFlag
   * \param dest_address
   * \param dest_seq_number
   */
  void AddDestinationAddressElement (
    bool doFlag,
    bool rfFlag,
    Mac48Address dest_address,
    uint32_t dest_seq_number
    );
  /**
   * Delete a destination address unit by destination
   * \param dest_address the destination address
   */
  void DelDestinationAddressElement (Mac48Address dest_address);
  /// Clear PREQ: remove all destinations
  void ClearDestinationAddressElements ();
  /**
   * Get all destinations, which are stored in PREQ:
   * \returns the list of destination addresses
   */
  std::vector<Ptr<DestinationAddressUnit> > GetDestinationList ();
  /// Set flag indicating that PREQ is unicast
  void SetUnicastPreq ();
  /**
   * \brief Set Proactive PREP subfield to off
   */
  void SetNeedNotPrep ();

  // Setters for fields:
  /**
   * Set number of hops from originator to mesh STA transmitting this 
   * element
   * \param hopcount the hop count
   */
  void SetHopcount (uint8_t hopcount);
  /**
   * Set remaining number of hops allowed for this element
   * \param ttl the TTL
   */
  void SetTTL (uint8_t ttl);
  /**
   * Set path discovery id field
   * \param id some unique id for this path discovery
   */
  void SetPreqID (uint32_t id);
  /**
   * Set originator address value
   * \param originator_address the originator MAC address
   */
  void SetOriginatorAddress (Mac48Address originator_address);
  /**
   * Set originator sequence number
   * \param originator_seq_number
   */
  void SetOriginatorSeqNumber (uint32_t originator_seq_number);
  /**
   * Set lifetime in TUs for the forwarding information to be considered valid
   * \param lifetime the lifetime in TUs
   */
  void SetLifetime (uint32_t lifetime);
  /**
   * Set metric value
   * \param metric the metric
   */
  void SetMetric (uint32_t metric);
  /**
   * Set destination count value
   * \param dest_count the destination count
   */
  void SetDestCount (uint8_t dest_count);

  // Getters for fields:
  /**
   * Is unicast PREQ function
   * \returns true if unicast PREQ
   */
  bool  IsUnicastPreq () const;
  /**
   * Check whether Proactive PREP subfield to off
   * \returns true if need not Proactive PREP subfield is off
   */
  bool  IsNeedNotPrep () const;
  /**
   * Get hop count value
   * \returns the hop count
   */
  uint8_t  GetHopCount () const;
  /**
   * Get TTL value
   * \returns the TTL
   */
  uint8_t  GetTtl () const;
  /**
   * Get path discovery id field
   * \returns the path discovery ID
   */
  uint32_t GetPreqID () const;
  /**
   * Get originator address value
   * \returns the originator MAC address
   */
  Mac48Address GetOriginatorAddress () const;
  /**
   * Get originator sequence numnber value
   * \returns the originator sequence number
   */
  uint32_t GetOriginatorSeqNumber () const;
  /**
   * Get lifetime value
   * \returns the lifetime in TUs
   */
  uint32_t GetLifetime () const;
  /**
   * Get metric value
   * \returns the metric
   */
  uint32_t GetMetric () const;
  /**
   * Get destination count
   * \returns the destination count
   */
  uint8_t  GetDestCount () const;

  /// Handle TTL
  void  DecrementTtl ();
  /**
   * Handle Metric:
   * \param metric the mteric to increment
   */
  void  IncrementMetric (uint32_t metric);
  /**
   * \brief Checks that preq's originator address equals to originator, and
   * this preq is not proactive
   * \param originator the originator address
   * \returns true if it may add
   */
  bool MayAddAddress (Mac48Address originator);
  /**
   * Is full function
   * \returns true if full
   */
  bool IsFull () const;
  
  // Inherited from WifiInformationElement
  virtual WifiInformationElementId ElementId () const;
  virtual void SerializeInformationField (Buffer::Iterator i) const;
  virtual uint8_t DeserializeInformationField (Buffer::Iterator i, uint8_t length);
  virtual uint8_t GetInformationFieldSize () const;
  virtual void Print (std::ostream& os) const;

private:
  /**
   * how many destinations we support
   * \todo make as an attribute
   */
  uint8_t m_maxSize; 

  uint8_t m_flags; ///< flags
  uint8_t m_hopCount; ///< hop count
  uint8_t m_ttl; ///< TTL
  uint32_t m_preqId; ///< PREQ ID
  Mac48Address m_originatorAddress; ///< originator address
  uint32_t m_originatorSeqNumber; ///< originator sequence number
  uint32_t m_lifetime; ///< lifetime
  uint32_t m_metric; ///< metric
  uint8_t  m_destCount; ///< destination count
  std::vector<Ptr<DestinationAddressUnit> >  m_destinations; ///< the destinations

  /**
   * equality operator
   *
   * \param a lhs
   * \param b rhs
   * \returns true if equal
   */
  friend bool operator== (const IePreq & a, const IePreq & b);
};

bool operator== (const DestinationAddressUnit & a, const DestinationAddressUnit & b);
bool operator== (const IePreq & a, const IePreq & b);
std::ostream &operator << (std::ostream &os, const IePreq &preq);

} // namespace dot11s
} // namespace ns3
#endif

