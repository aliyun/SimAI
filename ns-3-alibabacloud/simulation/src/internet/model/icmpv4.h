/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#ifndef ICMPV4_H
#define ICMPV4_H

#include "ns3/header.h"
#include "ns3/ptr.h"
#include "ns3/ipv4-header.h"
#include <stdint.h>

namespace ns3 {

class Packet;

/**
 * \ingroup icmp
 *
 * \brief Base class for all the ICMP packet headers.
 *
 * This header is the common part in all the ICMP packets.
 */
class Icmpv4Header : public Header
{
public:

  /**
   * ICMP type code.
   */
  enum Type_e {
    ICMPV4_ECHO_REPLY = 0,
    ICMPV4_DEST_UNREACH = 3,
    ICMPV4_ECHO = 8,
    ICMPV4_TIME_EXCEEDED = 11
  };

  /**
   * Enables ICMP Checksum calculation
   */
  void EnableChecksum (void);

  /**
   * Set ICMP type
   * \param type the ICMP type
   */
  void SetType (uint8_t type);

  /**
   * Set ICMP code
   * \param code the ICMP code
   */
  void SetCode (uint8_t code);

  /**
   * Get ICMP type
   * \returns the ICMP type
   */
  uint8_t GetType (void) const;
  /**
   * Get ICMP code
   * \returns the ICMP code
   */
  uint8_t GetCode (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  Icmpv4Header ();
  virtual ~Icmpv4Header ();

  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;

private:
  uint8_t m_type; //!< ICMP type
  uint8_t m_code; //!< ICMP code
  bool m_calcChecksum;  //!< true if checksum is calculated
};

/**
 * \ingroup icmp
 *
 * \brief ICMP Echo header
 */
class Icmpv4Echo : public Header
{
public:
  /**
   * \brief Set the Echo identifier
   * \param id the identifier
   */
  void SetIdentifier (uint16_t id);
  /**
   * \brief Set the Echo sequence number
   * \param seq the sequence number
   */
  void SetSequenceNumber (uint16_t seq);
  /**
   * \brief Set the Echo data
   * \param data the data
   */
  void SetData (Ptr<const Packet> data);
  /**
   * \brief Get the Echo identifier
   * \returns the identifier
   */
  uint16_t GetIdentifier (void) const;
  /**
   * \brief Get the Echo sequence number
   * \returns the sequence number
   */
  uint16_t GetSequenceNumber (void) const;
  /**
   * \brief Get the Echo data size
   * \returns the data size
   */
  uint32_t GetDataSize (void) const;
  /**
   * \brief Get the Echo data
   * \param payload the data (filled)
   * \returns the data length
   */
  uint32_t GetData (uint8_t payload[]) const;


  /**
   * Get ICMP type
   * \returns the ICMP type
   */
  static TypeId GetTypeId (void);
  Icmpv4Echo ();
  virtual ~Icmpv4Echo ();
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
private:
  uint16_t m_identifier; //!< identifier
  uint16_t m_sequence;   //!< sequence number
  uint8_t *m_data;       //!< data
  uint32_t m_dataSize;   //!< data size
};

/**
 * \ingroup icmp
 *
 * \brief ICMP Destination Unreachable header
 */
class Icmpv4DestinationUnreachable : public Header
{
public:
  /**
   * ICMP error code : Destination Unreachable
   */
  enum ErrorDestinationUnreachable_e {
    ICMPV4_NET_UNREACHABLE = 0,
    ICMPV4_HOST_UNREACHABLE = 1,
    ICMPV4_PROTOCOL_UNREACHABLE = 2,
    ICMPV4_PORT_UNREACHABLE = 3,
    ICMPV4_FRAG_NEEDED = 4,
    ICMPV4_SOURCE_ROUTE_FAILED = 5
  };

  /**
   * Get ICMP type
   * \returns the ICMP type
   */
  static TypeId GetTypeId (void);
  Icmpv4DestinationUnreachable ();
  virtual ~Icmpv4DestinationUnreachable ();

  /**
   * \brief Set the next hop MTU
   * \param mtu the MTU
   */
  void SetNextHopMtu (uint16_t mtu);
  /**
   * \brief Get the next hop MTU
   * \returns the MTU
   */
  uint16_t GetNextHopMtu (void) const;

  /**
   * \brief Set the ICMP carried data
   * \param data the data
   */
  void SetData (Ptr<const Packet> data);
  /**
   * \brief Set the ICMP carried IPv4 header
   * \param header the header
   */
  void SetHeader (Ipv4Header header);

  /**
   * \brief Get the ICMP carried data
   * \param payload the data (filled)
   */
  void GetData (uint8_t payload[8]) const;
  /**
   * \brief Get the ICMP carried IPv4 header
   * \returns the header
   */
  Ipv4Header GetHeader (void) const;

private:
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;
private:
  uint16_t m_nextHopMtu; //!< next hop MTU
  Ipv4Header m_header;   //!< carried IPv4 header
  uint8_t m_data[8];     //!< carried data
};


/**
 * \ingroup icmp
 *
 * \brief ICMP Time Exceeded header
 */
class Icmpv4TimeExceeded : public Header
{
public:
  /**
   * \brief ICMP error code : Time Exceeded
   */
  enum ErrorTimeExceeded_e
  {
    ICMPV4_TIME_TO_LIVE = 0,
    ICMPV4_FRAGMENT_REASSEMBLY = 1
  };

  /**
   * \brief Get the ICMP carried data
   * \param data the data
   */
  void SetData (Ptr<const Packet> data);
  /**
   * \brief Set the ICMP carried IPv4 header
   * \param header the header
   */
  void SetHeader (Ipv4Header header);

  /**
   * \brief Get the ICMP carried data
   * \param payload the data (filled)
   */
  void GetData (uint8_t payload[8]) const;
  /**
   * \brief Get the ICMP carried IPv4 header
   * \returns the header
   */
  Ipv4Header GetHeader (void) const;

  /**
   * Get ICMP type
   * \returns the ICMP type
   */
  static TypeId GetTypeId (void);
  Icmpv4TimeExceeded ();
  virtual ~Icmpv4TimeExceeded ();
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  virtual void Print (std::ostream &os) const;

private:
  Ipv4Header m_header;   //!< carried IPv4 header
  uint8_t m_data[8];     //!< carried data
};

} // namespace ns3

#endif /* ICMPV4_H */
