/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#ifndef ARP_HEADER_H
#define ARP_HEADER_H

#include "ns3/header.h"
#include "ns3/address.h"
#include "ns3/ipv4-address.h"
#include <string>

namespace ns3 {
/**
 * \ingroup arp
 * \brief The packet header for an ARP packet
 */
class ArpHeader : public Header 
{
public:
  /**
   * \brief Set the ARP request parameters
   * \param sourceHardwareAddress the source hardware address
   * \param sourceProtocolAddress the source IP address
   * \param destinationHardwareAddress the destination hardware address (usually the broadcast address)
   * \param destinationProtocolAddress the destination IP address
   */
  void SetRequest (Address sourceHardwareAddress,
                   Ipv4Address sourceProtocolAddress,
                   Address destinationHardwareAddress,
                   Ipv4Address destinationProtocolAddress);
  /**
   * \brief Set the ARP reply parameters
   * \param sourceHardwareAddress the source hardware address
   * \param sourceProtocolAddress the source IP address
   * \param destinationHardwareAddress the destination hardware address (usually the broadcast address)
   * \param destinationProtocolAddress the destination IP address
   */
  void SetReply (Address sourceHardwareAddress,
                 Ipv4Address sourceProtocolAddress,
                 Address destinationHardwareAddress,
                 Ipv4Address destinationProtocolAddress);

  /**
   * \brief Check if the ARP is a request
   * \returns true if it is a request
   */
  bool IsRequest (void) const;

  /**
   * \brief Check if the ARP is a reply
   * \returns true if it is a reply
   */
  bool IsReply (void) const;

  /**
   * \brief Returns the source hardware address
   * \returns the source hardware address
   */
  Address GetSourceHardwareAddress (void) const;

  /**
   * \brief Returns the destination hardware address
   * \returns the destination hardware address
   */
  Address GetDestinationHardwareAddress (void) const;

  /**
   * \brief Returns the source IP address
   * \returns the source IP address
   */
  Ipv4Address GetSourceIpv4Address (void) const;

  /**
   * \brief Returns the destination IP address
   * \returns the destination IP address
   */
  Ipv4Address GetDestinationIpv4Address (void) const;

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

  /**
   * \brief Enumeration listing the possible ARP types
   */
  enum ArpType_e {
    ARP_TYPE_REQUEST = 1,
    ARP_TYPE_REPLY   = 2
  };
  uint16_t m_type;           //!< type of the ICMP (ARP_TYPE_REQUEST)
  Address m_macSource;       //!< hardware source address
  Address m_macDest;         //!< hardware destination address
  Ipv4Address m_ipv4Source;  //!< IP source address
  Ipv4Address m_ipv4Dest;    //!< IP destination address
};

} // namespace ns3

#endif /* ARP_HEADER_H */
