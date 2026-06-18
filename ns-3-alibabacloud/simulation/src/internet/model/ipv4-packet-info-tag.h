/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 Hajime Tazaki
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
 * Authors: Hajime Tazaki <tazaki@sfc.wide.ad.jp>
 */

#ifndef IPV4_PACKET_INFO_TAG_H
#define IPV4_PACKET_INFO_TAG_H

#include "ns3/tag.h"
#include "ns3/ipv4-address.h"

namespace ns3 {


class Node;
class Packet;

/**
 * \ingroup ipv4
 *
 * \brief This class implements Linux struct pktinfo 
 * in order to deliver ancillary information to the socket interface.
 * This is used with socket option such as IP_PKTINFO, IP_RECVTTL, 
 * IP_RECVTOS. See linux manpage ip(7).
 *
 * See also SocketIpTosTag and SocketIpTtlTag
 *
 * The Tag does not carry the Local address (as it is not currently
 * used to force a source address).
 *
 * This tag in the send direction is presently not enabled but we
 * would accept a patch along those lines in the future.
 */
class Ipv4PacketInfoTag : public Tag
{
public:
  Ipv4PacketInfoTag ();

  /**
   * \brief Set the tag's address
   *
   * \param addr the address
   */
  void SetAddress (Ipv4Address addr);

  /**
   * \brief Get the tag's address
   *
   * \returns the address
   */
  Ipv4Address GetAddress (void) const;

  /**
   * \brief Set the tag's receiving interface
   *
   * \param ifindex the interface index
   */
  void SetRecvIf (uint32_t ifindex);
  /**
   * \brief Get the tag's receiving interface
   *
   * \returns the interface index
   */
  uint32_t GetRecvIf (void) const;

  /**
   * \brief Set the tag's Time to Live
   * Implemented, but not used in the stack yet
   * \param ttl the TTL
   */
  void SetTtl (uint8_t ttl);
  /**
   * \brief Get the tag's Time to Live
   * Implemented, but not used in the stack yet
   * \returns the TTL
   */
  uint8_t GetTtl (void) const;

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (TagBuffer i) const;
  virtual void Deserialize (TagBuffer i);
  virtual void Print (std::ostream &os) const;

private:
  // Linux IP_PKTINFO ip(7) implementation
  // 
  // struct in_pktinfo {
  //   unsigned int   ipi_ifindex;  /* Interface index */
  //   struct in_addr ipi_spec_dst; /* Local address */
  //   struct in_addr ipi_addr;     /* Header Destination
  //                                   address */
  // };

  Ipv4Address m_addr;     //!< Header destination address
  uint32_t m_ifindex;     //!< interface index

  // Used for IP_RECVTTL, though not implemented yet.
  uint8_t m_ttl; //!< Time to Live
};
} // namespace ns3

#endif /* IPV4_PACKET_INFO_TAG_H */
