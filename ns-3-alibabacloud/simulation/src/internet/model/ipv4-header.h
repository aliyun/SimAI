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

#ifndef IPV4_HEADER_H
#define IPV4_HEADER_H

#include "ns3/header.h"
#include "ns3/ipv4-address.h"

namespace ns3 {
/**
 * \ingroup ipv4
 *
 * \brief Packet header for IPv4
 */
class Ipv4Header : public Header 
{
public:
  /**
   * \brief Construct a null IPv4 header
   */
  Ipv4Header ();
  /**
   * \brief Enable checksum calculation for this header.
   */
  void EnableChecksum (void);
  /**
   * \param size the size of the payload in bytes
   */
  void SetPayloadSize (uint16_t size);
  /**
   * \param identification the Identification field of IPv4 packets.
   *
   * By default, set to zero.
   */
  void SetIdentification (uint16_t identification);
  /**
   * \param tos the 8 bits of Ipv4 TOS.
   */
  void SetTos (uint8_t tos);
public:
  /**
   * \enum DscpType
   * \brief DiffServ codepoints
   *
   * The values correspond to the 6-bit DSCP codepoint within the 8-bit
   * DS field defined in \RFC{2474}.  ECN bits are separately set with the
   * SetEcn() method.  Codepoints are defined in
   * Assured Forwarding (AF) \RFC{2597},
   * Expedited Forwarding (EF) \RFC{2598}, and
   * Default and Class Selector (CS) \RFC{2474}.
   */
  enum DscpType
    {
      DscpDefault = 0x00,

      // Prefixed with "DSCP" to avoid name clash (bug 1723)
      DSCP_CS1  = 0x08, // octal 010
      DSCP_AF11 = 0x0A, // octal 012
      DSCP_AF12 = 0x0C, // octal 014
      DSCP_AF13 = 0x0E, // octal 016

      DSCP_CS2  = 0x10, // octal 020
      DSCP_AF21 = 0x12, // octal 022
      DSCP_AF22 = 0x14, // octal 024
      DSCP_AF23 = 0x16, // octal 026

      DSCP_CS3  = 0x18, // octal 030
      DSCP_AF31 = 0x1A, // octal 032
      DSCP_AF32 = 0x1C, // octal 034
      DSCP_AF33 = 0x1E, // octal 036

      DSCP_CS4  = 0x20, // octal 040
      DSCP_AF41 = 0x22, // octal 042
      DSCP_AF42 = 0x24, // octal 044
      DSCP_AF43 = 0x26, // octal 046

      DSCP_CS5  = 0x28, // octal 050
      DSCP_EF   = 0x2E, // octal 056

      DSCP_CS6  = 0x30, // octal 060
      DSCP_CS7  = 0x38  // octal 070

    };
  /**
   * \brief Set DSCP Field
   * \param dscp DSCP value
   */
  void SetDscp (DscpType dscp);

  /**
   * \enum EcnType
   * \brief ECN Type defined in \RFC{3168}
   */
  enum EcnType
    {
      // Prefixed with "ECN" to avoid name clash (bug 1723)
      ECN_NotECT = 0x00,
      ECN_ECT1 = 0x01,
      ECN_ECT0 = 0x02,
      ECN_CE = 0x03
    }; 
  /**
   * \brief Set ECN Field
   * \param ecn ECN Type
   */
  void SetEcn (EcnType ecn);
  /**
   * This packet is not the last packet of a fragmented ipv4 packet.
   */
  void SetMoreFragments (void);
  /**
   * This packet is the last packet of a fragmented ipv4 packet.
   */
  void SetLastFragment (void);
  /**
   * Don't fragment this packet: if you need to anyway, drop it.
   */
  void SetDontFragment (void);
  /**
   * If you need to fragment this packet, you can do it.
   */
  void SetMayFragment (void);
  /**
   * The offset is measured in bytes for the packet start.
   * Mind that IPv4 "fragment offset" field is 13 bits long and is measured in 8-bytes words.
   * Hence, the function does enforce that the offset is a multiple of 8.
   * \param offsetBytes the ipv4 fragment offset measured in bytes from the start.
   */
  void SetFragmentOffset (uint16_t offsetBytes);
  /**
   * \param ttl the ipv4 TTL
   */
  void SetTtl (uint8_t ttl);
  /**
   * \param num the ipv4 protocol field
   */
  void SetProtocol (uint8_t num);
  /**
   * \param source the source of this packet
   */
  void SetSource (Ipv4Address source);
  /**
   * \param destination the destination of this packet.
   */
  void SetDestination (Ipv4Address destination);
  /**
   * \returns the size of the payload in bytes
   */
  uint16_t GetPayloadSize (void) const;
  /**
   * \returns the identification field of this packet.
   */
  uint16_t GetIdentification (void) const;
  /**
   * \returns the TOS field of this packet.
   */
  uint8_t GetTos (void) const;
  /**
   * \returns the DSCP field of this packet.
   */
  DscpType GetDscp (void) const;
  /**
   * \param dscp the dscp
   * \returns std::string of DSCPType
   */
  std::string DscpTypeToString (DscpType dscp) const;
  /**
   * \returns the ECN field of this packet.
   */
  EcnType GetEcn (void) const;
  /**
   * \param ecn the ECNType
   * \returns std::string of ECNType
   */
  std::string EcnTypeToString (EcnType ecn) const;
  /**
   * \returns true if this is the last fragment of a packet, false otherwise.
   */
  bool IsLastFragment (void) const;
  /**
   * \returns true if this is this packet can be fragmented.
   */
  bool IsDontFragment (void) const;
  /**
   * \returns the offset of this fragment measured in bytes from the start.
   */
  uint16_t GetFragmentOffset (void) const;
  /**
   * \returns the TTL field of this packet
   */
  uint8_t GetTtl (void) const;
  /**
   * \returns the protocol field of this packet
   */
  uint8_t GetProtocol (void) const;
  /**
   * \returns the source address of this packet
   */
  Ipv4Address GetSource (void) const;
  /**
   * \returns the destination address of this packet
   */
  Ipv4Address GetDestination (void) const;

  /**
   * \returns true if the ipv4 checksum is correct, false otherwise.
   *
   * If Ipv4Header::EnableChecksums has not been called prior to
   * deserializing this header, this method will always return true.
   */
  bool IsChecksumOk (void) const;

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

  /// flags related to IP fragmentation
  enum FlagsE {
    DONT_FRAGMENT = (1<<0),
    MORE_FRAGMENTS = (1<<1)
  };

  bool m_calcChecksum; //!< true if the checksum must be calculated

  uint16_t m_payloadSize; //!< payload size
  uint16_t m_identification; //!< identification
  uint32_t m_tos : 8; //!< TOS, also used as DSCP + ECN value
  uint32_t m_ttl : 8; //!< TTL
  uint32_t m_protocol : 8;  //!< Protocol
  uint32_t m_flags : 3; //!< flags
  uint16_t m_fragmentOffset;  //!< Fragment offset
  Ipv4Address m_source; //!< source address
  Ipv4Address m_destination; //!< destination address
  uint16_t m_checksum; //!< checksum
  bool m_goodChecksum; //!< true if checksum is correct
  uint16_t m_headerSize; //!< IP header size
};

} // namespace ns3


#endif /* IPV4_HEADER_H */
