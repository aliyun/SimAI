/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 *         Mehdi Benamor <benamor.mehdi@ensi.rnu.tn>
 *         David Gross <gdavid.devel@gmail.com>
 */

#ifndef ICMPV6_HEADER_H
#define ICMPV6_HEADER_H

#include "ns3/header.h"
#include "ns3/ipv6-address.h"
#include "ns3/packet.h"

namespace ns3
{

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 header.
 */
class Icmpv6Header : public Header
{
public:
  /**
   * \brief ICMPv6 type code.
   */
  enum Type_e
  {
    ICMPV6_ERROR_DESTINATION_UNREACHABLE = 1,
    ICMPV6_ERROR_PACKET_TOO_BIG,
    ICMPV6_ERROR_TIME_EXCEEDED,
    ICMPV6_ERROR_PARAMETER_ERROR,
    ICMPV6_ECHO_REQUEST = 128,
    ICMPV6_ECHO_REPLY,
    ICMPV6_SUBSCRIBE_REQUEST,
    ICMPV6_SUBSCRIBE_REPORT,
    ICMPV6_SUBSCRIVE_END,
    ICMPV6_ND_ROUTER_SOLICITATION,
    ICMPV6_ND_ROUTER_ADVERTISEMENT,
    ICMPV6_ND_NEIGHBOR_SOLICITATION,
    ICMPV6_ND_NEIGHBOR_ADVERTISEMENT,
    ICMPV6_ND_REDIRECTION,
    ICMPV6_ROUTER_RENUMBER,
    ICMPV6_INFORMATION_REQUEST,
    ICMPV6_INFORMATION_RESPONSE,
    ICMPV6_INVERSE_ND_SOLICITATION,
    ICMPV6_INVERSE_ND_ADVERSTISEMENT,
    ICMPV6_MLDV2_SUBSCRIBE_REPORT,
    ICMPV6_MOBILITY_HA_DISCOVER_REQUEST,
    ICMPV6_MOBILITY_HA_DISCOVER_RESPONSE,
    ICMPV6_MOBILITY_MOBILE_PREFIX_SOLICITATION,
    ICMPV6_SECURE_ND_CERTIFICATE_PATH_SOLICITATION,
    ICMPV6_SECURE_ND_CERTIFICATE_PATH_ADVERTISEMENT,
    ICMPV6_EXPERIMENTAL_MOBILITY
  };

  /**
   * \brief ICMPv6 Option type code.
   */
  enum OptionType_e
  {
    ICMPV6_OPT_LINK_LAYER_SOURCE = 1,
    ICMPV6_OPT_LINK_LAYER_TARGET,
    ICMPV6_OPT_PREFIX,
    ICMPV6_OPT_REDIRECTED,
    ICMPV6_OPT_MTU
  };

  /**
   * \brief ICMPv6 error code : Destination Unreachable
   */
  enum ErrorDestinationUnreachable_e
  {
    ICMPV6_NO_ROUTE = 0,
    ICMPV6_ADM_PROHIBITED,
    ICMPV6_NOT_NEIGHBOUR,
    ICMPV6_ADDR_UNREACHABLE,
    ICMPV6_PORT_UNREACHABLE
  };

  /**
   * \brief ICMPv6 error code : Time Exceeded
   */
  enum ErrorTimeExceeded_e
  {
    ICMPV6_HOPLIMIT = 0,
    ICMPV6_FRAGTIME
  };

  /**
   * \brief ICMPv6 error code : Parameter Error
   */
  enum ErrorParameterError_e
  {
    ICMPV6_MALFORMED_HEADER = 0,
    ICMPV6_UNKNOWN_NEXT_HEADER,
    ICMPV6_UNKNOWN_OPTION
  };

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Constructor.
   */
  Icmpv6Header ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6Header ();

  /**
   * \brief Get the type field.
   * \return type of ICMPv6 message
   */
  uint8_t GetType () const;

  /**
   * \brief Set the type.
   * \param type type to set
   */
  void SetType (uint8_t type);

  /**
   * \brief Get the code field.
   * \return code of ICMPv6 message
   */
  uint8_t GetCode () const;

  /**
   * \brief Set the code field.
   * \param code code to set
   */
  void SetCode (uint8_t code);

  /**
   * \brief Get the checksum.
   * \return checksum
   */
  uint16_t GetChecksum () const;

  /**
   * \brief Set the checksum.
   * \param checksum to set
   */
  void SetChecksum (uint16_t checksum);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

  /**
   * \brief Calculate pseudo header checksum for IPv6.
   * \param src source address
   * \param dst destination address
   * \param length length
   * \param protocol the protocol number to use in the
   * underlying IPv6 packet.
   */
  void CalculatePseudoHeaderChecksum (Ipv6Address src, Ipv6Address dst, uint16_t length, uint8_t protocol);

protected:
  /**
   * \brief Checksum enable or not.
   */
  bool m_calcChecksum;

  /**
   * \brief The checksum.
   */
  uint16_t m_checksum; 

private:
  /**
   * \brief The type.
   */
  uint8_t m_type;

  /**
   * \brief The code.
   */
  uint8_t m_code;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 option header.
 */
class Icmpv6OptionHeader : public Header
{
public:
  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Constructor.
   */
  Icmpv6OptionHeader ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6OptionHeader ();

  /**
   * \brief Get the type of the option.
   * \return type
   */
  uint8_t GetType () const;

  /**
   * \brief Set the type of the option.
   * \param type the type to set
   */
  void SetType (uint8_t type);

  /**
   * \brief Get the length of the option in 8 bytes unit.
   * \return length of the option
   */
  uint8_t GetLength () const;

  /**
   * \brief Set the length of the option.
   * \param len length value to set
   */
  void SetLength (uint8_t len);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The type.
   */
  uint8_t m_type;

  /**
   * \brief The length.
   */
  uint8_t m_len;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Neighbor Solicitation header.
 */
class Icmpv6NS : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   * \param target target IPv6 address
   */
  Icmpv6NS (Ipv6Address target);

  /**
   * \brief Constructor.
   */
  Icmpv6NS ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6NS ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Get the reserved field.
   * \return reserved value
   */
  uint32_t GetReserved () const;

  /**
   * \brief Set the reserved field.
   * \param reserved the reserved value
   */
  void SetReserved (uint32_t reserved);

  /**
   * \brief Get the IPv6 target field.
   * \return IPv6 address
   */
  Ipv6Address GetIpv6Target () const;

  /**
   * \brief Set the IPv6 target field.
   * \param target IPv6 address
   */
  void SetIpv6Target (Ipv6Address target);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:

  /**
   * \brief The reserved value.
   */
  uint32_t m_reserved;

  /**
   * \brief The IPv6 target address.
   */
  Ipv6Address m_target;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Neighbor Advertisement header.
 */
class Icmpv6NA : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6NA ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6NA ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Get the reserved field.
   * \return reserved value
   */
  uint32_t GetReserved () const;

  /**
   * \brief Set the reserved field.
   * \param reserved the reserved value
   */
  void SetReserved (uint32_t reserved);

  /**
   * \brief Get the IPv6 target field.
   * \return IPv6 address
   */
  Ipv6Address GetIpv6Target () const;

  /**
   * \brief Set the IPv6 target field.
   * \param target IPv6 address
   */
  void SetIpv6Target (Ipv6Address target);

  /**
   * \brief Get the R flag.
   * \return R flag
   */
  bool GetFlagR () const;

  /**
   * \brief Set the R flag.
   * \param r value
   */
  void SetFlagR (bool r);

  /**
   * \brief Get the S flag.
   * \return S flag
   */
  bool GetFlagS () const;

  /**
   * \brief Set the S flag.
   * \param s value
   */
  void SetFlagS (bool s);

  /**
   * \brief Get the O flag.
   * \return O flag
   */
  bool GetFlagO () const;

  /**
   * \brief Set the O flag.
   * \param o value
   */
  void SetFlagO (bool o);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The R flag.
   */
  bool m_flagR;

  /**
   * \brief The O flag.
   */
  bool m_flagS;

  /**
   * \brief The M flag.
   */
  bool m_flagO;

  /**
   * \brief The reserved value.
   */
  uint32_t m_reserved;

  /**
   * \brief The IPv6 target address.
   */
  Ipv6Address m_target;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Router Advertisement header.
 */
class Icmpv6RA : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6RA ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6RA ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Set the IPv6 maximum number of jumps.
   * \param m maximum jumps
   */
  void SetCurHopLimit (uint8_t m);

  /**
   * \brief Get the IPv6 maximum number of jumps.
   * \return maximum jumps
   */
  uint8_t GetCurHopLimit () const;

  /**
   * \brief Set the node Life time (Neighbor Discovery).
   * \param l life time
   */
  void SetLifeTime (uint16_t l);

  /**
   * \brief Get the node Life time (Neighbor Discovery).
   * \return life time
   */
  uint16_t GetLifeTime () const;

  /**
   * \brief Set the node Reachable time (Neighbor Discovery).
   * \param r Reachable time
   */
  void SetReachableTime (uint32_t r);

  /**
   * \brief Get the node Reachable time (Neighbor Discovery).
   * \return reachable time
   */
  uint32_t GetReachableTime () const;

  /**
   * \brief Set the node Retransmission time (Neighbor Discovery).
   * \param r Retransmission time
   */
  void SetRetransmissionTime (uint32_t r);

  /**
   * \brief Get the node Retransmission time (Neighbor Discovery).
   * \return retransmission time
   */
  uint32_t GetRetransmissionTime () const;

  /**
   * \brief Get the M flag.
   * \return M flag
   */
  bool GetFlagM () const;

  /**
   * \brief Set the M flag.
   * \param m value
   */
  void SetFlagM (bool m);

  /**
   * \brief Get the O flag.
   * \return O flag
   */
  bool GetFlagO () const;

  /**
   * \brief Set the O flag.
   * \param o value
   */
  void SetFlagO (bool o);

  /**
   * \brief Get the H flag.
   * \return H flag
   */
  bool GetFlagH () const;

  /**
   * \brief Set the H flag.
   * \param h value
   */
  void SetFlagH (bool h);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Getflags.
   * \return the flags value
   */
  NS_DEPRECATED_3_34
  uint8_t GetFlags () const;

  /**
   * \brief Setflags.
   * \param f the flags value
   */
  NS_DEPRECATED_3_34
  void SetFlags (uint8_t f);

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The M flag.
   */
  bool m_flagM;

  /**
   * \brief The O flag.
   */
  bool m_flagO;

  /**
   * \brief The H flag.
   */
  bool m_flagH;

  /**
   * \brief The lifetime value.
   */
  uint16_t m_LifeTime;

  /**
   * \brief The reachable time value.
   */
  uint32_t m_ReachableTime;

  /**
   * \brief The retransmission timer.
   */
  uint32_t m_RetransmissionTimer;

  /**
   * \brief The max jumps.
   */
  uint8_t m_curHopLimit;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Router Solicitation header.
 */
class Icmpv6RS : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6RS ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6RS ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Get the reserved field.
   * \return reserved value
   */
  uint32_t GetReserved () const;

  /**
   * \brief Set the reserved field.
   * \param reserved the reserved value
   */
  void SetReserved (uint32_t reserved);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The reserved value.
   */
  uint32_t m_reserved;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Redirection header.
 */
class Icmpv6Redirection : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6Redirection ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6Redirection ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Get the IPv6 target address.
   * \return the IPv6 target address
   */
  Ipv6Address GetTarget () const;

  /**
   * \brief Set the IPv6 target address.
   * \param target IPv6 target address
   */
  void SetTarget (Ipv6Address target);

  /**
   * \brief Get the IPv6 destination address.
   * \return the IPv6 destination address
   */
  Ipv6Address GetDestination () const;

  /**
   * \brief Set the IPv6 destination address.
   * \param destination IPv6 destination address
   */
  void SetDestination (Ipv6Address destination);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

  /**
   * \brief Get the reserved field.
   * \return reserved value
   */
  uint32_t GetReserved () const;

  /**
   * \brief Set the reserved field.
   * \param reserved the reserved value
   */
  void SetReserved (uint32_t reserved);

private:
  /**
   * \brief IPv6 target address.
   */
  Ipv6Address m_target;

  /**
   * \brief IPv6 destination address.
   */
  Ipv6Address m_destination;

  /**
   * \brief Reserved value.
   */
  uint32_t m_reserved;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Echo message.
 */
class Icmpv6Echo : public Icmpv6Header
{
public:
  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Default constructor.
   */
  Icmpv6Echo ();

  /**
   * \brief Constructor.
   * \param request request or reply message
   */
  Icmpv6Echo (bool request);

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6Echo ();

  /**
   * \brief Get the ID of the packet.
   * \return id
   */
  uint16_t GetId () const;

  /**
   * \brief Set the ID of the packet.
   * \param id id to set
   */
  void SetId (uint16_t id);

  /**
   * \brief Get the sequence number.
   * \return sequence number
   */
  uint16_t GetSeq () const;

  /**
   * \brief Set the sequence number.
   * \param seq sequence to set
   */
  void SetSeq (uint16_t seq);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief ID of the packet (to distinguish response between many ping program).
   */
  uint16_t m_id;

  /**
   * \brief Sequence number (to distinguish response).
   */
  uint16_t m_seq;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Error Destination Unreachable header.
 */
class Icmpv6DestinationUnreachable : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6DestinationUnreachable ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6DestinationUnreachable ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Set the incorrect packet.
   * \param p the incorrect packet
   */
  void SetPacket (Ptr<Packet> p);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The incorrect Packet.
   */
  Ptr<Packet> m_packet;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Error Too Big header.
 */
class Icmpv6TooBig : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6TooBig ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6TooBig ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Set the incorrect packet.
   * \param p the incorrect packet
   */
  void SetPacket (Ptr<Packet> p);

  /**
   * \brief Get the MTU field.
   * \return MTU value
   */
  uint32_t GetMtu () const;

  /**
   * \brief Set the MTU.
   * \param mtu the MTU
   */
  void SetMtu (uint32_t mtu);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:

  /**
   * \brief the incorrect packet.
   */
  Ptr<Packet> m_packet;

  /**
   * \brief The MTU value.
   */
  uint32_t m_mtu;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Error Time Exceeded header.
 */
class Icmpv6TimeExceeded : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6TimeExceeded ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6TimeExceeded ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Set the incorrect packet.
   * \param p the incorrect packet
   */
  void SetPacket (Ptr<Packet> p);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:

  /**
   * \brief The incorrect packet.
   */
  Ptr<Packet> m_packet;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Error Parameter Error header.
 */
class Icmpv6ParameterError : public Icmpv6Header
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6ParameterError ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6ParameterError ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Set the incorrect packet.
   * \param p the incorrect packet
   */
  void SetPacket (Ptr<Packet> p);

  /**
   * \brief Get the pointer field.
   * \return pointer value
   */
  uint32_t GetPtr () const;

  /**
   * \brief Set the pointer field.
   * \param ptr byte where the error is located in the incorrect packet
   */
  void SetPtr (uint32_t ptr);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:

  /**
   * \brief The incorrect packet.
   */
  Ptr<Packet> m_packet;

  /**
   * \brief The pointer field.
   */
  uint32_t m_ptr;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 MTU option.
 */
class Icmpv6OptionMtu : public Icmpv6OptionHeader
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6OptionMtu ();

  /**
   * \brief Constructor.
   * \param mtu MTU used.
   */
  Icmpv6OptionMtu (uint32_t mtu);

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6OptionMtu ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Get the reserved field.
   * \return the reserved value
   */
  uint16_t GetReserved () const;

  /**
   * \brief Set the reserved field.
   * \param reserved the reserved value
   */
  void SetReserved (uint16_t reserved);

  /**
   * \brief Get the MTU.
   * \return the MTU value
   */
  uint32_t GetMtu () const;

  /**
   * \brief Set the MTU.
   * \param mtu the MTU to set
   */
  void SetMtu (uint32_t mtu);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The reserved value
   */
  uint16_t m_reserved;

  /**
   * \brief The MTU value.
   */
  uint32_t m_mtu;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 Option Prefix Information.
 */
class Icmpv6OptionPrefixInformation : public Icmpv6OptionHeader
{
public:
  /**
   * \brief Constructor.
   */
  Icmpv6OptionPrefixInformation ();

  /**
   * \brief Constructor.
   * \param network prefix
   * \param prefixlen prefix length
   */
  Icmpv6OptionPrefixInformation (Ipv6Address network, uint8_t prefixlen);

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6OptionPrefixInformation ();

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Icmpv6 Option Prefix Information flag field values
   */
  typedef enum
  {
    NONE         = 0,   //!< No flags
    ROUTERADDR   = 32,  //!< Router Address
    AUTADDRCONF  = 64,  //!< Autonomous Address Configuration
    ONLINK       = 128  //!< On-link
  } Flags_t;

  /**
   * \brief Get the prefix length.
   * \return prefix length
   */
  uint8_t GetPrefixLength () const;

  /**
   * \brief Set the prefix length.
   * \param prefixLength the prefix length
   */
  void SetPrefixLength (uint8_t prefixLength);

  /**
   * \brief Get the flags.
   * \return the flags.
   */
  uint8_t GetFlags () const;

  /**
   * \brief Set the flags.
   * \param flags the flags to set
   */
  void SetFlags (uint8_t flags);

  /**
   * \brief Get the valid time of the information.
   * \return valid time
   */
  uint32_t GetValidTime () const;

  /**
   * \brief Set the valid time of the information.
   * \param validTime valid time
   */
  void SetValidTime (uint32_t validTime);

  /**
   * \brief Get the preferred time of the information.
   * \return preferred time
   */
  uint32_t GetPreferredTime () const;

  /**
   * \brief Set the preferred time of the information.
   * \param preferredTime preferred time
   */
  void SetPreferredTime (uint32_t preferredTime);

  /**
   * \brief Get the reserved field.
   * \return the reserved field (should be 0x00000000)
   */
  uint32_t GetReserved () const;

  /**
   * \brief Set the reserved field (normally it will be 0x00000000).
   * \param reserved reserved value
   */
  void SetReserved (uint32_t reserved);

  /**
   * \brief Get the IPv6 prefix.
   * \return IPv6 prefix
   */
  Ipv6Address GetPrefix () const;

  /**
   * \brief Set the IPv6 prefix.
   * \param prefix the IPv6 prefix
   */
  void SetPrefix (Ipv6Address prefix);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The prefix value.
   */
  Ipv6Address m_prefix;

  /**
   * \brief The length of the prefix.
   */
  uint8_t m_prefixLength;

  /**
   * \brief The flags.
   */
  uint8_t m_flags;

  /**
   * \brief The valid time.
   */
  uint32_t m_validTime;

  /**
   * \brief The preferred time.
   */
  uint32_t m_preferredTime;

  /**
   * \brief The reserved field.
   */
  uint32_t m_reserved;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 link-layer address option.
 */
class Icmpv6OptionLinkLayerAddress : public Icmpv6OptionHeader
{
public:
  /**
   * \brief Constructor.
   * \param source source hardware address or target hardware address for the option
   */
  Icmpv6OptionLinkLayerAddress (bool source);

  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId (void) const;

  /**
   * \brief Constructor.
   * \param source source hardware address or target hardware address for the option
   * \param addr hardware address
   */
  Icmpv6OptionLinkLayerAddress (bool source, Address addr);

  /**
   * \brief Constructor.
   */
  Icmpv6OptionLinkLayerAddress ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6OptionLinkLayerAddress ();

  /**
   * \brief Get the hardware address.
   * \return the hardware address
   */
  Address GetAddress () const;

  /**
   * \brief Set the hardware address.
   * \param addr the address to set
   */
  void SetAddress (Address addr);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The hardware address.
   */
  Address m_addr;
};

/**
 * \ingroup icmpv6
 *
 * \brief ICMPv6 redirected option.
 */
class Icmpv6OptionRedirected : public Icmpv6OptionHeader
{
public:
  /**
   * \brief Get the UID of this class.
   * \return UID
   */
  static TypeId GetTypeId ();

  /**
   * \brief Get the instance type ID.
   * \return instance type ID
   */
  virtual TypeId GetInstanceTypeId () const;

  /**
   * \brief Constructor.
   */
  Icmpv6OptionRedirected ();

  /**
   * \brief Destructor.
   */
  virtual ~Icmpv6OptionRedirected ();

  /**
   * \brief Set the redirected packet.
   * \param packet the redirected packet
   */
  void SetPacket (Ptr<Packet> packet);

  /**
   * \brief Print information.
   * \param os output stream
   */
  virtual void Print (std::ostream& os) const;

  /**
   * \brief Get the serialized size.
   * \return serialized size
   */
  virtual uint32_t GetSerializedSize () const;

  /**
   * \brief Serialize the packet.
   * \param start start offset
   */
  virtual void Serialize (Buffer::Iterator start) const;

  /**
   * \brief Deserialize the packet.
   * \param start start offset
   * \return length of packet
   */
  virtual uint32_t Deserialize (Buffer::Iterator start);

private:
  /**
   * \brief The redirected packet.
   */
  Ptr<Packet> m_packet;
};

} /* namespace ns3 */

#endif /* ICMPV6_HEADER_H */

