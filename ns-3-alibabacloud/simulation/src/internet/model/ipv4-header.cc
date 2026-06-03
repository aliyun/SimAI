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

#include "ns3/assert.h"
#include "ns3/abort.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "ipv4-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4Header");

NS_OBJECT_ENSURE_REGISTERED (Ipv4Header);

Ipv4Header::Ipv4Header ()
  : m_calcChecksum (false),
    m_payloadSize (0),
    m_identification (0),
    m_tos (0),
    m_ttl (0),
    m_protocol (0),
    m_flags (0),
    m_fragmentOffset (0),
    m_checksum (0),
    m_goodChecksum (true),
    m_headerSize(5*4)
{
}

void
Ipv4Header::EnableChecksum (void)
{
  NS_LOG_FUNCTION (this);
  m_calcChecksum = true;
}

void
Ipv4Header::SetPayloadSize (uint16_t size)
{
  NS_LOG_FUNCTION (this << size);
  m_payloadSize = size;
}
uint16_t
Ipv4Header::GetPayloadSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_payloadSize;
}

uint16_t
Ipv4Header::GetIdentification (void) const
{
  NS_LOG_FUNCTION (this);
  return m_identification;
}
void
Ipv4Header::SetIdentification (uint16_t identification)
{
  NS_LOG_FUNCTION (this << identification);
  m_identification = identification;
}

void 
Ipv4Header::SetTos (uint8_t tos)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (tos));
  m_tos = tos;
}

void
Ipv4Header::SetDscp (DscpType dscp)
{
  NS_LOG_FUNCTION (this << dscp);
  m_tos &= 0x3; // Clear out the DSCP part, retain 2 bits of ECN
  m_tos |= (dscp << 2);
}

void
Ipv4Header::SetEcn (EcnType ecn)
{
  NS_LOG_FUNCTION (this << ecn);
  m_tos &= 0xFC; // Clear out the ECN part, retain 6 bits of DSCP
  m_tos |= ecn;
}

Ipv4Header::DscpType 
Ipv4Header::GetDscp (void) const
{
  NS_LOG_FUNCTION (this);
  // Extract only first 6 bits of TOS byte, i.e 0xFC
  return DscpType ((m_tos & 0xFC) >> 2);
}

std::string 
Ipv4Header::DscpTypeToString (DscpType dscp) const
{
  NS_LOG_FUNCTION (this << dscp);
  switch (dscp)
    {
      case DscpDefault:
        return "Default";
      case DSCP_CS1:
        return "CS1";
      case DSCP_AF11:
        return "AF11";
      case DSCP_AF12:
        return "AF12";
      case DSCP_AF13:
        return "AF13";
      case DSCP_CS2:
        return "CS2";
      case DSCP_AF21:
        return "AF21";
      case DSCP_AF22:
        return "AF22";
      case DSCP_AF23:
        return "AF23";
      case DSCP_CS3:
        return "CS3";
      case DSCP_AF31:
        return "AF31";
      case DSCP_AF32:
        return "AF32";
      case DSCP_AF33:
        return "AF33";
      case DSCP_CS4:
        return "CS4";
      case DSCP_AF41:
        return "AF41";
      case DSCP_AF42:
        return "AF42";
      case DSCP_AF43:
        return "AF43";
      case DSCP_CS5:
        return "CS5";
      case DSCP_EF:
        return "EF";
      case DSCP_CS6:
        return "CS6";
      case DSCP_CS7:
        return "CS7";
      default:
        return "Unrecognized DSCP";
    };
}


Ipv4Header::EcnType 
Ipv4Header::GetEcn (void) const
{
  NS_LOG_FUNCTION (this);
  // Extract only last 2 bits of TOS byte, i.e 0x3
  return EcnType (m_tos & 0x3);
}

std::string 
Ipv4Header::EcnTypeToString (EcnType ecn) const
{
  NS_LOG_FUNCTION (this << ecn);
  switch (ecn)
    {
      case ECN_NotECT:
        return "Not-ECT";
      case ECN_ECT1:
        return "ECT (1)";
      case ECN_ECT0:
        return "ECT (0)";
      case ECN_CE:
        return "CE";      
      default:
        return "Unknown ECN";
    };
}

uint8_t 
Ipv4Header::GetTos (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tos;
}
void 
Ipv4Header::SetMoreFragments (void)
{
  NS_LOG_FUNCTION (this);
  m_flags |= MORE_FRAGMENTS;
}
void
Ipv4Header::SetLastFragment (void)
{
  NS_LOG_FUNCTION (this);
  m_flags &= ~MORE_FRAGMENTS;
}
bool 
Ipv4Header::IsLastFragment (void) const
{
  NS_LOG_FUNCTION (this);
  return !(m_flags & MORE_FRAGMENTS);
}

void 
Ipv4Header::SetDontFragment (void)
{
  NS_LOG_FUNCTION (this);
  m_flags |= DONT_FRAGMENT;
}
void 
Ipv4Header::SetMayFragment (void)
{
  NS_LOG_FUNCTION (this);
  m_flags &= ~DONT_FRAGMENT;
}
bool 
Ipv4Header::IsDontFragment (void) const
{
  NS_LOG_FUNCTION (this);
  return (m_flags & DONT_FRAGMENT);
}

void 
Ipv4Header::SetFragmentOffset (uint16_t offsetBytes)
{
  NS_LOG_FUNCTION (this << offsetBytes);
  // check if the user is trying to set an invalid offset
  NS_ABORT_MSG_IF ((offsetBytes & 0x7), "offsetBytes must be multiple of 8 bytes");
  m_fragmentOffset = offsetBytes;
}
uint16_t 
Ipv4Header::GetFragmentOffset (void) const
{
  NS_LOG_FUNCTION (this);
  // -fstrict-overflow sensitive, see bug 1868
  if ( m_fragmentOffset + m_payloadSize > 65535 - 5*4 )
    {
      NS_LOG_WARN("Fragment will exceed the maximum packet size once reassembled");
    }

  return m_fragmentOffset;
}

void 
Ipv4Header::SetTtl (uint8_t ttl)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (ttl));
  m_ttl = ttl;
}
uint8_t 
Ipv4Header::GetTtl (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ttl;
}

uint8_t 
Ipv4Header::GetProtocol (void) const
{
  NS_LOG_FUNCTION (this);
  return m_protocol;
}
void 
Ipv4Header::SetProtocol (uint8_t protocol)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (protocol));
  m_protocol = protocol;
}

void 
Ipv4Header::SetSource (Ipv4Address source)
{
  NS_LOG_FUNCTION (this << source);
  m_source = source;
}
Ipv4Address
Ipv4Header::GetSource (void) const
{
  NS_LOG_FUNCTION (this);
  return m_source;
}

void 
Ipv4Header::SetDestination (Ipv4Address dst)
{
  NS_LOG_FUNCTION (this << dst);
  m_destination = dst;
}
Ipv4Address
Ipv4Header::GetDestination (void) const
{
  NS_LOG_FUNCTION (this);
  return m_destination;
}


bool
Ipv4Header::IsChecksumOk (void) const
{
  NS_LOG_FUNCTION (this);
  return m_goodChecksum;
}

TypeId 
Ipv4Header::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv4Header")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv4Header> ()
  ;
  return tid;
}
TypeId 
Ipv4Header::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}
void 
Ipv4Header::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  // ipv4, right ?
  std::string flags;
  if (m_flags == 0)
    {
      flags = "none";
    }
  else if ((m_flags & MORE_FRAGMENTS) &&
           (m_flags & DONT_FRAGMENT))
    {
      flags = "MF|DF";
    }
  else if (m_flags & DONT_FRAGMENT)
    {
      flags = "DF";
    }
  else if (m_flags & MORE_FRAGMENTS)
    {
      flags = "MF";
    }
  else
    {
      flags = "XX";
    }
  os << "tos 0x" << std::hex << m_tos << std::dec << " "
     << "DSCP " << DscpTypeToString (GetDscp ()) << " "
     << "ECN " << EcnTypeToString (GetEcn ()) << " "
     << "ttl " << m_ttl << " "
     << "id " << m_identification << " "
     << "protocol " << m_protocol << " "
     << "offset (bytes) " << m_fragmentOffset << " "
     << "flags [" << flags << "] "
     << "length: " << (m_payloadSize + 5 * 4)
     << " " 
     << m_source << " > " << m_destination
  ;
}
uint32_t 
Ipv4Header::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  //return 5 * 4;
	return m_headerSize;
}

void
Ipv4Header::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;

  uint8_t verIhl = (4 << 4) | (5);
  i.WriteU8 (verIhl);
  i.WriteU8 (m_tos);
  i.WriteHtonU16 (m_payloadSize + 5*4);
  i.WriteHtonU16 (m_identification);
  uint32_t fragmentOffset = m_fragmentOffset / 8;
  uint8_t flagsFrag = (fragmentOffset >> 8) & 0x1f;
  if (m_flags & DONT_FRAGMENT) 
    {
      flagsFrag |= (1<<6);
    }
  if (m_flags & MORE_FRAGMENTS) 
    {
      flagsFrag |= (1<<5);
    }
  i.WriteU8 (flagsFrag);
  uint8_t frag = fragmentOffset & 0xff;
  i.WriteU8 (frag);
  i.WriteU8 (m_ttl);
  i.WriteU8 (m_protocol);
  i.WriteHtonU16 (0);
  i.WriteHtonU32 (m_source.Get ());
  i.WriteHtonU32 (m_destination.Get ());

  if (m_calcChecksum) 
    {
      i = start;
      uint16_t checksum = i.CalculateIpChecksum (20);
      NS_LOG_LOGIC ("checksum=" <<checksum);
      i = start;
      i.Next (10);
      i.WriteU16 (checksum);
    }
}
uint32_t
Ipv4Header::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;

  uint8_t verIhl = i.ReadU8 ();
  uint8_t ihl = verIhl & 0x0f; 
  uint16_t headerSize = ihl * 4;

  if ((verIhl >> 4) != 4)
    {
      NS_LOG_WARN ("Trying to decode a non-IPv4 header, refusing to do it.");
      return 0;
    }

  m_tos = i.ReadU8 ();
  uint16_t size = i.ReadNtohU16 ();
  m_payloadSize = size - headerSize;
  m_identification = i.ReadNtohU16 ();
  uint8_t flags = i.ReadU8 ();
  m_flags = 0;
  if (flags & (1<<6)) 
    {
      m_flags |= DONT_FRAGMENT;
    }
  if (flags & (1<<5)) 
    {
      m_flags |= MORE_FRAGMENTS;
    }
  i.Prev ();
  m_fragmentOffset = i.ReadU8 () & 0x1f;
  m_fragmentOffset <<= 8;
  m_fragmentOffset |= i.ReadU8 ();
  m_fragmentOffset <<= 3;
  m_ttl = i.ReadU8 ();
  m_protocol = i.ReadU8 ();
  m_checksum = i.ReadU16 ();
  /* i.Next (2); // checksum */
  m_source.Set (i.ReadNtohU32 ());
  m_destination.Set (i.ReadNtohU32 ());
  m_headerSize = headerSize;

  if (m_calcChecksum) 
    {
      i = start;
      uint16_t checksum = i.CalculateIpChecksum (headerSize);
      NS_LOG_LOGIC ("checksum=" <<checksum);

      m_goodChecksum = (checksum == 0);
    }
  return GetSerializedSize ();
}

} // namespace ns3
