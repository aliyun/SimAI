/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2008 Louis Pasteur University
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
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"

#include "ns3/address-utils.h"
#include "ipv6-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6Header");

NS_OBJECT_ENSURE_REGISTERED (Ipv6Header);

Ipv6Header::Ipv6Header ()
  : m_trafficClass (0),
    m_flowLabel (1),
    m_payloadLength (0),
    m_nextHeader (0),
    m_hopLimit (0)
{
  SetSource (Ipv6Address ("::"));
  SetDestination (Ipv6Address ("::"));
}

void Ipv6Header::SetTrafficClass (uint8_t traffic)
{
  m_trafficClass = traffic;
}

uint8_t Ipv6Header::GetTrafficClass () const
{
  return m_trafficClass;
}

void Ipv6Header::SetFlowLabel (uint32_t flow)
{
  m_flowLabel = flow;
}

uint32_t Ipv6Header::GetFlowLabel () const
{
  return m_flowLabel;
}

void Ipv6Header::SetPayloadLength (uint16_t len)
{
  m_payloadLength = len;
}

uint16_t Ipv6Header::GetPayloadLength () const
{
  return m_payloadLength;
}

void Ipv6Header::SetNextHeader (uint8_t next)
{
  m_nextHeader = next;
}

uint8_t Ipv6Header::GetNextHeader () const
{
  return m_nextHeader;
}

void Ipv6Header::SetHopLimit (uint8_t limit)
{
  m_hopLimit = limit;
}

uint8_t Ipv6Header::GetHopLimit () const
{
  return m_hopLimit;
}

void Ipv6Header::SetSource (Ipv6Address src)
{
  m_sourceAddress = src;
}

void Ipv6Header::SetSourceAddress (Ipv6Address src)
{
  SetSource (src);
}

Ipv6Address Ipv6Header::GetSource () const
{
  return m_sourceAddress;
}

Ipv6Address Ipv6Header::GetSourceAddress () const
{
  return GetSource ();
}

void Ipv6Header::SetDestination (Ipv6Address dst)
{
  m_destinationAddress = dst;
}

void Ipv6Header::SetDestinationAddress (Ipv6Address dst)
{
  SetDestination (dst);
}

Ipv6Address Ipv6Header::GetDestination () const
{
  return m_destinationAddress;
}

Ipv6Address Ipv6Header::GetDestinationAddress () const
{
  return GetDestination ();
}

TypeId Ipv6Header::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Ipv6Header")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<Ipv6Header> ()
  ;
  return tid;
}

TypeId Ipv6Header::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void Ipv6Header::Print (std::ostream& os) const
{
  os << "(Version 6 "
     << "Traffic class 0x" << std::hex << m_trafficClass << std::dec << " "
     << "DSCP " << DscpTypeToString (GetDscp ()) << " "
     << "Flow Label 0x" << std::hex << m_flowLabel << std::dec << " "
     << "Payload Length " << m_payloadLength << " "
     << "Next Header " << std::dec << (uint32_t) m_nextHeader << " "
     << "Hop Limit " << std::dec << (uint32_t)m_hopLimit << " )"
     <<  m_sourceAddress << " > " <<  m_destinationAddress
  ;
}

uint32_t Ipv6Header::GetSerializedSize () const
{
  return 10 * 4;
}

void Ipv6Header::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  uint32_t vTcFl = 0; /* version, Traffic Class and Flow Label fields */

  vTcFl = (6 << 28) | (m_trafficClass << 20) | (m_flowLabel);

  i.WriteHtonU32 (vTcFl);
  i.WriteHtonU16 (m_payloadLength);
  i.WriteU8 (m_nextHeader);
  i.WriteU8 (m_hopLimit);

  WriteTo (i, m_sourceAddress);
  WriteTo (i, m_destinationAddress);
}

uint32_t Ipv6Header::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint32_t vTcFl = 0;

  vTcFl = i.ReadNtohU32 ();
  if ((vTcFl >> 28) != 6)
    {
      NS_LOG_WARN ("Trying to decode a non-IPv6 header, refusing to do it.");
      return 0;
    }

  m_trafficClass = (uint8_t)((vTcFl >> 20) & 0x000000ff);
  m_flowLabel = vTcFl & 0xfffff;
  m_payloadLength = i.ReadNtohU16 ();
  m_nextHeader = i.ReadU8 ();
  m_hopLimit = i.ReadU8 ();

  ReadFrom (i, m_sourceAddress);
  ReadFrom (i, m_destinationAddress);

  return GetSerializedSize ();
}

void Ipv6Header::SetDscp (DscpType dscp)
{
  NS_LOG_FUNCTION (this << dscp);
  m_trafficClass &= 0x3; // Clear out the DSCP part, retain 2 bits of ECN
  m_trafficClass |= (dscp << 2);
}

void Ipv6Header::SetEcn (EcnType ecn)
{
  NS_LOG_FUNCTION (this << ecn);
  m_trafficClass &= 0xFC; // Clear out the ECN part, retain 6 bits of DSCP
  m_trafficClass |= ecn;
}

Ipv6Header::DscpType Ipv6Header::GetDscp (void) const
{
  NS_LOG_FUNCTION (this);
  // Extract only first 6 bits of TOS byte, i.e 0xFC
  return DscpType ((m_trafficClass & 0xFC) >> 2);
}

std::string Ipv6Header::DscpTypeToString (DscpType dscp) const
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

Ipv6Header::EcnType
Ipv6Header::GetEcn (void) const
{
  NS_LOG_FUNCTION (this);
  // Extract only last 2 bits of Traffic Class byte, i.e 0x3
  return EcnType (m_trafficClass & 0x3);
}

std::string Ipv6Header::EcnTypeToString (EcnType ecn) const
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
        return "Unknown ECN codepoint";
    };
}

} /* namespace ns3 */

