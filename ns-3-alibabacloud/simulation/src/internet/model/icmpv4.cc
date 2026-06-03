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

#include "icmpv4.h"
#include "ns3/packet.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Icmpv4Header");

/********************************************************
 *        Icmpv4Header
 ********************************************************/

NS_OBJECT_ENSURE_REGISTERED (Icmpv4Header);

TypeId 
Icmpv4Header::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Icmpv4Header")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<Icmpv4Header> ()
  ;
  return tid;
}
Icmpv4Header::Icmpv4Header ()
  : m_type (0),
    m_code (0),
    m_calcChecksum (false)
{
  NS_LOG_FUNCTION (this);
}
Icmpv4Header::~Icmpv4Header ()
{
  NS_LOG_FUNCTION (this);
}
void
Icmpv4Header::EnableChecksum (void)
{
  NS_LOG_FUNCTION (this);
  m_calcChecksum = true;
}
TypeId
Icmpv4Header::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}
uint32_t
Icmpv4Header::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 4;
}
void
Icmpv4Header::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  i.WriteU8 (m_type);
  i.WriteU8 (m_code);
  i.WriteHtonU16 (0);
  if (m_calcChecksum)
    {
      i = start;
      uint16_t checksum = i.CalculateIpChecksum (i.GetSize ());
      i = start;
      i.Next (2);
      i.WriteU16 (checksum);
    }

}
uint32_t 
Icmpv4Header::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  m_type = start.ReadU8 ();
  m_code = start.ReadU8 ();
  start.Next (2); // uint16_t checksum = start.ReadNtohU16 ();
  return 4;
}
void 
Icmpv4Header::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "type=" << (uint32_t)m_type << ", code=" << (uint32_t)m_code;
}

void 
Icmpv4Header::SetType (uint8_t type)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (type));
  m_type = type;
}
void 
Icmpv4Header::SetCode (uint8_t code)
{
  NS_LOG_FUNCTION (this << static_cast<uint32_t> (code));
  m_code = code;
}
uint8_t 
Icmpv4Header::GetType (void) const
{
  NS_LOG_FUNCTION (this);
  return m_type;
}
uint8_t 
Icmpv4Header::GetCode (void) const
{
  NS_LOG_FUNCTION (this); 
  return m_code;
}

/********************************************************
 *        Icmpv4Echo
 ********************************************************/

NS_OBJECT_ENSURE_REGISTERED (Icmpv4Echo);

void 
Icmpv4Echo::SetIdentifier (uint16_t id)
{
  NS_LOG_FUNCTION (this << id); 
  m_identifier = id;
}
void 
Icmpv4Echo::SetSequenceNumber (uint16_t seq)
{
  NS_LOG_FUNCTION (this << seq);
  m_sequence = seq;
}
void 
Icmpv4Echo::SetData (Ptr<const Packet> data)
{
  NS_LOG_FUNCTION (this << *data);
  
  uint32_t size = data->GetSize ();
  //
  // All kinds of optimizations are possible, but let's not get carried away
  // since this is probably a very uncommon thing in the big picture.
  //
  // N.B. Zero is a legal size for the alloc below even though a hardcoded zero
  // would result in  warning.
  //
  if (size != m_dataSize)
    {
      delete [] m_data;
      m_data = new uint8_t[size];
      m_dataSize = size;
    }
  data->CopyData (m_data, size);
}
uint16_t 
Icmpv4Echo::GetIdentifier (void) const
{
  NS_LOG_FUNCTION (this);
  return m_identifier;
}
uint16_t 
Icmpv4Echo::GetSequenceNumber (void) const
{
  NS_LOG_FUNCTION (this);
  return m_sequence;
}
uint32_t
Icmpv4Echo::GetDataSize (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dataSize;
}
uint32_t
Icmpv4Echo::GetData (uint8_t payload[]) const
{
  NS_LOG_FUNCTION (this << payload);
  memcpy (payload, m_data, m_dataSize);
  return m_dataSize;
}
TypeId 
Icmpv4Echo::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Icmpv4Echo")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<Icmpv4Echo> ()
  ;
  return tid;
}
Icmpv4Echo::Icmpv4Echo ()
  : m_identifier (0),
    m_sequence (0),
    m_dataSize (0)
{
  NS_LOG_FUNCTION (this);
  //
  // After construction, m_data is always valid until destruction.  This is true
  // even if m_dataSize is zero.
  //
  m_data = new uint8_t[m_dataSize];
}
Icmpv4Echo::~Icmpv4Echo ()
{
  NS_LOG_FUNCTION (this);
  delete [] m_data;
  m_data = 0;
  m_dataSize = 0;
}
TypeId 
Icmpv4Echo::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}
uint32_t 
Icmpv4Echo::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 4 + m_dataSize;
}
void 
Icmpv4Echo::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.WriteHtonU16 (m_identifier);
  start.WriteHtonU16 (m_sequence);
  start.Write (m_data, m_dataSize);
}
uint32_t 
Icmpv4Echo::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);

  uint32_t optionalPayloadSize = start.GetRemainingSize () -4;
  NS_ASSERT (start.GetRemainingSize () >= 4);

  m_identifier = start.ReadNtohU16 ();
  m_sequence = start.ReadNtohU16 ();
  if (optionalPayloadSize != m_dataSize)
    {
      delete [] m_data;
      m_dataSize = optionalPayloadSize;
      m_data = new uint8_t[m_dataSize];
    }
  start.Read (m_data, m_dataSize);
  return m_dataSize+4;
}
void 
Icmpv4Echo::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  os << "identifier=" << m_identifier << ", sequence="  << m_sequence << ", data size=" << m_dataSize;
}


/********************************************************
 *        Icmpv4DestinationUnreachable
 ********************************************************/

NS_OBJECT_ENSURE_REGISTERED (Icmpv4DestinationUnreachable);

TypeId 
Icmpv4DestinationUnreachable::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Icmpv4DestinationUnreachable")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<Icmpv4DestinationUnreachable> ()
  ;
  return tid;
}
Icmpv4DestinationUnreachable::Icmpv4DestinationUnreachable ()
{
  NS_LOG_FUNCTION (this);
  // make sure that thing is initialized to get initialized bytes
  // when the ip payload's size is smaller than 8 bytes.
  for (uint8_t j = 0; j < 8; j++)
    {
      m_data[j] = 0;
    }
}

void 
Icmpv4DestinationUnreachable::SetNextHopMtu (uint16_t mtu)
{
  NS_LOG_FUNCTION (this << mtu);
  m_nextHopMtu = mtu;
}
uint16_t 
Icmpv4DestinationUnreachable::GetNextHopMtu (void) const
{
  NS_LOG_FUNCTION (this);
  return m_nextHopMtu;
}

void 
Icmpv4DestinationUnreachable::SetData (Ptr<const Packet> data)
{
  NS_LOG_FUNCTION (this << *data);
  data->CopyData (m_data, 8);
}
void 
Icmpv4DestinationUnreachable::SetHeader (Ipv4Header header)
{
  NS_LOG_FUNCTION (this << header);
  m_header = header;
}
void 
Icmpv4DestinationUnreachable::GetData (uint8_t payload[8]) const
{
  NS_LOG_FUNCTION (this << payload);
  memcpy (payload, m_data, 8);
}
Ipv4Header 
Icmpv4DestinationUnreachable::GetHeader (void) const
{
  NS_LOG_FUNCTION (this);
  return m_header;
}


Icmpv4DestinationUnreachable::~Icmpv4DestinationUnreachable ()
{
}
TypeId
Icmpv4DestinationUnreachable::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}
uint32_t
Icmpv4DestinationUnreachable::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 4 + m_header.GetSerializedSize () + 8;
}
void
Icmpv4DestinationUnreachable::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.WriteU16 (0);
  start.WriteHtonU16 (m_nextHopMtu);
  uint32_t size = m_header.GetSerializedSize ();
  m_header.Serialize (start);
  start.Next (size);
  start.Write (m_data, 8);
}

uint32_t 
Icmpv4DestinationUnreachable::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  i.Next (2);
  m_nextHopMtu = i.ReadNtohU16 ();
  uint32_t read = m_header.Deserialize (i);
  i.Next (read);
  for (uint8_t j = 0; j < 8; j++)
    {
      m_data[j] = i.ReadU8 ();
    }
  return i.GetDistanceFrom (start);
}
void 
Icmpv4DestinationUnreachable::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  m_header.Print (os);
  os << " org data=";
  for (uint8_t i = 0; i < 8; i++)
    {
      os << (uint32_t) m_data[i];
      if (i != 8)
        {
          os << " ";
        }
    }
}

/********************************************************
 *        Icmpv4TimeExceeded
 ********************************************************/

NS_OBJECT_ENSURE_REGISTERED (Icmpv4TimeExceeded);

TypeId 
Icmpv4TimeExceeded::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Icmpv4TimeExceeded")
    .SetParent<Header> ()
    .SetGroupName ("Internet")
    .AddConstructor<Icmpv4TimeExceeded> ()
  ;
  return tid;
}
Icmpv4TimeExceeded::Icmpv4TimeExceeded ()
{
  NS_LOG_FUNCTION (this);
  // make sure that thing is initialized to get initialized bytes
  // when the ip payload's size is smaller than 8 bytes.
  for (uint8_t j = 0; j < 8; j++)
    {
      m_data[j] = 0;
    }
}


void 
Icmpv4TimeExceeded::SetData (Ptr<const Packet> data)
{
  NS_LOG_FUNCTION (this << *data);
  data->CopyData (m_data, 8);
}
void 
Icmpv4TimeExceeded::SetHeader (Ipv4Header header)
{
  NS_LOG_FUNCTION (this << header);
  m_header = header;
}
void 
Icmpv4TimeExceeded::GetData (uint8_t payload[8]) const
{
  NS_LOG_FUNCTION (this << payload);
  memcpy (payload, m_data, 8);
}
Ipv4Header 
Icmpv4TimeExceeded::GetHeader (void) const
{
  NS_LOG_FUNCTION (this);
  return m_header;
}


Icmpv4TimeExceeded::~Icmpv4TimeExceeded ()
{
  NS_LOG_FUNCTION (this);
}
TypeId
Icmpv4TimeExceeded::GetInstanceTypeId (void) const
{
  NS_LOG_FUNCTION (this);
  return GetTypeId ();
}
uint32_t
Icmpv4TimeExceeded::GetSerializedSize (void) const
{
  NS_LOG_FUNCTION (this);
  return 4 + m_header.GetSerializedSize () + 8;
}
void
Icmpv4TimeExceeded::Serialize (Buffer::Iterator start) const
{
  NS_LOG_FUNCTION (this << &start);
  start.WriteU32 (0);
  uint32_t size = m_header.GetSerializedSize ();
  m_header.Serialize (start);
  start.Next (size);
  start.Write (m_data, 8);
}

uint32_t 
Icmpv4TimeExceeded::Deserialize (Buffer::Iterator start)
{
  NS_LOG_FUNCTION (this << &start);
  Buffer::Iterator i = start;
  i.Next (4);
  uint32_t read = m_header.Deserialize (i);
  i.Next (read);
  for (uint8_t j = 0; j < 8; j++)
    {
      m_data[j] = i.ReadU8 ();
    }
  return i.GetDistanceFrom (start);
}
void 
Icmpv4TimeExceeded::Print (std::ostream &os) const
{
  NS_LOG_FUNCTION (this << &os);
  m_header.Print (os);
  os << " org data=";
  for (uint8_t i = 0; i < 8; i++)
    {
      os << (uint32_t) m_data[i];
      if (i != 8)
        {
          os << " ";
        }
    }
}

} // namespace ns3
