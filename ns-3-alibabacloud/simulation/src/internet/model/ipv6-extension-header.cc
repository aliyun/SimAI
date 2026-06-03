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
 * Author: David Gross <gdavid.devel@gmail.com>
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "ipv6-extension-header.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("Ipv6ExtensionHeader");

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionHeader);

TypeId Ipv6ExtensionHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionHeader")
    .AddConstructor<Ipv6ExtensionHeader> ()
    .SetParent<Header> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionHeader::Ipv6ExtensionHeader ()
  : m_length (0),
    m_nextHeader (0),
    m_data (0)
{
}

Ipv6ExtensionHeader::~Ipv6ExtensionHeader ()
{
}

void Ipv6ExtensionHeader::SetNextHeader (uint8_t nextHeader)
{
  m_nextHeader = nextHeader;
}

uint8_t Ipv6ExtensionHeader::GetNextHeader () const
{
  return m_nextHeader;
}

void Ipv6ExtensionHeader::SetLength (uint16_t length)
{
  NS_ASSERT_MSG (!(length & 0x7), "Invalid Ipv6ExtensionHeader Length, must be a multiple of 8 bytes.");
  NS_ASSERT_MSG (length > 0, "Invalid Ipv6ExtensionHeader Length, must be greater than 0.");
  NS_ASSERT_MSG (length < 2048, "Invalid Ipv6ExtensionHeader Length, must be a lower than 2048.");

  m_length = (length >> 3) - 1;
}

uint16_t Ipv6ExtensionHeader::GetLength () const
{
  return (m_length + 1) << 3;
}

void Ipv6ExtensionHeader::Print (std::ostream &os) const
{
  os << "( nextHeader = " << (uint32_t)GetNextHeader () << " length = " << (uint32_t)GetLength () << " )";
}

uint32_t Ipv6ExtensionHeader::GetSerializedSize () const
{
  return 2;
}

void Ipv6ExtensionHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (m_nextHeader);
  i.WriteU8 (m_length);
  i.Write (m_data.PeekData (), m_data.GetSize ());
}

uint32_t Ipv6ExtensionHeader::Deserialize (Buffer::Iterator start) 
{
  Buffer::Iterator i = start;

  m_nextHeader = i.ReadU8 ();
  m_length = i.ReadU8 ();

  uint32_t dataLength = GetLength () - 2;
  uint8_t* data = new uint8_t[dataLength];
  i.Read (data, dataLength);

  if (dataLength > m_data.GetSize ())
    {
      m_data.AddAtEnd (dataLength - m_data.GetSize ());
    }
  else 
    {
      m_data.RemoveAtEnd (m_data.GetSize () - dataLength);
    }

  i = m_data.Begin ();
  i.Write (data, dataLength);

  delete[] data;
  return GetSerializedSize ();
}

OptionField::OptionField (uint32_t optionsOffset)
  : m_optionData (0),
    m_optionsOffset (optionsOffset)
{
}

OptionField::~OptionField ()
{
}

uint32_t OptionField::GetSerializedSize () const
{
  return m_optionData.GetSize () + CalculatePad ((Ipv6OptionHeader::Alignment) { 8,0});
}

void OptionField::Serialize (Buffer::Iterator start) const
{
  start.Write (m_optionData.Begin (), m_optionData.End ());
  uint32_t fill = CalculatePad ((Ipv6OptionHeader::Alignment) { 8,0});
  NS_LOG_LOGIC ("fill with " << fill << " bytes padding");
  switch (fill)
    {
    case 0: return;
    case 1: Ipv6OptionPad1Header ().Serialize (start);
      return;
    default: Ipv6OptionPadnHeader (fill).Serialize (start);
      return;
    }
}

uint32_t OptionField::Deserialize (Buffer::Iterator start, uint32_t length)
{
  uint8_t* buf = new uint8_t[length];
  start.Read (buf, length);
  m_optionData = Buffer ();
  m_optionData.AddAtEnd (length);
  m_optionData.Begin ().Write (buf, length);
  delete[] buf;
  return length;
}

void OptionField::AddOption (Ipv6OptionHeader const& option)
{
  NS_LOG_FUNCTION (option);

  uint32_t pad = CalculatePad (option.GetAlignment ());
  NS_LOG_LOGIC ("need " << pad << " bytes padding");
  switch (pad)
    {
    case 0: break; //no padding needed
    case 1: AddOption (Ipv6OptionPad1Header ());
      break;
    default: AddOption (Ipv6OptionPadnHeader (pad));
      break;
    }

  m_optionData.AddAtEnd (option.GetSerializedSize ());
  Buffer::Iterator it = m_optionData.End ();
  it.Prev (option.GetSerializedSize ());
  option.Serialize (it);
}

uint32_t OptionField::CalculatePad (Ipv6OptionHeader::Alignment alignment) const
{
  return (alignment.offset - (m_optionData.GetSize () + m_optionsOffset)) % alignment.factor;
}

uint32_t OptionField::GetOptionsOffset ()
{
  return m_optionsOffset;
}

Buffer OptionField::GetOptionBuffer ()
{
  return m_optionData;
}


NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionHopByHopHeader);

TypeId Ipv6ExtensionHopByHopHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionHopByHopHeader")
    .AddConstructor<Ipv6ExtensionHopByHopHeader> ()
    .SetParent<Ipv6ExtensionHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionHopByHopHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionHopByHopHeader::Ipv6ExtensionHopByHopHeader ()
  : OptionField (2)
{
}

Ipv6ExtensionHopByHopHeader::~Ipv6ExtensionHopByHopHeader ()
{
}

void Ipv6ExtensionHopByHopHeader::Print (std::ostream &os) const
{
  os << "( nextHeader = " << (uint32_t)GetNextHeader () << " length = " << (uint32_t)GetLength () << " )";
}

uint32_t Ipv6ExtensionHopByHopHeader::GetSerializedSize () const
{
  return 2 + OptionField::GetSerializedSize ();
}

void Ipv6ExtensionHopByHopHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetNextHeader ());
  i.WriteU8 ((GetSerializedSize () >> 3) - 1);
  OptionField::Serialize (i);
}

uint32_t Ipv6ExtensionHopByHopHeader::Deserialize (Buffer::Iterator start) 
{
  Buffer::Iterator i = start;

  SetNextHeader (i.ReadU8 ());
  m_length = i.ReadU8 ();
  OptionField::Deserialize (i, GetLength () - 2);

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionDestinationHeader);

TypeId Ipv6ExtensionDestinationHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionDestinationHeader")
    .AddConstructor<Ipv6ExtensionDestinationHeader> ()
    .SetParent<Ipv6ExtensionHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionDestinationHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionDestinationHeader::Ipv6ExtensionDestinationHeader ()
  : OptionField (2)
{
}

Ipv6ExtensionDestinationHeader::~Ipv6ExtensionDestinationHeader ()
{
}

void Ipv6ExtensionDestinationHeader::Print (std::ostream &os) const
{
  os << "( nextHeader = " << (uint32_t)GetNextHeader () << " length = " << (uint32_t)GetLength () << " )";
}

uint32_t Ipv6ExtensionDestinationHeader::GetSerializedSize () const
{
  return 2 + OptionField::GetSerializedSize ();
}

void Ipv6ExtensionDestinationHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetNextHeader ());
  i.WriteU8 ((GetSerializedSize () >> 3) - 1);
  OptionField::Serialize (i);
}

uint32_t Ipv6ExtensionDestinationHeader::Deserialize (Buffer::Iterator start) 
{
  Buffer::Iterator i = start;

  SetNextHeader (i.ReadU8 ());
  m_length = i.ReadU8 ();
  OptionField::Deserialize (i, GetLength () - 2);

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionFragmentHeader);

TypeId Ipv6ExtensionFragmentHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionFragmentHeader")
    .AddConstructor<Ipv6ExtensionFragmentHeader> ()
    .SetParent<Ipv6ExtensionHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionFragmentHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionFragmentHeader::Ipv6ExtensionFragmentHeader ()
  : m_offset (0),
    m_identification (0)
{
  m_length = 0;
}

Ipv6ExtensionFragmentHeader::~Ipv6ExtensionFragmentHeader ()
{
}

void Ipv6ExtensionFragmentHeader::SetOffset (uint16_t offset)
{
  // Clear the offset, and save the MF bit
  m_offset &= 1;
  m_offset |= offset & (~7);
}

uint16_t Ipv6ExtensionFragmentHeader::GetOffset () const 
{
  return m_offset & (~1);
}

void Ipv6ExtensionFragmentHeader::SetMoreFragment (bool moreFragment)
{
  m_offset = moreFragment ? m_offset | 1 : m_offset & (~1);
}

bool Ipv6ExtensionFragmentHeader::GetMoreFragment () const
{
  return m_offset & 1;
}

void Ipv6ExtensionFragmentHeader::SetIdentification (uint32_t identification)
{
  m_identification = identification;
}

uint32_t Ipv6ExtensionFragmentHeader::GetIdentification () const
{
  return m_identification;
}

void Ipv6ExtensionFragmentHeader::Print (std::ostream &os) const
{
  os << "( nextHeader = " << (uint32_t)GetNextHeader () << " length = " << (uint32_t)GetLength () 
     << " offset = " << (uint32_t)GetOffset () << " MF = " << (uint32_t)GetMoreFragment () << " identification = " << (uint32_t)m_identification << " )";
}

uint32_t Ipv6ExtensionFragmentHeader::GetSerializedSize () const
{
  return 8;
}

void Ipv6ExtensionFragmentHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetNextHeader ());
  // Fragment header does not carry an extension length
  i.WriteU8 (0);
  i.WriteHtonU16 (m_offset);
  i.WriteHtonU32 (m_identification);
}

uint32_t Ipv6ExtensionFragmentHeader::Deserialize (Buffer::Iterator start) 
{
  Buffer::Iterator i = start;

  SetNextHeader (i.ReadU8 ());
  // Fragment header does not carry an extension length
  i.ReadU8 ();
  m_offset = i.ReadNtohU16 ();
  m_identification = i.ReadNtohU32 ();

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionRoutingHeader);

TypeId Ipv6ExtensionRoutingHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionRoutingHeader")
    .AddConstructor<Ipv6ExtensionRoutingHeader> ()
    .SetParent<Ipv6ExtensionHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionRoutingHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionRoutingHeader::Ipv6ExtensionRoutingHeader ()
  : m_typeRouting (0),
    m_segmentsLeft (0)
{
}

Ipv6ExtensionRoutingHeader::~Ipv6ExtensionRoutingHeader ()
{
}

void Ipv6ExtensionRoutingHeader::SetTypeRouting (uint8_t typeRouting) 
{
  m_typeRouting = typeRouting;
}

uint8_t Ipv6ExtensionRoutingHeader::GetTypeRouting () const
{
  return m_typeRouting;
}

void Ipv6ExtensionRoutingHeader::SetSegmentsLeft (uint8_t segmentsLeft)
{
  m_segmentsLeft = segmentsLeft;
}

uint8_t Ipv6ExtensionRoutingHeader::GetSegmentsLeft () const
{
  return m_segmentsLeft;
}

void Ipv6ExtensionRoutingHeader::Print (std::ostream &os) const
{
  os << "( nextHeader = " << (uint32_t)GetNextHeader () << " length = " << (uint32_t)GetLength () 
     << " typeRouting = " << (uint32_t)m_typeRouting << " segmentsLeft = " << (uint32_t)m_segmentsLeft << " )";
}

uint32_t Ipv6ExtensionRoutingHeader::GetSerializedSize () const
{
  return 4;
}

void Ipv6ExtensionRoutingHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetNextHeader ());
  i.WriteU8 (m_length);
  i.WriteU8 (m_typeRouting);
  i.WriteU8 (m_segmentsLeft);
}

uint32_t Ipv6ExtensionRoutingHeader::Deserialize (Buffer::Iterator start) 
{
  Buffer::Iterator i = start;

  SetNextHeader (i.ReadU8 ());
  m_length = i.ReadU8 ();
  m_typeRouting = i.ReadU8 ();
  m_segmentsLeft = i.ReadU8 ();

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionLooseRoutingHeader);

TypeId Ipv6ExtensionLooseRoutingHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionLooseRoutingHeader")
    .AddConstructor<Ipv6ExtensionLooseRoutingHeader> ()
    .SetParent<Ipv6ExtensionRoutingHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionLooseRoutingHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionLooseRoutingHeader::Ipv6ExtensionLooseRoutingHeader ()
  : m_routersAddress (0)
{
}

Ipv6ExtensionLooseRoutingHeader::~Ipv6ExtensionLooseRoutingHeader ()
{
}

void Ipv6ExtensionLooseRoutingHeader::SetNumberAddress (uint8_t n)
{
  m_routersAddress.clear ();
  m_routersAddress.assign (n, Ipv6Address (""));
}

void Ipv6ExtensionLooseRoutingHeader::SetRoutersAddress (std::vector<Ipv6Address> routersAddress)
{
  m_routersAddress = routersAddress;
}

std::vector<Ipv6Address> Ipv6ExtensionLooseRoutingHeader::GetRoutersAddress () const
{
  return m_routersAddress;
}

void Ipv6ExtensionLooseRoutingHeader::SetRouterAddress (uint8_t index, Ipv6Address addr)
{
  m_routersAddress.at (index) = addr;
}

Ipv6Address Ipv6ExtensionLooseRoutingHeader::GetRouterAddress (uint8_t index) const
{
  return m_routersAddress.at (index);
}

void Ipv6ExtensionLooseRoutingHeader::Print (std::ostream &os) const
{
  os << "( nextHeader = " << (uint32_t)GetNextHeader () << " length = " <<  (uint32_t)GetLength () 
     << " typeRouting = " << (uint32_t)GetTypeRouting () << " segmentsLeft = " << (uint32_t)GetSegmentsLeft () << " ";

  for (std::vector<Ipv6Address>::const_iterator it = m_routersAddress.begin (); it != m_routersAddress.end (); it++)
    {
      os << *it << " ";
    }

  os << " )";
}

uint32_t Ipv6ExtensionLooseRoutingHeader::GetSerializedSize () const
{
  return 8 + m_routersAddress.size () * 16;
}

void Ipv6ExtensionLooseRoutingHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  uint8_t buff[16];

  uint8_t addressNum = m_routersAddress.size ();

  i.WriteU8 (GetNextHeader ());
  i.WriteU8 (addressNum*2);
  i.WriteU8 (GetTypeRouting ());
  i.WriteU8 (GetSegmentsLeft ());
  i.WriteU32 (0);

  for (VectorIpv6Address_t::const_iterator it = m_routersAddress.begin (); it != m_routersAddress.end (); it++)
    {
      it->Serialize (buff);
      i.Write (buff, 16);
    }
}

uint32_t Ipv6ExtensionLooseRoutingHeader::Deserialize (Buffer::Iterator start) 
{
  Buffer::Iterator i = start;
  uint8_t buff[16];

  SetNextHeader (i.ReadU8 ());
  m_length = i.ReadU8 ();
  SetTypeRouting (i.ReadU8 ());
  SetSegmentsLeft (i.ReadU8 ());
  i.ReadU32 ();

  uint8_t addressNum = m_length / 2;
  SetNumberAddress (addressNum);
  for (uint8_t index = 0; index < addressNum; index++)
    {
      i.Read (buff, 16);
      SetRouterAddress (index, Ipv6Address (buff));
    }

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionESPHeader);

TypeId Ipv6ExtensionESPHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionESPHeader")
    .AddConstructor<Ipv6ExtensionESPHeader> ()
    .SetParent<Ipv6ExtensionHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionESPHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionESPHeader::Ipv6ExtensionESPHeader ()
{
}

Ipv6ExtensionESPHeader::~Ipv6ExtensionESPHeader ()
{
}

void Ipv6ExtensionESPHeader::Print (std::ostream &os) const
{
  /** \todo */
}

uint32_t Ipv6ExtensionESPHeader::GetSerializedSize () const
{
  /** \todo */
  return 0;
}

void Ipv6ExtensionESPHeader::Serialize (Buffer::Iterator start) const
{
  /** \todo */
}

uint32_t Ipv6ExtensionESPHeader::Deserialize (Buffer::Iterator start) 
{
  /** \todo */
  return 0;
}

NS_OBJECT_ENSURE_REGISTERED (Ipv6ExtensionAHHeader);

TypeId Ipv6ExtensionAHHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::Ipv6ExtensionAHHeader")
    .AddConstructor<Ipv6ExtensionAHHeader> ()
    .SetParent<Ipv6ExtensionHeader> ()
    .SetGroupName ("Internet")
  ;
  return tid;
}

TypeId Ipv6ExtensionAHHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

Ipv6ExtensionAHHeader::Ipv6ExtensionAHHeader ()
{
}

Ipv6ExtensionAHHeader::~Ipv6ExtensionAHHeader ()
{
}

void Ipv6ExtensionAHHeader::Print (std::ostream &os) const
{
  /** \todo */
}

uint32_t Ipv6ExtensionAHHeader::GetSerializedSize () const
{
  /** \todo */
  return 0;
}

void Ipv6ExtensionAHHeader::Serialize (Buffer::Iterator start) const
{
  /** \todo */
}

uint32_t Ipv6ExtensionAHHeader::Deserialize (Buffer::Iterator start) 
{
  /** \todo */ 
  return 0;
}

} /* namespace ns3 */

