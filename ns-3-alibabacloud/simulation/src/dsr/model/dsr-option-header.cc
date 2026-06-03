/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Yufei Cheng
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
 * Author: Yufei Cheng   <yfcheng@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#include "ns3/assert.h"
#include "ns3/log.h"
#include "ns3/header.h"
#include "dsr-option-header.h"
#include "ns3/ipv4-address.h"
#include "ns3/address-utils.h"
#include "ns3/packet.h"
#include "ns3/enum.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DsrOptionHeader");

namespace dsr {

NS_OBJECT_ENSURE_REGISTERED (DsrOptionHeader);

TypeId DsrOptionHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionHeader")
    .AddConstructor<DsrOptionHeader> ()
    .SetParent<Header> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionHeader::DsrOptionHeader ()
  : m_type (0),
    m_length (0)
{
}

DsrOptionHeader::~DsrOptionHeader ()
{
}

void DsrOptionHeader::SetType (uint8_t type)
{
  m_type = type;
}

uint8_t DsrOptionHeader::GetType () const
{
  return m_type;
}

void DsrOptionHeader::SetLength (uint8_t length)
{
  m_length = length;
}

uint8_t DsrOptionHeader::GetLength () const
{
  return m_length;
}

void DsrOptionHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)m_type << " length = " << (uint32_t)m_length << " )";
}

uint32_t DsrOptionHeader::GetSerializedSize () const
{
  return m_length + 2;
}

void DsrOptionHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (m_type);
  i.WriteU8 (m_length);
  i.Write (m_data.Begin (), m_data.End ());
}

uint32_t DsrOptionHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_type = i.ReadU8 ();
  m_length = i.ReadU8 ();

  m_data = Buffer ();
  m_data.AddAtEnd (m_length);
  Buffer::Iterator dataStart = i;
  i.Next (m_length);
  Buffer::Iterator dataEnd = i;
  m_data.Begin ().Write (dataStart, dataEnd);

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionHeader::GetAlignment () const
{
  Alignment retVal = { 1, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionPad1Header);

TypeId DsrOptionPad1Header::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionPad1Header")
    .AddConstructor<DsrOptionPad1Header> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionPad1Header::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionPad1Header::DsrOptionPad1Header ()
{
  SetType (224);
}

DsrOptionPad1Header::~DsrOptionPad1Header ()
{
}

void DsrOptionPad1Header::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " )";
}

uint32_t DsrOptionPad1Header::GetSerializedSize () const
{
  return 1;
}

void DsrOptionPad1Header::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
}

uint32_t DsrOptionPad1Header::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionPadnHeader);

TypeId DsrOptionPadnHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionPadnHeader")
    .AddConstructor<DsrOptionPadnHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionPadnHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionPadnHeader::DsrOptionPadnHeader (uint32_t pad)
{
  SetType (0);
  NS_ASSERT_MSG (pad >= 2, "PadN must be at least 2 bytes long");
  SetLength (pad - 2);
}

DsrOptionPadnHeader::~DsrOptionPadnHeader ()
{
}

void DsrOptionPadnHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength () << " )";
}

uint32_t DsrOptionPadnHeader::GetSerializedSize () const
{
  return GetLength () + 2;
}

void DsrOptionPadnHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());

  for (int padding = 0; padding < GetLength (); padding++)
    {
      i.WriteU8 (0);
    }
}

uint32_t DsrOptionPadnHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());

  return GetSerializedSize ();
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionRreqHeader);

TypeId DsrOptionRreqHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionRreqHeader")
    .AddConstructor<DsrOptionRreqHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionRreqHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionRreqHeader::DsrOptionRreqHeader ()
  : m_ipv4Address (0)
{
  SetType (1);
  SetLength (6 + m_ipv4Address.size () * 4);
}

DsrOptionRreqHeader::~DsrOptionRreqHeader ()
{
}

void DsrOptionRreqHeader::SetNumberAddress (uint8_t n)
{
  m_ipv4Address.clear ();
  m_ipv4Address.assign (n, Ipv4Address ());
}

Ipv4Address DsrOptionRreqHeader::GetTarget ()
{
  return m_target;
}

void DsrOptionRreqHeader::SetTarget (Ipv4Address target)
{
  m_target = target;
}

void DsrOptionRreqHeader::AddNodeAddress (Ipv4Address ipv4)
{
  m_ipv4Address.push_back (ipv4);
  SetLength (6 + m_ipv4Address.size () * 4);
}

void DsrOptionRreqHeader::SetNodesAddress (std::vector<Ipv4Address> ipv4Address)
{
  m_ipv4Address = ipv4Address;
  SetLength (6 + m_ipv4Address.size () * 4);
}

std::vector<Ipv4Address> DsrOptionRreqHeader::GetNodesAddresses () const
{
  return m_ipv4Address;
}

uint32_t DsrOptionRreqHeader::GetNodesNumber () const
{
  return m_ipv4Address.size ();
}

void DsrOptionRreqHeader::SetNodeAddress (uint8_t index, Ipv4Address addr)
{
  m_ipv4Address.at (index) = addr;
}

Ipv4Address DsrOptionRreqHeader::GetNodeAddress (uint8_t index) const
{
  return m_ipv4Address.at (index);
}

void DsrOptionRreqHeader::SetId (uint16_t identification)
{
  m_identification = identification;
}

uint16_t DsrOptionRreqHeader::GetId () const
{
  return m_identification;
}

void DsrOptionRreqHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength () << "";

  for (std::vector<Ipv4Address>::const_iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      os << *it << " ";
    }

  os << ")";
}

uint32_t DsrOptionRreqHeader::GetSerializedSize () const
{
  return 8 + m_ipv4Address.size () * 4;
}

void DsrOptionRreqHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  uint8_t buff[4];

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteHtonU16 (m_identification);
  WriteTo (i, m_target);

  for (VectorIpv4Address_t::const_iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      it->Serialize (buff);
      i.Write (buff, 4);
    }
}

uint32_t DsrOptionRreqHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t buff[4];

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  m_identification = i.ReadNtohU16 ();
  ReadFrom (i, m_target);

  uint8_t index = 0;
  for (std::vector<Ipv4Address>::iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      i.Read (buff, 4);
      m_address = it->Deserialize (buff);
      SetNodeAddress (index, m_address);
      ++index;
    }

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionRreqHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionRrepHeader);

TypeId DsrOptionRrepHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionRrepHeader")
    .AddConstructor<DsrOptionRrepHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionRrepHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionRrepHeader::DsrOptionRrepHeader ()
  : m_ipv4Address (0)
{
  SetType (2);
  SetLength (2 + m_ipv4Address.size () * 4);
}

DsrOptionRrepHeader::~DsrOptionRrepHeader ()
{
}

void DsrOptionRrepHeader::SetNumberAddress (uint8_t n)
{
  m_ipv4Address.clear ();
  m_ipv4Address.assign (n, Ipv4Address ());
}

void DsrOptionRrepHeader::SetNodesAddress (std::vector<Ipv4Address> ipv4Address)
{
  m_ipv4Address = ipv4Address;
  SetLength (2 + m_ipv4Address.size () * 4);
}

std::vector<Ipv4Address> DsrOptionRrepHeader::GetNodesAddress () const
{
  return m_ipv4Address;
}

void DsrOptionRrepHeader::SetNodeAddress (uint8_t index, Ipv4Address addr)
{
  m_ipv4Address.at (index) = addr;
}

Ipv4Address DsrOptionRrepHeader::GetNodeAddress (uint8_t index) const
{
  return m_ipv4Address.at (index);
}

Ipv4Address DsrOptionRrepHeader::GetTargetAddress (std::vector<Ipv4Address> ipv4Address) const
{
  return m_ipv4Address.at (ipv4Address.size () - 1);
}

void DsrOptionRrepHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength () << "";

  for (std::vector<Ipv4Address>::const_iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      os << *it << " ";
    }

  os << ")";
}

uint32_t DsrOptionRrepHeader::GetSerializedSize () const
{
  return 4 + m_ipv4Address.size () * 4;
}

void DsrOptionRrepHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  uint8_t buff[4];

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU8 (0);
  i.WriteU8 (0);

  for (VectorIpv4Address_t::const_iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      it->Serialize (buff);
      i.Write (buff, 4);
    }
}

uint32_t DsrOptionRrepHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t buff[4];

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  i.ReadU8 ();
  i.ReadU8 ();

  uint8_t index = 0;
  for (std::vector<Ipv4Address>::iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      i.Read (buff, 4);
      m_address = it->Deserialize (buff);
      SetNodeAddress (index, m_address);
      ++index;
    }

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionRrepHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionSRHeader);

TypeId DsrOptionSRHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionSRHeader")
    .AddConstructor<DsrOptionSRHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionSRHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionSRHeader::DsrOptionSRHeader ()
  : m_segmentsLeft (0),
    m_ipv4Address (0)
{
  SetType (96);
  SetLength (2 + m_ipv4Address.size () * 4);
}

DsrOptionSRHeader::~DsrOptionSRHeader ()
{
}

void DsrOptionSRHeader::SetSegmentsLeft (uint8_t segmentsLeft)
{
  m_segmentsLeft = segmentsLeft;
}

uint8_t DsrOptionSRHeader::GetSegmentsLeft () const
{
  return m_segmentsLeft;
}

void DsrOptionSRHeader::SetSalvage (uint8_t salvage)
{
  m_salvage = salvage;
}

uint8_t DsrOptionSRHeader::GetSalvage () const
{
  return m_salvage;
}

void DsrOptionSRHeader::SetNumberAddress (uint8_t n)
{
  m_ipv4Address.clear ();
  m_ipv4Address.assign (n, Ipv4Address ());
}

void DsrOptionSRHeader::SetNodesAddress (std::vector<Ipv4Address> ipv4Address)
{
  m_ipv4Address = ipv4Address;
  SetLength (2 + m_ipv4Address.size () * 4);
}

std::vector<Ipv4Address> DsrOptionSRHeader::GetNodesAddress () const
{
  return m_ipv4Address;
}

void DsrOptionSRHeader::SetNodeAddress (uint8_t index, Ipv4Address addr)
{
  m_ipv4Address.at (index) = addr;
}

Ipv4Address DsrOptionSRHeader::GetNodeAddress (uint8_t index) const
{
  return m_ipv4Address.at (index);
}

uint8_t DsrOptionSRHeader::GetNodeListSize () const
{
  return m_ipv4Address.size ();
}

void DsrOptionSRHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength () << "";

  for (std::vector<Ipv4Address>::const_iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      os << *it << " ";
    }

  os << ")";
}

uint32_t DsrOptionSRHeader::GetSerializedSize () const
{
  return 4 + m_ipv4Address.size () * 4;
}

void DsrOptionSRHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;
  uint8_t buff[4];

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU8 (m_salvage);
  i.WriteU8 (m_segmentsLeft);

  for (VectorIpv4Address_t::const_iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      it->Serialize (buff);
      i.Write (buff, 4);
    }
}

uint32_t DsrOptionSRHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t buff[4];

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  m_salvage = i.ReadU8 ();
  m_segmentsLeft = i.ReadU8 ();

  uint8_t index = 0;
  for (std::vector<Ipv4Address>::iterator it = m_ipv4Address.begin (); it != m_ipv4Address.end (); it++)
    {
      i.Read (buff, 4);
      m_address = it->Deserialize (buff);
      SetNodeAddress (index, m_address);
      ++index;
    }

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionSRHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionRerrHeader);

TypeId DsrOptionRerrHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionRerrHeader")
    .AddConstructor<DsrOptionRerrHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionRerrHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionRerrHeader::DsrOptionRerrHeader ()
  : m_errorType (0),
    m_salvage (0),
    m_errorLength (4)
{
  SetType (3);
  SetLength (18);
}

DsrOptionRerrHeader::~DsrOptionRerrHeader ()
{
}

void DsrOptionRerrHeader::SetErrorType (uint8_t errorType)
{
  m_errorType = errorType;
}

uint8_t DsrOptionRerrHeader::GetErrorType () const
{
  return m_errorType;
}

void DsrOptionRerrHeader::SetSalvage (uint8_t salvage)
{
  m_salvage = salvage;
}

uint8_t DsrOptionRerrHeader::GetSalvage () const
{
  return m_salvage;
}

void DsrOptionRerrHeader::SetErrorSrc (Ipv4Address errorSrcAddress)
{
  m_errorSrcAddress = errorSrcAddress;
}

Ipv4Address DsrOptionRerrHeader::GetErrorSrc () const
{
  return m_errorSrcAddress;
}

void DsrOptionRerrHeader::SetErrorDst (Ipv4Address errorDstAddress)
{
  m_errorDstAddress = errorDstAddress;
}

Ipv4Address DsrOptionRerrHeader::GetErrorDst () const
{
  return m_errorDstAddress;
}

void DsrOptionRerrHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength ()
     << " errorType = " << (uint32_t)m_errorType << " salvage = " << (uint32_t)m_salvage
     << " error source = " << m_errorSrcAddress << " error dst = " << m_errorDstAddress << " )";

}

uint32_t DsrOptionRerrHeader::GetSerializedSize () const
{
  return 20;
}

void DsrOptionRerrHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU8 (m_errorType);
  i.WriteU8 (m_salvage);
  WriteTo (i, m_errorSrcAddress);
  WriteTo (i, m_errorDstAddress);
  i.Write (m_errorData.Begin (), m_errorData.End ());
}

uint32_t DsrOptionRerrHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  m_errorType = i.ReadU8 ();
  m_salvage = i.ReadU8 ();
  ReadFrom (i, m_errorSrcAddress);
  ReadFrom (i, m_errorDstAddress);

  m_errorData = Buffer ();
  m_errorData.AddAtEnd (m_errorLength);
  Buffer::Iterator dataStart = i;
  i.Next (m_errorLength);
  Buffer::Iterator dataEnd = i;
  m_errorData.Begin ().Write (dataStart, dataEnd);

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionRerrHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionRerrUnreachHeader);

TypeId DsrOptionRerrUnreachHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionRerrUnreachHeader")
    .AddConstructor<DsrOptionRerrUnreachHeader> ()
    .SetParent<DsrOptionRerrHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionRerrUnreachHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionRerrUnreachHeader::DsrOptionRerrUnreachHeader ()
  : m_salvage (0)
{
  SetType (3);
  SetLength (18);
  SetErrorType (1);
}

DsrOptionRerrUnreachHeader::~DsrOptionRerrUnreachHeader ()
{
}

void DsrOptionRerrUnreachHeader::SetSalvage (uint8_t salvage)
{
  m_salvage = salvage;
}

uint8_t DsrOptionRerrUnreachHeader::GetSalvage () const
{
  return m_salvage;
}

void DsrOptionRerrUnreachHeader::SetErrorSrc (Ipv4Address errorSrcAddress)
{
  m_errorSrcAddress = errorSrcAddress;
}

Ipv4Address DsrOptionRerrUnreachHeader::GetErrorSrc () const
{
  return m_errorSrcAddress;
}

void DsrOptionRerrUnreachHeader::SetErrorDst (Ipv4Address errorDstAddress)
{
  m_errorDstAddress = errorDstAddress;
}

Ipv4Address DsrOptionRerrUnreachHeader::GetErrorDst () const
{
  return m_errorDstAddress;
}

void DsrOptionRerrUnreachHeader::SetUnreachNode (Ipv4Address unreachNode)
{
  m_unreachNode = unreachNode;
}

Ipv4Address DsrOptionRerrUnreachHeader::GetUnreachNode () const
{
  return m_unreachNode;
}

void DsrOptionRerrUnreachHeader::SetOriginalDst (Ipv4Address originalDst)
{
  m_originalDst = originalDst;
}

Ipv4Address DsrOptionRerrUnreachHeader::GetOriginalDst () const
{
  return m_originalDst;
}

void DsrOptionRerrUnreachHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength ()
     << " errorType = " << (uint32_t)m_errorType << " salvage = " << (uint32_t)m_salvage
     << " error source = " << m_errorSrcAddress << " error dst = " << m_errorDstAddress
     << " unreach node = " <<  m_unreachNode << " )";
}

uint32_t DsrOptionRerrUnreachHeader::GetSerializedSize () const
{
  return 20;
}

void DsrOptionRerrUnreachHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU8 (GetErrorType ());
  i.WriteU8 (m_salvage);
  WriteTo (i, m_errorSrcAddress);
  WriteTo (i, m_errorDstAddress);
  WriteTo (i, m_unreachNode);
  WriteTo (i, m_originalDst);
}

uint32_t DsrOptionRerrUnreachHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  SetErrorType (i.ReadU8 ());
  m_salvage = i.ReadU8 ();
  ReadFrom (i, m_errorSrcAddress);
  ReadFrom (i, m_errorDstAddress);
  ReadFrom (i, m_unreachNode);
  ReadFrom (i, m_originalDst);

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionRerrUnreachHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionRerrUnsupportHeader);

TypeId DsrOptionRerrUnsupportHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionRerrUnsupportHeader")
    .AddConstructor<DsrOptionRerrUnsupportHeader> ()
    .SetParent<DsrOptionRerrHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionRerrUnsupportHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionRerrUnsupportHeader::DsrOptionRerrUnsupportHeader ()
  : m_salvage (0)
{
  SetType (3);
  SetLength (14);
  SetErrorType (3);
}

DsrOptionRerrUnsupportHeader::~DsrOptionRerrUnsupportHeader ()
{
}

void DsrOptionRerrUnsupportHeader::SetSalvage (uint8_t salvage)
{
  m_salvage = salvage;
}

uint8_t DsrOptionRerrUnsupportHeader::GetSalvage () const
{
  return m_salvage;
}

void DsrOptionRerrUnsupportHeader::SetErrorSrc (Ipv4Address errorSrcAddress)
{
  m_errorSrcAddress = errorSrcAddress;
}

Ipv4Address DsrOptionRerrUnsupportHeader::GetErrorSrc () const
{
  return m_errorSrcAddress;
}

void DsrOptionRerrUnsupportHeader::SetErrorDst (Ipv4Address errorDstAddress)
{
  m_errorDstAddress = errorDstAddress;
}

Ipv4Address DsrOptionRerrUnsupportHeader::GetErrorDst () const
{
  return m_errorDstAddress;
}

void DsrOptionRerrUnsupportHeader::SetUnsupported (uint16_t unsupport)
{
  m_unsupport = unsupport;
}

uint16_t DsrOptionRerrUnsupportHeader::GetUnsupported () const
{
  return m_unsupport;
}

void DsrOptionRerrUnsupportHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength ()
     << " errorType = " << (uint32_t)m_errorType << " salvage = " << (uint32_t)m_salvage
     << " error source = " << m_errorSrcAddress << " error dst = " << m_errorDstAddress
     << " unsupported option = " <<  m_unsupport << " )";

}

uint32_t DsrOptionRerrUnsupportHeader::GetSerializedSize () const
{
  return 16;
}

void DsrOptionRerrUnsupportHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU8 (GetErrorType ());
  i.WriteU8 (m_salvage);
  WriteTo (i, m_errorSrcAddress);
  WriteTo (i, m_errorDstAddress);
  i.WriteU16 (m_unsupport);

}

uint32_t DsrOptionRerrUnsupportHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  SetErrorType (i.ReadU8 ());
  m_salvage = i.ReadU8 ();
  ReadFrom (i, m_errorSrcAddress);
  ReadFrom (i, m_errorDstAddress);
  m_unsupport = i.ReadU16 ();

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionRerrUnsupportHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionAckReqHeader);

TypeId DsrOptionAckReqHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionAckReqHeader")
    .AddConstructor<DsrOptionAckReqHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionAckReqHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionAckReqHeader::DsrOptionAckReqHeader ()
  : m_identification (0)

{
  SetType (160);
  SetLength (2);
}

DsrOptionAckReqHeader::~DsrOptionAckReqHeader ()
{
}

void DsrOptionAckReqHeader::SetAckId (uint16_t identification)
{
  m_identification = identification;
}

uint16_t DsrOptionAckReqHeader::GetAckId () const
{
  return m_identification;
}

void DsrOptionAckReqHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength ()
     << " id = " << m_identification << " )";
}

uint32_t DsrOptionAckReqHeader::GetSerializedSize () const
{
  return 4;
}

void DsrOptionAckReqHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU16 (m_identification);
}

uint32_t DsrOptionAckReqHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  m_identification = i.ReadU16 ();

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionAckReqHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}

NS_OBJECT_ENSURE_REGISTERED (DsrOptionAckHeader);

TypeId DsrOptionAckHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrOptionAckHeader")
    .AddConstructor<DsrOptionAckHeader> ()
    .SetParent<DsrOptionHeader> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrOptionAckHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrOptionAckHeader::DsrOptionAckHeader ()
  :    m_identification (0)
{
  SetType (32);
  SetLength (10);
}

DsrOptionAckHeader::~DsrOptionAckHeader ()
{
}

void DsrOptionAckHeader::SetAckId (uint16_t identification)
{
  m_identification = identification;
}

uint16_t DsrOptionAckHeader::GetAckId () const
{
  return m_identification;
}

void DsrOptionAckHeader::SetRealSrc (Ipv4Address realSrcAddress)
{
  m_realSrcAddress = realSrcAddress;
}

Ipv4Address DsrOptionAckHeader::GetRealSrc () const
{
  return m_realSrcAddress;
}

void DsrOptionAckHeader::SetRealDst (Ipv4Address realDstAddress)
{
  m_realDstAddress = realDstAddress;
}

Ipv4Address DsrOptionAckHeader::GetRealDst () const
{
  return m_realDstAddress;
}

void DsrOptionAckHeader::Print (std::ostream &os) const
{
  os << "( type = " << (uint32_t)GetType () << " length = " << (uint32_t)GetLength ()
     << " id = " << m_identification << " real src = " << m_realSrcAddress
     << " real dst = " << m_realDstAddress << " )";

}

uint32_t DsrOptionAckHeader::GetSerializedSize () const
{
  return 12;
}

void DsrOptionAckHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetType ());
  i.WriteU8 (GetLength ());
  i.WriteU16 (m_identification);
  WriteTo (i, m_realSrcAddress);
  WriteTo (i, m_realDstAddress);
}

uint32_t DsrOptionAckHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetType (i.ReadU8 ());
  SetLength (i.ReadU8 ());
  m_identification = i.ReadU16 ();
  ReadFrom (i, m_realSrcAddress);
  ReadFrom (i, m_realDstAddress);

  return GetSerializedSize ();
}

DsrOptionHeader::Alignment DsrOptionAckHeader::GetAlignment () const
{
  Alignment retVal = { 4, 0 };
  return retVal;
}
} /* namespace dsr */
} /* namespace ns3 */
