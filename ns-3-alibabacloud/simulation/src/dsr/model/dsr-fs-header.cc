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
#include "dsr-fs-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DsrFsHeader");

namespace dsr {

NS_OBJECT_ENSURE_REGISTERED (DsrFsHeader);

TypeId DsrFsHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrFsHeader")
    .AddConstructor<DsrFsHeader> ()
    .SetParent<Header> ()
    .SetGroupName ("Dsr")
  ;
  return tid;
}

TypeId DsrFsHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrFsHeader::DsrFsHeader ()
  : m_nextHeader (0),
    m_messageType (0),
    m_payloadLen (0),
    m_sourceId (0),
    m_destId (0),
    m_data (0)
{
}

DsrFsHeader::~DsrFsHeader ()
{
}

void DsrFsHeader::SetNextHeader (uint8_t protocol)
{
  m_nextHeader = protocol;
}

uint8_t DsrFsHeader::GetNextHeader () const
{
  return m_nextHeader;
}

void DsrFsHeader::SetMessageType (uint8_t messageType)
{
  m_messageType = messageType;
}

uint8_t DsrFsHeader::GetMessageType () const
{
  return m_messageType;
}

void DsrFsHeader::SetPayloadLength (uint16_t length)
{
  m_payloadLen = length;
}

uint16_t DsrFsHeader::GetPayloadLength () const
{
  return m_payloadLen;
}

void DsrFsHeader::SetSourceId (uint16_t sourceId)
{
  m_sourceId = sourceId;
}

uint16_t DsrFsHeader::GetSourceId () const
{
  return m_sourceId;
}

void DsrFsHeader::SetDestId (uint16_t destId)
{
  m_destId = destId;
}

uint16_t DsrFsHeader::GetDestId () const
{
  return m_destId;
}

void DsrFsHeader::Print (std::ostream &os) const
{
  os
  << "nextHeader: " << (uint32_t)GetNextHeader () << " messageType: " << (uint32_t)GetMessageType ()
  << " sourceId: " << (uint32_t)GetSourceId () << " destinationId: " << (uint32_t)GetDestId ()
  << " length: " << (uint32_t)GetPayloadLength ();
}

uint32_t DsrFsHeader::GetSerializedSize () const
{
  return 8;
}

void DsrFsHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (m_nextHeader);
  i.WriteU8 (m_messageType);
  i.WriteU16 (m_sourceId);
  i.WriteU16 (m_destId);
  i.WriteU16 (m_payloadLen);

  i.Write (m_data.PeekData (), m_data.GetSize ());
}

uint32_t DsrFsHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  m_nextHeader = i.ReadU8 ();
  m_messageType = i.ReadU8 ();
  m_sourceId = i.ReadU16 ();
  m_destId = i.ReadU16 ();
  m_payloadLen = i.ReadU16 ();

  uint32_t dataLength = GetPayloadLength ();
  uint8_t data[dataLength];
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

  return GetSerializedSize ();
}

DsrOptionField::DsrOptionField (uint32_t optionsOffset)
  : m_optionData (0),
    m_optionsOffset (optionsOffset)
{
}

DsrOptionField::~DsrOptionField ()
{
}

uint32_t DsrOptionField::GetSerializedSize () const
{
  DsrOptionHeader::Alignment align = {4,0};
  return m_optionData.GetSize () + CalculatePad (align);
}

void DsrOptionField::Serialize (Buffer::Iterator start) const
{
  start.Write (m_optionData.Begin (), m_optionData.End ());
  DsrOptionHeader::Alignment align = {4,0};
  uint32_t fill = CalculatePad (align);
  NS_LOG_LOGIC ("fill with " << fill << " bytes padding");
  switch (fill)
    {
    case 0:
      return;
    case 1:
      DsrOptionPad1Header ().Serialize (start);
      return;
    default:
      DsrOptionPadnHeader (fill).Serialize (start);
      return;
    }
}

uint32_t DsrOptionField::Deserialize (Buffer::Iterator start, uint32_t length)
{
  uint8_t buf[length];
  start.Read (buf, length);
  m_optionData = Buffer ();
  m_optionData.AddAtEnd (length);
  m_optionData.Begin ().Write (buf, length);
  return length;
}

void DsrOptionField::AddDsrOption (DsrOptionHeader const& option)
{
  NS_LOG_FUNCTION_NOARGS ();

  uint32_t pad = CalculatePad (option.GetAlignment ());
  NS_LOG_LOGIC ("need " << pad << " bytes padding");
  switch (pad)
    {
    case 0:
      break;       // no padding needed
    case 1:
      AddDsrOption (DsrOptionPad1Header ());
      break;
    default:
      AddDsrOption (DsrOptionPadnHeader (pad));
      break;
    }

  m_optionData.AddAtEnd (option.GetSerializedSize ());
  Buffer::Iterator it = m_optionData.End ();
  it.Prev (option.GetSerializedSize ());
  option.Serialize (it);
}

uint32_t DsrOptionField::CalculatePad (DsrOptionHeader::Alignment alignment) const
{
  return (alignment.offset - (m_optionData.GetSize () + m_optionsOffset)) % alignment.factor;
}

uint32_t DsrOptionField::GetDsrOptionsOffset ()
{
  return m_optionsOffset;
}

Buffer DsrOptionField::GetDsrOptionBuffer ()
{
  return m_optionData;
}

NS_OBJECT_ENSURE_REGISTERED (DsrRoutingHeader);

TypeId DsrRoutingHeader::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::DsrRoutingHeader")
    .AddConstructor<DsrRoutingHeader> ()
    .SetParent<DsrFsHeader> ()
  ;
  return tid;
}

TypeId DsrRoutingHeader::GetInstanceTypeId () const
{
  return GetTypeId ();
}

DsrRoutingHeader::DsrRoutingHeader ()
  : DsrOptionField (8)
{
}

DsrRoutingHeader::~DsrRoutingHeader ()
{
}

void DsrRoutingHeader::Print (std::ostream &os) const
{
  os
  << " nextHeader: " << (uint32_t)GetNextHeader () << " messageType: " << (uint32_t)GetMessageType ()
  << " sourceId: " << (uint32_t)GetSourceId () << " destinationId: " << (uint32_t)GetDestId ()
  << " length: " << (uint32_t)GetPayloadLength ();
}

uint32_t DsrRoutingHeader::GetSerializedSize () const
{
  // 8 bytes is the DsrFsHeader length
  return 8 + DsrOptionField::GetSerializedSize ();
}

void DsrRoutingHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 (GetNextHeader ());
  i.WriteU8 (GetMessageType ());
  i.WriteU16 (GetSourceId ());
  i.WriteU16 (GetDestId ());
  i.WriteU16 (GetPayloadLength ());

  DsrOptionField::Serialize (i);
}

uint32_t DsrRoutingHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;

  SetNextHeader (i.ReadU8 ());
  SetMessageType (i.ReadU8 ());
  SetSourceId (i.ReadU16 ());
  SetDestId (i.ReadU16 ());
  SetPayloadLength (i.ReadU16 ());

  DsrOptionField::Deserialize (i, GetPayloadLength ());

  return GetSerializedSize ();
}

}  /* namespace dsr */
}  /* namespace ns3 */
