/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Manuel Requena <manuel.requena@cttc.es>
 */

#include "ns3/log.h"

#include "ns3/lte-pdcp-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LtePdcpHeader");

NS_OBJECT_ENSURE_REGISTERED (LtePdcpHeader);

LtePdcpHeader::LtePdcpHeader ()
  : m_dcBit (0xff),
    m_sequenceNumber (0xfffa)
{
}

LtePdcpHeader::~LtePdcpHeader ()
{
  m_dcBit = 0xff;
  m_sequenceNumber = 0xfffb;
}

void
LtePdcpHeader::SetDcBit (uint8_t dcBit)
{
  m_dcBit = dcBit & 0x01;
}

void
LtePdcpHeader::SetSequenceNumber (uint16_t sequenceNumber)
{
  m_sequenceNumber = sequenceNumber & 0x0FFF;
}

uint8_t
LtePdcpHeader::GetDcBit () const
{
  return m_dcBit;
}

uint16_t
LtePdcpHeader::GetSequenceNumber () const
{
  return m_sequenceNumber;
}


TypeId
LtePdcpHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LtePdcpHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<LtePdcpHeader> ()
  ;
  return tid;
}

TypeId
LtePdcpHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void LtePdcpHeader::Print (std::ostream &os)  const
{
  os << "D/C=" << (uint16_t)m_dcBit;
  os << " SN=" << m_sequenceNumber;
}

uint32_t LtePdcpHeader::GetSerializedSize (void) const
{
  return 2;
}

void LtePdcpHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  i.WriteU8 ( (m_dcBit << 7) | (m_sequenceNumber & 0x0F00) >> 8 );
  i.WriteU8 ( (m_sequenceNumber & 0x00FF) );
}

uint32_t LtePdcpHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t byte_1;
  uint8_t byte_2;

  byte_1 = i.ReadU8 ();
  byte_2 = i.ReadU8 ();
  m_dcBit = (byte_1 & 0x80) > 7;
  // For now, we just support DATA PDUs
  NS_ASSERT (m_dcBit == DATA_PDU);
  m_sequenceNumber = ((byte_1 & 0x0F) << 8) | byte_2;

  return GetSerializedSize ();
}

}; // namespace ns3
