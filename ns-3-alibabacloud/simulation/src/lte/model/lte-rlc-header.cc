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

#include "ns3/lte-rlc-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteRlcHeader");

NS_OBJECT_ENSURE_REGISTERED (LteRlcHeader);

LteRlcHeader::LteRlcHeader ()
  : m_headerLength (0),
    m_framingInfo (0xff),
    m_sequenceNumber (0xfffa)
{
}

LteRlcHeader::~LteRlcHeader ()
{
    m_headerLength = 0;
    m_framingInfo = 0xff;
    m_sequenceNumber = 0xfffb;
}

void
LteRlcHeader::SetFramingInfo (uint8_t framingInfo)
{
  m_framingInfo = framingInfo & 0x03;
}

void
LteRlcHeader::SetSequenceNumber (SequenceNumber10 sequenceNumber)
{
  m_sequenceNumber = sequenceNumber;
}

uint8_t
LteRlcHeader::GetFramingInfo () const
{
  return m_framingInfo;
}

SequenceNumber10
LteRlcHeader::GetSequenceNumber () const
{
  return m_sequenceNumber;
}


void
LteRlcHeader::PushExtensionBit (uint8_t extensionBit)
{
  m_extensionBits.push_back (extensionBit);
  if (m_extensionBits.size() == 1)
    {
      m_headerLength = 2; // Only fixed part
    }
  else if (m_extensionBits.size() % 2)
    {
      m_headerLength += 1;
    }
  else
    {
      m_headerLength += 2;
    }
}

void
LteRlcHeader::PushLengthIndicator (uint16_t lengthIndicator)
{
  m_lengthIndicators.push_back (lengthIndicator);
}


uint8_t
LteRlcHeader::PopExtensionBit (void)
{
  uint8_t extensionBit = m_extensionBits.front ();
  m_extensionBits.pop_front ();

  return extensionBit;
}

uint16_t
LteRlcHeader::PopLengthIndicator (void)
{
  uint16_t lengthIndicator = m_lengthIndicators.front ();
  m_lengthIndicators.pop_front ();

  return lengthIndicator;
}


TypeId
LteRlcHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteRlcHeader")
    .SetParent<Header> ()
    .SetGroupName("Lte")
    .AddConstructor<LteRlcHeader> ()
  ;
  return tid;
}

TypeId
LteRlcHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void LteRlcHeader::Print (std::ostream &os)  const
{
  std::list <uint8_t>::const_iterator it1 = m_extensionBits.begin ();
  std::list <uint16_t>::const_iterator it2 = m_lengthIndicators.begin ();

  os << "Len=" << m_headerLength;
  os << " FI=" << (uint16_t)m_framingInfo;
  os << " E=" << (uint16_t)(*it1);
  os << " SN=" << m_sequenceNumber;

  it1++;
  if (it1 != m_extensionBits.end ())
    {
      os << " E=";
    }
  while ( it1 != m_extensionBits.end () )
    {
      os << (uint16_t)(*it1);
      it1++;
    }

  if (it2 != m_lengthIndicators.end ())
    {
      os << " LI=";
    }
  while ( it2 != m_lengthIndicators.end () )
    {
      os << (uint16_t)(*it2) << " ";
      it2++;
    }
}

uint32_t LteRlcHeader::GetSerializedSize (void) const
{
  return m_headerLength;
}

void LteRlcHeader::Serialize (Buffer::Iterator start) const
{
  Buffer::Iterator i = start;

  std::list <uint8_t>::const_iterator it1 = m_extensionBits.begin ();
  std::list <uint16_t>::const_iterator it2 = m_lengthIndicators.begin ();

  i.WriteU8 ( ((m_framingInfo << 3) & 0x18) |
              (((*it1) << 2) & 0x04) |
              ((m_sequenceNumber.GetValue () >> 8) & 0x0003) );
  i.WriteU8 ( m_sequenceNumber.GetValue () & 0x00FF );
  it1++;

  while ( it1 != m_extensionBits.end () &&
          it2 != m_lengthIndicators.end () )
    {
      uint16_t oddLi, evenLi;
      uint8_t oddE, evenE;

      oddE = *it1;
      oddLi = *it2;

      it1++;
      it2++;

      if ( it1 != m_extensionBits.end () &&
           it2 != m_lengthIndicators.end () )
        {
          evenE = *it1;
          evenLi = *it2;

          i.WriteU8 ( ((oddE << 7) & 0x80) | ((oddLi >> 4) & 0x007F) );
          i.WriteU8 ( ((oddLi << 4) & 0x00F0) | ((evenE << 3) & 0x08) | ((evenLi >> 8) & 0x0007) );
          i.WriteU8 ( evenLi & 0x00FF );

          it1++;
          it2++;
        }
      else
        {
          i.WriteU8 ( ((oddE << 7) & 0x80) | ((oddLi >> 4) & 0x007F) );
          i.WriteU8 ( ((oddLi << 4) & 0x00F0) ); // Padding is implicit
        }
    }
}

uint32_t LteRlcHeader::Deserialize (Buffer::Iterator start)
{
  Buffer::Iterator i = start;
  uint8_t byte_1;
  uint8_t byte_2;
  uint8_t byte_3;
  uint8_t extensionBit;

  byte_1 = i.ReadU8 ();
  byte_2 = i.ReadU8 ();
  m_headerLength = 2;
  m_framingInfo = (byte_1 & 0x18) >> 3;
  m_sequenceNumber = ((byte_1 & 0x03) << 8) | byte_2;

  extensionBit = (byte_1 & 0x04) >> 2;
  m_extensionBits.push_back (extensionBit);

  if (extensionBit == DATA_FIELD_FOLLOWS)
    {
      return GetSerializedSize ();
    }

  uint16_t oddLi, evenLi;
  uint8_t oddE, evenE;
  bool moreLiFields = (extensionBit == E_LI_FIELDS_FOLLOWS);

  while (moreLiFields)
    {
      byte_1 = i.ReadU8 ();
      byte_2 = i.ReadU8 ();

      oddE = (byte_1 & 0x80) >> 7;
      oddLi = ((byte_1 & 0x7F) << 4) | ((byte_2 & 0xF0) >> 4);
      moreLiFields = (oddE == E_LI_FIELDS_FOLLOWS);

      m_extensionBits.push_back (oddE);
      m_lengthIndicators.push_back (oddLi);
      m_headerLength += 2;

      if (moreLiFields)
        {
          byte_3 = i.ReadU8 ();

          evenE = (byte_2 & 0x08) >> 3;
          evenLi = ((byte_2 & 0x07) << 8) | (byte_3 & 0xFF);
          moreLiFields = (evenE == E_LI_FIELDS_FOLLOWS);

          m_extensionBits.push_back (evenE);
          m_lengthIndicators.push_back (evenLi);

          m_headerLength += 1;
        }
    }

  return GetSerializedSize ();
}

}; // namespace ns3
