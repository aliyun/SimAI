/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Sébastien Deronne
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "erp-information.h"

namespace ns3 {

ErpInformation::ErpInformation ()
  : m_erpInformation (0),
    m_erpSupported (0)
{
}

WifiInformationElementId
ErpInformation::ElementId () const
{
  return IE_ERP_INFORMATION;
}

void
ErpInformation::SetErpSupported (uint8_t erpSupported)
{
  m_erpSupported = erpSupported;
}

void
ErpInformation::SetBarkerPreambleMode (uint8_t barkerPreambleMode)
{
  m_erpInformation |= (barkerPreambleMode & 0x01) << 2;
}

void
ErpInformation::SetUseProtection (uint8_t useProtection)
{
  m_erpInformation |= (useProtection & 0x01) << 1;
}

void
ErpInformation::SetNonErpPresent (uint8_t nonErpPresent)
{
  m_erpInformation |= nonErpPresent & 0x01;
}

uint8_t
ErpInformation::GetBarkerPreambleMode (void) const
{
  return ((m_erpInformation >> 2) & 0x01);
}

uint8_t
ErpInformation::GetUseProtection (void) const
{
  return ((m_erpInformation >> 1) & 0x01);
}

uint8_t
ErpInformation::GetNonErpPresent (void) const
{
  return (m_erpInformation & 0x01);
}

uint8_t
ErpInformation::GetInformationFieldSize () const
{
  NS_ASSERT (m_erpSupported);
  return 1;
}

Buffer::Iterator
ErpInformation::Serialize (Buffer::Iterator i) const
{
  if (!m_erpSupported)
    {
      return i;
    }
  return WifiInformationElement::Serialize (i);
}

uint16_t
ErpInformation::GetSerializedSize () const
{
  if (!m_erpSupported)
    {
      return 0;
    }
  return WifiInformationElement::GetSerializedSize ();
}

void
ErpInformation::SerializeInformationField (Buffer::Iterator start) const
{
  if (m_erpSupported)
    {
      start.WriteU8 (m_erpInformation);
    }
}

uint8_t
ErpInformation::DeserializeInformationField (Buffer::Iterator start,
                                             uint8_t length)
{
  Buffer::Iterator i = start;
  m_erpInformation = i.ReadU8 ();
  return length;
}

std::ostream &
operator << (std::ostream &os, const ErpInformation &erpInformation)
{
  os <<  bool (erpInformation.GetBarkerPreambleMode ())
     << "|" << bool (erpInformation.GetUseProtection ())
     << "|" << bool (erpInformation.GetNonErpPresent ());

  return os;
}

} //namespace ns3
