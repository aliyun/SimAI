/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#include <ns3/log.h>
#include "aloha-noack-mac-header.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("AlohaNoackMacHeader");

NS_OBJECT_ENSURE_REGISTERED (AlohaNoackMacHeader);

TypeId
AlohaNoackMacHeader::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::AlohaNoackMacHeader")
    .SetParent<Header> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<AlohaNoackMacHeader> ()
  ;
  return tid;
}

TypeId
AlohaNoackMacHeader::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}



uint32_t
AlohaNoackMacHeader::GetSerializedSize (void) const
{
  return 12;
}



void
AlohaNoackMacHeader::Serialize (Buffer::Iterator start) const
{
  WriteTo (start, m_destination);
  WriteTo (start, m_source);
}

uint32_t
AlohaNoackMacHeader::Deserialize (Buffer::Iterator start)
{
  ReadFrom (start, m_destination);
  ReadFrom (start, m_source);
  return GetSerializedSize ();
}


void
AlohaNoackMacHeader::Print (std::ostream &os) const
{
  os << "src="  << m_source
     << "dst="  << m_destination;
}


void
AlohaNoackMacHeader::SetSource (Mac48Address source)
{
  m_source = source;
}

Mac48Address
AlohaNoackMacHeader::GetSource () const
{
  return m_source;
}

void
AlohaNoackMacHeader::SetDestination (Mac48Address dst)
{
  m_destination = dst;
}

Mac48Address
AlohaNoackMacHeader::GetDestination () const
{
  return m_destination;
}




} // namespace ns3


