/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 The Boeing Company
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
 * Author:
 *  kwong yin <kwong-sang.yin@boeing.com>
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 *  Erwan Livolant <erwan.livolant@inria.fr>
 */
#include "lr-wpan-mac-trailer.h"
#include <ns3/packet.h>

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (LrWpanMacTrailer);

const uint16_t LrWpanMacTrailer::LR_WPAN_MAC_FCS_LENGTH = 2;

LrWpanMacTrailer::LrWpanMacTrailer (void)
  : m_fcs (0),
    m_calcFcs (false)
{
}

TypeId
LrWpanMacTrailer::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LrWpanMacTrailer")
    .SetParent<Trailer> ()
    .SetGroupName ("LrWpan")
    .AddConstructor<LrWpanMacTrailer> ()
  ;
  return tid;
}

TypeId
LrWpanMacTrailer::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

void
LrWpanMacTrailer::Print (std::ostream &os) const
{
  os << " FCS = " << m_fcs;
}

uint32_t
LrWpanMacTrailer::GetSerializedSize (void) const
{
  return LR_WPAN_MAC_FCS_LENGTH;
}

void
LrWpanMacTrailer::Serialize (Buffer::Iterator start) const
{
  start.Prev (LR_WPAN_MAC_FCS_LENGTH);
  start.WriteU16 (m_fcs);
}

uint32_t
LrWpanMacTrailer::Deserialize (Buffer::Iterator start)
{
  start.Prev (LR_WPAN_MAC_FCS_LENGTH);
  m_fcs = start.ReadU16 ();

  return LR_WPAN_MAC_FCS_LENGTH;
}

uint16_t
LrWpanMacTrailer::GetFcs (void) const
{
  return m_fcs;
}

void
LrWpanMacTrailer::SetFcs (Ptr<const Packet> p)
{
  if (m_calcFcs)
    {
      uint16_t size = p->GetSize ();
      uint8_t *serial_packet = new uint8_t[size];

      p->CopyData (serial_packet, size);

      m_fcs = GenerateCrc16 (serial_packet, size);
      delete[] serial_packet;
    }
}

/* Be sure to have removed the trailer and only the trailer
 * from the packet before to use CheckFcs */
bool
LrWpanMacTrailer::CheckFcs (Ptr<const Packet> p)
{
  if (!m_calcFcs)
    {
      return true;
    }
  else
    {
      uint16_t checkFcs;
      uint16_t size = p->GetSize ();
      uint8_t *serial_packet = new uint8_t[size];

      p->CopyData (serial_packet, size);

      checkFcs = GenerateCrc16 (serial_packet, size);
      delete[] serial_packet;
      return (checkFcs == GetFcs ());
    }
}

void
LrWpanMacTrailer::EnableFcs (bool enable)
{
  m_calcFcs = enable;
  if (!enable)
    {
      m_fcs = 0;
    }
}

bool
LrWpanMacTrailer::IsFcsEnabled (void)
{
  return m_calcFcs;
}

uint16_t
LrWpanMacTrailer::GenerateCrc16 (uint8_t *data, int length)
{
  int i;
  uint16_t accumulator = 0;

  for (i = 0; i < length; ++i)
    {
      accumulator ^= *data;
      accumulator  = (accumulator >> 8) | (accumulator << 8);
      accumulator ^= (accumulator & 0xff00) << 4;
      accumulator ^= (accumulator >> 8) >> 4;
      accumulator ^= (accumulator & 0xff00) >> 5;
      ++data;
    }
  return accumulator;
}

} //namespace ns3
