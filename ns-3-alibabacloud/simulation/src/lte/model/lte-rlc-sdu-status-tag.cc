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

#include "ns3/lte-rlc-sdu-status-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (LteRlcSduStatusTag);

LteRlcSduStatusTag::LteRlcSduStatusTag ()
{
}

void
LteRlcSduStatusTag::SetStatus (uint8_t status)
{
  m_sduStatus = status;
}

uint8_t
LteRlcSduStatusTag::GetStatus (void) const
{
  return m_sduStatus;
}

TypeId
LteRlcSduStatusTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteRlcSduStatusTag")
    .SetParent<Tag> ()
    .SetGroupName("Lte")
    .AddConstructor<LteRlcSduStatusTag> ()
  ;
  return tid;
}
TypeId
LteRlcSduStatusTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t
LteRlcSduStatusTag::GetSerializedSize (void) const
{
  return 1;
}
void
LteRlcSduStatusTag::Serialize (TagBuffer i) const
{
  i.WriteU8 (m_sduStatus);
}
void
LteRlcSduStatusTag::Deserialize (TagBuffer i)
{
  m_sduStatus = i.ReadU8 ();
}
void
LteRlcSduStatusTag::Print (std::ostream &os) const
{
  os << "SDU Status=" << (uint32_t) m_sduStatus;
}

}; // namespace ns3
