/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Universita' degli Studi di Napoli Federico II
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
 */

#include "ns3/log.h"
#include "arp-queue-disc-item.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ArpQueueDiscItem");

ArpQueueDiscItem::ArpQueueDiscItem (Ptr<Packet> p, const Address& addr,
                                    uint16_t protocol, const ArpHeader & header)
  : QueueDiscItem (p, addr, protocol),
    m_header (header),
    m_headerAdded (false)
{
}

ArpQueueDiscItem::~ArpQueueDiscItem ()
{
  NS_LOG_FUNCTION (this);
}

uint32_t ArpQueueDiscItem::GetSize (void) const
{
  NS_LOG_FUNCTION (this);
  Ptr<Packet> p = GetPacket ();
  NS_ASSERT (p != 0);
  uint32_t ret = p->GetSize ();
  if (!m_headerAdded)
    {
      ret += m_header.GetSerializedSize ();
    }
  return ret;
}

const ArpHeader&
ArpQueueDiscItem::GetHeader (void) const
{
  return m_header;
}

void ArpQueueDiscItem::AddHeader (void)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (!m_headerAdded, "The header has been already added to the packet");
  Ptr<Packet> p = GetPacket ();
  NS_ASSERT (p != 0);
  p->AddHeader (m_header);
  m_headerAdded = true;
}

void
ArpQueueDiscItem::Print (std::ostream& os) const
{
  if (!m_headerAdded)
    {
      os << m_header << " ";
    }
  os << GetPacket () << " "
     << "Dst addr " << GetAddress () << " "
     << "proto " << (uint16_t) GetProtocol () << " "
     << "txq " << (uint8_t) GetTxQueueIndex ()
  ;
}

bool
ArpQueueDiscItem::Mark (void)
{
  NS_LOG_FUNCTION (this);
  return false;
}

uint32_t
ArpQueueDiscItem::Hash (uint32_t perturbation) const
{
  NS_LOG_FUNCTION (this << perturbation);

  Ipv4Address ipv4Src = m_header.GetSourceIpv4Address ();
  Ipv4Address ipv4Dst = m_header.GetDestinationIpv4Address ();
  Address macSrc = m_header.GetSourceHardwareAddress ();
  Address macDst = m_header.GetDestinationHardwareAddress ();
  uint8_t type = m_header.IsRequest () ? ArpHeader::ARP_TYPE_REQUEST : ArpHeader::ARP_TYPE_REPLY;

  /* serialize the addresses and the perturbation in buf */
  uint8_t tmp = 8 + macSrc.GetLength () + macDst.GetLength ();
  uint8_t buf[tmp + 5];
  ipv4Src.Serialize (buf);
  ipv4Dst.Serialize (buf + 4);
  macSrc.CopyTo (buf + 8);
  macDst.CopyTo (buf + 8 + macSrc.GetLength ());
  buf[tmp] = type;
  buf[tmp+1] = (perturbation >> 24) & 0xff;
  buf[tmp+2] = (perturbation >> 16) & 0xff;
  buf[tmp+3] = (perturbation >> 8) & 0xff;
  buf[tmp+4] = perturbation & 0xff;

  // Linux calculates jhash2 (jenkins hash), we calculate murmur3 because it is
  // already available in ns-3

  uint32_t hash = Hash32 ((char*) buf, tmp+5);

  NS_LOG_DEBUG ("Hash value " << hash);

  return hash;
}

} // namespace ns3
