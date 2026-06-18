/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2016 Universita' degli Studi di Napoli Federico II
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
#include "ipv6-queue-disc-item.h"
#include "ns3/tcp-header.h"
#include "ns3/udp-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv6QueueDiscItem");

Ipv6QueueDiscItem::Ipv6QueueDiscItem (Ptr<Packet> p, const Address& addr,
                                      uint16_t protocol, const Ipv6Header & header)
  : QueueDiscItem (p, addr, protocol),
    m_header (header),
    m_headerAdded (false)
{
}

Ipv6QueueDiscItem::~Ipv6QueueDiscItem ()
{
  NS_LOG_FUNCTION (this);
}

uint32_t Ipv6QueueDiscItem::GetSize (void) const
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

const Ipv6Header&
Ipv6QueueDiscItem::GetHeader (void) const
{
  return m_header;
}

void Ipv6QueueDiscItem::AddHeader (void)
{
  NS_LOG_FUNCTION (this);

  NS_ASSERT_MSG (!m_headerAdded, "The header has been already added to the packet");
  Ptr<Packet> p = GetPacket ();
  NS_ASSERT (p != 0);
  p->AddHeader (m_header);
  m_headerAdded = true;
}

void
Ipv6QueueDiscItem::Print (std::ostream& os) const
{
  if (!m_headerAdded)
    {
      os << m_header << " ";
    }
  os << GetPacket () << " "
     << "Dst addr " << GetAddress () << " "
     << "proto " << (uint16_t) GetProtocol () << " "
     << "txq " << (uint16_t) GetTxQueueIndex ()
  ;
}

bool
Ipv6QueueDiscItem::Mark (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_headerAdded && m_header.GetEcn () != Ipv6Header::ECN_NotECT)
    {
      m_header.SetEcn (Ipv6Header::ECN_CE);
      return true;
    }
  return false;
}

bool
Ipv6QueueDiscItem::GetUint8Value (QueueItem::Uint8Values field, uint8_t& value) const
{
  bool ret = false;

  switch (field)
    {
    case IP_DSFIELD:
      value = m_header.GetTrafficClass ();
      ret = true;
      break;
    }

  return ret;
}

uint32_t
Ipv6QueueDiscItem::Hash (uint32_t perturbation) const
{
  NS_LOG_FUNCTION (this << perturbation);

  Ipv6Address src = m_header.GetSource ();
  Ipv6Address dest = m_header.GetDestination ();
  uint8_t prot = m_header.GetNextHeader ();

  TcpHeader tcpHdr;
  UdpHeader udpHdr;
  uint16_t srcPort = 0;
  uint16_t destPort = 0;

  if (prot == 6) // TCP
    {
      GetPacket ()->PeekHeader (tcpHdr);
      srcPort = tcpHdr.GetSourcePort ();
      destPort = tcpHdr.GetDestinationPort ();
    }
  else if (prot == 17) // UDP
    {
      GetPacket ()->PeekHeader (udpHdr);
      srcPort = udpHdr.GetSourcePort ();
      destPort = udpHdr.GetDestinationPort ();
    }
  if (prot != 6 && prot != 17)
    {
      NS_LOG_WARN ("Unknown transport protocol, no port number included in hash computation");
    }

  /* serialize the 5-tuple and the perturbation in buf */
  uint8_t buf[41];
  src.Serialize (buf);
  dest.Serialize (buf + 16);
  buf[32] = prot;
  buf[33] = (srcPort >> 8) & 0xff;
  buf[34] = srcPort & 0xff;
  buf[35] = (destPort >> 8) & 0xff;
  buf[36] = destPort & 0xff;
  buf[37] = (perturbation >> 24) & 0xff;
  buf[38] = (perturbation >> 16) & 0xff;
  buf[39] = (perturbation >> 8) & 0xff;
  buf[40] = perturbation & 0xff;

  // Linux calculates jhash2 (jenkins hash), we calculate murmur3 because it is
  // already available in ns-3
  uint32_t hash = Hash32 ((char*) buf, 41);

  NS_LOG_DEBUG ("Found Ipv6 packet; hash of the five tuple " << hash);

  return hash;
}

} // namespace ns3
