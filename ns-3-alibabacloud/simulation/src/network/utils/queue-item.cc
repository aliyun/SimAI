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
 *
 * Author: Stefano Avallone <stefano.avallone@unina.it>
 */

#include "queue-item.h"
#include "ns3/packet.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("QueueItem");

QueueItem::QueueItem (Ptr<Packet> p)
{
  NS_LOG_FUNCTION (this << p);
  m_packet = p;
}

QueueItem::~QueueItem ()
{
  NS_LOG_FUNCTION (this);
  m_packet = 0;
}

Ptr<Packet>
QueueItem::GetPacket (void) const
{
  NS_LOG_FUNCTION (this);
  return m_packet;
}

uint32_t
QueueItem::GetSize (void) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_packet != 0);
  return m_packet->GetSize ();
}

bool
QueueItem::GetUint8Value (QueueItem::Uint8Values field, uint8_t& value) const
{
  NS_LOG_FUNCTION (this);
  return false;
}

void
QueueItem::Print (std::ostream& os) const
{
  os << GetPacket();
}

std::ostream & operator << (std::ostream &os, const QueueItem &item)
{
  item.Print (os);
  return os;
}


QueueDiscItem::QueueDiscItem (Ptr<Packet> p, const Address& addr, uint16_t protocol)
  : QueueItem (p),
    m_address (addr),
    m_protocol (protocol),
    m_txq (0)
{
  NS_LOG_FUNCTION (this << p << addr << protocol);
}

QueueDiscItem::~QueueDiscItem()
{
  NS_LOG_FUNCTION (this);
}

Address
QueueDiscItem::GetAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return m_address;
}

uint16_t
QueueDiscItem::GetProtocol (void) const
{
  NS_LOG_FUNCTION (this);
  return m_protocol;
}

uint8_t
QueueDiscItem::GetTxQueueIndex (void) const
{
  NS_LOG_FUNCTION (this);
  return m_txq;
}

void
QueueDiscItem::SetTxQueueIndex (uint8_t txq)
{
  NS_LOG_FUNCTION (this << (uint16_t) txq);
  m_txq = txq;
}

Time
QueueDiscItem::GetTimeStamp (void) const
{
  NS_LOG_FUNCTION (this);
  return m_tstamp;
}

void
QueueDiscItem::SetTimeStamp (Time t)
{
  NS_LOG_FUNCTION (this << t);
  m_tstamp = t;
}

void
QueueDiscItem::Print (std::ostream& os) const
{
  os << GetPacket () << " "
     << "Dst addr " << m_address << " "
     << "proto " << m_protocol << " "
     << "txq " << +m_txq
  ;
}

uint32_t
QueueDiscItem::Hash (uint32_t perturbation) const
{
  NS_LOG_WARN ("The Hash method should be redefined by subclasses");
  return 0;
}

} // namespace ns3
