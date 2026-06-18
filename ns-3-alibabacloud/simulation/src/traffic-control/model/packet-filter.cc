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
#include "ns3/integer.h"
#include "queue-disc.h"
#include "packet-filter.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("PacketFilter");

NS_OBJECT_ENSURE_REGISTERED (PacketFilter);

TypeId 
PacketFilter::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::PacketFilter")
    .SetParent<Object> ()
    .SetGroupName ("TrafficControl")
  ;
  return tid;
}

PacketFilter::PacketFilter ()
{
  NS_LOG_FUNCTION (this);
}

PacketFilter::~PacketFilter()
{
  NS_LOG_FUNCTION (this);
}

int32_t
PacketFilter::Classify (Ptr<QueueDiscItem> item) const
{
  NS_LOG_FUNCTION (this << item);

  if (!CheckProtocol (item))
  {
    NS_LOG_LOGIC ("Unable to classify packets of this protocol");
    return PF_NO_MATCH;
  }

  return DoClassify (item);
}

} // namespace ns3
