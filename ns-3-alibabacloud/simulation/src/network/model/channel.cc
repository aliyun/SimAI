/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 University of Washington
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

#include "channel.h"
#include "channel-list.h"
#include "net-device.h"

#include "ns3/log.h"
#include "ns3/uinteger.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Channel");

NS_OBJECT_ENSURE_REGISTERED (Channel);

TypeId 
Channel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Channel")
    .SetParent<Object> ()
    .SetGroupName("Network")
    .AddAttribute ("Id", "The id (unique integer) of this Channel.",
                   TypeId::ATTR_GET,
                   UintegerValue (0),
                   MakeUintegerAccessor (&Channel::m_id),
                   MakeUintegerChecker<uint32_t> ());
  return tid;
}

Channel::Channel ()
  : m_id (0)
{
  NS_LOG_FUNCTION (this);
  m_id = ChannelList::Add (this);
}

Channel::~Channel ()
{
  NS_LOG_FUNCTION (this);
}

uint32_t 
Channel::GetId (void) const
{
  NS_LOG_FUNCTION (this);
  return m_id;
}

} // namespace ns3
