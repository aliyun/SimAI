/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005, 2009 INRIA
 * Copyright (c) 2009 MIRKO BANCHI
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
 * Author: Mirko Banchi <mk.banchi@gmail.com>
 */

#include "ns3/mac48-address.h"
#include "qos-blocked-destinations.h"

namespace ns3 {

QosBlockedDestinations::QosBlockedDestinations ()
{
}

QosBlockedDestinations::~QosBlockedDestinations ()
{
}

bool
QosBlockedDestinations::IsBlocked (Mac48Address dest, uint8_t tid) const
{
  return m_blockedQosPackets.find ({dest, tid}) != m_blockedQosPackets.end ();
}

void
QosBlockedDestinations::Block (Mac48Address dest, uint8_t tid)
{
  m_blockedQosPackets.insert ({dest, tid});
}

void
QosBlockedDestinations::Unblock (Mac48Address dest, uint8_t tid)
{
  m_blockedQosPackets.erase ({dest, tid});
}

} //namespace ns3
