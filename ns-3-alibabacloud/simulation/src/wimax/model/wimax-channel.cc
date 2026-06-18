/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007,2008 INRIA
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
 * Author: Jahanzeb Farooq <jahanzeb.farooq@sophia.inria.fr>
 */

#include "ns3/assert.h"
#include "ns3/net-device.h"
#include "wimax-channel.h"
#include "wimax-phy.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WimaxChannel");

NS_OBJECT_ENSURE_REGISTERED (WimaxChannel);

TypeId WimaxChannel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::WimaxChannel")
    .SetParent<Channel> ()
    .SetGroupName("Wimax");
  return tid;
}

WimaxChannel::WimaxChannel (void)
{
}

WimaxChannel::~WimaxChannel (void)
{
}

void
WimaxChannel::Attach (Ptr<WimaxPhy> phy)
{
  DoAttach (phy);
}

std::size_t
WimaxChannel::GetNDevices (void) const
{
  return DoGetNDevices ();
}

Ptr<NetDevice>
WimaxChannel::GetDevice (std::size_t index) const
{
  return DoGetDevice (index);
}

} // namespace ns3
