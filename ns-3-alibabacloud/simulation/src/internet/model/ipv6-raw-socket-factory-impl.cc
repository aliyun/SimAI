/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

#include "ns3/socket.h"

#include "ipv6-raw-socket-factory-impl.h"
#include "ipv6-l3-protocol.h"

namespace ns3
{

Ptr<Socket> Ipv6RawSocketFactoryImpl::CreateSocket ()
{
  Ptr<Ipv6L3Protocol> ipv6 = GetObject<Ipv6L3Protocol> ();
  Ptr<Socket> socket = ipv6->CreateRawSocket ();
  return socket;
}

} /* namespace ns3 */

