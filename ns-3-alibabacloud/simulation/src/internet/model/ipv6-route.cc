/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007-2009 Strasbourg University
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
 * Author: Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
 */

#include <iostream>

#include "ns3/net-device.h"
#include "ipv6-route.h"

namespace ns3
{

Ipv6Route::Ipv6Route ()
{
}

Ipv6Route::~Ipv6Route ()
{
}

void Ipv6Route::SetDestination (Ipv6Address dest)
{
  m_dest = dest;
}

Ipv6Address Ipv6Route::GetDestination () const
{
  return m_dest;
}

void Ipv6Route::SetSource (Ipv6Address src)
{
  m_source = src;
}

Ipv6Address Ipv6Route::GetSource () const
{
  return m_source;
}

void Ipv6Route::SetGateway (Ipv6Address gw)
{
  m_gateway = gw;
}

Ipv6Address Ipv6Route::GetGateway () const
{
  return m_gateway;
}

void Ipv6Route::SetOutputDevice (Ptr<NetDevice> outputDevice)
{
  m_outputDevice = outputDevice;
}

Ptr<NetDevice> Ipv6Route::GetOutputDevice () const
{
  return m_outputDevice;
}

std::ostream& operator<< (std::ostream& os, Ipv6Route const& route)
{
  os << "source=" << route.GetSource () << " dest="<< route.GetDestination () <<" gw=" << route.GetGateway ();
  return os;
}

Ipv6MulticastRoute::Ipv6MulticastRoute ()
{
  m_ttls.clear ();
}

Ipv6MulticastRoute::~Ipv6MulticastRoute ()
{
}

void Ipv6MulticastRoute::SetGroup (const Ipv6Address group)
{
  m_group = group;
}

Ipv6Address Ipv6MulticastRoute::GetGroup () const
{
  return m_group;
}

void Ipv6MulticastRoute::SetOrigin (const Ipv6Address origin)
{
  m_origin = origin;
}

Ipv6Address Ipv6MulticastRoute::GetOrigin () const
{
  return m_origin;
}

void Ipv6MulticastRoute::SetParent (uint32_t parent)
{
  m_parent = parent;
}

uint32_t Ipv6MulticastRoute::GetParent () const
{
  return m_parent;
}

void Ipv6MulticastRoute::SetOutputTtl (uint32_t oif, uint32_t ttl)
{
  if (ttl >= MAX_TTL)
    {
      // This TTL value effectively disables the interface
      std::map<uint32_t, uint32_t>::iterator iter;
      iter = m_ttls.find (oif);
      if (iter != m_ttls.end ())
        {
          m_ttls.erase (iter);
        }
    }
  else
    {
      m_ttls[oif] = ttl;
    }
}

std::map<uint32_t, uint32_t> Ipv6MulticastRoute::GetOutputTtlMap () const
{
  return(m_ttls);
}

std::ostream& operator<< (std::ostream& os, Ipv6MulticastRoute const& route)
{
  os << "origin=" << route.GetOrigin () << " group="<< route.GetGroup () <<" parent=" << route.GetParent ();
  return os;
}

} /* namespace ns3 */

