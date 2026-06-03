/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 */

#include "ipv4-route.h"
#include "ns3/net-device.h"
#include "ns3/assert.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Ipv4Route");

Ipv4Route::Ipv4Route ()
{
  NS_LOG_FUNCTION (this);
}

void
Ipv4Route::SetDestination (Ipv4Address dest)
{
  NS_LOG_FUNCTION (this << dest);
  m_dest = dest;
}

Ipv4Address
Ipv4Route::GetDestination (void) const
{
  NS_LOG_FUNCTION (this);
  return m_dest;
}

void
Ipv4Route::SetSource (Ipv4Address src)
{
  NS_LOG_FUNCTION (this << src);
  m_source = src;
}

Ipv4Address
Ipv4Route::GetSource (void) const
{
  NS_LOG_FUNCTION (this);
  return m_source;
}

void
Ipv4Route::SetGateway (Ipv4Address gw)
{
  NS_LOG_FUNCTION (this << gw);
  m_gateway = gw;
}

Ipv4Address
Ipv4Route::GetGateway (void) const
{
  NS_LOG_FUNCTION (this);
  return m_gateway;
}

void
Ipv4Route::SetOutputDevice (Ptr<NetDevice> outputDevice)
{
  NS_LOG_FUNCTION (this << outputDevice);
  m_outputDevice = outputDevice;
}

Ptr<NetDevice>
Ipv4Route::GetOutputDevice (void) const
{
  NS_LOG_FUNCTION (this);
  return m_outputDevice;
}

std::ostream& operator<< (std::ostream& os, Ipv4Route const& route)
{
  os << "source=" << route.GetSource () << " dest="<< route.GetDestination () <<" gw=" << route.GetGateway ();
  return os;
}

Ipv4MulticastRoute::Ipv4MulticastRoute ()
{
  NS_LOG_FUNCTION (this);
  m_ttls.clear ();
}

void 
Ipv4MulticastRoute::SetGroup (const Ipv4Address group)
{
  NS_LOG_FUNCTION (this << group);
  m_group = group;
}

Ipv4Address 
Ipv4MulticastRoute::GetGroup (void) const
{
  NS_LOG_FUNCTION (this);
  return m_group;
}

void 
Ipv4MulticastRoute::SetOrigin (const Ipv4Address origin)
{
  NS_LOG_FUNCTION (this << origin);
  m_origin = origin;
}

Ipv4Address 
Ipv4MulticastRoute::GetOrigin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_origin;
}

void 
Ipv4MulticastRoute::SetParent (uint32_t parent)
{
  NS_LOG_FUNCTION (this << parent);
  m_parent = parent;
}

uint32_t 
Ipv4MulticastRoute::GetParent (void) const
{
  NS_LOG_FUNCTION (this);
  return m_parent;
}

void 
Ipv4MulticastRoute::SetOutputTtl (uint32_t oif, uint32_t ttl)
{
  NS_LOG_FUNCTION (this << oif << ttl);
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

std::map<uint32_t, uint32_t>
Ipv4MulticastRoute::GetOutputTtlMap () const
{
  NS_LOG_FUNCTION (this);
  return(m_ttls);
}

} // namespace ns3
