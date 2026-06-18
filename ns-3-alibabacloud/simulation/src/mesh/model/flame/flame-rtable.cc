/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 IITP RAS
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
 * Author: Kirill Andreev <andreev@iitp.ru>
 */
#include "ns3/assert.h"
#include "ns3/simulator.h"
#include "ns3/test.h"
#include "ns3/log.h"

#include "flame-rtable.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FlameRtable");

namespace flame {
  
NS_OBJECT_ENSURE_REGISTERED (FlameRtable);

TypeId
FlameRtable::GetTypeId ()
{
  static TypeId tid =
    TypeId ("ns3::flame::FlameRtable")
    .SetParent<Object> ()
    .SetGroupName ("Mesh")
    .AddConstructor<FlameRtable> ()
    .AddAttribute ( "Lifetime",
                    "The lifetime of the routing entry",
                    TimeValue (Seconds (120)), MakeTimeAccessor (
                      &FlameRtable::m_lifetime),
                    MakeTimeChecker ()
                    )
  ;
  return tid;
}
FlameRtable::FlameRtable () :
  m_lifetime (Seconds (120))
{
}
FlameRtable::~FlameRtable ()
{
}
void
FlameRtable::DoDispose ()
{
  m_routes.clear ();
}
void
FlameRtable::AddPath (const Mac48Address destination, const Mac48Address retransmitter,
                      const uint32_t interface, const uint8_t cost, const uint16_t seqnum)
{
  std::map<Mac48Address, Route>::iterator i = m_routes.find (destination);
  if (i == m_routes.end ())
    {
      Route newroute;
      newroute.cost = cost;
      newroute.retransmitter = retransmitter;
      newroute.interface = interface;
      newroute.whenExpire = Simulator::Now () + m_lifetime;
      newroute.seqnum = seqnum;
      m_routes[destination] = newroute;
      return;
    }
  i->second.seqnum = seqnum;
  NS_ASSERT (i != m_routes.end ());
  i->second.retransmitter = retransmitter;
  i->second.interface = interface;
  i->second.cost = cost;
  i->second.whenExpire = Simulator::Now () + m_lifetime;
}
FlameRtable::LookupResult
FlameRtable::Lookup (Mac48Address destination)
{
  std::map<Mac48Address, Route>::iterator i = m_routes.find (destination);
  if (i == m_routes.end ())
    {
      return LookupResult ();
    }
  if ((i->second.whenExpire < Simulator::Now ()))
    {
      NS_LOG_DEBUG ("Route has expired, sorry.");
      m_routes.erase (i);
      return LookupResult ();
    }
  return LookupResult (i->second.retransmitter, i->second.interface, i->second.cost, i->second.seqnum);
}
bool
FlameRtable::LookupResult::operator== (const FlameRtable::LookupResult & o) const
{
  return (retransmitter == o.retransmitter && ifIndex == o.ifIndex && cost == o.cost && seqnum == o.seqnum);
}

bool
FlameRtable::LookupResult::IsValid () const
{
  return !(retransmitter == Mac48Address::GetBroadcast () && ifIndex == INTERFACE_ANY && cost == MAX_COST
           && seqnum == 0);
}

} // namespace flame
} // namespace ns3
