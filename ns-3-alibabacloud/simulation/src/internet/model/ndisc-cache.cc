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

#include "ns3/log.h"
#include "ns3/uinteger.h"
#include "ns3/node.h"
#include "ns3/names.h"

#include "ipv6-l3-protocol.h" 
#include "icmpv6-l4-protocol.h"
#include "ndisc-cache.h"
#include "ipv6-interface.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("NdiscCache");

NS_OBJECT_ENSURE_REGISTERED (NdiscCache);

TypeId NdiscCache::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::NdiscCache")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddAttribute ("UnresolvedQueueSize",
                   "Size of the queue for packets pending an NA reply.",
                   UintegerValue (DEFAULT_UNRES_QLEN),
                   MakeUintegerAccessor (&NdiscCache::m_unresQlen),
                   MakeUintegerChecker<uint32_t> ())
  ;
  return tid;
} 

NdiscCache::NdiscCache ()
{
  NS_LOG_FUNCTION (this);
}

NdiscCache::~NdiscCache ()
{
  NS_LOG_FUNCTION (this);
  Flush ();
}

void NdiscCache::DoDispose ()
{
  NS_LOG_FUNCTION (this);
  Flush ();
  m_device = 0;
  m_interface = 0;
  m_icmpv6 = 0;
  Object::DoDispose ();
}

void NdiscCache::SetDevice (Ptr<NetDevice> device, Ptr<Ipv6Interface> interface, Ptr<Icmpv6L4Protocol> icmpv6)
{
  NS_LOG_FUNCTION (this << device << interface);
  m_device = device;
  m_interface = interface;
  m_icmpv6 = icmpv6;
}

Ptr<Ipv6Interface> NdiscCache::GetInterface () const
{
  NS_LOG_FUNCTION (this);
  return m_interface;
}

Ptr<NetDevice> NdiscCache::GetDevice () const
{
  NS_LOG_FUNCTION (this);
  return m_device;
}

NdiscCache::Entry* NdiscCache::Lookup (Ipv6Address dst)
{
  NS_LOG_FUNCTION (this << dst);

  if (m_ndCache.find (dst) != m_ndCache.end ())
    {
      NdiscCache::Entry* entry = m_ndCache[dst];
      NS_LOG_LOGIC ("Found an entry: " << *entry);

      return entry;
    }
  NS_LOG_LOGIC ("Nothing found");
  return 0;
}

std::list<NdiscCache::Entry*> NdiscCache::LookupInverse (Address dst)
{
  NS_LOG_FUNCTION (this << dst);

  std::list<NdiscCache::Entry *> entryList;
  for (CacheI i = m_ndCache.begin (); i != m_ndCache.end (); i++)
    {
      NdiscCache::Entry *entry = (*i).second;
      if (entry->GetMacAddress () == dst)
        {
          NS_LOG_LOGIC ("Found an entry:" << (*entry));
          entryList.push_back (entry);
        }
    }
  return entryList;
}


NdiscCache::Entry* NdiscCache::Add (Ipv6Address to)
{
  NS_LOG_FUNCTION (this << to);
  NS_ASSERT (m_ndCache.find (to) == m_ndCache.end ());

  NdiscCache::Entry* entry = new NdiscCache::Entry (this);
  entry->SetIpv6Address (to);
  m_ndCache[to] = entry;
  return entry;
}

void NdiscCache::Remove (NdiscCache::Entry* entry)
{
  NS_LOG_FUNCTION (this << entry);

  for (CacheI i = m_ndCache.begin (); i != m_ndCache.end (); i++)
    {
      if ((*i).second == entry)
        {
          m_ndCache.erase (i);
          entry->ClearWaitingPacket ();
          delete entry;
          return;
        }
    }
}

void NdiscCache::Flush ()
{
  NS_LOG_FUNCTION (this);

  for (CacheI i = m_ndCache.begin (); i != m_ndCache.end (); i++)
    {
      delete (*i).second; /* delete the pointer NdiscCache::Entry */
    }

  m_ndCache.erase (m_ndCache.begin (), m_ndCache.end ());
}

void NdiscCache::SetUnresQlen (uint32_t unresQlen)
{
  NS_LOG_FUNCTION (this << unresQlen);
  m_unresQlen = unresQlen;
}

uint32_t NdiscCache::GetUnresQlen ()
{
  NS_LOG_FUNCTION (this);
  return m_unresQlen;
}

void NdiscCache::PrintNdiscCache (Ptr<OutputStreamWrapper> stream)
{
  NS_LOG_FUNCTION (this << stream);
  std::ostream* os = stream->GetStream ();

  for (CacheI i = m_ndCache.begin (); i != m_ndCache.end (); i++)
    {
      *os << i->first << " dev ";
      std::string found = Names::FindName (m_device);
      if (Names::FindName (m_device) != "")
        {
          *os << found;
        }
      else
        {
          *os << static_cast<int> (m_device->GetIfIndex ());
        }

      *os << " lladdr " << i->second->GetMacAddress ();

      if (i->second->IsReachable ())
        {
          *os << " REACHABLE\n";
        }
      else if (i->second->IsDelay ())
        {
          *os << " DELAY\n";
        }
      else if (i->second->IsIncomplete ())
        {
          *os << " INCOMPLETE\n";
        }
      else if (i->second->IsProbe ())
        {
          *os << " PROBE\n";
        }
      else if (i->second->IsStale ())
        {
          *os << " STALE\n";
        }
      else if (i->second->IsPermanent ())
        {
          *os << " PERMANENT\n";
        }
      else
        {
          NS_FATAL_ERROR ("Test for possibly unreachable code-- please file a bug report, with a test case, if this is ever hit");
        }
    }
}

NdiscCache::Entry::Entry (NdiscCache* nd)
  : m_ndCache (nd),
    m_waiting (),
    m_router (false),
    m_nudTimer (Timer::CANCEL_ON_DESTROY),
    m_lastReachabilityConfirmation (Seconds (0.0)),
    m_nsRetransmit (0)
{
  NS_LOG_FUNCTION (this);
}

void NdiscCache::Entry::SetRouter (bool router)
{
  NS_LOG_FUNCTION (this << router);
  m_router = router;
}

bool NdiscCache::Entry::IsRouter () const
{
  NS_LOG_FUNCTION (this);
  return m_router;
}

void NdiscCache::Entry::AddWaitingPacket (Ipv6PayloadHeaderPair p)
{
  NS_LOG_FUNCTION (this << p.second << p.first);

  if (m_waiting.size () >= m_ndCache->GetUnresQlen ())
    {
      /* we store only m_unresQlen packet => first packet in first packet remove */
      /** \todo report packet as 'dropped' */
      m_waiting.pop_front ();
    }
  m_waiting.push_back (p);
}

void NdiscCache::Entry::ClearWaitingPacket ()
{
  NS_LOG_FUNCTION (this);
  /** \todo report packets as 'dropped' */
  m_waiting.clear ();
}

void NdiscCache::Entry::FunctionReachableTimeout ()
{
  NS_LOG_FUNCTION (this);
  this->MarkStale ();
}

void NdiscCache::Entry::FunctionRetransmitTimeout ()
{
  NS_LOG_FUNCTION (this);
  Ipv6Address addr;

  /* determine source address */
  if (m_ipv6Address.IsLinkLocal ())
    {
      addr = m_ndCache->GetInterface ()->GetLinkLocalAddress ().GetAddress ();
    }
  else if (!m_ipv6Address.IsAny ())
    {
      addr = m_ndCache->GetInterface ()->GetAddressMatchingDestination (m_ipv6Address).GetAddress ();

      if (addr.IsAny ()) /* maybe address has expired */
        {
          /* delete the entry */
          m_ndCache->Remove (this);
          return;
        }
    }

  if (m_nsRetransmit < m_ndCache->m_icmpv6->GetMaxMulticastSolicit ())
    {
      m_nsRetransmit++;

      m_ndCache->m_icmpv6->SendNS (addr, Ipv6Address::MakeSolicitedAddress (m_ipv6Address), m_ipv6Address, m_ndCache->GetDevice ()->GetAddress ());
      /* arm the timer again */
      StartRetransmitTimer ();
    }
  else
    {
      Ipv6PayloadHeaderPair malformedPacket = m_waiting.front ();
      if (malformedPacket.first == 0)
        {
          malformedPacket.first = Create<Packet> ();
        }
      else
        {
          malformedPacket.first->AddHeader (malformedPacket.second);
        }

      m_ndCache->m_icmpv6->SendErrorDestinationUnreachable (malformedPacket.first, addr, Icmpv6Header::ICMPV6_ADDR_UNREACHABLE);

      /* delete the entry */
      m_ndCache->Remove (this);
    }
}

void NdiscCache::Entry::FunctionDelayTimeout ()
{
  NS_LOG_FUNCTION (this);
  Ipv6Address addr;

  this->MarkProbe ();

  if (m_ipv6Address.IsLinkLocal ())
    {
      addr = m_ndCache->GetInterface ()->GetLinkLocalAddress ().GetAddress ();
    }
  else if (!m_ipv6Address.IsAny ())
    {
      addr = m_ndCache->GetInterface ()->GetAddressMatchingDestination (m_ipv6Address).GetAddress ();
      if (addr.IsAny ()) /* maybe address has expired */
        {
          /* delete the entry */
          m_ndCache->Remove (this);
          return;
        }
    }
  else
    {
      /* should not happen */
      return;
    }

  Ipv6PayloadHeaderPair p = m_ndCache->m_icmpv6->ForgeNS (addr, m_ipv6Address, m_ipv6Address, m_ndCache->GetDevice ()->GetAddress ());
  p.first->AddHeader (p.second);
  m_ndCache->GetDevice ()->Send (p.first, this->GetMacAddress (), Ipv6L3Protocol::PROT_NUMBER);

  m_nsRetransmit = 1;
  StartProbeTimer ();
}

void NdiscCache::Entry::FunctionProbeTimeout ()
{
  NS_LOG_FUNCTION (this);

  if (m_nsRetransmit < m_ndCache->m_icmpv6->GetMaxUnicastSolicit ())
    {
      m_nsRetransmit++;

      Ipv6Address addr;

      if (m_ipv6Address.IsLinkLocal ())
        {
          addr = m_ndCache->GetInterface ()->GetLinkLocalAddress ().GetAddress ();
        }
      else if (!m_ipv6Address.IsAny ())
        {
          addr = m_ndCache->GetInterface ()->GetAddressMatchingDestination (m_ipv6Address).GetAddress ();
          if (addr.IsAny ()) /* maybe address has expired */
            {
              /* delete the entry */
              m_ndCache->Remove (this);
              return;
            }
        }
      else
        {
          /* should not happen */
          return;
        }

      /* icmpv6->SendNS (m_ndCache->GetInterface ()->GetLinkLocalAddress (), m_ipv6Address, m_ipv6Address, m_ndCache->GetDevice ()->GetAddress ()); */
      Ipv6PayloadHeaderPair p = m_ndCache->m_icmpv6->ForgeNS (addr, m_ipv6Address, m_ipv6Address, m_ndCache->GetDevice ()->GetAddress ());
      p.first->AddHeader (p.second);
      m_ndCache->GetDevice ()->Send (p.first, this->GetMacAddress (), Ipv6L3Protocol::PROT_NUMBER);

      /* arm the timer again */
      StartProbeTimer ();
    }
  else
    {
      /* delete the entry */
      m_ndCache->Remove (this);
    }
}

void NdiscCache::Entry::SetIpv6Address (Ipv6Address ipv6Address)
{
  NS_LOG_FUNCTION (this << ipv6Address);
  m_ipv6Address = ipv6Address;
}

Ipv6Address NdiscCache::Entry::GetIpv6Address (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ipv6Address;
}

Time NdiscCache::Entry::GetLastReachabilityConfirmation () const
{
  NS_LOG_FUNCTION (this);
  return m_lastReachabilityConfirmation;
}

void NdiscCache::Entry::StartReachableTimer ()
{
  NS_LOG_FUNCTION (this);
  if (m_nudTimer.IsRunning ())
    {
      m_nudTimer.Cancel ();
    }

  m_lastReachabilityConfirmation = Simulator::Now ();
  m_nudTimer.SetFunction (&NdiscCache::Entry::FunctionReachableTimeout, this);
  m_nudTimer.SetDelay (m_ndCache->m_icmpv6->GetReachableTime ());
  m_nudTimer.Schedule ();
}

void NdiscCache::Entry::UpdateReachableTimer ()
{
  NS_LOG_FUNCTION (this);

  if (m_state == REACHABLE)
    {
      m_lastReachabilityConfirmation = Simulator::Now ();
      if (m_nudTimer.IsRunning ())
        {
          m_nudTimer.Cancel ();
        }
      m_nudTimer.Schedule ();
    }
}

void NdiscCache::Entry::StartProbeTimer ()
{
  NS_LOG_FUNCTION (this);
  if (m_nudTimer.IsRunning ())
    {
      m_nudTimer.Cancel ();
    }

  m_nudTimer.SetFunction (&NdiscCache::Entry::FunctionProbeTimeout, this);
  m_nudTimer.SetDelay (m_ndCache->m_icmpv6->GetRetransmissionTime ());
  m_nudTimer.Schedule ();
}

void NdiscCache::Entry::StartDelayTimer ()
{
  NS_LOG_FUNCTION (this);
  if (m_nudTimer.IsRunning ())
    {
      m_nudTimer.Cancel ();
    }

  m_nudTimer.SetFunction (&NdiscCache::Entry::FunctionDelayTimeout, this);
  m_nudTimer.SetDelay (m_ndCache->m_icmpv6->GetDelayFirstProbe ());
  m_nudTimer.Schedule ();
}

void NdiscCache::Entry::StartRetransmitTimer ()
{
  NS_LOG_FUNCTION (this);
  if (m_nudTimer.IsRunning ())
    {
      m_nudTimer.Cancel ();
    }

  m_nudTimer.SetFunction (&NdiscCache::Entry::FunctionRetransmitTimeout, this);
  m_nudTimer.SetDelay (m_ndCache->m_icmpv6->GetRetransmissionTime ());
  m_nudTimer.Schedule ();
}

void NdiscCache::Entry::StopNudTimer ()
{
  NS_LOG_FUNCTION (this);
  m_nudTimer.Cancel ();
  m_nsRetransmit = 0;
}

void NdiscCache::Entry::MarkIncomplete (Ipv6PayloadHeaderPair p)
{
  NS_LOG_FUNCTION (this << p.second << p.first);
  m_state = INCOMPLETE;

  if (p.first)
    {
      m_waiting.push_back (p);
    }
}

std::list<NdiscCache::Ipv6PayloadHeaderPair> NdiscCache::Entry::MarkReachable (Address mac)
{
  NS_LOG_FUNCTION (this << mac);
  m_state = REACHABLE;
  m_macAddress = mac;
  return m_waiting;
}

void NdiscCache::Entry::MarkProbe ()
{
  NS_LOG_FUNCTION (this);
  m_state = PROBE;
}

void NdiscCache::Entry::MarkStale ()
{
  NS_LOG_FUNCTION (this);
  m_state = STALE;
}

void NdiscCache::Entry::MarkReachable ()
{
  NS_LOG_FUNCTION (this);
  m_state = REACHABLE;
}

std::list<NdiscCache::Ipv6PayloadHeaderPair> NdiscCache::Entry::MarkStale (Address mac)
{
  NS_LOG_FUNCTION (this << mac);
  m_state = STALE;
  m_macAddress = mac;
  return m_waiting;
}

void NdiscCache::Entry::MarkDelay ()
{
  NS_LOG_FUNCTION (this);
  m_state = DELAY;
}

void NdiscCache::Entry::MarkPermanent ()
{
  NS_LOG_FUNCTION (this);
  StopNudTimer ();
  m_state = PERMANENT;
}

bool NdiscCache::Entry::IsStale () const
{
  NS_LOG_FUNCTION (this);
  return (m_state == STALE);
}

bool NdiscCache::Entry::IsReachable () const
{
  NS_LOG_FUNCTION (this);
  return (m_state == REACHABLE);
}

bool NdiscCache::Entry::IsDelay () const
{
  NS_LOG_FUNCTION (this);
  return (m_state == DELAY);
}

bool NdiscCache::Entry::IsIncomplete () const
{
  NS_LOG_FUNCTION (this);
  return (m_state == INCOMPLETE);
}

bool NdiscCache::Entry::IsProbe () const
{
  NS_LOG_FUNCTION (this);
  return (m_state == PROBE);
}

bool NdiscCache::Entry::IsPermanent () const
{
  NS_LOG_FUNCTION (this);
  return (m_state == PERMANENT);
}

Address NdiscCache::Entry::GetMacAddress () const
{
  NS_LOG_FUNCTION (this);
  return m_macAddress;
}

void NdiscCache::Entry::SetMacAddress (Address mac)
{
  NS_LOG_FUNCTION (this << mac << int(m_state));
  m_macAddress = mac;
}

void NdiscCache::Entry::Print (std::ostream &os) const
{
  os << m_ipv6Address << " lladdr " << m_macAddress << " state ";
  switch (m_state)
  {
    case INCOMPLETE:
      os << "INCOMPLETE";
      break;
    case REACHABLE:
      os << "REACHABLE";
      break;
    case STALE:
      os << "STALE";
      break;
    case DELAY:
      os << "DELAY";
      break;
    case PROBE:
      os << "PROBE";
      break;
    case PERMANENT:
      os << "PERMANENT";
      break;
  }
}

std::ostream& operator << (std::ostream& os, NdiscCache::Entry const& entry)
{
  entry.Print (os);
  return os;
}

} /* namespace ns3 */

