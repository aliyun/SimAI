/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006 INRIA
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
#include "ns3/assert.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/uinteger.h"
#include "ns3/log.h"
#include "ns3/node.h"
#include "ns3/trace-source-accessor.h"
#include "ns3/names.h"

#include "arp-cache.h"
#include "arp-header.h"
#include "ipv4-interface.h"
#include "ipv4-header.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ArpCache");

NS_OBJECT_ENSURE_REGISTERED (ArpCache);

TypeId
ArpCache::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ArpCache")
    .SetParent<Object> ()
    .SetGroupName ("Internet")
    .AddAttribute ("AliveTimeout",
                   "When this timeout expires, "
                   "the matching cache entry needs refreshing",
                   TimeValue (Seconds (120)),
                   MakeTimeAccessor (&ArpCache::m_aliveTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("DeadTimeout",
                   "When this timeout expires, "
                   "a new attempt to resolve the matching entry is made",
                   TimeValue (Seconds (100)),
                   MakeTimeAccessor (&ArpCache::m_deadTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("WaitReplyTimeout",
                   "When this timeout expires, "
                   "the cache entries will be scanned and "
                   "entries in WaitReply state will resend ArpRequest "
                   "unless MaxRetries has been exceeded, "
                   "in which case the entry is marked dead",
                   TimeValue (Seconds (1)),
                   MakeTimeAccessor (&ArpCache::m_waitReplyTimeout),
                   MakeTimeChecker ())
    .AddAttribute ("MaxRetries",
                   "Number of retransmissions of ArpRequest "
                   "before marking dead",
                   UintegerValue (3),
                   MakeUintegerAccessor (&ArpCache::m_maxRetries),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("PendingQueueSize",
                   "The size of the queue for packets pending an arp reply.",
                   UintegerValue (3),
                   MakeUintegerAccessor (&ArpCache::m_pendingQueueSize),
                   MakeUintegerChecker<uint32_t> ())
    .AddTraceSource ("Drop",
                     "Packet dropped due to ArpCache entry "
                     "in WaitReply expiring.",
                     MakeTraceSourceAccessor (&ArpCache::m_dropTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

ArpCache::ArpCache ()
  : m_device (0),
    m_interface (0)
{
  NS_LOG_FUNCTION (this);
}

ArpCache::~ArpCache ()
{
  NS_LOG_FUNCTION (this);
}

void
ArpCache::DoDispose (void)
{
  NS_LOG_FUNCTION (this);
  Flush ();
  m_device = 0;
  m_interface = 0;
  if (!m_waitReplyTimer.IsRunning ())
    {
      m_waitReplyTimer.Cancel ();
    }
  Object::DoDispose ();
}

void
ArpCache::SetDevice (Ptr<NetDevice> device, Ptr<Ipv4Interface> interface)
{
  NS_LOG_FUNCTION (this << device << interface);
  m_device = device;
  m_interface = interface;
}

Ptr<NetDevice>
ArpCache::GetDevice (void) const
{
  NS_LOG_FUNCTION (this);
  return m_device;
}

Ptr<Ipv4Interface>
ArpCache::GetInterface (void) const
{
  NS_LOG_FUNCTION (this);
  return m_interface;
}

void
ArpCache::SetAliveTimeout (Time aliveTimeout)
{
  NS_LOG_FUNCTION (this << aliveTimeout);
  m_aliveTimeout = aliveTimeout;
}
void
ArpCache::SetDeadTimeout (Time deadTimeout)
{
  NS_LOG_FUNCTION (this << deadTimeout);
  m_deadTimeout = deadTimeout;
}
void
ArpCache::SetWaitReplyTimeout (Time waitReplyTimeout)
{
  NS_LOG_FUNCTION (this << waitReplyTimeout);
  m_waitReplyTimeout = waitReplyTimeout;
}

Time
ArpCache::GetAliveTimeout (void) const
{
  NS_LOG_FUNCTION (this);
  return m_aliveTimeout;
}
Time
ArpCache::GetDeadTimeout (void) const
{
  NS_LOG_FUNCTION (this);
  return m_deadTimeout;
}
Time
ArpCache::GetWaitReplyTimeout (void) const
{
  NS_LOG_FUNCTION (this);
  return m_waitReplyTimeout;
}

void
ArpCache::SetArpRequestCallback (Callback<void, Ptr<const ArpCache>,
                                          Ipv4Address> arpRequestCallback)
{
  NS_LOG_FUNCTION (this << &arpRequestCallback);
  m_arpRequestCallback = arpRequestCallback;
}

void
ArpCache::StartWaitReplyTimer (void)
{
  NS_LOG_FUNCTION (this);
  if (!m_waitReplyTimer.IsRunning ())
    {
      NS_LOG_LOGIC ("Starting WaitReplyTimer at " << Simulator::Now () << " for " <<
                    m_waitReplyTimeout);
      m_waitReplyTimer = Simulator::Schedule (m_waitReplyTimeout,
                                              &ArpCache::HandleWaitReplyTimeout, this);
    }
}

void
ArpCache::HandleWaitReplyTimeout (void)
{
  NS_LOG_FUNCTION (this);
  ArpCache::Entry* entry;
  bool restartWaitReplyTimer = false;
  for (CacheI i = m_arpCache.begin (); i != m_arpCache.end (); i++)
    {
      entry = (*i).second;
      if (entry != 0 && entry->IsWaitReply ())
        {
          if (entry->GetRetries () < m_maxRetries)
            {
              NS_LOG_LOGIC ("node="<< m_device->GetNode ()->GetId () <<
                            ", ArpWaitTimeout for " << entry->GetIpv4Address () <<
                            " expired -- retransmitting arp request since retries = " <<
                            entry->GetRetries ());
              m_arpRequestCallback (this, entry->GetIpv4Address ());
              restartWaitReplyTimer = true;
              entry->IncrementRetries ();
            }
          else
            {
              NS_LOG_LOGIC ("node="<<m_device->GetNode ()->GetId () <<
                            ", wait reply for " << entry->GetIpv4Address () <<
                            " expired -- drop since max retries exceeded: " <<
                            entry->GetRetries ());
              entry->MarkDead ();
              entry->ClearRetries ();
              Ipv4PayloadHeaderPair pending = entry->DequeuePending ();
              while (pending.first != 0)
                {
                  // add the Ipv4 header for tracing purposes
                  pending.first->AddHeader (pending.second);
                  m_dropTrace (pending.first);
                  pending = entry->DequeuePending ();
                }
            }
        }

    }
  if (restartWaitReplyTimer)
    {
      NS_LOG_LOGIC ("Restarting WaitReplyTimer at " << Simulator::Now ().GetSeconds ());
      m_waitReplyTimer = Simulator::Schedule (m_waitReplyTimeout,
                                              &ArpCache::HandleWaitReplyTimeout, this);
    }
}

void
ArpCache::Flush (void)
{
  NS_LOG_FUNCTION (this);
  for (CacheI i = m_arpCache.begin (); i != m_arpCache.end (); i++)
    {
      delete (*i).second;
    }
  m_arpCache.erase (m_arpCache.begin (), m_arpCache.end ());
  if (m_waitReplyTimer.IsRunning ())
    {
      NS_LOG_LOGIC ("Stopping WaitReplyTimer at " << Simulator::Now ().GetSeconds () << " due to ArpCache flush");
      m_waitReplyTimer.Cancel ();
    }
}

void
ArpCache::PrintArpCache (Ptr<OutputStreamWrapper> stream)
{
  NS_LOG_FUNCTION (this << stream);
  std::ostream* os = stream->GetStream ();

  for (CacheI i = m_arpCache.begin (); i != m_arpCache.end (); i++)
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

      if (i->second->IsAlive ())
        {
          *os << " REACHABLE\n";
        }
      else if (i->second->IsWaitReply ())
        {
          *os << " DELAY\n";
        }
      else if (i->second->IsPermanent ())
	{
	  *os << " PERMANENT\n";
	}
      else
        {
          *os << " STALE\n";
        }
    }
}

std::list<ArpCache::Entry *>
ArpCache::LookupInverse (Address to)
{
  NS_LOG_FUNCTION (this << to);

  std::list<ArpCache::Entry *> entryList;
  for (CacheI i = m_arpCache.begin (); i != m_arpCache.end (); i++)
    {
      ArpCache::Entry *entry = (*i).second;
      if (entry->GetMacAddress () == to)
        {
          entryList.push_back (entry);
        }
    }
  return entryList;
}


ArpCache::Entry *
ArpCache::Lookup (Ipv4Address to)
{
  NS_LOG_FUNCTION (this << to);
  CacheI it = m_arpCache.find (to);
  if (it != m_arpCache.end ())
    {
      return it->second;
    }
  return 0;
}

ArpCache::Entry *
ArpCache::Add (Ipv4Address to)
{
  NS_LOG_FUNCTION (this << to);
  NS_ASSERT (m_arpCache.find (to) == m_arpCache.end ());

  ArpCache::Entry *entry = new ArpCache::Entry (this);
  m_arpCache[to] = entry;
  entry->SetIpv4Address (to);
  return entry;
}

void
ArpCache::Remove (ArpCache::Entry *entry)
{
  NS_LOG_FUNCTION (this << entry);

  for (CacheI i = m_arpCache.begin (); i != m_arpCache.end (); i++)
    {
      if ((*i).second == entry)
        {
          m_arpCache.erase (i);
          entry->ClearPendingPacket (); //clear the pending packets for entry's ipaddress
          delete entry;
          return;
        }
    }
  NS_LOG_WARN ("Entry not found in this ARP Cache");
}

ArpCache::Entry::Entry (ArpCache *arp)
  : m_arp (arp),
    m_state (ALIVE),
    m_retries (0)
{
  NS_LOG_FUNCTION (this << arp);
}


bool
ArpCache::Entry::IsDead (void)
{
  NS_LOG_FUNCTION (this);
  return (m_state == DEAD);
}
bool
ArpCache::Entry::IsAlive (void)
{
  NS_LOG_FUNCTION (this);
  return (m_state == ALIVE);
}
bool
ArpCache::Entry::IsWaitReply (void)
{
  NS_LOG_FUNCTION (this);
  return (m_state == WAIT_REPLY);
}
bool
ArpCache::Entry::IsPermanent (void)
{
  NS_LOG_FUNCTION (this);
  return (m_state == PERMANENT);
}


void
ArpCache::Entry::MarkDead (void)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_state == ALIVE || m_state == WAIT_REPLY || m_state == DEAD);
  m_state = DEAD;
  ClearRetries ();
  UpdateSeen ();
}
void
ArpCache::Entry::MarkAlive (Address macAddress)
{
  NS_LOG_FUNCTION (this << macAddress);
  NS_ASSERT (m_state == WAIT_REPLY);
  m_macAddress = macAddress;
  m_state = ALIVE;
  ClearRetries ();
  UpdateSeen ();
}
void
ArpCache::Entry::MarkPermanent (void)
{
  NS_LOG_FUNCTION (this << m_macAddress);
  NS_ASSERT (!m_macAddress.IsInvalid ());

  m_state = PERMANENT;
  ClearRetries ();
  UpdateSeen ();
}
bool
ArpCache::Entry::UpdateWaitReply (Ipv4PayloadHeaderPair waiting)
{
  NS_LOG_FUNCTION (this << waiting.first);
  NS_ASSERT (m_state == WAIT_REPLY);
  /* We are already waiting for an answer so
   * we dump the previously waiting packet and
   * replace it with this one.
   */
  if (m_pending.size () >= m_arp->m_pendingQueueSize)
    {
      return false;
    }
  m_pending.push_back (waiting);
  return true;
}
void
ArpCache::Entry::MarkWaitReply (Ipv4PayloadHeaderPair waiting)
{
  NS_LOG_FUNCTION (this << waiting.first);
  NS_ASSERT (m_state == ALIVE || m_state == DEAD);
  NS_ASSERT (m_pending.empty ());
  NS_ASSERT_MSG (waiting.first, "Can not add a null packet to the ARP queue");

  m_state = WAIT_REPLY;
  m_pending.push_back (waiting);
  UpdateSeen ();
  m_arp->StartWaitReplyTimer ();
}

Address
ArpCache::Entry::GetMacAddress (void) const
{
  NS_LOG_FUNCTION (this);
  return m_macAddress;
}
void
ArpCache::Entry::SetMacAddress (Address macAddress)
{
  NS_LOG_FUNCTION (this);
  m_macAddress = macAddress;
}
Ipv4Address
ArpCache::Entry::GetIpv4Address (void) const
{
  NS_LOG_FUNCTION (this);
  return m_ipv4Address;
}
void
ArpCache::Entry::SetIpv4Address (Ipv4Address destination)
{
  NS_LOG_FUNCTION (this << destination);
  m_ipv4Address = destination;
}
Time
ArpCache::Entry::GetTimeout (void) const
{
  NS_LOG_FUNCTION (this);
  switch (m_state) {
    case ArpCache::Entry::WAIT_REPLY:
      return m_arp->GetWaitReplyTimeout ();
    case ArpCache::Entry::DEAD:
      return m_arp->GetDeadTimeout ();
    case ArpCache::Entry::ALIVE:
      return m_arp->GetAliveTimeout ();
    case ArpCache::Entry::PERMANENT:
      return Time::Max ();
    default:
      NS_ASSERT (false);
      return Seconds (0);
      /* NOTREACHED */
    }
}
bool
ArpCache::Entry::IsExpired (void) const
{
  NS_LOG_FUNCTION (this);
  Time timeout = GetTimeout ();
  Time delta = Simulator::Now () - m_lastSeen;
  NS_LOG_DEBUG ("delta=" << delta.GetSeconds () << "s");
  if (delta > timeout)
    {
      return true;
    }
  return false;
}
ArpCache::Ipv4PayloadHeaderPair
ArpCache::Entry::DequeuePending (void)
{
  NS_LOG_FUNCTION (this);
  if (m_pending.empty ())
    {
      Ipv4Header h;
      return Ipv4PayloadHeaderPair (0, h);
    }
  else
    {
      Ipv4PayloadHeaderPair p = m_pending.front ();
      m_pending.pop_front ();
      return p;
    }
}
void
ArpCache::Entry::ClearPendingPacket (void)
{
  NS_LOG_FUNCTION (this);
  m_pending.clear ();
}
void
ArpCache::Entry::UpdateSeen (void)
{
  NS_LOG_FUNCTION (this);
  m_lastSeen = Simulator::Now ();
}
uint32_t
ArpCache::Entry::GetRetries (void) const
{
  NS_LOG_FUNCTION (this);
  return m_retries;
}
void
ArpCache::Entry::IncrementRetries (void)
{
  NS_LOG_FUNCTION (this);
  m_retries++;
  UpdateSeen ();
}
void
ArpCache::Entry::ClearRetries (void)
{
  NS_LOG_FUNCTION (this);
  m_retries = 0;
}

} // namespace ns3

