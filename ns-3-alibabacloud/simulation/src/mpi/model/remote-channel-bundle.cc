/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright 2013. Lawrence Livermore National Security, LLC.
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
 * Author: Steven Smith <smith84@llnl.gov>
 *
 */

/**
 * \file
 * \ingroup mpi
 * Implementation of class ns3::RemoteChannelBundle.
 */

#include "remote-channel-bundle.h"

#include "null-message-mpi-interface.h"
#include "null-message-simulator-impl.h"

#include <ns3/simulator.h>

namespace ns3 {


TypeId RemoteChannelBundle::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RemoteChannelBundle")
    .SetParent<Object> ()
    .SetGroupName ("Mpi")
    .AddConstructor <RemoteChannelBundle> ();
  return tid;
}

RemoteChannelBundle::RemoteChannelBundle ()
  : m_remoteSystemId (UINT32_MAX),
    m_guaranteeTime (0),
    m_delay (Time::Max ())
{
}

RemoteChannelBundle::RemoteChannelBundle (const uint32_t remoteSystemId)
  : m_remoteSystemId (remoteSystemId),
    m_guaranteeTime (0),
    m_delay (Time::Max ())
{
}

void
RemoteChannelBundle::AddChannel (Ptr<Channel> channel, Time delay)
{
  m_channels[channel->GetId ()] = channel;
  m_delay = ns3::Min (m_delay, delay);
}

uint32_t
RemoteChannelBundle::GetSystemId () const
{
  return m_remoteSystemId;
}

Time
RemoteChannelBundle::GetGuaranteeTime (void) const
{
  return m_guaranteeTime;
}

void
RemoteChannelBundle::SetGuaranteeTime (Time time)
{
  NS_ASSERT (time >= Simulator::Now ());

  m_guaranteeTime = time;
}

Time
RemoteChannelBundle::GetDelay (void) const
{
  return m_delay;
}

void
RemoteChannelBundle::SetEventId (EventId id)
{
  m_nullEventId = id;
}

EventId
RemoteChannelBundle::GetEventId (void) const
{
  return m_nullEventId;
}

std::size_t
RemoteChannelBundle::GetSize (void) const
{
  return m_channels.size ();
}

void 
RemoteChannelBundle::Send(Time time)
{
  NullMessageMpiInterface::SendNullMessage (time, this);  
}

std::ostream& operator<< (std::ostream& out, ns3::RemoteChannelBundle& bundle )
{
  out << "RemoteChannelBundle Rank = " << bundle.m_remoteSystemId
      << ", GuaranteeTime = "  << bundle.m_guaranteeTime
      << ", Delay = " << bundle.m_delay << std::endl;

  for (auto element : bundle.m_channels)
    {
      out << "\t" << element.second << std::endl;
    }
  
  return out;
}


} // namespace ns3

