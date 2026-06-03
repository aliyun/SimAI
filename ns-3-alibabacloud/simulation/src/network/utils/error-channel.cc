/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Universita' di Firenze, Italy
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
 * Author: Tommaso Pecorella <tommaso.pecorella@unifi.it>
 */

#include "ns3/simple-net-device.h"
#include "ns3/simulator.h"
#include "ns3/packet.h"
#include "ns3/node.h"
#include "ns3/log.h"
#include "error-channel.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ErrorChannel");

NS_OBJECT_ENSURE_REGISTERED (ErrorChannel);

TypeId
ErrorChannel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ErrorChannel")
    .SetParent<SimpleChannel> ()
    .SetGroupName ("Network")
    .AddConstructor<ErrorChannel> ()
  ;
  return tid;
}

ErrorChannel::ErrorChannel ()
{
  m_jumpingTime = Seconds (0.5);
  m_jumping = false;
  m_jumpingState = 0;
  m_duplicateTime = Seconds (0.1);
  m_duplicate = false;
  m_duplicateState = 0;
}

void
ErrorChannel::SetJumpingTime (Time delay)
{
  m_jumpingTime = delay;
}

void
ErrorChannel::SetJumpingMode (bool mode)
{
  m_jumping = mode;
  m_jumpingState = 0;
}

void
ErrorChannel::SetDuplicateTime (Time delay)
{
  m_duplicateTime = delay;
}

void
ErrorChannel::SetDuplicateMode (bool mode)
{
  m_duplicate = mode;
  m_duplicateState = 0;
}

void
ErrorChannel::Send (Ptr<Packet> p, uint16_t protocol,
                          Mac48Address to, Mac48Address from,
                          Ptr<SimpleNetDevice> sender)
{
  NS_LOG_FUNCTION (p << protocol << to << from << sender);
  for (std::vector<Ptr<SimpleNetDevice> >::const_iterator i = m_devices.begin (); i != m_devices.end (); ++i)
    {
      Ptr<SimpleNetDevice> tmp = *i;
      if (tmp == sender)
        {
          continue;
        }
      if (m_jumping)
        {
          if (m_jumpingState % 2)
            {
              Simulator::ScheduleWithContext (tmp->GetNode ()->GetId (), Seconds (0),
                                              &SimpleNetDevice::Receive, tmp, p->Copy (), protocol, to, from);
            }
          else
            {
              Simulator::ScheduleWithContext (tmp->GetNode ()->GetId (), m_jumpingTime,
                                              &SimpleNetDevice::Receive, tmp, p->Copy (), protocol, to, from);
            }
          m_jumpingState++;
        }
      else if (m_duplicate)
        {
          if (m_duplicateState % 2)
            {
              Simulator::ScheduleWithContext (tmp->GetNode ()->GetId (), Seconds (0),
                                              &SimpleNetDevice::Receive, tmp, p->Copy (), protocol, to, from);
            }
          else
            {
              Simulator::ScheduleWithContext (tmp->GetNode ()->GetId (), Seconds (0),
                                              &SimpleNetDevice::Receive, tmp, p->Copy (), protocol, to, from);
              Simulator::ScheduleWithContext (tmp->GetNode ()->GetId (), m_duplicateTime,
                                              &SimpleNetDevice::Receive, tmp, p->Copy (), protocol, to, from);
            }
          m_duplicateState++;
        }
      else
        {
          Simulator::ScheduleWithContext (tmp->GetNode ()->GetId (), Seconds (0),
                                          &SimpleNetDevice::Receive, tmp, p->Copy (), protocol, to, from);
        }
    }
}

void
ErrorChannel::Add (Ptr<SimpleNetDevice> device)
{
  m_devices.push_back (device);
}

std::size_t
ErrorChannel::GetNDevices (void) const
{
  return m_devices.size ();
}

Ptr<NetDevice>
ErrorChannel::GetDevice (std::size_t i) const
{
  return m_devices[i];
}


} // namespace ns3
