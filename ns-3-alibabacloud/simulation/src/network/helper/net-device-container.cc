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

#include "net-device-container.h"
#include "ns3/names.h"

namespace ns3 {

NetDeviceContainer::NetDeviceContainer ()
{
}
NetDeviceContainer::NetDeviceContainer (Ptr<NetDevice> dev)
{
  m_devices.push_back (dev);
}
NetDeviceContainer::NetDeviceContainer (std::string devName)
{
  Ptr<NetDevice> dev = Names::Find<NetDevice> (devName);
  m_devices.push_back (dev);
}
NetDeviceContainer::NetDeviceContainer (const NetDeviceContainer &a, const NetDeviceContainer &b)
{
  *this = a;
  Add (b);
}


NetDeviceContainer::Iterator 
NetDeviceContainer::Begin (void) const
{
  return m_devices.begin ();
}
NetDeviceContainer::Iterator 
NetDeviceContainer::End (void) const
{
  return m_devices.end ();
}

uint32_t 
NetDeviceContainer::GetN (void) const
{
  return m_devices.size ();
}
Ptr<NetDevice> 
NetDeviceContainer::Get (uint32_t i) const
{
  return m_devices[i];
}
void 
NetDeviceContainer::Add (NetDeviceContainer other)
{
  for (Iterator i = other.Begin (); i != other.End (); i++)
    {
      m_devices.push_back (*i);
    }
}
void 
NetDeviceContainer::Add (Ptr<NetDevice> device)
{
  m_devices.push_back (device);
}
void 
NetDeviceContainer::Add (std::string deviceName)
{
  Ptr<NetDevice> device = Names::Find<NetDevice> (deviceName);
  m_devices.push_back (device);
}

} // namespace ns3
