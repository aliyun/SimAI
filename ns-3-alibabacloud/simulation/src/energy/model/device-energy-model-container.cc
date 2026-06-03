/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2010 Network Security Lab, University of Washington, Seattle.
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
 * Authors: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 *          Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#include "device-energy-model-container.h"
#include "ns3/names.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DeviceEnergyModelContainer");

DeviceEnergyModelContainer::DeviceEnergyModelContainer ()
{
  NS_LOG_FUNCTION (this);
}

DeviceEnergyModelContainer::DeviceEnergyModelContainer (Ptr<DeviceEnergyModel> model)
{
  NS_LOG_FUNCTION (this << model);
  NS_ASSERT (model != NULL);
  m_models.push_back (model);
}

DeviceEnergyModelContainer::DeviceEnergyModelContainer (std::string modelName)
{
  NS_LOG_FUNCTION (this << modelName);
  Ptr<DeviceEnergyModel> model = Names::Find<DeviceEnergyModel> (modelName);
  NS_ASSERT (model != NULL);
  m_models.push_back (model);
}

DeviceEnergyModelContainer::DeviceEnergyModelContainer (const DeviceEnergyModelContainer &a,
                                                        const DeviceEnergyModelContainer &b)
{
  NS_LOG_FUNCTION (this << &a << &b);
  *this = a;
  Add (b);
}

DeviceEnergyModelContainer::Iterator
DeviceEnergyModelContainer::Begin (void) const
{
  NS_LOG_FUNCTION (this);
  return m_models.begin ();
}

DeviceEnergyModelContainer::Iterator
DeviceEnergyModelContainer::End (void) const
{
  NS_LOG_FUNCTION (this);
  return m_models.end ();
}

uint32_t
DeviceEnergyModelContainer::GetN (void) const
{
  NS_LOG_FUNCTION (this);
  return m_models.size ();
}

Ptr<DeviceEnergyModel>
DeviceEnergyModelContainer::Get (uint32_t i) const
{
  NS_LOG_FUNCTION (this << i);
  return m_models[i];
}

void
DeviceEnergyModelContainer::Add (DeviceEnergyModelContainer container)
{
  NS_LOG_FUNCTION (this << &container);
  for (Iterator i = container.Begin (); i != container.End (); i++)
    {
      m_models.push_back (*i);
    }
}

void
DeviceEnergyModelContainer::Add (Ptr<DeviceEnergyModel> model)
{
  NS_LOG_FUNCTION (this << model);
  NS_ASSERT (model != NULL);
  m_models.push_back (model);
}

void
DeviceEnergyModelContainer::Add (std::string modelName)
{
  NS_LOG_FUNCTION (this << modelName);
  Ptr<DeviceEnergyModel> model = Names::Find<DeviceEnergyModel> (modelName);
  NS_ASSERT (model != NULL);
  m_models.push_back (model);
}

void
DeviceEnergyModelContainer::Clear (void)
{
  NS_LOG_FUNCTION (this);
  m_models.clear ();
}

} // namespace ns3
