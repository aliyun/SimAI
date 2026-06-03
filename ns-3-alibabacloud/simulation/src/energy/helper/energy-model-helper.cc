/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Authors: Sidharth Nabar <snabar@uw.edu>, He Wu <mdzz@u.washington.edu>
 */

#include "energy-model-helper.h"
#include "ns3/config.h"
#include "ns3/names.h"

namespace ns3 {

/*
 * EnergySourceHelper
 */
EnergySourceHelper::~EnergySourceHelper ()
{
}

EnergySourceContainer
EnergySourceHelper::Install (Ptr<Node> node) const
{
  return Install (NodeContainer (node));
}

EnergySourceContainer
EnergySourceHelper::Install (NodeContainer c) const
{
  EnergySourceContainer container;
  for (NodeContainer::Iterator i = c.Begin (); i != c.End (); ++i)
    {
      Ptr<EnergySource> src = DoInstall (*i);
      container.Add (src);
      /*
       * Check if EnergySourceContainer is already aggregated to target node. If
       * not, create a new EnergySourceContainer and aggregate it to node.
       */
      Ptr<EnergySourceContainer> EnergySourceContainerOnNode =
        (*i)->GetObject<EnergySourceContainer> ();
      if (EnergySourceContainerOnNode == NULL)
        {
          ObjectFactory fac;
          fac.SetTypeId ("ns3::EnergySourceContainer");
          EnergySourceContainerOnNode = fac.Create<EnergySourceContainer> ();
          EnergySourceContainerOnNode->Add (src);
          (*i)->AggregateObject (EnergySourceContainerOnNode);
        }
      else
        {
          EnergySourceContainerOnNode->Add (src);  // append new EnergySource
        }
    }
  return container;
}

EnergySourceContainer
EnergySourceHelper::Install (std::string nodeName) const
{
  Ptr<Node> node = Names::Find<Node> (nodeName);
  return Install (node);
}

EnergySourceContainer
EnergySourceHelper::InstallAll (void) const
{
  return Install (NodeContainer::GetGlobal ());
}

/*
 * DeviceEnergyModelHelper
 */
DeviceEnergyModelHelper::~DeviceEnergyModelHelper ()
{
}

DeviceEnergyModelContainer
DeviceEnergyModelHelper::Install (Ptr<NetDevice> device,
                                  Ptr<EnergySource> source) const
{
  NS_ASSERT (device != NULL);
  NS_ASSERT (source != NULL);
  // check to make sure source and net device are on the same node
  NS_ASSERT (device->GetNode () == source->GetNode ());
  DeviceEnergyModelContainer container (DoInstall (device, source));
  return container;
}

DeviceEnergyModelContainer
DeviceEnergyModelHelper::Install (NetDeviceContainer deviceContainer,
                                  EnergySourceContainer sourceContainer) const
{
  NS_ASSERT (deviceContainer.GetN () <= sourceContainer.GetN ());
  DeviceEnergyModelContainer container;
  NetDeviceContainer::Iterator dev = deviceContainer.Begin ();
  EnergySourceContainer::Iterator src = sourceContainer.Begin ();
  while (dev != deviceContainer.End ())
    {
      // check to make sure source and net device are on the same node
      NS_ASSERT ((*dev)->GetNode () == (*src)->GetNode ());
      Ptr<DeviceEnergyModel> model = DoInstall (*dev, *src);
      container.Add (model);
      dev++;
      src++;
    }
  return container;
}

} // namespace ns3
