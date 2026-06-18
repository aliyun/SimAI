/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Wireless Communications and Networking Group (WCNG),
 * University of Rochester, Rochester, NY, USA.
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
 * Author: Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#include "energy-harvester-helper.h"
#include "ns3/config.h"
#include "ns3/names.h"

namespace ns3 {
  
/*
 * EnergyHarvesterHelper
 */
EnergyHarvesterHelper::~EnergyHarvesterHelper ()
{
}

EnergyHarvesterContainer
EnergyHarvesterHelper::Install (Ptr<EnergySource> source) const
{
  return Install (EnergySourceContainer (source));
}

EnergyHarvesterContainer
EnergyHarvesterHelper::Install (EnergySourceContainer sourceContainer) const
{
  EnergyHarvesterContainer container;
  for (EnergySourceContainer::Iterator i = sourceContainer.Begin (); i != sourceContainer.End (); ++i)
  {
    Ptr<EnergyHarvester> harvester = DoInstall (*i);
    container.Add (harvester);
    Ptr<Node> node = (*i)->GetNode ();
    /*
     * Check if EnergyHarvesterContainer is already aggregated to target node. If
     * not, create a new EnergyHarvesterContainer and aggregate it to the node.
     */
    Ptr<EnergyHarvesterContainer> EnergyHarvesterContainerOnNode =
    node->GetObject<EnergyHarvesterContainer> ();
    if (EnergyHarvesterContainerOnNode == 0)
    {
      ObjectFactory fac;
      fac.SetTypeId ("ns3::EnergyHarvesterContainer");
      EnergyHarvesterContainerOnNode = fac.Create<EnergyHarvesterContainer> ();
      EnergyHarvesterContainerOnNode->Add (harvester);
      node->AggregateObject (EnergyHarvesterContainerOnNode);
    }
    else
    {
      EnergyHarvesterContainerOnNode->Add (harvester);  // append new EnergyHarvester
    }
  }
  return container;
}

EnergyHarvesterContainer
EnergyHarvesterHelper::Install (std::string sourceName) const
{
  Ptr<EnergySource> source  = Names::Find<EnergySource> (sourceName);
  return Install (source);
}
    
} // namespace ns3
