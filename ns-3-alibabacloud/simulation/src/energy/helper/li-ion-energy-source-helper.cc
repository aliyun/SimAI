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
 * Authors: Hoda Ayatollahi <hayatoll@ur.rochester.edu>
 *          Cristiano Tapparello <cristiano.tapparello@rochester.edu>
 */

#include "li-ion-energy-source-helper.h"
#include "ns3/energy-source.h"

namespace ns3 {

LiIonEnergySourceHelper::LiIonEnergySourceHelper ()
{
  m_liIonEnergySource.SetTypeId ("ns3::LiIonEnergySource");
}

LiIonEnergySourceHelper::~LiIonEnergySourceHelper ()
{
}

void 
LiIonEnergySourceHelper::Set (std::string name, const AttributeValue &v)
{
  m_liIonEnergySource.Set (name, v);
}

Ptr<EnergySource> 
LiIonEnergySourceHelper::DoInstall (Ptr<Node> node) const
{
  NS_ASSERT (node != NULL);
  Ptr<EnergySource> source = m_liIonEnergySource.Create<EnergySource> ();
  NS_ASSERT (source != NULL);
  source->SetNode (node);
  return source;
}

} // namespace ns3
