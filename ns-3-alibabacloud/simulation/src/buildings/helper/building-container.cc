/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2008 INRIA
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr> (original node-container.cc)
 *         Nicola Baldo (wrote building-container.cc based on node-container.cc)
 */
#include "building-container.h"
#include "ns3/building-list.h"
#include "ns3/names.h"

namespace ns3 {

BuildingContainer::BuildingContainer ()
{
}

BuildingContainer::BuildingContainer (Ptr<Building> building)
{
  m_buildings.push_back (building);
}
BuildingContainer::BuildingContainer (std::string buildingName)
{
  Ptr<Building> building = Names::Find<Building> (buildingName);
  m_buildings.push_back (building);
}

BuildingContainer::Iterator 
BuildingContainer::Begin (void) const
{
  return m_buildings.begin ();
}
BuildingContainer::Iterator 
BuildingContainer::End (void) const
{
  return m_buildings.end ();
}

uint32_t 
BuildingContainer::GetN (void) const
{
  return m_buildings.size ();
}
Ptr<Building> 
BuildingContainer::Get (uint32_t i) const
{
  return m_buildings[i];
}
void 
BuildingContainer::Create (uint32_t n)
{
  for (uint32_t i = 0; i < n; i++)
    {
      m_buildings.push_back (CreateObject<Building> ());
    }
}
void 
BuildingContainer::Add (BuildingContainer other)
{
  for (Iterator i = other.Begin (); i != other.End (); i++)
    {
      m_buildings.push_back (*i);
    }
}
void 
BuildingContainer::Add (Ptr<Building> building)
{
  m_buildings.push_back (building);
}
void 
BuildingContainer::Add (std::string buildingName)
{
  Ptr<Building> building = Names::Find<Building> (buildingName);
  m_buildings.push_back (building);
}

BuildingContainer 
BuildingContainer::GetGlobal (void)
{
  BuildingContainer c;
  for (BuildingList::Iterator i = BuildingList::Begin (); i != BuildingList::End (); ++i)
    {
      c.Add (*i);
    }
  return c;
}

} // namespace ns3
