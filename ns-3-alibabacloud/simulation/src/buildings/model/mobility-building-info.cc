/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 * 
 */

#include <ns3/simulator.h>
#include <ns3/position-allocator.h>
#include <ns3/building-list.h>
#include <ns3/mobility-building-info.h>
#include <ns3/pointer.h>
#include <ns3/log.h>
#include <ns3/assert.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("MobilityBuildingInfo");

NS_OBJECT_ENSURE_REGISTERED (MobilityBuildingInfo);

TypeId
MobilityBuildingInfo::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::MobilityBuildingInfo")
    .SetParent<Object> ()
    .SetGroupName ("Buildings")
    .AddConstructor<MobilityBuildingInfo> ();

  return tid;
}

void
MobilityBuildingInfo::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
  Ptr<MobilityModel> mm = this->GetObject<MobilityModel> ();
  MakeConsistent (mm);
}


MobilityBuildingInfo::MobilityBuildingInfo ()
{
  NS_LOG_FUNCTION (this);
  m_indoor = false;
  m_nFloor = 1;
  m_roomX = 1;
  m_roomY = 1;
  m_cachedPosition = Vector (0, 0, 0);
}


MobilityBuildingInfo::MobilityBuildingInfo (Ptr<Building> building)
  : m_myBuilding (building)
{
  NS_LOG_FUNCTION (this);
  m_indoor = false;
  m_nFloor = 1;
  m_roomX = 1;
  m_roomY = 1;
}

bool
MobilityBuildingInfo::IsIndoor (void)
{
  NS_LOG_FUNCTION (this);
  Ptr<MobilityModel> mm = this->GetObject<MobilityModel> ();
  Vector currentPosition = mm->GetPosition ();
  bool posNotEqual = (currentPosition < m_cachedPosition) || (m_cachedPosition < currentPosition);
  if (posNotEqual)
    {
      MakeConsistent (mm);
    }

  return m_indoor;
}

void
MobilityBuildingInfo::SetIndoor (Ptr<Building> building, uint8_t nfloor, uint8_t nroomx, uint8_t nroomy)
{
  NS_LOG_FUNCTION (this);
  m_indoor = true;
  m_myBuilding = building;
  m_nFloor = nfloor;
  m_roomX = nroomx;
  m_roomY = nroomy;
  

  NS_ASSERT (m_roomX > 0);
  NS_ASSERT (m_roomX <= building->GetNRoomsX ());
  NS_ASSERT (m_roomY > 0);
  NS_ASSERT (m_roomY <= building->GetNRoomsY ());
  NS_ASSERT (m_nFloor > 0);
  NS_ASSERT (m_nFloor <= building->GetNFloors ());

}


void
MobilityBuildingInfo::SetIndoor (uint8_t nfloor, uint8_t nroomx, uint8_t nroomy)
{
  NS_LOG_FUNCTION (this);
  m_indoor = true;
  m_nFloor = nfloor;
  m_roomX = nroomx;
  m_roomY = nroomy;

  NS_ASSERT_MSG (m_myBuilding, "Node does not have any building defined");
  NS_ASSERT (m_roomX > 0);
  NS_ASSERT (m_roomX <= m_myBuilding->GetNRoomsX ());
  NS_ASSERT (m_roomY > 0);
  NS_ASSERT (m_roomY <= m_myBuilding->GetNRoomsY ());
  NS_ASSERT (m_nFloor > 0);
  NS_ASSERT (m_nFloor <= m_myBuilding->GetNFloors ());

}


void
MobilityBuildingInfo::SetOutdoor (void)
{
  NS_LOG_FUNCTION (this);
  m_indoor = false;
}

uint8_t
MobilityBuildingInfo::GetFloorNumber (void)
{
  NS_LOG_FUNCTION (this);
  return (m_nFloor);
}

uint8_t
MobilityBuildingInfo::GetRoomNumberX (void)
{
  NS_LOG_FUNCTION (this);
  return (m_roomX);
}

uint8_t
MobilityBuildingInfo::GetRoomNumberY (void)
{
  NS_LOG_FUNCTION (this);
  return (m_roomY);
}


Ptr<Building>
MobilityBuildingInfo::GetBuilding ()
{
  NS_LOG_FUNCTION (this);
  return (m_myBuilding);
}

void
MobilityBuildingInfo::MakeConsistent (Ptr<MobilityModel> mm)
{
  bool found = false;
  Vector pos = mm->GetPosition ();
  for (BuildingList::Iterator bit = BuildingList::Begin (); bit != BuildingList::End (); ++bit)
    {
      NS_LOG_LOGIC ("checking building " << (*bit)->GetId () << " with boundaries " << (*bit)->GetBoundaries ());
      if ((*bit)->IsInside (pos))
        {
          NS_LOG_LOGIC ("MobilityBuildingInfo " << this << " pos " << pos << " falls inside building " << (*bit)->GetId ());
          NS_ABORT_MSG_UNLESS (found == false, " MobilityBuildingInfo already inside another building!");
          found = true;
          uint16_t floor = (*bit)->GetFloor (pos);
          uint16_t roomX = (*bit)->GetRoomX (pos);
          uint16_t roomY = (*bit)->GetRoomY (pos);
          SetIndoor (*bit, floor, roomX, roomY);
        }
    }
  if (!found)
    {
      NS_LOG_LOGIC ("MobilityBuildingInfo " << this << " pos " << pos  << " is outdoor");
      SetOutdoor ();
    }

  m_cachedPosition = pos;

}

  
} // namespace
