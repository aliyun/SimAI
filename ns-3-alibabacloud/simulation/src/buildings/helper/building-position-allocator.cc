/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (C)  2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 *         Michele Polese <michele.polese@gmail.com> for the OutdoorPositionAllocator class
 */
#include "building-position-allocator.h"
#include <ns3/mobility-building-info.h>
#include "ns3/mobility-model.h"
#include "ns3/buildings-helper.h"
#include "ns3/random-variable-stream.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/enum.h"
#include "ns3/boolean.h"
#include "ns3/log.h"
#include "ns3/box.h"
#include "ns3/building.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include <cmath>

#include "ns3/building-list.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BuildingPositionAllocator");

NS_OBJECT_ENSURE_REGISTERED (RandomBuildingPositionAllocator);


RandomBuildingPositionAllocator::RandomBuildingPositionAllocator ()
{
  m_rand = CreateObject<UniformRandomVariable> ();
}

TypeId
RandomBuildingPositionAllocator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RandomBuildingPositionAllocator")
    .SetParent<PositionAllocator> ()
    .SetGroupName ("Buildings")
    .AddConstructor<RandomBuildingPositionAllocator> ()
    .AddAttribute ("WithReplacement",
                   "If true, the building will be randomly selected with replacement. "
                   "If false, no replacement will occur, until the list of buildings "
                   "to select becomes empty, at which point it will be filled again "
                   "with the list of all buildings.",
                   BooleanValue (false),
                   MakeBooleanAccessor (&RandomBuildingPositionAllocator::m_withReplacement),
                   MakeBooleanChecker ());
  return tid;
}

Vector 
RandomBuildingPositionAllocator::GetNext () const
{
  NS_ASSERT_MSG (BuildingList::GetNBuildings () > 0, "no building found");
  Ptr<Building> b;
  if (m_withReplacement)
    {
      uint32_t n = m_rand->GetInteger (0, BuildingList::GetNBuildings () - 1);
      b = BuildingList::GetBuilding (n);
    }
  else
    {
      if (m_buildingListWithoutReplacement.empty ())
        {
          for (BuildingList::Iterator bit = BuildingList::Begin (); bit != BuildingList::End (); ++bit)
            {
              m_buildingListWithoutReplacement.push_back (*bit);
            }
        }
      uint32_t n = m_rand->GetInteger (0, m_buildingListWithoutReplacement.size () - 1);
      b = m_buildingListWithoutReplacement.at (n);
      m_buildingListWithoutReplacement.erase (m_buildingListWithoutReplacement.begin () + n);
    }

  Ptr<RandomBoxPositionAllocator> pa = CreateObject<RandomBoxPositionAllocator> ();
  BoxValue bv;
  b->GetAttribute ("Boundaries", bv);
  double x = m_rand->GetValue (bv.Get ().xMin, bv.Get ().xMax);
  double y = m_rand->GetValue (bv.Get ().yMin, bv.Get ().yMax);
  double z = m_rand->GetValue (bv.Get ().zMin, bv.Get ().zMax);
  return Vector (x, y, z);
}

int64_t
RandomBuildingPositionAllocator::AssignStreams (int64_t stream)
{
  m_rand->SetStream (stream);
  return 1;
}


NS_OBJECT_ENSURE_REGISTERED (OutdoorPositionAllocator);

OutdoorPositionAllocator::OutdoorPositionAllocator ()
{
}

TypeId
OutdoorPositionAllocator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OutdoorPositionAllocator")
    .SetParent<PositionAllocator> ()
    .SetGroupName ("Buildings")
    .AddConstructor<OutdoorPositionAllocator> ()
    .AddAttribute ("X",
                   "A random variable which represents the x coordinate of a position in a random box.",
                   StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1.0]"),
                   MakePointerAccessor (&OutdoorPositionAllocator::m_x),
                   MakePointerChecker<RandomVariableStream> ())
    .AddAttribute ("Y",
                   "A random variable which represents the y coordinate of a position in a random box.",
                   StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1.0]"),
                   MakePointerAccessor (&OutdoorPositionAllocator::m_y),
                   MakePointerChecker<RandomVariableStream> ())
    .AddAttribute ("Z",
                   "A random variable which represents the z coordinate of a position in a random box.",
                   StringValue ("ns3::UniformRandomVariable[Min=0.0|Max=1.0]"),
                   MakePointerAccessor (&OutdoorPositionAllocator::m_z),
                   MakePointerChecker<RandomVariableStream> ())
    .AddAttribute ("MaxAttempts",
                   "Maximum number of attempts for the rejection sampling before giving up.",
                   UintegerValue (1000),
                   MakeUintegerAccessor (&OutdoorPositionAllocator::m_maxAttempts),
                   MakeUintegerChecker<uint32_t> ())
  ;

  return tid;
}

void
OutdoorPositionAllocator::SetX (Ptr<RandomVariableStream> x)
{
  m_x = x;
}
void
OutdoorPositionAllocator::SetY (Ptr<RandomVariableStream> y)
{
  m_y = y;
}
void
OutdoorPositionAllocator::SetZ (Ptr<RandomVariableStream> z)
{
  m_z = z;
}

Vector
OutdoorPositionAllocator::GetNext () const
{
  NS_ABORT_MSG_IF (BuildingList::GetNBuildings () == 0, "no building found");

  bool outdoor = false;
  uint32_t attempts = 0;
  Vector position = Vector (0,0,0);

  while (!outdoor && attempts < m_maxAttempts)
    {
      // get a random position
      double x = m_x->GetValue ();
      double y = m_y->GetValue ();
      double z = m_z->GetValue ();

      position = Vector (x, y, z);

      NS_LOG_INFO ("Position " << position);

      bool inside = false;
      for (BuildingList::Iterator bit = BuildingList::Begin (); bit != BuildingList::End (); ++bit)
        {
          if ((*bit)->IsInside (position))
            {
              NS_LOG_INFO ("Position " << position << " is inside the building with boundaries "
                                       << (*bit)->GetBoundaries ().xMin << " " << (*bit)->GetBoundaries ().xMax << " "
                                       << (*bit)->GetBoundaries ().yMin << " " << (*bit)->GetBoundaries ().yMax << " "
                                       << (*bit)->GetBoundaries ().zMin << " " << (*bit)->GetBoundaries ().zMax);
              inside = true;
              break;
            }
        }

      if (inside)
        {
          NS_LOG_INFO ("Inside a building, attempt " << attempts << " out of " << m_maxAttempts);
          attempts++;
        }
      else
        {
          NS_LOG_INFO ("Outdoor position found " << position);
          outdoor = true;
        }
    }

  NS_ABORT_MSG_IF (attempts >= m_maxAttempts, "Too many attempts, give up");
  NS_ABORT_MSG_IF (!outdoor, "Still indoor, give up");
  return position;
}

int64_t
OutdoorPositionAllocator::AssignStreams (int64_t stream)
{
  m_x->SetStream (stream);
  m_y->SetStream (stream + 1);
  m_z->SetStream (stream + 2);
  return 3;
}


NS_OBJECT_ENSURE_REGISTERED (RandomRoomPositionAllocator);


RandomRoomPositionAllocator::RandomRoomPositionAllocator ()
{
  m_rand = CreateObject<UniformRandomVariable> ();
}

TypeId
RandomRoomPositionAllocator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::RandomRoomPositionAllocator")
    .SetParent<PositionAllocator> ()
    .SetGroupName ("Buildings")
    .AddConstructor<RandomRoomPositionAllocator> ();
  return tid;
}

Vector 
RandomRoomPositionAllocator::GetNext () const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (BuildingList::GetNBuildings () > 0, "no building found");
 
  if (m_roomListWithoutReplacement.empty ())
    {
      for (BuildingList::Iterator bit = BuildingList::Begin (); bit != BuildingList::End (); ++bit)
        {
          NS_LOG_LOGIC ("building " << (*bit)->GetId ());
          for (uint32_t rx = 1; rx <= (*bit)->GetNRoomsX (); ++rx)
            {
              for (uint32_t ry = 1; ry <= (*bit)->GetNRoomsY (); ++ry)
                {
                  for (uint32_t f = 1; f <= (*bit)->GetNFloors (); ++f)
                    {
                      RoomInfo i;
                      i.roomx = rx;
                      i.roomy = ry;
                      i.floor = f; 
                      i.b = *bit;
                      NS_LOG_LOGIC ("adding room (" << rx << ", " << ry << ", " << f << ")");
                      m_roomListWithoutReplacement.push_back (i);
                    }
                }
            }
        }
    }
  uint32_t n = m_rand->GetInteger (0,m_roomListWithoutReplacement.size () - 1);
  RoomInfo r = m_roomListWithoutReplacement.at (n);
  m_roomListWithoutReplacement.erase (m_roomListWithoutReplacement.begin () + n);
  NS_LOG_LOGIC ("considering building " << r.b->GetId () << " room (" << r.roomx << ", " << r.roomy << ", " << r.floor << ")");

  Ptr<RandomBoxPositionAllocator> pa = CreateObject<RandomBoxPositionAllocator> ();
  BoxValue bv;
  r.b->GetAttribute ("Boundaries", bv);
  Box box = bv.Get ();
  double rdx =  (box.xMax - box.xMin) / r.b->GetNRoomsX ();
  double rdy =  (box.yMax - box.yMin) / r.b->GetNRoomsY ();
  double rdz =  (box.zMax - box.zMin) / r.b->GetNFloors ();
  double x1 = box.xMin + rdx * (r.roomx - 1);
  double x2 = box.xMin + rdx * r.roomx;
  double y1 = box.yMin + rdy * (r.roomy -1);
  double y2 = box.yMin + rdy * r.roomy;
  double z1 = box.zMin + rdz * (r.floor - 1);
  double z2 = box.zMin + rdz * r.floor;
  NS_LOG_LOGIC ("randomly allocating position in "
                << " (" << x1 << "," << x2 << ") "
                << "x (" << y1 << "," << y2 << ") "
                << "x (" << z1 << "," << z2 << ") ");

  double x = m_rand->GetValue (x1, x2);
  double y = m_rand->GetValue (y1, y2);
  double z = m_rand->GetValue (z1, z2);

  return Vector (x, y, z);
}

int64_t
RandomRoomPositionAllocator::AssignStreams (int64_t stream)
{
  m_rand->SetStream (stream);
  return 1;
}





NS_OBJECT_ENSURE_REGISTERED (SameRoomPositionAllocator);

SameRoomPositionAllocator::SameRoomPositionAllocator ()
{
  NS_FATAL_ERROR (" Constructor \"SameRoomPositionAllocator ()\" should not be used");
}


SameRoomPositionAllocator::SameRoomPositionAllocator (NodeContainer c)
  : m_nodes (c)
{
  m_rand = CreateObject<UniformRandomVariable> ();
  m_nodeIt = m_nodes.Begin ();
  // this is needed to make sure the building models associated with c have been initialized
  for (NodeContainer::Iterator it = m_nodes.Begin (); it != m_nodes.End (); ++it)
    {
      Ptr<MobilityModel> mm = (*it)->GetObject<MobilityModel> ();
      NS_ASSERT_MSG (mm, "no mobility model aggregated to this node");
      Ptr<MobilityBuildingInfo> bmm = mm->GetObject <MobilityBuildingInfo> ();
      NS_ASSERT_MSG (bmm, "MobilityBuildingInfo has not been aggregated to this node mobility model");
      bmm->MakeConsistent (mm);
    }
}

TypeId
SameRoomPositionAllocator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SameRoomPositionAllocator")
    .SetParent<PositionAllocator> ()
    .SetGroupName ("Buildings")
    .AddConstructor<SameRoomPositionAllocator> ();
  return tid;
}

Vector 
SameRoomPositionAllocator::GetNext () const
{
  NS_LOG_FUNCTION (this);
  if (m_nodeIt == m_nodes.End ())
    {
      m_nodeIt  = m_nodes.Begin ();
    }

  NS_ASSERT_MSG (m_nodeIt != m_nodes.End (), "no node in container");

  NS_LOG_LOGIC ("considering node " << (*m_nodeIt)->GetId ());
  Ptr<MobilityModel> mm = (*m_nodeIt)->GetObject<MobilityModel> ();
  NS_ASSERT_MSG (mm, "no mobility model aggregated to this node");
  Ptr<MobilityBuildingInfo> bmm = mm->GetObject<MobilityBuildingInfo> ();
  NS_ASSERT_MSG (bmm, "MobilityBuildingInfo has not been aggregated to this node mobility model");

  ++m_nodeIt;
  uint32_t roomx = bmm->GetRoomNumberX ();
  uint32_t roomy = bmm->GetRoomNumberY ();
  uint32_t floor = bmm->GetFloorNumber ();
  NS_LOG_LOGIC ("considering building " << bmm->GetBuilding ()->GetId () << " room (" << roomx << ", " << roomy << ", " << floor << ")");

  Ptr<Building> b = bmm->GetBuilding ();
  Ptr<RandomBoxPositionAllocator> pa = CreateObject<RandomBoxPositionAllocator> ();
  BoxValue bv;
  b->GetAttribute ("Boundaries", bv);
  Box box = bv.Get ();
  double rdx =  (box.xMax - box.xMin) / b->GetNRoomsX ();
  double rdy =  (box.yMax - box.yMin) / b->GetNRoomsY ();
  double rdz =  (box.zMax - box.zMin) / b->GetNFloors ();
  double x1 = box.xMin + rdx * (roomx - 1);
  double x2 = box.xMin + rdx * roomx;
  double y1 = box.yMin + rdy * (roomy -1);
  double y2 = box.yMin + rdy * roomy;
  double z1 = box.zMin + rdz * (floor - 1);
  double z2 = box.zMin + rdz * floor;
  NS_LOG_LOGIC ("randomly allocating position in "
                << " (" << x1 << "," << x2 << ") "
                << "x (" << y1 << "," << y2 << ") "
                << "x (" << z1 << "," << z2 << ") ");

  double x = m_rand->GetValue (x1, x2);
  double y = m_rand->GetValue (y1, y2);
  double z = m_rand->GetValue (z1, z2);

  return Vector (x, y, z);
}

int64_t
SameRoomPositionAllocator::AssignStreams (int64_t stream)
{
  m_rand->SetStream (stream);
  return 1;
}

NS_OBJECT_ENSURE_REGISTERED (FixedRoomPositionAllocator);


FixedRoomPositionAllocator::FixedRoomPositionAllocator (
  uint32_t x,
  uint32_t y,
  uint32_t z,
  Ptr<Building> pbtr)
{
  m_rand = CreateObject<UniformRandomVariable> ();
  roomx = x;
  roomy = y;
  floor = z;
  bptr = pbtr;
}

TypeId
FixedRoomPositionAllocator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::FixedRoomPositionAllocator")
    .SetParent<PositionAllocator> ()
    .SetGroupName ("Buildings")
    .AddConstructor<SameRoomPositionAllocator> ();
  return tid;
}

Vector
FixedRoomPositionAllocator::GetNext () const
{

  NS_LOG_LOGIC ("considering building " << bptr->GetId () << " room (" << roomx << ", " << roomy << ", " << floor << ")");

  Ptr<RandomBoxPositionAllocator> pa = CreateObject<RandomBoxPositionAllocator> ();

  Box box = bptr->GetBoundaries ();
  double rdx =  (box.xMax - box.xMin) / bptr->GetNRoomsX ();
  double rdy =  (box.yMax - box.yMin) / bptr->GetNRoomsY ();
  double rdz =  (box.zMax - box.zMin) / bptr->GetNFloors ();
  double x1 = box.xMin + rdx * (roomx - 1);
  double x2 = box.xMin + rdx * roomx;
  double y1 = box.yMin + rdy * (roomy -1);
  double y2 = box.yMin + rdy * roomy;
  double z1 = box.zMin + rdz * (floor - 1);
  double z2 = box.zMin + rdz * floor;
  NS_LOG_LOGIC ("randomly allocating position in "
                << " (" << x1 << "," << x2 << ") "
                << "x (" << y1 << "," << y2 << ") "
                << "x (" << z1 << "," << z2 << ") ");

  double x = m_rand->GetValue (x1, x2);
  double y = m_rand->GetValue (y1, y2);
  double z = m_rand->GetValue (z1, z2);
  return Vector (x, y, z);
}


int64_t
FixedRoomPositionAllocator::AssignStreams (int64_t stream)
{
  m_rand->SetStream (stream);
  return 1;
}

}  // namespace ns3
