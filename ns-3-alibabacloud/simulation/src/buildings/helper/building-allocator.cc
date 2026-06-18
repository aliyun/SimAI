/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
 * Copyright (C) 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Nicola Baldo <nbaldo@cttc.es> (took position-allocator and turned it into building-allocator)
 */
#include "building-allocator.h"
#include "ns3/building.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/enum.h"
#include "ns3/log.h"
#include <cmath>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BuildingAllocator");

NS_OBJECT_ENSURE_REGISTERED (GridBuildingAllocator);

GridBuildingAllocator::GridBuildingAllocator ()
  : m_current (0)
{
  m_buildingFactory.SetTypeId ("ns3::Building");
  m_lowerLeftPositionAllocator = CreateObject<GridPositionAllocator> ();
  m_upperRightPositionAllocator = CreateObject<GridPositionAllocator> ();
}

GridBuildingAllocator::~GridBuildingAllocator ()
{
}

TypeId 
GridBuildingAllocator::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::GridBuildingAllocator")
    .SetParent<Object> ()
    .AddConstructor<GridBuildingAllocator> ()
    .SetGroupName ("Buildings")
    .AddAttribute ("GridWidth", "The number of objects laid out on a line.",
                   UintegerValue (10),
                   MakeUintegerAccessor (&GridBuildingAllocator::m_n),
                   MakeUintegerChecker<uint32_t> ())
    .AddAttribute ("MinX", "The x coordinate where the grid starts.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_xMin),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("MinY", "The y coordinate where the grid starts.",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_yMin),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("LengthX", "The length of the wall of each building along the X axis.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_lengthX),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("LengthY", "The length of the wall of each building along the X axis.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_lengthY),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("DeltaX", "The x space between buildings.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_deltaX),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("DeltaY", "The y space between buildings.",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_deltaY),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Height", "The height of the building (roof level)",
                   DoubleValue (10),
                   MakeDoubleAccessor (&GridBuildingAllocator::m_height),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("LayoutType", "The type of layout.",
                   EnumValue (GridPositionAllocator::ROW_FIRST),
                   MakeEnumAccessor (&GridBuildingAllocator::m_layoutType),
                   MakeEnumChecker (GridPositionAllocator::ROW_FIRST, "RowFirst",
                                    GridPositionAllocator::COLUMN_FIRST, "ColumnFirst"))
  ;
  return tid;
}

void
GridBuildingAllocator::SetBuildingAttribute (std::string n, const AttributeValue &v)
{
  NS_LOG_FUNCTION (this);
  m_buildingFactory.Set (n, v);
}

BuildingContainer
GridBuildingAllocator::Create (uint32_t n) const
{
  NS_LOG_FUNCTION (this);
  PushAttributes ();
  BuildingContainer bc;
  uint32_t limit = n + m_current;
  for (; m_current < limit; ++m_current)
    {
      Vector lowerLeft = m_lowerLeftPositionAllocator->GetNext ();
      Vector upperRight = m_upperRightPositionAllocator->GetNext ();
      Box box (lowerLeft.x, upperRight.x, lowerLeft.y, upperRight.y, 0, m_height);
      NS_LOG_LOGIC ("new building : " <<  box);
      BoxValue boxValue (box);
      m_buildingFactory.Set ("Boundaries", boxValue);
      Ptr<Building> b  = m_buildingFactory.Create<Building> ();
      bc.Add (b);     
    }
  return bc;
}

void
GridBuildingAllocator::PushAttributes () const
{
  NS_LOG_FUNCTION (this);
  m_lowerLeftPositionAllocator->SetMinX (m_xMin);
  m_upperRightPositionAllocator->SetMinX (m_xMin + m_lengthX);
  m_lowerLeftPositionAllocator->SetDeltaX (m_lengthX + m_deltaX);
  m_upperRightPositionAllocator->SetDeltaX (m_lengthX + m_deltaX);
  
  m_lowerLeftPositionAllocator->SetMinY (m_yMin);
  m_upperRightPositionAllocator->SetMinY (m_yMin + m_lengthY);
  m_lowerLeftPositionAllocator->SetDeltaY (m_lengthY + m_deltaY);
  m_upperRightPositionAllocator->SetDeltaY (m_lengthY + m_deltaY); 

  m_lowerLeftPositionAllocator->SetLayoutType (m_layoutType);
  m_upperRightPositionAllocator->SetLayoutType (m_layoutType);

  m_lowerLeftPositionAllocator->SetN (m_n);
  m_upperRightPositionAllocator->SetN (m_n);
}


} // namespace ns3 
