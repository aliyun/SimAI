/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015, NYU WIRELESS, Tandon School of Engineering, New York
 * University
 * Copyright (c) 2019 SIGNET Lab, Department of Information Engineering,
 * University of Padova
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
 */

#include "ns3/buildings-channel-condition-model.h"
#include "ns3/mobility-model.h"
#include "ns3/mobility-building-info.h"
#include "ns3/building-list.h"
#include "ns3/log.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BuildingsChannelConditionModel");

NS_OBJECT_ENSURE_REGISTERED (BuildingsChannelConditionModel);

TypeId
BuildingsChannelConditionModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BuildingsChannelConditionModel")
    .SetParent<ChannelConditionModel> ()
    .SetGroupName ("Buildings")
    .AddConstructor<BuildingsChannelConditionModel> ()
  ;
  return tid;
}

BuildingsChannelConditionModel::BuildingsChannelConditionModel ()
  : ChannelConditionModel ()
{
}

BuildingsChannelConditionModel::~BuildingsChannelConditionModel ()
{
}

Ptr<ChannelCondition>
BuildingsChannelConditionModel::GetChannelCondition (Ptr<const MobilityModel> a,
                                                     Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);
  Ptr<MobilityBuildingInfo> a1 = a->GetObject<MobilityBuildingInfo> ();
  Ptr<MobilityBuildingInfo> b1 = b->GetObject<MobilityBuildingInfo> ();
  NS_ASSERT_MSG ((a1 != nullptr) && (b1 != nullptr),
                 "BuildingsChannelConditionModel only works with MobilityBuildingInfo");

  Ptr<ChannelCondition> cond = CreateObject<ChannelCondition> ();

  bool isAIndoor = a1->IsIndoor ();
  bool isBIndoor = b1->IsIndoor ();


  if (!isAIndoor && !isBIndoor) // a and b are outdoor
    {
      cond->SetO2iCondition (ChannelCondition::O2iConditionValue::O2O);
      
      // The outdoor case, determine LOS/NLOS
      // The channel condition should be LOS if the line of sight is not blocked,
      // otherwise NLOS
      bool blocked = IsLineOfSightBlocked (a->GetPosition (), b->GetPosition ());
      NS_LOG_DEBUG ("a and b are outdoor, blocked " << blocked);
      if (!blocked)
        {
          NS_LOG_DEBUG ("Set LOS");
          cond->SetLosCondition (ChannelCondition::LosConditionValue::LOS);
        }
      else
        {
          cond->SetLosCondition (ChannelCondition::LosConditionValue::NLOS);
        }
    }
  else if (isAIndoor && isBIndoor) // a and b are indoor
    {
      cond->SetO2iCondition (ChannelCondition::O2iConditionValue::I2I);
      
      // Indoor case, determine is the two nodes are inside the same building
      // or not
      if (a1->GetBuilding () == b1->GetBuilding ())
        {
          NS_LOG_DEBUG ("a and b are indoor in the same building");
          cond->SetLosCondition (ChannelCondition::LosConditionValue::LOS);
        }
      else
        {
          NS_LOG_DEBUG ("a and b are indoor in different buildings");
          cond->SetLosCondition (ChannelCondition::LosConditionValue::NLOS);
        }
    }
  else //outdoor to indoor case
    {
      cond->SetO2iCondition (ChannelCondition::O2iConditionValue::O2I);
      
      NS_LOG_DEBUG ("a is indoor and b outdoor or viceversa");
      cond->SetLosCondition (ChannelCondition::LosConditionValue::NLOS);
    }

  return cond;
}

bool
BuildingsChannelConditionModel::IsLineOfSightBlocked (const ns3::Vector &l1, const ns3::Vector &l2) const
{
  for (BuildingList::Iterator bit = BuildingList::Begin (); bit != BuildingList::End (); ++bit)
    {
      if ((*bit)->IsIntersect (l1, l2))
        {
          // The line of sight should be blocked if the line-segment between
          // l1 and l2 intersects one of the buildings.
          return true;
        }
    }

  // The line of sight should not be blocked if the line-segment between
  // l1 and l2 did not intersect any building.
  return false;
}

int64_t
BuildingsChannelConditionModel::AssignStreams ([[maybe_unused]] int64_t stream)
{
  return 0;
}

} // end namespace ns3
