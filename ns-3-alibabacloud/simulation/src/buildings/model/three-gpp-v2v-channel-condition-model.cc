/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 SIGNET Lab, Department of Information Engineering,
 * University of Padova
 * Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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

#include "three-gpp-v2v-channel-condition-model.h"
#include "ns3/mobility-model.h"
#include "ns3/log.h"
#include <ns3/building-list.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ThreeGppV2vChannelConditionModel");

NS_OBJECT_ENSURE_REGISTERED (ThreeGppV2vUrbanChannelConditionModel);

TypeId
ThreeGppV2vUrbanChannelConditionModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppV2vUrbanChannelConditionModel")
    .SetParent<ThreeGppChannelConditionModel> ()
    .SetGroupName ("Buildings")
    .AddConstructor<ThreeGppV2vUrbanChannelConditionModel> ()
  ;
  return tid;
}

ThreeGppV2vUrbanChannelConditionModel::ThreeGppV2vUrbanChannelConditionModel ()
  : ThreeGppChannelConditionModel ()
{
  m_buildingsCcm = CreateObject<BuildingsChannelConditionModel> ();
}

ThreeGppV2vUrbanChannelConditionModel::~ThreeGppV2vUrbanChannelConditionModel ()
{}

double
ThreeGppV2vUrbanChannelConditionModel::ComputePlos (Ptr<const MobilityModel> a,
                                                    Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);

  // determine if there is a building in between the tx and rx
  Ptr<ChannelCondition> cond = m_buildingsCcm->GetChannelCondition (a, b);
  NS_ASSERT_MSG (cond->IsO2o (), "The nodes should be outdoor");

  double pLos = 0.0;
  if (cond->IsLos ())
    {
      // compute the 2D distance between a and b
      double distance2D = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

      // compute the LOS probability (see 3GPP TR 37.885, Table 6.2-1)
      pLos = std::min (1.0, 1.05 * exp (-0.0114 * distance2D));
    }

  return pLos;
}

double
ThreeGppV2vUrbanChannelConditionModel::ComputePnlos (Ptr<const MobilityModel> a,
                                                     Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);

  // determine the NLOS due to buildings
  Ptr<ChannelCondition> cond = m_buildingsCcm->GetChannelCondition (a, b);
  NS_ASSERT_MSG (cond->IsO2o (), "The nodes should be outdoor");

  double pNlos = 0.0;
  if (cond->IsNlos ())
    {
      pNlos = 1.0;
    }

  return pNlos;
}

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppV2vHighwayChannelConditionModel);

TypeId
ThreeGppV2vHighwayChannelConditionModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppV2vHighwayChannelConditionModel")
    .SetParent<ThreeGppChannelConditionModel> ()
    .SetGroupName ("Buildings")
    .AddConstructor<ThreeGppV2vHighwayChannelConditionModel> ()
  ;
  return tid;
}

ThreeGppV2vHighwayChannelConditionModel::ThreeGppV2vHighwayChannelConditionModel ()
  : ThreeGppChannelConditionModel ()
{
  m_buildingsCcm = CreateObject<BuildingsChannelConditionModel> ();
  ComputeChCond = std::bind (&ThreeGppV2vHighwayChannelConditionModel::GetChCondAndFixCallback, this,
                             std::placeholders::_1, std::placeholders::_2);
}

ThreeGppV2vHighwayChannelConditionModel::~ThreeGppV2vHighwayChannelConditionModel ()
{}

double
ThreeGppV2vHighwayChannelConditionModel::ComputePlos (Ptr<const MobilityModel> a,
                                                      Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);

  // determine if there is a building in between the tx and rx
  Ptr<ChannelCondition> cond = ComputeChCond (a, b);
  NS_ASSERT_MSG (cond->IsO2o (), "The nodes should be outdoor");

  double pLos = 0.0;
  if (cond->IsLos ())
    {
      // compute the 2D distance between a and b
      double distance2D = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

      // compute the LOS probability (see 3GPP TR 37.885, Table 6.2-1)
      if (distance2D <= 475.0)
        {
          pLos = std::min (1.0, 2.1013e-6 * distance2D * distance2D - 0.002 * distance2D + 1.0193);
        }
      else
        {
          pLos = std::max (0.0, 0.54 - 0.001 * (distance2D - 475.0));
        }
    }

  return pLos;
}

double
ThreeGppV2vHighwayChannelConditionModel::ComputePnlos (Ptr<const MobilityModel> a,
                                                       Ptr<const MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);

  // determine the NLOS due to buildings
  Ptr<ChannelCondition> cond = ComputeChCond (a, b);
  NS_ASSERT_MSG (cond->IsO2o (), "The nodes should be outdoor");

  double pNlos = 0;
  if (cond->IsNlos ())
    {
      pNlos = 1.0;
    }

  return pNlos;
}

Ptr<ChannelCondition>
ThreeGppV2vHighwayChannelConditionModel::GetChCondAndFixCallback (Ptr<const MobilityModel> a,
                                                                   Ptr<const MobilityModel> b)
{
  Ptr<ChannelCondition> cond;
  if (BuildingList::Begin () != BuildingList::End ())
    {
      ComputeChCond = std::bind (&ThreeGppV2vHighwayChannelConditionModel::GetChCondWithBuildings, this,
                                 std::placeholders::_1, std::placeholders::_2);
      cond = GetChCondWithBuildings (a, b);
    }
  else
    {
      ComputeChCond = std::bind (&ThreeGppV2vHighwayChannelConditionModel::GetChCondWithNoBuildings, this,
                                 std::placeholders::_1, std::placeholders::_2);
      cond = GetChCondWithNoBuildings (a, b);
    }
  return cond;
}

Ptr<ChannelCondition>
ThreeGppV2vHighwayChannelConditionModel::GetChCondWithBuildings (Ptr<const MobilityModel> a,
                                                                  Ptr<const MobilityModel> b) const
{
  Ptr<ChannelCondition> cond = m_buildingsCcm->GetChannelCondition (a, b);
  return cond;
}

Ptr<ChannelCondition>
ThreeGppV2vHighwayChannelConditionModel::GetChCondWithNoBuildings (Ptr<const MobilityModel> a,
                                                                    Ptr<const MobilityModel> b) const
{
  Ptr<ChannelCondition> cond = CreateObject<ChannelCondition> ();
  cond->SetO2iCondition (ChannelCondition::O2iConditionValue::O2O);
  cond->SetLosCondition (ChannelCondition::LosConditionValue::LOS);
  return cond;
}

} // end namespace ns3
