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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>,
 *         Nicola Baldo <nbaldo@cttc.es>
 * 
 */

#include "ns3/propagation-loss-model.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/double.h"
#include "ns3/pointer.h"
#include <cmath>
#include "buildings-propagation-loss-model.h"
#include <ns3/mobility-building-info.h>
#include "ns3/enum.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BuildingsPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (BuildingsPropagationLossModel);

BuildingsPropagationLossModel::ShadowingLoss::ShadowingLoss ()
{
}

BuildingsPropagationLossModel::ShadowingLoss::ShadowingLoss (double shadowingValue, Ptr<MobilityModel> receiver)
  : m_shadowingValue (shadowingValue), m_receiver (receiver)
{
  NS_LOG_INFO (this << " New Shadowing value " << m_shadowingValue);
}

double
BuildingsPropagationLossModel::ShadowingLoss::GetLoss () const
{
  return (m_shadowingValue);
}

Ptr<MobilityModel>
BuildingsPropagationLossModel::ShadowingLoss::GetReceiver () const
{
  return m_receiver;
}

TypeId
BuildingsPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::BuildingsPropagationLossModel")

    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Buildings")


    .AddAttribute ("ShadowSigmaOutdoor",
                   "Standard deviation of the normal distribution used for calculate the shadowing for outdoor nodes",
                   DoubleValue (7.0),
                   MakeDoubleAccessor (&BuildingsPropagationLossModel::m_shadowingSigmaOutdoor),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("ShadowSigmaIndoor",
                   "Standard deviation of the normal distribution used for calculate the shadowing for indoor nodes ",
                   DoubleValue (8.0),
                   MakeDoubleAccessor (&BuildingsPropagationLossModel::m_shadowingSigmaIndoor),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ShadowSigmaExtWalls",
                   "Standard deviation of the normal distribution used for calculate the shadowing due to ext walls ",
                   DoubleValue (5.0),
                   MakeDoubleAccessor (&BuildingsPropagationLossModel::m_shadowingSigmaExtWalls),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("InternalWallLoss",
                   "Additional loss for each internal wall [dB]",
                   DoubleValue (5.0),
                   MakeDoubleAccessor (&BuildingsPropagationLossModel::m_lossInternalWall),
                   MakeDoubleChecker<double> ());


  return tid;
}

BuildingsPropagationLossModel::BuildingsPropagationLossModel ()
{
  m_randVariable = CreateObject<NormalRandomVariable> ();
}

double
BuildingsPropagationLossModel::ExternalWallLoss (Ptr<MobilityBuildingInfo> a) const
{
  double loss = 0.0;
  Ptr<Building> aBuilding = a->GetBuilding ();
  if (aBuilding->GetExtWallsType () == Building::Wood)
    {
      loss = 4;
    }
  else if (aBuilding->GetExtWallsType () == Building::ConcreteWithWindows)
    {
      loss = 7;
    }
  else if (aBuilding->GetExtWallsType () == Building::ConcreteWithoutWindows)
    {
      loss = 15; // 10 ~ 20 dB
    }
  else if (aBuilding->GetExtWallsType () == Building::StoneBlocks)
    {
      loss = 12;
    }
  return (loss);
}

double
BuildingsPropagationLossModel::HeightLoss (Ptr<MobilityBuildingInfo> node) const
{
  double loss = 0.0;

  int nfloors  = node->GetFloorNumber () - 1;
  loss = -2 * (nfloors);
  return (loss);
}

double
BuildingsPropagationLossModel::InternalWallsLoss (Ptr<MobilityBuildingInfo> a, Ptr<MobilityBuildingInfo>b) const
{
  // approximate the number of internal walls with the Manhattan distance in "rooms" units
  double dx = std::abs (a->GetRoomNumberX () - b->GetRoomNumberX ());
  double dy = std::abs (a->GetRoomNumberY () - b->GetRoomNumberY ());    
  return m_lossInternalWall * (dx+dy);
}



double
BuildingsPropagationLossModel::GetShadowing (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
const
{
    Ptr<MobilityBuildingInfo> a1 = a->GetObject <MobilityBuildingInfo> ();
    Ptr<MobilityBuildingInfo> b1 = b->GetObject <MobilityBuildingInfo> ();
    NS_ASSERT_MSG ((a1 != 0) && (b1 != 0), "BuildingsPropagationLossModel only works with MobilityBuildingInfo");
  
  std::map<Ptr<MobilityModel>,  std::map<Ptr<MobilityModel>, ShadowingLoss> >::iterator ait = m_shadowingLossMap.find (a);
  if (ait != m_shadowingLossMap.end ())
    {
      std::map<Ptr<MobilityModel>, ShadowingLoss>::iterator bit = ait->second.find (b);
      if (bit != ait->second.end ())
        {
          return (bit->second.GetLoss ());
        }
      else
        {
          double sigma = EvaluateSigma (a1, b1);
          // side effect: will create new entry          
          // sigma is standard deviation, not variance
          double shadowingValue = m_randVariable->GetValue (0.0, (sigma*sigma));
          ait->second[b] = ShadowingLoss (shadowingValue, b);          
          return (ait->second[b].GetLoss ());
        }
    }
  else
    {
      double sigma = EvaluateSigma (a1, b1);
      // side effect: will create new entries in both maps
      // sigma is standard deviation, not variance
      double shadowingValue = m_randVariable->GetValue (0.0, (sigma*sigma));
      m_shadowingLossMap[a][b] = ShadowingLoss (shadowingValue, b);  
      return (m_shadowingLossMap[a][b].GetLoss ());       
    }
}



double
BuildingsPropagationLossModel::EvaluateSigma (Ptr<MobilityBuildingInfo> a, Ptr<MobilityBuildingInfo> b)
const
{
  bool isAIndoor = a->IsIndoor ();
  bool isBIndoor = b->IsIndoor ();

  if (!isAIndoor) // a is outdoor
    {
      if (!isBIndoor) // b is outdoor
        {
          return (m_shadowingSigmaOutdoor);
        }
      else
        {
          double sigma = std::sqrt ((m_shadowingSigmaOutdoor * m_shadowingSigmaOutdoor) + (m_shadowingSigmaExtWalls * m_shadowingSigmaExtWalls));
          return (sigma);
        }
    }
  else
  if (isBIndoor) // b is indoor
    {
      return (m_shadowingSigmaIndoor);
    }
  else
    {
      double sigma = std::sqrt ((m_shadowingSigmaOutdoor * m_shadowingSigmaOutdoor) + (m_shadowingSigmaExtWalls * m_shadowingSigmaExtWalls));
      return (sigma);
    }
}


double
BuildingsPropagationLossModel::DoCalcRxPower (double txPowerDbm, Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  return txPowerDbm - GetLoss (a, b) - GetShadowing (a, b);
}

int64_t
BuildingsPropagationLossModel::DoAssignStreams (int64_t stream)
{
  m_randVariable->SetStream (stream);
  return 1;
}


} // namespace ns3
