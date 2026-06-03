/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 SIGNET Lab, Department of Information Engineering,
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

#include "probabilistic-v2v-channel-condition-model.h"
#include "ns3/log.h"
#include "ns3/string.h"
#include "ns3/mobility-model.h"
#include "ns3/enum.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ProbabilisticV2vChannelConditionModel");

NS_OBJECT_ENSURE_REGISTERED (ProbabilisticV2vUrbanChannelConditionModel);

TypeId
ProbabilisticV2vUrbanChannelConditionModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ProbabilisticV2vUrbanChannelConditionModel")
    .SetParent<ThreeGppChannelConditionModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ProbabilisticV2vUrbanChannelConditionModel> ()
    .AddAttribute ("Density", "Specifies the density of the vehicles in the scenario."
                   "It can be set to Low, Medium or High.",
                   EnumValue (VehicleDensity::LOW),
                   MakeEnumAccessor (&ProbabilisticV2vUrbanChannelConditionModel::m_densityUrban),
                   MakeEnumChecker (VehicleDensity::LOW, "Low",
                                    VehicleDensity::MEDIUM, "Medium",
                                    VehicleDensity::HIGH, "High"))
  ;
  return tid;
}

ProbabilisticV2vUrbanChannelConditionModel::ProbabilisticV2vUrbanChannelConditionModel ()
  : ThreeGppChannelConditionModel ()
{}

ProbabilisticV2vUrbanChannelConditionModel::~ProbabilisticV2vUrbanChannelConditionModel ()
{}

double
ProbabilisticV2vUrbanChannelConditionModel::ComputePlos (Ptr<const MobilityModel> a,
                                                         Ptr<const MobilityModel> b) const
{
  // compute the 2D distance between a and b
  double distance2D = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

  double pLos = 0.0;
  switch (m_densityUrban)
    {
      case VehicleDensity::LOW:
        pLos = std::min (1.0, std::max (0.0, 0.8548 * exp (-0.0064 * distance2D)));
        break;
      case VehicleDensity::MEDIUM:
        pLos = std::min (1.0, std::max (0.0, 0.8372 * exp (-0.0114 * distance2D)));
        break;
      case VehicleDensity::HIGH:
        pLos = std::min (1.0, std::max (0.0, 0.8962 * exp (-0.017 * distance2D)));
        break;
      default:
        NS_FATAL_ERROR ("Undefined density, choose between Low, Medium and High");
    }

  return pLos;
}

double
ProbabilisticV2vUrbanChannelConditionModel::ComputePnlos (Ptr<const MobilityModel> a,
                                                          Ptr<const MobilityModel> b) const
{
  // compute the 2D distance between a and b
  double distance2D = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

  // compute the NLOSv probability
  double pNlosv = 0.0;
  switch (m_densityUrban)
    {
      case VehicleDensity::LOW:
        pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0396 * distance2D) * exp (-(log (distance2D) - 5.2718) * (log (distance2D) - 5.2718) / 3.4827)));
        break;
      case VehicleDensity::MEDIUM:
        pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0312 * distance2D) * exp (-(log (distance2D) - 5.0063) * (log (distance2D) - 5.0063) / 2.4544)));
        break;
      case VehicleDensity::HIGH:
        pNlosv = std::min (1.0, std::max (0.0, 1 / (0.0242 * distance2D) * exp (-(log (distance2D) - 5.0115) * (log (distance2D) - 5.0115) / 2.2092)));
        break;
      default:
        NS_FATAL_ERROR ("Undefined density, choose between Low, Medium and High");
    }

  // derive the NLOS probability
  double pNlos = 1 - ComputePlos (a, b) - pNlosv;
  return pNlos;
}

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ProbabilisticV2vHighwayChannelConditionModel);

TypeId
ProbabilisticV2vHighwayChannelConditionModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ProbabilisticV2vHighwayChannelConditionModel")
    .SetParent<ThreeGppChannelConditionModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ProbabilisticV2vHighwayChannelConditionModel> ()
    .AddAttribute ("Density", "Specifies the density of the vehicles in the scenario."
                   "It can be set to Low, Medium or High.",
                   EnumValue (VehicleDensity::LOW),
                   MakeEnumAccessor (&ProbabilisticV2vHighwayChannelConditionModel::m_densityHighway),
                   MakeEnumChecker (VehicleDensity::LOW, "Low",
                                    VehicleDensity::MEDIUM, "Medium",
                                    VehicleDensity::HIGH, "High"))
  ;
  return tid;
}

ProbabilisticV2vHighwayChannelConditionModel::ProbabilisticV2vHighwayChannelConditionModel ()
  : ThreeGppChannelConditionModel ()
{}

ProbabilisticV2vHighwayChannelConditionModel::~ProbabilisticV2vHighwayChannelConditionModel ()
{}

double
ProbabilisticV2vHighwayChannelConditionModel::ComputePlos (Ptr<const MobilityModel> a,
                                                           Ptr<const MobilityModel> b) const
{
  // compute the 2D distance between a and b
  double distance2D = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

  double aLos = 0.0;
  double bLos = 0.0;
  double cLos = 0.0;
  switch (m_densityHighway)
    {
      case VehicleDensity::LOW:
        aLos = 1.5e-6;
        bLos = -0.0015;
        cLos = 1.0;
        break;
      case VehicleDensity::MEDIUM:
        aLos = 2.7e-6;
        bLos = -0.0025;
        cLos = 1.0;
        break;
      case VehicleDensity::HIGH:
        aLos = 3.2e-6;
        bLos = -0.003;
        cLos = 1.0;
        break;
      default:
        NS_FATAL_ERROR ("Undefined density, choose between Low, Medium and High");
    }

  double pLos = std::min (1.0, std::max (0.0, aLos * distance2D * distance2D + bLos * distance2D + cLos));

  return pLos;
}

double
ProbabilisticV2vHighwayChannelConditionModel::ComputePnlos (Ptr<const MobilityModel> a,
                                                            Ptr<const MobilityModel> b) const
{
  // compute the 2D distance between a and b
  double distance2D = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

  double aNlos = 0.0;
  double bNlos = 0.0;
  double cNlos = 0.0;
  switch (m_densityHighway)
    {
      case VehicleDensity::LOW:
        aNlos = -2.9e-7;
        bNlos = 0.00059;
        cNlos = 0.0017;
        break;
      case VehicleDensity::MEDIUM:
        aNlos = -3.7e-7;
        bNlos = 0.00061;
        cNlos = 0.015;
        break;
      case VehicleDensity::HIGH:
        aNlos = -4.1e-7;
        bNlos = 0.00067;
        cNlos = 0.0;
        break;
      default:
        NS_FATAL_ERROR ("Undefined density, choose between Low, Medium and High");
    }

  double pNlos = std::min (1.0, std::max (0.0, aNlos * pow (distance2D, 2) + bNlos * distance2D + cNlos));

  return pNlos;
}

} // end namespace ns3
