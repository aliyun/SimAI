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

#include "three-gpp-v2v-propagation-loss-model.h"
#include "ns3/double.h"
#include "ns3/log.h"
#include "ns3/string.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ThreeGppV2vPropagationLossModel");

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppV2vUrbanPropagationLossModel);

TypeId
ThreeGppV2vUrbanPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppV2vUrbanPropagationLossModel")
    .SetParent<ThreeGppPropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ThreeGppV2vUrbanPropagationLossModel> ()
    .AddAttribute ("PercType3Vehicles",
                   "The percentage of vehicles of type 3 (i.e., trucks) in the scenario",
                   DoubleValue (0.0),
                   MakeDoubleAccessor (&ThreeGppV2vUrbanPropagationLossModel::m_percType3Vehicles),
                   MakeDoubleChecker<double> (0.0, 100.0))
  ;
  return tid;
}

ThreeGppV2vUrbanPropagationLossModel::ThreeGppV2vUrbanPropagationLossModel ()
  : ThreeGppPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
  m_uniformVar = CreateObject<UniformRandomVariable> ();
  m_logNorVar = CreateObject<LogNormalRandomVariable> ();

  // set a default channel condition model
  // TODO the default ccm needs buildings, how to do this?
  // m_channelConditionModel = CreateObject<ThreeGppRmaChannelConditionModel> ();
}

ThreeGppV2vUrbanPropagationLossModel::~ThreeGppV2vUrbanPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

double
ThreeGppV2vUrbanPropagationLossModel::GetLossLos (double /* distance2D */, double distance3D, double /* hUt */, double /* hBs */) const
{
  NS_LOG_FUNCTION (this);

  // compute the pathloss (see 3GPP TR 37.885, Table 6.2.1-1)
  double loss = 38.77 + 16.7 * log10 (distance3D) + 18.2 * log10 (m_frequency / 1e9);

  return loss;
}

double
ThreeGppV2vUrbanPropagationLossModel::GetLossNlosv (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);

  // compute the pathloss (see 3GPP TR 37.885, Table 6.2.1-1)
  double loss = GetLossLos (distance2D, distance3D, hUt, hBs) + GetAdditionalNlosvLoss (distance3D, hUt, hBs);

  return loss;
}

double
ThreeGppV2vUrbanPropagationLossModel::GetAdditionalNlosvLoss (double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);
  // From TR 37.885 v15.2.0
  // When a V2V link is in NLOSv, additional vehicle blockage loss is
  // added as follows:
  // 1. The blocker height is the vehicle height which is randomly selected
  // out of the three vehicle types according to the portion of the vehicle
  // types in the simulated scenario.
  double additionalLoss = 0;
  double blockerHeight = 0;
  double mu_a = 0;
  double sigma_a = 0;
  double randomValue = m_uniformVar->GetValue () * 100.0;
  if (randomValue < m_percType3Vehicles)
    {
      // vehicles of type 3 have height 3 meters
      blockerHeight = 3.0;
    }
  else
    {
      // vehicles of type 1 and 2 have height 1.6 meters
      blockerHeight = 1.6;
    }

  // The additional blockage loss is max {0 dB, a log-normal random variable}
  if (std::min (hUt, hBs) > blockerHeight)
    {
      // Case 1: Minimum antenna height value of TX and RX > Blocker height
      additionalLoss = 0;
    }
  else if (std::max (hUt, hBs) < blockerHeight)
    {
      // Case 2: Maximum antenna height value of TX and RX < Blocker height
      mu_a = 9.0 + std::max (0.0, 15 * log10 (distance3D) - 41.0);
      sigma_a = 4.5;
      m_logNorVar->SetAttribute ("Mu", DoubleValue (log (pow (mu_a, 2) / sqrt (pow (sigma_a, 2) + pow (mu_a, 2)))));
      m_logNorVar->SetAttribute ("Sigma", DoubleValue (sqrt (log (pow (sigma_a, 2) / pow (mu_a, 2) + 1))));
      additionalLoss = std::max (0.0, m_logNorVar->GetValue ());
    }
  else
    {
      // Case 3: Otherwise
      mu_a = 5.0 + std::max (0.0, 15 * log10 (distance3D) - 41.0);
      sigma_a = 4.0;

      m_logNorVar->SetAttribute ("Mu", DoubleValue (log (pow (mu_a,2) / sqrt (pow (sigma_a, 2) + pow (mu_a, 2)))));
      m_logNorVar->SetAttribute ("Sigma", DoubleValue (sqrt (log (pow (sigma_a,2) / pow (mu_a, 2) + 1))));
      additionalLoss = std::max (0.0, m_logNorVar->GetValue ());
    }

  return additionalLoss;
}

double
ThreeGppV2vUrbanPropagationLossModel::GetLossNlos (double /* distance2D */, double distance3D, double /* hUt */, double /* hBs */) const
{
  NS_LOG_FUNCTION (this);

  double loss = 36.85 + 30 * log10 (distance3D) + 18.9 * log10 (m_frequency / 1e9);

  return loss;
}

double
ThreeGppV2vUrbanPropagationLossModel::GetShadowingStd (Ptr<MobilityModel> /* a */, Ptr<MobilityModel> /* b */, ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double shadowingStd;

  if (cond == ChannelCondition::LosConditionValue::LOS || cond == ChannelCondition::LosConditionValue::NLOSv)
    {
      shadowingStd = 3.0;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      shadowingStd = 4.0;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return shadowingStd;
}

double
ThreeGppV2vUrbanPropagationLossModel::GetShadowingCorrelationDistance (ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double correlationDistance;

  // See 3GPP TR 37.885, Table 6.2.3-1
  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      correlationDistance = 10;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOSv || cond == ChannelCondition::LosConditionValue::NLOS)
    {
      correlationDistance = 13;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return correlationDistance;
}

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppV2vHighwayPropagationLossModel);

TypeId
ThreeGppV2vHighwayPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppV2vHighwayPropagationLossModel")
    .SetParent<ThreeGppV2vUrbanPropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ThreeGppV2vHighwayPropagationLossModel> ()
  ;
  return tid;
}

ThreeGppV2vHighwayPropagationLossModel::ThreeGppV2vHighwayPropagationLossModel ()
  : ThreeGppV2vUrbanPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

ThreeGppV2vHighwayPropagationLossModel::~ThreeGppV2vHighwayPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

double
ThreeGppV2vHighwayPropagationLossModel::GetLossLos (double /* distance2D */, double distance3D, double /* hUt */, double /* hBs */) const
{
  NS_LOG_FUNCTION (this);

  // compute the pathloss (see 3GPP TR 37.885, Table 6.2.1-1)
  double loss = 32.4 + 20 * log10 (distance3D) + 20 * log10 (m_frequency / 1e9);

  return loss;
}

} // namespace ns3
