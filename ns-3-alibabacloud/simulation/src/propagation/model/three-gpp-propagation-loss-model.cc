/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
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

#include "three-gpp-propagation-loss-model.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/channel-condition-model.h"
#include "ns3/double.h"
#include "ns3/boolean.h"
#include "ns3/pointer.h"
#include <cmath>
#include "ns3/node.h"
#include "ns3/simulator.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ThreeGppPropagationLossModel");

static const double M_C = 3.0e8; //!< propagation velocity in free space

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppPropagationLossModel);

TypeId
ThreeGppPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppPropagationLossModel")
    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddAttribute ("Frequency", "The centre frequency in Hz.",
                   DoubleValue (500.0e6),
                   MakeDoubleAccessor (&ThreeGppPropagationLossModel::SetFrequency,
                                       &ThreeGppPropagationLossModel::GetFrequency),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("ShadowingEnabled", "Enable/disable shadowing.",
                   BooleanValue (true),
                   MakeBooleanAccessor (&ThreeGppPropagationLossModel::m_shadowingEnabled),
                   MakeBooleanChecker ())
    .AddAttribute ("ChannelConditionModel", "Pointer to the channel condition model.",
                   PointerValue (),
                   MakePointerAccessor (&ThreeGppPropagationLossModel::SetChannelConditionModel,
                                        &ThreeGppPropagationLossModel::GetChannelConditionModel),
                   MakePointerChecker<ChannelConditionModel> ())
  ;
  return tid;
}

ThreeGppPropagationLossModel::ThreeGppPropagationLossModel ()
  : PropagationLossModel ()
{
  NS_LOG_FUNCTION (this);

  // initialize the normal random variable
  m_normRandomVariable = CreateObject<NormalRandomVariable> ();
  m_normRandomVariable->SetAttribute ("Mean", DoubleValue (0));
  m_normRandomVariable->SetAttribute ("Variance", DoubleValue (1));
}

ThreeGppPropagationLossModel::~ThreeGppPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

void
ThreeGppPropagationLossModel::DoDispose ()
{
  m_channelConditionModel->Dispose ();
  m_channelConditionModel = nullptr;
  m_shadowingMap.clear ();
}

void
ThreeGppPropagationLossModel::SetChannelConditionModel (Ptr<ChannelConditionModel> model)
{
  NS_LOG_FUNCTION (this);
  m_channelConditionModel = model;
}

Ptr<ChannelConditionModel>
ThreeGppPropagationLossModel::GetChannelConditionModel () const
{
  NS_LOG_FUNCTION (this);
  return m_channelConditionModel;
}

void
ThreeGppPropagationLossModel::SetFrequency (double f)
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (f >= 500.0e6 && f <= 100.0e9, "Frequency should be between 0.5 and 100 GHz but is " << f);
  m_frequency = f;
}

double
ThreeGppPropagationLossModel::GetFrequency () const
{
  NS_LOG_FUNCTION (this);
  return m_frequency;
}

double
ThreeGppPropagationLossModel::DoCalcRxPower (double txPowerDbm,
                                             Ptr<MobilityModel> a,
                                             Ptr<MobilityModel> b) const
{
  NS_LOG_FUNCTION (this);

  // check if the model is initialized
  NS_ASSERT_MSG (m_frequency != 0.0, "First set the centre frequency");

  // retrieve the channel condition
  NS_ASSERT_MSG (m_channelConditionModel, "First set the channel condition model");
  Ptr<ChannelCondition> cond = m_channelConditionModel->GetChannelCondition (a, b);

  // compute the 2D distance between a and b
  double distance2d = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

  // compute the 3D distance between a and b
  double distance3d = CalculateDistance (a->GetPosition (), b->GetPosition ());

  // compute hUT and hBS
  std::pair<double, double> heights = GetUtAndBsHeights (a->GetPosition ().z, b->GetPosition ().z);

  double rxPow = txPowerDbm;
  rxPow -= GetLoss (cond, distance2d, distance3d, heights.first, heights.second); 
  
  if (m_shadowingEnabled)
    {
      rxPow -= GetShadowing (a, b, cond->GetLosCondition ());
    }

  return rxPow;
}

double
ThreeGppPropagationLossModel::GetLoss (Ptr<ChannelCondition> cond, double distance2d, double distance3d, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);
  
  double loss = 0;
  if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::LOS)
    {
      loss = GetLossLos (distance2d, distance3d, hUt, hBs);
    }
  else if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::NLOSv)
    {
      loss = GetLossNlosv (distance2d, distance3d, hUt, hBs);
    }
  else if (cond->GetLosCondition () == ChannelCondition::LosConditionValue::NLOS)
    {
      loss = GetLossNlos (distance2d, distance3d, hUt, hBs);
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }
  return loss;
}

double
ThreeGppPropagationLossModel::GetLossNlosv (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);
  NS_FATAL_ERROR ("Unsupported channel condition (NLOSv)");
  return 0;
}

double
ThreeGppPropagationLossModel::GetShadowing (Ptr<MobilityModel> a, Ptr<MobilityModel> b, ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);

  double shadowingValue;

  // compute the channel key
  uint32_t key = GetKey (a, b);

  bool notFound = false; // indicates if the shadowing value has not been computed yet
  bool newCondition = false; // indicates if the channel condition has changed
  Vector newDistance; // the distance vector, that is not a distance but a difference
  auto it = m_shadowingMap.end (); // the shadowing map iterator
  if (m_shadowingMap.find (key) != m_shadowingMap.end ())
    {
      // found the shadowing value in the map
      it = m_shadowingMap.find (key);
      newDistance = GetVectorDifference (a, b);
      newCondition = (it->second.m_condition != cond); // true if the condition changed
    }
  else
    {
      notFound = true;

      // add a new entry in the map and update the iterator
      ShadowingMapItem newItem;
      it = m_shadowingMap.insert (it, std::make_pair (key, newItem));
    }

  if (notFound || newCondition)
    {
      // generate a new independent realization
      shadowingValue = m_normRandomVariable->GetValue () * GetShadowingStd (a, b, cond);
    }
  else
    {
      // compute a new correlated shadowing loss
      Vector2D displacement (newDistance.x - it->second.m_distance.x, newDistance.y - it->second.m_distance.y);
      double R = exp (-1 * displacement.GetLength () / GetShadowingCorrelationDistance (cond));
      shadowingValue =  R * it->second.m_shadowing + sqrt (1 - R * R) * m_normRandomVariable->GetValue () * GetShadowingStd (a, b, cond);
    }

  // update the entry in the map
  it->second.m_shadowing = shadowingValue;
  it->second.m_distance = newDistance; // Save the (0,0,0) vector in case it's the first time we are calculating this value
  it->second.m_condition = cond;

  return shadowingValue;
}

std::pair<double, double>
ThreeGppPropagationLossModel::GetUtAndBsHeights (double za, double zb) const
{
  // The default implementation assumes that the tallest node is the BS and the
  // smallest is the UT.
  double hUt = std::min (za, zb);
  double hBs = std::max (za, zb);

  return std::pair<double, double> (hUt, hBs);
}

int64_t
ThreeGppPropagationLossModel::DoAssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this);

  m_normRandomVariable->SetStream (stream);
  return 1;
}

double
ThreeGppPropagationLossModel::Calculate2dDistance (Vector a, Vector b)
{
  double x = a.x - b.x;
  double y = a.y - b.y;
  double distance2D = sqrt (x * x + y * y);

  return distance2D;
}

uint32_t
ThreeGppPropagationLossModel::GetKey (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  // use the nodes ids to obtain an unique key for the channel between a and b
  // sort the nodes ids so that the key is reciprocal
  uint32_t x1 = std::min (a->GetObject<Node> ()->GetId (), b->GetObject<Node> ()->GetId ());
  uint32_t x2 = std::max (a->GetObject<Node> ()->GetId (), b->GetObject<Node> ()->GetId ());

  // use the cantor function to obtain the key
  uint32_t key = (((x1 + x2) * (x1 + x2 + 1)) / 2) + x2;

  return key;
}

Vector
ThreeGppPropagationLossModel::GetVectorDifference (Ptr<MobilityModel> a, Ptr<MobilityModel> b)
{
  uint32_t x1 = a->GetObject<Node> ()->GetId ();
  uint32_t x2 = b->GetObject<Node> ()->GetId ();

  if (x1 < x2)
    {
      return b->GetPosition () - a->GetPosition ();
    }
  else
    {
      return a->GetPosition () - b->GetPosition ();
    }
}

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppRmaPropagationLossModel);

TypeId
ThreeGppRmaPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppRmaPropagationLossModel")
    .SetParent<ThreeGppPropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ThreeGppRmaPropagationLossModel> ()
    .AddAttribute ("AvgBuildingHeight", "The average building height in meters.",
                   DoubleValue (5.0),
                   MakeDoubleAccessor (&ThreeGppRmaPropagationLossModel::m_h),
                   MakeDoubleChecker<double> (5.0, 50.0))
    .AddAttribute ("AvgStreetWidth", "The average street width in meters.",
                   DoubleValue (20.0),
                   MakeDoubleAccessor (&ThreeGppRmaPropagationLossModel::m_w),
                   MakeDoubleChecker<double> (5.0, 50.0))
  ;
  return tid;
}

ThreeGppRmaPropagationLossModel::ThreeGppRmaPropagationLossModel ()
  : ThreeGppPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);

  // set a default channel condition model
  m_channelConditionModel = CreateObject<ThreeGppRmaChannelConditionModel> ();
}

ThreeGppRmaPropagationLossModel::~ThreeGppRmaPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

double
ThreeGppRmaPropagationLossModel::GetLossLos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_frequency <= 30.0e9, "RMa scenario is valid for frequencies between 0.5 and 30 GHz.");

  // check if hBS and hUT are within the specified validity range
  if (hUt < 1.0 || hUt > 10.0)
    {
      NS_LOG_WARN ("The height of the UT should be between 1 and 10 m (see TR 38.901, Table 7.4.1-1)");
    }

  if (hBs < 10.0 || hBs > 150.0)
    {
      NS_LOG_WARN ("The height of the BS should be between 10 and 150 m (see TR 38.901, Table 7.4.1-1)");
    }

  // NOTE The model is intended to be used for BS-UT links, however we may need to
  // compute the pathloss between two BSs or UTs, e.g., to evaluate the
  // interference. In order to apply the model, we need to retrieve the values of
  // hBS and hUT, but in these cases one of the two falls outside the validity
  // range and the warning message is printed (hBS for the UT-UT case and hUT
  // for the BS-BS case).

  double distanceBp = GetBpDistance (m_frequency, hBs, hUt);
  NS_LOG_DEBUG ("breakpoint distance " << distanceBp);

  // check if the distance is outside the validity range
  if (distance2D < 10.0 || distance2D > 10.0e3)
    {
      NS_LOG_WARN ("The 2D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss (see 3GPP TR 38.901, Table 7.4.1-1)
  double loss = 0;
  if (distance2D <= distanceBp)
    {
      // use PL1
      loss = Pl1 (m_frequency, distance3D, m_h, m_w);
    }
  else
    {
      // use PL2
      loss = Pl1 (m_frequency, distanceBp, m_h, m_w) + 40 * log10 (distance3D / distanceBp);
    }

  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppRmaPropagationLossModel::GetLossNlos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT_MSG (m_frequency <= 30.0e9, "RMa scenario is valid for frequencies between 0.5 and 30 GHz.");

  // check if hBs and hUt are within the validity range
  if (hUt < 1.0 || hUt > 10.0)
    {
      NS_LOG_WARN ("The height of the UT should be between 1 and 10 m (see TR 38.901, Table 7.4.1-1)");
    }

  if (hBs < 10.0 || hBs > 150.0)
    {
      NS_LOG_WARN ("The height of the BS should be between 10 and 150 m (see TR 38.901, Table 7.4.1-1)");
    }

  // NOTE The model is intended to be used for BS-UT links, however we may need to
  // compute the pathloss between two BSs or UTs, e.g., to evaluate the
  // interference. In order to apply the model, we need to retrieve the values of
  // hBS and hUT, but in these cases one of the two falls outside the validity
  // range and the warning message is printed (hBS for the UT-UT case and hUT
  // for the BS-BS case).

  // check if the distance is outside the validity range
  if (distance2D < 10.0 || distance2D > 5.0e3)
    {
      NS_LOG_WARN ("The 2D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss
  double plNlos = 161.04 - 7.1 * log10 (m_w) + 7.5 * log10 (m_h) - (24.37 - 3.7 * pow ((m_h / hBs), 2)) * log10 (hBs) + (43.42 - 3.1 * log10 (hBs)) * (log10 (distance3D) - 3.0) + 20.0 * log10 (m_frequency / 1e9) - (3.2 * pow (log10 (11.75 * hUt), 2) - 4.97);

  double loss = std::max (GetLossLos (distance2D, distance3D, hUt, hBs), plNlos);

  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppRmaPropagationLossModel::GetShadowingStd (Ptr<MobilityModel> a, Ptr<MobilityModel> b, ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double shadowingStd;

  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      // compute the 2D distance between the two nodes
      double distance2d = Calculate2dDistance (a->GetPosition (), b->GetPosition ());

      // compute the breakpoint distance (see 3GPP TR 38.901, Table 7.4.1-1, note 5)
      double distanceBp = GetBpDistance (m_frequency, a->GetPosition ().z, b->GetPosition ().z);

      if (distance2d <= distanceBp)
        {
          shadowingStd = 4.0;
        }
      else
        {
          shadowingStd = 6.0;
        }
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      shadowingStd = 8.0;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return shadowingStd;
}

double
ThreeGppRmaPropagationLossModel::GetShadowingCorrelationDistance (ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double correlationDistance;

  // See 3GPP TR 38.901, Table 7.5-6
  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      correlationDistance = 37;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      correlationDistance = 120;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return correlationDistance;
}

double
ThreeGppRmaPropagationLossModel::Pl1 (double frequency, double distance3D, double h, [[maybe_unused]] double w)
{
  double loss = 20.0 * log10 (40.0 * M_PI * distance3D * frequency / 1e9 / 3.0) + std::min (0.03 * pow (h, 1.72), 10.0) * log10 (distance3D) - std::min (0.044 * pow (h, 1.72), 14.77) + 0.002 * log10 (h) * distance3D;
  return loss;
}

double
ThreeGppRmaPropagationLossModel::GetBpDistance (double frequency, double hA, double hB)
{
  double distanceBp = 2.0 * M_PI * hA * hB * frequency / M_C;
  return distanceBp;
}

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppUmaPropagationLossModel);

TypeId
ThreeGppUmaPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppUmaPropagationLossModel")
    .SetParent<ThreeGppPropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ThreeGppUmaPropagationLossModel> ()
  ;
  return tid;
}

ThreeGppUmaPropagationLossModel::ThreeGppUmaPropagationLossModel ()
  : ThreeGppPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
  m_uniformVar = CreateObject<UniformRandomVariable> ();

  // set a default channel condition model
  m_channelConditionModel = CreateObject<ThreeGppUmaChannelConditionModel> ();
}

ThreeGppUmaPropagationLossModel::~ThreeGppUmaPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

double
ThreeGppUmaPropagationLossModel::GetBpDistance (double hUt, double hBs, double distance2D) const
{
  NS_LOG_FUNCTION (this);

  // compute g (d2D) (see 3GPP TR 38.901, Table 7.4.1-1, Note 1)
  double g = 0.0;
  if (distance2D > 18.0)
    {
      g = 5.0 / 4.0 * pow (distance2D / 100.0, 3) * exp (-distance2D / 150.0);
    }

  // compute C (hUt, d2D) (see 3GPP TR 38.901, Table 7.4.1-1, Note 1)
  double c = 0.0;
  if (hUt >= 13.0)
    {
      c = pow ((hUt - 13.0) / 10.0, 1.5) * g;
    }

  // compute hE (see 3GPP TR 38.901, Table 7.4.1-1, Note 1)
  double prob = 1.0 / (1.0 + c);
  double hE = 0.0;
  if (m_uniformVar->GetValue () < prob)
    {
      hE = 1.0;
    }
  else
    {
      int random = m_uniformVar->GetInteger (12, (int)(hUt - 1.5));
      hE = (double)floor (random / 3.0) * 3.0;
    }

  // compute dBP' (see 3GPP TR 38.901, Table 7.4.1-1, Note 1)
  double distanceBp = 4 * (hBs - hE) * (hUt - hE) * m_frequency / M_C;

  return distanceBp;
}

double
ThreeGppUmaPropagationLossModel::GetLossLos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);

  // check if hBS and hUT are within the validity range
  if (hUt < 1.5 || hUt > 22.5)
    {
      NS_LOG_WARN ("The height of the UT should be between 1.5 and 22.5 m (see TR 38.901, Table 7.4.1-1)");
    }

  if (hBs != 25.0)
    {
      NS_LOG_WARN ("The height of the BS should be equal to 25 m (see TR 38.901, Table 7.4.1-1)");
    }

  // NOTE The model is intended to be used for BS-UT links, however we may need to
  // compute the pathloss between two BSs or UTs, e.g., to evaluate the
  // interference. In order to apply the model, we need to retrieve the values of
  // hBS and hUT, but in these cases one of the two falls outside the validity
  // range and the warning message is printed (hBS for the UT-UT case and hUT
  // for the BS-BS case).

  // compute the breakpoint distance (see 3GPP TR 38.901, Table 7.4.1-1, note 1)
  double distanceBp = GetBpDistance (hUt, hBs, distance2D);
  NS_LOG_DEBUG ("breakpoint distance " << distanceBp);

  // check if the distance is outside the validity range
  if (distance2D < 10.0 || distance2D > 5.0e3)
    {
      NS_LOG_WARN ("The 2D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss (see 3GPP TR 38.901, Table 7.4.1-1)
  double loss = 0;
  if (distance2D <= distanceBp)
    {
      // use PL1
      loss = 28.0 + 22.0 * log10 (distance3D) + 20.0 * log10 (m_frequency / 1e9);
    }
  else
    {
      // use PL2
      loss = 28.0 + 40.0 * log10 (distance3D) + 20.0 * log10 (m_frequency / 1e9) - 9.0 * log10 (pow (distanceBp, 2) + pow (hBs - hUt, 2));
    }

  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppUmaPropagationLossModel::GetLossNlos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);

  // check if hBS and hUT are within the vaalidity range
  if (hUt < 1.5 || hUt > 22.5)
    {
      NS_LOG_WARN ("The height of the UT should be between 1.5 and 22.5 m (see TR 38.901, Table 7.4.1-1)");
    }

  if (hBs != 25.0)
    {
      NS_LOG_WARN ("The height of the BS should be equal to 25 m (see TR 38.901, Table 7.4.1-1)");
    }

  // NOTE The model is intended to be used for BS-UT links, however we may need to
  // compute the pathloss between two BSs or UTs, e.g., to evaluate the
  // interference. In order to apply the model, we need to retrieve the values of
  // hBS and hUT, but in these cases one of the two falls outside the validity
  // range and the warning message is printed (hBS for the UT-UT case and hUT
  // for the BS-BS case).

  // check if the distance is outside the validity range
  if (distance2D < 10.0 || distance2D > 5.0e3)
    {
      NS_LOG_WARN ("The 2D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss
  double plNlos = 13.54 + 39.08 * log10 (distance3D) + 20.0 * log10 (m_frequency / 1e9) - 0.6 * (hUt - 1.5);
  double loss = std::max (GetLossLos (distance2D, distance3D, hUt, hBs), plNlos);
  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppUmaPropagationLossModel::GetShadowingStd ([[maybe_unused]] Ptr<MobilityModel> a, 
                                                  [[maybe_unused]] Ptr<MobilityModel> b, 
                                                  ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double shadowingStd;

  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      shadowingStd = 4.0;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      shadowingStd = 6.0;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return shadowingStd;
}

double
ThreeGppUmaPropagationLossModel::GetShadowingCorrelationDistance (ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double correlationDistance;

  // See 3GPP TR 38.901, Table 7.5-6
  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      correlationDistance = 37;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      correlationDistance = 50;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return correlationDistance;
}

int64_t
ThreeGppUmaPropagationLossModel::DoAssignStreams (int64_t stream)
{
  NS_LOG_FUNCTION (this);

  m_normRandomVariable->SetStream (stream);
  m_uniformVar->SetStream (stream);
  return 2;
}

// ------------------------------------------------------------------------- //

NS_OBJECT_ENSURE_REGISTERED (ThreeGppUmiStreetCanyonPropagationLossModel);

TypeId
ThreeGppUmiStreetCanyonPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppUmiStreetCanyonPropagationLossModel")
    .SetParent<ThreeGppPropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ThreeGppUmiStreetCanyonPropagationLossModel> ()
  ;
  return tid;
}
ThreeGppUmiStreetCanyonPropagationLossModel::ThreeGppUmiStreetCanyonPropagationLossModel ()
  : ThreeGppPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);

  // set a default channel condition model
  m_channelConditionModel = CreateObject<ThreeGppUmiStreetCanyonChannelConditionModel> ();
}

ThreeGppUmiStreetCanyonPropagationLossModel::~ThreeGppUmiStreetCanyonPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

double
ThreeGppUmiStreetCanyonPropagationLossModel::GetBpDistance (double hUt, double hBs, 
                                                            [[maybe_unused]] double distance2D) const
{
  NS_LOG_FUNCTION (this);

  // compute hE (see 3GPP TR 38.901, Table 7.4.1-1, Note 1)
  double hE = 1.0;

  // compute dBP' (see 3GPP TR 38.901, Table 7.4.1-1, Note 1)
  double distanceBp = 4 * (hBs - hE) * (hUt - hE) * m_frequency / M_C;

  return distanceBp;
}

double
ThreeGppUmiStreetCanyonPropagationLossModel::GetLossLos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);

  // check if hBS and hUT are within the validity range
  if (hUt < 1.5 || hUt >= 10.0)
    {
      NS_LOG_WARN ("The height of the UT should be between 1.5 and 22.5 m (see TR 38.901, Table 7.4.1-1). We further assume hUT < hBS, then hUT is upper bounded by hBS, which should be 10 m");
    }

  if (hBs != 10.0)
    {
      NS_LOG_WARN ("The height of the BS should be equal to 10 m (see TR 38.901, Table 7.4.1-1)");
    }

  // NOTE The model is intended to be used for BS-UT links, however we may need to
  // compute the pathloss between two BSs or UTs, e.g., to evaluate the
  // interference. In order to apply the model, we need to retrieve the values of
  // hBS and hUT, but in these cases one of the two falls outside the validity
  // range and the warning message is printed (hBS for the UT-UT case and hUT
  // for the BS-BS case).

  // compute the breakpoint distance (see 3GPP TR 38.901, Table 7.4.1-1, note 1)
  double distanceBp = GetBpDistance (hUt, hBs, distance2D);
  NS_LOG_DEBUG ("breakpoint distance " << distanceBp);

  // check if the distance is outside the validity range
  if (distance2D < 10.0 || distance2D > 5.0e3)
    {
      NS_LOG_WARN ("The 2D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss (see 3GPP TR 38.901, Table 7.4.1-1)
  double loss = 0;
  if (distance2D <= distanceBp)
    {
      // use PL1
      loss = 32.4 + 21.0 * log10 (distance3D) + 20.0 * log10 (m_frequency / 1e9);
    }
  else
    {
      // use PL2
      loss = 32.4 + 40.0 * log10 (distance3D) + 20.0 * log10 (m_frequency / 1e9) - 9.5 * log10 (pow (distanceBp, 2) + pow (hBs - hUt, 2));
    }

  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppUmiStreetCanyonPropagationLossModel::GetLossNlos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);

  // check if hBS and hUT are within the validity range
  if (hUt < 1.5 || hUt >= 10.0)
    {
      NS_LOG_WARN ("The height of the UT should be between 1.5 and 22.5 m (see TR 38.901, Table 7.4.1-1). We further assume hUT < hBS, then hUT is upper bounded by hBS, which should be 10 m");
    }

  if (hBs != 10.0)
    {
      NS_LOG_WARN ("The height of the BS should be equal to 10 m (see TR 38.901, Table 7.4.1-1)");
    }

  // NOTE The model is intended to be used for BS-UT links, however we may need to
  // compute the pathloss between two BSs or UTs, e.g., to evaluate the
  // interference. In order to apply the model, we need to retrieve the values of
  // hBS and hUT, but in these cases one of the two falls outside the validity
  // range and the warning message is printed (hBS for the UT-UT case and hUT
  // for the BS-BS case).

  // check if the distance is outside the validity range
  if (distance2D < 10.0 || distance2D > 5.0e3)
    {
      NS_LOG_WARN ("The 2D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss
  double plNlos = 22.4 + 35.3 * log10 (distance3D) + 21.3 * log10 (m_frequency / 1e9) - 0.3 * (hUt - 1.5);
  double loss = std::max (GetLossLos (distance2D, distance3D, hUt, hBs), plNlos);
  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

std::pair<double, double>
ThreeGppUmiStreetCanyonPropagationLossModel::GetUtAndBsHeights (double za, double zb) const
{
  NS_LOG_FUNCTION (this);
  // TR 38.901 specifies hBS = 10 m and 1.5 <= hUT <= 22.5
  double hBs, hUt;
  if (za == 10.0)
    {
      // node A is the BS and node B is the UT
      hBs = za;
      hUt = zb;
    }
  else if (zb == 10.0)
    {
      // node B is the BS and node A is the UT
      hBs = zb;
      hUt = za;
    }
  else
    {
      // We cannot know who is the BS and who is the UT, we assume that the
      // tallest node is the BS and the smallest is the UT
      hBs = std::max (za, zb);
      hUt = std::min (za, za);
    }

  return std::pair<double, double> (hUt, hBs);
}

double
ThreeGppUmiStreetCanyonPropagationLossModel::GetShadowingStd ([[maybe_unused]] Ptr<MobilityModel> a, 
                                                              [[maybe_unused]] Ptr<MobilityModel> b, 
                                                              ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double shadowingStd;

  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      shadowingStd = 4.0;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      shadowingStd = 7.82;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return shadowingStd;
}

double
ThreeGppUmiStreetCanyonPropagationLossModel::GetShadowingCorrelationDistance (ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double correlationDistance;

  // See 3GPP TR 38.901, Table 7.5-6
  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      correlationDistance = 10;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
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

NS_OBJECT_ENSURE_REGISTERED (ThreeGppIndoorOfficePropagationLossModel);

TypeId
ThreeGppIndoorOfficePropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ThreeGppIndoorOfficePropagationLossModel")
    .SetParent<ThreeGppPropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<ThreeGppIndoorOfficePropagationLossModel> ()
  ;
  return tid;
}
ThreeGppIndoorOfficePropagationLossModel::ThreeGppIndoorOfficePropagationLossModel ()
  : ThreeGppPropagationLossModel ()
{
  NS_LOG_FUNCTION (this);

  // set a default channel condition model
  m_channelConditionModel = CreateObject<ThreeGppIndoorOpenOfficeChannelConditionModel> ();
}

ThreeGppIndoorOfficePropagationLossModel::~ThreeGppIndoorOfficePropagationLossModel ()
{
  NS_LOG_FUNCTION (this);
}

double
ThreeGppIndoorOfficePropagationLossModel::GetLossLos ([[maybe_unused]] double distance2D, 
                                                      [[maybe_unused]] double distance3D, 
                                                      [[maybe_unused]] double hUt, 
                                                      [[maybe_unused]] double hBs) const
{
  NS_LOG_FUNCTION (this);

  // check if the distance is outside the validity range
  if (distance3D < 1.0 || distance3D > 150.0)
    {
      NS_LOG_WARN ("The 3D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss (see 3GPP TR 38.901, Table 7.4.1-1)
  double loss = 32.4 + 17.3 * log10 (distance3D) + 20.0 * log10 (m_frequency / 1e9);

  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppIndoorOfficePropagationLossModel::GetLossNlos (double distance2D, double distance3D, double hUt, double hBs) const
{
  NS_LOG_FUNCTION (this);

  // check if the distance is outside the validity range
  if (distance3D < 1.0 || distance3D > 150.0)
    {
      NS_LOG_WARN ("The 3D distance is outside the validity range, the pathloss value may not be accurate");
    }

  // compute the pathloss
  double plNlos = 17.3 + 38.3 * log10 (distance3D) + 24.9 * log10 (m_frequency / 1e9);
  double loss = std::max (GetLossLos (distance2D, distance3D, hUt, hBs), plNlos);

  NS_LOG_DEBUG ("Loss " << loss);

  return loss;
}

double
ThreeGppIndoorOfficePropagationLossModel::GetShadowingStd ([[maybe_unused]] Ptr<MobilityModel> a, 
                                                           [[maybe_unused]] Ptr<MobilityModel> b, 
                                                           ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);
  double shadowingStd;

  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      shadowingStd = 3.0;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      shadowingStd = 8.03;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return shadowingStd;
}

double
ThreeGppIndoorOfficePropagationLossModel::GetShadowingCorrelationDistance (ChannelCondition::LosConditionValue cond) const
{
  NS_LOG_FUNCTION (this);

  // See 3GPP TR 38.901, Table 7.5-6
  double correlationDistance;

  if (cond == ChannelCondition::LosConditionValue::LOS)
    {
      correlationDistance = 10;
    }
  else if (cond == ChannelCondition::LosConditionValue::NLOS)
    {
      correlationDistance = 6;
    }
  else
    {
      NS_FATAL_ERROR ("Unknown channel condition");
    }

  return correlationDistance;
}

} // namespace ns3
