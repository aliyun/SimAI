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
 * Author: Marco Miozzo <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 * 
 */

#include <cmath>

#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/double.h"
#include "ns3/pointer.h"
#include "ns3/okumura-hata-propagation-loss-model.h"
#include "ns3/itu-r-1411-los-propagation-loss-model.h"
#include "ns3/itu-r-1411-nlos-over-rooftop-propagation-loss-model.h"
#include "ns3/itu-r-1238-propagation-loss-model.h"
#include "ns3/kun-2600-mhz-propagation-loss-model.h"
#include <ns3/mobility-building-info.h>
#include "ns3/enum.h"

#include "hybrid-buildings-propagation-loss-model.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("HybridBuildingsPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (HybridBuildingsPropagationLossModel);



HybridBuildingsPropagationLossModel::HybridBuildingsPropagationLossModel ()
{
  m_okumuraHata = CreateObject<OkumuraHataPropagationLossModel> ();
  m_ituR1411Los = CreateObject<ItuR1411LosPropagationLossModel> ();
  m_ituR1411NlosOverRooftop = CreateObject<ItuR1411NlosOverRooftopPropagationLossModel> ();
  m_ituR1238 = CreateObject<ItuR1238PropagationLossModel> ();
  m_kun2600Mhz = CreateObject<Kun2600MhzPropagationLossModel> ();
}

HybridBuildingsPropagationLossModel::~HybridBuildingsPropagationLossModel ()
{
}

TypeId
HybridBuildingsPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::HybridBuildingsPropagationLossModel")
    
    .SetParent<BuildingsPropagationLossModel> ()
    
    .AddConstructor<HybridBuildingsPropagationLossModel> ()
    .SetGroupName ("Buildings")
    
    .AddAttribute ("Frequency",
                   "The Frequency  (default is 2.106 GHz).",
                   DoubleValue (2160e6),
                   MakeDoubleAccessor (&HybridBuildingsPropagationLossModel::SetFrequency),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("Los2NlosThr",
                   " Threshold from LoS to NLoS in ITU 1411 [m].",
                   DoubleValue (200.0),
                   MakeDoubleAccessor (&HybridBuildingsPropagationLossModel::m_itu1411NlosThreshold),
                   MakeDoubleChecker<double> ())

    .AddAttribute ("Environment",
                   "Environment Scenario",
                   EnumValue (UrbanEnvironment),
                   MakeEnumAccessor (&HybridBuildingsPropagationLossModel::SetEnvironment),
                   MakeEnumChecker (UrbanEnvironment, "Urban",
                                    SubUrbanEnvironment, "SubUrban",
                                    OpenAreasEnvironment, "OpenAreas"))

    .AddAttribute ("CitySize",
                   "Dimension of the city",
                   EnumValue (LargeCity),
                   MakeEnumAccessor (&HybridBuildingsPropagationLossModel::SetCitySize),
                   MakeEnumChecker (SmallCity, "Small",
                                    MediumCity, "Medium",
                                    LargeCity, "Large"))

    .AddAttribute ("RooftopLevel",
                   "The height of the rooftop level in meters",
                   DoubleValue (20.0),
                   MakeDoubleAccessor (&HybridBuildingsPropagationLossModel::SetRooftopHeight),
                   MakeDoubleChecker<double> (0.0, 90.0))

    ;
  
  return tid;
}

void
HybridBuildingsPropagationLossModel::SetEnvironment (EnvironmentType env)
{
  m_okumuraHata->SetAttribute ("Environment", EnumValue (env));
  m_ituR1411NlosOverRooftop->SetAttribute ("Environment", EnumValue (env));
}

void
HybridBuildingsPropagationLossModel::SetCitySize (CitySize size)
{
  m_okumuraHata->SetAttribute ("CitySize", EnumValue (size));
  m_ituR1411NlosOverRooftop->SetAttribute ("CitySize", EnumValue (size));
}

void
HybridBuildingsPropagationLossModel::SetFrequency (double freq)
{
  m_okumuraHata->SetAttribute ("Frequency", DoubleValue (freq));
  m_ituR1411Los->SetAttribute ("Frequency", DoubleValue (freq));
  m_ituR1411NlosOverRooftop->SetAttribute ("Frequency", DoubleValue (freq));
  m_ituR1238->SetAttribute ("Frequency", DoubleValue (freq));
  m_frequency = freq;
}

void
HybridBuildingsPropagationLossModel::SetRooftopHeight (double rooftopHeight)
{
  m_rooftopHeight = rooftopHeight;
  m_ituR1411NlosOverRooftop->SetAttribute ("RooftopLevel", DoubleValue (rooftopHeight));
}


double
HybridBuildingsPropagationLossModel::GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  NS_ASSERT_MSG ((a->GetPosition ().z >= 0) && (b->GetPosition ().z >= 0), "HybridBuildingsPropagationLossModel does not support underground nodes (placed at z < 0)");

  
  double distance = a->GetDistanceFrom (b);

  // get the MobilityBuildingInfo pointers
  Ptr<MobilityBuildingInfo> a1 = a->GetObject<MobilityBuildingInfo> ();
  Ptr<MobilityBuildingInfo> b1 = b->GetObject<MobilityBuildingInfo> ();
  NS_ASSERT_MSG ((a1 != 0) && (b1 != 0), "HybridBuildingsPropagationLossModel only works with MobilityBuildingInfo");

  double loss = 0.0;
  bool isAIndoor = a1->IsIndoor ();
  bool isBIndoor = b1->IsIndoor ();


  if (!isAIndoor) // a is outdoor
    {
      if (!isBIndoor) // b is outdoor
        {
          if (distance > 1000)
            {
              NS_LOG_INFO (this << a->GetPosition ().z << b->GetPosition ().z << m_rooftopHeight);
              if ((a->GetPosition ().z < m_rooftopHeight)
                  && (b->GetPosition ().z < m_rooftopHeight))
                {
                  loss = ItuR1411 (a, b);
                  NS_LOG_INFO (this << " 0-0 (>1000): below rooftop -> ITUR1411 : " << loss);
                }
              else
                {
                  // Over the rooftop tranmission -> Okumura Hata
                  loss = OkumuraHata (a, b);
                  NS_LOG_INFO (this << " O-O (>1000): above rooftop -> OH : " << loss);
                }
            }
          else
            {
              // short range outdoor communication
              loss = ItuR1411 (a, b);
              NS_LOG_INFO (this << " 0-0 (<1000) Street canyon -> ITUR1411 : " << loss);
            }
        }
      else
        {
          // b indoor
          if (distance > 1000)
            {
              if ((a->GetPosition ().z < m_rooftopHeight)
                  && (b->GetPosition ().z < m_rooftopHeight))
                {                  
                  loss = ItuR1411 (a, b) + ExternalWallLoss (b1) + HeightLoss (b1);
                  NS_LOG_INFO (this << " 0-I (>1000): below rooftop -> ITUR1411 : " << loss);
                }
              else
                {
                  loss = OkumuraHata (a, b) + ExternalWallLoss (b1);
                  NS_LOG_INFO (this << " O-I (>1000): above the rooftop -> OH : " << loss);
                }
            }
          else
            {
              loss = ItuR1411 (a, b) + ExternalWallLoss (b1) + HeightLoss (b1);
              NS_LOG_INFO (this << " 0-I (<1000) ITUR1411 + BEL : " << loss);
            }
        } // end b1->isIndoor ()
    }
  else
    {
      // a is indoor
      if (isBIndoor) // b is indoor
        {
          if (a1->GetBuilding () == b1->GetBuilding ())
            {
              // nodes are in same building -> indoor communication ITU-R P.1238
              loss = ItuR1238 (a, b) + InternalWallsLoss (a1, b1);
              NS_LOG_INFO (this << " I-I (same building) ITUR1238 : " << loss);

            }
          else
            {
              // nodes are in different buildings
              loss = ItuR1411 (a, b) + ExternalWallLoss (a1) + ExternalWallLoss (b1);
              NS_LOG_INFO (this << " I-I (different) ITUR1238 + 2*BEL : " << loss);
            }
        }
      else
        {
          // b is outdoor
          if (distance > 1000)
            {
              if ((a->GetPosition ().z < m_rooftopHeight)
                  && (b->GetPosition ().z < m_rooftopHeight))
                {
                  loss = ItuR1411 (a, b) + ExternalWallLoss (a1) + HeightLoss (a1);
                  NS_LOG_INFO (this << " I-O (>1000): down rooftop -> ITUR1411 : " << loss);
                }
              else
                {
                  // above rooftop -> OH
                  loss = OkumuraHata (a, b) + ExternalWallLoss (a1) + HeightLoss (a1);
                  NS_LOG_INFO (this << " =I-O (>1000) over rooftop OH + BEL + HG: " << loss);
                }
            }
          else
            {
              loss = ItuR1411 (a, b) + ExternalWallLoss (a1)  + HeightLoss (a1);
              NS_LOG_INFO (this << " I-O (<1000)  ITUR1411 + BEL + HG: " << loss);
            }
        } // end if (isBIndoor)
    } // end if (!isAIndoor)

  loss = std::max (loss, 0.0);

  return loss;
}


double
HybridBuildingsPropagationLossModel::OkumuraHata (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  if (m_frequency <= 2.3e9)
    {
      return m_okumuraHata->GetLoss (a, b);
    }
  else
    {
      return m_kun2600Mhz->GetLoss (a, b);
    }
}

double
HybridBuildingsPropagationLossModel::ItuR1411 (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  if (a->GetDistanceFrom (b) < m_itu1411NlosThreshold)
    {
      return (m_ituR1411Los->GetLoss (a, b));
    }
  else
    {
      return (m_ituR1411NlosOverRooftop->GetLoss (a, b));
    }
}

double
HybridBuildingsPropagationLossModel::ItuR1238 (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  return m_ituR1238->GetLoss (a,b);
}


} // namespace ns3
