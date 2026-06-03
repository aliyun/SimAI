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

#include "ns3/propagation-loss-model.h"
#include "ns3/log.h"
#include "ns3/mobility-model.h"
#include "ns3/double.h"
#include "ns3/pointer.h"
#include <cmath>
#include "oh-buildings-propagation-loss-model.h"
#include <ns3/mobility-building-info.h>
#include "ns3/okumura-hata-propagation-loss-model.h"
#include "ns3/enum.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("OhBuildingsPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (OhBuildingsPropagationLossModel);



OhBuildingsPropagationLossModel::OhBuildingsPropagationLossModel ()
{
  m_okumuraHata = CreateObject<OkumuraHataPropagationLossModel> ();
}

OhBuildingsPropagationLossModel::~OhBuildingsPropagationLossModel ()
{
}

TypeId
OhBuildingsPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::OhBuildingsPropagationLossModel")
  
  .SetParent<BuildingsPropagationLossModel> ()
  .SetGroupName ("Buildings")
  
  .AddConstructor<OhBuildingsPropagationLossModel> ();
  
  return tid;
}


double
OhBuildingsPropagationLossModel::GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  NS_LOG_FUNCTION (this << a << b);

  // get the MobilityBuildingInfo pointers
  Ptr<MobilityBuildingInfo> a1 = a->GetObject<MobilityBuildingInfo> ();
  Ptr<MobilityBuildingInfo> b1 = b->GetObject<MobilityBuildingInfo> ();
  NS_ASSERT_MSG ((a1 != 0) && (b1 != 0), "OhBuildingsPropagationLossModel only works with MobilityBuildingInfo");

  double loss = 0.0;

  bool isAIndoor = a1->IsIndoor ();
  bool isBIndoor = b1->IsIndoor ();

  if (!isAIndoor) // a is outdoor
    {
      if (!isBIndoor) // b is outdoor
        {
          loss = m_okumuraHata->GetLoss (a, b);
          NS_LOG_INFO (this << " O-O : " << loss);
        }
      else
        {
          // b indoor
          loss = m_okumuraHata->GetLoss (a, b) + ExternalWallLoss (b1);
          NS_LOG_INFO (this << " O-I : " << loss);
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
              loss = m_okumuraHata->GetLoss (a, b) + InternalWallsLoss (a1, b1);
              NS_LOG_INFO (this << " I-I (same building)" << loss);

            }
          else
            {
              // nodes are in different buildings
              loss = m_okumuraHata->GetLoss (a, b) + ExternalWallLoss (a1) + ExternalWallLoss (b1);
              NS_LOG_INFO (this << " I-O-I (different buildings): " << loss);
            }
        }
      else
        {
          loss = m_okumuraHata->GetLoss (a, b) + ExternalWallLoss (a1);
          NS_LOG_INFO (this << " I-O : " << loss);
        } // end if (isBIndoor)
    } // end if (!isAIndoor)

  loss = std::max (0.0, loss);
  return loss;
}


} // namespace ns3
