/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
#include "ns3/log.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/mobility-model.h"
#include <cmath>

#include "kun-2600-mhz-propagation-loss-model.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Kun2600MhzPropagationLossModel");

NS_OBJECT_ENSURE_REGISTERED (Kun2600MhzPropagationLossModel);


TypeId
Kun2600MhzPropagationLossModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::Kun2600MhzPropagationLossModel")
    .SetParent<PropagationLossModel> ()
    .SetGroupName ("Propagation")
    .AddConstructor<Kun2600MhzPropagationLossModel> ()
    ;

  return tid;
}

Kun2600MhzPropagationLossModel::Kun2600MhzPropagationLossModel ()
  : PropagationLossModel ()
{
}

Kun2600MhzPropagationLossModel::~Kun2600MhzPropagationLossModel ()
{
}

double
Kun2600MhzPropagationLossModel::GetLoss (Ptr<MobilityModel> a, Ptr<MobilityModel> b) const
{
  double dist = a->GetDistanceFrom (b);  
  double loss = 36 + 26 * std::log10 (dist);
  return loss;
}

double 
Kun2600MhzPropagationLossModel::DoCalcRxPower (double txPowerDbm,
					       Ptr<MobilityModel> a,
					       Ptr<MobilityModel> b) const
{
  return (txPowerDbm - GetLoss (a, b));
}

int64_t
Kun2600MhzPropagationLossModel::DoAssignStreams (int64_t stream)
{
  return 0;
}


} // namespace ns3
